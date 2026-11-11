"""Request-rewriting pipeline — the KV-cache guard.

record mode: everything passes through untouched (capture only).

stabilize mode: keep the prompt sent to a cache-capable engine append-only per
conversation.  The engine caches by token prefix (no cache-id hook exists),
so the contract is byte-stability of everything before the new turn:

* one canonical conversation per session (key = volatile-normalized system
  prompt + first user message), frozen verbatim once seen;
* images matched by *slot* (position in history), never by bytes — first-seen
  bytes win and are resurrected over re-encodes or drops, which also keeps
  the engine's embedding cache (keyed by image hash) warm;
* volatile regions (dates, plan-mode guardrails, file-state blocks learned
  from captures) are normalized only for *matching*, never in what we forward;
* requests matching sidecar rules (title-gen etc.) go to the 8083 model so
  they can't evict the main model's single conversation-cache sequence.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
from pathlib import Path

log = logging.getLogger("prefixrelay.pipeline")

# ---------------------------------------------------------------------------
# helpers


def sha(s: str | bytes) -> str:
    if isinstance(s, str):
        s = s.encode("utf-8", "surrogatepass")
    return hashlib.sha256(s).hexdigest()


def _sidecar_spec(cfg) -> dict:
    """Sidecar tuning, from ``backends.sidecar`` or the legacy top-level map."""
    sb = cfg.backend_for("sidecar") if getattr(cfg, "backends", None) else None
    if sb is not None and isinstance(sb.raw, dict) and sb.raw:
        return sb.raw
    legacy = cfg.raw.get("sidecar")
    return legacy if isinstance(legacy, dict) else {}


def _compile_rules(cfg):
    out = []
    rules = _sidecar_spec(cfg).get("rules", [])
    for r in rules:
        out.append({
            "system_re": re.compile(r.get("system_re", "."), re.S),
            "user_re": re.compile(r.get("user_re", "."), re.S),
            "max_msgs": int(r.get("max_msgs", 2)),
            "max_tokens": int(r.get("max_tokens", 256)),
        })
    return out


DEFAULT_VOLATILE = [
    r"Today's date: [^\n]+",
    r"<timestamp>[^<]+</timestamp>",
    r"<open_and_recently_viewed_files>[\s\S]*?</open_and_recently_viewed_files>",
    r"<agent_transcripts>[\s\S]*?</agent_transcripts>",
    r"<available_subagent_types>[\s\S]*?</available_subagent_types>",
    r"<available_subagent_models>[\s\S]*?</available_subagent_models>",
    r"<dynamic_tool_catalog>[\s\S]*?</dynamic_tool_catalog>",
    r"<dynamic_tools>[\s\S]*?</dynamic_tools>",
    r"<mcp_file_system_info>[\s\S]*?</mcp_file_system_info>",
    r"Is directory a git repo: [^\n]+",
]
# Toggle-style volatile blocks: Cursor inserts/removes these wholesale (plan
# mode on/off), so present and absent must normalize to the *same* text.
# They are replaced with nothing — including the block's trailing blank line,
# which travels with it — not with "<V>" (2026-10-30: unnormalized, each
# plan-mode toggle re-keyed live sessions and dropped the whole KV cache).
DEFAULT_VOLATILE_ABSENT = [
    r"<plan_mode_guardrails>[\s\S]*?</plan_mode_guardrails>\n*",
]
FIRST_USER_KEY_CHARS = 800
# Break exact specials so compatible engines/Cursor don't treat quoted tags as markers.
_ZW = "\u200b"
CHAT_SPECIALS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>")
COMPACTION_MARKERS = (
    "[Previous conversation summary]",
    "[Conversation summary]",
    "The conversation has been summarized",
)


def _volatile_re(cfg):
    raw_list = cfg.raw.get("volatile_system")
    if raw_list is None:
        pats = DEFAULT_VOLATILE
    else:
        pats = list(raw_list)
        for d in DEFAULT_VOLATILE:
            if d not in pats:
                pats.append(d)
    out = [(re.compile(p, re.S | re.M), "<V>") for p in pats]
    for p in DEFAULT_VOLATILE_ABSENT:
        if not any(p == compiled.pattern for compiled, _ in out):
            out.append((re.compile(p, re.S | re.M), ""))
    return out


_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.DOTALL)


def defang_chat_specials(s: str, think_tags: bool = False,
                         chat_specials: bool = True) -> str:
    """Insert a ZWSP so quoted chat/think tags are not exact specials.

    ``think_tags`` guards the think-wrap markers Cursor parses out of content;
    ``chat_specials`` guards the engine's own <|im_*|> tokens. A hosted backend
    that must receive byte-exact text can turn the second one off while keeping
    the first, so a model quoting ``</think>`` cannot close Cursor's think block
    early.
    """
    if not s:
        return s
    if think_tags:
        s = s.replace("</think>", "</think" + _ZW + ">")
        s = s.replace("<think>", "<think" + _ZW + ">")
    if chat_specials:
        for tag in CHAT_SPECIALS:
            s = s.replace(tag, tag[:-2] + _ZW + tag[-2:])
    return s


def _defang_value(v, think_tags: bool = False):
    if isinstance(v, str):
        return defang_chat_specials(v, think_tags=think_tags)
    if isinstance(v, dict):
        return {k: _defang_value(x, think_tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_defang_value(x, think_tags) for x in v]
    return v


def defang_request_body(body: dict) -> dict:
    """Copy the forwarded body and break <|im_*|> in message/tool text.

    Canonical session state stays unmodified; this is last-mile for the engine.
    """
    body = dict(body)
    if "messages" in body:
        body["messages"] = _defang_value(body["messages"])
    if "tools" in body:
        body["tools"] = _defang_value(body["tools"])
    return body


def apply_body_defaults(body: dict, defaults: dict) -> list[str]:
    """Merge per-route request defaults in with setdefault semantics.

    Nested dicts merge key-by-key so ``{reasoning: {effort: high}}`` adds
    ``effort`` without clobbering an unrelated ``reasoning`` option the caller
    sent. Returns the dotted keys actually applied, for the capture meta.
    """
    import copy

    applied: list[str] = []

    def rec(dst: dict, src: dict, prefix: str = "") -> None:
        for k, v in (src or {}).items():
            if k not in dst:
                dst[k] = copy.deepcopy(v)
                applied.append(prefix + str(k))
            elif isinstance(v, dict) and isinstance(dst.get(k), dict):
                rec(dst[k], v, prefix + str(k) + ".")

    rec(body, defaults)
    return applied


def _split_think(text: str) -> tuple[str, list[str]]:
    """Pull <think> blocks out of assistant content. Unclosed open-tag → rest is thought."""
    if "<think>" not in text:
        return text, []
    thoughts: list[str] = []
    if "</think>" in text:
        for m in _THINK_RE.finditer(text):
            t = m.group(1).strip()
            if t:
                thoughts.append(t)
        return _THINK_RE.sub("", text).strip(), thoughts
    i = text.find("<think>")
    before = text[:i].strip()
    thought = text[i + len("<think>"):].strip()
    if thought:
        thoughts.append(thought)
    return before, thoughts


def _blank_if_ws(s: str) -> str:
    """Think-wrap leftover newlines must not count as real assistant text."""
    return "" if not (s or "").strip() else s


def _clean_msg_think(msg: dict) -> dict:
    msg = dict(msg)
    if msg.get("role") != "assistant":
        return msg
    content = msg.get("content")
    thoughts: list[str] = []
    if isinstance(content, str):
        cleaned, thoughts = _split_think(content)
        msg["content"] = _blank_if_ws(cleaned)
    elif isinstance(content, list):
        has_think = any(isinstance(p, dict) and "<think>" in p.get("text", "")
                        for p in content)
        if has_think:
            new_parts = []
            for p in content:
                if isinstance(p, dict) and p.get("type") == "text":
                    txt = p.get("text", "")
                    if "<think>" in txt:
                        cleaned_txt, th = _split_think(txt)
                        thoughts.extend(th)
                        cleaned_txt = _blank_if_ws(cleaned_txt)
                        if cleaned_txt:
                            new_p = dict(p)
                            new_p["text"] = cleaned_txt
                            new_parts.append(new_p)
                    else:
                        new_parts.append(p)
                else:
                    new_parts.append(p)
            msg["content"] = new_parts if new_parts else ""
        else:
            # Cursor often strips <think> and leaves the close-tag's leftover newlines.
            texts = [p.get("text", "") for p in content if isinstance(p, dict) and "text" in p]
            if texts and not "".join(texts).strip() and not any(
                    isinstance(p, dict) and p.get("type") not in (None, "text") for p in content):
                msg["content"] = ""
    elif content is not None and not str(content).strip():
        msg["content"] = ""
    if thoughts and not msg.get("reasoning_content"):
        msg["reasoning_content"] = "\n\n".join(thoughts)
    return msg


def _norm_args(args) -> str:
    """Canonical comparable form of tool-call ``arguments``.

    The model streams arguments as a JSON string with its own spacing while
    Cursor re-serializes them compactly when echoing history back
    (``{"command": "ls"}`` vs ``{"command":"ls"}``); byte comparison then
    fails every tool-call turn and rebases the canonical tail.  Parse and
    re-serialize so both sides compare equal regardless of whitespace or key
    order.  Anything that is not JSON (or does not parse) falls back to byte
    comparison — conservative, since real tool args are JSON.
    """
    if isinstance(args, (dict, list)):
        return json.dumps(args, sort_keys=True, separators=(",", ":"))
    if isinstance(args, str):
        try:
            return json.dumps(json.loads(args), sort_keys=True,
                              separators=(",", ":"))
        except Exception:
            return args
    return args or ""


def _tool_sig(msg: dict):
    tcs = msg.get("tool_calls")
    if not tcs:
        return None
    sig = []
    for tc in tcs:
        fn = (tc or {}).get("function") or {}
        sig.append((tc.get("id") or "", fn.get("name") or "",
                    _norm_args(fn.get("arguments", ""))))
    return sig


def _text_of(msg: dict) -> str:
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict) and "text" in p)
    return ""


def _images_of(msg: dict) -> list[str]:
    """Image payload strings (b64 data URLs or raw b64) in order."""
    out = []
    c = msg.get("content")
    if isinstance(c, list):
        for p in c:
            if not isinstance(p, dict):
                continue
            if p.get("type") == "image_url" or "image_url" in p:
                out.append((p.get("image_url") or {}).get("url", ""))
            elif p.get("type") == "image" or "image" in p:
                out.append(str(p.get("image", "")))
    return out


def _set_image(msg: dict, idx: int, url: str) -> None:
    n = -1
    for p in msg.get("content") or []:
        if isinstance(p, dict) and (p.get("type") in ("image_url", "image")
                                    or "image_url" in p or "image" in p):
            n += 1
            if n == idx:
                if "image_url" in p or p.get("type") == "image_url":
                    p["image_url"] = {"url": url}
                else:
                    p["image"] = url
                return


# ---------------------------------------------------------------------------
# session store


class Session:
    """Canonical conversation: messages frozen at first sight, image slots
    mapped to canonical bytes on disk."""

    def __init__(self, key: str, path: Path):
        self.key = key
        self.path = path
        self.msgs: list[dict] = []
        self.image_files: dict[str, str] = {}   # "m{i}k{j}" -> filename in images dir
        self.created = time.time()
        self.updated = time.time()
        self.lock = threading.Lock()

    def to_json(self) -> dict:
        return {"key": self.key, "created": self.created, "updated": self.updated,
                "msgs": self.msgs, "image_files": self.image_files}

    @classmethod
    def from_json(cls, j: dict, path: Path) -> "Session":
        s = cls(j["key"], path)
        s.msgs = j.get("msgs") or []
        s.image_files = j.get("image_files") or {}
        s.created = j.get("created", time.time())
        s.updated = j.get("updated", time.time())
        return s

    def save(self) -> None:
        self.updated = time.time()
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_json()))
        tmp.replace(self.path)


class SessionStore:
    def __init__(self, cfg):
        self.dir = cfg.path(cfg.state_dir) / "sessions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.images_dir = cfg.path(cfg.state_dir) / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self._volatile = _volatile_re(cfg)
        for p in self.dir.glob("*.json"):
            try:
                s = Session.from_json(json.loads(p.read_text()), p)
                self.sessions[s.key] = s
            except Exception:
                log.exception("dropping corrupt session %s", p)

    def _apply_volatile(self, s: str) -> str:
        for pat, repl in self._volatile:
            s = pat.sub(repl, s)
        return s

    def session_key(self, body: dict) -> str:
        msgs = body.get("messages") or []
        sys_txt = self._apply_volatile(_text_of(next((m for m in msgs if m.get("role") == "system"), {})))
        first_user = next((m for m in msgs if m.get("role") != "system"), {})
        u_txt = self._apply_volatile(_text_of(first_user))[:FIRST_USER_KEY_CHARS]
        n_img = len(_images_of(first_user))
        return sha(f"{sys_txt}||{u_txt}||{n_img}")[:16]

    def get(self, key: str) -> Session | None:
        with self._lock:
            return self.sessions.get(key)

    def get_or_create(self, key: str) -> Session:
        with self._lock:
            s = self.sessions.get(key)
            if s is None:
                s = Session(key, self.dir / f"{key}.json")
                self.sessions[key] = s
            return s

    # -- image store ------------------------------------------------------
    def image_path(self, fname: str) -> Path:
        return self.images_dir / fname

    def store_image(self, slot: str, url: str, sess: Session) -> str:
        payload = url.split(",", 1)[1] if "," in url[:64] else url
        h = sha(payload)[:32]
        fname = f"{h}.img"
        f = self.image_path(fname)
        if not f.exists():
            f.write_text(url)
        sess.image_files[slot] = fname
        return fname

    def load_image(self, fname: str) -> str | None:
        f = self.image_path(fname)
        return f.read_text() if f.exists() else None

    # -- matching ----------------------------------------------------------
    def _msg_matches(self, sess: Session, i: int, canon: dict, inc: dict) -> bool:
        if canon.get("role") != inc.get("role"):
            return False
        if _tool_sig(canon) != _tool_sig(inc):
            return False
        a, b = _text_of(canon), _text_of(inc)
        if canon.get("role") == "assistant":
            a, b = a.strip(), b.strip()
        if a != b:
            if self._apply_volatile(a) != self._apply_volatile(b):
                return False
        return len(_images_of(canon)) == len(_images_of(inc))

    def _anchor_matches(self, canon: dict, inc: dict) -> bool:
        """First non-system message: same session-key surface, ignore catalog drift after it."""
        if canon.get("role") != inc.get("role"):
            return False
        if len(_images_of(canon)) != len(_images_of(inc)):
            return False
        ca = self._apply_volatile(_text_of(canon))[:FIRST_USER_KEY_CHARS]
        ia = self._apply_volatile(_text_of(inc))[:FIRST_USER_KEY_CHARS]
        return ca == ia

    def _is_compaction(self, canon: list[dict], inc: list[dict]) -> bool:
        """Cursor replaced a long history with a summary (+ short recent tail)."""
        for msg in inc:
            if msg.get("role") != "user":
                continue
            text = _text_of(msg)
            if any(m in text for m in COMPACTION_MARKERS):
                return True
        return bool(len(canon) >= 12 and len(inc) <= max(8, len(canon) // 3))

    def _commit_tail(self, sess: Session, body: dict, start: int) -> None:
        """Freeze incoming messages[start:] into the canonical history."""
        msgs = body.get("messages") or []
        for i in range(start, len(msgs)):
            m = json.loads(json.dumps(msgs[i]))  # deep copy
            for j, url in enumerate(_images_of(m)):
                self.store_image(f"m{i}k{j}", url, sess)
            sess.msgs.append(m)
        sess.save()

    def _apply_images(self, sess: Session, msgs: list[dict], offset: int,
                      meta: dict) -> None:
        """Replace image payloads in msgs[offset:] with canonical bytes by slot."""
        restored = 0
        for i in range(offset, len(msgs)):
            for j, url in enumerate(_images_of(msgs[i])):
                slot = f"m{i}k{j}"
                fname = sess.image_files.get(slot)
                if fname:
                    canon = self.load_image(fname)
                    if canon and canon != url:
                        payload = url.split(",", 1)[-1]
                        cpayload = canon.split(",", 1)[-1]
                        if payload != cpayload:
                            _set_image(msgs[i], j, canon)
                            restored += 1
                else:
                    self.store_image(slot, url, sess)
        if restored:
            meta["images_restored"] = restored

    def stabilize(self, body: dict, meta: dict, namespace: str = "") -> tuple[dict, str]:
        """Main-route rewrite. Returns (new_body, route).

        ``namespace`` separates canonical histories per KV backend, so
        alternating models inside one Cursor chat cannot fight over one
        frozen prefix. Left empty for a single-KV config (including legacy),
        which keeps existing session files - and their parked KV cache -
        addressable.
        """
        key = self.session_key(body)
        if namespace:
            key = f"{namespace}-{key}"
        sess = self.get_or_create(key)
        meta["session"] = key
        with sess.lock:
            inc = [_clean_msg_think(m) for m in (body.get("messages") or [])]
            canon = sess.msgs
            if not canon:
                sess.msgs = []
                body = dict(body)
                body["messages"] = inc
                self._commit_tail(sess, body, 0)
                meta["canonical"] = f"new:{len(inc)}"
                return body, "main"
            # Compaction: Cursor drops most history and injects a summary.
            # Keep a matching system prompt (KV prefix), adopt the rest from
            # incoming. Never copy reasoning by index — canon[j] is a different
            # turn after a rewrite, and that paste is what confused the agent.
            compaction = self._is_compaction(canon, inc)
            if compaction:
                meta["compaction"] = True
                d = 0
                if (canon and inc and canon[0].get("role") == "system"
                        and self._msg_matches(sess, 0, canon[0], inc[0])):
                    d = 1
                meta["divergence"] = f"{d}/{len(canon)} (incoming {len(inc)} msgs)"
                body = dict(body)
                new_msgs = json.loads(json.dumps(canon[:d])) + \
                    json.loads(json.dumps(inc[d:]))
                body["messages"] = new_msgs
                self._apply_images(sess, body["messages"], d, meta)
                sess.msgs = body["messages"]
                self._reindex_images(sess)
                sess.save()
                meta["canonical"] = f"compact:{d}+new:{len(new_msgs) - d}"
                return body, "main"

            # find divergence. The first non-system message is the session
            # anchor: Cursor mutates catalogs in that blob after the 800-char
            # key surface, which used to rebase the whole history at d=1.
            d = 0
            limit = min(len(canon), len(inc))
            anchor = next((i for i, m in enumerate(canon) if m.get("role") != "system"), None)
            while d < limit:
                if self._msg_matches(sess, d, canon[d], inc[d]):
                    d += 1
                    continue
                if anchor is not None and d == anchor:
                    if self._anchor_matches(canon[d], inc[d]):
                        meta["anchor"] = "key-surface"
                        d += 1
                        continue
                break
            if d == len(canon):
                # normal append: freeze canonical prefix, re-serialize it
                body = dict(body)
                n_canon = len(canon)
                tail = json.loads(json.dumps(inc[limit:]))
                body["messages"] = json.loads(json.dumps(canon)) + tail
                self._apply_images(sess, body["messages"], 0, meta)
                self._commit_tail(sess, body, n_canon)
                meta["canonical"] = f"frozen:{n_canon}+new:{len(tail)}"
                return body, "main"
            if d < len(canon):
                # Cursor edited/trimmed history at d — preserve the longest
                # frozen prefix we still share, adopt the rest from incoming.
                meta["divergence"] = f"{d}/{len(canon)} (incoming {len(inc)} msgs)"
                body = dict(body)
                new_msgs = json.loads(json.dumps(canon[:d])) + \
                    json.loads(json.dumps(inc[d:]))
                # Same-index restore is valid here: d is a real shared-prefix
                # cut, so canon[j] and incoming[j] are the same turn.
                for j in range(d, min(len(new_msgs), len(canon))):
                    if new_msgs[j].get("role") == "assistant":
                        if canon[j].get("reasoning_content") and not new_msgs[j].get("reasoning_content"):
                            new_msgs[j]["reasoning_content"] = canon[j]["reasoning_content"]
                body["messages"] = new_msgs
                self._apply_images(sess, body["messages"], d, meta)
                sess.msgs = body["messages"]
                self._reindex_images(sess)
                sess.save()
                meta["canonical"] = f"rebase:{d}+new:{len(new_msgs) - d}"
                return body, "main"
            # unreachable: d == limit == len(inc) < len(canon) means trimmed
            meta["canonical"] = "pass"
            return body, "main"

    def _reindex_images(self, sess: Session) -> None:
        # after a rebase, slot ids may shift; canonical bytes stay the same
        for i, m in enumerate(sess.msgs):
            for j, url in enumerate(_images_of(m)):
                sess.image_files[f"m{i}k{j}"] = self.store_image(f"m{i}k{j}", url, sess)


# ---------------------------------------------------------------------------
# module entry

_store: SessionStore | None = None
_sidecar_rules = None
_init_lock = threading.Lock()


def _init(cfg):
    global _store, _sidecar_rules
    with _init_lock:
        if _store is None:
            _sidecar_rules = _compile_rules(cfg)
            _store = SessionStore(cfg)


def _sidecar_match(body: dict) -> dict | None:
    msgs = body.get("messages") or []
    for rule in _sidecar_rules:
        if len(msgs) > rule["max_msgs"]:
            continue
        sys_txt = _text_of(next((m for m in msgs if m.get("role") == "system"), {}))
        first_user = next((m for m in msgs if m.get("role") != "system"), {})
        if sys_txt and not rule["system_re"].search(sys_txt):
            continue
        if not rule["user_re"].search(_text_of(first_user)):
            continue
        return rule
    return None


def session_key(cfg, body: dict) -> str:
    _init(cfg)
    return _store.session_key(body)


def _kv_namespace(cfg, backend) -> str:
    """Only needed when more than one KV-cached backend could serve a chat."""
    if backend is not None and getattr(cfg, "kv_backends", 0) > 1:
        return backend.name
    return ""


def process(cfg, body: dict, rid: str, backend=None,
            model_name: str | None = None) -> tuple[dict, str, dict]:
    """Return (body_to_forward, route, meta). route: main | sidecar | stub.

    ``backend`` is the resolved :class:`~proxy.config.Backend` for the model
    name Cursor asked for. KV stabilization applies only when enabled and
    configured in stabilize mode. Context compaction is independent: it can
    protect a hosted endpoint even when that backend has no reusable KV cache.
    """
    meta: dict = {}
    if cfg.mode != "stabilize":
        if backend is not None and backend.compact_enabled:
            from proxy import compaction
            body = compaction.compact_if_needed(cfg, body, backend, meta)
        return body, "main", meta
    _init(cfg)
    kv_cached = backend is None or backend.stabilize
    if not kv_cached:
        meta["stabilize"] = "skipped:" + backend.name
        if backend.compact_enabled:
            from proxy import compaction
            body = compaction.compact_if_needed(cfg, body, backend, meta)
        return body, "main", meta

    sidecar_backend = cfg.backend_for("sidecar") if getattr(cfg, "backends", None) else None
    rule = _sidecar_match(body)
    if rule and sidecar_backend is not None:
        body = dict(body)
        meta["sidecar"] = "matched"
        cur = body.get("max_tokens") or 0
        body["max_tokens"] = min(cur, rule["max_tokens"]) if cur else rule["max_tokens"]
        think_off = _sidecar_spec(cfg).get("force_thinking_off", True)
        if think_off and sidecar_backend.llama_kwargs:
            ctk = body.setdefault("chat_template_kwargs", {})
            ctk.setdefault("enable_thinking", False)
        return body, "sidecar", meta
    if rule:
        # A rule matched but no sidecar backend is configured; keep it on main
        # rather than dropping the user's title-gen request.
        meta["sidecar"] = "no-backend"
    new_body, _route = _store.stabilize(body, meta, namespace=_kv_namespace(cfg, backend))
    # Silent compaction of the outgoing projection only (canonical history
    # untouched). Enabled by default; a route can disable it per backend.
    if backend is not None and backend.compact_enabled:
        from proxy import compaction
        new_body = compaction.compact_if_needed(cfg, new_body, backend, meta)
    return new_body, "main", meta


def record_assistant_response(cfg, session_key: str, assistant_msg: dict) -> None:
    """Commit completed assistant response (with reasoning_content and tool_calls)
    to the canonical session so subsequent turns see the exact generated prefix."""
    if cfg.mode != "stabilize":
        return
    _init(cfg)
    sess = _store.get(session_key)
    if not sess:
        return
    m = _clean_msg_think(json.loads(json.dumps(assistant_msg)))
    if not (m.get("content") or m.get("reasoning_content") or m.get("tool_calls")):
        return
    with sess.lock:
        if not sess.msgs:
            sess.msgs.append(m)
            sess.save()
            return
        last = sess.msgs[-1]
        if last.get("role") == "assistant":
            # Update existing assistant message if already present at the tail
            sess.msgs[-1] = m
        else:
            sess.msgs.append(m)
        sess.save()
