"""Silent context compaction for OpenAI-compatible model backends.

When an outgoing request exceeds ``compact_at`` of the backend's per-slot
context, compact the OUTGOING projection only. Context can be configured per
route or discovered from a compatible local server:

    system + user("[Previous conversation summary] ...") + recent tail

The canonical session (``sess.msgs``) is never touched, so Cursor's history
reconciliation keeps working and next turn's ``stabilize`` still matches the
frozen prefix. The marker string is one of ``COMPACTION_MARKERS``, so a
Cursor-native compaction and a proxy compaction compose.
"""
from __future__ import annotations

import json
import ipaddress
import logging
import re
import threading
import time
from urllib.parse import urlsplit
import urllib.request

log = logging.getLogger("prefixrelay.compaction")

# Conservative pre-check: only pay for an exact /tokenize call near threshold.
CHARS_PER_TOKEN = 4
SAFETY = 1.2
IMAGE_TOKEN_ESTIMATE = 4096       # conservative allowance per image part

_prop_cache: dict[str, tuple[float, int]] = {}   # base_url -> (ts, n_ctx)
_prop_lock = threading.Lock()
_PROP_TTL = 3600.0


# ---------------------------------------------------------------------------
# token counting / n_ctx discovery

def _props_n_ctx(backend, timeout: float = 5.0) -> int:
    """Per-slot n_ctx from llama.cpp /props, cached per base_url."""
    raw = getattr(backend, "raw", {}) or {}
    host = urlsplit(backend.base_url).hostname or ""
    try:
        is_local = ipaddress.ip_address(host).is_private or ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_local = host.casefold() == "localhost" or host.casefold().endswith(".localhost")
    if raw.get("context_discovery") is True:
        is_local = True
    elif raw.get("context_discovery") is False:
        is_local = False
    if not is_local:
        return 0
    now = time.time()
    with _prop_lock:
        hit = _prop_cache.get(backend.base_url)
        if hit and now - hit[0] < _PROP_TTL:
            return hit[1]
    try:
        req = urllib.request.Request(
            _aux_url(backend, "props"),
            headers=backend.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        n_ctx = int(
            data.get("default_generation_settings", {}).get("n_ctx")
            or data.get("n_ctx") or 0)
    except Exception as e:
        log.warning("n_ctx discovery failed for %s: %s", backend.base_url, e)
        n_ctx = 0
    with _prop_lock:
        _prop_cache[backend.base_url] = (now, n_ctx)
    return n_ctx


def _context_tokens(backend) -> int:
    """Configured window, or discovered n_ctx; 0 means unknown (never compact)."""
    if backend.context_tokens:
        return backend.context_tokens
    return _props_n_ctx(backend)


def _aux_url(backend, path: str) -> str:
    """Build a local-server extension URL from an OpenAI API base URL."""
    root = backend.base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root + "/" + path.lstrip("/")


def _serialize_for_count(messages: list[dict]) -> str:
    """One string covering everything the engine reads from ``messages``."""
    parts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            # text parts count; image parts counted as a flat per-image charge
            texts = [p.get("text", "") for p in c if isinstance(p, dict) and "text" in p]
            n_img = sum(1 for p in c if isinstance(p, dict)
                        and ("image_url" in p or "image" in p
                             or p.get("type") in ("image_url", "image")))
            parts.append(" ".join(texts) + ("\n[image]\n" * n_img))
        elif c is not None:
            parts.append(str(c))
        for tc in m.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            parts.append(f"{fn.get('name', '')} {fn.get('arguments', '')}")
        if m.get("reasoning_content"):
            parts.append(str(m["reasoning_content"]))
    return "\n".join(parts)


def _estimate_tokens(messages: list[dict]) -> int:
    # 1.2 safety multiplier: when in doubt, assume the prompt is larger.
    serialized = _serialize_for_count(messages)
    image_count = serialized.count("[image]")
    return int(len(serialized) / CHARS_PER_TOKEN * SAFETY) + image_count * IMAGE_TOKEN_ESTIMATE


def _exact_tokens(backend, messages: list[dict], timeout: float = 10.0) -> int | None:
    """llama.cpp /tokenize on the serialized chat. None on any failure."""
    try:
        req = urllib.request.Request(
            _aux_url(backend, "tokenize"),
            data=json.dumps({"content": _serialize_for_count(messages)}).encode(),
            headers=backend.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        toks = data.get("tokens")
        return len(toks) if isinstance(toks, list) else None
    except Exception as e:
        log.warning("/tokenize failed for %s: %s", backend.base_url, e)
        return None


def _token_count(backend, messages: list[dict]) -> int:
    """Estimate first; exact-count only when the estimate says we might be over."""
    est = _estimate_tokens(messages)
    configured_n_ctx = int(getattr(backend, "context_tokens", 0) or 0)
    n_ctx = _context_tokens(backend)
    if not n_ctx:
        return est
    # If even the upper-bound estimate is under threshold, trust it (cheap path).
    if est < int(n_ctx * backend.compact_at) - 4096:
        return est
    # The private /tokenize endpoint is not part of the OpenAI-compatible API.
    # Only call it when the backend also exposed llama.cpp-style /props; an
    # explicitly configured hosted-model limit uses the safe estimate instead.
    exact = _exact_tokens(backend, messages) if not configured_n_ctx else None
    return exact if exact is not None else est


# ---------------------------------------------------------------------------
# tail boundary (tool-pair safe)

def _is_tool_result(m: dict) -> bool:
    return m.get("role") == "tool"


def _has_tool_calls(m: dict) -> bool:
    return bool(m.get("tool_calls"))


def _tail_boundary(messages: list[dict], want: int) -> int:
    """Index of the first message of the kept tail, tool-pair safe.

    Start from ``len - want`` and extend backwards while the boundary would
    separate an ``assistant(tool_calls)`` from its ``tool`` result(s): if the
    cut lands on a tool result, back up to include its assistant call; if it
    lands right after an assistant-with-calls whose results got cut, pull the
    whole group into the tail. Never returns before 0.
    """
    n = len(messages)
    if want <= 0:
        return 0
    start = max(0, min(n - 1, n - want))
    # boundary lands on a tool result -> include the assistant call above it
    while start > 0 and _is_tool_result(messages[start]):
        start -= 1
    return start
    # assistant with tool_calls at the boundary, results below the cut -> this
    # cannot happen: the forward walk above guarantees the boundary never sits
    # on a tool result, so an assistant-with-calls at `start` always has its
    # results inside the tail. Nothing to do.
    return max(0, start)


# ---------------------------------------------------------------------------
# elision

ELIDE_MIN_CHARS = 400          # only bother eliding meaningful outputs
ELIDE_HEAD = 200               # keep a recognizable head of an elided output
ELIDE_TAIL = 120               # ...and tail

# Marker spliced into elided tool-call arguments. Unlike tool outputs (plain
# text), ``arguments`` is itself a serialized JSON document, so the marker's
# newlines MUST be the two-character escape sequence "\n" — a raw U+000A
# inside a JSON string is exactly the parse error llama.cpp 500s on after the
# model imitates elided history containing one.
_ELIDE_ARGS_MARKER = "\\n...[arguments elided: %d chars total]...\\n"

_INCOMPLETE_U_RE = re.compile(r"\\u[0-9a-fA-F]{0,3}\Z")


def _backslash_run(s: str, i: int) -> int:
    """Count of consecutive backslashes ending at (and including) index i."""
    n = 0
    while i >= 0 and s[i] == "\\":
        n += 1
        i -= 1
    return n


def _trim_partial_escape(s: str) -> str:
    """Drop a trailing escape sequence cut in half by the head slice.

    A splice head ending in a dangling backslash or a partial \\uXXXX run
    re-joins into invalid JSON. Complete escapes (\\n, \\\\, \\u00e9) are kept.
    """
    if _backslash_run(s, len(s) - 1) % 2 == 1:
        s = s[:-1]                       # dangling backslash: never valid
    m = _INCOMPLETE_U_RE.search(s)
    if m and _backslash_run(s, m.start()) % 2 == 1:
        s = s[:m.start()]                # real (unescaped) partial \uXXXX
    return s


def _shorten_json_value(v, total: int):
    """Elide long strings inside a parsed arguments value tree."""
    if isinstance(v, str) and len(v) > ELIDE_MIN_CHARS:
        return (v[:ELIDE_HEAD] + "\n...[arguments elided: "
                f"{total} chars total]...\n" + v[-ELIDE_TAIL:])
    if isinstance(v, list):
        return [_shorten_json_value(x, total) for x in v]
    if isinstance(v, dict):
        return {k: _shorten_json_value(x, total) for k, x in v.items()}
    return v


def _elide_text(text: str) -> str:
    if len(text) <= ELIDE_MIN_CHARS + 8:
        return text
    return (text[:ELIDE_HEAD] + "\n...[tool output elided: "
            f"{len(text)} chars total]...\n" + text[-ELIDE_TAIL:])


def _shorten_args(args, max_len: int = 400) -> object:
    """Long tool-call arguments (e.g. file contents) get the same treatment.

    The result must still be a parseable JSON document: llama.cpp embeds the
    arguments string verbatim in the chat template, and a corrupted sample
    teaches the model to emit tool calls with unescaped control characters,
    which then fail json::parse with U+000A errors.
    """
    if not isinstance(args, str) or len(args) <= max_len + 8:
        return args
    # The usual cut lands inside the dominant string value; with the escaped
    # marker and a repaired head cut that re-joins as valid JSON. (The tail
    # slice never needs repair: a suffix is just literal continuation text.)
    spliced = (_trim_partial_escape(args[:ELIDE_HEAD])
               + (_ELIDE_ARGS_MARKER % len(args)) + args[-ELIDE_TAIL:])
    try:
        json.loads(spliced)
        return spliced
    except ValueError:
        pass
    # Awkward cut (e.g. the head landed between JSON tokens), or the original
    # was never valid JSON. If it was parseable, elide inside the parsed value
    # tree and re-serialize: guaranteed-parseable and still deterministic.
    try:
        obj = json.loads(args)
    except ValueError:
        return spliced      # wasn't valid JSON to begin with; best effort
    try:
        return json.dumps(_shorten_json_value(obj, len(args)),
                          separators=(",", ":"))
    except (ValueError, TypeError):
        return spliced


def _elide_tool_outputs(messages: list[dict], keep_recent: int) -> tuple[list[dict], int]:
    """Return (copy, elided_count). The last ``keep_recent`` messages are never
    touched; older tool outputs and oversized tool-call arguments are replaced
    with a short placeholder. Image payloads in the elided region are dropped
    (they are summarized away next if elision is not enough)."""
    n = len(messages)
    boundary = _tail_boundary(messages, keep_recent)
    out = []
    elided = 0
    for i, m in enumerate(messages):
        if i >= boundary:
            out.append(json.loads(json.dumps(m)))
            continue
        c = m.get("content")
        if m.get("role") == "tool" and isinstance(c, (str, list)):
            if isinstance(c, list):
                newc = []
                for p in c:
                    if isinstance(p, dict) and "text" in p and len(p.get("text", "")) > ELIDE_MIN_CHARS:
                        p = dict(p)
                        p["text"] = _elide_text(p["text"])
                        elided += 1
                    newc.append(p)
                m = dict(m)
                m["content"] = newc
            else:
                new = _elide_text(str(c))
                if new != c:
                    elided += 1
                m = dict(m)
                m["content"] = new
        tcs = m.get("tool_calls")
        if tcs:
            new_tcs = []
            changed = False
            for tc in tcs:
                fn = dict((tc or {}).get("function") or {})
                args = fn.get("arguments")
                short = _shorten_args(args)
                if short is not args:
                    fn["arguments"] = short
                    changed = True
                new_tcs.append(dict(tc or {}, function=fn) if changed else tc)
            if changed:
                m = dict(m)
                m["tool_calls"] = new_tcs
                elided += 1
        # drop image payloads in the elided region; slot resurrection in the
        # canonical path is unaffected because we never write this back
        if isinstance(c := m.get("content"), list) and any(
                isinstance(p, dict) and ("image_url" in p or "image" in p
                                         or p.get("type") in ("image_url", "image"))
                for p in c):
            m = dict(m)
            m["content"] = [p for p in c if not (isinstance(p, dict) and (
                "image_url" in p or "image" in p or p.get("type") in ("image_url", "image")))]
        out.append(m)
    return out, elided


# ---------------------------------------------------------------------------
# side-call summarizer

SUMMARY_MARKER = "[Previous conversation summary]"


def _summary_prompt(system: str, older: list[dict]) -> list[dict]:
    older_txt = _serialize_for_count(older)
    instr = (
        "Summarize the conversation below into a continuation brief for an AI "
        "coding agent. Be dense and concrete; keep:\n"
        "- the user's goal and any explicit instructions/constraints\n"
        "- decisions made and their reasons\n"
        "- exact file paths, branch/PR names, commands, error messages\n"
        "- open tool-call state: what was called, what results are pending\n"
        "- the exact next steps that were planned\n"
        "Do not add commentary. Plain text, under 1200 chars.\n\n"
        "CONVERSATION:\n" + older_txt[-60000:])
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system[:8000]})
    msgs.append({"role": "user", "content": instr})
    return msgs


def _summarize(backend, system: str, older: list[dict],
               max_tokens: int, timeout: float = 120.0) -> str | None:
    """Direct side-call to the same backend. NOT through process(): no
    recursion, no canonical-session write, no captures."""
    body = {
        "messages": _summary_prompt(system, older),
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    if getattr(backend, "default_model", ""):
        body["model"] = backend.default_model
    # llama_kwargs off for this call: summary must work regardless of template
    # kwargs handling, and thinking would eat the tiny token budget.
    try:
        req = urllib.request.Request(
            backend.chat_url(),
            data=json.dumps(body).encode(),
            headers=backend.headers())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if not text:
            text = msg.get("reasoning_content") or msg.get("reasoning") or ""
        return text.strip() or None
    except Exception as e:
        log.warning("compaction side-call failed for %s: %s", backend.base_url, e)
        return None


def compact_if_needed(cfg, body: dict, backend, meta: dict,
                      _summarize_fn=None) -> dict:
    """Return the (possibly compacted) body to forward. Never raises; on any
    failure the original body is returned unchanged (llama.cpp overflow error
    remains the existing degradation path). ``_summarize_fn`` allows tests to
    stub the side-call without any upstream server."""
    if not backend.compact_enabled:
        return body
    try:
        return _compact(cfg, body, backend, meta, _summarize_fn or _summarize)
    except Exception:
        log.exception("compaction failed; forwarding uncompacted body")
        return body


def _compact(cfg, body: dict, backend, meta: dict, summarize_fn) -> dict:
    msgs = body.get("messages") or []
    if len(msgs) < backend.compact_tail_msgs + 2:
        return body  # too small to compact; not worth a count
    if any(mk in _serialize_for_count(msgs) for mk in
           ("[Previous conversation summary]", "[Conversation summary]",
            "The conversation has been summarized")):
        # Cursor already compacted (or we did last turn via its marker);
        # do not double-compact. Note: our own summaries carry the marker too,
        # but they are regenerated fresh each turn from the canonical history,
        # so this only short-circuits Cursor-native summaries.
        meta["compact"] = {"stage": "skipped", "reason": "marker-present"}
        return body
    n_ctx = _context_tokens(backend)
    if not n_ctx:
        meta["compact"] = {"stage": "skipped", "reason": "no-n_ctx"}
        return body
    threshold = int(n_ctx * backend.compact_at)
    total = _token_count(backend, msgs)
    if total < threshold:
        meta["compact"] = {"stage": "under-threshold", "prompt_tokens_est": total,
                           "threshold": threshold}
        return body

    system = msgs[0] if msgs and msgs[0].get("role") == "system" else None
    rest = msgs[1:] if system else msgs
    stage = "elide"
    projected, elided = _elide_tool_outputs(rest, backend.compact_tail_msgs)
    candidate = ([system] if system else []) + projected
    size = _token_count(backend, candidate)
    if size >= threshold and len(rest) > backend.compact_tail_msgs:
        stage = "summarize"
        # keep the tail verbatim, summarize everything before it
        boundary = _tail_boundary(rest, backend.compact_tail_msgs)
        older = rest[:boundary]
        tail = rest[boundary:]
        summary = summarize_fn(backend, (system or {}).get("content", "") if system else "",
                               older, backend.compact_summary_max_tokens)
        if summary:
            summary_msg = {
                "role": "user",
                "content": SUMMARY_MARKER + "\n" + summary
                + "\n\nThe full history is preserved above this summary's source; "
                  "continue seamlessly from the next steps listed.",
            }
            candidate = ([system] if system else []) + [summary_msg] + tail
            size = _token_count(backend, candidate)
        else:
            meta["compact"] = {"stage": "summarize-failed", "prompt_tokens_est": size,
                               "threshold": threshold}
            return body  # fallback: uncompacted (existing overflow behavior)
        if size >= threshold:
            meta["compact"] = {"stage": "still-over-threshold", "prompt_tokens_est": size,
                               "threshold": threshold, "n_ctx": n_ctx}
            return body
    meta["compact"] = {"stage": stage, "elided_outputs": elided,
                       "prompt_tokens_est": size, "threshold": threshold,
                       "n_ctx": n_ctx, "msgs_in": len(msgs), "msgs_out": len(candidate)}
    out = dict(body)
    out["messages"] = candidate
    return out
