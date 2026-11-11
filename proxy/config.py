"""Configuration loading for PrefixRelay.

Two eras coexist here.  Legacy configs name exactly one model (``model_id``)
behind one upstream and every request is normalized to it.  Routing configs
name a table: each model name Cursor may send maps to a backend plus the real
upstream model id, so a local KV-cached engine and a hosted API can share one
proxy.  When ``routes:`` is absent the legacy behavior is synthesized, so
existing configs and rollback copies keep working unchanged.
"""
from __future__ import annotations

import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _expand_env(v: str) -> str:
    return os.path.expandvars(v) if isinstance(v, str) else v


def _public_url(value: str) -> str:
    """Remove credentials from URLs before they appear in diagnostics."""
    try:
        parts = urlsplit(value)
        host = parts.netloc.rsplit("@", 1)[-1]
        query = [(k, "<redacted>" if any(s in k.casefold() for s in
                  ("key", "token", "secret", "auth")) else v)
                 for k, v in parse_qsl(parts.query, keep_blank_values=True)]
        return urlunsplit((parts.scheme, host, parts.path, urlencode(query), parts.fragment))
    except Exception:
        return "<configured>"


@dataclass
class Backend:
    """One upstream server plus how to talk to it.

    ``stabilize`` is the KV-cache switch: only backends with a token-prefix
    cache worth protecting get the append-only rewrite, session writes and
    image resurrection.  ``usage_style`` picks the dialect for asking the
    upstream to report usage (many local servers want ``stream_options``;
    some hosted APIs want ``usage: {include: true}``).
    """

    name: str
    base_url: str
    stabilize: bool = True
    api_key: str = ""
    default_model: str = ""                      # upstream id when a route names none
    usage_style: str = "stream_options"          # stream_options | usage_object | none
    reasoning_field: str = "auto"                # auto | reasoning | reasoning_content | none
    wrap_reasoning: bool = True                  # think-tag wrap for Cursor
    defang_specials: bool = True                 # ZWSP-guard <|im_*|> in upstream output
    llama_kwargs: bool = True                    # chat_template_kwargs allowed upstream
    extra_headers: dict = field(default_factory=dict)
    pricing: dict = field(default_factory=dict)  # per-token; /stats estimates only
    # -- context compaction (enabled by default; unknown limits safely skip) -
    # context_tokens: per-slot context window; 0 = discover via /props and cache.
    compact_enabled: bool = True
    context_tokens: int = 0                      # 0 = auto-discover from /props
    compact_at: float = 0.8                      # fraction of context that triggers
    compact_tail_msgs: int = 8                   # recent messages always kept verbatim
    compact_summary_max_tokens: int = 1024       # side-call summary budget
    raw: dict = field(default_factory=dict, repr=False)  # YAML spec, for backend extras

    def chat_url(self) -> str:
        """Full chat-completions URL.  A base ending in /v1 already carries the
        version segment; appending it again would 404 on OpenRouter."""
        b = self.base_url.rstrip("/")
        return b + "/chat/completions" if b.endswith("/v1") else b + "/v1/chat/completions"

    def headers(self) -> dict:
        h = {"content-type": "application/json"}
        if self.api_key:
            h["authorization"] = "Bearer " + self.api_key
        h.update({k: str(v) for k, v in (self.extra_headers or {}).items()})
        return h

    def estimate(self, usage: dict | None) -> float | None:
        if not usage or not self.pricing:
            return None
        pt = usage.get("prompt_tokens") or 0
        ct = usage.get("completion_tokens") or 0
        return (pt * float(self.pricing.get("prompt", 0))
                + ct * float(self.pricing.get("completion", 0)))

    def public(self) -> dict:
        """Healthz view - never includes the key."""
        return {"base_url": _public_url(self.base_url), "stabilize": self.stabilize,
                "usage_style": self.usage_style, "reasoning_field": self.reasoning_field,
                "wrap_reasoning": self.wrap_reasoning, "defang_specials": self.defang_specials,
                "llama_kwargs": self.llama_kwargs, "default_model": self.default_model,
                "has_key": bool(self.api_key),
                "compact_enabled": self.compact_enabled}


@dataclass
class ModelRoute:
    """A name Cursor may put in ``model`` and where it leads.

    ``extra_body`` holds request defaults for this model only, applied with
    setdefault semantics so a value Cursor actually sent still wins - e.g.
    pinning GLM's thinking effort to high without fighting the caller.
    """

    name: str
    backend: str
    upstream_model: str = ""
    context_tokens: int = 0                      # per-model override; 0 = backend discovery/config
    kind: str = "model"                          # model | alias
    extra_body: dict = field(default_factory=dict)
    drop_body: list = field(default_factory=list)  # request keys to strip before forwarding


@dataclass
class Config:
    host: str = "127.0.0.1"
    port: int = 8081
    mode: str = "stabilize"                   # stabilize | record
    upstream: str = "http://127.0.0.1:1234/v1"
    sidecar: str = "http://127.0.0.1:1234/v1"
    model_id: str = "local-model"
    sidecar_model: str = ""
    default_model: str = ""                   # route name for unknown/absent model ids
    token_file: str = "token.txt"
    enforce_auth: bool = True
    force_usage_in_stream: bool = True
    captures_dir: str = "captures"
    bodies_dir: str = "bodies"
    state_dir: str = "state"
    record_bodies: bool = False                 # full prompt/response capture is opt-in
    log_response_bodies: bool = False
    response_body_cap: int = 2_000_000
    stream_reasoning_to_client: bool = True
    stream_reasoning_as_think_tags: bool = True
    # Client-key bootstrap is disabled by default. Prefer a generated token
    # file so a public tunnel never accepts an arbitrary first caller.
    learn_first_key: bool = False
    learned_keys_file: str = "state/learned_keys.txt"
    raw: dict = field(default_factory=dict, repr=False)
    # Routing tables, filled by build_routing() from raw + legacy fields.
    backends: dict = field(default_factory=dict, repr=False)   # name -> Backend
    routes: list = field(default_factory=list, repr=False)     # ModelRoute (models + aliases)
    by_name: dict = field(default_factory=dict, repr=False)    # name -> ModelRoute
    routing_enabled: bool = False
    default_route: ModelRoute | None = field(default=None, repr=False)
    kv_backends: int = 0

    # -- routing ------------------------------------------------------------
    def build_routing(self) -> "Config":
        """(Re)build backends/routes from raw YAML, or synthesize the legacy
        single-model behavior when no ``routes:`` table is present."""
        self.backends = {}
        self.routes = []
        self.by_name = {}
        self.routing_enabled = bool(self.raw.get("routes"))

        for name, spec in (self.raw.get("backends") or {}).items():
            spec = spec if isinstance(spec, dict) else {"base_url": str(spec)}
            key = str(spec.get("api_key") or "").strip()
            if not key and spec.get("api_key_file"):
                kf = self.path(str(spec["api_key_file"]))
                if kf.exists():
                    key = kf.read_text().strip()
            if not key and spec.get("api_key_env"):
                key = os.environ.get(str(spec["api_key_env"]), "")
            self.backends[str(name)] = Backend(
                name=str(name),
                base_url=_expand_env(str(spec.get("base_url", ""))),
                stabilize=bool(spec.get("stabilize", True)),
                api_key=key,
                default_model=str(spec.get("default_model", "") or ""),
                usage_style=str(spec.get("usage_style", "stream_options")).lower(),
                reasoning_field=str(spec.get("reasoning_field", "auto")).lower(),
                wrap_reasoning=bool(spec.get("wrap_reasoning", True)),
                defang_specials=bool(spec.get("defang_specials", True)),
                llama_kwargs=bool(spec.get("llama_kwargs", True)),
                extra_headers=dict(spec.get("extra_headers") or {}),
                pricing=dict(spec.get("pricing") or {}),
                    compact_enabled=bool(spec.get("compact_enabled", True)),
                context_tokens=int(spec.get("context_tokens", 0) or 0),
                compact_at=float(spec.get("compact_at", 0.8)),
                compact_tail_msgs=int(spec.get("compact_tail_msgs", 8)),
                compact_summary_max_tokens=int(spec.get("compact_summary_max_tokens", 1024)),
                raw=dict(spec),
            )

        if self.routing_enabled:
            for spec in self.raw.get("routes") or []:
                if not isinstance(spec, dict) or not spec.get("name"):
                    continue
                bname = str(spec.get("backend") or "")
                if bname not in self.backends:
                    raise ValueError(f"route {spec['name']!r} names unknown backend {bname!r}")
                self._add(ModelRoute(
                    name=str(spec["name"]), backend=bname,
                    upstream_model=str(spec.get("upstream_model", "") or ""),
                    context_tokens=int(spec.get("context_tokens", 0) or 0),
                    extra_body=dict(spec.get("extra_body") or {}),
                    drop_body=[str(k) for k in (spec.get("drop_body") or [])]))
            for alias, target in (self.raw.get("aliases") or {}).items():
                tgt = self.by_name.get(str(target))
                if tgt is not None:
                    self._add(ModelRoute(name=str(alias), backend=tgt.backend,
                                         upstream_model=tgt.upstream_model,
                                         context_tokens=tgt.context_tokens,
                                         extra_body=dict(tgt.extra_body),
                                         drop_body=list(tgt.drop_body), kind="alias"))
            want = self.default_model or (self.routes[0].name if self.routes else "")
            self.default_route = self.by_name.get(want) or (self.routes[0] if self.routes else None)
        else:
            # Legacy: everything normalized onto model_id behind ``upstream``.
            self.backends = {
                "local": Backend(name="local", base_url=self.upstream, stabilize=True,
                                  default_model=self.model_id),
                "sidecar": Backend(name="sidecar", base_url=self.sidecar, stabilize=True,
                                   default_model=self.sidecar_model),
            }
            legacy = ModelRoute(name=self.model_id, backend="local", upstream_model=self.model_id)
            self._add(legacy)
            self.default_route = legacy

        # Only backends a *named route* can land on count for session
        # namespacing: the sidecar is reached by shape rule, not by name, and
        # re-keying sessions needlessly would orphan the parked KV cache.
        self.kv_backends = len({r.backend for r in self.routes
                                if r.kind == "model" and self.backends[r.backend].stabilize})
        return self

    def _add(self, route: ModelRoute) -> None:
        self.by_name[route.name] = route
        self.routes.append(route)

    def resolve(self, name: str | None) -> tuple[ModelRoute, Backend, str]:
        """Return (route, backend, source) for a requested model name.

        source is exact | alias | default | empty.  Unknown names land on
        ``default_model``, which keeps Cursor's key-validation challenge and
        any unlisted name off paid APIs by construction.
        """
        route: ModelRoute | None = None
        source = "empty"
        key = (name or "").strip()
        if key and key in self.by_name:
            route = self.by_name[key]
            source = "exact" if route.kind == "model" else "alias"
        if route is None:
            route = self.default_route
            if route is None:
                raise ValueError("no model routes configured")
            source = "default"
        return route, self.backends[route.backend], source

    def backend_for(self, name: str) -> Backend | None:
        return self.backends.get(name)

    def model_names(self) -> list[str]:
        """Names to publish in /v1/models and Cursor's availableAPIKeyModels."""
        return [r.name for r in self.routes if r.kind == "model"]

    def alias_names(self) -> list[str]:
        return [r.name for r in self.routes if r.kind == "alias"]

    def upstream_model_for(self, route: ModelRoute, backend: Backend) -> str:
        return route.upstream_model or backend.default_model

    def models_payload(self) -> dict:
        data = []
        for r in self.routes:
            entry = {"id": r.name, "object": "model", "created": 0, "owned_by": r.backend}
            if r.kind == "alias":
                entry["alias_of"] = self.upstream_model_for(r, self.backends[r.backend]) or r.backend
            data.append(entry)
        return {"object": "list", "data": data}

    def health(self) -> dict:
        db = self.default_route and self.backends[self.default_route.backend]
        return {
            "ok": True, "mode": self.mode,
            "upstream": _public_url(db.base_url) if db else _public_url(self.upstream),
            "model": (self.upstream_model_for(self.default_route, db)
                      if self.default_route and db else self.model_id),
            "default_model": self.default_route.name if self.default_route else None,
            "routing": self.routing_enabled,
            "kv_backends": self.kv_backends,
            "backends": {n: b.public() for n, b in self.backends.items()},
            "routes": [{"name": r.name, "backend": r.backend, "kind": r.kind,
                        "upstream_model": self.upstream_model_for(r, self.backends[r.backend]),
                        "context_tokens": r.context_tokens or self.backends[r.backend].context_tokens or None,
                        **({"extra_body": r.extra_body} if r.extra_body else {}),
                        **({"drop_body": r.drop_body} if r.drop_body else {})}
                       for r in self.routes],
        }

    @classmethod
    def load(cls, path: str | None = None) -> "Config":
        p = Path(path or os.environ.get("CPROXY_CONFIG", ROOT / "config.yaml"))
        data: dict = {}
        if p.exists():
            data = yaml.safe_load(p.read_text()) or {}
        known = {f.name for f in fields(cls)} - {"raw"}
        cfg = cls(**{k: v for k, v in data.items() if k in known})
        cfg.raw = data
        return cfg.build_routing()

    def token(self) -> str:
        tf = Path(self.token_file)
        if not tf.is_absolute():
            tf = ROOT / tf
        if tf.exists():
            return tf.read_text().strip()
        return os.environ.get("CPROXY_TOKEN", "")

    def path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p
