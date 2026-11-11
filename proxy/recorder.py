"""Metadata recording and optional request/response body capture.

Each request gets an index row with timing/usage metadata. Full bodies are
written only when explicitly enabled. Auth headers are redacted before disk.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

_SENSITIVE_MARKERS = ("authorization", "api-key", "token", "secret", "password", "cookie")


def _safe_headers(headers: dict) -> dict:
    return {k: ("<redacted>" if any(marker in k.casefold()
                                    for marker in _SENSITIVE_MARKERS) else v)
            for k, v in headers.items()}


def parse_sse(buf: bytes) -> tuple[dict | None, str]:
    """(last usage object, concatenated completion text) from an SSE body."""
    usage = None
    parts: list[str] = []
    for line in buf.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if payload in (b"", b"[DONE]"):
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if isinstance(obj, dict):
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices") or []:
                c = (ch.get("delta") or {}).get("content")
                if isinstance(c, str):
                    parts.append(c)
    return usage, "".join(parts)


def parse_json_body(buf: bytes) -> tuple[dict | None, str]:
    try:
        obj = json.loads(buf)
    except Exception:
        return None, ""
    if not isinstance(obj, dict):
        return None, ""
    usage = obj.get("usage")
    text = ""
    try:
        text = obj["choices"][0]["message"].get("content") or ""
    except Exception:
        pass
    return usage, text


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bodies: Path = cfg.path(cfg.bodies_dir)
        self.bodies.mkdir(parents=True, exist_ok=True)
        self.index: Path = cfg.path(cfg.captures_dir) / "index.jsonl"
        self.index.parent.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, dict] = {}
        self._lock = threading.Lock()

    def begin(self, rid: str, headers: dict, raw: bytes) -> None:
        if self.cfg.record_bodies:
            (self.bodies / f"{rid}.recv.json").write_bytes(raw)
        with self._lock:
            self._open[rid] = {
                "rid": rid,
                "ts": time.time(),
                "headers": _safe_headers(headers),
                "req_bytes": len(raw),
            }

    def set_forwarded(self, rid: str, payload: bytes, route: str, meta: dict | None = None) -> None:
        if self.cfg.record_bodies:
            (self.bodies / f"{rid}.fwd.json").write_bytes(payload)
        with self._lock:
            rec = self._open.get(rid)
            if rec is not None:
                rec["route"] = route
                rec["fwd_bytes"] = len(payload)
                if meta:
                    rec["meta"] = meta

    def finish(self, rid: str, raw: bytes | None, status: int = 0,
               dur_ms: float = 0, ttft_ms: float | None = None,
               error: str | None = None) -> None:
        with self._lock:
            rec = self._open.pop(rid, None)
        rec = rec or {"rid": rid, "ts": time.time()}
        rec["status"] = status
        rec["dur_ms"] = round(dur_ms, 1)
        if ttft_ms is not None:
            rec["ttft_ms"] = round(ttft_ms, 1)
        usage = None
        text = ""
        if raw is not None:
            if raw.lstrip()[:1] == b"{":
                usage, text = parse_json_body(raw)
            else:
                usage, text = parse_sse(raw)
            rec["resp_bytes"] = len(raw)
            if self.cfg.record_bodies and self.cfg.log_response_bodies:
                cap = self.cfg.response_body_cap
                (self.bodies / f"{rid}.resp").write_bytes(raw[:cap])
        rec["usage"] = usage
        if self.cfg.record_bodies and text:
            rec["text_head"] = text[:300]
        if error:
            rec["error"] = error
        self._write(rec)

    def _write(self, row: dict) -> None:
        with self._lock:
            with self.index.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
