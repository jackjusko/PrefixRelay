#!/usr/bin/env python3
"""Live smoke test for the routing table.

Sends one tiny request per route *through the proxy* (so the routing decision,
upstream id, auth header and usage dialect all get exercised) and prints what
came back.  Use --tools to check tool calling, which Cursor depends on for
agent turns regardless of the selected name.

    .venv/bin/python scripts/probe_upstreams.py                    # all routes
    .venv/bin/python scripts/probe_upstreams.py glm-flash --tools  # one route, tools on
    .venv/bin/python scripts/probe_upstreams.py --list-models      # hosted catalog
"""
import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from proxy.config import Config  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {"name": "echo", "description": "Echo text back",
                 "parameters": {"type": "object",
                                "properties": {"text": {"type": "string"}},
                                "required": ["text"]}},
}]


def list_models(cfg: Config, backend: str, grep: str) -> None:
    be = cfg.backend_for(backend)
    if be is None:
        sys.exit(f"unknown backend {backend!r}; have {sorted(cfg.backends)}")
    url = be.base_url.rstrip("/") + ("/models" if be.base_url.rstrip("/").endswith("/v1")
                                      else "/v1/models")
    r = httpx.get(url, headers=be.headers(), timeout=60)
    r.raise_for_status()
    rows = r.json().get("data") or []
    hits = [m for m in rows if grep.lower() in str(m.get("id", ""))]
    print(f"{len(hits)}/{len(rows)} models match {grep!r} on {backend}")
    for m in hits[:40]:
        sp = set(m.get("supported_parameters") or [])
        pr = m.get("pricing") or {}
        print(f"  {m['id']:44} ctx={m.get('context_length')} "
              f"tools={'tools' in sp} reasoning={'reasoning' in sp} "
              f"in=${float(pr.get('prompt', 0)) * 1e6:.2f}/M "
              f"out=${float(pr.get('completion', 0)) * 1e6:.2f}/M")


def probe(cfg: Config, proxy: str, name: str, tools: bool, stream: bool,
          max_tokens: int | None = None) -> bool:
    route, be, source = cfg.resolve(name)
    upstream = cfg.upstream_model_for(route, be)
    body = {"model": name,
            "messages": [{"role": "user", "content": "Reply with exactly: PROBE-OK"}]}
    if max_tokens:
        body["max_tokens"] = max_tokens
    if tools:
        body["tools"] = TOOLS
        body["tool_choice"] = {"type": "function", "function": {"name": "echo"}}
    if stream:
        body["stream"] = True
    headers = {"authorization": "Bearer " + (cfg.token() or "")}
    t0 = time.time()
    try:
        with httpx.stream("POST", proxy.rstrip("/") + "/v1/chat/completions",
                          json=body, headers=headers, timeout=180) as r:
            if r.status_code != 200:
                r.read()
                print(f"{name:14} HTTP {r.status_code}: {r.text[:300]}")
                return False
            if not stream:
                r.read()
                obj = r.json()
            else:
                obj, text, reasoning = {}, [], ""
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("", "[DONE]"):
                        continue
                    try:
                        c = json.loads(payload)
                    except Exception:
                        continue
                    obj.setdefault("model", c.get("model"))
                    if c.get("usage"):
                        obj["usage"] = c["usage"]
                    for ch in c.get("choices") or []:
                        d = ch.get("delta") or {}
                        if d.get("content"):
                            text.append(d["content"])
                        if d.get("reasoning") or d.get("reasoning_content"):
                            reasoning += d.get("reasoning") or d.get("reasoning_content")
                        if ch.get("finish_reason"):
                            obj["finish_reason"] = ch["finish_reason"]
                obj["text"] = "".join(text)
                obj["reasoning_head"] = reasoning[:80]
    except Exception as e:
        print(f"{name:14} ERROR {e!r}")
        return False
    dt = time.time() - t0
    u = obj.get("usage") or {}
    det = u.get("prompt_tokens_details") or {}
    msg = (obj.get("choices") or [{}])[0].get("message") or {}
    tcs = msg.get("tool_calls") or []
    content = msg.get("content") if not stream else obj.get("text")
    est = be.estimate(u)
    print(f"{name:14} {dt:5.1f}s src={source:7} backend={be.name} up={upstream}")
    choice = (obj.get("choices") or [{}])[0]
    print(f"{'':14} returned model={obj.get('model')!r} "
          f"finish={choice.get('finish_reason') or obj.get('finish_reason')}")
    print(f"{'':14} usage in={u.get('prompt_tokens')} out={u.get('completion_tokens')} "
          f"cached={det.get('cached_tokens')} reasoning="
          f"{(u.get('completion_tokens_details') or {}).get('reasoning_tokens')}"
          + (f" est=${est:.6f}" if est else ""))
    if obj.get("reasoning_head") is not None and stream:
        print(f"{'':14} reasoning_head={obj['reasoning_head']!r}")
    if tcs:
        print(f"{'':14} tool_calls={json.dumps(tcs)[:200]}")
    print(f"{'':14} content={(content or '')[:120]!r}")
    ok = bool(content or tcs)
    if not ok:
        print(f"{'':14} !! no content and no tool call")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="route names (default: all model routes)")
    ap.add_argument("--proxy", default="http://127.0.0.1:8081")
    ap.add_argument("--tools", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="cap output for this probe only; note routes may drop max_tokens")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--backend", default="openrouter")
    ap.add_argument("--grep", default="glm")
    a = ap.parse_args()
    cfg = Config.load()
    if a.list_models:
        list_models(cfg, a.backend, a.grep)
        return
    names = a.names or cfg.model_names()
    ok = 0
    for n in names:
        if n not in cfg.by_name:
            print(f"{n:14} not a configured route; have {cfg.model_names()}")
            continue
        ok += probe(cfg, a.proxy, n, a.tools, a.stream, a.max_tokens)
    print(f"{ok}/{len(names)} routes answered")
    sys.exit(0 if ok == len(names) else 1)


if __name__ == "__main__":
    main()
