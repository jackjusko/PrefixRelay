#!/usr/bin/env python3
"""Offline unit test for model routing: no network, no GPU, no Cursor.

Writes a temp config with a fake local (KV-cached) backend and a fake hosted
backend, swaps app.client for an httpx.MockTransport that records exactly what
we would have put on the wire, then drives the real FastAPI app.  Asserts the
decisions that matter: which server a name selects, which upstream id it gets,
which usage dialect it is asked for, whether the KV pipeline touched it, and
what model name Cursor sees back.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="cproute-"))
CFG = tmp / "config.yaml"
proxy_key = "test-cursor-proxy-key"
(tmp / "proxy.token").write_text(proxy_key)
CFG.write_text(f"""
port: 8085
mode: stabilize
default_model: local-model
enforce_auth: true
token_file: {tmp}/proxy.token
learn_first_key: false
state_dir: {tmp}/state
captures_dir: {tmp}/captures
bodies_dir: {tmp}/bodies
backends:
  local:
    base_url: http://local.test:9001
    stabilize: true
    default_model: engine-local-id
  sidecar:
    base_url: http://sidecar.test:9003
    stabilize: true
    default_model: side-id
    force_thinking_off: true
    rules:
      - system_re: "title"
        user_re: "TITLE-PROBE-ME"
        max_msgs: 2
        max_tokens: 64
  hosted:
    base_url: https://hosted.test/api/v1
    stabilize: false
    api_key: test-upstream-key
    usage_style: usage_object
    reasoning_field: auto
    defang_specials: false
    llama_kwargs: false
    pricing: {{prompt: 0.00000015, completion: 0.0000005}}
    extra_headers: {{X-Title: test}}
routes:
  - name: local-model
    backend: local
    upstream_model: engine-local-id
  - name: hosted-model
    backend: hosted
    upstream_model: vendor/model-x
    context_tokens: 65536
    extra_body:
      reasoning: {{effort: high}}
    drop_body: [max_tokens]
aliases:
  chip-name: local-model
""")
os.environ["CPROXY_CONFIG"] = str(CFG)

from fastapi.testclient import TestClient  # noqa: E402

from proxy import app as app_module  # noqa: E402
from proxy import pipeline  # noqa: E402
from proxy.config import Config  # noqa: E402

sent: list[dict] = []

HOSTED_SSE = [
    {"model": "vendor/model-x", "choices": [{"index": 0, "delta": {"role": "assistant"},
                                             "finish_reason": None}]},
    {"model": "vendor/model-x", "choices": [{"index": 0, "delta": {"reasoning": "deep  thought"},
                                             "finish_reason": None}]},
    {"model": "vendor/model-x", "choices": [{"index": 0, "delta": {"content": "answer 42  <|im_end|>"},
                                             "finish_reason": None}]},
    {"model": "vendor/model-x", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 40, "completion_tokens": 10}},
]
LOCAL_SSE = [
    {"model": "engine-local-id", "choices": [{"index": 0, "delta": {"reasoning_content": "local thought"},
                                              "finish_reason": None}]},
    {"model": "engine-local-id", "choices": [{"index": 0, "delta": {"content": "local answer"},
                                              "finish_reason": "stop"}],
     "usage": {"prompt_tokens": 30, "completion_tokens": 5,
               "prompt_tokens_details": {"cached_tokens": 28}}},
]


class _Stream(httpx.AsyncByteStream):
    """Minimal once-only async body, so the app's stream=True path is real."""

    def __init__(self, data: bytes):
        self._data = data

    async def __aiter__(self):
        yield self._data

    async def aclose(self) -> None:
        pass


def _sse(chunks):
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks).encode() + b"data: [DONE]\n\n"


def handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    hosted = "hosted.test" in str(request.url)
    sent.append({"url": str(request.url), "headers": dict(request.headers), "body": body,
                 "hosted": hosted})
    if body.get("stream"):
        return httpx.Response(200, stream=_Stream(_sse(HOSTED_SSE if hosted else LOCAL_SSE)),
                              headers={"content-type": "text/event-stream"})
    return httpx.Response(200, json={
        "id": "x", "object": "chat.completion", "created": 1, "model": body.get("model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 4,
                  "prompt_tokens_details": {"cached_tokens": 18}}})


app_module.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
tc = TestClient(app_module.app)

fails: list[str] = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


check(tc.get("/healthz").status_code == 401, "healthz must require the proxy key")
check(tc.get("/v1/models").status_code == 401, "model list must require the proxy key")
check(tc.post("/v1/chat/completions", json={"messages": []}).status_code == 401,
      "chat completions must require the proxy key")


def post(model, stream=False, messages=None, **extra):
    sent.clear()
    body = {"messages": messages or [{"role": "user", "content": "routing probe " + str(model)}]}
    body.update(extra)
    if model is not None:
        body["model"] = model
    if stream:
        body["stream"] = True
    r = tc.post("/v1/chat/completions", json=body,
                headers={"authorization": "Bearer " + proxy_key})
    check(r.status_code == 200, f"{model}: HTTP {r.status_code} {r.text[:200]}")
    return r, (sent[0] if sent else None)


def sse_events(text):
    return [json.loads(l[5:].strip()) for l in text.splitlines()
            if l.startswith("data:") and l[5:].strip() not in ("", "[DONE]")]


sessions_dir = tmp / "state" / "sessions"

# 1. local name -> local backend, engine id, no auth header, stream_options dialect
r, req = post("local-model", stream=True)
check(req["url"] == "http://local.test:9001/v1/chat/completions", f"local url: {req['url']}")
check("authorization" not in req["headers"], "local backend must not get an auth header")
check(req["body"]["model"] == "engine-local-id", f"local upstream id: {req['body']['model']}")
check(req["body"].get("stream_options", {}).get("include_usage") is True, "local stream_options")
check("usage" not in req["body"], "local backend must not get usage.include")
evs = sse_events(r.text)
check(all(e["model"] == "local-model" for e in evs), "local stream must echo requested name")
first = evs[0]["choices"][0]["delta"].get("content", "")
check(first.startswith("<think>\nlocal thought"),
      f"local reasoning not think-wrapped: {first!r}")

# 2. hosted name -> hosted backend, vendor id, auth header, usage dialect
r, req = post("hosted-model", stream=True)
check(req["url"] == "https://hosted.test/api/v1/chat/completions",
      f"hosted url must not double /v1: {req['url']}")
check(req["headers"].get("authorization") == "Bearer test-upstream-key", "hosted auth header")
check(req["headers"].get("x-title") == "test", "hosted extra_headers not sent")
check(req["body"]["model"] == "vendor/model-x", f"hosted upstream id: {req['body']['model']}")
check(req["body"].get("usage", {}).get("include") is True, "hosted usage.include")
check("stream_options" not in req["body"], "hosted backend must not get stream_options")
check("chat_template_kwargs" not in req["body"], "llama-only kwargs leaked to hosted backend")
check(req["body"].get("reasoning") == {"effort": "high"},
      f"route extra_body not applied: {req['body'].get('reasoning')}")
evs = sse_events(r.text)
check(all(e["model"] == "hosted-model" for e in evs), "hosted stream must echo requested name")
joined = "".join((e["choices"][0].get("delta") or {}).get("content", "") for e in evs)
check("<think>\ndeep  thought" in joined, f"hosted reasoning not think-wrapped: {joined!r}")
check("answer 42" in joined, f"hosted answer lost: {joined!r}")
check("</think>" in joined, f"think block never closed: {joined!r}")
check(all("reasoning" not in (e["choices"][0].get("delta") or {}) for e in evs),
      "reasoning field should be consumed into the think wrap, not forwarded raw")
check("<|im_end|>" in joined,
      f"defang_specials=false must pass chat specials byte-exact: {joined!r}")

# 3. hosted route must not create a canonical session (no KV cache to protect)
before = set(sessions_dir.glob("*.json")) if sessions_dir.exists() else set()
post("hosted-model")
after = set(sessions_dir.glob("*.json")) if sessions_dir.exists() else set()
check(before == after, "hosted route wrote a session file")

# 4. non-stream response model is echoed back as requested
r, req = post("hosted-model")
check(r.json()["model"] == "hosted-model", f"non-stream echo: {r.json().get('model')}")

# 4b. no thinking cap: this route strips Cursor's max_tokens, others keep it
r, req = post("hosted-model", max_tokens=120)
check("max_tokens" not in req["body"],
      f"route must strip max_tokens so reasoning is not capped: {req['body'].get('max_tokens')}")
check(req["body"].get("reasoning") == {"effort": "high"}, "effort default lost after drop")
r, req = post("local-model", max_tokens=120)
check(req["body"].get("max_tokens") == 120,
      "max_tokens must survive on routes that do not drop it")

# 4c. shape rules shunt side requests to the sidecar - but never for a hosted
# route, which must not spend credits or get llama-only kwargs
title_msgs = [{"role": "system", "content": "You generate titles"},
              {"role": "user", "content": "TITLE-PROBE-ME please"}]
r, req = post("local-model", messages=title_msgs, max_tokens=900)
check(req["url"] == "http://sidecar.test:9003/v1/chat/completions",
      f"side request not shunted: {req['url']}")
check(req["body"]["model"] == "side-id", f"sidecar model id: {req['body']['model']}")
check(req["body"]["max_tokens"] == 64, f"sidecar max_tokens cap: {req['body']['max_tokens']}")
check(req["body"].get("chat_template_kwargs", {}).get("enable_thinking") is False,
      "sidecar thinking not forced off")
r, req = post("hosted-model", messages=title_msgs, max_tokens=900)
check("hosted.test" in req["url"],
      f"hosted route must not be shunted to the sidecar: {req['url']}")
check("chat_template_kwargs" not in req["body"], "llama kwargs leaked via sidecar path")
strow = [x for x in tc.get("/stats?limit=50",
                           headers={"authorization": "Bearer " + proxy_key}).json()["rows"]
         if x.get("model") == "local-model" and x.get("route") == "sidecar"]
check(strow and strow[-1]["backend"] == "sidecar",
      f"sidecar turn misattributed in stats: {strow[-1] if strow else None}")

# 5. alias resolves to its target backend; unknown name falls back to default
r, req = post("chip-name")
check(req["url"].startswith("http://local.test:9001"), f"alias backend: {req['url']}")
r, req = post("some-unlisted-chip")
check(req["url"].startswith("http://local.test:9001"), f"unknown -> default backend: {req['url']}")
check(req["body"]["model"] == "engine-local-id", "unknown name must normalize to default upstream")

# 6. /v1/models publishes routes and aliases
data = tc.get("/v1/models", headers={"authorization": "Bearer " + proxy_key}).json()["data"]
ids = [d["id"] for d in data]
check({"local-model", "hosted-model", "chip-name"} <= set(ids), f"/v1/models ids: {ids}")
hosted_entry = next(d for d in data if d["id"] == "hosted-model")
check(hosted_entry["owned_by"] == "hosted", f"owned_by: {hosted_entry}")
alias_entry = next(d for d in data if d["id"] == "chip-name")
check(alias_entry.get("alias_of") == "engine-local-id", f"alias_of: {alias_entry}")

# 7. healthz exposes the table but never the key
h = tc.get("/healthz", headers={"authorization": "Bearer " + proxy_key}).json()
raw = json.dumps(h)
check("test-upstream-key" not in raw and proxy_key not in raw, "healthz leaked a key")
check(h["backends"]["hosted"]["has_key"] is True and h["backends"]["local"]["has_key"] is False,
      "has_key flags wrong")
check(h["default_model"] == "local-model", f"default_model: {h.get('default_model')}")
check(any(r_["name"] == "hosted-model" and r_["extra_body"] == {"reasoning": {"effort": "high"}}
          for r_ in h["routes"]), f"healthz routes: {h['routes']}")
check(next(r_ for r_ in h["routes"] if r_["name"] == "hosted-model")["context_tokens"] == 65536,
      "route-specific context limit not exposed")

# 8. /stats splits by backend and only reports cache ratio for KV backends
st = tc.get("/stats?limit=50", headers={"authorization": "Bearer " + proxy_key}).json()
bb = st["summary"]["by_backend"]
check({"local", "hosted"} <= set(bb), f"by_backend: {list(bb)}")
check(bb["hosted"]["kv_cache"] is False and bb["hosted"]["cache_ratio"] is None,
      f"hosted must not report a cache ratio: {bb['hosted']}")
check(bb["hosted"]["est_cost"] and bb["hosted"]["est_cost"] > 0, f"no cost estimate: {bb['hosted']}")
check(bb["local"]["kv_cache"] is True and bb["local"]["cache_ratio"] not in (None, 0),
      f"local cache ratio missing: {bb['local']}")
check(st["summary"]["cache_ratio"] is not None, "summary.cache_ratio lost (README watch depends on it)")
row = [x for x in st["rows"] if x.get("backend") == "hosted"]
check(row and row[-1]["model"] == "hosted-model"
      and row[-1]["model_upstream"] == "vendor/model-x", f"row meta: {row[-1] if row else None}")

# 9. two KV backends get separate canonical histories
alt = Config.load(str(CFG))
alt.raw["backends"]["local2"] = {"base_url": "http://local2.test:9002", "stabilize": True,
                                "default_model": "engine-two"}
alt.raw["routes"].append({"name": "local2-model", "backend": "local2",
                          "upstream_model": "engine-two"})
alt.build_routing()
check(alt.kv_backends == 2, f"kv_backends should be 2, got {alt.kv_backends}")
pipeline._store = None
pipeline._sidecar_rules = None
msgs = [{"role": "user", "content": "same conversation, two engines"}]
_, _, m1 = pipeline.process(alt, {"model": "local-model", "messages": msgs}, "n1",
                            backend=alt.resolve("local-model")[1])
_, _, m2 = pipeline.process(alt, {"model": "local2-model", "messages": msgs}, "n2",
                            backend=alt.resolve("local2-model")[1])
check(m1["session"] != m2["session"], f"sessions collided across KV backends: {m1} {m2}")
check(m1["session"].startswith("local-") and m2["session"].startswith("local2-"),
      f"session namespaces: {m1['session']} {m2['session']}")

# 10. single-KV config keeps legacy session keys (parked cache stays addressable)
leg = Config.load(str(ROOT / "config.yaml"))
check(leg.kv_backends == 1, f"shipped config should be single-KV, got {leg.kv_backends}")
pipeline._store = None
pipeline._sidecar_rules = None
_, _, m3 = pipeline.process(leg, {"model": "whatever", "messages": msgs}, "n3",
                            backend=leg.resolve("whatever")[1])
sess = m3["session"]
check("-" not in sess and len(sess) == 16, f"legacy session key changed shape: {sess}")

shutil.rmtree(tmp, ignore_errors=True)
if fails:
    print("FAILED:")
    for f in fails:
        print(" -", f)
    sys.exit(1)
print("ALL PASS - routing picks backend/upstream/dialect, keeps hosted traffic "
      "out of the KV pipeline, echoes names, splits stats and sessions")
