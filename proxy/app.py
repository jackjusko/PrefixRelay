"""PrefixRelay: KV-cache-friendly OpenAI-compatible proxy for Cursor.

Cursor sends conversation history on each turn; changes to earlier messages
can invalidate a server's prefix cache. In stabilize mode, the proxy preserves
a canonical history and reports cache usage supplied by the upstream.
"""
from __future__ import annotations

from dataclasses import replace
import asyncio
import codecs
import json
import logging
import time
import uuid

import httpx
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import affinity, pipeline
from .config import Config
from .recorder import Recorder
from .streaming import batch_text_events, ClosingStreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prefixrelay")

# Where a backend puts reasoning in a stream delta.
REASONING_FIELDS = ("reasoning_content", "reasoning")


def _reasoning_keys(backend) -> tuple[str, ...]:
    field = getattr(backend, "reasoning_field", "auto")
    if field == "none":
        return ()
    if field in (None, "auto"):
        return REASONING_FIELDS
    return (field,)


def _rewrite_model(raw: bytes, name: str) -> bytes:
    """Point a non-stream response's model field at the name Cursor asked for
    so its chip label matches what it sent.  Original bytes on anything odd."""
    try:
        obj = json.loads(raw)
    except Exception:
        return raw
    if not isinstance(obj, dict) or "model" not in obj:
        return raw
    obj["model"] = name
    try:
        return json.dumps(obj).encode()
    except Exception:
        return raw

CONFIG = Config.load()
rec = Recorder(CONFIG)
client = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=3600.0, write=60.0, pool=10.0)
)
app = FastAPI(title="PrefixRelay", docs_url=None, redoc_url=None)
app.add_middleware(affinity.ReleaseReplica)
pools = {name: affinity.Pool(backend.raw["replicas"])
         for name, backend in CONFIG.backends.items() if backend.raw.get("replicas")}


_LEARNED_PATH = CONFIG.path(CONFIG.learned_keys_file)
_learned: set[str] = set()


def _load_learned() -> None:
    if _LEARNED_PATH.exists():
        _learned.update(x for x in _LEARNED_PATH.read_text().splitlines() if x)


_load_learned()


def _learn(key: str) -> None:
    _learned.add(key)
    _LEARNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LEARNED_PATH.open("a") as f:
        f.write(key + "\n")
    log.warning("learned client API key (…%s); subsequent requests must match a known key", key[-4:])


def _supplied_key(request: Request) -> str:
    auth = (request.headers.get("authorization") or "").strip()
    if auth.casefold().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("api-key") or "").strip()


def _authed(request: Request) -> bool:
    if not CONFIG.enforce_auth:
        return True
    supplied = _supplied_key(request)
    tok = CONFIG.token()
    if tok and supplied == tok:
        return True
    if supplied and supplied in _learned:
        return True
    if CONFIG.learn_first_key and not _learned and supplied:
        _learn(supplied)
        return True
    return False


def _err(status: int, msg: str, err_type: str = "proxy_error") -> JSONResponse:
    return JSONResponse(
        {"error": {"message": msg, "type": err_type, "code": status}},
        status_code=status,
    )


@app.get("/healthz")
async def healthz(request: Request):
    if not _authed(request):
        return _err(401, "invalid or missing API key", "authentication_error")
    # Config.health() reports backends/routes with keys stripped.
    return CONFIG.health()


@app.get("/stats")
async def stats(request: Request, limit: int = 25):
    if not _authed(request):
        return _err(401, "invalid or missing API key", "authentication_error")
    rows: list[dict] = []
    try:
        lines = rec.index.read_text().splitlines()
    except FileNotFoundError:
        lines = []
    for x in lines:
        try:
            rows.append(json.loads(x))
        except Exception:
            pass
    def _backend(r) -> str:
        return (r.get("meta") or {}).get("backend") or "unknown"

    def _cache_tracked(r) -> bool:
        # Rows recorded before routing existed carry no backend; they were all
        # local, so keep counting them in the cache ratio.
        name = _backend(r)
        if name == "unknown":
            return True
        be = CONFIG.backend_for(name)
        return bool(be and be.stabilize)

    all_main = [r for r in rows if r.get("route", "main") == "main"]
    cache_rows = [r for r in all_main if _cache_tracked(r)]

    def _pt(r):
        return (r.get("usage") or {}).get("prompt_tokens") or 0

    def _ct(r):
        return ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0

    def _co(r):
        return (r.get("usage") or {}).get("completion_tokens") or 0

    tp = sum(_pt(r) for r in cache_rows)
    tc = sum(_ct(r) for r in cache_rows)
    by_backend: dict[str, dict] = {}
    for r in rows:
        name = _backend(r)
        b = by_backend.setdefault(name, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                        "cached_tokens": 0, "cache_ratio": None,
                                        "kv_cache": _cache_tracked(r), "est_cost": None})
        be = CONFIG.backend_for(name)
        b["requests"] += 1
        b["prompt_tokens"] += _pt(r)
        b["completion_tokens"] += _co(r)
        b["cached_tokens"] += _ct(r)
        if b["kv_cache"] and b["prompt_tokens"]:
            b["cache_ratio"] = round(b["cached_tokens"] / b["prompt_tokens"], 4)
        est = be.estimate(r.get("usage")) if be else None
        if est:
            b["est_cost"] = round((b["est_cost"] or 0.0) + est, 6)
    summary = {"requests": len(rows), "main_requests": len(all_main),
               "prompt_tokens": tp, "cached_tokens": tc,
               "cache_ratio": round(tc / tp, 4) if tp else None,
               "kv_requests": len(cache_rows), "by_backend": by_backend}
    rows = rows[-max(1, min(limit, 500)):]
    out = []
    for r in rows:
        u = r.get("usage") or {}
        det = u.get("prompt_tokens_details") or {}
        pt = u.get("prompt_tokens")
        ct = det.get("cached_tokens")
        rmeta = r.get("meta") or {}
        out.append({
            "rid": r.get("rid"), "ts": r.get("ts"), "route": r.get("route"),
            "backend": rmeta.get("backend"), "model": rmeta.get("model_requested"),
            "model_upstream": rmeta.get("model_upstream"),
            "status": r.get("status"), "prompt": pt, "cached": ct,
            "ratio": (round(ct / pt, 3) if pt and ct is not None else None),
            "completion": u.get("completion_tokens"),
            "dur_ms": r.get("dur_ms"), "ttft_ms": r.get("ttft_ms"),
            "error": r.get("error"),
        })
    return {"summary": summary, "rows": out}


@app.get("/v1/models")
@app.get("/models")
async def models(request: Request):
    if not _authed(request):
        return _err(401, "invalid or missing API key", "authentication_error")
    # Cursor validates its key against one of these and lists them in the
    # picker; every name here is a route Cursor may send back to us.
    return CONFIG.models_payload()


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    if not _authed(request):
        return _err(401, "invalid or missing API key", "authentication_error")
    t0 = time.monotonic()
    raw = await request.body()
    try:
        body = json.loads(raw)
    except Exception:
        return _err(400, "request body is not valid JSON", "invalid_request_error")

    if not isinstance(body, dict):
        return _err(400, "request body must be a JSON object", "invalid_request_error")
    messages = body.get("messages")
    if messages is not None and (
        not isinstance(messages, list) or any(not isinstance(msg, dict) for msg in messages)
    ):
        return _err(400, "messages must be an array of objects", "invalid_request_error")

    rid = uuid.uuid4().hex[:12]
    await asyncio.to_thread(rec.begin, rid, dict(request.headers), raw)

    # Route on the name Cursor sent before anything else touches the body: it
    # picks the server and decides whether the KV-cache pipeline applies at all.
    requested = str(body.get("model") or "").strip()
    try:
        route_obj, backend, source = CONFIG.resolve(requested)
    except Exception as e:
        return _err(400, f"model routing failed: {e}", "invalid_request_error")
    echo_name = requested or route_obj.name
    # Compaction and its summarizer must use this exact model's context limit,
    # upstream model id, and credentials, even when routes share one backend.
    backend = replace(
        backend,
        context_tokens=route_obj.context_tokens or backend.context_tokens,
        default_model=CONFIG.upstream_model_for(route_obj, backend) or requested,
    )
    meta = {"model_requested": requested or None, "model_route": route_obj.name,
            "model_source": source, "backend": backend.name}

    pool = pools.get(backend.name)
    replica = None
    if pool:
        queue_started = time.monotonic()
        replica = await pool.acquire(client, pipeline.session_key(CONFIG, body), affinity.fingerprint(body))
        if replica is None:
            return _err(503, "no healthy minion replicas")
        request.scope["minion_lease"] = (pool, replica)
        backend = replace(backend, base_url=replica.base_url, default_model=replica.model)
        meta["replica"] = replica.model
        meta["queue_ms"] = round((time.monotonic() - queue_started) * 1000, 1)

    pipeline_started = time.monotonic()
    try:
        fwd_body, route, pmeta = await asyncio.to_thread(
            pipeline.process, CONFIG, body, rid, backend=backend, model_name=route_obj.name
        )
    except Exception as e:  # never take the user's turn down because of us
        log.exception("pipeline failed rid=%s", rid)
        fwd_body, route, pmeta = body, "main", {"pipeline_error": repr(e)}
    meta = {**meta, **pmeta, "pipeline_ms": round((time.monotonic() - pipeline_started) * 1000, 1)}
    if route == "sidecar":
        sb = CONFIG.backend_for("sidecar")
        if sb is not None:
            backend = sb
            meta["backend"] = sb.name   # stats must attribute it to the sidecar

    if route == "stub":
        now = int(time.time())
        await asyncio.to_thread(rec.finish, rid, b"", status=200, dur_ms=(time.monotonic() - t0) * 1000,
                   error="stubbed")
        return JSONResponse({
            "id": f"proxy-stub-{rid}", "object": "chat.completion", "created": now,
            "model": echo_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })

    # Rewrite the model field to the id the chosen backend actually serves;
    # an empty upstream id means "pass the caller's name through".
    upstream_model = CONFIG.upstream_model_for(route_obj, backend) if route == "main" \
        else backend.default_model
    if replica is not None and route == "main":
        upstream_model = replica.model
        request.scope["minion_forwarded"] = True
    meta["model_upstream"] = upstream_model or None
    if upstream_model and fwd_body.get("model") != upstream_model:
        meta["model_replaced"] = fwd_body.get("model")
        fwd_body["model"] = upstream_model

    # Per-route request defaults (thinking effort and friends). Only for the
    # main route: a sidecar redirect is a different model with different knobs.
    if route == "main" and (route_obj.extra_body or route_obj.drop_body):
        fwd_body = dict(fwd_body)
        applied = pipeline.apply_body_defaults(fwd_body, route_obj.extra_body)
        if applied:
            meta["body_defaults"] = applied
        dropped = [k for k in route_obj.drop_body if fwd_body.pop(k, None) is not None]
        if dropped:
            meta["body_dropped"] = dropped

    url = backend.chat_url()
    stream = bool(fwd_body.get("stream"))
    if stream and CONFIG.force_usage_in_stream:
        # Usage reporting is dialect-specific: OpenRouter rejects
        # stream_options and wants usage.include instead.
        if backend.usage_style == "stream_options":
            so = fwd_body.setdefault("stream_options", {})
            so.setdefault("include_usage", True)
        elif backend.usage_style == "usage_object":
            u = fwd_body.setdefault("usage", {})
            if isinstance(u, dict):
                u.setdefault("include", True)

    payload = json.dumps(fwd_body).encode()
    await asyncio.to_thread(rec.set_forwarded, rid, payload, route, meta)
    log.info("rid=%s route=%s backend=%s %s -> %s (%s msgs, stream=%s)",
             rid, route, backend.name, requested or "-", url,
             len(fwd_body.get("messages") or []), stream)
    try:
        req = client.build_request("POST", url, content=payload, headers=backend.headers())
        resp = await client.send(req, stream=True)
    except Exception as e:
        await asyncio.to_thread(rec.finish, rid, None, status=0, dur_ms=(time.monotonic() - t0) * 1000,
                   error=repr(e))
        return _err(502, f"upstream unreachable: {e}")

    if stream and resp.is_success:
        state = {"ttft": None}
        session_key = meta.get("session") if route == "main" else None
        stream_think = CONFIG.stream_reasoning_as_think_tags and backend.wrap_reasoning
        content_defang = backend.defang_specials
        reasoning_keys = _reasoning_keys(backend)
        close_model_default = echo_name

        def _think_close_bytes(last_chunk_obj: dict) -> bytes:
            # One newline after </think>, not a blank line: Cursor strips the
            # think block and was storing leftover "\n\n" as assistant content,
            # which then failed prefix matching every turn.
            close_obj = {
                "id": last_chunk_obj.get("id", f"chatcmpl-{rid}"),
                "object": "chat.completion.chunk",
                "created": last_chunk_obj.get("created", int(time.time())),
                "model": last_chunk_obj.get("model") or close_model_default,
                "choices": [{"index": 0, "delta": {"content": "\n</think>\n"}, "finish_reason": None}],
            }
            return f"data: {json.dumps(close_obj)}\n\n".encode("utf-8")

        async def gen():
            chunks: list[bytes] = []
            accum_reasoning: list[str] = []
            accum_content: list[str] = []
            tool_calls_map: dict[int, dict] = {}
            thinking_open = False
            last_reasoning_heartbeat = time.monotonic()
            last_chunk_obj: dict = {}
            buffer = ""
            finished = False
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            try:
                async for raw_bytes in resp.aiter_bytes():
                    if state["ttft"] is None:
                        state["ttft"] = time.monotonic() - t0

                    buffer += decoder.decode(raw_bytes)
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line_stripped = line.rstrip("\r")

                        if not line_stripped:
                            continue

                        if line_stripped.startswith(":"):
                            out_bytes = (line_stripped + "\n\n").encode("utf-8")
                            chunks.append(out_bytes)
                            yield out_bytes
                            continue

                        if not line_stripped.startswith("data:"):
                            out_bytes = (line_stripped + "\n").encode("utf-8")
                            chunks.append(out_bytes)
                            yield out_bytes
                            continue

                        payload = line_stripped[5:].strip()
                        if payload == "[DONE]":
                            finished = True
                            if stream_think and thinking_open:
                                close_bytes = _think_close_bytes(last_chunk_obj)
                                chunks.append(close_bytes)
                                yield close_bytes
                                thinking_open = False
                            done_bytes = b"data: [DONE]\n\n"
                            chunks.append(done_bytes)
                            yield done_bytes
                            continue

                        try:
                            obj = json.loads(payload)
                        except Exception:
                            out_bytes = (line_stripped + "\n\n").encode("utf-8")
                            chunks.append(out_bytes)
                            yield out_bytes
                            continue

                        if not isinstance(obj, dict):
                            out_bytes = (line_stripped + "\n\n").encode("utf-8")
                            chunks.append(out_bytes)
                            yield out_bytes
                            continue

                        obj["model"] = echo_name   # keep Cursor's chip label
                        last_chunk_obj = obj
                        choices = obj.get("choices") or []
                        if not choices:
                            out_bytes = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
                            chunks.append(out_bytes)
                            yield out_bytes
                            continue

                        delta = choices[0].get("delta")
                        if not isinstance(delta, dict):
                            delta = {}
                            choices[0]["delta"] = delta
                        r_key = next((k for k in reasoning_keys if delta.get(k)), None)
                        r_text = delta.get(r_key) if r_key else None
                        c_text = delta.get("content")
                        t_calls = delta.get("tool_calls")
                        finish = choices[0].get("finish_reason")
                        finished |= bool(finish)

                        if t_calls:
                            for tc in t_calls:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_map:
                                    tool_calls_map[idx] = {
                                        "id": tc.get("id") or "",
                                        "type": tc.get("type") or "function",
                                        "function": {
                                            "name": (tc.get("function") or {}).get("name") or "",
                                            "arguments": (tc.get("function") or {}).get("arguments") or "",
                                        },
                                    }
                                else:
                                    if tc.get("id"):
                                        tool_calls_map[idx]["id"] = tc["id"]
                                    if tc.get("type"):
                                        tool_calls_map[idx]["type"] = tc["type"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name"):
                                        tool_calls_map[idx]["function"]["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        tool_calls_map[idx]["function"]["arguments"] += fn["arguments"]

                        if r_text:
                            accum_reasoning.append(r_text)
                        if not CONFIG.stream_reasoning_to_client:
                            for key in reasoning_keys:
                                delta.pop(key, None)
                            if not delta and not finish and not obj.get("usage"):
                                if time.monotonic() - last_reasoning_heartbeat >= 10:
                                    yield b": reasoning in progress\n\n"
                                    last_reasoning_heartbeat = time.monotonic()
                                continue

                        if r_text and CONFIG.stream_reasoning_to_client:
                            if stream_think:
                                wrapped = pipeline.defang_chat_specials(
                                    r_text, think_tags=True, chat_specials=content_defang)
                                if not thinking_open:
                                    thinking_open = True
                                    delta["content"] = "<think>\n" + wrapped
                                else:
                                    delta["content"] = wrapped
                                # Don't forward raw specials in the reasoning field —
                                # Cursor treats <|im_end|> / </think> as stream killers.
                                delta.pop(r_key, None)
                            else:
                                # OpenAI-compatible reasoning stays separate from
                                # visible answer content. Normalize provider dialects.
                                for key in reasoning_keys:
                                    delta.pop(key, None)
                                delta["reasoning_content"] = r_text
                            if not (c_text or t_calls or finish):
                                out_bytes = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
                                chunks.append(out_bytes)
                                yield out_bytes
                                continue

                        if stream_think and thinking_open and (c_text or t_calls or finish):
                            close_bytes = _think_close_bytes(obj)
                            chunks.append(close_bytes)
                            yield close_bytes
                            thinking_open = False

                        if c_text:
                            accum_content.append(c_text)
                            delta["content"] = pipeline.defang_chat_specials(
                                c_text, think_tags=True, chat_specials=content_defang)

                        out_bytes = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
                        chunks.append(out_bytes)
                        yield out_bytes

                buffer += decoder.decode(b"", final=True)
                if buffer.strip():
                    line_stripped = buffer.rstrip("\r\n")
                    if line_stripped.startswith("data:"):
                        payload = line_stripped[5:].strip()
                        finished |= payload == "[DONE]"
                        if payload == "[DONE]" and stream_think and thinking_open:
                            close_bytes = _think_close_bytes(last_chunk_obj)
                            chunks.append(close_bytes)
                            yield close_bytes
                            thinking_open = False
                    out_bytes = (line_stripped + "\n\n").encode("utf-8")
                    chunks.append(out_bytes)
                    yield out_bytes

            except Exception as e:
                if stream_think and thinking_open:
                    close_bytes = _think_close_bytes(last_chunk_obj)
                    chunks.append(close_bytes)
                    yield close_bytes
                    thinking_open = False
                await asyncio.to_thread(rec.finish, rid, b"".join(chunks), status=resp.status_code,
                           dur_ms=(time.monotonic() - t0) * 1000, error=repr(e))
                raise
            finally:
                with anyio.CancelScope(shield=True):
                    await resp.aclose()
            if not finished:
                request.scope["minion_forwarded"] = False
            elif session_key and (accum_content or accum_reasoning or tool_calls_map):
                t_calls_list = [tool_calls_map[i] for i in sorted(tool_calls_map.keys())] if tool_calls_map else None
                asst_msg = {
                    "role": "assistant",
                    "content": "".join(accum_content),
                }
                if accum_reasoning:
                    asst_msg["reasoning_content"] = "".join(accum_reasoning)
                if t_calls_list:
                    asst_msg["tool_calls"] = t_calls_list
                try:
                    await asyncio.to_thread(pipeline.record_assistant_response, CONFIG, session_key, asst_msg)
                except Exception:
                    log.exception("failed to record assistant response for %s", session_key)

            await asyncio.to_thread(rec.finish, rid, b"".join(chunks), status=resp.status_code,
                       dur_ms=(time.monotonic() - t0) * 1000,
                       ttft_ms=state["ttft"] * 1000 if state["ttft"] else None)

        return ClosingStreamingResponse(
            batch_text_events(gen(), request_id=rid), status_code=resp.status_code,
            headers={
                "content-type": resp.headers.get("content-type", "text/event-stream"),
                "x-accel-buffering": "no",
                "cache-control": "no-cache, no-transform",
            },
        )

    try:
        raw_resp = await resp.aread()
    finally:
        await resp.aclose()
    if resp.status_code == 200:
        raw_resp = _rewrite_model(raw_resp, echo_name)
    await asyncio.to_thread(rec.finish, rid, raw_resp, status=resp.status_code,
               dur_ms=(time.monotonic() - t0) * 1000)
    if route == "main" and meta.get("session") and resp.status_code == 200:
        try:
            res_obj = json.loads(raw_resp)
            msg = (res_obj.get("choices") or [{}])[0].get("message")
            if msg:
                await asyncio.to_thread(pipeline.record_assistant_response, CONFIG, meta["session"], msg)
        except Exception:
            log.exception("failed to record non-streaming assistant response for %s", meta.get("session"))
    return Response(content=raw_resp, status_code=resp.status_code,
                    media_type=resp.headers.get("content-type", "application/json"))
