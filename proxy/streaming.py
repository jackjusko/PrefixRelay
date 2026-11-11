"""Reduce downstream SSE update frequency without delaying idle streams."""
import asyncio
import json
import logging
import copy

import anyio
from starlette.responses import StreamingResponse


def _text_event(raw):
    try:
        if not raw.startswith(b'data: '):
            return None
        obj = json.loads(raw[6:])
        choices = obj.get('choices')
        if len(choices) != 1 or obj.get('usage'):
            return None
        choice = choices[0]
        delta = choice['delta']
        if choice.get('finish_reason') or not delta:
            return None
        if set(delta) == {'tool_calls'}:
            calls = delta['tool_calls']
            if not isinstance(calls, list) or not calls:
                return None
            for call in calls:
                if not isinstance(call, dict) or not isinstance(call.get('index'), int):
                    return None
                if set(call) - {'index', 'id', 'type', 'function'}:
                    return None
                fn = call.get('function', {})
                if not isinstance(fn, dict) or set(fn) - {'name', 'arguments'}:
                    return None
                if not all(isinstance(v, str) for v in fn.values()):
                    return None
            return obj
        if not set(delta) <= {'content', 'reasoning', 'reasoning_content'}:
            return None
        if not all(isinstance(v, str) for v in delta.values()):
            return None
        return obj
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def _merge(left, right):
    if {k: v for k, v in left.items() if k != 'choices'} != {k: v for k, v in right.items() if k != 'choices'}:
        return False
    a, b = left['choices'][0], right['choices'][0]
    if {k: v for k, v in a.items() if k != 'delta'} != {k: v for k, v in b.items() if k != 'delta'}:
        return False
    if a['delta'].keys() != b['delta'].keys():
        return False
    if 'tool_calls' in a['delta']:
        calls = copy.deepcopy(a['delta']['tool_calls'])
        for incoming in b['delta']['tool_calls']:
            existing = next((c for c in calls if c['index'] == incoming['index']), None)
            if existing is None:
                calls.append(copy.deepcopy(incoming))
                continue
            for key in ('id', 'type'):
                value = incoming.get(key)
                if value and existing.get(key) and value != existing[key]:
                    return False
                if value:
                    existing[key] = value
            fn = existing.setdefault('function', {})
            for key, value in incoming.get('function', {}).items():
                fn[key] = fn.get(key, '') + value
        a['delta']['tool_calls'] = calls
        return True
    for key, value in b['delta'].items():
        a['delta'][key] += value
    return True


async def batch_text_events(source, interval=3.0, max_bytes=1048576, request_id=None):
    """Coalesce compatible text deltas for at most 3 seconds or 1MiB of input events.

    Keep one pending read across timer expirations: cancelling an async
    generator's read on timeout would close the upstream stream.
    """
    iterator = source.__aiter__()
    inputs = 0
    started = asyncio.get_running_loop().time()
    pending = None
    batch = None
    size = 0
    deadline = None
    loop = asyncio.get_running_loop()

    def encode(obj):
        return b'data: ' + json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode() + b'\n\n'

    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))
            timeout = max(0, deadline - loop.time()) if batch is not None else None
            done, _ = await asyncio.wait({pending}, timeout=timeout)
            if not done:
                yield encode(batch)
                batch, size, deadline = None, 0, None
                continue
            try:
                raw = pending.result()
                inputs += 1
            except StopAsyncIteration:
                pending = None
                break
            pending = None
            obj = _text_event(raw)
            if batch is not None and (obj is None or not _merge(batch, obj)):
                yield encode(batch)
                batch, size, deadline = None, 0, None
            elif batch is not None:
                size += len(raw)
                if size >= max_bytes or loop.time() >= deadline:
                    yield encode(batch)
                    batch, size, deadline = None, 0, None
                continue
            if obj is None:
                yield raw
            else:
                batch, size, deadline = obj, len(raw), loop.time() + interval
        if batch is not None:
            yield encode(batch)
    finally:
        logging.getLogger("prefixrelay").info("rid=%s batch_input_events=%s elapsed=%.3fs interval=%.1fs", request_id, inputs, loop.time() - started, interval)
        # Starlette cancels streams inside an AnyIO cancellation scope.
        # Shield cleanup so the detached upstream read is actually stopped.
        with anyio.CancelScope(shield=True):
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await iterator.aclose()


class ClosingStreamingResponse(StreamingResponse):
    """Close the iterator even when disconnect happens during a socket send."""

    async def stream_response(self, send):
        count = 0
        async def measured_send(message):
            nonlocal count
            if message['type'] == 'http.response.body' and message.get('body'):
                count += 1
            await send(message)
        try:
            await super().stream_response(measured_send)
        finally:
            logging.getLogger("prefixrelay").info("stream_output_events=%s", count)
            with anyio.CancelScope(shield=True):
                await self.body_iterator.aclose()
