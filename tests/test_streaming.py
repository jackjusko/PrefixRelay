import asyncio
import json
from proxy.streaming import batch_text_events


def event(text):
    return ('data: ' + json.dumps({'id': 'test', 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}) + '\n\n').encode()


def test_batch_preserves_text_and_boundaries():
    async def run():
        async def source():
            for _ in range(1000):
                yield event('hello')
            yield b'data: [DONE]\n\n'
        result = [raw async for raw in batch_text_events(source())]
        assert result[-1] == b'data: [DONE]\n\n'
        assert len(result) < 50
        assert ''.join(json.loads(raw[6:])['choices'][0]['delta']['content'] for raw in result[:-1]) == 'hello' * 1000
        print('1000 text events reduced to', len(result)-1)
    asyncio.run(run())


def test_idle_flush():
    async def run():
        resume = asyncio.Event()
        async def source():
            yield event('first')
            await resume.wait()
            yield event('second')
        stream = batch_text_events(source(), interval=0.005)
        first = await asyncio.wait_for(anext(stream), 0.5)
        assert json.loads(first[6:])['choices'][0]['delta']['content'] == 'first'
        resume.set()
        rest = [raw async for raw in stream]
        assert json.loads(rest[0][6:])['choices'][0]['delta']['content'] == 'second'
    asyncio.run(run())

if __name__ == '__main__':
    test_batch_preserves_text_and_boundaries()
    test_idle_flush()


def test_disconnect_cancels_pending_read():
    import anyio
    from proxy.streaming import ClosingStreamingResponse

    async def run():
        closed = asyncio.Event()
        async def source():
            try:
                yield event('first')
                await asyncio.Event().wait()
            finally:
                # Exercise asynchronous cleanup inside the cancelled scope.
                await asyncio.sleep(0)
                closed.set()
        async def send(message):
            if message['type'] == 'http.response.body':
                scope.cancel()
                await anyio.sleep(0)
        with anyio.CancelScope() as scope:
            response = ClosingStreamingResponse(batch_text_events(source(), interval=0.001))
            await response.stream_response(send)
        assert closed.is_set()
    asyncio.run(run())


def test_tool_argument_batches_preserve_fragments():
    async def run():
        def tool(fragment):
            return ('data: ' + json.dumps({'id': 'test', 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': fragment}}]}, 'finish_reason': None}]}) + '\n\n').encode()
        async def source():
            for fragment in ['{"code":"', 'hello' * 100, '"}']:
                yield tool(fragment)
            yield b'data: [DONE]\n\n'
        result = [raw async for raw in batch_text_events(source())]
        assert len(result) == 2
        args = json.loads(result[0][6:])['choices'][0]['delta']['tool_calls'][0]['function']['arguments']
        assert json.loads(args) == {'code': 'hello' * 100}
    asyncio.run(run())
