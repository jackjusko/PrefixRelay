"""Unit tests for the compaction module (offline; no upstream needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxy import compaction as C                      # noqa: E402


def msgs_fixture():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "result"},
        {"role": "assistant", "content": "done1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "done2"},
    ]


def test_boundary_never_splits_tool_pair():
    m = msgs_fixture()
    # naive cut at index 3 would land on the tool result -> must back up to 2
    # so the tail starts on its assistant(tool_calls) message
    assert C._tail_boundary(m, 4) == 2
    # boundary landing directly on the assistant-with-calls is also valid
    assert C._tail_boundary(m, 5) == 2
    # plain cut lands fine on user messages
    assert C._tail_boundary(m, 2) == 5
    # clamps at 0
    assert C._tail_boundary(m, 99) == 0
    # want=0 -> 0
    assert C._tail_boundary(m, 0) == 0


def test_elision_keeps_tail_verbatim():
    m = ([{"role": "user", "content": "x"},
          {"role": "assistant", "tool_calls": [
              {"id": "t", "function": {"name": "read", "arguments": "p"}}]},
          {"role": "tool", "tool_call_id": "t", "content": "Y" * 20000},
          {"role": "tool", "tool_call_id": "t", "content": "short"}]
         + [{"role": "user", "content": f"m{i}"} for i in range(8)])
    out, n = C._elide_tool_outputs(m, 4)
    assert n == 1
    assert "elided: 20000 chars" in out[2]["content"]
    assert out[2]["content"].startswith("Y" * C.ELIDE_HEAD)
    assert out[-1]["content"] == "m7"          # tail untouched
    assert out[3]["content"] == "short"        # short outputs untouched
    # original list untouched (projection only)
    assert len(m[2]["content"]) == 20000


def test_elision_drops_images_and_shortens_args():
    m = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]},
         {"role": "assistant", "tool_calls": [
             {"id": "t", "function": {"name": "w", "arguments": "z" * 5000}}]},
         {"role": "tool", "tool_call_id": "t", "content": "ok"},
         {"role": "user", "content": "next"}]
    out, n = C._elide_tool_outputs(m, 1)
    assert n >= 1
    assert not any(isinstance(p, dict) and "image_url" in p for p in out[0]["content"])
    assert "elided: 5000 chars" in out[1]["tool_calls"][0]["function"]["arguments"]


def test_marker_short_circuits():
    class B:
        compact_enabled = True
        context_tokens = 100
        compact_at = 0.8
        compact_tail_msgs = 8
        compact_summary_max_tokens = 1024
        base_url = "http://127.0.0.1:1"
    body = {"messages": [{"role": "system", "content": "s"}]
            + [{"role": "user", "content": "[Previous conversation summary] x"}] * 20}
    meta = {}
    out = C.compact_if_needed(None, body, B(), meta)
    assert out is body
    assert meta["compact"]["stage"] == "skipped"


def test_no_nctx_skips():
    class B:
        compact_enabled = True
        context_tokens = 0
        compact_at = 0.8
        compact_tail_msgs = 8
        compact_summary_max_tokens = 1024
        base_url = "http://127.0.0.1:1"
    C._prop_cache.clear()
    import time
    C._prop_cache[B.base_url] = (time.time(), 0)
    body = {"messages": [{"role": "user", "content": "hi"}] * 30}
    meta = {}
    out = C.compact_if_needed(None, body, B(), meta)
    assert out is body
    assert meta["compact"]["stage"] == "skipped"


def test_disabled_backend_is_noop():
    class B:
        compact_enabled = False
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert C.compact_if_needed(None, body, B(), {}) is body


def test_forced_compaction_elides_and_flags():
    """Small context_tokens forces the elide stage without any upstream call."""
    class B:
        compact_enabled = True
        context_tokens = 10000        # enough room after older tool output is elided
        compact_at = 0.8
        compact_tail_msgs = 2
        compact_summary_max_tokens = 1024
        base_url = "http://127.0.0.1:1"

        def chat_url(self):
            return self.base_url + "/v1/chat/completions"

    big = [{"role": "system", "content": "sys"}]
    for i in range(10):
        big.append({"role": "assistant", "tool_calls": [
            {"id": f"t{i}", "function": {"name": "read", "arguments": "p"}}]})
        big.append({"role": "tool", "tool_call_id": f"t{i}",
                    "content": "Q" * 3000})
    big.append({"role": "user", "content": "latest question"})
    meta = {}
    out = C.compact_if_needed(None, {"messages": big}, B(), meta,
                              _summarize_fn=lambda *a, **k: "Test summary of earlier turns.")
    assert meta["compact"]["stage"] == "elide"
    assert meta["compact"]["threshold"] == 8000
    sent = out["messages"]
    assert sent[0]["role"] == "system"
    # tail is tool-pair safe: last kept messages must not start on a tool result
    tool_idx = [i for i, m in enumerate(sent) if m.get("role") == "tool"]
    if tool_idx:
        assert sent[tool_idx[0] - 1].get("role") == "assistant" and sent[tool_idx[0] - 1].get("tool_calls")
    # the last user message survives verbatim
    assert sent[-1] == {"role": "user", "content": "latest question"}
    # original body untouched
    assert big[-2]["content"] == "Q" * 3000


def test_summarize_stage_builds_summary_body():
    """When elision alone cannot get under threshold, the summarizer path
    produces system + marker summary + verbatim tail."""
    class B:
        compact_enabled = True
        context_tokens = 1000
        compact_at = 0.8
        compact_tail_msgs = 2
        compact_summary_max_tokens = 1024
        base_url = "http://127.0.0.1:1"

    big = [{"role": "system", "content": "sys"}]
    for i in range(6):
        big.append({"role": "assistant", "tool_calls": [
            {"id": f"t{i}", "function": {"name": "read", "arguments": "p"}}]})
        big.append({"role": "tool", "tool_call_id": f"t{i}",
                    "content": "Q" * 3000})
    big.append({"role": "user", "content": "latest question"})
    orig_count = C._token_count
    counts = iter((10000, 900, 500))  # full prompt, elided projection, summarized prompt
    C._token_count = lambda *a, **k: next(counts)
    meta = {}
    try:
        out = C.compact_if_needed(None, {"messages": big}, B(), meta,
                                  _summarize_fn=lambda *a, **k: "Brief.")
    finally:
        C._token_count = orig_count
    assert meta["compact"]["stage"] == "summarize"
    sent = out["messages"]
    # system + summary user msg + tool-pair-safe tail + latest user message
    assert len(sent) == 5
    assert sent[0]["role"] == "system"
    assert "[Previous conversation summary]" in sent[1]["content"]
    assert "Brief." in sent[1]["content"]
    assert sent[-1] == {"role": "user", "content": "latest question"}
    # tail is tool-pair safe: never starts on a tool result
    assert sent[2]["role"] == "assistant" and sent[2].get("tool_calls")
    assert sent[3]["role"] == "tool"


def test_serialize_counts_all_roles():
    s = C._serialize_for_count([
        {"role": "assistant", "content": "hi", "reasoning_content": "think",
         "tool_calls": [{"id": "1", "function": {"name": "f", "arguments": '{"a": 1}'}}]},
        {"role": "user", "content": [
            {"type": "text", "text": "part"},
            {"type": "image_url", "image_url": {"url": "data:..."}},
        ]}])
    assert "hi" in s and "think" in s and "f" in s and "part" in s
    assert s.count("[image]") == 1


def test_summary_request_uses_upstream_model_and_auth(monkeypatch):
    class Backend:
        base_url = "https://api.example/v1"
        default_model = "vendor/model-small"

        def chat_url(self):
            return self.base_url + "/chat/completions"

        def headers(self):
            return {"content-type": "application/json", "authorization": "Bearer upstream-secret"}

    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "summary"}}]}).encode()

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert C._summarize(Backend(), "system", [{"role": "user", "content": "old"}], 64) == "summary"
    assert captured["body"]["model"] == "vendor/model-small"
    assert captured["headers"]["Authorization"] == "Bearer upstream-secret"


# ---------------------------------------------------------------------------
# elided tool-call arguments must stay valid JSON (regression: a raw U+000A
# spliced into the arguments string can teach models to emit tool calls
# with unescaped control characters, which strict servers reject with
# json.exception.parse_error.101)

def test_shorten_args_roundtrip_valid_json():
    # mirrors the captured corruption: a heredoc-shaped command, long enough
    # to trigger elision (> max_len + 8 after serialization)
    cmd = ("cd /workspace/example && python3 - <<'EOF'\nimport example_package\n"
           + "x" * 400)
    args = json.dumps({"command": cmd})
    assert len(args) > 408
    out = C._shorten_args(args)
    parsed = json.loads(out)                       # must re-parse
    assert "arguments elided" in parsed["command"]
    assert parsed["command"].startswith(cmd[:40])  # head preserved verbatim
    # the marker's newlines are the two-character escape sequence, never raw
    assert chr(10) not in out
    assert "\\n...[arguments elided:" in out       # backslash + n, not U+000A


def test_shorten_args_repairs_dangling_backslash_cut():
    # place a lone backslash exactly at the head cut so the splice would end
    # mid-escape sequence
    prefix = '{"command": "'
    pad = "a" * (C.ELIDE_HEAD - len(prefix) - 1) + "\\"
    args = json.dumps({"command": pad + "\\n" + "b" * 500})
    out = C._shorten_args(args)
    parsed = json.loads(out)                       # must re-parse
    assert "arguments elided" in parsed["command"]
    # the dangling backslash is gone: the kept head no longer ends mid-escape
    assert not parsed["command"].split("...[arguments elided")[0].endswith("\\")
    assert chr(10) not in out


def test_shorten_args_repairs_partial_unicode_escape_cut():
    # head cut lands inside a \uXXXX escape: backslash at index 196, so the
    # head slice ends with a partial "\u00" run
    doc = '{"text": "' + "x" * (C.ELIDE_HEAD - 11) + "\\u00e9" + "y" * 500 + '"}'
    assert len(doc) > 408
    out = C._shorten_args(doc)
    parsed = json.loads(out)                       # must re-parse
    assert "arguments elided" in parsed["text"]
    assert chr(10) not in out


def test_shorten_args_unparseable_input_keeps_marker():
    # legacy behavior: non-JSON arguments still get the byte-level splice
    out = C._shorten_args("z" * 5000)
    assert "elided: 5000 chars" in out
    assert chr(10) not in out


def test_shorten_args_structural_cut_falls_back_to_value_elision():
    # valid JSON whose char-200 splice lands between tokens (many short
    # values): the fallback elides inside the parsed value tree instead
    many = json.dumps({("k%02d" % i): ("v" * 70) for i in range(40)})
    out = C._shorten_args(many)
    parsed = json.loads(out)                       # must re-parse
    assert parsed == json.loads(out)               # stable
    # nothing was long enough to elide, but the output must stay valid
    assert out != many or "arguments elided" in out


def test_shorten_args_is_deterministic():
    args = json.dumps({"command": "x" * 500})
    assert C._shorten_args(args) == C._shorten_args(args)


def test_shorten_args_untouched_below_threshold():
    args = json.dumps({"command": "short"})
    assert C._shorten_args(args) is args
