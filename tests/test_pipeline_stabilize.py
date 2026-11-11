"""Regression tests for the KV-cache guard (pipeline.stabilize).

Both cases reproduce real cache-loss bugs observed on 2026-10-30:

1. Tool-call arguments were compared byte-for-byte, but the model streams
   ``{"command": "ls"}`` while Cursor echoes history back compactly
   (``{"command":"ls"}``) — so every tool-call turn failed prefix matching
   at the recorded assistant message and rebased the canonical tail,
   invalidating the KV cache once per tool call.
2. Cursor injects a ``<plan_mode_guardrails>`` block into the system prompt
   while plan mode is active and removes it afterwards; the block was not
   volatile-normalized, so each toggle re-keyed the session and dropped the
   entire parked KV cache.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from proxy import pipeline as P                       # noqa: E402
from proxy.config import Config                       # noqa: E402


def tmp_cfg(tmp_path) -> Config:
    cfg = Config(mode="stabilize", state_dir=str(tmp_path / "state"))
    return cfg


# ---------------------------------------------------------------------------
# 1. tool-call argument normalization


def test_tool_sig_ignores_argument_spacing():
    streamed = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "Shell",
            "arguments": '{"command": "ls -la /workspace/example"}'}}]}
    echoed = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {
            "name": "Shell",
            "arguments": '{"command":"ls -la /workspace/example"}'}}]}
    assert P._tool_sig(streamed) == P._tool_sig(echoed)


def test_tool_sig_normalizes_key_order_and_dict_args():
    str_args = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": '{"b":2,"a":1}'}}]}
    dict_args = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": {"a": 1, "b": 2}}}]}
    assert P._tool_sig(str_args) == P._tool_sig(dict_args)


def test_tool_sig_non_json_args_fall_back_to_bytes():
    m1 = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": "raw text one"}}]}
    m2 = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": "raw text two"}}]}
    assert P._tool_sig(m1) != P._tool_sig(m2)
    assert P._tool_sig(m1) == P._tool_sig(m1)


def test_tool_sig_different_args_still_differ():
    m1 = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": '{"a": 1}'}}]}
    m2 = {"role": "assistant", "tool_calls": [
        {"id": "c1", "function": {"name": "f", "arguments": '{"a": 2}'}}]}
    assert P._tool_sig(m1) != P._tool_sig(m2)


# ---------------------------------------------------------------------------
# 2. end-to-end: a tool-call turn must append, not rebase


def test_tool_call_turn_appends_instead_of_rebasing(tmp_path):
    store = P.SessionStore(tmp_cfg(tmp_path))
    sys_txt = "You are helpful."
    user_txt = "list the boards dir please"

    meta: dict = {}
    store.stabilize({"model": "minion", "messages": [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt}]}, meta)
    assert meta["canonical"] == "new:2"

    # What app.py freezes via record_assistant_response: the model streamed
    # the tool call with its own (spaced) JSON formatting.
    streamed = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_abc", "type": "function", "function": {
            "name": "Shell",
            "arguments": '{"command": "ls -la /workspace/example/boards"}'}}]}
    sess = store.get(meta["session"])
    sess.msgs.append(streamed)
    sess.save()

    # Cursor's next request echoes the assistant turn with compact args and
    # appends the tool result plus a follow-up user message.
    meta2: dict = {}
    new_body, _route = store.stabilize({"model": "minion", "messages": [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_abc", "type": "function", "function": {
                "name": "Shell",
                "arguments": '{"command":"ls -la /workspace/example/boards"}'}}]},
        {"role": "tool", "tool_call_id": "call_abc", "content": "file1.md"},
        {"role": "user", "content": "now read file1.md"}]}, meta2)

    # Before the fix this was rebase:2+new:2 (divergence at len(canon)-1),
    # invalidating the KV cache on every tool-call turn.
    assert meta2["canonical"] == "frozen:3+new:2"
    assert "divergence" not in meta2
    # The forwarded prefix must be byte-stable: the streamed spacing stays
    # frozen (matching-only normalization never touches what we forward).
    assert new_body["messages"][:3] == [
        {"role": "system", "content": sys_txt},
        {"role": "user", "content": user_txt},
        streamed,
    ]


# ---------------------------------------------------------------------------
# 3. session key survives plan-mode guardrails injection


def test_session_key_survives_plan_mode_guardrails(tmp_path):
    store = P.SessionStore(tmp_cfg(tmp_path))
    # Splice layout as observed in real captures: Cursor inserts the block
    # (with its own trailing blank line) between two existing sections.
    base_sys = ("Use terminal commands exclusively for system operations.\n"
                "</tool_calling>\n\n<citing_code>\nDisplay code blocks.\n"
                "</citing_code>")
    plan_sys = ("Use terminal commands exclusively for system operations.\n"
                "</tool_calling>\n\n<plan_mode_guardrails>\n"
                "- In plan mode, only edit markdown files.\n"
                "- If the user explicitly asks you to build, switch to agent "
                "mode by calling `switchMode` with `target_mode_id=agent`.\n"
                "</plan_mode_guardrails>\n\n<citing_code>\nDisplay code "
                "blocks.\n</citing_code>")
    b_plain = {"model": "minion", "messages": [
        {"role": "system", "content": base_sys},
        {"role": "user", "content": "tailor my resume"}]}
    b_plan = {"model": "minion", "messages": [
        {"role": "system", "content": plan_sys},
        {"role": "user", "content": "tailor my resume"}]}
    # Before the fix these hashed to different sessions: the plan-mode
    # toggle dropped the entire parked KV cache (ct=0 on a 54k prompt).
    assert store.session_key(b_plain) == store.session_key(b_plan)


def test_msg_matches_ignores_guardrails_in_system(tmp_path):
    store = P.SessionStore(tmp_cfg(tmp_path))
    plain = "You are a coding agent.\n</tool_calling>\n\n<citing_code>\nx\n"
    guarded = ("You are a coding agent.\n</tool_calling>\n\n"
               "<plan_mode_guardrails>\n- only edit markdown files\n"
               "</plan_mode_guardrails>\n\n<citing_code>\nx\n")
    sess = store.get_or_create("testkey")
    # toggle on: canonical frozen without guardrails, Cursor now sends them
    assert store._msg_matches(sess, 0, {"role": "system", "content": plain},
                              {"role": "system", "content": guarded})
    # toggle off: canonical frozen with guardrails, Cursor now omits them
    assert store._msg_matches(sess, 0, {"role": "system", "content": guarded},
                              {"role": "system", "content": plain})


def test_guardrails_normalization_is_scoped_to_the_block():
    # The whole block (plus its trailing blank line) normalizes away, but
    # regular prose that merely mentions the tag name survives untouched.
    store = P.SessionStore.__new__(P.SessionStore)
    store._volatile = P._volatile_re(type("C", (), {"raw": {}})())
    text = "x<plan_mode_guardrails>a\nb</plan_mode_guardrails>\n\ny"
    assert store._apply_volatile(text) == "xy"
    prose = "Remember the plan_mode_guardrails rules when planning."
    assert store._apply_volatile(prose) == prose
