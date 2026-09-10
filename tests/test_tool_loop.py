"""Regression coverage for _parse_worker_reply and _tool_loop -- the shared
ACTION/FINAL loop every role node (planner/solver/summarizer/finder) AND
the evaluator drive their conversation through (agent/pipeline/nodes.py,
2026-09-10 architecture replacement of the swarm pipeline's
_worker_loop/_synthesize_loop).

_parse_worker_reply itself is carried over verbatim from the retired
pipeline (same name, same behavior, same bug history -- see
tests/test_worker_reply_validation.py, now retired alongside it) --
re-tested here rather than assumed, since this IS a from-scratch rewrite of
the module it lives in, not a copy-paste.

The ACTION-protocol-leak fix (never let a tool-call reply's raw text become
the exhaustion fallback `output`) is carried over too, and re-tested below
for the same reason: it was caught live in the swarm pipeline
(debug_pipeline_agents3.py, nphard_tsp_02) and deliberately built into
_tool_loop from the start rather than rediscovered, but "written correctly"
and "verified correct" are not the same claim.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.pipeline import nodes as pn
from agent.pipeline import tools as pt


def test_marker_less_reply_is_unparseable_not_final():
    kind, tool_name, body = pn._parse_worker_reply("just some prose with no marker at all")
    assert kind == "unparseable"


def test_empty_reply_is_unparseable():
    kind, tool_name, body = pn._parse_worker_reply("")
    assert kind == "unparseable"


def test_final_marker_with_nothing_after_it_is_unparseable():
    kind, tool_name, body = pn._parse_worker_reply("I'm done.\nFINAL:")
    assert kind == "unparseable"


def test_final_marker_with_only_whitespace_after_it_is_unparseable():
    kind, tool_name, body = pn._parse_worker_reply("FINAL:\n   \n")
    assert kind == "unparseable"


def test_final_marker_with_real_content_is_still_final():
    kind, tool_name, body = pn._parse_worker_reply("FINAL:\ndef f(): return 1")
    assert kind == "final"
    assert body == "def f(): return 1"


def test_action_marker_parses_the_tool_name_and_code_body():
    kind, tool_name, body = pn._parse_worker_reply("ACTION: execute_python\nCODE:\nprint(1)")
    assert kind == "action"
    assert tool_name == "execute_python"
    assert body == "print(1)"


class _FakeMultiStreamModel:
    """Returns each of `replies` in order, one per .stream() call -- for
    exercising a tool loop that must retry N times before succeeding."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[str]] = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        reply = self._replies.pop(0)
        yield AIMessageChunk(content=reply)


def test_tool_loop_returns_a_final_body_with_the_fence_stripped():
    llm = _FakeMultiStreamModel(["FINAL:\n```python\ndef f(): return 1\n```"])

    output = pn._tool_loop(llm, [])

    assert output == "def f(): return 1"
    assert len(llm.calls) == 1


def test_tool_loop_retries_a_marker_less_reply_with_corrective_feedback():
    llm = _FakeMultiStreamModel([
        "I think the answer is probably something like this",  # unparseable
        "FINAL:\ndef f(): return 1",
    ])

    output = pn._tool_loop(llm, [])

    assert output == "def f(): return 1"
    assert len(llm.calls) == 2
    assert pn.UNPARSEABLE_FEEDBACK in llm.calls[1][-1]


def test_tool_loop_retries_an_empty_final_reply():
    llm = _FakeMultiStreamModel(["FINAL:\n", "FINAL:\ndef f(): return 2"])

    output = pn._tool_loop(llm, [])

    assert output == "def f(): return 2"
    assert len(llm.calls) == 2


def test_tool_loop_dispatches_a_real_tool_and_feeds_back_its_result():
    llm = _FakeMultiStreamModel([
        "ACTION: execute_python\nCODE:\nprint(1 + 1)",
        "FINAL:\ndone",
    ])

    output = pn._tool_loop(llm, [])

    assert output == "done"
    tool_result = llm.calls[1][-1]
    assert "TOOL RESULT" in tool_result
    assert "stdout:\n2" in tool_result


def test_tool_loop_refuses_a_tool_outside_the_allowed_set():
    llm = _FakeMultiStreamModel([
        "ACTION: delete_everything\nCODE:\nrm -rf /",
        "FINAL:\ndone",
    ])

    output = pn._tool_loop(llm, [])

    assert output == "done"
    assert "is not available" in llm.calls[1][-1]


def test_tool_loop_exhausts_gracefully_when_every_reply_is_unparseable():
    replies = [f"unparseable attempt {i}" for i in range(pn.MAX_TOOL_ITERATIONS)]
    llm = _FakeMultiStreamModel(replies)

    output = pn._tool_loop(llm, [])

    assert output == replies[-1]
    assert len(llm.calls) == pn.MAX_TOOL_ITERATIONS


def test_tool_loop_never_leaks_raw_action_protocol_text_when_it_exhausts_mid_action():
    # Regression: a tool-call reply is not a candidate answer, so if the
    # loop exhausts its budget right after an ACTION step, the fallback
    # output must never be that reply's literal "ACTION: ...\nCODE:\n..."
    # protocol text.
    replies = [f"ACTION: execute_python\nCODE:\nprint({i})\n" for i in range(pn.MAX_TOOL_ITERATIONS)]
    llm = _FakeMultiStreamModel(replies)

    output = pn._tool_loop(llm, [])

    assert output == ""
    assert "ACTION:" not in output


def test_tool_loop_falls_back_to_the_last_final_attempt_not_a_later_action_reply(monkeypatch):
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 2)
    llm = _FakeMultiStreamModel([
        "FINAL:\ndef f(:\n",  # a real (broken) candidate answer
        "ACTION: execute_python\nCODE:\nprint(1)\n",
    ])

    output = pn._tool_loop(llm, [])

    assert output == "def f(:"


# ---- a reply that contains SEVERAL tool calls ----------------------------


def test_code_body_stops_at_the_next_action_rather_than_running_the_whole_reply():
    """Asked for one step, the model will sometimes lay out its whole plan at
    once. The body used to be "everything after the first CODE:", so the
    literal lines "ACTION: execute_bash" and "CODE:" were handed to the shell
    as part of the command -- caught in a Terminal-Bench transcript as
    `ACTION:: command not found` and an exit code of 127 for a command whose
    real work had actually succeeded.
    """
    reply = (
        "ACTION: execute_bash\nCODE:\n"
        "tar -czf a.tgz -C /opt data\n\n"
        "ACTION: execute_bash\nCODE:\n"
        "gpg --symmetric a.tgz\n"
    )

    kind, tool_name, body = pn._parse_worker_reply(reply)

    assert (kind, tool_name) == ("action", "execute_bash")
    assert body == "tar -czf a.tgz -C /opt data"


def test_code_body_stops_at_a_trailing_final_block():
    reply = "ACTION: execute_bash\nCODE:\nls -la\n\nFINAL:\neverything is done\n"

    _, _, body = pn._parse_worker_reply(reply)

    assert body == "ls -la"


def test_a_directive_word_inside_a_command_is_not_a_directive():
    """Anchored to the start of a line, so a command that merely mentions one
    keeps working."""
    reply = 'ACTION: execute_bash\nCODE:\necho "ACTION: done" && echo "FINAL: yes"\n'

    _, _, body = pn._parse_worker_reply(reply)

    assert body == 'echo "ACTION: done" && echo "FINAL: yes"'


def test_only_the_first_tool_call_of_a_multi_call_reply_is_executed(monkeypatch):
    """The rest is not lost -- the loop feeds the first result back and the
    model reissues what it still wants."""
    calls = []

    def _fake_bash(body):
        calls.append(body)
        return pt.ToolResult(stdout="ok", stderr="", returncode=0)

    monkeypatch.setitem(pn.TOOL_DISPATCH, "execute_bash", _fake_bash)
    llm = _FakeMultiStreamModel([
        "ACTION: execute_bash\nCODE:\nfirst\n\nACTION: execute_bash\nCODE:\nsecond",
        "FINAL:\ndone",
    ])

    assert pn._tool_loop(llm, [HumanMessage("go")]) == "done"
    assert calls == ["first"]
