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
from langchain_core.messages import AIMessageChunk, HumanMessage, HumanMessage

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


def test_tool_loop_gives_up_after_a_run_of_unparseable_replies():
    """It used to spend the whole budget telling a silent model off. It now
    stops after MAX_CONSECUTIVE_DEAD_REPLIES, still handing back the last
    attempt, which is what the caller falls back on."""
    replies = [f"unparseable attempt {i}" for i in range(pn.MAX_TOOL_ITERATIONS)]
    llm = _FakeMultiStreamModel(replies)

    output = pn._tool_loop(llm, [])

    assert output == replies[pn.MAX_CONSECUTIVE_DEAD_REPLIES - 1]
    assert len(llm.calls) == pn.MAX_CONSECUTIVE_DEAD_REPLIES


def test_one_bad_reply_is_still_retried_and_recovered():
    """The early exit must not cost the case the retry was built for."""
    llm = _FakeMultiStreamModel(["no markers here", "FINAL:\nthe answer"])

    assert pn._tool_loop(llm, [HumanMessage("go")]) == "the answer"


def test_the_dead_reply_run_resets_after_a_good_one(monkeypatch):
    """Only CONSECUTIVE non-answers end it -- a model alternating between a
    bad reply and a real tool call is making progress, slowly."""
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 10)
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "execute_bash",
        lambda body: pt.ToolResult(stdout="ok", stderr="", returncode=0),
    )
    llm = _FakeMultiStreamModel([
        "", "", "ACTION: execute_bash\nCODE:\nls",
        "", "", "ACTION: execute_bash\nCODE:\nls",
        "FINAL:\ndone",
    ])

    assert pn._tool_loop(llm, [HumanMessage("go")]) == "done"


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


# ---- the record of what has already been done ----------------------------


def test_tool_loop_records_each_call_it_makes(monkeypatch):
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "write_file",
        lambda body: pt.ToolResult(stdout="wrote m.py (3 lines)", stderr="", returncode=0),
    )
    llm = _FakeMultiStreamModel([
        "ACTION: write_file\nCODE:\nm.py\nprint(1)",
        "FINAL:\ndone",
    ])
    taken = []

    pn._tool_loop(llm, [HumanMessage("go")], taken)

    assert taken == ["write_file m.py -> ok: wrote m.py (3 lines)"]


def test_a_failing_call_records_the_error_not_the_success():
    """A compiler error is the thing worth carrying into the next attempt."""
    result = pt.ToolResult(
        stdout="", stderr="error[E0433]: failed to resolve\n  --> main.rs:4:5", returncode=1,
    )

    line = pn._summarise_action("execute_bash", "rustc main.c.rs", result)

    assert line.startswith("execute_bash rustc main.c.rs -> FAILED (exit 1)")
    assert "E0433" in line


def test_the_action_record_survives_into_the_judgment():
    """The loop this fixes: a role's tool conversation used to die with the
    node, so a re-invoked solver had no idea it had already written the file --
    measured at 166 writes to one path across five rounds, none ever compiled.

    The loop keeps the real conversation now, so the lossy record is no longer
    the only memory. It is still what the EVALUATOR reads, which is the half
    that could never see it at all.
    """
    state = {
        "messages": [HumanMessage("build it")],
        "actions": ["solve: write_file main.rs -> ok: wrote main.rs (56 lines)"],
    }

    block = pn._actions_block(state)

    assert "WHAT HAS ALREADY BEEN DONE" in block
    assert "write_file main.rs" in block


def test_nothing_is_shown_before_anything_has_been_done():
    assert pn._actions_block({"messages": [HumanMessage("build it")], "actions": []}) == ""


def test_only_the_most_recent_actions_are_shown():
    """Bounded: a long run accumulates hundreds and this goes into the
    judgment prompt."""
    state = {
        "messages": [HumanMessage("go")],
        "actions": [f"solve: execute_bash step{i} -> ok" for i in range(pn._ACTIONS_SHOWN + 10)],
    }

    block = pn._actions_block(state)

    assert "step0 " not in block
    assert f"step{pn._ACTIONS_SHOWN + 9} " in block


def test_repeated_writes_to_one_file_are_named_in_the_result(monkeypatch):
    """The loop worth catching does not repeat verbatim: a slightly different
    draft every time looks different byte for byte and is the same
    non-progress -- 166 writes to one path, none of them ever compiled."""
    drafts = iter(range(100))

    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "write_file",
        lambda body: pt.ToolResult(stdout="wrote m.rs", stderr="", returncode=0),
    )
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 8)
    llm = _FakeMultiStreamModel([
        f"ACTION: write_file\nCODE:\nm.rs\nversion {next(drafts)}" for _ in range(4)
    ] + ["FINAL:\ndone"])

    pn._tool_loop(llm, [HumanMessage("go")])

    assert "NOTE:" not in llm.calls[1][-1]                      # 1st write
    assert "NOTE:" not in llm.calls[2][-1]                      # 2nd, still fine
    assert "3 write_file calls in a row against `m.rs`" in llm.calls[3][-1]


def test_a_different_target_resets_the_run(monkeypatch):
    """Only CONSECUTIVE calls against one target count -- alternating between
    writing and running is exactly the behaviour this is trying to produce."""
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "write_file",
        lambda body: pt.ToolResult(stdout="wrote", stderr="", returncode=0),
    )
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "execute_bash",
        lambda body: pt.ToolResult(stdout="ok", stderr="", returncode=0),
    )
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 8)
    llm = _FakeMultiStreamModel([
        "ACTION: write_file\nCODE:\nm.rs\na",
        "ACTION: execute_bash\nCODE:\nrustc m.rs",
        "ACTION: write_file\nCODE:\nm.rs\nb",
        "ACTION: execute_bash\nCODE:\nrustc m.rs",
        "FINAL:\ndone",
    ])

    pn._tool_loop(llm, [HumanMessage("go")])

    assert all("NOTE:" not in call[-1] for call in llm.calls)


def test_action_target_is_the_path_for_file_tools_and_the_command_otherwise():
    assert pn._action_target("write_file", "src/m.rs\nbody here") == "src/m.rs"
    assert pn._action_target("execute_bash", "rustc m.rs").startswith("execute_bash:rustc")


def test_chat_template_tokens_are_stripped_from_a_command():
    """Observed in a Terminal-Bench run: the model emitted <|tool_call_start|>
    mid-body, and bash answered "syntax error near unexpected token `|'".
    They are markup for the serialiser, never content."""
    reply = "ACTION: execute_bash\nCODE:\ncurl -v example.com\n\n<|tool_call_start|> 2>&1 | head -50"

    _, _, body = pn._parse_worker_reply(reply)

    assert "<|" not in body
    assert body.startswith("curl -v example.com")


def test_a_second_failure_against_the_same_target_is_escalated_at_once(monkeypatch):
    """Re-running something that succeeded is wasteful; re-running something
    that failed, unchanged, means the error was never read -- and it is the
    cheapest thing for a model to produce."""
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "execute_bash",
        lambda body: pt.ToolResult(stdout="", stderr="boom", returncode=1),
    )
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 8)
    llm = _FakeMultiStreamModel([
        "ACTION: execute_bash\nCODE:\nmake",
        "ACTION: execute_bash\nCODE:\nmake",
        "FINAL:\ndone",
    ])

    pn._tool_loop(llm, [HumanMessage("go")])

    assert "NOTE:" not in llm.calls[1][-1]                       # first failure
    assert "has already failed once" in llm.calls[2][-1]          # second


def test_a_failure_after_a_success_is_not_escalated(monkeypatch):
    """Only a repeat of something that already failed -- a command that worked
    before and broke now is new information, not a repeated mistake."""
    outcomes = iter([0, 1])
    monkeypatch.setitem(
        pn.TOOL_DISPATCH, "execute_bash",
        lambda body: pt.ToolResult(stdout="", stderr="", returncode=next(outcomes)),
    )
    monkeypatch.setattr(pn, "MAX_TOOL_ITERATIONS", 8)
    llm = _FakeMultiStreamModel([
        "ACTION: execute_bash\nCODE:\nmake",
        "ACTION: execute_bash\nCODE:\nmake",
        "FINAL:\ndone",
    ])

    pn._tool_loop(llm, [HumanMessage("go")])

    assert all("has already failed" not in call[-1] for call in llm.calls)


# --------------------------------------------------------------------------
# The emit-path contract
# --------------------------------------------------------------------------
#
# A text protocol has nothing structurally enforcing its action schema, which
# is the dominant production failure class: plausible reasoning decoupled from
# the action contract. The counter-evidence is that it closes in code -- one
# harness eliminated every illegal move across 145 environments by validating
# on the emit path. These are the shapes real models produced against THIS
# protocol, each of which used to cost a wasted model call.

ALLOWED = pt.TOOL_DISPATCH


def _resolved(reply: str) -> str:
    _, tool_name, _ = pn._parse_worker_reply(reply, allowed=ALLOWED)
    return tool_name


def test_prose_after_the_tool_name_still_names_the_tool():
    """`ACTION: execute_bash to list the files` reached the dispatch table
    verbatim, missed, and came back as "not available"."""
    assert _resolved("ACTION: execute_bash to list the files\nCODE:\nls") == "execute_bash"


def test_a_tool_name_in_backticks_or_bold_is_still_a_tool_name():
    """Models format. The protocol should not care."""
    assert _resolved("ACTION: `execute_bash`\nCODE:\nls") == "execute_bash"
    assert _resolved("ACTION: **execute_bash**\nCODE:\nls") == "execute_bash"


def test_a_full_stop_does_not_invent_a_new_tool():
    assert _resolved("ACTION: execute_bash.\nCODE:\nls") == "execute_bash"


def test_a_typo_resolves_to_what_it_obviously_meant():
    assert _resolved("ACTION: execute_pyton\nCODE:\nprint(1)") == "execute_python"


def test_a_genuinely_different_name_is_not_guessed_at():
    """`read_file` and `edit_file` differ by more than a typo, and guessing
    wrong runs the WRONG TOOL rather than wasting a round trip. The cutoff is
    set so this stays a miss."""
    assert _resolved("ACTION: frobnicate_thing\nCODE:\nx") == "frobnicate_thing"


def test_a_fuzzy_match_is_only_tried_on_the_first_word():
    """Otherwise a parser starts inventing tool calls out of prose. An exact
    match anywhere wins; a near miss only from the token in the tool slot."""
    kind, tool_name, _ = pn._parse_worker_reply(
        "ACTION: please go and reed the file for me\nCODE:\nx", allowed=ALLOWED,
    )
    assert kind == "action"
    assert tool_name == "please"


def test_whichever_marker_comes_first_wins():
    """A reply that answered and then suggested a follow-up step used to have
    the ANSWER silently discarded and the command run instead -- the loop then
    had nothing to return and paid for another exchange to be told again."""
    kind, _, body = pn._parse_worker_reply(
        "FINAL:\nthe report is written\nACTION: execute_bash\nCODE:\ncat report.md",
        allowed=ALLOWED,
    )
    assert kind == "final"
    assert body.startswith("the report is written")


def test_an_action_before_an_answer_is_still_an_action():
    kind, tool_name, body = pn._parse_worker_reply(
        "ACTION: execute_bash\nCODE:\nls\nFINAL:\nand here is the answer",
        allowed=ALLOWED,
    )
    assert (kind, tool_name, body) == ("action", "execute_bash", "ls")


def test_an_action_with_no_body_is_refused_before_it_runs():
    """It used to reach the shell as an empty command and come back as a
    returncode the model then had to interpret."""
    problem = pn._action_problem("execute_bash", "", ALLOWED)
    assert "no CODE: body" in problem


def test_an_unavailable_tool_is_still_refused_by_name():
    problem = pn._action_problem("frobnicate", "x", ALLOWED)
    assert "not available" in problem
    assert "frobnicate" in problem


def test_a_well_formed_action_has_no_problem():
    assert pn._action_problem("execute_bash", "ls", ALLOWED) == ""


def test_the_loops_own_pseudo_tools_survive_resolution():
    """`switch_mode`, `delegate` and `ask_user` are not in TOOL_DISPATCH --
    they change the loop's state rather than returning a result -- so name
    resolution has to know them or it would fuzzy-match them into something
    else. Whether a given loop HONOURS one is that loop's decision."""
    for name in ("switch_mode", "delegate", "ask_user"):
        assert _resolved(f"ACTION: **{name}**\nCODE:\nx") == name


def test_a_run_with_extra_tools_bound_resolves_those_too(monkeypatch):
    """A benchmark's task-specific tools are the names a model is MOST likely
    to decorate or misspell, having seen them once in a prompt."""
    allowed = {**pt.TOOL_DISPATCH, "gmail_send_message": lambda body: None}
    _, tool_name, _ = pn._parse_worker_reply(
        "ACTION: `gmail_send_message`\nCODE:\n{}", allowed=allowed,
    )
    assert tool_name == "gmail_send_message"
