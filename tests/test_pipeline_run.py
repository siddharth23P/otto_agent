"""Coverage for agent/pipeline/run.py's own helpers -- _initial(), _config(),
_graph_thread_id() -- without driving the real graph (that needs a live
LLM; agent/eval/'s golden harness and manual `otto eval`/`otto chat` runs
are what exercise the graph for real, per this repo's established offline-
vs-live test split).

No `agents` parameter anywhere here (2026-09-10): the router/planner/
solver/summarizer/finder/evaluator graph that replaced the swarm pipeline
has nothing to size -- see run.py's own module docstring.

_config()'s recursion_limit is sized off nodes.py's _RECURSION_SAFETY_NET
(2026-09-10, second revision), not a multiple of a round cap -- there is no
round cap anymore (nodes.py's own module docstring), just a generous, pure
infra backstop against a genuinely runaway loop.

_initial()'s `history` parameter (2026-09-10, same day, sixth refinement
to nodes.py's module docstring -- live-tested: "improve above solution"
had nothing to improve, because every turn started `messages` from
scratch) prepends the conversation so far onto `messages`, ahead of the
current turn's own text. Defaults to `()` so every existing single-turn
caller (agent/eval/'s golden runner, debug_pipeline.py) is unaffected --
covered here by the pre-existing no-history test still passing unchanged.

_as_ask_event() (2026-09-10, same day, seventh refinement -- module
docstring, "Pausing for a person mid-run") is the one other piece of this
file that's a pure function, safe to test the same "no live LLM/graph"
way: turning LangGraph's own `{"__interrupt__": (Interrupt(...),)}`
update shape (yielded by app.stream() the moment nodes.py's ask_user node
calls interrupt()) into the `{"__ask__": {"question", "choices",
"thread_id"}}` event run_pipeline_stream()/resume_pipeline_stream()
actually yield to callers. The full pause-and-resume MECHANISM (a real
interrupt(), a real Command(resume=...), the checkpointer) needs an
actual LangGraph run to exercise -- that's tests/test_ask_user_node.py's
job, against the real compiled graph (pn.app) directly, sidestepping this
file's Langfuse wrapping entirely; genuinely new territory for this
offline suite (nothing before it drove the graph at all), justified by
how much of that mechanism nothing else here would ever catch a
regression in.
"""
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Interrupt

from agent.pipeline.nodes import _RECURSION_SAFETY_NET
from agent.pipeline.run import _as_ask_event, _config, _graph_thread_id, _initial


def test_initial_state_matches_the_agentstate_shape_with_empty_start_values():
    state = _initial("do the thing")

    assert [m.content for m in state["messages"]] == ["do the thing"]
    assert state["board"] == []
    assert state["round"] == 0
    assert state["node"] is None
    assert state["feedback"] == ""
    assert state["output"] is None
    assert state["context"] == ""
    assert state["plan"] is None
    assert state["active_step"] is None
    assert state["node_error"] is None
    assert state["pending_question"] is None
    assert state["pending_choices"] is None
    assert state["asking_role"] is None
    assert state["final_output"] is None


def test_initial_with_no_history_argument_is_unchanged_from_before():
    # No `history` kwarg at all -- the exact call every pre-existing caller
    # makes. Must produce exactly the single-message list it always has.
    state = _initial("do the thing")
    assert [m.content for m in state["messages"]] == ["do the thing"]


def test_initial_prepends_history_onto_messages_ahead_of_the_current_text():
    history = [HumanMessage("Hi"), AIMessage("hello!"), HumanMessage("solve N queens")]
    state = _initial("improve above solution", history=history)

    assert [m.content for m in state["messages"]] == [
        "Hi", "hello!", "solve N queens", "improve above solution",
    ]
    # the current turn's text is always the LAST message -- nodes.py's
    # body-builders rely on that to find "the task" vs. "prior turns".
    assert state["messages"][-1].content == "improve above solution"


def test_initial_with_empty_history_list_behaves_like_no_history():
    state = _initial("do the thing", history=[])
    assert [m.content for m in state["messages"]] == ["do the thing"]


def test_initial_seeds_context_from_memory_context(monkeypatch):
    # Phase 2 of claude/otto-tiered-memory-design.md: whatever a session's
    # TieredQueue has compacted away (agent/memory/wiring.py's
    # history_for_graph()) lands in state["context"] -- the SAME field
    # every prompt-builder in nodes.py already shows via "CONTEXT GATHERED
    # SO FAR:", not a new prompt section.
    state = _initial("do the thing", memory_context="EARLIER CONVERSATION (compacted):\n- talked about X")

    assert state["context"] == "EARLIER CONVERSATION (compacted):\n- talked about X"


def test_initial_with_no_memory_context_argument_is_unchanged_from_before():
    # No `memory_context` kwarg at all -- every pre-existing caller.
    state = _initial("do the thing")
    assert state["context"] == ""


def test_config_sizes_recursion_limit_off_the_recursion_safety_net(monkeypatch):
    config = _config("thread-1", handler=object())

    assert config["recursion_limit"] == _RECURSION_SAFETY_NET
    assert config["configurable"] == {"thread_id": "thread-1"}


def test_graph_thread_id_is_unique_per_call_and_keeps_the_session_id_prefix():
    a = _graph_thread_id("session-abc")
    b = _graph_thread_id("session-abc")

    assert a != b
    assert a.startswith("session-abc:")
    assert b.startswith("session-abc:")


def test_as_ask_event_returns_none_for_an_ordinary_update():
    assert _as_ask_event({"router": {"board": ["round 1: ..."]}}, "thread-1") is None


def test_as_ask_event_extracts_question_choices_and_thread_id():
    update = {"__interrupt__": (Interrupt(value={"question": "which N?", "choices": ["4", "8"]}, id="abc"),)}

    ask = _as_ask_event(update, "session-1:deadbeef")

    assert ask == {
        "__ask__": {"question": "which N?", "choices": ["4", "8"], "thread_id": "session-1:deadbeef"}
    }


def test_as_ask_event_defaults_choices_to_empty_list_when_absent():
    update = {"__interrupt__": (Interrupt(value={"question": "which N?"}, id="abc"),)}

    ask = _as_ask_event(update, "thread-1")

    assert ask["__ask__"]["choices"] == []


# --------------------------------------------------------------------------
# _stream_events -- the loop rewrite's stream shape
# --------------------------------------------------------------------------
#
# app.stream() yields a bare payload for one stream mode and a (mode, payload)
# TUPLE for several, so asking for "custom" alongside "updates" changes the
# shape every caller sees. _stream_events is the one place that knows it.
#
# Why "custom" is needed at all: "updates" emits once per node RETURN. That was
# fine when a node returned every few seconds. With one long-running agent loop
# it means nothing reaches the screen until the loop finishes -- `otto chat`
# would sit silent for minutes and then print one panel.

from agent.pipeline.run import _stream_events


def test_a_custom_event_passes_straight_through():
    """The loop emits these already node-shaped, so chat.py and tui.py need no
    changes -- their `next(iter(update.items()))` still works."""
    payload = {"agent": {"board": ["solve: execute_bash pytest -> exit 1"]}}
    events = list(_stream_events([("custom", payload)], "t1"))
    assert events == [(payload, False)]


def test_an_ordinary_update_passes_through_too():
    payload = {"evaluator": {"board": ["approved"]}}
    assert list(_stream_events([("updates", payload)], "t1")) == [(payload, False)]


def test_an_interrupt_in_the_updates_stream_becomes_an_ask_event():
    update = {"__interrupt__": (Interrupt(value={"question": "which one?", "choices": ["a", "b"]}),)}
    [(event, is_ask)] = list(_stream_events([("updates", update)], "thread-9"))
    assert is_ask
    assert event["__ask__"] == {"question": "which one?", "choices": ["a", "b"], "thread_id": "thread-9"}


def test_a_custom_event_is_never_mistaken_for_an_interrupt():
    """The loop's own events are not checked for `__interrupt__` -- only
    LangGraph puts that key in an updates payload, and scanning custom
    payloads for it would let a board line ending a run by accident."""
    payload = {"agent": {"board": ["__interrupt__ is just text here"]}}
    [(event, is_ask)] = list(_stream_events([("custom", payload)], "t1"))
    assert not is_ask


def test_live_events_arrive_before_the_node_returns():
    """The ordering that makes this worth doing: custom events are yielded as
    they happen, and the node's own update lands last."""
    stream = [
        ("custom", {"agent": {"board": ["step 1"]}}),
        ("custom", {"agent": {"board": ["step 2"]}}),
        ("updates", {"agent": {"board": ["done"]}}),
    ]
    boards = [e["agent"]["board"][0] for e, _ in _stream_events(stream, "t1")]
    assert boards == ["step 1", "step 2", "done"]


# --------------------------------------------------------------------------
# _salvage / _paused -- a run must not lose work it already did
# --------------------------------------------------------------------------
#
# Claw-Eval task C01 is why these exist. The agent had computed and verified a
# mortgage comparison; then the model put prose where a file path goes,
# Path.exists() raised OSError(ENAMETOOLONG), and 1096 seconds of correct work
# was thrown away. The grader saw a conversation with no assistant messages at
# all and scored completion 0.00.

from agent.pipeline.run import _paused, _salvage


def test_an_approved_answer_is_left_alone():
    assert _salvage({"final_output": "the real answer", "output": "draft"})["final_output"] == "the real answer"


def test_an_unapproved_candidate_is_promoted_rather_than_lost():
    """`output` is the agent's own best attempt. Reporting it beats reporting
    nothing, which is what a run that never reached approval used to do."""
    assert _salvage({"final_output": None, "output": "best effort"})["final_output"] == "best effort"


def test_a_blank_candidate_is_not_promoted_over_nothing():
    assert not _salvage({"final_output": None, "output": "   "}).get("final_output")


def test_a_crash_keeps_the_work_and_says_what_happened():
    state = _salvage({"output": "verified numbers"}, OSError("File name too long"))
    assert state["final_output"] == "verified numbers"
    assert any("File name too long" in line for line in state["board"])


def test_a_crash_with_nothing_reached_still_reports_the_failure():
    state = _salvage({}, RuntimeError("boom"))
    assert any("boom" in line for line in state["board"])


def test_an_interrupt_is_detected_rather_than_read_as_a_finished_run():
    """app.invoke returns NORMALLY on an interrupt, with final_output still
    None, and says so nowhere a caller looks."""
    assert _paused({"__interrupt__": [object()]})
    assert _paused({"pending_question": "which one?"})


def test_an_ordinary_finished_run_is_not_mistaken_for_a_pause():
    assert not _paused({"final_output": "done", "pending_question": None})
