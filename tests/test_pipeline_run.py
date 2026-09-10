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
