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
"""
from langchain_core.messages import AIMessage, HumanMessage

from agent.pipeline.nodes import _RECURSION_SAFETY_NET
from agent.pipeline.run import _config, _graph_thread_id, _initial


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
