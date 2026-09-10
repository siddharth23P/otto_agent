"""Coverage for agent/pipeline/run.py's own helpers -- _initial(), _config(),
_graph_thread_id() -- without driving the real graph (that needs a live
LLM; agent/eval/'s golden harness and manual `otto eval`/`otto chat` runs
are what exercise the graph for real, per this repo's established offline-
vs-live test split).

No `agents` parameter anywhere here (2026-09-10): the router/planner/
solver/summarizer/finder/evaluator graph that replaced the swarm pipeline
has nothing to size -- see run.py's own module docstring.
"""
from agent.pipeline.nodes import MAX_DISPATCH_ROUNDS
from agent.pipeline.run import _config, _graph_thread_id, _initial


def test_initial_state_matches_the_agentstate_shape_with_empty_start_values():
    state = _initial("do the thing")

    assert [m.content for m in state["messages"]] == ["do the thing"]
    assert state["board"] == []
    assert state["round"] == 0
    assert state["node"] is None
    assert state["feedback"] == ""
    assert state["output"] is None
    assert state["final_output"] is None


def test_config_sizes_recursion_limit_off_max_dispatch_rounds():
    config = _config("thread-1", handler=object())

    assert config["recursion_limit"] == MAX_DISPATCH_ROUNDS * 4
    assert config["configurable"] == {"thread_id": "thread-1"}


def test_graph_thread_id_is_unique_per_call_and_keeps_the_session_id_prefix():
    a = _graph_thread_id("session-abc")
    b = _graph_thread_id("session-abc")

    assert a != b
    assert a.startswith("session-abc:")
    assert b.startswith("session-abc:")
