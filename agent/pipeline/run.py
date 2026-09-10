"""Prewarm, invoke/stream, score, return -- same shape as the retired
agent/graph/code_run.py and the swarm pipeline's own run.py it replaces,
retargeted at the router/planner/solver/summarizer/finder/evaluator graph
in agent.pipeline.nodes.

No `agents` parameter anywhere in this file (2026-09-10): the new graph has
nothing to size -- one router dispatch, one specialist, one evaluator, per
round -- so there is no swarm-width knob left to validate, thread through,
or score. Callers that used to pass `agents=` (agent/eval/, agent/cli/chat.py,
agent/cli/tui.py) all lose that parameter in the same change.

Conversation memory (2026-09-10, same day, design call: live-tested with
"Hi" / "Solve N Queens with brute force" / "improve above solution" -- the
third turn had no idea what "above solution" meant and looped). Before
this, `_initial()` only ever seeded `messages` with the CURRENT turn's
text -- every turn started the graph from a blank slate, even though
chat.py's/tui.py's own `Session.history` was already tracking the whole
conversation client-side (used for nothing but /new resets and the visual
transcript). `history` (new, optional, keyword-only on `_initial()`,
`run_pipeline()` and `run_pipeline_stream()`) is that same list, handed
straight to the graph: `messages` becomes `[*history, HumanMessage(text)]`
instead of just `[HumanMessage(text)]`. `text` stays the required
positional argument and `history` defaults to `()` so every existing
caller that only ever runs one turn per session (agent/eval/'s golden
runner, debug_pipeline.py) needs zero changes and sees zero behavior
difference. agent/pipeline/nodes.py's prompt-builders read the rest of
`state["messages"]` (everything before the last one) as prior
conversation -- see its module docstring for that half.

Deliberately NOT built here: any kind of summarization, truncation, or
cross-SESSION persistence. `history` is exactly the in-memory list the
REPL/TUI already had; a long-running chat's prompt grows every turn with
nothing capping it. That's the same "project profile"/"personal lessons"
territory claude/otto-memory-design.md already specs out for a *different*
purpose (learning across runs, not remembering within one) against the
now-replaced code hive -- worth revisiting there, not smuggled into this
fix, which only closes the "does the graph even see what I said two turns
ago" gap.
"""
import logging
import uuid
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agent.pipeline.nodes import _RECURSION_SAFETY_NET, ROUTER, app
from agent.pipeline.state import AgentState

logger = logging.getLogger(__name__)


def _initial(text: str, *, history: Sequence[BaseMessage] = ()) -> dict:
    return {
        "messages": [*history, HumanMessage(text)],
        "board": [],
        "round": 0,
        "node": None,
        "feedback": "",
        "output": None,
        "context": "",
        "plan": None,
        "active_step": None,
        "node_error": None,
        "final_output": None,
    }


def _config(graph_thread_id: str, handler) -> dict:
    # The overseer (router) is re-invoked after every node with no cap on
    # how many times it may retry (nodes.py, 2026-09-10 design call) -- the
    # only backstop left is LangGraph's own recursion_limit, sized by
    # nodes.py's _RECURSION_SAFETY_NET as pure infra insurance against a
    # genuinely runaway loop, not a business rule a real request should hit.
    return {
        "recursion_limit": _RECURSION_SAFETY_NET,
        "configurable": {"thread_id": graph_thread_id},
        "callbacks": [handler],
    }


def _score(run_span, final: AgentState) -> None:
    run_span.update(output=final["final_output"])
    run_span.score_trace(name="dispatch_rounds", value=final["round"], data_type="NUMERIC")
    if final.get("node"):
        run_span.score_trace(name="last_node", value=final["node"], data_type="CATEGORICAL")


def _graph_thread_id(session_id: str) -> str:
    """A fresh LangGraph checkpoint thread for THIS turn only.

    Same reasoning as the retired code_run.py's/swarm pipeline's identical
    helper: every reducer-annotated list in AgentState (board) accumulates
    on every write, including the first one -- reusing one checkpoint
    thread across turns would let turn 1's leftovers bleed into turn 2's
    state. A disposable thread id per turn makes _initial()'s empty lists
    actually the starting state, every time.
    """
    return f"{session_id}:{uuid.uuid4().hex[:8]}"


def run_pipeline(text: str, *, session_id: str, history: Sequence[BaseMessage] = ()) -> AgentState:
    """Prewarm, invoke, score, return the finished AgentState.

    One call in, one finished result out -- for scripted/benchmark callers
    (agent/eval/'s golden-dataset runner). The interactive shell wants to
    watch it happen instead -- that's run_pipeline_stream(), below, not a
    different mode of this function.

    `history` is the conversation BEFORE `text` -- module docstring. Empty
    by default, so every existing single-turn caller is unaffected.
    """
    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, history=history)
    client = get_client()
    handler = CallbackHandler()
    config = _config(_graph_thread_id(session_id), handler)

    with propagate_attributes(
        trace_name="otto:pipeline",
        session_id=session_id,
        tags=["pipeline"],
    ):
        with client.start_as_current_observation(
            name="otto:pipeline", as_type="agent", input=text
        ) as run_span:
            final = app.invoke(initial, config)
            _score(run_span, final)

    return final


def run_pipeline_stream(text: str, *, session_id: str, history: Sequence[BaseMessage] = ()):
    """Same prewarm, tracing and scoring as run_pipeline(), but yields each
    graph update as it happens (`stream_mode="updates"`) instead of
    invoking and returning. What the interactive shell/TUI watch live.

    `history` is the conversation BEFORE `text` -- module docstring. Empty
    by default, so every existing single-turn caller is unaffected.

    The last item yielded is always `{"__final__": AgentState,
    "__trace_id__": str | None}` -- a plain dict, not a Command/node delta,
    matching the retired run_pipeline_stream()'s/run_code_stream()'s exact
    shape.
    """
    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, history=history)
    client = get_client()
    handler = CallbackHandler()
    config = _config(_graph_thread_id(session_id), handler)

    with propagate_attributes(
        trace_name="otto:pipeline",
        session_id=session_id,
        tags=["pipeline"],
    ):
        with client.start_as_current_observation(
            name="otto:pipeline", as_type="agent", input=text
        ) as run_span:
            for update in app.stream(initial, config, stream_mode="updates"):
                yield update
            final = app.get_state(config).values
            _score(run_span, final)
            trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}
