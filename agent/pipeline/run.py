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

Bounding that conversation memory (2026-09-10, same day, Phase 2 of
claude/otto-tiered-memory-design.md -- the paragraph above flagged
"deliberately NOT built here" as its own explicit follow-up): `history` no
longer needs to be the whole raw, unbounded list. `_initial()` gained a
`memory_context` parameter, seeded from whatever `agent/memory/wiring.py`'s
`history_for_graph()` produces out of a session's TieredQueue -- the
recent, still-verbatim tail becomes `history` exactly as before (so
`messages`/`_conversation_so_far()` need no changes at all), and anything
older that's been compacted away becomes `memory_context`, landing in
`state["context"]`'s existing "CONTEXT GATHERED SO FAR:" display instead.
Both `run_pipeline()` and `run_pipeline_stream()` also now open this
session's MemoryStore (agent/memory/store.py's `~/.otto/memory/
<session_id>.db`) and bind it for the duration of the graph call
(agent/memory/session.py's `bind_store()`) -- how agent/pipeline/tools.py's
new `recall_memory` tool finds out which session's history to search,
since a plain TOOL_DISPATCH function has no other way to see `session_id`.

Pausing for a person mid-run (2026-09-10, same day, agent/pipeline/
nodes.py's seventh refinement -- has its own module docstring section
with the full design discussion): app.compile() already carries a
checkpointer (InMemorySaver, nodes.py) from before this, for an unrelated
reason (giving every turn its own disposable thread id -- _graph_thread_id
below). That's also exactly what LangGraph's own interrupt()/
Command(resume=...) mechanism needs, so nodes.py's new `ask_user` node
reuses it rather than inventing a second one. When a run hits that node,
`app.stream()` yields LangGraph's own `{"__interrupt__": (Interrupt(...),)
}` update instead of continuing -- run_pipeline_stream() (below) turns
that into a `{"__ask__": {"question", "choices", "thread_id"}}` event for
callers and then RETURNS (the generator just ends there, mid-turn; there
is no `__final__` yet). `resume_pipeline_stream()` (below, new) is how a
caller with an answer continues that SAME checkpoint thread -- not a new
turn, `Command(resume=answer)` picks the graph up exactly where ask_user()
paused it, plan/context/active_step/everything intact. It can itself hit
another `__ask__` (a chained question) or finish with `__final__`, same as
run_pipeline_stream() -- agent/cli/chat.py's and agent/cli/tui.py's own
run-turn loops handle both the same way, looping until `__final__` shows
up. Each half gets its own Langfuse observation rather than one span held
open across however long a person takes to answer -- tagged the same way
(session_id, "pipeline" in the tags) so both halves of one externally
visible turn are still easy to find together.
"""
import logging
import uuid
from collections.abc import Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from langgraph.types import Command

from agent.memory.session import bind_store
from agent.memory.store import MemoryStore
from agent.pipeline.nodes import _RECURSION_SAFETY_NET, ROUTER, app
from agent.pipeline.state import AgentState

logger = logging.getLogger(__name__)


def _initial(text: str, *, history: Sequence[BaseMessage] = (), memory_context: str = "") -> dict:
    return {
        "messages": [*history, HumanMessage(text)],
        "board": [],
        "round": 0,
        "actions": [],
        "node": None,
        "feedback": "",
        "output": None,
        "context": memory_context,
        "plan": None,
        "active_step": None,
        "node_error": None,
        "pending_question": None,
        "pending_choices": None,
        "asking_role": None,
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


def _as_ask_event(update: dict, graph_thread_id: str) -> dict | None:
    """If `update` is LangGraph's own interrupt shape (`{"__interrupt__":
    (Interrupt(value=..., id=...),)}`, yielded by app.stream() the moment
    a node calls interrupt() -- nodes.py's ask_user, seventh refinement),
    return the `{"__ask__": ...}` event run_pipeline_stream()/
    resume_pipeline_stream() actually yield to callers. None otherwise --
    every other update passes through unchanged.
    """
    payload = update.get("__interrupt__")
    if not payload:
        return None
    value = payload[0].value or {}
    return {
        "__ask__": {
            "question": value.get("question", ""),
            "choices": value.get("choices") or [],
            "thread_id": graph_thread_id,
        }
    }


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


def run_pipeline(
    text: str, *, session_id: str, history: Sequence[BaseMessage] = (), memory_context: str = "",
) -> AgentState:
    """Prewarm, invoke, score, return the finished AgentState.

    One call in, one finished result out -- for scripted/benchmark callers
    (agent/eval/'s golden-dataset runner). The interactive shell wants to
    watch it happen instead -- that's run_pipeline_stream(), below, not a
    different mode of this function.

    `history`/`memory_context` are the conversation BEFORE `text` -- module
    docstring's "Bounding that conversation memory". Both empty by default,
    so every existing single-turn caller is unaffected.
    """
    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, history=history, memory_context=memory_context)
    client = get_client()
    handler = CallbackHandler()
    config = _config(_graph_thread_id(session_id), handler)
    store = MemoryStore.for_session(session_id)

    with bind_store(store):
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


def run_pipeline_stream(
    text: str, *, session_id: str, history: Sequence[BaseMessage] = (), memory_context: str = "",
):
    """Same prewarm, tracing and scoring as run_pipeline(), but yields each
    graph update as it happens (`stream_mode="updates"`) instead of
    invoking and returning. What the interactive shell/TUI watch live.

    `history`/`memory_context` are the conversation BEFORE `text` -- module
    docstring's "Bounding that conversation memory". Both empty by default,
    so every existing single-turn caller is unaffected.

    The last item yielded is usually `{"__final__": AgentState,
    "__trace_id__": str | None}` -- a plain dict, not a Command/node delta,
    matching the retired run_pipeline_stream()'s/run_code_stream()'s exact
    shape. The one exception (module docstring, "Pausing for a person
    mid-run"): a run that hits nodes.py's ask_user node instead yields
    `{"__ask__": {"question", "choices", "thread_id"}}` and ENDS there,
    mid-turn -- no `__final__` this call. `resume_pipeline_stream()`,
    below, is how a caller with an answer continues that same run -- it
    re-binds this same session's MemoryStore itself, so recall_memory stays
    usable across a resume too.
    """
    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, history=history, memory_context=memory_context)
    client = get_client()
    handler = CallbackHandler()
    graph_thread_id = _graph_thread_id(session_id)
    config = _config(graph_thread_id, handler)
    store = MemoryStore.for_session(session_id)

    with bind_store(store):
        with propagate_attributes(
            trace_name="otto:pipeline",
            session_id=session_id,
            tags=["pipeline"],
        ):
            with client.start_as_current_observation(
                name="otto:pipeline", as_type="agent", input=text
            ) as run_span:
                for update in app.stream(initial, config, stream_mode="updates"):
                    ask = _as_ask_event(update, graph_thread_id)
                    if ask is not None:
                        run_span.update(output="(paused -- awaiting your answer)")
                        yield ask
                        return
                    yield update
                final = app.get_state(config).values
                _score(run_span, final)
                trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}


def resume_pipeline_stream(answer, *, thread_id: str, session_id: str):
    """Continue a run that paused on an `{"__ask__": ...}` event (either
    run_pipeline_stream()'s or a previous resume_pipeline_stream()'s) --
    module docstring, "Pausing for a person mid-run". `thread_id` is that
    event's own `"thread_id"` -- the SAME LangGraph checkpoint thread the
    original run_pipeline_stream() call used, resumed via
    `Command(resume=answer)` exactly where nodes.py's ask_user node
    paused it (plan/context/active_step/everything else untouched), not
    restarted as a fresh turn.

    Same yield shape as run_pipeline_stream(): usually ends in
    `{"__final__": ...}`, but can itself end in another `{"__ask__": ...}`
    if the re-invoked specialist gets stuck again -- callers loop on this
    the same way they loop on run_pipeline_stream() (agent/cli/chat.py,
    agent/cli/tui.py). Re-binds `session_id`'s own MemoryStore for this
    continuation too (module docstring, "Bounding that conversation
    memory") -- recall_memory stays usable after a resume, not just on a
    run's first `run_pipeline_stream()` call.
    """
    client = get_client()
    handler = CallbackHandler()
    config = _config(thread_id, handler)
    store = MemoryStore.for_session(session_id)

    with bind_store(store):
        with propagate_attributes(
            trace_name="otto:pipeline",
            session_id=session_id,
            tags=["pipeline", "resumed"],
        ):
            with client.start_as_current_observation(
                name="otto:pipeline:resume", as_type="agent", input=str(answer)
            ) as run_span:
                for update in app.stream(Command(resume=answer), config, stream_mode="updates"):
                    ask = _as_ask_event(update, thread_id)
                    if ask is not None:
                        run_span.update(output="(paused -- awaiting your answer)")
                        yield ask
                        return
                    yield update
                final = app.get_state(config).values
                _score(run_span, final)
                trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}
