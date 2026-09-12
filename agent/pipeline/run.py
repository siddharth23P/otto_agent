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

A `workspace` for every entry point (2026-09-12, design call: "we need
filesystem management so we can use it to write code and work on already
implemented codebases"). Binding a workspace was already possible and only
agent/eval/'s harnesses did it, each wrapping its OWN `with
bind_workspace(...)` around the call below. That worked for them and left
`otto chat`/`otto tui` with no way to reach it, because a contextvar set on
the CLI's thread is not visible inside a generator consumed on a Textual
worker thread. So the binding moves in here, next to `bind_budget` and
`bind_store`, which exist for exactly the same reason: a run-scoped fact the
graph cannot be handed as an argument has to be bound where the graph actually
runs. `workspace=None` is the old behaviour and stays the default, so the
harnesses' own outer `bind_workspace` still wins for their calls (nesting
unwinds correctly -- workspace.py's contract) and no single-turn caller
changes.
"""
import logging
import uuid
from collections.abc import Sequence
from contextlib import nullcontext

from langchain_core.messages import BaseMessage, HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from langgraph.types import Command

from agent.memory.session import bind_store
from agent.pipeline.budget import bind_budget, current_budget, default_budget
from agent.memory.store import MemoryStore
from agent.pipeline.nodes import _RECURSION_SAFETY_NET, ROUTER, app
from agent.pipeline.state import AgentState
from agent.pipeline.usage import UsageLedger, bind_usage, current_usage
from agent.pipeline.workspace import bind_workspace

logger = logging.getLogger(__name__)


def _initial(text: str, *, history: Sequence[BaseMessage] = (), memory_context: str = "") -> dict:
    return {
        "messages": [*history, HumanMessage(text)],
        "board": [],
        "actions": [],
        "node": None,
        "feedback": "",
        "output": None,
        "context": memory_context,
        "node_error": None,
        "pending_question": None,
        "pending_choices": None,
        "asking_role": None,
        "user_answer": None,
        "asked_qa": [],
        # Per TURN, not per session: a new request gets its own budget of
        # questions (agent/pipeline/state.py's `asks`).
        "asks": 0,
        # The agent loop's own conversation, carried across node returns
        # (agent/pipeline/state.py). None means "not started" -- the loop seeds
        # it on its first entry and hands back the version it finished with.
        "transcript": None,
        "mode": None,
        "mode_log": [],
        "model_calls": 0,
        "rejections": 0,
        "checklist": None,
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


def _salvage(final: dict | None, exc: Exception | None = None) -> dict:
    """Make sure a run hands back the best answer it actually reached.

    Three ways a run used to return nothing despite having done the work:

    * an unhandled exception anywhere unwound the whole graph. Seen live on
      Claw-Eval task C01 -- the agent had computed and verified a mortgage
      comparison, then put prose where a file path goes, `Path.exists()` raised
      OSError(ENAMETOOLONG), and 1096 seconds of correct work was thrown away.
      The grader saw a conversation with no assistant messages at all.
    * `app.invoke` returns NORMALLY on an interrupt, with `__interrupt__`
      spliced into the state and `final_output` still None. Every caller reads
      `final_output` and records an empty answer, and none of them checks.
    * the evaluator never approved, so `output` held a real candidate that
      `final_output` never received.

    All three have the same fix and it belongs in one place: `output` is the
    agent's own best attempt, and reporting it -- labelled as unverified -- beats
    reporting nothing.
    """
    state = dict(final or {})
    if exc is not None:
        state.setdefault("board", [])
        state["board"] = [*state.get("board", []), f"the run failed: {type(exc).__name__}: {exc}"]
    if not (state.get("final_output") or "").strip():
        candidate = (state.get("output") or "").strip()
        if not candidate and state.get("pending_question"):
            # The run stopped to ask something and nothing here can answer --
            # `run_pipeline` has no resume path, that is the streaming API.
            # The question IS the answer in that case, and saying it is both
            # honest and useful: a caller that wanted a decision learns which
            # decision is missing, instead of receiving nothing.
            #
            # Seen live on Claw-Eval T026, where the mutation gate correctly
            # stopped the agent guessing between three contacts named Zhang.
            # It asked, exactly as the grader requires, and the task recorded
            # no assistant output whatsoever.
            question = state["pending_question"]
            choices = state.get("pending_choices") or []
            candidate = question if not choices else (
                f"{question}\n\n" + "\n".join(f"- {c}" for c in choices)
            )
        if candidate:
            state["final_output"] = candidate
    return state


def _paused(final: dict | None) -> bool:
    """Whether `app.invoke` came back on an interrupt rather than a finish.

    It does not raise and it does not say so anywhere a caller looks -- the
    only signals are the `__interrupt__` key it splices in and the
    `pending_question` the paused node never got to clear.
    """
    return bool(final) and ("__interrupt__" in final or final.get("pending_question"))


def _score(run_span, final: AgentState) -> None:
    """The facts worth filtering a Langfuse trace by, scored on it.

    `model_calls` replaces the old `dispatch_rounds`: with one agent loop there
    are no overseer rounds left to count, and model requests are the number that
    maps to both latency and spend -- measured on Claw-Eval, tool execution was
    0.1 to 0.4 seconds of runs lasting 119 to 946, so everything else was this.

    `mode_swaps` is here so that "did the model park on one model, or thrash
    between them?" is a query across a batch rather than an opinion. Nothing in
    the loop can settle that with a rule; only the numbers can.
    """
    run_span.update(output=final.get("final_output"))
    run_span.score_trace(
        name="model_calls", value=final.get("model_calls") or 0, data_type="NUMERIC",
    )
    swaps = [line for line in (final.get("mode_log") or []) if "->" in line]
    run_span.score_trace(name="mode_swaps", value=len(swaps), data_type="NUMERIC")
    if final.get("mode"):
        run_span.score_trace(name="final_mode", value=final["mode"], data_type="CATEGORICAL")
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


#: What app.stream() is asked for. "updates" is what it always yielded -- one
#: dict per node RETURN. That was enough when a node returned every few seconds;
#: with one long-running agent loop it means nothing reaches the screen until
#: the whole loop finishes, so `otto chat` would sit silent for minutes and then
#: print one panel. "custom" is the loop's own per-iteration events
#: (langgraph.config.get_stream_writer), emitted in the same node-shaped form so
#: agent/cli/chat.py and agent/cli/tui.py need no changes at all.
_STREAM_MODES = ["updates", "custom"]


def _stream_events(app_stream, graph_thread_id: str):
    """Unwrap app.stream()'s `(mode, payload)` tuples into the flat updates
    callers have always seen, turning an interrupt into an `__ask__` event.

    Asking for more than one stream mode changes the yield shape from a bare
    payload to a tuple, so this is the one place that knows about it. Custom
    payloads pass straight through: the loop already emits them node-shaped.
    Yields `(event, is_ask)`.
    """
    for mode, payload in app_stream:
        if mode == "custom":
            yield payload, False
            continue
        ask = _as_ask_event(payload, graph_thread_id)
        yield (ask, True) if ask is not None else (payload, False)


def _workspace_binding(workspace: str | None):
    """`bind_workspace(workspace)`, or nothing at all when the caller named none.

    Deliberately NOT `bind_workspace(None)`. That is an explicit "this run has
    no workspace", and it sets the contextvar, which CLOBBERS an outer binding
    -- exactly what agent/eval/'s harnesses depend on, each wrapping its own
    `with bind_workspace(scratch)` around the call below and passing no
    `workspace=` argument at all. Adding the parameter without this would have
    silently taken file access away from both benchmark harnesses while every
    test they have kept passing, because none of them asserts on the binding.
    A test does now (tests/test_workspace_session.py), and it caught this.
    """
    return bind_workspace(workspace) if workspace is not None else nullcontext()


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
    text: str, *, session_id: str, history: Sequence[BaseMessage] = (),
    memory_context: str = "", workspace: str | None = None,
    usage: UsageLedger | None = None,
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
    # A run with nobody watching still needs a ceiling. Both benchmark harnesses
    # bind their own Budget; an `otto chat` turn bound NOTHING, so
    # OTTO_MAX_MODEL_CALLS was a dead env var and an interactive turn could
    # spend without limit. `current_budget() or default_budget()` keeps a
    # harness's own budget when there is one.
    budget = current_budget() or default_budget()
    ledger = usage or current_usage() or UsageLedger()

    with bind_budget(budget), bind_usage(ledger), bind_store(store), _workspace_binding(workspace):
        with propagate_attributes(
            trace_name="otto:pipeline",
            session_id=session_id,
            tags=["pipeline"],
        ):
            with client.start_as_current_observation(
                name="otto:pipeline", as_type="agent", input=text
            ) as run_span:
                try:
                    final = _salvage(app.invoke(initial, config))
                except Exception as exc:
                    # One salvage point, deliberately, rather than a catch
                    # inside the loop: a real bug should still surface loudly
                    # here and in the logs, but it must not cost the caller the
                    # work that was already done. See _salvage.
                    logger.exception("pipeline run failed; salvaging what it reached")
                    final = _salvage(app.get_state(config).values, exc)
                if _paused(final):
                    # run_pipeline has no resume path -- resume is the streaming
                    # API. A caller here would otherwise record an empty answer
                    # and never learn that a question was asked.
                    logger.warning(
                        "run_pipeline: the graph paused to ask %r and cannot be "
                        "resumed from here; answering with what it had",
                        final.get("pending_question"),
                    )
                _score(run_span, final)

    return final


def run_pipeline_stream(
    text: str, *, session_id: str, history: Sequence[BaseMessage] = (),
    memory_context: str = "", workspace: str | None = None,
    usage: UsageLedger | None = None,
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
    # A run with nobody watching still needs a ceiling. Both benchmark harnesses
    # bind their own Budget; an `otto chat` turn bound NOTHING, so
    # OTTO_MAX_MODEL_CALLS was a dead env var and an interactive turn could
    # spend without limit. `current_budget() or default_budget()` keeps a
    # harness's own budget when there is one.
    budget = current_budget() or default_budget()

    # The caller's ledger when it has one (the TUI keeps one per SESSION,
    # so its panel is cumulative across turns and resumes), otherwise a
    # throwaway -- agent/pipeline/usage.py.
    ledger = usage or current_usage() or UsageLedger()
    with bind_budget(budget), bind_usage(ledger), bind_store(store), _workspace_binding(workspace):
        with propagate_attributes(
            trace_name="otto:pipeline",
            session_id=session_id,
            tags=["pipeline"],
        ):
            with client.start_as_current_observation(
                name="otto:pipeline", as_type="agent", input=text
            ) as run_span:
                stream = app.stream(initial, config, stream_mode=_STREAM_MODES)
                for event, is_ask in _stream_events(stream, graph_thread_id):
                    if is_ask:
                        run_span.update(output="(paused -- awaiting your answer)")
                        yield event
                        return
                    yield event
                final = app.get_state(config).values
                _score(run_span, final)
                trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}


def resume_pipeline_stream(
    answer, *, thread_id: str, session_id: str, workspace: str | None = None,
    usage: UsageLedger | None = None,
):
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
    # A run with nobody watching still needs a ceiling. Both benchmark harnesses
    # bind their own Budget; an `otto chat` turn bound NOTHING, so
    # OTTO_MAX_MODEL_CALLS was a dead env var and an interactive turn could
    # spend without limit. `current_budget() or default_budget()` keeps a
    # harness's own budget when there is one.
    budget = current_budget() or default_budget()

    # The caller's ledger when it has one (the TUI keeps one per SESSION,
    # so its panel is cumulative across turns and resumes), otherwise a
    # throwaway -- agent/pipeline/usage.py.
    ledger = usage or current_usage() or UsageLedger()
    with bind_budget(budget), bind_usage(ledger), bind_store(store), _workspace_binding(workspace):
        with propagate_attributes(
            trace_name="otto:pipeline",
            session_id=session_id,
            tags=["pipeline", "resumed"],
        ):
            with client.start_as_current_observation(
                name="otto:pipeline:resume", as_type="agent", input=str(answer)
            ) as run_span:
                stream = app.stream(Command(resume=answer), config, stream_mode=_STREAM_MODES)
                for event, is_ask in _stream_events(stream, thread_id):
                    if is_ask:
                        run_span.update(output="(paused -- awaiting your answer)")
                        yield event
                        return
                    yield event
                final = app.get_state(config).values
                _score(run_span, final)
                trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}
