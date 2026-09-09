import logging

from langchain_core.messages import HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agent.graph.code_nodes import app, ROUTER, MAX_ROUNDS
from agent.graph.code_state import CodeTask
from agent.graph.state import ALLOWED_AGENTS

logger = logging.getLogger(__name__)


def _validate(agents: int) -> None:
    if agents not in ALLOWED_AGENTS:
        raise ValueError(f"{agents} agents are not allowed, must be one of {ALLOWED_AGENTS}")


def _initial(text: str, agents: int) -> dict:
    return {
        "messages": [HumanMessage(text)],
        "board": [],
        "agents": agents,
        "decomp_round": 0,
        "proposals": [],
        "proposal_votes": [],
        "part_specs": None,
        "part_round": {},
        "part_status": {},
        "submissions": [],
        "reviews": [],
        "final_code": None,
    }


def _config(thread_id: str, agents: int, handler) -> dict:
    return {
        "recursion_limit": agents * MAX_ROUNDS * 4,
        "configurable": {"thread_id": thread_id},
        "callbacks": [handler],
    }


def _score(run_span, final: CodeTask, agents: int) -> None:
    accepted = sum(1 for s in final["part_status"].values() if s == "accepted")
    exhausted = sum(1 for s in final["part_status"].values() if s == "exhausted")
    run_span.update(output=final["final_code"])
    run_span.score_trace(name="agents", value=agents, data_type="NUMERIC")
    run_span.score_trace(name="decomp_rounds", value=final["decomp_round"], data_type="NUMERIC")
    run_span.score_trace(name="parts_accepted", value=accepted, data_type="NUMERIC")
    run_span.score_trace(name="parts_exhausted", value=exhausted, data_type="NUMERIC")


def run_code(text: str, *, thread_id: str, agents: int) -> CodeTask:
    """Validate, prewarm, invoke, score, return the finished CodeTask.

    Stays exactly this shape for scripted/benchmark callers (12.8, run_smart())
    that want one call in, one finished result out. The interactive shell
    (9.7/9.8) wants to watch it happen instead -- that is run_code_stream(),
    below, not a different mode of this function.
    """
    _validate(agents)

    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, agents)
    client = get_client()
    handler = CallbackHandler()
    config = _config(thread_id, agents, handler)

    with propagate_attributes(
        trace_name="otto:code",
        session_id=thread_id,
        tags=["code", f"agents:{agents}"],
    ):
        with client.start_as_current_observation(
            name="otto:code", as_type="agent", input=text
        ) as run_span:
            final = app.invoke(initial, config)
            _score(run_span, final, agents)

    return final


def run_code_stream(text: str, *, thread_id: str, agents: int):
    """Same validation, tracing and scoring as run_code(), but yields each
    graph update as it happens (`stream_mode="updates"`) instead of invoking
    and returning. What the interactive shell watches live.

    The last item yielded is always `{"__final__": CodeTask, "__trace_id__":
    str | None}` -- a plain dict, not a Command/node delta, so a caller doing
    `node, delta = next(iter(update.items()))` on every *other* item never has
    to special-case it inline; it only has to check for the key once.
    """
    _validate(agents)

    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = _initial(text, agents)
    client = get_client()
    handler = CallbackHandler()
    config = _config(thread_id, agents, handler)

    with propagate_attributes(
        trace_name="otto:code",
        session_id=thread_id,
        tags=["code", f"agents:{agents}"],
    ):
        with client.start_as_current_observation(
            name="otto:code", as_type="agent", input=text
        ) as run_span:
            for update in app.stream(initial, config, stream_mode="updates"):
                yield update
            final = app.get_state(config).values
            _score(run_span, final, agents)
            trace_id = run_span.trace_id

    yield {"__final__": final, "__trace_id__": trace_id}
