import logging

from langchain_core.messages import HumanMessage

from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler

from agent.graph.nodes import app, ROUTER, MAX_ROUNDS
from agent.graph.state import ALLOWED_AGENTS, Hive

logger = logging.getLogger(__name__)


def run(text: str, *, thread_id: str, agents: int, secondary_seats: int = 0) -> Hive:
    if agents not in ALLOWED_AGENTS:
        raise ValueError(f"{agents} agents are not allowed, choose from {ALLOWED_AGENTS}")
    if not 0 <= secondary_seats <= agents:
        raise ValueError(f"Invalid number of secondary seats {secondary_seats}")
    if secondary_seats > 0 and ROUTER.secondary is None:
        raise ValueError(f"No secondary model available add one before retrying!")

    failures = ROUTER.prewarm()
    if failures:
        logger.warning("prewarm: %s", failures)

    initial = {
        "messages": [HumanMessage(text)],
        "board": [],
        "votes": [],
        "agents": agents,
        "secondary_seats": secondary_seats,
        "round": 0,
        "decision": None,
    }

    client = get_client()
    handler = CallbackHandler()

    config = {
        "recursion_limit": 3 * MAX_ROUNDS + 2,
        "configurable": {"thread_id": thread_id},
        "callbacks": [handler],
    }

    with propagate_attributes(
        trace_name="otto:hive",
        session_id=thread_id,
        tags=["hive", f"agents:{agents}", f"seats:{secondary_seats}"],
    ):
        with client.start_as_current_observation(
            name="otto:hive", as_type="agent", input=text
        ) as run_span:
            final = app.invoke(initial, config)

            last_round = final["round"]
            agreed = sum(
                1
                for v in final["votes"]
                if v["round"] == last_round and v["answer"] == final["decision"]
            )

            run_span.update(output=final["decision"])
            run_span.score_trace(name="agents", value=agents, data_type="NUMERIC")
            run_span.score_trace(name="secondary_seats", value=secondary_seats, data_type="NUMERIC")
            run_span.score_trace(name="rounds", value=last_round, data_type="NUMERIC")
            run_span.score_trace(name="agreement", value=agreed / agents, data_type="NUMERIC")
            run_span.score_trace(
                name="abstained", value=int(final["decision"] is None), data_type="BOOLEAN"
            )

    return final