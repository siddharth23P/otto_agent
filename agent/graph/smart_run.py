from agent.graph.code_run import run_code, run_code_stream
from agent.graph.code_state import CodeTask
from agent.graph.size import size


def run_smart(text: str, *, thread_id: str) -> CodeTask:
    agents = size(text)
    return run_code(text, thread_id=thread_id, agents=agents)


def run_smart_stream(text: str, *, thread_id: str):
    """run_code_stream(), sized first. Yields `{"__sizing__": agents}` before
    anything else, so a caller building a per-agent display (9.8's animation)
    knows the count before the graph's first real update arrives."""
    agents = size(text)
    yield {"__sizing__": agents}
    yield from run_code_stream(text, thread_id=thread_id, agents=agents)
