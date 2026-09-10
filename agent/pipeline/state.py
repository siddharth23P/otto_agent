"""State schema for the router/planner/solver/summarizer/finder/evaluator
graph that replaced the orchestrator/worker/evaluate/subtask_consensus/
synthesize swarm pipeline (nodes.py, retired 2026-09-10).

That swarm was built around parallel independent subtasks (N workers, N
evaluators, a Judge rule, a synthesizer to reconcile them) -- this graph is
a single sequential loop instead: one ROUTER decides which ONE specialist
(planner/solver/summarizer/finder) should attempt the whole request, that
specialist may use tools before answering, and one EVALUATOR judges the
result -- approve and stop, or reject and let the router decide who tries
next (possibly the same specialist with feedback, possibly a different one
if the feedback suggests this was the wrong kind of task for whoever tried
it). No fan-out, no consensus vote, no synthesis step: there is only ever
one candidate answer in flight at a time, so nothing needs reconciling.

This is a deliberate architecture swap, not a tune -- see nodes.py's module
docstring for the design discussion. `agents` (a swarm size) has no
equivalent here: there is nothing to size.
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    board: Annotated[list[str], operator.add]
    #: How many router-dispatch rounds have happened so far (starts at 0,
    #: incremented by router() before each dispatch). Bounds retries via
    #: MAX_DISPATCH_ROUNDS (nodes.py) the same way the swarm's per-subtask
    #: round counter did.
    round: int
    #: Which specialist the router most recently dispatched to -- one of
    #: ROLE_NODES (nodes.py), or None before the first dispatch. This is
    #: also, once a role node has run, "who produced the pending `output`".
    node: str | None
    #: The evaluator's rejection reason for the most recent attempt, fed to
    #: the next round (router's re-decision AND whichever role node is
    #: dispatched next). Empty string on round 1 -- nothing to revise yet.
    feedback: str
    #: The most recently dispatched role node's candidate answer, pending
    #: evaluation. Not yet trusted -- final_output is the only field a
    #: caller should treat as the finished result.
    output: str | None
    final_output: str | None
