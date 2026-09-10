"""State schema for the overseer/planner/solver/summarizer/finder/evaluator
graph (agent/pipeline/nodes.py).

Second revision of this graph (2026-09-10, same day it replaced the
orchestrator/worker/evaluate/subtask_consensus/synthesize swarm pipeline):
ROUTER stopped being a one-shot dispatcher and became the overseer -- it is
re-invoked after EVERY node (not just after an evaluator rejection) and
decides the single next action from everything accumulated so far. Two
fields exist because of that shift that didn't before: `context` (material
finder/summarizer hand forward) and `plan` (a plan once the evaluator has
actually approved it, as distinct from a plan still pending judgment,
which lives in `output` like any other unjudged specialist attempt).

`round` is telemetry only now, not a budget -- there is deliberately no
constant anywhere in this graph that caps how many times the overseer may
retry a task (2026-09-10 design call: "we dont need any variable to limit
number of rounds"). The only thing that can still end a run early is
LangGraph's own recursion_limit (nodes.py's `_RECURSION_SAFETY_NET`), and
that exists to catch a genuinely runaway loop (a bug), never to be the
reason a real, converging request stops.
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    board: Annotated[list[str], operator.add]
    #: How many times the overseer has been invoked so far (starts at 0,
    #: incremented by router() on every call). Telemetry/tracing only --
    #: nothing in this graph reads it to force a stop; see the module
    #: docstring.
    round: int
    #: Whichever node MOST RECENTLY produced `output` -- one of ROLE_NODES
    #: (nodes.py), or None before anything has run. Set by that node itself
    #: (self-reported), not predicted by router() ahead of time. `_run_role`
    #: uses `node == <this role>` to tell "I am being re-dispatched, and
    #: whatever's in `output` is MY previous attempt" apart from "a
    #: different specialist ran last, unrelated to what I'm about to do."
    node: str | None
    #: The evaluator's most recent rejection reason -- cleared back to ""
    #: on approval, overwritten on the next rejection. Read by both router()
    #: (deciding what to do about it) and whichever role node runs next
    #: (revising, or just noting it as background if it was about a
    #: DIFFERENT specialist's attempt -- see _run_role).
    feedback: str
    #: The most recently produced specialist output, pending judgment --
    #: a plan (from planner) or a candidate final answer (from solver,
    #: or occasionally summarizer/finder if their own output already
    #: answers the request). Not yet trusted either way; `final_output`
    #: is the only field a caller should treat as the finished result.
    output: str | None
    #: Material finder/summarizer hand forward for planner/solver to use --
    #: finder APPENDS what it gathered, summarizer REPLACES it with a
    #: condensed version (agent/pipeline/nodes.py's `_run_role`,
    #: `context_op`). Empty string until either has run.
    context: str
    #: The current plan, once the evaluator has actually approved one --
    #: distinct from a plan still pending judgment (which lives in
    #: `output` like anything else awaiting evaluation). None until a plan
    #: is approved; a task the overseer judges as not needing one just
    #: never sets this and goes straight to solver.
    plan: str | None
    final_output: str | None
