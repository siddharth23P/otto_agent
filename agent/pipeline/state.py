"""State schema for the overseer/planner/solver/summarizer/finder/evaluator
graph (agent/pipeline/nodes.py).

Second revision of this graph (2026-09-10, same day it replaced the
orchestrator/worker/evaluate/subtask_consensus/synthesize swarm pipeline):
ROUTER stopped being a one-shot dispatcher and became the overseer -- it is
re-invoked after EVERY node (not just after an evaluator rejection) and
decides the single next action from everything accumulated so far. Two
fields exist because of that shift that didn't before: `context` (material
finder/summarizer hand forward) and `plan`.

`round` is telemetry only now, not a budget -- there is deliberately no
constant anywhere in this graph that caps how many times the overseer may
retry a task (2026-09-10 design call: "we dont need any variable to limit
number of rounds"). The only thing that can still end a run early is
LangGraph's own recursion_limit (nodes.py's `_RECURSION_SAFETY_NET`), and
that exists to catch a genuinely runaway loop (a bug), never to be the
reason a real, converging request stops.

Third refinement, same day: `plan` stopped being free text. It is a JSON-
shaped list of step dicts once the evaluator approves it --
`{"task": str, "route_to": str | None, "output": str | None}` each -- with
`task` filled in by the planner and `route_to` filled in by the overseer,
one step at a time, as it assigns each step to whichever of
solver/summarizer/finder should execute it (agent/pipeline/nodes.py's
STEP_TARGETS/STEP_ROUTE_PROMPT). `active_step` exists because of that: the
index into `plan` of whichever step is currently in flight, so a role node
revising a rejected final-answer judgment (which, once a plan is active, is
really just the LAST step's output) knows which step's `output` to
overwrite rather than only ever appending a fresh, disconnected context
entry alongside a now-stale one.
"""
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class PlanStep(TypedDict):
    #: What this step needs to accomplish -- written by the planner, never
    #: rewritten afterward (a step that turns out to be wrong is corrected
    #: by re-planning from scratch, not edited in place; see router()'s
    #: "escalating back to planner resets the whole plan" behavior).
    task: str
    #: Which specialist executes this step -- one of nodes.py's
    #: STEP_TARGETS ("solver", "summarizer", "finder"; never "planner" or
    #: "evaluator" -- no re-planning or per-step judgment mid-plan in this
    #: design). None until the overseer assigns it, right before dispatch.
    route_to: str | None
    #: That specialist's result for this step. None means "not run yet" --
    #: the sentinel _next_pending_step_index (nodes.py) scans for; an empty
    #: string is a legitimate (if useless) completed result, not the same
    #: as unset.
    output: str | None


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
    #: a plan (from planner, still raw/unparsed JSON text) or a candidate
    #: final answer (from solver, or occasionally summarizer/finder if
    #: their own output already answers the request, or from whichever
    #: specialist ran the LAST plan step once a plan is active). Not yet
    #: trusted either way; `final_output` is the only field a caller should
    #: treat as the finished result.
    output: str | None
    #: Material finder/summarizer hand forward for planner/solver to use --
    #: finder APPENDS what it gathered, summarizer REPLACES it with a
    #: condensed version (agent/pipeline/nodes.py's `_run_role`,
    #: `context_op`); once a plan is active, EVERY executed step also
    #: appends a labeled "step N (role): task -> result" entry here
    #: (overriding that role's own default context_op for the duration of
    #: plan execution), so later steps can see earlier steps' results.
    #: Empty string until any of the above has run.
    context: str
    #: The approved plan, as a list of PlanStep dicts -- None until the
    #: evaluator approves one (see evaluator()'s plan-judging mode, which
    #: parses the planner's raw JSON-ish output into this shape). A task
    #: the overseer judges as not needing one just never sets this and
    #: goes straight to solver.
    plan: list[PlanStep] | None
    #: Index into `plan` of whichever step is currently in flight -- None
    #: when no plan is active, or once every step has run (there's no
    #: "current" step left; the overseer moves on to evaluator). Set by
    #: router() right before dispatching a step's route_to; read by
    #: _run_role to know which PlanStep's `output` to fill in (and, on a
    #: revise, which one to overwrite rather than append a new one).
    active_step: int | None
    final_output: str | None
