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

Fifth refinement, same day (2026-09-10 design call: "if evaluator fails
router should again go to planner for planning next steps based on
current output"): a live run crashed the whole graph on an
httpx.ReadTimeout raised mid-stream by the Inception provider, uncaught,
all the way up through nodes.py's `_call`. `node_error` exists to turn
that from a crash into a normal graph edge -- ANY node's own LLM call
(a role node's, or the evaluator's) can raise a ProviderError (nodes.py's
`_run_role`/`evaluator` now catch it instead of letting it propagate),
and when that happens the node returns to router() with `node_error` set
instead of a real `output`/verdict. router() checks `node_error` FIRST,
ahead of everything else, and deterministically (no LLM call -- the LLM
call is exactly what just failed) escalates to planner, discarding any
active plan the same way an ordinary rejection-driven escalation already
does. The failure text also gets written into `feedback` so planner sees
it via the existing "PREVIOUS ATTEMPT BY <role>" background display
(_role_body) alongside whatever `output` already existed -- the "based on
current output" part of the request -- without needing a second display
mechanism. See nodes.py's module docstring for the full reasoning.

Seventh refinement, same day (nodes.py's module docstring has the full
design discussion): `pending_question`/`pending_choices`/`asking_role`
exist so a role node or the evaluator can pause the whole run -- via
LangGraph's own interrupt()/Command(resume=...) -- and ask the person
something it's genuinely stuck without, instead of guessing or looping.
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
    #: Set (to a short description of what failed and where) when a node's
    #: own LLM call raised a ProviderError instead of producing a real
    #: output/verdict -- a provider/network failure (a timeout, an outage),
    #: NOT an evaluator rejection. router() checks this first, ahead of
    #: everything else, and deterministically escalates to planner; the
    #: node that sets it also clears back to "" whatever it would normally
    #: have overwritten (`output` stays whatever it already was), and
    #: writes a human-readable version into `feedback` so planner sees the
    #: failure the same way it would see any other rejected attempt (see
    #: nodes.py's module docstring, fifth refinement). Cleared back to None
    #: by router() once it has escalated.
    node_error: str | None
    #: Set together, by whichever of planner/solver/summarizer/finder/
    #: evaluator's own _tool_loop call raised NeedsUserInput (nodes.py) --
    #: it got stuck on something only the person can supply. `asking_role`
    #: is who to hand back to (ask_user() has no other way to know, since
    #: it's a dedicated node, not part of that role's own function).
    #: `pending_choices` is the "multi choice" half of the "multi choice +
    #: text bar" UI (agent/cli/chat.py, agent/cli/tui.py) -- an empty list
    #: means an open-ended, free-text-only question, not "no question."
    #: All three are cleared back to None the moment ask_user() resumes
    #: (nodes.py, seventh refinement) -- nothing about this is meant to
    #: outlive that one pause.
    pending_question: str | None
    pending_choices: list[str] | None
    asking_role: str | None
    #: One compact line per tool call any role has made, across the WHOLE
    #: run -- "solver: write_file main.c.rs -> ok (56 lines)",
    #: "solver: execute_bash rustc main.c.rs -> exit 1: error[E0433]...".
    #:
    #: A role's tool conversation lives in a local list inside _tool_loop and
    #: is thrown away when that node returns, so a re-invoked role used to
    #: start blind: it could see the task, the plan, the gathered context and
    #: a rejected attempt, but nothing about what it had actually DONE.
    #: Measured on a hard task: the solver wrote the same file 166 times over
    #: five overseer rounds, a different draft each time, never compiling any
    #: of them -- each round genuinely did not know the work had been tried.
    #: This is what it reads instead (nodes.py's _role_body).
    #:
    #: Accumulating (operator.add) like `board`, and deliberately separate
    #: from it: `board` is a human-readable narration of which node ran,
    #: this is the agent's own record of what it did and what came back.
    actions: Annotated[list[str], operator.add]
    final_output: str | None
