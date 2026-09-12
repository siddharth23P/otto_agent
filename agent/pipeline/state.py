"""What one run carries between the two nodes.

`AgentState` is a TypedDict, not the conversation: the conversation lives in
`transcript` as plain dicts, because LangGraph serialises state on every
super-step and Langfuse serialises it into every span, so storing message
OBJECTS would put the whole exchange through both on each pass.

The fields that decide behaviour rather than merely record it:

  checklist   what a correct answer must contain, written from the TASK before
              any attempt exists and never rewritten. The evaluator scores
              against it and the loop is shown it. A run holds one.
  mode        which model and guidance the loop is running on now; `mode_log`
              is every switch, which is also how `_record_seat` knows whether
              a run was single-mode enough to credit anything.
  model_calls what the run has spent, counted in REQUESTS.
  actions     one line per tool call, which is what survives compaction and
              what both the loop and the judge read instead of the transcript.

An earlier design had the overseer build a JSON `plan` of `PlanStep` dicts and
walk it one step at a time, with `active_step` and `round` tracking where it
was. That graph is gone -- planning is a MODE the one agent switches into --
and those fields went with it rather than lingering as state nothing reads.
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
    #: The agent loop's OWN conversation -- system prompt, every model reply,
    #: every tool result -- carried across node returns instead of discarded.
    #:
    #: This is the field the loop rewrite exists for. Before it, every node
    #: rebuilt `[SystemMessage, HumanMessage]` from scratch and the tool
    #: conversation died with the node, so a re-invoked role paid to re-derive
    #: what the last one had just learned. `actions` above was the patch for
    #: that: a lossy one line per call, which is what you can afford to carry
    #: when the real thing is gone. Both survive, and they are complementary --
    #: this one is verbatim and recent, `actions` is lossy and whole-run.
    #:
    #: NO REDUCER, deliberately. `board` and `actions` accumulate because many
    #: nodes append to them; a transcript is last-write-wins, like `context`
    #: and `output`, because the loop owns the whole list and hands back the
    #: version it finished with.
    #:
    #: Two shapes that must never enter it, both silent failures rather than
    #: loud ones. A SystemMessage anywhere after the opening run: langchain's
    #: Anthropic adapter raises on non-consecutive system messages and its
    #: Gemini adapter hoists a mid-list one out of position or drops it. And an
    #: empty AIMessage: `_call` returns "" on an empty stream, Anthropic
    #: rejects empty text blocks, and in a list that never resets one of those
    #: breaks every later Anthropic call for the rest of the run.
    transcript: list[AnyMessage] | None
    #: Which mode the loop is working in -- a key of agent/pipeline/modes.py's
    #: MODES, naming both the model answering and the guidance it is under.
    #: Survives an evaluator rejection, so a run that switched to `plan` and
    #: got rejected resumes planning rather than silently reverting.
    mode: str | None
    #: One line per accepted or refused mode swap, for the board and for
    #: telemetry -- "call 7: solve -> plan (needs ordering first)". Accumulating
    #: like `board`, because it is a narration of what happened rather than a
    #: current value. Read by `_score` so "did the model park on one model?" is
    #: a number rather than an argument.
    mode_log: Annotated[list[str], operator.add]
    #: How many model requests this run has spent. Replaces `round` as the
    #: number worth tracing: with one loop there are no overseer rounds to
    #: count, and calls are the thing that maps to both latency and spend.
    model_calls: int
    #: How many times the evaluator has rejected an answer in this run. Bounded
    #: so judgment cannot eat the whole budget -- see nodes.py's MAX_REJECTIONS.
    rejections: int
    #: How many times the judge failed with a PROVIDER error, as opposed to
    #: returning a verdict. Counted apart from `rejections` because it is a
    #: different fact: a rejection is a verdict and retrying is the point, a
    #: provider failure is no verdict at all. Its cap (nodes.py's
    #: MAX_JUDGE_ERRORS) is what stops the agent and the evaluator handing a
    #: dead provider back and forth until the graph's recursion limit ends
    #: the run with no answer.
    judge_errors: int
    #: What this run has to be true at the end, as records rather than prose.
    #: `[{"text": str, "status": "pending"|"met"|"blocked", "evidence": str}]`.
    #:
    #: This is the run's working STATE, and it is deliberately not the
    #: conversation. The measured case: in one ablation a verified working
    #: state was worth +24 points where an experience library over the same
    #: task was worth +2, and injecting more library text with no state signal
    #: scored 16 points BELOW state alone. What matters is knowing what is
    #: still open, not having more to read.
    #:
    #: Written once at the start of a run, from the task alone, before any
    #: attempt exists -- so nothing here can be shaped by an attempt trying to
    #: satisfy it. The loop reads it to know what is left; the evaluator judges
    #: against it and is the only thing that may change a status, because an
    #: executor's claim about its own work is not evidence.
    checklist: list[dict] | None
    final_output: str | None
