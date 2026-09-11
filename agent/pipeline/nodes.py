"""The overseer/planner/solver/summarizer/finder/evaluator graph.

Second revision of this graph (2026-09-10, same day the first revision
replaced the orchestrator/worker/evaluate/subtask_consensus/synthesize swarm
pipeline). The first revision's ROUTER was a one-shot dispatcher: decide once,
run one specialist, judge once, re-decide only on rejection. This revision
changes what ROUTER *is*, per that day's second design discussion:

  * ROUTER is the OVERSEER now, not a one-shot dispatcher. It is re-invoked
    after EVERY node -- not just after an evaluator rejection -- and decides
    the single next action from everything accumulated so far: gathered
    context, an approved plan (if any), the most recent pending output, and
    the most recent rejection feedback (if any). Five targets, not four:
    planner, solver, summarizer, finder, or evaluator. Sending something to
    evaluator is now itself a decision the overseer makes explicitly, not
    something that happens automatically after every specialist run -- a
    finder call that only gathered background material, say, should usually
    go straight to whichever specialist needs it next, not to the evaluator.

  * No cap on how many times the overseer may retry a task (design call:
    "we dont need any variable to limit number of rounds a agent runs for").
    `round` (AgentState) is telemetry now, incremented on every overseer
    invocation, read by nothing that decides to stop. The only backstop left
    is LangGraph's own recursion_limit, sized generously by
    _RECURSION_SAFETY_NET below and by agent/pipeline/run.py's `_config` --
    infra insurance against a genuinely runaway loop (a bug), never a
    business rule a real, converging request is expected to hit.

  * Retry-same-specialist is the DEFAULT policy on a rejection, not a
    hardcoded rule (design call: "we dont have overlap between specialist so
    retry with same specialist with feedback rather than different"). It
    lives entirely in ROUTER_PROMPT's wording below, with one deliberate,
    explicit exception: if the feedback shows a task was attempted without a
    plan and that was the actual problem, the overseer is told to dispatch
    to planner instead, even though a different specialist tried it --
    "learn from its mistake for which task required planning and which did
    not," scoped to THIS run only (see the note at the bottom of this
    docstring on the larger, cross-turn version of that same idea).

  * planner / solver / summarizer / finder -- same ACTION/tool-then-FINAL
    loop as before (_tool_loop), same six tools. What changed is what they
    hand back: every one of them now returns to "router" (not "evaluator")
    when done, and self-reports its own identity into `state["node"]` --
    the overseer no longer predicts ahead of time who is about to run, it
    only decides who runs next. See agent/pipeline/state.py for the full
    field-by-field reasoning, and the third-refinement note below for how
    `context` gets written during plan execution specifically.

  * evaluator is dual-mode now, not single-mode: it judges a PLAN
    (`state["node"] == "planner"`) or a candidate FINAL ANSWER (anything
    else) -- different question, different prompt. Approving a plan sets
    `state["plan"]` and returns to the overseer (there is more work left --
    the plan hasn't been executed yet); approving a final answer sets
    `state["final_output"]` and ends the run. Rejecting either always
    returns to the overseer with the reason -- there is no longer an
    exhaustion branch that gives up after N rounds (removed along with
    MAX_DISPATCH_ROUNDS, see above).

Third refinement, same day: a plan stopped being free text the moment it's
approved. PLANNER_PROMPT now asks for a JSON array of `{"task": ...}`
objects; evaluator(), on approving one, parses it (_parse_plan_steps) into
a list of PlanStep dicts (agent/pipeline/state.py) --
`{"task", "route_to": None, "output": None}` each -- and CLEARS
`state["active_step"]` back to None. From there, executing the plan is
deterministic/assignment-driven, not re-litigated through the general
overseer prompt on every round:

  * router() first checks for an ACTIVE, feedback-free plan
    (`state["plan"]` is a non-empty list and `state["feedback"]` is empty).
    If so, it finds the first step whose `output` is still None
    (_next_pending_step_index) and:
      - if there isn't one (every step has run), it dispatches straight to
        evaluator for the whole-answer judgment -- no LLM call needed, the
        decision is unambiguous once the plan says "done".
      - if there is one, it makes a NARROWER LLM call (STEP_ROUTE_PROMPT,
        restricted to STEP_TARGETS = solver/summarizer/finder -- no
        re-planning or per-step judgment mid-plan in this design), writes
        the chosen specialist into that step's own `route_to`, and
        dispatches there.
    The general ROUTER_PROMPT (all five targets) is only consulted again
    either before any plan exists (the very first decision on a request)
    or after a REJECTION (deciding how to retry) -- both cases where real
    judgment, not just plan bookkeeping, is needed. A rejection while a
    plan is active is really a rejection of the LAST step's output (see
    below), so the general prompt's existing "retry same specialist by
    default, escalate to planner if it needed one" policy still applies
    unchanged; if it picks "planner" while an old plan is still hanging
    around, router() resets `plan`/`active_step` to None first -- the old
    plan turned out to be the problem, not just one step of it, so this
    starts over rather than leaving stale step state behind.

  * `active_step` (state.py) tracks which step is currently in flight.
    _run_role checks it: when set, this call is executing (or REVISING,
    after a rejection) that specific step, so its output overwrites that
    step's `output` in `state["plan"]` -- not just the flat `state["output"]`
    used outside plan execution -- and a labeled "step N (role): task ->
    result" entry is always appended onto `context` (overriding that role's
    own normal context_op, e.g. summarizer's usual "replace", for the
    duration of plan execution -- erasing earlier steps' results because
    the current step happens to be a summarizer step would defeat the whole
    point of sequencing). This is the concrete mechanism behind point 4's
    "big task -> plan -> later steps build on earlier ones' results."

Fourth refinement, same day: both of the overseer's LLM calls (the general
5-way decision and the narrower per-step assignment) now retry IN PLACE
(_decide, _MAX_ROUTER_PARSE_RETRIES) if the reply doesn't parse at all,
before falling back to _parse_router's "default to solver" safety net.
Observed live: the model occasionally rambles instead of the required
two-line NODE:/WHY: format -- most often the very first time a PENDING
OUTPUT shows up in its prompt (i.e. right when it should say "evaluator"
for the first time) -- rather than genuinely being unsure. Silently
defaulting to solver in that case doesn't just misroute once: it burns an
entire extra specialist round (a full ACTION/FINAL tool loop) to recover,
every time it happens. A retry of just the one small router call is far
cheaper and, empirically, usually succeeds on the first retry.

Fifth refinement, same day (design call, verbatim: "if evaluator fails
router should again go to planner for planning next steps based on
current output"): a live run crashed with an uncaught httpx.ReadTimeout,
raised mid-stream deep inside the Inception provider (solver's own call,
that time -- not evaluator's), propagating all the way up through this
file's `_call` and out of `app.invoke()` entirely. Two things changed:

  * agent/router/llm_provider/inception_provider.py's `_stream` only
    translated exceptions raised by the initial `client.chat.completions.
    create(...)` call into ProviderError -- NOT exceptions raised while
    iterating the returned stream itself (`for chunk in stream:`), which
    is exactly where a read timeout happens (the request already
    succeeded; the response body is still arriving). That gap is now
    closed: the iteration is wrapped the same way the initial call is,
    and raw httpx errors (not just the SDK's own InceptionError subtree)
    are translated too, since a read timeout mid-stream surfaces as a raw
    httpx.ReadTimeout, never one of the SDK's own wrapped types. The
    invariant this restores (base.py's own docstring already states it):
    nothing but ProviderError should ever cross the provider boundary.

  * With that invariant actually holding, `_run_role` and `evaluator`
    (below) now catch ProviderError around their own `_tool_loop` call
    instead of letting it crash the graph. Rather than a real output or
    verdict, they return to router() with `state["node_error"]` set
    (state.py) -- and router() checks that FIRST, ahead of both the
    plan-execution shortcut and the general 5-way decision, escalating to
    planner DETERMINISTICALLY (no LLM call -- an LLM call is exactly what
    just failed) with any active plan discarded, same as an ordinary
    rejection-driven escalation. The literal request said "if evaluator
    fails" -- but the crash that prompted it happened in solver, and
    there's no principled reason a specialist's own network hiccup should
    be handled differently from the evaluator's, so this is general: any
    of the five nodes' own LLM call failing routes back to planner the
    same way. "based on current output" is handled by reusing
    _role_body's existing "PREVIOUS ATTEMPT BY <role>" background display
    -- the failing node writes a plain-language explanation into
    `feedback` (state["output"] itself is left untouched by the failure),
    so planner sees exactly what a different specialist's rejected
    attempt already looks like to it, just with a failure reason instead
    of an evaluator's.

Sixth refinement, same day, live-tested: "Hi" / "Solve N Queens with
brute force" / "improve above solution" -- the third turn had no idea
what "above solution" was and looped trying to guess. Root cause: every
turn started `state["messages"]` from scratch (agent/pipeline/run.py's
`_initial()` only ever seeded it with the CURRENT turn's text), even
though chat.py's/tui.py's own `Session.history` was already tracking the
whole conversation client-side -- it just never got handed to the graph.
Two halves, both needed: run.py's new `history` parameter actually feeds
prior turns into `state["messages"]`; `_conversation_so_far()` (below)
is what makes every prompt-builder in THIS file (`_router_body`,
`_step_route_body`, `_role_body`, and evaluator()'s own human_body) show
it, as a "CONVERSATION SO FAR:" block ahead of TASK:/ORIGINAL REQUEST:.
Without both halves this doesn't work -- state carrying the messages but
no prompt ever displaying them would be just as blind as before.
Deliberately NOT a fix for "the model asks the user a clarifying question
mid-run" (there was no such capability anywhere in this graph yet, a
separate and larger gap the same live test surfaced) -- this only made
sure the model has what it needs to not HAVE to ask in a case like
"improve above solution", where the answer was one turn away the whole
time. See the seventh refinement, directly below, for that other gap.

Seventh refinement, same day, same live test's other half (verbatim:
"it got stuck in a loop trying to find what above solution is and was
thinking to ask user but it didnt have that capability... not able to ask
something to user mid thinking if it get's confused"): planner / solver /
summarizer / finder / evaluator can now genuinely pause a run and ask the
person something, instead of guessing or looping. Scoped with the person
before building it (three separate calls, all "any node can ask" /
"widget with multi choice + text bar" / "LangGraph interrupt()/
Command(resume=...)"):

  * `ask_user` joins the six existing tools (execute_python, execute_bash,
    web_search, rag, complete_code, predict_edit) as a seventh ACTION any
    of the five nodes above may reach for, mid-_tool_loop, exactly like
    any other tool -- see PLANNER_PROMPT/SOLVER_PROMPT/SUMMARIZER_PROMPT/
    FINDER_PROMPT/EVALUATOR_PROMPT's shared wording on when to use it
    (sparingly -- it pauses the whole run and costs the person real time).

  * Unlike every other tool, though, `ask_user` cannot just hand a result
    back into _tool_loop's own local `messages` list and keep going --
    the answer has to come from an actual human, which means the WHOLE
    GRAPH has to pause (LangGraph's own checkpointed interrupt()/
    Command(resume=...) mechanism -- app.compile(checkpointer=
    InMemorySaver()) already had a checkpointer wired up, from before this
    refinement, for an unrelated reason: giving every turn its own
    disposable thread id, agent/pipeline/run.py's `_graph_thread_id`).
    _tool_loop, on seeing ACTION: ask_user, raises NeedsUserInput (below)
    instead of dispatching it like a normal tool -- unwinding out of
    _tool_loop and out of whichever role node (or evaluator()) called it,
    all the way to a NEW dedicated `ask_user` node (bottom of this file).
    That node's entire body is "read the question off state, call
    interrupt(), write the answer down, hand back to whoever asked" --
    deliberately nothing else, because langgraph.types.interrupt's own
    docstring is explicit that a node resumes by RE-EXECUTING ITS WHOLE
    BODY from the top; a node with an LLM call or a tool dispatch BEFORE
    its interrupt() would redo that work every single time the person
    answers. Keeping the actual pause point in its own minimal node, with
    NeedsUserInput as the unwind signal that gets it there, is what avoids
    that -- the specialist that got stuck is simply re-invoked fresh
    afterward (ask_user's own Command(goto=<the role that asked>)), not
    resumed mid-loop.

  * The answer goes into `state["context"]` (a `you asked: "..." / the
    user answered: "..."` line, appended the same way finder's own
    gathered material is), NOT into `state["messages"]` -- every node in
    this graph, router() included, reads `state["messages"][-1]` as THE
    TASK for the whole turn; appending the Q&A there would silently
    replace the actual task the next time anything looked. `context`
    already means "material gathered so far for planner/solver to use"
    (state.py) and is already shown to every prompt below via "CONTEXT
    GATHERED SO FAR:" -- reusing it needs no new display mechanism, and
    the re-invoked specialist sees the answer as ordinary background,
    the same way it would see anything finder dug up.

  * Two new AgentState fields carry a pending question across the pause:
    `pending_question`/`pending_choices` (what to show; choices is the
    "multi choice" half of "multi choice + text bar" -- OPTIONAL, an
    empty list means open-ended free text only) and `asking_role` (who to
    hand back to once answered -- ask_user() itself has no other way to
    know). All three are cleared back to None by ask_user() the moment it
    resumes; nothing about this is meant to survive past that one pause.

  * The interrupt surfaces to callers of agent/pipeline/run.py's
    run_pipeline_stream() as a `{"__ask__": {"question", "choices",
    "thread_id"}}` event (instead of the usual `{"__final__": ...}` at
    the very end) -- the stream simply ends there, mid-turn, same
    thread_id and all, and a NEW function, `resume_pipeline_stream()`,
    continues that exact same LangGraph checkpoint thread once the caller
    has an answer. agent/cli/chat.py and agent/cli/tui.py both loop on
    that: render/collect the question, resume, keep going -- possibly
    more than once, if the re-invoked specialist gets stuck again.

Deliberately out of scope for this revision, same as the first:
  - web_search and rag are still STUBBED (tools.py).
  - No domain-specific verification beyond the evaluator's own tool access.
  - No PER-STEP evaluation -- only the whole plan (before execution) and
    the whole final answer (after every step has run) are ever judged. A
    step that turns out wrong is caught at that final judgment and handled
    like any other rejection (retry the specialist that produced it, or
    escalate to a fresh plan) rather than being individually re-judged
    mid-plan.

Deliberately out of scope for a DIFFERENT reason -- not unbuilt-but-planned
inside this file, but a separate, larger subsystem that already has its own
design doc (claude/otto-memory-design.md, Phase 14 "Personal lessons"):
persistent, CROSS-TURN learning about which tasks need a plan (retrieval
from a local embeddings index, a promotion gate, etc.) is not implemented
here. What IS implemented here is the narrower, IN-TURN version -- the
overseer sees this run's own rejection history in its prompt and can act on
it for the rest of THIS run -- because nothing durable persists once the
run ends. Building the durable version is follow-up work against that doc,
not a silent scope-expansion of this change.

Eighth refinement, same day: Phase 2 of claude/otto-tiered-memory-design.md
(the project doc has the full design) -- this graph's own two halves of
that wiring, a DIFFERENT memory system from the paragraph just above (that
one is cross-turn LEARNING; this one is cross-turn conversation MEMORY,
short-term, not persisted past a session). agent/pipeline/run.py's
`_initial()` gained a `memory_context` parameter: agent/cli/shell.py's
`Session` now keeps conversation history in an `agent.memory.queue.
TieredQueue` instead of an unbounded list, and seeds `state["context"]`
with whatever's fallen out of that queue's verbatim recent tier (`X`) --
the queue's own bounded, real Human/AIMessage reconstruction still becomes
`history`/`state["messages"]` exactly as before this refinement, so
`_conversation_so_far()` above needed zero changes; only what counts as
"recent enough to stay verbatim" is now capped. tools.py's new
`recall_memory` tool is the other half -- registered like any other tool
(TOOL_DISPATCH/TOOL_TIERS) and mentioned in every role/evaluator prompt's
ACTION list, same as every other tool (tests/test_prompt_tool_sync.py's
existing "every prompt mentions every dispatchable tool" check is what
keeps this in sync, unchanged by this refinement). FINDER_PROMPT's own
listing gives it the most emphasis (Prefer, in this order: ...) since the
spec's own framing was specifically "if AGENT decides it needs more
details... FINDER will find the relevant stuff" -- but any role stuck on
something from earlier in a long conversation can reach for it directly
rather than routing through finder first.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agent.pipeline.state import AgentState, PlanStep
from agent.pipeline.budget import Budget, current_budget, default_budget
from agent.pipeline.modes import DEFAULT_MODE, MODES, mode_names, mode_reason, parse_mode_body
from agent.pipeline.tools import READ_ONLY, TOOL_DISPATCH, TOOL_TIERS
from agent.pipeline.toolkit import dispatch_table, render_note
from agent.router.llm_provider.base import ProviderError, translate_unknown
from agent.router.mapping import Task
from agent.router.router import Router

ROUTER = Router()  # the LLM-provider router every node calls into -- NOT
                    # the router() graph node below (same naming collision
                    # noted in the first revision: the two are unrelated,
                    # case matters, Python doesn't confuse them, only a
                    # reader skimming might).

#: The four specialists that actually produce work. Always exactly these
#: four; adding a fifth means adding its prompt, its node function, a line
#: in the graph wiring at the bottom of this file, and a mention in
#: ROUTER_PROMPT.
#: Cut from 150 to 30 with the loop rewrite. 150 was sized for a graph that
#: bounced router -> specialist -> router for every few tool calls, so a real
#: run legitimately used dozens of super-steps. One loop uses three or four
#: (agent -> evaluator -> agent -> evaluator), so 30 is still a wide margin and
#: catches a genuine runaway in seconds rather than minutes.
_RECURSION_SAFETY_NET = 30

#: Same rationale as the first revision's identical constant: a diffusing
#: (Mercury) call cut off at max_tokens is not a clean prefix, it's an
#: unconverged snapshot, and must be retried rather than accepted.
MAX_DIFFUSION_RETRIES = 3

#: Appended to a tool result the model has already produced, verbatim, earlier
#: in this same attempt.
#:
#: A plain success gives it nothing to react to. Measured on a hard task: the
#: solver wrote the same file over and over -- every write succeeding, every
#: result saying so -- because "wrote main.c.rs (30 lines)" is not a reason to
#: do anything different, and continuing an established pattern is the easiest
#: thing for a model to do. Naming the repetition is: it turns an
#: indistinguishable success into a fact about the attempt's own history.
#:
#: Deliberately not an error and not a refusal -- the call really did succeed,
#: and re-running something is sometimes right (re-reading a file after
#: changing it). This only says it happened and that repeating cannot change
#: the outcome.
REPEATED_CALL_NOTE = (
    "NOTE: that is {n} {tool} calls in a row against `{target}`, with nothing "
    "run in between. Rewriting it again is guessing. Run it -- compile it, "
    "execute it, test it -- and use what that actually reports."
)

#: Appended when a call fails against something that has already failed in
#: this attempt.
#:
#: Separate from REPEATED_CALL_NOTE, and fires on the SECOND occurrence rather
#: than the third, because the two are not the same mistake. Re-running
#: something that succeeded is wasteful; re-running something that failed,
#: unchanged, means the error was never read. It is also the cheapest thing
#: for a model to produce, which is why it needs saying out loud.
REPEATED_FAILURE_NOTE = (
    "NOTE: `{target}` has already failed once in this attempt and has now "
    "failed again. Do not run it a third time. Read the error above and work "
    "out what is actually wrong -- check the state of the thing you are acting "
    "on, and whether something else is undoing or blocking your change. Fix "
    "what you find, then try."
)

#: How many calls in a row against the same target before the note appears.
#: 3, not 2: a second attempt straight after the first is ordinary (a typo
#: spotted on re-reading), while a third with still nothing run against it is
#: the pattern this exists to break.
REPEATS_BEFORE_NOTE = 3


def _action_target(tool_name: str, body: str) -> str:
    """What a tool call is acting ON -- the file path for the file tools, the
    command itself for the shell ones.

    The target rather than the whole body, because the loop worth catching
    does not repeat verbatim: rewriting the same file with a slightly
    different draft every time looks different byte for byte and is the same
    non-progress. Measured on a hard task: 166 writes to one path, of 8, 721,
    36, 37 and 35 lines, without the compiler ever being run once.
    """
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    if tool_name == "view_image":
        # The path alone is the wrong target here. Asking a second, narrower
        # question about one image is exactly how this tool is meant to be
        # used -- a vision model returns words, so narrowing across calls is
        # the only way to get at detail -- and keying on the path would flag
        # that as repetition and tell the agent to stop.
        return f"view_image:{body.strip()[:200]}"
    if tool_name in {"write_file", "edit_file", "read_file", "list_files"}:
        return first_line.strip()[:120]
    return f"{tool_name}:{first_line.strip()[:120]}"

#: How many of the run's most recent recorded actions get shown (state.py's
#: `actions`). Bounded because this goes into every role and overseer prompt
#: and a long run accumulates hundreds; the most recent describe the world as
#: it is now, and an older attempt since superseded is the one worth
#: forgetting.
_ACTIONS_SHOWN = 40

#: How many unparseable replies IN A ROW end a node's tool loop early.
#:
#: The corrective retry (UNPARSEABLE_FEEDBACK) is worth having: a model that
#: forgot the format once usually gets it right when told. What it cannot fix
#: is a model that is not answering at all. Measured on a hard task: a broken
#: reasoning_effort setting (agent/router/mapping.py's Task.REASON, now
#: corrected) made the provider return empty streams, and the solver spent 39
#: of its 40 turns feeding "you didn't follow the format" to a model that had
#: sent back nothing, issuing 2 real commands in six minutes. The route is
#: fixed; nothing about the loop knew to stop, and the next provider hiccup
#: would spend a whole budget the same way.
#:
#: 3 rather than 2 because the retry genuinely does recover a one-off, and
#: rather than 5 because by the third identical non-answer the budget is
#: better spent letting the overseer pick a different step.
MAX_CONSECUTIVE_DEAD_REPLIES = 3

#: Bound on one node's OWN tool-calling loop (ACTION/execute_TOOL
#: round-trips) before its last reply is used as-is. Unrelated to the
#: (now-removed) overseer retry cap -- this bounds a single node's single
#: turn, not how many turns the overseer may hand out. Applies identically
#: to every role node and the evaluator (_tool_loop is shared by all of
#: them).
#: How many ACTION/tool exchanges one node gets before it must answer with
#: what it has. 5 is right for what this graph was built for -- a node
#: checking its own work, where a sixth exchange usually means it is stuck
#: rather than making progress -- and it stays the default.
#:
#: It is overridable because agentic benchmarks are a genuinely different
#: shape of task: published SWE-bench and terminal-agent traces routinely run
#: dozens of tool calls in one attempt (read a file, edit it, run the tests,
#: read the failure, edit again), and a cap of 5 would measure the cap rather
#: than the agent. An env var rather than a parameter because _tool_loop is
#: reached through five node functions and a LangGraph call, none of which
#: take agent-level configuration today, and threading one through all of
#: them to serve the harness would be a worse trade than this.
#: Temperature now lives in the routing table, per candidate, because only a
#: candidate knows which vendor it is talking to: Anthropic, OpenAI and Gemini
#: honour 0.0, while Inception resets anything below 0.5 to the model default
#: of 1.0 (agent/router/llm_provider/inception_provider.py clamps as a
#: backstop). A single constant here could not be right for both ends of a
#: chain, and passing one at the call site actively overrode the table --
#: Router.model_for merges {**route_params, **overrides}.
MAX_TOOL_ITERATIONS = int(os.environ.get("OTTO_MAX_TOOL_ITERATIONS", "5"))

#: The ACTION: enumeration every role/evaluator prompt shows, built FROM the
#: registry rather than typed out in each prompt. It was hand-kept in sync
#: through three rounds of tool additions, with tests/test_prompt_tool_sync.py
#: standing guard over the copies; deriving it means the next tool added to
#: TOOL_DISPATCH appears in all five prompts by construction, and that test
#: now checks the derivation reaches them rather than checking five hand-
#: written lists against one registry. `ask_user` is appended by hand because
#: it is deliberately NOT in TOOL_DISPATCH -- it unwinds the whole tool loop
#: by raising (see NeedsUserInput below) instead of returning a ToolResult.
#: `ask_user` and `switch_mode` are appended by hand because neither is in
#: TOOL_DISPATCH: they change the LOOP's own state rather than returning a
#: ToolResult -- one pauses the run, the other changes which model answers
#: next and under what guidance.
_TOOL_MENU = "|".join((*TOOL_DISPATCH, "ask_user", "switch_mode"))

#: KEEP THIS SHORT. A fifth habit was added and measured -- "search for the
#: exact text of an error rather than reasoning about it", which for the task
#: in question was one command that would have named the culprit outright. It
#: did not merely fail to take (zero such searches); it wiped out the
#: behaviour the first four had produced, taking system-inspection commands
#: from 17 to 0 and total commands from 80 to 53 on the same task. One run
#: each, so treat the size of that with suspicion but not the sign: past some
#: length the model acts on none of this rather than more of it. Adding a
#: habit here means measuring that the others survive it.
#: General debugging habits, shared by the prompts of the roles that actually
#: change things. Not advice about any particular kind of task -- these are the
#: two things a competent engineer does that a model, left alone, reliably
#: does not.
#:
#: The first is looking at the system rather than at the thing that failed.
#: Observed on a benchmark task: 99 commands, 28 of them re-running the one
#: command that was broken, and not a single `ps`, `crontab` or look at what
#: was running. The fix it applied was correct and kept being undone by
#: processes it never went looking for.
#:
#: The second is not repeating an action that already failed. A model will
#: re-issue a failing command verbatim several times over, because re-trying
#: is cheaper to produce than diagnosing. Saying plainly that a repeat needs a
#: reason is what turns the second attempt into a question about the first.
#: The CODE: body format for each tool, written once instead of five times.
#:
#: It used to be spelled out inline in every prompt at 1054 characters -- 35%
#: of the solver's whole prompt, repeated five ways, and hand-edited five ways
#: every time a tool was added. Together with the habits block below that left
#: the solver's own instructions as about a sixth of what it was reading,
#: which matters more here than it would elsewhere: a fifth habit measurably
#: erased the effect of the four before it, so length on this model is not
#: free.
#:
#: Trimmed to the formats a caller cannot guess. execute_bash takes a command,
#: execute_python takes code, web_search and rag take a query -- saying so
#: costs tokens and tells the model nothing it did not already know from the
#: tool's name.
_TOOL_BODY_HINT = (
    "read_file: a path, optionally `path:START-END`. "
    "list_files: a directory. "
    "write_file: the path on the FIRST line, the file's whole content after "
    "it -- no separator, no JSON. "
    "edit_file: the path, then a line `---OLD---`, the exact text to replace, "
    "a line `---NEW---`, the replacement. "
    "complete_code: code, optionally `---SUFFIX---` then trailing code. "
    "predict_edit: code only, no instruction. "
    "recall_memory: a search query. "
    "switch_mode: one word -- " + "|".join(mode_names()) + " -- and optionally "
    "why on the next line. The conversation continues; nothing is lost. "
    "ask_user: a question, optionally then `CHOICES: a | b`. Ask when an "
    "action you are about to take cannot be undone AND more than one target "
    "fits -- which recipient, which record, which file. Picking one and "
    "hoping is the worst option available."
)

#: Which standing tools change things, named from the registry rather than
#: typed out, so the next tool added to a mutating tier says so by itself.
#:
#: This is agent/pipeline/tools.py's TOOL_TIERS finally having a reader in
#: production. It has always been declared and always been enforced only by a
#: unit test, which is a strange place for the one distinction that decides
#: whether a mistake can be taken back. Claw-Eval task T026 is what it costs:
#: three contacts matched "Manager Zhang", the grader required asking which,
#: and the agent sent to the first -- then sent twice more. Safety is a
#: multiplier in that benchmark's formula, so the whole task scored zero.
#:
#: The rule in the ask_user hint above is about the ACTION rather than the
#: tool, because a benchmark's own tools (agent/pipeline/toolkit.py) carry no
#: tier and sending mail is exactly the case that matters. This line is the
#: half that CAN be derived, and it keeps the registry honest.
_MUTATING_TOOLS = tuple(
    name for name, tier in TOOL_TIERS.items() if tier != READ_ONLY
)

#: The ACTION/CODE protocol, assembled once for every prompt that offers tools.
_ACTION_BLOCK = (
    "reply with exactly\nACTION: <" + _TOOL_MENU + ">\nCODE:\n<"
    + _TOOL_BODY_HINT
    + ">\nand you will be shown the result, then you can continue. "
    + ", ".join(_MUTATING_TOOLS) + " change things and cannot be undone. "
)

_DIAGNOSTIC_HABITS = (
    "Four habits, whatever the task:\n"
    "1. Look at the system, not just at the thing that broke. Before you "
    "conclude why something fails, find out what state it is actually in -- "
    "what is running, what is scheduled, what a file really contains, what "
    "changed recently. If a fix does not hold, or something reverts, then "
    "something else is acting on it and finding THAT is the task.\n"
    "2. Never run the same thing twice hoping for a different answer. If a "
    "command failed, or did not achieve what you expected, work out why "
    "before you touch it again -- read the actual error, check your "
    "assumption about what it was going to do. Repeating an action is only "
    "worth doing when you have changed something that would change its "
    "result, and you should be able to say what.\n"
    "3. Look wide before you look narrow. A filter hides everything you did "
    "not think to ask for, so a `grep` that comes back empty or unsurprising "
    "is not evidence -- it is a guess about what the answer looks like. When "
    "a filtered search tells you nothing, run it again unfiltered and read "
    "what is really there before you conclude anything.\n"
    "4. If something goes back to how it was, that is not your change "
    "failing -- it is something else changing it. Ask what could: a process "
    "still running, a scheduled or repeating job, a service, something "
    "watching the file. Enumerate them and look, rather than applying the "
    "same fix harder.\n\n"
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

#: Phase one of judging: work out what a good answer would have to contain,
#: from the TASK ALONE, before the attempt is visible.
#:
#: This is the whole reason the evaluator is worth its calls. RefineBench
#: measured self-refinement over five turns at 31.3% for the best model and
#: -2.5% to 0% for most; the SAME models reach 90-98% when given an external
#: checklist. An evaluator that only re-reads the actor's own output is
#: measured at approximately nothing, and that is what this one was: it saw the
#: answer, the action record, the transcript tail and the gathered context --
#: all of it produced by the actor it was judging.
#:
#: A rubric derived from the task is the cheapest thing the actor does not have.
#: Generating it in a SEPARATE phase is load-bearing rather than tidy: writing
#: criteria while looking at an answer produces criteria that the answer
#: happens to meet. The verifier paper reports this structure taking agreement
#: with humans from 0.26-0.31 to 0.64 -- inside the human inter-annotator band
#: of 0.53-0.57 -- and false positives from 0.40-0.45 to 0.01, and shows the
#: gain survives giving the baselines the same model, so it is architectural
#: rather than a better judge.
RUBRIC_PROMPT = (
    "You are about to judge someone else's attempt at the task below. First, "
    "before you see any attempt, write down what a correct answer would have "
    "to contain.\n\n"
    "Reply with 2 to 5 criteria, one per line, each starting with `- `. Each "
    "must be something you could CHECK rather than an opinion: a value that "
    "must be right, a file that must exist, a command that must succeed, a "
    "question that must be answered. Make them independent -- overlapping "
    "criteria double-count one mistake.\n\n"
    "Do not write criteria about style, effort or presentation. Nothing else "
    "in your reply, no preamble."
)

EVALUATOR_PROMPT = (
    "Judge whether the {target} below actually satisfies the original "
    "request -- {target_note}.\n\n"
    "CRITERIA, written before this attempt was visible. Judge against these "
    "and nothing else:\n{rubric}\n\n"
    "Take each in turn. A criterion is met, not met, or blocked by something "
    "outside the agent's control -- a missing file it could not create, a "
    "service that was down, a credential it was never given. That last case "
    "is NOT the agent being wrong, and saying so is how a real obstacle stops "
    "being counted as a failure.\n\n"
    "CHECK IT, DO NOT TAKE ITS WORD. If the request asked for something to "
    "be changed, fixed, built or made to work, run a command that would fail "
    "if it had not been -- and judge what that command actually reports, not "
    "what the output above claims. An output that says the work is done is "
    "not evidence the work is done. If your own check contradicts it, reject "
    "and quote exactly what you saw.\n\n"
    "Check the RESULT, not the steps. A step that succeeded does not mean "
    "the thing the request asked for is true now -- something else may have "
    "undone it, or the step may have been aimed at the wrong target. If a "
    "check passes, consider whether it would still pass a minute from now, "
    "and if you have reason to doubt it, look for what would change it "
    "back.\n\n"
    "You may check ONE thing with a tool if a criterion genuinely cannot be "
    "settled from what you were shown: "
    + _ACTION_BLOCK +
    "When you are done, reply with exactly\nFINAL:\n"
    "MET: the number of criteria met, then / then the number of criteria\n"
    "BLOCKED: yes or no -- whether anything unmet was outside the agent's "
    "control\n"
    "APPROVE: yes or no\n"
    "WHY: one sentence naming the first criterion that failed, or why it "
    "passed\n"
    "You have at most {max_iter} exchanges before your last reply is used "
    "as-is."
)
#: The whole agent, in one prompt.
#:
#: This replaces PLANNER_PROMPT, SOLVER_PROMPT, SUMMARIZER_PROMPT and
#: FINDER_PROMPT, and it is SHORTER than the four of them put together by a
#: wide margin -- each of those carried its own copy of _ACTION_BLOCK (831
#: characters), so the protocol alone was stated four times. Said once here,
#: with what is specific to each role moved into agent/pipeline/modes.py as a
#: guidance block under 400 characters, the text a call actually carries goes
#: DOWN even though the agent can now do everything all four roles could.
#:
#: That matters more here than it would elsewhere. The comment above
#: _DIAGNOSTIC_HABITS records a measurement where a fifth habit did not merely
#: fail to take -- it erased the effect of the four before it, taking
#: system-inspection commands from 17 to 0. Length is not free on this model,
#: so the budget freed by de-duplicating the protocol is the budget that pays
#: for the mode machinery.
AGENT_PROMPT = (
    "You are an engineer with a shell, working a task end to end: find out "
    "what is true, do the work, and confirm it actually holds.\n\n"
    + _DIAGNOSTIC_HABITS +
    "You work in a MODE, which sets both how you are thinking and which model "
    "you are running on. You start in " + DEFAULT_MODE + ". Switch when the "
    "KIND of work changes -- `ACTION: switch_mode` with one of "
    + "|".join(mode_names()) + " -- not to restate what you are already doing. "
    "The conversation carries over: everything above stays, and you keep every "
    "tool.\n\n"
    + _ACTION_BLOCK +
    "One tool call per reply.\n\n"
    "Before you finish, run something that would FAIL if the task were not "
    "done, and read what it prints. An explanation is not evidence. When you "
    "have seen it work, reply with exactly\nFINAL:\n<the answer itself -- the "
    "numbers, the names, the decision -- not a description of what you did>"
)


#: Fed back inside _tool_loop when a reply has neither ACTION: nor FINAL:
#: (or a FINAL: with nothing after it). A marker-less or empty reply is
#: never silently accepted as an answer -- the node is told what went wrong
#: and made to retry, spending one of MAX_TOOL_ITERATIONS exchanges.
UNPARSEABLE_FEEDBACK = (
    "Your last reply didn't match either required format -- it had no "
    "ACTION: with a tool call, and no FINAL: followed by an answer (or "
    "FINAL: was there but empty). Reply again using exactly one of those "
    "two formats, with no other text."
)


# --------------------------------------------------------------------------
# Shared LLM-call plumbing -- unchanged from the first revision.
# --------------------------------------------------------------------------

def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _call(llm, messages: list) -> str:
    """Run one streamed chat call and return its settled text.

    A diffusing (Mercury) call whose stream ends with finish_reason "length"
    is retried with doubled max_tokens rather than accepted -- see
    MAX_DIFFUSION_RETRIES. A non-diffusing truncation is returned as-is: it
    is an incomplete but internally clean prefix, the ordinary "ran out of
    budget" case every LLM caller already has to tolerate.
    """
    # Every model REQUEST is counted here, not every logical call, and the
    # retries below are the reason. An empty-stream retry and a diffusion
    # max_tokens doubling are each a fresh HTTP request that costs real money,
    # so a budget that counted _call once would undercount by up to 3x exactly
    # when a run is going badly. agent/pipeline/budget.py has the rest of the
    # argument; nothing bound is the normal case and makes this a no-op.
    budget = current_budget()
    current = llm
    for attempt in range(MAX_DIFFUSION_RETRIES):
        if budget is not None:
            budget.spend()
        reply = None
        try:
            for chunk in current.stream(messages):
                reply = chunk if reply is None else reply + chunk
        except ValueError as exc:
            # langchain_core raises ValueError("No generation chunks were
            # returned") from inside stream() when the provider yields nothing
            # at all -- so the `reply is None` check below never gets the
            # chance to see it. This module always meant to treat an empty
            # stream as an empty answer (the caller's unparseable-reply retry
            # then does its job); an uncaught ValueError instead unwinds the
            # whole graph, which is how one benchmark task died outright
            # rather than scoring badly.
            if "No generation chunks" not in str(exc):
                raise
            logger.warning(
                "%s returned an empty stream (attempt %d/%d)",
                _model_label(current), attempt + 1, MAX_DIFFUSION_RETRIES,
            )
            if attempt < MAX_DIFFUSION_RETRIES - 1:
                continue  # transient often enough to be worth one more try
            return ""
        except ProviderError:
            # Inception's own provider already translated this one; re-wrapping
            # would bury the specific subclass the callers branch on.
            raise
        except Exception as exc:
            # The chat model here may be a LangChain class Otto does not own,
            # which raises its vendor's SDK errors straight out of .stream().
            # _run_role and evaluator catch only ProviderError, so an untranslated
            # one unwinds the whole graph instead of becoming a clean edge back
            # to the overseer.
            #
            # Clause order is load-bearing twice over. The ValueError clause
            # must stay FIRST, or a bare `except Exception` swallows the
            # empty-stream retry above. And a `raise` from an earlier clause is
            # not caught by a later clause of the same try, which is what keeps
            # an unrelated ValueError propagating. Collapsing these into one
            # handler with isinstance checks breaks both.
            raise translate_unknown(
                exc,
                provider=getattr(current, "_otto_provider", ""),
                model_id=_model_label(current),
            ) from exc
        if reply is None:
            return ""

        finish_reason = (reply.response_metadata or {}).get("finish_reason")
        if not (getattr(current, "diffusing", False) and finish_reason == "length"):
            return _content_text(reply.content)

        # Everything below is Inception-only by construction: `diffusing` is a
        # ChatInception field, so no other vendor's model reaches it.
        if attempt == MAX_DIFFUSION_RETRIES - 1:
            raise ProviderError(
                f"{_model_label(current)} truncated a diffusing response at "
                f"max_tokens={getattr(current, 'max_tokens', None)!r} on every one of "
                f"{MAX_DIFFUSION_RETRIES} attempts -- refusing to hand an "
                f"unconverged diffusion snapshot to the rest of the graph"
            )
        bumped = max(getattr(current, "max_tokens", None) or 1024, 1024) * 2
        logger.warning(
            "%s truncated a diffusing response at max_tokens=%r "
            "(attempt %d/%d) -- retrying at max_tokens=%d",
            _model_label(current), getattr(current, "max_tokens", None),
            attempt + 1, MAX_DIFFUSION_RETRIES, bumped,
        )
        current = current.model_copy(update={"max_tokens": bumped})
    return ""  # unreachable -- the loop always returns or raises


def _parse_rubric(text: str) -> list[str]:
    """The criteria out of a rubric reply. Bullet lines only.

    Bounded at RUBRIC_MAX because criteria are scored one by one: an
    unbounded list turns one judgment into an unbounded number of them, and
    the verifier paper's own guidance is a small set of NON-OVERLAPPING
    criteria, since overlapping ones double-count a single mistake.
    """
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        for marker in ("- ", "* ", "\u2022 "):
            if line.startswith(marker):
                line = line[len(marker):].strip()
                break
        else:
            continue
        if line:
            lines.append(line)
    return lines[:RUBRIC_MAX]


def _parse_verdict(text: str) -> "Verdict":
    """The four things a judgment has to say, out of one FINAL body.

    Separated because one number cannot carry them. An attempt can take every
    right step and be stopped by a missing credential, or reach the right
    answer by accident -- and a single approve/reject collapses those into the
    same signal, which is how a judge ends up training the agent on noise.
    """
    approve, reason = _parse_approval(text)
    met = total = 0
    if "MET:" in text:
        fragment = text.split("MET:")[1].split("\n")[0]
        numbers = [int(n) for n in re.findall(r"\d+", fragment)[:2]]
        if len(numbers) == 2:
            met, total = numbers
    blocked_line = text.split("BLOCKED:")[1].split("\n")[0].lower() if "BLOCKED:" in text else ""
    return Verdict(
        approved=approve,
        reason=reason,
        # The continuous score. Kept apart from `approved` on purpose: "three
        # of four criteria met" and "nothing worked" are both rejections and
        # should not look alike to anything reading this back.
        process=round(met / total, 3) if total else (1.0 if approve else 0.0),
        criteria=(met, total),
        blocked="yes" in blocked_line,
    )


def _parse_approval(text: str) -> tuple[bool, str]:
    after = text.split("APPROVE:")[-1]
    line, _, rest = after.partition("\n")
    approve = "yes" in line.strip().lower()
    reason = rest.split("WHY:")[-1].strip() if "WHY:" in rest else rest.strip()
    return approve, reason


def _strip_code_fence(text: str) -> str:
    """Strip a whole-reply ```lang\\n...\\n``` wrapper, if present.

    Applied to every FINAL body regardless of which node produced it: the
    solver's code needs to be raw, fence-free text to be usable directly
    (e.g. by a golden-set checker that execs it), and stripping a fence
    from a plain-text answer that happens to be entirely fenced is
    harmless. Only strips when the ENTIRE reply is one fenced block (starts
    with ``` and ends with a lone ``` line) -- a reply that merely mentions
    backticks mid-text is left alone.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1])
    return text


class NeedsUserInput(Exception):
    """Raised by _tool_loop (below) when a reply's ACTION: is ask_user --
    a signal, not a tool result. Every other ACTION gets dispatched via
    TOOL_DISPATCH and its result fed straight back into this same loop's
    local `messages`, but an ask_user answer has to come from an actual
    human, which means the WHOLE GRAPH has to pause -- not just this one
    node's own tool-calling loop. Raising unwinds out of _tool_loop and out
    of whichever role node (or evaluator()) called it, all the way to a
    dedicated `ask_user` graph node built to do nothing else (module
    docstring, seventh refinement, explains why the interrupt() call
    itself can't just live here: LangGraph re-executes a node's ENTIRE
    body from the top on resume, so anything with side effects before it
    -- including the LLM calls this loop already made -- would replay a
    second time).
    """

    def __init__(self, question: str, choices: list[str]):
        self.question = question
        self.choices = choices
        super().__init__(question)


def _parse_ask_user_body(body: str) -> tuple[str, list[str]]:
    """Split an ask_user ACTION's CODE: body into (question, choices).

    A line starting with "CHOICES:" (case-insensitive, anywhere in the
    body) holds pipe-separated options; everything else is the question.
    No such line -- the common case, a genuinely open-ended question --
    means choices=[] and the whole body is the question.
    """
    lines = body.splitlines()
    choices: list[str] = []
    question_lines = []
    for line in lines:
        if line.strip().upper().startswith("CHOICES:"):
            raw = line.split(":", 1)[1]
            choices = [c.strip() for c in raw.split("|") if c.strip()]
        else:
            question_lines.append(line)
    return "\n".join(question_lines).strip(), choices


#: A line that STARTS a new directive, and therefore ENDS the CODE: body
#: above it. Anchored to the start of a line so a mention of the word inside
#: a command ("echo ACTION: done") doesn't truncate anything.
_NEXT_DIRECTIVE = re.compile(r"^[ \t]*(?:ACTION|FINAL):", re.MULTILINE)


#: Chat-template control tokens a model sometimes emits into its own visible
#: output -- "<|tool_call_start|>" and friends. They are markup for the
#: serialiser, never content, and they arrive mid-body where nothing else
#: strips them: observed once in a Terminal-Bench run, where the first command
#: of the task became `curl -v example.com\n\n<|tool_call_start|> 2>&1 | head`
#: and bash answered "syntax error near unexpected token `|'". Rare, and free
#: to remove.
_SPECIAL_TOKEN = re.compile(r"<\|[a-z_]+\|>")


def _code_body(text: str) -> str:
    """The CODE: body of the FIRST action in `text`, ending where the next
    ACTION:/FINAL: begins.

    It used to be "everything after the first CODE:, to the end of the reply".
    That is correct only while the model emits exactly one tool call per reply,
    which it does not: asked for one step it will sometimes lay out the whole
    plan at once, and the entire rest of the reply -- the literal lines
    "ACTION: execute_bash", "CODE:", and a trailing "FINAL:" block -- was then
    handed to the shell as part of the command. Terminal-Bench transcripts
    caught it: bash reporting `ACTION:: command not found` mid-task, an exit
    code of 127 for a command whose real work had actually succeeded, and an
    agent reading that as failure. Only the first call is executed either way;
    the loop feeds its result back and the model reissues the rest.
    """
    if "CODE:" not in text:
        return ""
    after = text.split("CODE:", 1)[1]
    match = _NEXT_DIRECTIVE.search(after)
    body = after[: match.start()] if match else after
    return _SPECIAL_TOKEN.sub("", body).strip()


def _parse_worker_reply(text: str) -> tuple[Literal["action", "final", "unparseable"], str, str]:
    """Split a reply into (kind, tool_name, body).

    For an ACTION: (`"action"`, the tool name on that line, the CODE: body).
    For a FINAL: with a non-empty body: (`"final"`, `""`, the answer text).
    Anything else -- no ACTION:/FINAL: marker anywhere, or a FINAL: with
    nothing (or only whitespace) after it -- is `"unparseable"`, which
    _tool_loop treats as a signal to retry with corrective feedback rather
    than accepting it.
    """
    if "ACTION:" in text:
        action_line = text.split("ACTION:", 1)[1].split("\n", 1)[0].strip()
        code = _code_body(text)
        return "action", action_line, code
    if "FINAL:" in text:
        body = text.split("FINAL:", 1)[-1].strip()
        if body:
            return "final", "", body
        return "unparseable", "", text.strip()
    return "unparseable", "", text.strip()


def _model_label(llm) -> str:
    """A model's id for a log line, whichever vendor's chat class this is.

    `ChatInception`, `ChatAnthropic` and `ChatGoogleGenerativeAI` expose
    `model`; `ChatOpenAI` exposes `model_name` and keeps `model` only as a
    populate-by-name alias, so the plain attribute access this replaced raised
    AttributeError there. The messages that used it were also hardcoded to say
    "inception:", which is simply untrue for any other vendor.
    """
    for attr in ("model", "model_name", "model_id"):
        value = getattr(llm, attr, None)
        if isinstance(value, str):
            return value
    return type(llm).__name__


def _summarise_action(tool_name: str, body: str, result) -> str:
    """One line describing a tool call and how it went, for the record the
    overseer and a re-invoked role read (agent/pipeline/state.py's `actions`).

    Deliberately lossy: the point is "you already tried this, here is what came
    back", not a replayable transcript. The first line of the body identifies
    the call (a path, a command), and only a FAILING result's text is carried
    forward -- a compiler error is exactly what the next attempt needs, the
    full stdout of a successful build is noise.
    """
    first_line = next((line for line in body.splitlines() if line.strip()), "")
    call = f"{tool_name} {first_line.strip()[:90]}"
    if result.returncode == 0:
        detail = (result.stdout or "").strip().splitlines()
        return f"{call} -> ok{': ' + detail[0][:120] if detail else ''}"
    problem = (result.stderr or result.stdout or "").strip().replace("\n", " ")
    return f"{call} -> FAILED (exit {result.returncode}): {problem[:200]}"


def _tool_loop(llm, messages: list, actions: list[str] | None = None,
               *, max_iterations: int | None = None) -> str:
    """Run the shared ACTION/FINAL tool-calling loop -- every role node AND
    the evaluator drive their conversation through this one function. A
    role node's FINAL body IS its candidate answer; the evaluator's FINAL
    body is instead parsed by _parse_approval() at the call site, since
    "APPROVE: yes/no\\nWHY: ..." is answer-shaped text as far as this loop
    is concerned, just interpreted differently by its caller.

    `output` (returned if the loop exhausts without ever hitting the
    `return` below) is ONLY ever set from an actual attempted answer -- a
    FINAL body, or, failing that, the last unparseable reply. An ACTION
    reply never touches it: a tool call is not a candidate answer.
    """
    # TOOL_DISPATCH plus whatever agent/pipeline/toolkit.py has bound for this
    # run -- empty in every ordinary turn, so this is a dict copy of nothing.
    # A benchmark that hands the agent task-specific tools (agent/eval/
    # claw_bench.py) binds them there rather than mutating the registry, and
    # the note is how the prompt finds out they exist: the menu in
    # _ACTION_BLOCK is derived at import and cannot know about them.
    dispatch = dispatch_table()
    note = render_note()
    if note:
        messages.insert(1, SystemMessage(note))

    budget = current_budget()
    output = ""
    dead_replies = 0
    last_target: str | None = None
    repeats = 0
    failed_targets: set[str] = set()
    for _ in range(max_iterations or MAX_TOOL_ITERATIONS):
        # The evaluator drives this loop, and a judgment that kept checking
        # past the ceiling would spend the budget the agent was stopped to
        # protect. Its own MAX_TOOL_ITERATIONS cap stays as the inner bound.
        if budget is not None and budget.spent():
            return output
        text = _call(llm, messages)
        kind_of_reply, tool_name, body = _parse_worker_reply(text)

        if kind_of_reply == "final":
            return _strip_code_fence(body)

        if kind_of_reply == "unparseable":
            dead_replies += 1
            output = text  # fallback if the loop ends here, exhausted or not
            if dead_replies >= MAX_CONSECUTIVE_DEAD_REPLIES:
                logger.warning(
                    "tool loop: %d unparseable replies in a row -- giving up "
                    "on this node rather than spending the rest of its budget",
                    dead_replies,
                )
                return output
            # Never an EMPTY AIMessage. `_call` returns "" on an empty stream,
            # Anthropic rejects empty text blocks, and this loop is the
            # evaluator's -- which runs on Anthropic. One empty reply here
            # breaks every later call in the same judgment. The same guard
            # exists in `_agent_loop`; this copy did not get it, which is the
            # argument for there being one loop rather than two.
            if text:
                messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue

        dead_replies = 0

        # ACTION -- deliberately does NOT touch `output` (see docstring).
        if tool_name == "ask_user":
            # Not a normal tool -- see NeedsUserInput's own docstring for
            # why this has to unwind all the way out rather than being
            # just another TOOL_DISPATCH entry.
            question, choices = _parse_ask_user_body(body)
            raise NeedsUserInput(question or "(no question given)", choices)
        if tool_name not in dispatch:
            evidence = (
                f"tool {tool_name!r} is not available "
                f"(allowed: {sorted(dispatch)})"
            )
        else:
            result = dispatch[tool_name](body)
            if actions is not None:
                actions.append(_summarise_action(tool_name, body, result))
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
            target = _action_target(tool_name, body)
            repeats = repeats + 1 if target == last_target else 1
            last_target = target
            failed_before = target in failed_targets
            if result.returncode != 0:
                failed_targets.add(target)
            if result.returncode != 0 and failed_before:
                # A failing call repeated is the strongest possible signal
                # that the last one was not understood -- act on it at once
                # rather than waiting for a run of three.
                evidence += "\n\n" + REPEATED_FAILURE_NOTE.format(target=target)
            elif repeats >= REPEATS_BEFORE_NOTE:
                evidence += "\n\n" + REPEATED_CALL_NOTE.format(
                    n=repeats, tool=tool_name, target=target,
                )
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))
    return output


# --------------------------------------------------------------------------
# The agent loop -- one conversation, many modes.
#
# What this replaces: router -> role -> router -> evaluator -> router, where
# every node rebuilt [SystemMessage, HumanMessage] from scratch and the tool
# conversation died with the node. Measured on Claw-Eval traces, the boundaries
# between those nodes cost 20 to 126 seconds each and were 45 to 69% of a run's
# wall time, while every tool call a task made totalled 0.1 to 0.4 seconds. One
# round of work was five model calls, three of them the router and evaluator.
#
# The agent forgot what it had just done AND paid to be reminded. Those were
# never two problems: they were the same boundary. So there is one loop now, and
# changing role is an appended message rather than a node transition.
# --------------------------------------------------------------------------

#: How many times a run may switch mode before it is told to finish in the one
#: it is in. A backstop, not the main guard -- a swap already costs a model call
#: and yields no tool result, so it competes for budget on its own, and
#: `_refuse_idle_swap` below catches ping-ponging much earlier.
MAX_MODE_SWAPS = 8

#: Said when a swap follows a swap with nothing done in between. Deliberately
#: the same shape as REPEATED_CALL_NOTE, which measurement already showed this
#: model acts on.
IDLE_SWAP_NOTE = (
    "NOTE: you switched to {last} and now to {want} with nothing done in "
    "between. Switching is not progress. Do the work in the mode you are in."
)


def _emit(payload: dict) -> None:
    """Send one live event to whoever is streaming this run, or do nothing.

    LangGraph's "updates" stream only emits when a node RETURNS. That was fine
    when a node returned every few seconds; with one loop that runs for minutes
    it would mean `otto chat` sits silent and then prints one panel. This is the
    "custom" stream (agent/pipeline/run.py's _STREAM_MODES), and the payload is
    deliberately shaped exactly like a node update so agent/cli/chat.py and
    agent/cli/tui.py need no changes at all.

    Best-effort by design: outside a graph run, or when nothing subscribes to
    the custom stream, the writer is absent or a no-op -- so run_pipeline() and
    every offline test are unaffected.
    """
    try:
        from langgraph.config import get_stream_writer

        get_stream_writer()(payload)
    except Exception:  # not in a graph, or no custom subscriber
        pass


def _mode_message(name: str) -> HumanMessage:
    """The message that puts the loop into a mode.

    A HumanMessage, NEVER a SystemMessage, and this is load-bearing rather than
    stylistic: langchain_anthropic raises ValueError("Received multiple
    non-consecutive system messages") and langchain_google_genai hoists a
    mid-list system message into `system_instruction` out of position or drops
    it at a bare `else: pass`. Three of the routed Tasks are pinned to
    Anthropic, so a system message here would be a hard 400 in some modes and a
    silent no-op in others.
    """
    return HumanMessage(
        f"MODE: {name}\n{MODES[name].guidance}\n"
        "Same task, same tools, same conversation -- everything above still applies."
    )


def _seed_transcript(state: AgentState, task_text: str) -> list:
    """The conversation a run starts from. Built once per run, never rebuilt.

    The two system messages are adjacent at indices 0 and 1 on purpose: a run of
    system messages at the very start is legal on every vendor here, and the
    moment one appears later it is not (see _mode_message). Everything after
    them is Human/AI for the life of the run.
    """
    history = _conversation_so_far(state)
    context = state.get("context") or ""
    body = "\n\n".join(part for part in (
        f"CONVERSATION SO FAR:\n{history}" if history else "",
        f"CONTEXT GATHERED SO FAR:\n{context}" if context else "",
        f"TASK:\n{task_text}",
    ) if part)

    note = render_note()
    messages: list = [SystemMessage(AGENT_PROMPT)]
    if note:
        messages.append(SystemMessage(note))
    messages.append(HumanMessage(body))
    messages.append(_mode_message(state.get("mode") or DEFAULT_MODE))
    return messages


def _plain(messages: list) -> list[dict]:
    """A transcript as JSON-able dicts, for the checkpointer and the trace.

    LangGraph serialises state on every super-step and Langfuse serialises it
    into every span, so storing message OBJECTS would put the whole
    conversation through both on each pass.
    """
    kinds = {HumanMessage: "human", AIMessage: "ai"}
    return [
        {"kind": kinds[type(m)], "content": _content_text(m.content)}
        for m in messages
        # System messages are deliberately NOT stored. The only ones in the list
        # are the opening pair, which `agent` rebuilds on resume; storing them
        # would round-trip the same two constants through the checkpointer and
        # every Langfuse span on each super-step, and would risk one being
        # revived into the middle of a list -- the exact shape that raises on
        # Anthropic and is silently dropped on Gemini.
        if type(m) in kinds and _content_text(m.content).strip()
    ]


def _revive(stored) -> list:
    """The inverse of _plain. A stored SystemMessage is dropped rather than
    revived: the only ones ever stored are the opening pair, which
    _seed_transcript rebuilds, and reviving one into the middle of a list is
    the exact failure _mode_message exists to avoid."""
    if not stored:
        return []
    builders = {"human": HumanMessage, "ai": AIMessage}
    out = []
    for entry in stored:
        builder = builders.get(entry.get("kind", "human"))
        content = entry.get("content") or ""
        if builder is not None and content:
            out.append(builder(content))
    return out


def _conversation_so_far(state: AgentState) -> str:
    """Prior turns of this session's conversation, as plain dialogue lines
    -- everything in state["messages"] except the very last one (today's
    task, which every caller below already shows separately as TASK:).

    2026-09-10 design call, live-tested: "improve above solution" as a
    fresh turn had nothing to improve -- state["messages"] held only that
    one sentence, because agent/pipeline/run.py's _initial() never seeded
    it with anything earlier. That half of the fix is run.py's new
    `history` parameter; THIS half is making every prompt below actually
    show what it carries once it's there. Empty on a session's first turn,
    and for every eval/debug caller that never passes `history` at all
    (run.py's default `()`) -- in both cases `state["messages"]` has
    exactly one entry, so this returns "" and nothing changes for them.
    """
    prior = state["messages"][:-1]
    if not prior:
        return ""
    lines = []
    for m in prior:
        speaker = "you" if isinstance(m, HumanMessage) else "otto"
        lines.append(f"{speaker}: {_content_text(m.content)}")
    return "\n".join(lines)


def _actions_block(state: AgentState) -> str:
    """What has already been done in this run, as prompt text -- or "" if
    nothing has.

    Shown to BOTH the overseer and the role it dispatches, because the loop
    this fixes needed both to see it. A role's tool conversation lives in a
    local list inside _tool_loop and dies when that node returns, so a
    re-invoked role started blind; and the overseer, deciding what to do next,
    could see only that "solver produced an output", never what solver had
    tried. Measured on a hard task: the solver wrote the same file 166 times
    across five rounds, a different draft each time, never compiling any of
    them. Neither half of the loop knew the work had already been done.
    """
    already = state.get("actions") or []
    if not already:
        return ""
    return (
        "WHAT HAS ALREADY BEEN DONE (tool calls from earlier attempts in this "
        "run, oldest first). These already happened and their effects are "
        "real. Do not repeat one expecting a different result -- build on it, "
        "or find out why it did not work:\n"
        + "\n".join(f"- {line}" for line in already[-_ACTIONS_SHOWN:])
    )


#: How many exchanges the evaluator gets to reach a verdict.
#:
#: Measured, after giving it the evidence it had been missing: it went from
#: judging blind to spending its whole five-iteration budget checking, on every
#: judgment. On a task as small as "write fib.py and run it", one run cost 24
#: model calls and 106 seconds -- and FIFTEEN of those 24 were the evaluator,
#: three judgments of five calls each. The loop doing the actual work used six.
#:
#: That is the opposite of what the evidence was for. The point was one
#: better-informed judgment, not four extra checks per judgment: it can now see
#: the commands that were run and what they printed, so the common case needs no
#: tool call at all. Two leaves room for one real check when something genuinely
#: cannot be taken on trust.
MAX_EVALUATOR_ITERATIONS = 2

#: Criteria per judgment. Small and non-overlapping is the point -- each is
#: scored in turn, and overlapping criteria double-count one mistake.
RUBRIC_MAX = 5


@dataclass(frozen=True, slots=True)
class Verdict:
    """What one judgment concluded, kept as four separate facts."""

    approved: bool
    reason: str
    #: 0.0-1.0, the share of criteria met. The continuous half.
    process: float
    #: (met, total), so a caller can say "3 of 4" rather than "0.75".
    criteria: tuple[int, int]
    #: Whether what went unmet was outside the agent's control. An environment
    #: blocker is not the agent being wrong, and counting it as one teaches
    #: the wrong lesson to anything downstream.
    blocked: bool


#: How many rejections a run may collect before the answer stands anyway.
#: Judgment is worth paying for; judgment without a bound is a way to spend a
#: whole budget re-reading the same answer.
MAX_REJECTIONS = 2

#: Ceiling on each piece of evidence handed to the evaluator. Two-ended, like
#: agent/pipeline/tools.py's own clip, for the same reason: the start says what
#: was attempted and the end says how it came out, and keeping only the head
#: loses the half that decides the verdict.
_EVIDENCE_CHARS = 6000

#: How much of the loop's own conversation the evaluator sees. The tail, because
#: what it needs to check is how the run FINISHED -- the commands that were meant
#: to prove the work, and what they actually printed.
_EVIDENCE_MESSAGES = 6


def _clip_evidence(text: str) -> str:
    if len(text) <= _EVIDENCE_CHARS:
        return text
    half = _EVIDENCE_CHARS // 2
    return (
        f"{text[:half]}\n... [{len(text) - _EVIDENCE_CHARS} characters omitted] "
        f"...\n{text[-half:]}"
    )


def _evidence_tail(state: AgentState) -> str:
    """The end of the agent's own working, for the evaluator to check against.

    This is the evidence that used not to exist. The evaluator saw an answer and
    no way to verify it, so "I cannot confirm this" came back as a rejection --
    and each rejection cost a full extra round of the graph.
    """
    stored = state.get("transcript") or []
    tail = stored[-_EVIDENCE_MESSAGES:]
    if not tail:
        return ""
    speaker = {"ai": "otto", "human": "result"}
    lines = [
        f"{speaker.get(entry.get('kind'), 'result')}: {entry.get('content', '')}"
        for entry in tail
    ]
    return _clip_evidence("\n".join(lines))


#: When the loop starts compacting itself, in characters of conversation.
#: Roughly 40k tokens at four characters a token -- comfortably inside every
#: routed model's window, and chosen so compaction is rare and chunky rather
#: than incremental. That matters for cost as well as noise: every vendor
#: caches its own prefix, and rewriting history invalidates the cache from the
#: rewrite point, so many small compactions would cost more than they save.
LOOP_COMPACT_AT = 160_000

#: How many recent messages stay verbatim. The tail is what the model is
#: actually reasoning over; the head is what it has already acted on.
KEEP_VERBATIM = 8

#: What a compacted tool result is cut down to. Enough to remember the shape
#: of what came back -- an error class, a count, the first line of output --
#: without carrying the whole thing for the rest of the run.
COMPACTED_RESULT_CHARS = 240


def _compact(messages: list) -> int:
    """Shrink the oldest tool results in place. Returns how many it rewrote.

    Costs NOTHING -- no model call -- which is why it is the first tier and why
    it runs before anything cleverer. On a tool-heavy run the transcript is
    mostly tool output by volume, so cutting the old ones down reclaims most of
    it, and `actions` still carries the one-line record of everything.

    Deliberately lossy in the same way `_summarise_action` is: "you already
    tried this, here is roughly what came back". What it must never touch is
    the seed (the task, the opening prompts) or the recent tail, so the run
    keeps both what it was asked and what it is in the middle of.

    This BOUNDS the transcript on its own -- after compaction a run at the
    default 120-call ceiling holds roughly 8 verbatim results plus a hundred
    240-character stubs, some 15k tokens, well inside every routed window. What
    it does not do is make the dropped bytes retrievable, which is the second
    tier: feeding evicted exchanges through a `kind="context"` TieredQueue
    (agent/memory/) so `recall_memory` can search them. That is worth building
    and is deliberately NOT half-built here -- until it exists the honest thing
    to tell the model is that the result can be produced again, which is true,
    rather than that it can be searched for, which is not.
    """
    rewritten = 0
    protected = len(messages) - KEEP_VERBATIM
    for i, message in enumerate(messages):
        if i >= protected or not isinstance(message, HumanMessage):
            continue
        text = _content_text(message.content)
        if not text.startswith("TOOL RESULT:") or len(text) <= COMPACTED_RESULT_CHARS:
            continue
        messages[i] = HumanMessage(
            text[:COMPACTED_RESULT_CHARS]
            + f"\n... [{len(text) - COMPACTED_RESULT_CHARS} characters of this "
            "result dropped to make room. Run it again if you need the rest.]"
        )
        rewritten += 1
    return rewritten


def _transcript_size(messages: list) -> int:
    return sum(len(_content_text(m.content)) for m in messages)


def _agent_loop(state: AgentState, messages: list, *, mode: str,
                actions: list[str], mode_log: list[str]) -> tuple[str, str, str]:
    """Run one conversation until it answers, pauses, or runs out of budget.

    Returns `(output, why_it_stopped, mode)`. `output` is only ever set from an
    attempted ANSWER -- a FINAL body, or failing that the last unparseable
    reply. A tool call is not a candidate answer, which is the same rule the
    loop this replaces had and the same reason: the ACTION protocol's raw text
    leaking out as an answer is a bug that already happened once.

    `messages` is mutated in place. The caller persists it, so a run that pauses
    for a question or dies on a budget still hands back everything it learned.
    """
    budget = current_budget()
    output = ""
    dead_replies = 0
    last_target: str | None = None
    repeats = 0
    failed_targets: set[str] = set()
    swaps = 0
    did_work_since_swap = True
    llm = ROUTER.chat_model(MODES[mode].task)

    while True:
        if _transcript_size(messages) > LOOP_COMPACT_AT:
            dropped = _compact(messages)
            if dropped:
                logger.info("agent loop: compacted %d old tool result(s)", dropped)
                _emit({"agent": {"board": [f"compacted {dropped} older tool result(s)"]}})

        if budget is not None:
            if budget.spent():
                return output, "budget", mode
            note = budget.wrap_up_once()
            if note:
                messages.append(HumanMessage(note))

        text = _call(llm, messages)
        kind_of_reply, tool_name, body = _parse_worker_reply(text)

        if kind_of_reply == "final":
            return _strip_code_fence(body), "final", mode

        if kind_of_reply == "unparseable":
            dead_replies += 1
            output = text or output
            if dead_replies >= MAX_CONSECUTIVE_DEAD_REPLIES:
                logger.warning(
                    "agent loop: %d unparseable replies in a row -- giving up on "
                    "this attempt rather than spending the rest of the budget",
                    dead_replies,
                )
                return output, "dead", mode
            # An empty reply must not become an empty AIMessage. `_call` returns
            # "" on an empty stream, Anthropic rejects empty text blocks, and in
            # a transcript that never resets one of those would break every
            # later Anthropic call for the rest of the run.
            if text:
                messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue

        dead_replies = 0

        if tool_name == "ask_user":
            question, choices = _parse_ask_user_body(body)
            raise NeedsUserInput(question or "(no question given)", choices)

        if tool_name == "switch_mode":
            messages.append(AIMessage(text))
            mode, swaps, did_work_since_swap = _switch_mode(
                messages, body, mode=mode, swaps=swaps,
                did_work=did_work_since_swap, mode_log=mode_log,
                calls=budget.calls if budget else 0,
            )
            llm = ROUTER.chat_model(MODES[mode].task)
            continue

        did_work_since_swap = True
        dispatch = dispatch_table()
        if tool_name not in dispatch:
            evidence = (
                f"tool {tool_name!r} is not available "
                f"(allowed: {sorted(dispatch)})"
            )
        else:
            result = dispatch[tool_name](body)
            line = _summarise_action(tool_name, body, result)
            actions.append(f"{mode}: {line}")
            _emit({"agent": {"board": [f"{mode}: {line}"]}})
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
            target = _action_target(tool_name, body)
            repeats = repeats + 1 if target == last_target else 1
            last_target = target
            failed_before = target in failed_targets
            if result.returncode != 0:
                failed_targets.add(target)
            if result.returncode != 0 and failed_before:
                evidence += "\n\n" + REPEATED_FAILURE_NOTE.format(target=target)
            elif repeats >= REPEATS_BEFORE_NOTE:
                evidence += "\n\n" + REPEATED_CALL_NOTE.format(
                    n=repeats, tool=tool_name, target=target,
                )
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))


def _switch_mode(messages: list, body: str, *, mode: str, swaps: int,
                 did_work: bool, mode_log: list[str], calls: int) -> tuple[str, int, bool]:
    """Handle one `switch_mode` request. Returns `(mode, swaps, did_work)`.

    Three refusals, cheapest first, and none of them raises -- a refusal is a
    message the model reads and acts on, exactly like a failed tool result.
    """
    want = parse_mode_body(body)
    if want is None:
        messages.append(HumanMessage(
            f"switch_mode: that is not a mode. Choose one of "
            f"{', '.join(mode_names())}. You are still in {mode}."
        ))
        mode_log.append(f"call {calls}: refused (unknown mode)")
        return mode, swaps, did_work

    if want == mode:
        # Deliberately does NOT re-append the guidance: it is already above,
        # and repeating it would grow the prompt every time the model asked
        # for what it already has.
        messages.append(HumanMessage(
            f"You are already in {mode} mode; its guidance is above. Continue."
        ))
        mode_log.append(f"call {calls}: refused (already in {mode})")
        return mode, swaps, did_work

    if not did_work:
        messages.append(HumanMessage(IDLE_SWAP_NOTE.format(last=mode, want=want)))
        mode_log.append(f"call {calls}: refused ({mode} -> {want}, no work between)")
        return mode, swaps, did_work

    if swaps >= MAX_MODE_SWAPS:
        messages.append(HumanMessage(
            f"You have switched modes {swaps} times. Finish in the mode you "
            "are in."
        ))
        mode_log.append(f"call {calls}: refused (swap limit)")
        return mode, swaps, did_work

    reason = mode_reason(body)
    messages.append(_mode_message(want))
    mode_log.append(f"call {calls}: {mode} -> {want}" + (f" ({reason})" if reason else ""))
    _emit({"agent": {"board": [f"switched to {want} mode" + (f" -- {reason}" if reason else "")]}})
    return want, swaps + 1, False


def agent(state: AgentState) -> Command[Literal["evaluator", "ask_user"]]:
    """The one working node. Everything the four specialists did, in one
    conversation that is never thrown away.

    Every exit writes `transcript` back. That is a correctness rule rather than
    an optimisation: a run that pauses on `ask_user` and never persisted its
    conversation would resume having forgotten the work it paused in the middle
    of, which is the failure this whole rewrite exists to remove.
    """
    task_text = state["messages"][-1].content
    mode = state.get("mode") or DEFAULT_MODE
    stored = _revive(state.get("transcript"))
    resuming = bool(stored)

    if resuming:
        messages = [SystemMessage(AGENT_PROMPT), *(
            [SystemMessage(render_note())] if render_note() else []
        ), *stored]
        feedback = state.get("feedback") or ""
        if feedback:
            messages.append(HumanMessage(
                f"EVALUATOR REJECTED THAT:\n{feedback}\n"
                "Fix it. You still have everything above."
            ))
    else:
        messages = _seed_transcript(state, task_text)

    actions: list[str] = []
    mode_log: list[str] = []
    budget = current_budget()

    def carry(**extra) -> dict:
        """Every return path persists the same four things."""
        update = {
            "transcript": _plain(messages),
            "mode": mode,
            "model_calls": budget.calls if budget else state.get("model_calls") or 0,
        }
        if actions:
            update["actions"] = actions
        if mode_log:
            update["mode_log"] = mode_log
        update.update(extra)
        return update

    try:
        output, why, mode = _agent_loop(
            state, messages, mode=mode, actions=actions, mode_log=mode_log,
        )
    except NeedsUserInput as exc:
        return Command(
            update=carry(
                pending_question=exc.question,
                pending_choices=exc.choices,
                asking_role="agent",
                board=["otto is asking you a question"],
            ),
            goto="ask_user",
        )
    except ProviderError as exc:
        # A provider failure is not the agent being wrong. Hand over whatever
        # was already produced rather than losing the run -- the evaluator can
        # judge a partial answer, and an empty one is what the old graph
        # produced here.
        return Command(
            update=carry(
                node="agent",
                node_error=f"agent: {exc}",
                feedback=f"a provider/network failure interrupted the run: {exc}",
                board=[f"otto hit a provider error: {exc}"],
            ),
            goto="evaluator",
        )

    if why == "budget":
        # Straight to the end, not to the evaluator. There is nothing left to
        # pay a judgment with, and handing on would put the run in a loop: the
        # evaluator rejects for lack of evidence, the loop returns immediately
        # because it is still out of budget, and the two bounce until the
        # recursion limit. Answering with what it has is the whole reason
        # exhaustion returns instead of raising.
        return Command(
            update=carry(
                node="agent",
                output=output,
                final_output=output,
                board=["otto ran out of budget -- answering with what it has, unverified"],
            ),
            goto=END,
        )

    board = {
        "final": "otto has an answer",
        "dead": "otto could not produce a usable reply and is handing over what it has",
    }[why]
    return Command(
        update=carry(node="agent", output=output, feedback="", board=[board]),
        goto="evaluator",
    )


def _criteria(llm, task_text: str) -> list[str]:
    """Phase one: the rubric, from the task alone.

    A separate call on purpose. Criteria written while looking at an answer
    are criteria the answer happens to meet, which is the failure mode that
    makes a self-judging loop measure zero.

    Fails soft: a provider hiccup here must not cost the judgment. An empty
    rubric degrades the evaluator to what it was before this change, which is
    worse but not broken.
    """
    try:
        reply = _call(llm, [
            SystemMessage(RUBRIC_PROMPT),
            HumanMessage(f"TASK:\n{task_text}"),
        ])
    except ProviderError as exc:
        logger.warning("rubric generation failed, judging without one: %s", exc)
        return []
    return _parse_rubric(reply)


def evaluator(state: AgentState) -> Command[Literal["agent", "__end__", "ask_user"]]:
    task_text = state["messages"][-1].content
    node = state.get("node") or "agent"
    output = state.get("output") or ""
    judging_plan = False

    llm = ROUTER.chat_model(Task.EVALUATE)
    target, target_note = "ANSWER", "as a finished answer"
    human_label = "ANSWER"
    # The SAME number the loop below is actually run with. It used to be
    # MAX_TOOL_ITERATIONS, so the prompt promised five exchanges and the loop
    # allowed two -- a model told it has budget it does not have will plan to
    # use it.
    # PHASE ONE: what would a correct answer have to contain? Asked from the
    # task alone, before the attempt is visible. This is the only information
    # in the whole judgment that the actor did not produce, and it is why the
    # judgment is worth its calls at all -- see RUBRIC_PROMPT.
    rubric = _criteria(llm, task_text)

    system_prompt = EVALUATOR_PROMPT.format(
        target=target, target_note=target_note, max_iter=MAX_EVALUATOR_ITERATIONS,
        rubric="\n".join(f"- {c}" for c in rubric) or "- the request is satisfied",
    )
    # It used to judge blind: conversation, request, output, nothing else. So
    # "I cannot verify this" came back as a rejection, and a rejection cost a
    # whole extra round. Everything added below already exists and is already
    # bounded -- _actions_block caps at _ACTIONS_SHOWN, the transcript tail is
    # clipped -- so one better-informed judgment costs a fraction of the round
    # it avoids.
    human_body = "\n\n".join(part for part in (
        (f"CONVERSATION SO FAR:\n{_conversation_so_far(state)}"
         if _conversation_so_far(state) else ""),
        f"ORIGINAL REQUEST:\n{task_text}",
        f"{human_label}:\n{output}",
        _actions_block(state),
        (f"MODES USED:\n" + "; ".join(state.get("mode_log") or [])
         if state.get("mode_log") else ""),
        (f"HOW IT FINISHED (the end of otto's own working):\n{_evidence_tail(state)}"
         if _evidence_tail(state) else ""),
        (f"CONTEXT GATHERED:\n{_clip_evidence(state.get('context') or '')}"
         if state.get("context") else ""),
    ) if part)
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    try:
        reply = _tool_loop(llm, messages, max_iterations=MAX_EVALUATOR_ITERATIONS)
    except NeedsUserInput as exc:
        # Seventh refinement (module docstring): the evaluator itself got
        # stuck judging something and needs the person's input to settle
        # it. `output` (still pending judgment) is left untouched.
        return Command(
            update={
                "pending_question": exc.question,
                "pending_choices": exc.choices,
                "asking_role": "evaluator",
                "board": ["evaluator is asking you a question"],
            },
            goto="ask_user",
        )
    except ProviderError as exc:
        # A provider/network failure interrupted the evaluator itself --
        # fifth refinement (module docstring). `output` (whatever is
        # pending judgment) is left untouched; the failure text goes into
        # `feedback` so planner sees it the same way it would see any
        # other specialist's rejected attempt, via _role_body's existing
        # background display.
        what = "plan" if judging_plan else "output"
        return Command(
            update={
                "node_error": f"evaluator: {exc}",
                "feedback": (
                    f"the evaluator was interrupted by a provider/network "
                    f"failure before it could judge {node}'s {what} -- not "
                    f"a real rejection: {exc}"
                ),
                "board": [f"evaluator failed with a provider error: {exc}"],
            },
            goto="agent",
        )
    # _parse_approval defaults to approve=False whenever "APPROVE:" isn't
    # found in `reply` at all (e.g. _tool_loop exhausted on unparseable
    # replies) -- fails CLOSED by construction: an evaluator that never
    # rendered a real verdict is not evidence the answer is fine.
    verdict = _parse_verdict(reply)
    approve, reason = verdict.approved, verdict.reason
    if verdict.criteria[1]:
        reason = f"{verdict.criteria[0]}/{verdict.criteria[1]} criteria met -- {reason}"
    if verdict.blocked and not approve:
        # An environment blocker is not the agent being wrong. Saying so keeps
        # a real obstacle from being counted as a failure -- and from being
        # retried identically, which is what a plain rejection invites.
        reason = f"blocked by the environment, not by the attempt: {reason}"
    # The judgment's own calls count too. Left out, `model_calls` reports what
    # the loop spent rather than what the run spent -- measured live at 4
    # against an actual 9, because the evaluator checks the answer with tools
    # and those are model calls like any other.
    spent = {"model_calls": budget.calls} if (budget := current_budget()) else {}

    if approve:
        return Command(
            update={
                **spent,
                "final_output": output,
                "rejections": 0,
                "board": ["evaluator approved the answer"],
            },
            goto=END,
        )
    rejections = (state.get("rejections") or 0) + 1
    if rejections > MAX_REJECTIONS:
        # Judgment must not eat the whole budget. Past the cap the answer
        # stands, said plainly rather than silently -- an unverified answer the
        # reader is told about beats a run that spent everything re-judging.
        return Command(
            update={
                **spent,
                "final_output": output,
                "rejections": rejections,
                "board": [
                    f"evaluator rejected {node} {rejections} times; answering "
                    "anyway, unverified: " + reason
                ],
            },
            goto=END,
        )
    return Command(
        update={
            **spent,
            "feedback": reason,
            "rejections": rejections,
            "board": [f"evaluator rejected {node}: {reason}"],
        },
        goto="agent",
    )


# --------------------------------------------------------------------------
# ask_user -- seventh refinement (module docstring). The ONLY node in this
# graph that calls interrupt(), and deliberately does nothing else: read
# the pending question off state, pause, write the answer down, hand back
# to whoever asked. See NeedsUserInput's docstring above for why an LLM
# call or a tool dispatch has no business happening in this node -- resume
# re-executes this whole function from the top, so anything with a side
# effect before interrupt() would redo it every time the person answers.
# --------------------------------------------------------------------------

def ask_user(state: AgentState) -> Command[Literal["agent", "evaluator"]]:
    question = state.get("pending_question") or ""
    choices = state.get("pending_choices") or []
    answer = interrupt({"question": question, "choices": choices})

    # Into `context`, NOT `messages` -- every node here (router included)
    # reads state["messages"][-1] as THE TASK for the whole turn;
    # appending onto it would silently replace the actual task the next
    # time anything looked (module docstring). `context` already means
    # "material gathered so far for planner/solver to use" and is already
    # shown to every prompt below via "CONTEXT GATHERED SO FAR:" -- the
    # role that asked sees this Q&A as ordinary background on its next
    # (fresh) attempt, the same way it would see anything finder dug up.
    prior = state.get("context") or ""
    qa = f'you asked: "{question}"\nthe user answered: "{answer}"'
    role = state.get("asking_role") or "agent"
    return Command(
        update={
            "context": f"{prior}\n\n{qa}" if prior else qa,
            "pending_question": None,
            "pending_choices": None,
            "asking_role": None,
            "board": [f"you answered -- {role} is carrying on"],
        },
        goto=role,
    )


g = StateGraph(AgentState)
for _name, _fn in (
    ("agent", agent),
    ("evaluator", evaluator),
    ("ask_user", ask_user),
):
    g.add_node(_name, _fn)
g.add_edge(START, "agent")
# Every other edge is a Command(goto=...) from the node function itself --
# router -> whichever of the five it picks (or a plan step's route_to);
# every specialist -> router OR ask_user (got stuck, NeedsUserInput --
# seventh refinement); evaluator -> router (reject, or plan approved),
# END (final answer approved), or ask_user (same as a specialist);
# ask_user -> whichever specialist/evaluator asked, once answered --
# nothing else to wire here.

app = g.compile(checkpointer=InMemorySaver())
