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
Deliberately still NOT a fix for "the model asks the user a clarifying
question mid-run" (there is no such capability anywhere in this graph
yet, a separate and larger gap the same live test surfaced) -- this only
makes sure the model has what it needs to not HAVE to ask in a case like
"improve above solution", where the answer was one turn away the whole
time.

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
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.pipeline.state import AgentState, PlanStep
from agent.pipeline.tools import TOOL_DISPATCH
from agent.router.llm_provider.base import ProviderError
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
ROLE_NODES = ("planner", "solver", "summarizer", "finder")

#: Everywhere the overseer may dispatch to -- the four specialists plus
#: evaluator itself. Judging something is now a decision the overseer makes
#: explicitly (see module docstring) rather than something automatic.
DISPATCH_TARGETS = ROLE_NODES + ("evaluator",)

#: Specialists a single PLAN STEP may be assigned to (router()'s narrower,
#: deterministic-except-for-this-one-choice step-assignment call) --
#: deliberately excludes "planner" (no re-planning mid-step; abandoning a
#: bad plan for a fresh one is a WHOLE-plan decision the general overseer
#: prompt makes, see module docstring) and "evaluator" (no per-step
#: judgment in this design -- see module docstring).
STEP_TARGETS = ("solver", "summarizer", "finder")

#: Pure infra safety net (see module docstring) -- NOT a business rule.
#: Sized generously: a real converging request is expected to stay well
#: under this; agent/pipeline/run.py's `_config` uses this directly as
#: LangGraph's `recursion_limit`. If a run ever actually hits this, that is
#: a bug (a genuinely non-converging loop) to go fix, not a signal to raise
#: the number further.
_RECURSION_SAFETY_NET = 150

#: Same rationale as the first revision's identical constant: a diffusing
#: (Mercury) call cut off at max_tokens is not a clean prefix, it's an
#: unconverged snapshot, and must be retried rather than accepted.
MAX_DIFFUSION_RETRIES = 3

#: Bound on one node's OWN tool-calling loop (ACTION/execute_TOOL
#: round-trips) before its last reply is used as-is. Unrelated to the
#: (now-removed) overseer retry cap -- this bounds a single node's single
#: turn, not how many turns the overseer may hand out. Applies identically
#: to every role node and the evaluator (_tool_loop is shared by all of
#: them).
MAX_TOOL_ITERATIONS = 5

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

ROUTER_PROMPT = (
    "You are the OVERSEER. You are re-invoked after every step; decide the "
    "single next action given everything accumulated so far (shown below). "
    "Choose exactly one:\n"
    "planner -- break a multi-step or complex task into an ordered plan (a "
    "JSON list of steps) before anyone executes it -- each step then gets "
    "assigned to a specialist one at a time as it runs. Also the right "
    "move on a retry if the feedback below suggests the previous attempt "
    "failed FOR WANT OF A PLAN (it jumped straight to solving something "
    "that actually needed steps worked out first).\n"
    "solver -- work out a concrete answer, writing/running code where that "
    "helps. Use directly for a task simple enough to need no plan and no "
    "lookup; use once a plan and/or context is ready if the task needed "
    "either.\n"
    "summarizer -- the context gathered so far (below) has gotten large or "
    "noisy; condense it before anyone else has to read all of it.\n"
    "finder -- something is missing that has to be looked up before the "
    "task can be solved: local files/a directory/a codebase/GitHub (via "
    "execute_bash: grep, find, git, gh, ls, cat, ...), a knowledge base "
    "(rag), or the open web (web_search).\n"
    "evaluator -- judge whatever was most recently produced -- a PLAN "
    "pending approval, or a candidate FINAL ANSWER -- before deciding "
    "whether the task is actually done. Only choose this when something "
    "below is genuinely pending judgment; never on the very first turn, "
    "and not right after a step that only gathered background material. "
    "(Note: once a plan is approved and running, you are normally NOT "
    "asked this question at all between its steps -- this choice only "
    "comes up before any plan exists, or after a rejection.)\n"
    "\n"
    "Default policy on a REJECTED ATTEMPT below: retry the SAME specialist "
    "that just tried it, with the feedback in view -- do not switch "
    "specialists just because one attempt failed; the specialists don't "
    "overlap, so a different one is not simply an alternate way to do the "
    "same job. The one deliberate exception: if the feedback shows the "
    "task was attempted without a plan and that was the actual problem, "
    "dispatch to planner instead, even though a different specialist tried "
    "it. Learn from which kind of task in THIS run actually needed a plan "
    "and which one didn't, and route similar tasks later in this same run "
    "accordingly.\n"
    "\n"
    "Reply with exactly two lines, and always start the first one with the "
    "literal word \"NODE:\" -- yes, even when your answer is evaluator, "
    "don't just write the bare word:\nNODE: one of planner, solver, "
    "summarizer, finder, evaluator\nWHY: one sentence"
)

#: The narrower call router() makes to assign ONE plan step to whichever
#: specialist should execute it, once a plan is approved and this step's
#: `route_to` is still unset (module docstring, third refinement). The
#: plan itself is already settled -- this is purely an execution
#: assignment, so there's no "evaluator"/"planner" option here at all
#: (STEP_TARGETS).
STEP_ROUTE_PROMPT = (
    "You are the OVERSEER, assigning ONE step of an already-approved plan "
    "to whichever specialist should execute it. The plan itself is settled "
    "-- this is purely an execution assignment for THIS step. Choose "
    "exactly one:\n"
    "solver -- this step needs a concrete answer worked out, writing/"
    "running code where that helps.\n"
    "summarizer -- this step needs the context gathered so far condensed "
    "or rewritten, not something new solved or looked up.\n"
    "finder -- this step needs something looked up first: local files/a "
    "directory/a codebase/GitHub, a knowledge base, or the open web.\n"
    "\n"
    "Reply with exactly two lines, and always start the first one with the "
    "literal word \"NODE:\":\nNODE: one of solver, summarizer, finder\n"
    "WHY: one sentence"
)

PLANNER_PROMPT = (
    "You are the PLANNER. Break the request below into an ordered list of "
    "concrete, executable steps for OTHER specialists to carry out -- you "
    "do not execute any step yourself, and you do not decide who executes "
    "each step (the overseer assigns that later, one step at a time, as "
    "each one runs). You may check your reasoning with a tool: reply with "
    "exactly\nACTION: <execute_python|execute_bash|web_search|rag|"
    "complete_code|predict_edit>\nCODE:\n<input for that tool -- "
    "complete_code: prefix code, optionally then a line \"---SUFFIX---\" "
    "and trailing code; predict_edit: code, optionally with a <|cursor|> "
    "marker, no instruction -- it only predicts the next edit>\nand you "
    "will be shown the result, then you can continue. When you are done, "
    "reply with exactly\nFINAL:\n<a JSON array of steps, each an object "
    "with exactly one key \"task\" holding that step's description -- "
    "e.g. [{{\"task\": \"write the core function\"}}, {{\"task\": \"add "
    "input validation\"}}] -- nothing else, no markdown code fence, no "
    "explanation, and no \"route_to\" (that's decided later, per step, "
    "not by you)>\nYou have at most {max_iter} exchanges before your last "
    "reply is used as-is."
)
SOLVER_PROMPT = (
    "You are the SOLVER. Work out a concrete answer to the request below, "
    "writing and running code where that helps you check it. Reply with "
    "exactly\nACTION: <execute_python|execute_bash|web_search|rag|"
    "complete_code|predict_edit>\nCODE:\n<input for that tool -- "
    "complete_code: prefix code, optionally then a line \"---SUFFIX---\" "
    "and trailing code; predict_edit: code, optionally with a <|cursor|> "
    "marker, no instruction -- it only predicts the next edit>\nand you "
    "will be shown the result, then you can continue. When you are done, "
    "reply with exactly\nFINAL:\n<the complete answer, nothing else -- no "
    "markdown code fences, no explanation>\nYou have at most {max_iter} "
    "exchanges before your last reply is used as-is."
)
SUMMARIZER_PROMPT = (
    "You are the SUMMARIZER. Condense or rewrite the given content below to "
    "satisfy the request -- you are not looking anything new up or solving "
    "a new problem. You may still use a tool if it helps verify something: "
    "reply with exactly\nACTION: <execute_python|execute_bash|web_search|"
    "rag|complete_code|predict_edit>\nCODE:\n<input for that tool -- "
    "complete_code: prefix code, optionally then a line \"---SUFFIX---\" "
    "and trailing code; predict_edit: code, optionally with a <|cursor|> "
    "marker, no instruction -- it only predicts the next edit>\nand you "
    "will be shown the result, then you can continue. When you are done, "
    "reply with exactly\nFINAL:\n<the summary, nothing else -- no markdown "
    "code fences, no explanation>\nYou have at most {max_iter} exchanges "
    "before your last reply is used as-is."
)
FINDER_PROMPT = (
    "You are the FINDER. Look up whatever the request below needs before "
    "answering. Prefer, in this order: execute_bash for anything on the "
    "local system -- a directory, a codebase, git history/blame, or GitHub "
    "via the gh CLI if it's on PATH (grep, find, ls, cat, git, gh, ...); "
    "rag for a knowledge base; web_search for the open web (rag and "
    "web_search are both stubbed today and will tell you so -- if that "
    "happens, fall back to your own knowledge and say plainly in your "
    "answer that you could not verify it). Reply with exactly\nACTION: "
    "<execute_python|execute_bash|web_search|rag|complete_code|"
    "predict_edit>\nCODE:\n<input for that tool -- complete_code: prefix "
    "code, optionally then a line \"---SUFFIX---\" and trailing code; "
    "predict_edit: code, optionally with a <|cursor|> marker, no "
    "instruction -- it only predicts the next edit>\nand you will be shown "
    "the result, then you can continue. When you are done, reply with "
    "exactly\nFINAL:\n<what you found, nothing else -- no markdown code "
    "fences, no explanation>\nYou have at most {max_iter} exchanges before "
    "your last reply is used as-is."
)
#: Used only when the SAME specialist that made the rejected attempt is the
#: one retrying it (state["node"] == role at dispatch time) -- see
#: _run_role. Shared by all four roles (parameterized by {role_upper})
#: since the instruction is identical regardless of which specialist is
#: revising its own work.
ROLE_REVISE_PROMPT = (
    "You are the {role_upper}. Your previous attempt at the task below was "
    "rejected by the evaluator -- see your previous attempt and the "
    "evaluator's feedback for what to fix or avoid. Produce a better "
    "answer, reusing anything from the previous attempt that was actually "
    "fine. Same reply format as before: ACTION/CODE to use a tool, or "
    "FINAL: when done. You have at most {max_iter} exchanges before your "
    "last reply is used as-is."
)
#: One shared evaluator prompt, parameterized by what's being judged --
#: {target} is "PLAN" or "<ROLE> OUTPUT", {target_note} distinguishes
#: "would this plan work if followed" from "is this actually a finished
#: answer" (see evaluator()'s dual-mode dispatch below).
EVALUATOR_PROMPT = (
    "Judge whether the {target} below actually satisfies the original "
    "request -- {target_note}. You may check your judgment with a tool: "
    "reply with exactly\nACTION: <execute_python|execute_bash|web_search|"
    "rag|complete_code|predict_edit>\nCODE:\n<input for that tool -- "
    "complete_code: prefix code, optionally then a line \"---SUFFIX---\" "
    "and trailing code; predict_edit: code, optionally with a <|cursor|> "
    "marker, no instruction -- it only predicts the next edit>\nand you "
    "will be shown the result, then you can continue. When you are done, "
    "reply with exactly\nFINAL:\nAPPROVE: yes or no\nWHY: one sentence\n"
    "You have at most {max_iter} exchanges before your last reply is used "
    "as-is."
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
    current = llm
    for attempt in range(MAX_DIFFUSION_RETRIES):
        reply = None
        for chunk in current.stream(messages):
            reply = chunk if reply is None else reply + chunk
        if reply is None:
            return ""

        finish_reason = (reply.response_metadata or {}).get("finish_reason")
        if not (getattr(current, "diffusing", False) and finish_reason == "length"):
            return _content_text(reply.content)

        if attempt == MAX_DIFFUSION_RETRIES - 1:
            raise ProviderError(
                f"inception: {current.model} truncated a diffusing response at "
                f"max_tokens={current.max_tokens!r} on every one of "
                f"{MAX_DIFFUSION_RETRIES} attempts -- refusing to hand an "
                f"unconverged diffusion snapshot to the rest of the graph"
            )
        bumped = max(current.max_tokens or 1024, 1024) * 2
        logger.warning(
            "inception: %s truncated a diffusing response at max_tokens=%r "
            "(attempt %d/%d) -- retrying at max_tokens=%d",
            current.model, current.max_tokens, attempt + 1,
            MAX_DIFFUSION_RETRIES, bumped,
        )
        current = current.model_copy(update={"max_tokens": bumped})
    return ""  # unreachable -- the loop always returns or raises


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
        code = text.split("CODE:", 1)[-1].strip() if "CODE:" in text else ""
        return "action", action_line, code
    if "FINAL:" in text:
        body = text.split("FINAL:", 1)[-1].strip()
        if body:
            return "final", "", body
        return "unparseable", "", text.strip()
    return "unparseable", "", text.strip()


def _tool_loop(llm, messages: list) -> str:
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
    output = ""
    for _ in range(MAX_TOOL_ITERATIONS):
        text = _call(llm, messages)
        kind_of_reply, tool_name, body = _parse_worker_reply(text)

        if kind_of_reply == "final":
            return _strip_code_fence(body)

        if kind_of_reply == "unparseable":
            output = text  # fallback if the loop is exhausted here
            messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue

        # ACTION -- deliberately does NOT touch `output` (see docstring).
        if tool_name not in TOOL_DISPATCH:
            evidence = (
                f"tool {tool_name!r} is not available "
                f"(allowed: {sorted(TOOL_DISPATCH)})"
            )
        else:
            result = TOOL_DISPATCH[tool_name](body)
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))
    return output


def _extract_node(text: str, targets: tuple[str, ...]) -> tuple[str | None, str]:
    """Try to find a NODE:/WHY: pair in `text`, restricted to `targets`.
    Returns `(None, "")` if nothing in `targets` was found at all -- this
    function bakes in no fallback, so a caller can decide for itself what
    "nothing found" means: _parse_router defaults to "solver"; _decide
    (below) retries the LLM call first and only falls back once retries
    are exhausted.

    Tolerates one specific malformed shape, observed live (2026-09-10): the
    model sometimes drops the literal "NODE:" label and replies with just
    the bare target name on its own line (most often when the answer is
    "evaluator", as if it read past its own format instruction) --
    "evaluator\\nWHY: ..." instead of "NODE: evaluator\\nWHY: ...". Only a
    line that is EXACTLY one of `targets` (whole line, stripped) counts --
    never a substring match, so a WHY sentence that happens to mention
    "solver" is never mistaken for a NODE: line.
    """
    node = None
    why = ""
    lines = text.splitlines()
    for line in lines:
        upper = line.upper()
        if upper.startswith("NODE:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in targets:
                node = candidate
        elif upper.startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    if node is None:
        for line in lines:
            candidate = line.strip().lower()
            if candidate in targets:
                node = candidate
                break
    return node, why


def _parse_router(text: str, targets: tuple[str, ...] = DISPATCH_TARGETS) -> tuple[str, str]:
    """Parse a NODE:/WHY: reply against `targets` (DISPATCH_TARGETS by
    default; router()'s step-assignment call passes STEP_TARGETS instead) --
    a thin wrapper over _extract_node that supplies the ONE-SHOT fallback:
    an unparseable or unrecognised NODE: value fails toward "solver" (or
    `targets[0]` if "solver" isn't even in this call's target set) rather
    than crashing or leaving `node` unset; the WHY string says so
    explicitly when that happens, visible on the board and to the
    dispatched node's own prompt. router() itself doesn't call this
    directly anymore -- it calls _decide(), which retries before ever
    reaching this fallback -- but it's kept as the single-shot primitive
    other callers and tests reason about.
    """
    node, why = _extract_node(text, targets)
    if node is None:
        fallback = "solver" if "solver" in targets else targets[0]
        return fallback, f"could not parse a NODE: line from {text.strip()[:200]!r}, defaulting to {fallback}"
    return node, why


#: How many times _decide() retries the SAME small router call in place
#: before giving up and falling back to _parse_router's "defaulting to
#: solver" safety net. Observed live (2026-09-10): the model occasionally
#: rambles instead of the required two-line format -- most often the
#: first time a PENDING OUTPUT shows up in its prompt (i.e. right when it
#: should say "evaluator" for the first time) -- rather than genuinely
#: being unsure. A cheap retry of just this one call recovers that far
#: more often than silently burning an entire extra specialist round on
#: the fallback would.
_MAX_ROUTER_PARSE_RETRIES = 2


def _router_retry_feedback(targets: tuple[str, ...]) -> str:
    return (
        "Reply again using EXACTLY two lines and nothing else -- no "
        "preamble, no explanation outside WHY:. Start the first line with "
        f"the literal word \"NODE:\", followed by one of: {', '.join(targets)}."
    )


def _decide(system_prompt: str, human_body: str, *, targets: tuple[str, ...]) -> tuple[str, str]:
    """Make one router-model call and extract (node, why) from it,
    retrying the SAME call in place (see _MAX_ROUTER_PARSE_RETRIES) if the
    reply doesn't parse at all, before falling back to _parse_router's
    fallback. Shared by both of router()'s LLM calls (the general 5-way
    decision and the narrower per-step assignment) -- identical retry
    logic either way, just different prompts/targets.
    """
    llm = ROUTER.chat_model(Task.CHAT_FAST, temperature=0.2)
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    text = _call(llm, messages)
    node, why = _extract_node(text, targets)
    attempts = 0
    while node is None and attempts < _MAX_ROUTER_PARSE_RETRIES:
        attempts += 1
        messages.append(AIMessage(text))
        messages.append(HumanMessage(_router_retry_feedback(targets)))
        text = _call(llm, messages)
        node, why = _extract_node(text, targets)
    if node is None:
        fallback = "solver" if "solver" in targets else targets[0]
        return fallback, (
            f"could not parse a NODE: line after {attempts + 1} attempt(s), "
            f"last reply {text.strip()[:200]!r}, defaulting to {fallback}"
        )
    return node, why


def _parse_plan_steps(text: str) -> list[PlanStep]:
    """Parse the planner's approved FINAL body into a list of PlanStep
    dicts -- `{"task": str, "route_to": None, "output": None}` each, both
    of the None fields filled in later (route_to by router()'s step
    assignment, output by whichever specialist executes that step).

    PLANNER_PROMPT asks for a JSON array of `{"task": ...}` objects.
    Tolerant of a reply that isn't clean JSON -- stray prose around it, a
    fence _strip_code_fence didn't fully catch -- by also trying just the
    substring between the first "[" and the last "]". If nothing parses as
    a non-empty list of tasks at all, falls back to treating the WHOLE raw
    reply as ONE step rather than dropping the plan or crashing: a
    malformed-but-present plan should still be executable, just as one big
    step instead of several small ones.
    """
    raw = text.strip()
    candidates = [raw]
    start, end = raw.find("["), raw.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidates.append(raw[start:end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, list) and parsed:
            steps: list[PlanStep] = []
            for item in parsed:
                if isinstance(item, dict) and str(item.get("task") or "").strip():
                    steps.append({"task": str(item["task"]).strip(), "route_to": None, "output": None})
                elif isinstance(item, str) and item.strip():
                    steps.append({"task": item.strip(), "route_to": None, "output": None})
            if steps:
                return steps
    return [{"task": raw, "route_to": None, "output": None}]


def _next_pending_step_index(plan: list[PlanStep]) -> int | None:
    """The first step whose `output` is still None -- "not yet run" (an
    empty string IS a completed, if useless, result -- see PlanStep's own
    docstring in state.py). None if every step has already run.
    """
    for i, step in enumerate(plan):
        if step.get("output") is None:
            return i
    return None


def _format_plan(plan: list[PlanStep]) -> str:
    """Render a plan for display inside a human message -- to the overseer
    (deciding what's next), a role node (showing it the whole plan for
    context while it executes one step of it), or as part of a rejected
    attempt's background. Not meant to round-trip; purely readable text.
    """
    lines = []
    for i, step in enumerate(plan, start=1):
        if step.get("output") is not None:
            status = f"done, by {step.get('route_to')}"
        elif step.get("route_to"):
            status = f"assigned to {step['route_to']}, running"
        else:
            status = "not yet assigned"
        lines.append(f"{i}. [{status}] {step['task']}")
        if step.get("output") is not None:
            lines.append(f"   result: {step['output']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Shared human-message-body builders -- both the overseer and the role
# nodes show the same accumulated state (context/plan/pending output or
# feedback), just framed for their own purpose.
# --------------------------------------------------------------------------

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


def _router_body(state: AgentState, task_text: str) -> str:
    parts = []
    history = _conversation_so_far(state)
    if history:
        parts.append(f"CONVERSATION SO FAR:\n{history}")
    parts.append(f"TASK:\n{task_text}")
    context = state.get("context") or ""
    if context:
        parts.append(f"CONTEXT GATHERED SO FAR:\n{context}")
    plan = state.get("plan")
    if plan:
        parts.append(f"PLAN:\n{_format_plan(plan)}")
    feedback = state.get("feedback") or ""
    output = state.get("output") or ""
    node = state.get("node")
    if feedback:
        parts.append(
            f"REJECTED ATTEMPT (by {node}):\n{output}\n\n"
            f"EVALUATOR FEEDBACK:\n{feedback}"
        )
    elif output and node:
        parts.append(f"PENDING OUTPUT (from {node}, not yet judged):\n{output}")
    return "\n\n".join(parts)


def _step_route_body(state: AgentState, task_text: str, plan: list[PlanStep], idx: int) -> str:
    parts = []
    history = _conversation_so_far(state)
    if history:
        parts.append(f"CONVERSATION SO FAR:\n{history}")
    parts += [
        f"TASK:\n{task_text}",
        f"PLAN:\n{_format_plan(plan)}",
        f"STEP TO ASSIGN (step {idx + 1}):\n{plan[idx]['task']}",
    ]
    context = state.get("context") or ""
    if context:
        parts.append(f"CONTEXT GATHERED SO FAR:\n{context}")
    return "\n\n".join(parts)


def _role_body(state: AgentState, task_text: str, *, role: str, revising: bool, executing_step: bool, active_step: int | None) -> str:
    parts = []
    history = _conversation_so_far(state)
    if history:
        parts.append(f"CONVERSATION SO FAR:\n{history}")
    parts.append(f"TASK:\n{task_text}")
    plan = state.get("plan")
    if plan:
        parts.append(f"PLAN:\n{_format_plan(plan)}")
    if executing_step and active_step is not None:
        parts.append(f"YOUR CURRENT STEP (step {active_step + 1}):\n{plan[active_step]['task']}")
    context = state.get("context") or ""
    if context:
        parts.append(f"CONTEXT GATHERED SO FAR:\n{context}")
    feedback = state.get("feedback") or ""
    if feedback:
        previous_node = state.get("node") or "a specialist"
        previous_output = state.get("output") or ""
        if revising:
            parts.append(
                f"YOUR PREVIOUS ATTEMPT (rejected):\n{previous_output}\n\n"
                f"EVALUATOR FEEDBACK -- fix this:\n{feedback}"
            )
        else:
            # A different specialist's rejected attempt, shown as
            # background only -- e.g. the overseer just escalated from
            # solver to planner because solving without a plan failed.
            parts.append(
                f"PREVIOUS ATTEMPT BY {previous_node.upper()} (rejected):\n"
                f"{previous_output}\n\n"
                f"EVALUATOR FEEDBACK (why it was rejected):\n{feedback}"
            )
    return "\n\n".join(parts)


# --------------------------------------------------------------------------
# router -- the overseer. Re-invoked after every node; one call, no tools
# (two calls, on the two paths described in the module docstring: the
# general 5-way decision, or a narrower per-step assignment while an
# approved plan is actively executing).
# --------------------------------------------------------------------------

def router(state: AgentState) -> Command[Literal["planner", "solver", "summarizer", "finder", "evaluator"]]:
    round_ = state["round"] + 1
    task_text = state["messages"][-1].content
    plan = state.get("plan")
    feedback = state.get("feedback") or ""
    node_error = state.get("node_error") or ""

    if node_error:
        # A provider/network failure interrupted whoever just ran -- a
        # specialist mid-attempt (possibly mid-plan-step), or the
        # evaluator itself (module docstring, fifth refinement).
        # Deterministic, no LLM call: an LLM call is exactly what just
        # failed, so asking another one to decide what to do about it
        # would be both ironic and just as likely to fail again. Discards
        # any active plan the same way an ordinary rejection-driven
        # escalation to planner already does -- a plan whose own step
        # runner just blew up cannot simply resume where it left off.
        update: dict = {
            "round": round_,
            "node_error": "",
            "board": [f"round {round_}: overseer -- provider failure, escalating to planner ({node_error})"],
        }
        if plan is not None:
            update["plan"] = None
            update["active_step"] = None
        return Command(update=update, goto="planner")

    if isinstance(plan, list) and plan and not feedback:
        # An approved plan is active and nothing is currently rejected --
        # executing it is assignment-driven, not a fresh 5-way judgment
        # call every round (module docstring, third refinement).
        idx = _next_pending_step_index(plan)
        if idx is None:
            # Every step has run -- straight to evaluator for the whole-
            # answer judgment. Unambiguous once the plan says "done": no
            # LLM call needed to make this decision.
            return Command(
                update={
                    "round": round_,
                    "board": [f"round {round_}: overseer -- plan complete, dispatching to evaluator"],
                },
                goto="evaluator",
            )
        step = plan[idx]
        if step.get("route_to"):
            # Defensive/idempotent: shouldn't normally recur, since a
            # Command's goto hands off immediately after route_to is set.
            return Command(
                update={
                    "round": round_,
                    "active_step": idx,
                    "board": [f"round {round_}: overseer -- step {idx + 1} already assigned to {step['route_to']}"],
                },
                goto=step["route_to"],
            )
        route_to, why = _decide(STEP_ROUTE_PROMPT, _step_route_body(state, task_text, plan, idx), targets=STEP_TARGETS)

        new_plan = [dict(s) for s in plan]
        new_plan[idx] = {**new_plan[idx], "route_to": route_to}
        return Command(
            update={
                "round": round_,
                "plan": new_plan,
                "active_step": idx,
                "board": [f"round {round_}: overseer routed step {idx + 1} ({step['task']}) to {route_to} ({why})"],
            },
            goto=route_to,
        )

    # No active plan-with-pending-steps -- either no plan exists yet, or a
    # rejection just happened and needs real judgment (retry the same
    # specialist by default, or escalate to planner). The original 5-way
    # overseer decision.
    node, why = _decide(ROUTER_PROMPT, _router_body(state, task_text), targets=DISPATCH_TARGETS)

    if node == "evaluator" and not (state.get("output") or "").strip():
        # Defensive net, same "fail toward the safe default" spirit as
        # _parse_router's own fallback: nothing is actually pending
        # judgment (nothing has run yet, or a plan was just approved and
        # cleared) -- evaluator would have nothing to judge.
        why = f"picked evaluator with nothing pending judgment yet ({why}) -- defaulting to solver"
        node = "solver"

    update: dict = {
        "round": round_,
        "board": [f"round {round_}: overseer dispatched to {node} ({why})"],
    }
    if node == "planner" and plan is not None:
        # Escalating to a fresh plan while an old one is still hanging
        # around (e.g. the FINAL answer was rejected after every step ran,
        # and the whole plan -- not just its last step -- looks like the
        # problem) -- start over rather than leaving stale step state
        # (a completed plan whose last step is about to be overwritten by
        # an unrelated fresh planning pass) behind.
        update["plan"] = None
        update["active_step"] = None

    return Command(update=update, goto=node)


# --------------------------------------------------------------------------
# The four specialists -- one shared implementation (_run_role), four thin
# named wrappers (LangGraph nodes need their own registered function/name).
# Every one of them now returns to "router" (the overseer decides what
# happens next), not straight to "evaluator".
# --------------------------------------------------------------------------

def _run_role(
    state: AgentState,
    *,
    role: str,
    task: Task,
    temperature: float,
    prompt: str,
    context_op: Literal["append", "replace"] | None = None,
) -> Command[Literal["router"]]:
    plan = state.get("plan")
    active_step = state.get("active_step")
    executing_step = isinstance(plan, list) and isinstance(active_step, int) and 0 <= active_step < len(plan)

    task_text = state["messages"][-1].content
    feedback = state.get("feedback") or ""
    # Revising MY OWN rejected attempt vs a fresh attempt (even if a
    # DIFFERENT specialist's rejected attempt or gathered context exists --
    # see _role_body's revising=False branch for that case).
    revising = bool(feedback) and state.get("node") == role

    llm = ROUTER.chat_model(task, temperature=temperature)
    system_prompt = (
        ROLE_REVISE_PROMPT.format(role_upper=role.upper(), max_iter=MAX_TOOL_ITERATIONS)
        if revising else prompt.format(max_iter=MAX_TOOL_ITERATIONS)
    )
    human_body = _role_body(state, task_text, role=role, revising=revising, executing_step=executing_step, active_step=active_step)
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    try:
        output = _tool_loop(llm, messages)
    except ProviderError as exc:
        # A provider/network failure interrupted this attempt before it
        # produced anything -- fifth refinement (module docstring).
        # Deliberately does NOT touch state["output"]/"plan"/"context": if
        # this was executing a plan step, the step stays exactly as it
        # was (still pending) -- router() discards the whole plan on the
        # node_error branch anyway, so there is nothing to write back.
        return Command(
            update={
                "node_error": f"{role}: {exc}",
                "feedback": (
                    f"the previous attempt by {role} was interrupted by a "
                    f"provider/network failure before it produced anything "
                    f"-- not rejected by the evaluator: {exc}"
                ),
                "board": [f"{role} failed with a provider error (round {state['round']}): {exc}"],
            },
            goto="router",
        )

    update: dict = {
        "output": output,
        "node": role,
        # This attempt hasn't been judged yet -- any feedback it was shown
        # has now been consumed (either fixed, or noted as background).
        "feedback": "",
        "board": [f"{role} produced an output (round {state['round']})"],
    }

    if executing_step:
        # Write back into THIS step (not just the flat state["output"]) so
        # the overseer's _next_pending_step_index sees it as done, and so a
        # later revision of this same step (after a whole-answer rejection)
        # overwrites the right slot instead of leaving a stale one behind.
        new_plan = [dict(s) for s in plan]
        new_plan[active_step] = {**new_plan[active_step], "output": output}
        update["plan"] = new_plan
        prior = state.get("context") or ""
        step_label = f"step {active_step + 1} ({role}): {plan[active_step]['task']}\n-> {output}"
        # Always APPEND while executing a plan step, regardless of this
        # role's own normal context_op (e.g. summarizer's usual "replace")
        # -- overwriting earlier steps' results mid-plan would defeat the
        # whole point of sequencing them.
        update["context"] = f"{prior}\n\n{step_label}" if prior else step_label
    elif context_op == "append":
        prior = state.get("context") or ""
        update["context"] = f"{prior}\n\n{output}" if prior else output
    elif context_op == "replace":
        update["context"] = output

    return Command(update=update, goto="router")


def planner(state: AgentState) -> Command[Literal["router"]]:
    return _run_role(state, role="planner", task=Task.PLAN, temperature=0.4, prompt=PLANNER_PROMPT)


def solver(state: AgentState) -> Command[Literal["router"]]:
    return _run_role(state, role="solver", task=Task.REASON, temperature=0.5, prompt=SOLVER_PROMPT)


def summarizer(state: AgentState) -> Command[Literal["router"]]:
    return _run_role(
        state, role="summarizer", task=Task.SUMMARIZE, temperature=0.2,
        prompt=SUMMARIZER_PROMPT, context_op="replace",
    )


def finder(state: AgentState) -> Command[Literal["router"]]:
    # No dedicated Task route for "look something up" exists yet -- CHAT_FAST
    # (fast turnaround, light reasoning) fits a node whose real work is
    # supposed to be the tool call, not deliberation. Revisit if/when
    # web_search/rag stop being stubs and finder's actual job gets harder.
    return _run_role(
        state, role="finder", task=Task.CHAT_FAST, temperature=0.3,
        prompt=FINDER_PROMPT, context_op="append",
    )


# --------------------------------------------------------------------------
# evaluator -- dual-mode: judges a PLAN (state["node"] == "planner") or a
# candidate FINAL ANSWER (anything else), with the same tool access every
# role node has. Approving a plan PARSES it into a structured step list
# (_parse_plan_steps) and hands back to the overseer (there's more work
# left -- the plan hasn't been executed yet); approving a final answer ends
# the run. Rejecting either always goes back to the overseer -- no
# exhaustion branch (see module docstring).
# --------------------------------------------------------------------------

def evaluator(state: AgentState) -> Command[Literal["router", "__end__"]]:
    task_text = state["messages"][-1].content
    node = state.get("node") or "solver"
    output = state.get("output") or ""
    judging_plan = node == "planner"

    llm = ROUTER.chat_model(Task.REASON, temperature=0.0)
    if judging_plan:
        target, target_note = "PLAN", (
            "a properly ordered, complete JSON list of executable steps "
            "that would satisfy the request if followed -- not whether "
            "the task is already done"
        )
        human_label = f"PLAN (from {node}, should be a JSON array of steps)"
    else:
        target, target_note = f"{node.upper()} OUTPUT", "as a finished answer"
        human_label = f"{node.upper()} OUTPUT"
    system_prompt = EVALUATOR_PROMPT.format(target=target, target_note=target_note, max_iter=MAX_TOOL_ITERATIONS)
    history = _conversation_so_far(state)
    human_body = (
        (f"CONVERSATION SO FAR:\n{history}\n\n" if history else "")
        + f"ORIGINAL REQUEST:\n{task_text}\n\n{human_label}:\n{output}"
    )
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    try:
        reply = _tool_loop(llm, messages)
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
                "board": [f"evaluator failed with a provider error (round {state['round']}): {exc}"],
            },
            goto="router",
        )
    # _parse_approval defaults to approve=False whenever "APPROVE:" isn't
    # found in `reply` at all (e.g. _tool_loop exhausted on unparseable
    # replies) -- fails CLOSED by construction: an evaluator that never
    # rendered a real verdict is not evidence the answer is fine.
    approve, reason = _parse_approval(reply)

    if approve and judging_plan:
        steps = _parse_plan_steps(output)
        return Command(
            update={
                "plan": steps,
                "active_step": None,
                "output": None,
                "feedback": "",
                "board": [f"evaluator approved {node}'s plan ({len(steps)} step(s)) (round {state['round']})"],
            },
            goto="router",
        )
    if approve:
        return Command(
            update={"final_output": output, "board": [f"evaluator approved {node}'s answer (round {state['round']})"]},
            goto=END,
        )
    return Command(
        update={"feedback": reason, "board": [f"evaluator rejected {node}: {reason}"]},
        goto="router",
    )


g = StateGraph(AgentState)
for _name, _fn in (
    ("router", router),
    ("planner", planner),
    ("solver", solver),
    ("summarizer", summarizer),
    ("finder", finder),
    ("evaluator", evaluator),
):
    g.add_node(_name, _fn)
g.add_edge(START, "router")
# Every other edge is a Command(goto=...) from the node function itself --
# router -> whichever of the five it picks (or a plan step's route_to);
# every specialist -> router; evaluator -> router (reject, or plan
# approved) or END (final answer approved) -- nothing else to wire here.

app = g.compile(checkpointer=InMemorySaver())
