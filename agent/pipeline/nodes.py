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
    only decides who runs next. finder APPENDS its output onto `context`;
    summarizer REPLACES `context` with its (condensed) output; planner and
    solver leave `context` untouched. See agent/pipeline/state.py for the
    full field-by-field reasoning.

  * evaluator is dual-mode now, not single-mode: it judges a PLAN
    (`state["node"] == "planner"`) or a candidate FINAL ANSWER (anything
    else) -- different question, different prompt. Approving a plan sets
    `state["plan"]` and returns to the overseer (there is more work left --
    the plan hasn't been executed yet); approving a final answer sets
    `state["final_output"]` and ends the run. Rejecting either always
    returns to the overseer with the reason -- there is no longer an
    exhaustion branch that gives up after N rounds (removed along with
    MAX_DISPATCH_ROUNDS, see above).

Deliberately out of scope for this revision, same as the first:
  - web_search and rag are still STUBBED (tools.py).
  - No domain-specific verification beyond the evaluator's own tool access.

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

import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.pipeline.state import AgentState
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
    "planner -- break a multi-step or complex task into an ordered plan "
    "before anyone executes it. Also the right move on a retry if the "
    "feedback below suggests the previous attempt failed FOR WANT OF A "
    "PLAN (it jumped straight to solving something that actually needed "
    "steps worked out first).\n"
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
    "and not right after a step that only gathered background material.\n"
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
    "Reply with exactly two lines:\nNODE: one of planner, solver, "
    "summarizer, finder, evaluator\nWHY: one sentence"
)

PLANNER_PROMPT = (
    "You are the PLANNER. Break the request below into a clear, ordered "
    "plan or set of steps -- you do not execute the plan yourself. You may "
    "check your reasoning with a tool: reply with exactly\nACTION: "
    "<execute_python|execute_bash|web_search|rag|complete_code|"
    "predict_edit>\nCODE:\n<input for that tool -- complete_code: prefix "
    "code, optionally then a line \"---SUFFIX---\" and trailing code; "
    "predict_edit: code, optionally with a <|cursor|> marker, no "
    "instruction -- it only predicts the next edit>\nand you will be shown "
    "the result, then you can continue. When you are done, reply with "
    "exactly\nFINAL:\n<the plan, nothing else -- no markdown code fences, "
    "no explanation>\nYou have at most {max_iter} exchanges before your "
    "last reply is used as-is."
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


def _parse_router(text: str) -> tuple[str, str]:
    """Parse router()'s NODE:/WHY: reply. An unparseable or unrecognised
    NODE: value fails toward "solver" -- the most general-purpose
    specialist -- rather than crashing or leaving `node` unset; the WHY
    string says so explicitly when that happens, visible on the board and
    to the dispatched node's own prompt.
    """
    node = None
    why = ""
    for line in text.splitlines():
        upper = line.upper()
        if upper.startswith("NODE:"):
            candidate = line.split(":", 1)[1].strip().lower()
            if candidate in DISPATCH_TARGETS:
                node = candidate
        elif upper.startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    if node is None:
        return "solver", f"could not parse a NODE: line from {text.strip()[:200]!r}, defaulting to solver"
    return node, why


# --------------------------------------------------------------------------
# Shared human-message-body builders -- both the overseer and the role
# nodes show the same accumulated state (context/plan/pending output or
# feedback), just framed for their own purpose.
# --------------------------------------------------------------------------

def _router_body(state: AgentState, task_text: str) -> str:
    parts = [f"TASK:\n{task_text}"]
    context = state.get("context") or ""
    if context:
        parts.append(f"CONTEXT GATHERED SO FAR:\n{context}")
    plan = state.get("plan")
    if plan:
        parts.append(f"APPROVED PLAN:\n{plan}")
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


def _role_body(state: AgentState, task_text: str, *, role: str, revising: bool) -> str:
    parts = [f"TASK:\n{task_text}"]
    context = state.get("context") or ""
    if context:
        parts.append(f"CONTEXT GATHERED SO FAR:\n{context}")
    plan = state.get("plan")
    if plan:
        parts.append(f"APPROVED PLAN:\n{plan}")
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
# router -- the overseer. Re-invoked after every node; one call, no tools.
# --------------------------------------------------------------------------

def router(state: AgentState) -> Command[Literal["planner", "solver", "summarizer", "finder", "evaluator"]]:
    round_ = state["round"] + 1
    task_text = state["messages"][-1].content

    llm = ROUTER.chat_model(Task.CHAT_FAST, temperature=0.2)
    messages = [SystemMessage(ROUTER_PROMPT), HumanMessage(_router_body(state, task_text))]
    text = _call(llm, messages)
    node, why = _parse_router(text)

    if node == "evaluator" and not (state.get("output") or "").strip():
        # Defensive net, same "fail toward the safe default" spirit as
        # _parse_router's own fallback: nothing is actually pending
        # judgment (nothing has run yet, or a plan was just approved and
        # cleared) -- evaluator would have nothing to judge.
        why = f"picked evaluator with nothing pending judgment yet ({why}) -- defaulting to solver"
        node = "solver"

    return Command(
        update={
            "round": round_,
            "board": [f"round {round_}: overseer dispatched to {node} ({why})"],
        },
        goto=node,
    )


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
    human_body = _role_body(state, task_text, role=role, revising=revising)
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    output = _tool_loop(llm, messages)

    update: dict = {
        "output": output,
        "node": role,
        # This attempt hasn't been judged yet -- any feedback it was shown
        # has now been consumed (either fixed, or noted as background).
        "feedback": "",
        "board": [f"{role} produced an output (round {state['round']})"],
    }
    if context_op == "append":
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
# role node has. Approving a plan hands back to the overseer (there's more
# work left); approving a final answer ends the run. Rejecting either always
# goes back to the overseer -- no exhaustion branch (see module docstring).
# --------------------------------------------------------------------------

def evaluator(state: AgentState) -> Command[Literal["router", "__end__"]]:
    task_text = state["messages"][-1].content
    node = state.get("node") or "solver"
    output = state.get("output") or ""
    judging_plan = node == "planner"

    llm = ROUTER.chat_model(Task.REASON, temperature=0.0)
    if judging_plan:
        target, target_note = "PLAN", (
            "would produce a satisfying result if followed, not whether "
            "the task is already done"
        )
        human_label = f"PLAN (from {node})"
    else:
        target, target_note = f"{node.upper()} OUTPUT", "as a finished answer"
        human_label = f"{node.upper()} OUTPUT"
    system_prompt = EVALUATOR_PROMPT.format(target=target, target_note=target_note, max_iter=MAX_TOOL_ITERATIONS)
    human_body = f"ORIGINAL REQUEST:\n{task_text}\n\n{human_label}:\n{output}"
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    reply = _tool_loop(llm, messages)
    # _parse_approval defaults to approve=False whenever "APPROVE:" isn't
    # found in `reply` at all (e.g. _tool_loop exhausted on unparseable
    # replies) -- fails CLOSED by construction: an evaluator that never
    # rendered a real verdict is not evidence the answer is fine.
    approve, reason = _parse_approval(reply)

    if approve and judging_plan:
        return Command(
            update={
                "plan": output,
                "output": None,
                "feedback": "",
                "board": [f"evaluator approved {node}'s plan (round {state['round']})"],
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
# router -> whichever of the five it picks; every specialist -> router;
# evaluator -> router (reject, or plan approved) or END (final answer
# approved) -- nothing else to wire here.

app = g.compile(checkpointer=InMemorySaver())
