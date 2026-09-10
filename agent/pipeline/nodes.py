"""The router/planner/solver/summarizer/finder/evaluator graph.

Replaces the orchestrator/worker/evaluate/subtask_consensus/synthesize
swarm pipeline (retired 2026-09-10, same day it was finished -- see below
for why) after a design discussion that produced a genuinely different
shape, not a tune of the old one:

  * ROUTER -- a single classification call, no vote, no tool use. Given the
    request (round 1) or the previous attempt plus the evaluator's
    rejection reason (retry), it picks exactly ONE specialist -- planner,
    solver, summarizer, or finder -- to attempt the WHOLE request. On
    retry it may pick a DIFFERENT specialist than the one that just failed,
    if the feedback suggests the wrong kind of node tried it (explicit
    design call, 2026-09-10: "router re-dispatches" over "always retry the
    same node").

  * planner / solver / summarizer / finder -- each a single specialist with
    its own narrow prompt and the SAME ACTION/execute_tool-then-FINAL loop
    (_tool_loop below) -- may call execute_python, execute_bash,
    web_search, rag, complete_code, or predict_edit before committing to
    an answer. Exactly one of these runs per round; there is no fan-out,
    so there is nothing to reconcile afterward (no synthesize() equivalent
    exists here).

  * evaluator -- judges the dispatched specialist's answer against the
    original request, with the SAME tool access (2026-09-10 design
    direction carried over unchanged from the swarm pipeline's own
    evaluator-roles work earlier the same day: "give similar tools to
    evaluators as well"). Approve -> done. Reject -> feedback goes back to
    ROUTER, not straight back to the same node -- the retry loop is
    router-mediated, matching this graph's whole "one router, several
    specialists" shape instead of the worker-in-a-loop shape the old
    pipeline used.

Why the swap, in short: the swarm pipeline's value came from genuinely
independent parallel attempts catching each other's blind spots (an
evaluator grounded in execution, N peer authors). That shape fits a task
that decomposes into independent pieces. It's a worse fit for a request
that is really just "pick the right kind of specialist and let them work,"
which has no pieces to parallelize -- forcing a 3-way split onto something
like "summarize this document" produces three redundant or arbitrary
fragments to reconcile, not three genuine subtasks. This graph is built for
that second shape instead: route once, work once, judge once, retry by
re-routing if wrong.

Deliberately out of scope for this first pass (2026-09-10 design calls):
  - web_search and rag are STUBBED (tools.py) -- prove the graph skeleton
    first, wire a real search API / knowledge base once it's working.
  - No domain-specific verification (the old evaluate()'s compile-gate /
    math-recompute-and-compare branches) -- the evaluator's own tool access
    is the ONE verification mechanism now, for every kind of request alike
    (it can execute_python a candidate itself and reject on failure, same
    as any other check it might do). Simpler, and consistent with there
    being only one evaluator role instead of three specialized reviewers.

Added later the same day, once the skeleton above was proven against the
golden set (10/10 on domain=code): complete_code and predict_edit
(tools.py), Mercury's FIM and edit endpoints (Task.CODE_COMPLETE/
CODE_EDIT, mapping.py) -- real capabilities that already existed in the
router for a different, now-retired caller, just never wired into this
graph's tool box until asked for. Both are READ_ONLY like everything else
here; predict_edit in particular takes no instruction (see its own
docstring) -- it is closer to "what would come next" than to an editable
tool, and is offered to every role/the evaluator on that understanding.
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
                    # the swarm pipeline never had to name, since none of
                    # its own node functions were called "router"; the two
                    # are unrelated, case matters, Python doesn't confuse
                    # them, only a reader skimming might).

#: The four specialists ROUTER may dispatch to. Always exactly these four;
#: adding a fifth means adding its prompt, its node function, and a line in
#: the graph wiring at the bottom of this file.
ROLE_NODES = ("planner", "solver", "summarizer", "finder")

#: Bound on how many router-dispatch rounds a single request may go through
#: before the evaluator's last rejection is given up on and its last
#: attempt is used as-is. Same rationale as the swarm pipeline's
#: MAX_SUBTASK_ROUNDS (bumped there the same day for the same underlying
#: reason: don't let a strict evaluator starve a real task of retries).
MAX_DISPATCH_ROUNDS = 6

#: Same rationale as the retired pipeline's identical constant: a diffusing
#: (Mercury) call cut off at max_tokens is not a clean prefix, it's an
#: unconverged snapshot, and must be retried rather than accepted.
MAX_DIFFUSION_RETRIES = 3

#: Bound on one node's tool-calling loop (ACTION/execute_TOOL round-trips)
#: before its last reply is used as-is. Applies identically to every role
#: node and the evaluator (_tool_loop is shared by all of them) -- keeps a
#: confused node from looping forever instead of surfacing its best attempt.
MAX_TOOL_ITERATIONS = 5

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

ROUTER_PROMPT = (
    "Decide which single specialist should handle the request below. "
    "Choose exactly one:\n"
    "planner -- breaks a multi-step goal into an ordered plan; produces no "
    "code or looked-up facts of its own\n"
    "solver -- writes and/or runs code, works out a concrete answer to a "
    "well-defined problem\n"
    "summarizer -- condenses or rewrites GIVEN text/content; nothing new to "
    "look up or solve\n"
    "finder -- needs to look something up (web search or a knowledge base) "
    "before it can answer\n"
    "Reply with exactly two lines:\nNODE: one of planner, solver, "
    "summarizer, finder\nWHY: one sentence"
)
#: Round 2+ of router(): the router re-decides given WHY the previous
#: attempt was rejected -- may keep the same specialist or pick a
#: different one (2026-09-10 design call: "router re-dispatches", not
#: "always retry the same node").
ROUTER_RETRY_PROMPT = (
    "The previous attempt below was rejected by the evaluator. Decide which "
    "specialist should try next -- the same one, with the feedback in mind, "
    "or a different one if the feedback suggests this was the wrong kind of "
    "task for whoever tried it (e.g. it needed a lookup nobody did, or code "
    "nobody ran, or it needed rewriting rather than solving from scratch). "
    "Reply with exactly two lines:\nNODE: one of planner, solver, "
    "summarizer, finder\nWHY: one sentence"
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
    "answering -- prefer the web_search or rag tool over guessing (both are "
    "stubbed today and will tell you so; if that happens, fall back to your "
    "own knowledge and say plainly in your answer that you could not "
    "verify it). Reply with exactly\nACTION: <execute_python|execute_bash|"
    "web_search|rag|complete_code|predict_edit>\nCODE:\n<input for that "
    "tool -- complete_code: prefix code, optionally then a line "
    "\"---SUFFIX---\" and trailing code; predict_edit: code, optionally "
    "with a <|cursor|> marker, no instruction -- it only predicts the next "
    "edit>\nand you will be shown the result, then you can continue. When "
    "you are done, reply with exactly\nFINAL:\n<the answer, nothing else -- "
    "no markdown code fences, no explanation>\nYou have at most {max_iter} "
    "exchanges before your last reply is used as-is."
)
#: Round 2+ of any role node: revise with the previous attempt and the
#: evaluator's feedback in view, not a fresh blind attempt -- same
#: "don't regenerate blind" reasoning as every retry prompt in the retired
#: pipeline. Shared by all four roles (parameterized by {role_upper}) since
#: the instruction is identical regardless of which specialist is retrying
#: -- including the case where the CURRENT specialist is not the one whose
#: attempt got rejected (router() may have re-dispatched to someone new).
ROLE_REVISE_PROMPT = (
    "You are the {role_upper}. A previous attempt at the task below was "
    "rejected by the evaluator (possibly made by a different specialist "
    "than you) -- see the previous attempt and the evaluator's feedback for "
    "what to fix or avoid. Produce a better answer, reusing anything from "
    "the previous attempt that was actually fine. Same reply format as "
    "before: ACTION/CODE to use a tool, or FINAL: when done. You have at "
    "most {max_iter} exchanges before your last reply is used as-is."
)
EVALUATOR_PROMPT = (
    "Judge whether the {role} output below actually satisfies the original "
    "request. You may check your judgment with a tool: reply with exactly\n"
    "ACTION: <execute_python|execute_bash|web_search|rag|complete_code|"
    "predict_edit>\nCODE:\n<input for that tool -- complete_code: prefix "
    "code, optionally then a line \"---SUFFIX---\" and trailing code; "
    "predict_edit: code, optionally with a <|cursor|> marker, no "
    "instruction -- it only predicts the next edit>\nand you will be shown "
    "the result, then you can continue. When you are done, reply with "
    "exactly\nFINAL:\nAPPROVE: yes or no\nWHY: one sentence\nYou have at "
    "most {max_iter} exchanges before your last reply is used as-is."
)
#: Fed back inside _tool_loop when a reply has neither ACTION: nor FINAL:
#: (or a FINAL: with nothing after it). Ported from the retired pipeline's
#: identical fix (2026-09-10): a marker-less or empty reply is never
#: silently accepted as an answer -- the node is told what went wrong and
#: made to retry, spending one of MAX_TOOL_ITERATIONS exchanges.
UNPARSEABLE_FEEDBACK = (
    "Your last reply didn't match either required format -- it had no "
    "ACTION: with a tool call, and no FINAL: followed by an answer (or "
    "FINAL: was there but empty). Reply again using exactly one of those "
    "two formats, with no other text."
)


# --------------------------------------------------------------------------
# Shared LLM-call plumbing -- ported from the retired pipeline verbatim.
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

    Applied to every FINAL body regardless of which node produced it
    (ported from the retired pipeline's synthesize()-side fix): the
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
    than accepting it. Ported from the retired pipeline verbatim -- same
    fix, same reasoning, this graph has the identical failure mode to guard
    against (a marker-less or empty reply becoming an answer with zero
    validation).
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
    reply never touches it: a tool call is not a candidate answer. This is
    the exact fix the retired pipeline's _worker_loop/_synthesize_loop
    needed (2026-09-10, caught live via debug_pipeline_agents3.py on
    nphard_tsp_02: exhausting right after an ACTION step used to leak that
    reply's raw "ACTION: ...\\nCODE:\\n..." protocol text straight through
    as the final answer) -- carried over here from the start rather than
    re-discovering it.
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
            if candidate in ROLE_NODES:
                node = candidate
        elif upper.startswith("WHY:"):
            why = line.split(":", 1)[1].strip()
    if node is None:
        return "solver", f"could not parse a NODE: line from {text.strip()[:200]!r}, defaulting to solver"
    return node, why


# --------------------------------------------------------------------------
# router -- single classification call, no vote, no tool use.
# --------------------------------------------------------------------------

def router(state: AgentState) -> Command[Literal["planner", "solver", "summarizer", "finder"]]:
    round_ = state["round"] + 1
    task_text = state["messages"][-1].content
    feedback = state.get("feedback") or ""

    llm = ROUTER.chat_model(Task.CHAT_FAST, temperature=0.2)
    if not feedback:
        system_prompt = ROUTER_PROMPT
        human_body = task_text
    else:
        system_prompt = ROUTER_RETRY_PROMPT
        human_body = (
            f"TASK:\n{task_text}\n\n"
            f"PREVIOUS ATTEMPT (by {state.get('node')}):\n{state.get('output') or ''}\n\n"
            f"EVALUATOR FEEDBACK:\n{feedback}"
        )
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    text = _call(llm, messages)
    node, why = _parse_router(text)

    return Command(
        update={
            "round": round_, "node": node,
            "board": [f"round {round_}: router dispatched to {node} ({why})"],
        },
        goto=node,
    )


# --------------------------------------------------------------------------
# The four specialists -- one shared implementation (_run_role), four thin
# named wrappers (LangGraph nodes need their own registered function/name).
# --------------------------------------------------------------------------

def _run_role(state: AgentState, *, role: str, task: Task, temperature: float, prompt: str) -> Command[Literal["evaluator"]]:
    task_text = state["messages"][-1].content
    feedback = state.get("feedback") or ""

    llm = ROUTER.chat_model(task, temperature=temperature)
    if not feedback:
        system_prompt = prompt.format(max_iter=MAX_TOOL_ITERATIONS)
        human_body = task_text
    else:
        # Revise with the previous attempt and the evaluator's feedback in
        # view -- even if `role` is not who made that previous attempt
        # (router() may have re-dispatched to a different specialist).
        system_prompt = ROLE_REVISE_PROMPT.format(role_upper=role.upper(), max_iter=MAX_TOOL_ITERATIONS)
        human_body = (
            f"TASK:\n{task_text}\n\n"
            f"PREVIOUS ATTEMPT:\n{state.get('output') or ''}\n\n"
            f"EVALUATOR FEEDBACK:\n{feedback}"
        )
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    output = _tool_loop(llm, messages)

    return Command(
        update={"output": output, "board": [f"{role} produced an answer (round {state['round']})"]},
        goto="evaluator",
    )


def planner(state: AgentState) -> Command[Literal["evaluator"]]:
    return _run_role(state, role="planner", task=Task.PLAN, temperature=0.4, prompt=PLANNER_PROMPT)


def solver(state: AgentState) -> Command[Literal["evaluator"]]:
    return _run_role(state, role="solver", task=Task.REASON, temperature=0.5, prompt=SOLVER_PROMPT)


def summarizer(state: AgentState) -> Command[Literal["evaluator"]]:
    return _run_role(state, role="summarizer", task=Task.SUMMARIZE, temperature=0.2, prompt=SUMMARIZER_PROMPT)


def finder(state: AgentState) -> Command[Literal["evaluator"]]:
    # No dedicated Task route for "look something up" exists yet -- CHAT_FAST
    # (fast turnaround, light reasoning) fits a node whose real work is
    # supposed to be the tool call, not deliberation. Revisit if/when
    # web_search/rag stop being stubs and finder's actual job gets harder.
    return _run_role(state, role="finder", task=Task.CHAT_FAST, temperature=0.3, prompt=FINDER_PROMPT)


# --------------------------------------------------------------------------
# evaluator -- judges the dispatched specialist's answer, with the same
# tool access every role node has. Approve -> done. Reject -> back to
# router (not straight back to the same node -- see module docstring).
# --------------------------------------------------------------------------

def evaluator(state: AgentState) -> Command[Literal["router", "__end__"]]:
    task_text = state["messages"][-1].content
    node = state.get("node") or "solver"
    output = state.get("output") or ""

    llm = ROUTER.chat_model(Task.REASON, temperature=0.0)
    system_prompt = EVALUATOR_PROMPT.format(role=node, max_iter=MAX_TOOL_ITERATIONS)
    human_body = f"ORIGINAL REQUEST:\n{task_text}\n\n{node.upper()} OUTPUT:\n{output}"
    messages = [SystemMessage(system_prompt), HumanMessage(human_body)]
    reply = _tool_loop(llm, messages)
    # _parse_approval defaults to approve=False whenever "APPROVE:" isn't
    # found in `reply` at all (e.g. _tool_loop exhausted on unparseable
    # replies) -- fails CLOSED by construction, no special-casing needed:
    # an evaluator that never rendered a real verdict is not evidence the
    # answer is fine.
    approve, reason = _parse_approval(reply)

    if approve:
        return Command(
            update={"final_output": output, "board": [f"evaluator approved {node}'s answer (round {state['round']})"]},
            goto=END,
        )
    if state["round"] >= MAX_DISPATCH_ROUNDS:
        return Command(
            update={
                "final_output": output,
                "board": [f"exhausted after {MAX_DISPATCH_ROUNDS} rounds -- using {node}'s last answer"],
            },
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
# Every other edge is a Command(goto=...) from the node function itself
# (router -> one of the four specialists; each specialist -> evaluator;
# evaluator -> router or END) -- nothing else to wire here.

app = g.compile(checkpointer=InMemorySaver())
