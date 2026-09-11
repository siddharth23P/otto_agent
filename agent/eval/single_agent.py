"""A single-agent control, for answering "is the graph the problem?" with a
measurement instead of an opinion.

The graph (agent/pipeline/nodes.py) is router -> role -> router -> evaluator ->
router, and EVERY one of those nodes begins with exactly two messages: a system
prompt and a freshly rendered body. A role's tool conversation lives in a local
list inside _tool_loop and is discarded when the node returns; what crosses the
boundary is the role's final answer text, a one-line-per-call action record,
and whatever the evaluator said. So on a long tool-heavy task the agent does
not work continuously -- it restarts from a summary, repeatedly, and each
restart pays for re-deriving what it already knew.

That is a real cost and it might be worth paying. Specialised roles, a plan,
and an independent judge are the things the graph buys. This module is the
control that says what they cost: the same model, the same tools, the same
deadline, one system prompt and ONE conversation that never resets, with the
agent deciding for itself when it is finished.

It is deliberately not wired into the CLI and nothing in agent/pipeline
imports it. It exists to be run side by side with the real pipeline on the
same benchmark task, and to be deleted if the answer turns out to be "the
graph is fine."
"""
from __future__ import annotations

import logging
from typing import Callable

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.pipeline.nodes import (
    ROUTER,
    UNPARSEABLE_FEEDBACK,
    _TOOL_MENU,
    _DIAGNOSTIC_HABITS,
    _call,
    _parse_worker_reply,
    _strip_code_fence,
    _summarise_action,
)
from agent.pipeline.tools import TOOL_DISPATCH
from agent.router.mapping import Task

logger = logging.getLogger(__name__)

#: The whole agent, in one prompt. Says the same things the graph distributes
#: across SOLVER_PROMPT (do the work), EVALUATOR_PROMPT (check it for real,
#: don't trust the claim) and the overseer (decide what to do next) -- so the
#: comparison is about the ARCHITECTURE, not about which instructions each
#: side happened to be given.
SINGLE_AGENT_PROMPT = (
    "You are an engineer with a shell. Finish the task below yourself, "
    "end to end: work out what is wrong, change it, and confirm the change "
    "actually holds.\n\n"
    + _DIAGNOSTIC_HABITS +
    "Reply with exactly\nACTION: <" + _TOOL_MENU + ">\nCODE:\n<input for that "
    "tool>\nand you will be shown the result, then you continue. One tool call "
    "per reply.\n\n"
    "Before you finish, run a command that would FAIL if the task were not "
    "done, and read what it actually prints. An explanation is not evidence. "
    "When -- and only when -- you have seen it work, reply with exactly\n"
    "FINAL:\n<what you did and what proved it>\n\n"
    "You have at most {max_steps} tool calls. Spend them on finding out what "
    "is true, not on repeating what you already tried."
)


def run_single_agent(
    task: str,
    *,
    max_steps: int = 60,
    on_action: Callable[[str], None] | None = None,
) -> tuple[str, list[str]]:
    """Run `task` to completion in one conversation. Returns (final answer,
    one line per tool call) -- the same action summaries the graph records, so
    the two can be compared on identical terms.
    """
    llm = ROUTER.chat_model(Task.REASON)
    messages = [
        SystemMessage(SINGLE_AGENT_PROMPT.format(max_steps=max_steps)),
        HumanMessage(f"TASK:\n{task}"),
    ]
    taken: list[str] = []
    answer = ""

    for _ in range(max_steps):
        text = _call(llm, messages)
        kind, tool_name, body = _parse_worker_reply(text)

        if kind == "final":
            return _strip_code_fence(body), taken
        if kind == "unparseable":
            answer = text
            messages.append(AIMessage(text))
            messages.append(HumanMessage(UNPARSEABLE_FEEDBACK))
            continue
        if tool_name not in TOOL_DISPATCH:
            evidence = f"tool {tool_name!r} is not available (allowed: {sorted(TOOL_DISPATCH)})"
        else:
            result = TOOL_DISPATCH[tool_name](body)
            line = _summarise_action(tool_name, body, result)
            taken.append(line)
            if on_action is not None:
                on_action(line)
            evidence = (
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}\n"
                f"returncode: {result.returncode}"
            )
        messages.append(AIMessage(text))
        messages.append(HumanMessage(f"TOOL RESULT:\n{evidence}"))

    return answer, taken
