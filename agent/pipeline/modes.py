"""The modes one agent loop can work in, and the model each one runs on.

This replaces four graph nodes with four entries in a dict, and that is the
whole point. `planner`, `solver`, `summarizer` and `finder` were separate
LangGraph nodes, so changing role meant crossing a node boundary: a new system
prompt, a freshly rendered body, and the previous tool conversation discarded.
Measured on Claw-Eval traces, those boundaries cost 20 to 126 seconds each and
were 45 to 69% of a run's wall time, for three overhead model calls out of every
five. The agent forgot what it had just done AND paid to be reminded.

A mode is the same four roles, kept as data. Switching one appends a message to
a conversation that keeps running -- so the work survives by construction and
the switch costs one call instead of a boundary.

A MODE IS A PAIR: a routing Task (hence a model) and that role's guidance. Both
change together, because they were always two halves of the same decision --
`solver` ran on openai:gpt-5-mini AND was told to write and run code; splitting
those would let the loop end up reasoning hard in a prompt that asks it to
summarise.

WHAT THE MODEL NAMES IS A CAPABILITY, NEVER A VENDOR. `Task` is as far as this
module goes; agent/router/mapping.py's TASK_ROUTES stays the only place that
knows which vendor serves what, and it holds measured decisions (see its own
comment on reasoning_effort). A prompt naming "claude" or "gemini" would move
that policy into model judgment and make the routing table a lie.

GUIDANCE IS SHORT ON PURPOSE. agent/pipeline/nodes.py records that a fifth
debugging habit did not merely fail to take -- it erased the effect of the four
before it. These blocks carry only what is specific to the role; the ACTION/CODE
protocol, the tool menu and the debugging habits are said once, in the loop's own
system prompt, instead of four times. Total prompt text per call goes DOWN.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.router.mapping import Task


@dataclass(frozen=True, slots=True)
class Mode:
    """One way of working: which model answers, and what it is told."""

    name: str
    task: Task
    #: Appended to the live conversation as a HumanMessage when this mode is
    #: entered. NEVER a SystemMessage: langchain_anthropic raises on
    #: non-consecutive system messages and langchain_google_genai hoists a
    #: mid-list one out of position or drops it silently, so a system message
    #: after the opening run is a hard failure on two of the four vendors here.
    guidance: str


#: The mode a run starts in. `solve` rather than `plan`, because most requests
#: are not multi-step and paying for a plan first was one of the things the old
#: overseer spent a model call deciding.
DEFAULT_MODE = "solve"

MODES: dict[str, Mode] = {
    "solve": Mode(
        name="solve",
        task=Task.REASON,
        guidance=(
            "Work out a concrete answer, writing and running code where that "
            "helps you check it. Prefer evidence you produced over reasoning "
            "you did not check."
        ),
    ),
    "plan": Mode(
        name="plan",
        task=Task.PLAN,
        guidance=(
            "Break the work into an ordered list of concrete steps before "
            "doing more of it. Say what each step produces and how you will "
            "know it worked. Then switch back and carry the steps out "
            "yourself -- nobody else is going to."
        ),
    ),
    "summarize": Mode(
        name="summarize",
        task=Task.SUMMARIZE,
        guidance=(
            "Condense what you already have. You are not looking anything new "
            "up or solving a new problem. Keep the specifics -- names, "
            "numbers, paths -- and drop the narration."
        ),
    ),
    "find": Mode(
        name="find",
        task=Task.CHAT_FAST,
        guidance=(
            "Look something up before answering. In order: recall_memory for "
            "something from earlier in this conversation that was compacted "
            "away; execute_bash for anything on this machine (grep, find, "
            "git, gh); rag for a knowledge base; web_search for the open web."
        ),
    ),
}


def mode_names() -> tuple[str, ...]:
    """The modes, in the order a prompt should list them."""
    return tuple(MODES)


def parse_mode_body(body: str) -> str | None:
    """The mode a `switch_mode` CODE: body names, or None if it names none.

    Only the first non-empty line is the mode; anything after it is the
    model's reason for switching, which is worth recording on the board but is
    deliberately never fed back to it -- it already knows why it switched, and
    echoing it would spend context saying so.
    """
    for line in body.splitlines():
        candidate = line.strip().strip(".`\"'").lower()
        if candidate:
            return candidate if candidate in MODES else None
    return None


def mode_reason(body: str) -> str:
    """Whatever the model said after the mode name, for the board."""
    lines = [line.strip() for line in body.splitlines() if line.strip()]
    return " ".join(lines[1:])[:200]
