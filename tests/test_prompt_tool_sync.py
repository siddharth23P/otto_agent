"""A prompt's ACTION: enumeration is meant to list exactly what TOOL_DISPATCH
dispatches (agent/pipeline/nodes.py's prompts, agent/pipeline/tools.py's
registry). A prompt advertising a tool that is not registered sends the model
down a dead end; a registered tool no prompt mentions never gets used.

There used to be four role prompts to keep in sync, each carrying its own copy
of the protocol. They are one AGENT_PROMPT now, with the role-specific part
moved into agent/pipeline/modes.py -- so the checks here changed shape: one
prompt instead of four, plus the two things the mode table has to keep true.

The length cap is here on purpose rather than as a style rule. nodes.py records
a measurement where a fifth instruction block did not merely fail to take, it
erased the effect of the four before it, taking system-inspection commands from
17 to 0. The next person who wants to explain more to this model should have to
notice they are doing it.
"""
from agent.pipeline import nodes as pn
from agent.pipeline.modes import MODES, mode_names
from agent.pipeline.tools import TOOL_DISPATCH
from agent.router.mapping import TASK_ROUTES


def _formatted_evaluator(*, target: str, target_note: str) -> str:
    return pn.EVALUATOR_PROMPT.format(
        max_iter=pn.MAX_EVALUATOR_ITERATIONS, target=target, target_note=target_note,
        rubric='- a checkable criterion',
    )


def test_the_agent_prompt_mentions_every_dispatchable_tool():
    for tool_name in TOOL_DISPATCH:
        assert tool_name in pn.AGENT_PROMPT, f"AGENT_PROMPT never mentions tool {tool_name!r}"


def test_the_agent_prompt_offers_every_mode():
    """A mode the prompt never names is a mode the model cannot reach."""
    for name in mode_names():
        assert name in pn.AGENT_PROMPT, f"AGENT_PROMPT never offers mode {name!r}"


def test_every_mode_routes_somewhere_the_router_serves():
    for mode in MODES.values():
        assert mode.task in TASK_ROUTES, f"mode {mode.name} routes to an unserved task"


def test_the_agent_prompt_offers_the_two_tools_that_are_not_in_the_registry():
    """`ask_user` and `switch_mode` change the loop's own state rather than
    returning a ToolResult, so they are appended to the menu by hand -- which
    is exactly the kind of thing that gets forgotten."""
    assert "ask_user" in pn.AGENT_PROMPT
    assert "switch_mode" in pn.AGENT_PROMPT


def test_one_agent_prompt_is_cheaper_than_the_four_role_prompts_it_replaced():
    """PLANNER/SOLVER/SUMMARIZER/FINDER came to 7663 characters between them,
    because each carried its own copy of the 831-character protocol block. The
    cap allows the solver's old size plus a fifth, which is what the mode
    machinery is allowed to cost."""
    assert len(pn.AGENT_PROMPT) <= 3300, (
        f"AGENT_PROMPT is {len(pn.AGENT_PROMPT)} chars; the old SOLVER_PROMPT "
        "was 2681 and the cap is 1.2x that"
    )


def test_evaluator_prompt_mentions_every_dispatchable_tool():
    """It judges by checking, not by reading -- so it needs the tools too."""
    text = _formatted_evaluator(target="ANSWER", target_note="as a finished answer")
    for tool_name in TOOL_DISPATCH:
        assert tool_name in text, f"EVALUATOR_PROMPT never mentions tool {tool_name!r}"
