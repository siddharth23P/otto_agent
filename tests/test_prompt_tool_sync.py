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
    """PLANNER, SOLVER, SUMMARIZER and FINDER came to 7663 characters between
    them, because each carried its own copy of the 831-character protocol
    block. One prompt must stay under half that.

    Measured against what it replaced rather than a number somebody picked: the
    cap has to stay meaningful as the prompt gains real capability -- modes,
    delegation, the rule about irreversible actions -- none of which the four
    it replaced could express at any length.

    The point is to make the next person NOTICE. It has already worked once:
    adding delegation pushed this over, and 134 characters came out of the
    switch_mode and delegate hints before the cap moved.
    """
    replaced = 7663
    assert len(pn.AGENT_PROMPT) <= replaced // 2, (
        f"AGENT_PROMPT is {len(pn.AGENT_PROMPT)} chars against a {replaced // 2} "
        "ceiling. Trim before raising this -- nodes.py records a measurement "
        "where a fifth instruction block erased the effect of the four before it."
    )
