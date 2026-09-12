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
import agent.pipeline.tools as pt
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


def test_the_prompt_pushes_back_on_scope_before_writing_code():
    """Measured against the same agent without it, on twelve real tickets in a
    real repository: 54% fewer lines, 22% fewer tokens, 20% lower cost, 27%
    faster. It was the only variant tested that cut every metric at once."""
    ladder = pn._MINIMALITY_LADDER
    assert "stop at the first" in ladder
    for rung in ("need not exist", "codebase already has it",
                 "standard library", "platform", "already installed",
                 "one line"):
        assert rung.split()[0] in ladder, f"the {rung!r} rung is missing"


def test_being_lazy_never_reaches_the_safety_guards():
    """The bare "write one-liners" arm in the same experiment WAS cheaper and
    dropped a safety guard doing it, scoring 95% where every other arm held
    100%. This sentence is the difference, not decoration."""
    ladder = pn._MINIMALITY_LADDER.lower()
    for guard in ("validation", "data-loss", "security", "accessibility"):
        assert guard in ladder, f"{guard} is no longer protected from the ladder"
    assert "never about the reading" in ladder, (
        "the ladder is about the solution; without this it reads as permission "
        "to skip understanding the problem"
    )


def test_no_rule_is_stated_more_than_twice():
    """The prompt was not full, it was redundant.

    `switch_mode` appeared three times, the mode list twice, and "the
    conversation carries over" twice -- because `_TOOL_BODY_HINT` is derived
    from the tool registry and the mode paragraph is hand-written prose, and
    neither block knew the other existed. Removing the second copies freed 229
    characters without deleting a single behavioural rule, taking the headroom
    under the cap from 73 to 302.

    A character cap cannot express that. This can: at most twice is once where
    the tool is listed and once where it is explained. A third is drift.
    """
    for phrase in ("switch_mode", "delegate", "conversation carries",
                   "|".join(pn.mode_names())):
        assert pn.AGENT_PROMPT.count(phrase) <= 2, (
            f"{phrase!r} is stated {pn.AGENT_PROMPT.count(phrase)} times -- "
            "the prompt is accreting duplicates again"
        )


def test_the_mode_names_survive_in_exactly_one_place():
    """`test_the_agent_prompt_offers_every_mode` above reads the mode names
    out of AGENT_PROMPT, and after the compression they appear only inside
    `_MODE_TOOL_HINT`. Trimming that line would silently break mode
    reachability, so the dependency is asserted rather than left implicit."""
    assert "|".join(pn.mode_names()) in pn._MODE_TOOL_HINT


# --------------------------------------------------------------------------
# The menu is composed from what the run can reach
# --------------------------------------------------------------------------
#
# Filtering the PROMPT, never `dispatch_table()`. A model that names a tool
# left out of the menu still reaches it and still gets that tool's own
# refusal, so being wrong about reachability costs exactly what being right
# costs today -- no new gate, no extra round trip.

def _live(workspace=None, container=None):
    from agent.pipeline.execution import bind_command_runner
    from agent.pipeline.tools import reachable_tools
    from agent.pipeline.workspace import bind_workspace
    import contextlib

    with contextlib.ExitStack() as stack:
        if workspace:
            stack.enter_context(bind_workspace(workspace))
        if container:
            stack.enter_context(bind_command_runner(lambda c, t: ("", "", 0)))
        return reachable_tools()


def test_the_reachability_table_covers_every_tool():
    """A fact about a tool belongs next to the tool, so the next one added
    says its own preconditions instead of being discovered missing."""
    from agent.pipeline.tools import TOOL_NEEDS, TOOL_TIERS

    assert set(TOOL_NEEDS) == set(TOOL_TIERS)


def test_a_chat_turn_is_not_offered_tools_it_cannot_reach(tmp_path):
    """No workspace and no container: a browser, a screen and a file index
    are all unreachable, and advertising them invites a call that can only
    fail."""
    live = _live()

    for unreachable in ("browse", "browse_act", "look", "look_act",
                        "rag", "code_map", "read_file", "write_file"):
        assert unreachable not in live, unreachable
    for always in ("execute_bash", "execute_python", "web_search", "recall_memory"):
        assert always in live, always


def test_a_container_run_gets_the_browser_and_the_screen(tmp_path):
    live = _live(container=True)

    assert {"browse", "look", "read_file"} <= set(live)
    assert "rag" not in live, "rag indexes files on this machine"


def test_a_workspace_run_gets_the_file_tools(tmp_path):
    live = _live(workspace=str(tmp_path))

    assert {"rag", "code_map", "read_file", "write_file"} <= set(live)
    assert "browse" not in live


def test_a_benchmark_run_reaches_everything(tmp_path):
    """Both bound, which is what SWE-bench and Claw-Eval do -- so filtering
    saves nothing there. Worth pinning, because I claimed otherwise once."""
    assert len(_live(workspace=str(tmp_path), container=True)) == len(pt.TOOL_TIERS)


def test_the_composed_prompt_names_every_reachable_tool_and_no_other(tmp_path):
    live = _live(workspace=str(tmp_path))
    prompt = pn.compose_agent_prompt(live)

    for name in live:
        assert name in prompt, name
    assert "browse_act" not in prompt
    assert "look_act" not in prompt


def test_a_smaller_menu_is_a_smaller_prompt(tmp_path):
    """The whole point. Measured: 553 chars on a chat turn, 662 on a
    delegated child, 123 on terminal-bench, 0 on Claw-Eval."""
    everything = len(pn.compose_agent_prompt(pt.TOOL_TIERS))
    chat = len(pn.compose_agent_prompt(_live()))
    child = len(pn.compose_agent_prompt(_live(), may_delegate=False))

    assert everything - chat > 400
    assert child < chat, "a child that cannot delegate should not be told it can"


def test_a_delegated_child_is_not_offered_delegate():
    """It was advertised and then refused at the dispatch -- a tool the child
    could name, could not use, and paid an exchange to discover."""
    child = pn.compose_agent_prompt(_live(), may_delegate=False)

    assert "delegate" not in child
    assert "switch_mode" in child


def test_the_evaluator_is_not_offered_delegate():
    """`_tool_loop`, which the evaluator runs in, refuses it."""
    assert "delegate" not in pn.EVALUATOR_PROMPT


def test_no_body_hint_contains_a_brace():
    """`EVALUATOR_PROMPT` embeds a composed action block and is then
    `.format()`-ed. A `{` or `}` in any hint would be read as a format field
    and raise at import. There are none today, which makes it a trap rather
    than a bug."""
    for name, hint in pn._BODY_HINTS.items():
        assert "{" not in hint and "}" not in hint, name
