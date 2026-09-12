"""Coverage for agent/pipeline/modes.py -- four graph nodes kept as data.

The properties worth locking in are not about any one mode. They are the two
invariants that make the whole design work: every mode routes through a Task
that the routing table actually serves, and no mode names a vendor. A mode that
named "claude" would move vendor policy out of agent/router/mapping.py, which
holds measured decisions, and into a prompt.

Guidance length is tested too, deliberately. nodes.py records a measurement
where a fifth instruction block erased the effect of the four before it, so the
next person who wants to explain more has to notice they are doing it.
"""
import pytest

from agent.pipeline.modes import (
    DEFAULT_MODE,
    MODES,
    Mode,
    mode_names,
    mode_reason,
    parse_mode_body,
)
from agent.router.mapping import TASK_ROUTES, Task


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------

def test_every_mode_routes_to_a_task_the_router_actually_serves():
    """A mode pinned to a Task with no route would fail at the first call in
    that mode, and only in that mode -- the worst kind of late failure."""
    for mode in MODES.values():
        assert mode.task in TASK_ROUTES, f"{mode.name} routes to an unserved task"


def test_the_default_mode_exists():
    assert DEFAULT_MODE in MODES


def test_the_default_is_solve_not_plan():
    """Most requests are not multi-step. Paying for a plan first was one of
    the things the old overseer spent a model call deciding."""
    assert DEFAULT_MODE == "solve"


def test_no_mode_names_a_vendor():
    """The model asks for a capability; mapping.py owns which vendor serves
    it. A vendor name here makes the routing table a lie."""
    vendors = ("claude", "anthropic", "gemini", "google", "openai", "gpt",
               "inception", "mercury", "haiku")
    for mode in MODES.values():
        lowered = mode.guidance.lower()
        for vendor in vendors:
            assert vendor not in lowered, f"{mode.name} guidance names {vendor}"


def test_each_mode_knows_its_own_name():
    for name, mode in MODES.items():
        assert mode.name == name


def test_guidance_stays_short():
    """KEEP THIS SHORT, for the reason nodes.py records: past some length the
    model acts on none of it rather than more of it."""
    for mode in MODES.values():
        assert len(mode.guidance) <= 400, f"{mode.name} guidance is {len(mode.guidance)} chars"


def test_guidance_carries_no_protocol_boilerplate():
    """The ACTION/CODE format is said once, in the loop's system prompt. Said
    again per mode it would be four copies of the thing that already crowds
    out the role-specific instruction."""
    for mode in MODES.values():
        assert "ACTION:" not in mode.guidance
        assert "FINAL:" not in mode.guidance


def test_the_roles_that_were_nodes_all_survive():
    """planner, solver, summarizer and finder become modes rather than being
    deleted -- their instructions were measured, not guessed."""
    assert set(mode_names()) == {"solve", "plan", "summarize", "find"}


def test_solve_and_plan_do_not_share_a_model():
    """If every mode routed to one model, swapping would buy only the
    guidance and the routing table would be doing nothing."""
    assert MODES["solve"].task is not MODES["plan"].task


# --------------------------------------------------------------------------
# Parsing what the model asked for
# --------------------------------------------------------------------------

def test_a_bare_mode_name_parses():
    assert parse_mode_body("plan") == "plan"


def test_leading_whitespace_and_punctuation_are_tolerated():
    """Models reliably write `PLAN.` or wrap it in backticks; refusing those
    costs a whole iteration to teach something the parse can handle."""
    assert parse_mode_body("  `PLAN`.  ") == "plan"


def test_only_the_first_line_is_the_mode():
    assert parse_mode_body("find\nbecause I need the config file") == "find"


def test_an_unknown_mode_is_none_rather_than_a_guess():
    """Guessing would silently put the run on the wrong model."""
    assert parse_mode_body("refactor") is None


def test_an_empty_body_is_none():
    assert parse_mode_body("   \n  ") is None


def test_the_reason_is_kept_for_the_board_but_separate_from_the_mode():
    assert mode_reason("plan\nthis needs ordering first") == "this needs ordering first"


def test_a_mode_with_no_reason_gives_an_empty_string():
    assert mode_reason("solve") == ""


def test_a_very_long_reason_is_clipped():
    assert len(mode_reason("solve\n" + "x" * 1000)) <= 200


# --------------------------------------------------------------------------
# Depth: an ordering, for one decision
# --------------------------------------------------------------------------

def test_every_mode_declares_a_depth():
    for mode in MODES.values():
        assert isinstance(mode.depth, int)


def test_the_orderings_are_not_all_the_same():
    """If every mode sat at one depth, no swap would ever be an escalation and
    the asymmetric handoff would be dead code."""
    assert len({m.depth for m in MODES.values()}) > 1


def test_the_reasoning_mode_is_the_deepest():
    assert MODES["solve"].depth == max(m.depth for m in MODES.values())


def test_lookup_is_shallower_than_reasoning():
    """Otherwise `find` -> `solve` would not restart, which is the case the
    measurement is about."""
    assert MODES["find"].depth < MODES["solve"].depth


def test_depth_names_no_model_and_no_price():
    """Which vendor serves a Task is mapping.py's business, and depth has to
    stay true when that table changes."""
    import inspect

    from agent.pipeline import modes

    source = inspect.getsource(modes)
    for vendor in ("gpt", "claude", "gemini", "mercury", "haiku"):
        assert vendor not in source.lower().split("depth")[-1][:200]
