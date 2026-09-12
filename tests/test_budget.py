"""Coverage for agent/pipeline/budget.py -- the ceiling a run spends against.

Otto had no budget at all before this. The reason it needs one is measured:
across four Claw-Eval tasks every tool call a task made totalled 0.1 to 0.4
seconds while the runs took 119 to 946, so a deadline that only checks inside a
tool -- which is what both benchmark harnesses had -- cannot see where the time
goes. C01 ran 1096 seconds against a 900-second budget without its deadline
firing once.

The tests that matter most here are the two-stage ones. A run killed mid-command
reports nothing; a run told to wrap up writes down what it found. That
difference is the whole reason `check` returns text instead of raising.
"""
import time

from agent.pipeline import budget as bd
from agent.pipeline.budget import (
    DEFAULT_MAX_MODEL_CALLS,
    WRAP_UP_NOTE,
    Budget,
    bind_budget,
    current_budget,
    default_budget,
)


# --------------------------------------------------------------------------
# Counting calls
# --------------------------------------------------------------------------

def test_a_fresh_budget_is_ok():
    assert Budget(max_model_calls=10).phase() == "ok"


def test_spending_every_call_reaches_the_ceiling():
    b = Budget(max_model_calls=4)
    for _ in range(4):
        b.spend()
    assert b.phase() == "spent"


def test_the_wrap_up_stage_opens_before_the_ceiling():
    """If it opened at the ceiling there would be no calls left to answer
    with, which is the whole point of having a first stage."""
    b = Budget(max_model_calls=10)
    for _ in range(8):
        b.spend()
    assert b.phase() == "wrap_up"


def test_a_budget_with_no_ceiling_at_all_never_stops():
    b = Budget()
    for _ in range(1000):
        b.spend()
    assert b.phase() == "ok"


# --------------------------------------------------------------------------
# The clock
# --------------------------------------------------------------------------

def test_a_passed_deadline_is_spent():
    now = time.monotonic()
    assert Budget(hard_at=now - 1, wrap_up_at=now - 2).phase() == "spent"


def test_a_passed_wrap_up_time_is_not_yet_spent():
    now = time.monotonic()
    assert Budget(hard_at=now + 60, wrap_up_at=now - 1).phase() == "wrap_up"


def test_of_puts_wrap_up_before_the_hard_stop():
    b = Budget.of(300)
    assert b.wrap_up_at < b.hard_at


def test_a_budget_shorter_than_its_own_margin_still_leaves_a_window():
    """A one-second budget must not produce a deadline already in the past,
    which would fail every task before it started."""
    b = Budget.of(1)
    assert b.hard_at > time.monotonic()


def test_until_shares_a_clock_a_caller_already_has():
    """A harness with its own deadline binds against it, so the two cannot
    disagree about when the task ends."""
    deadline = time.monotonic() + 120
    b = Budget.until(deadline, margin=0.0)
    assert b.hard_at <= deadline + 0.5


def test_calls_and_the_clock_are_both_ceilings():
    now = time.monotonic()
    b = Budget(max_model_calls=100, hard_at=now - 1, wrap_up_at=now - 2)
    assert b.phase() == "spent"


# --------------------------------------------------------------------------
# The wrap-up note
# --------------------------------------------------------------------------

def test_the_note_is_said_once_and_not_again():
    """Said every iteration it would be both noise and a prompt that grows
    for the rest of the run."""
    b = Budget(max_model_calls=10)
    for _ in range(8):
        b.spend()
    assert b.wrap_up_once() == WRAP_UP_NOTE
    assert b.wrap_up_once() is None


def test_no_note_while_there_is_budget_left():
    assert Budget(max_model_calls=10).wrap_up_once() is None


def test_the_note_tells_the_agent_to_answer_rather_than_stopping_it():
    assert "FINAL" in WRAP_UP_NOTE


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------

def test_nothing_is_bound_by_default():
    assert current_budget() is None


def test_binding_makes_it_reachable_and_unbinding_restores():
    b = Budget(max_model_calls=5)
    with bind_budget(b):
        assert current_budget() is b
    assert current_budget() is None


def test_binding_none_unbinds():
    """A harness binds per task; one task's spent budget must not leak into
    the next."""
    with bind_budget(Budget(max_model_calls=5)):
        with bind_budget(None):
            assert current_budget() is None


def test_the_default_ceiling_counts_calls_and_not_seconds(monkeypatch):
    """No wall-clock default on purpose: a chat turn with a person waiting
    should not be killed by a clock."""
    monkeypatch.delenv("OTTO_MAX_MODEL_CALLS", raising=False)
    b = default_budget()
    assert b.max_model_calls == DEFAULT_MAX_MODEL_CALLS
    assert b.hard_at is None


def test_the_default_ceiling_is_env_overridable(monkeypatch):
    monkeypatch.setenv("OTTO_MAX_MODEL_CALLS", "7")
    assert default_budget().max_model_calls == 7


def test_a_junk_env_value_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("OTTO_MAX_MODEL_CALLS", "not-a-number")
    assert default_budget().max_model_calls == DEFAULT_MAX_MODEL_CALLS


# --------------------------------------------------------------------------
# The ceiling actually reaches a run
# --------------------------------------------------------------------------

def test_every_pipeline_entry_point_binds_a_budget():
    """`default_budget` was imported and never called, so OTTO_MAX_MODEL_CALLS
    was a dead env var and an interactive turn had no ceiling on spend at all.
    The docstring said otherwise, which is how it survived."""
    import inspect

    from agent.pipeline import run

    for fn in (run.run_pipeline, run.run_pipeline_stream, run.resume_pipeline_stream):
        assert "bind_budget" in inspect.getsource(fn), f"{fn.__name__} binds no budget"


def test_a_harness_budget_is_not_replaced_by_the_default():
    """`current_budget() or default_budget()` -- a benchmark binds its own
    deadline and must keep it."""
    import inspect

    from agent.pipeline import run

    assert "current_budget() or default_budget()" in inspect.getsource(run.run_pipeline)


# --------------------------------------------------------------------------
# Multi-turn rationing
# --------------------------------------------------------------------------
#
# A multi-turn task is many runs sharing one budget, and nothing stopped the
# first run taking all of it. Measured on Claw-Eval: C03 and C04 each reached
# ~930 seconds and ~46 model calls and then stopped answering, with the graders
# saying so outright -- "failed to provide a response to the final three user
# prompts", "failed to provide the actual Python script requested". C04 scored
# 1.00 on gathering requirements and 0.10 on content, which is exactly the
# shape of a run that spent everything before the conversation reached its
# point.

def test_each_turn_gets_a_share_rather_than_the_lot():
    budget = Budget(max_model_calls=60)
    spent_per_turn = []

    for turns_left in (3, 2, 1):
        budget.begin_turn(turns_left)
        before = budget.calls
        while not budget.spent():
            budget.spend()
        spent_per_turn.append(budget.calls - before)

    assert spent_per_turn == [20, 20, 20]
    assert budget.calls == 60


def test_a_turn_that_finishes_early_hands_its_share_forward():
    """Recomputed from what is actually left, not from a fixed slice, so a
    cheap first turn buys the last one more room."""
    budget = Budget(max_model_calls=60)

    budget.begin_turn(3)
    for _ in range(4):          # used 4 of its 20
        budget.spend()
    budget.begin_turn(2)

    assert budget.turn_max_calls == 4 + (60 - 4) // 2


def test_the_last_turn_may_use_everything_that_is_left():
    budget = Budget(max_model_calls=30)
    budget.begin_turn(1)

    assert budget.turn_max_calls == 30


def test_an_ordinary_single_turn_run_is_unchanged():
    """Every `otto chat` turn and most benchmark tasks never call
    `begin_turn`, and must behave exactly as before."""
    budget = Budget(max_model_calls=10)

    assert budget.turn_max_calls is None
    while not budget.spent():
        budget.spend()
    assert budget.calls == 10


def test_a_later_turn_is_not_told_to_wrap_up_the_moment_it_starts():
    """The wrap-up point is a fraction of THIS turn's share. Measured against
    the whole run instead, turn two would open past 80% and be told to finish
    before it had done anything."""
    budget = Budget(max_model_calls=100)
    budget.begin_turn(2)
    while not budget.spent():
        budget.spend()

    budget.begin_turn(1)
    assert budget.phase() == "ok", "turn two opened in wrap-up"


def test_every_turn_gets_its_own_wrap_up_note():
    """Said once per RUN, a later turn would never be told to finish."""
    budget = Budget(max_model_calls=20)

    budget.begin_turn(2)
    while budget.phase() != "wrap_up":
        budget.spend()
    assert budget.wrap_up_once() is not None
    assert budget.wrap_up_once() is None, "said twice in one turn"

    budget.begin_turn(1)
    while budget.phase() != "wrap_up":
        budget.spend()
    assert budget.wrap_up_once() is not None, "turn two was never told to finish"


def test_the_run_ceiling_still_wins_over_a_turn_share():
    """A turn cannot be granted budget the run does not have."""
    budget = Budget(max_model_calls=5)
    budget.begin_turn(1)
    while not budget.spent():
        budget.spend()

    budget.begin_turn(1)
    assert budget.spent(), "a new turn resurrected an exhausted run"


def test_nothing_rations_per_turn_in_production():
    """`begin_turn` works and is deliberately UNUSED.

    It was written for issue #12, measured on C04, and made the task worse at
    every budget tried -- 0.43 unrationed against 0.32 rationed, with the
    grader reporting no assistant responses at all. A turn whose share runs
    out returns no answer, so rationing converted one mediocre answer into
    nine empty turns.

    This test is what stops it being switched back on without the missing
    half: a turn that ends its share by ANSWERING rather than by returning
    nothing. Delete this test in the same change that adds that.
    """
    import inspect

    from agent.eval import claw_bench

    source = inspect.getsource(claw_bench.run_one)
    called = [line for line in source.splitlines()
              if "begin_turn" in line and not line.lstrip().startswith("#")]
    assert called == [], called


# --------------------------------------------------------------------------
# The floor, which the first version did not have
# --------------------------------------------------------------------------
#
# Rationing 480 seconds across C04's nine turns gave each one 53 seconds. The
# grader's verdict on that run: "the provided conversation only contains user
# messages and lacks any responses from the assistant" -- 0.32, against 0.43
# for the unrationed run it was meant to improve. A share too small to answer
# with is worse than no rationing at all.

def test_a_turn_is_never_given_less_than_it_takes_to_answer():
    """A turn pays for the criteria call before the loop and the judgment
    after it, so a share of three calls is spent before any work happens."""
    budget = Budget(max_model_calls=20)
    budget.begin_turn(9)

    assert budget.turn_max_calls - budget.turn_start_calls >= bd.MIN_TURN_CALLS


def test_the_clock_share_has_a_floor_too():
    import time

    budget = Budget.of(480.0)
    budget.begin_turn(9)

    assert budget.turn_hard_at - time.monotonic() >= bd.MIN_TURN_SECONDS - 1


def test_a_budget_that_cannot_pay_for_every_turn_pays_for_fewer():
    """Fewer turns answered properly beats every turn answered with nothing.
    480 seconds buys five 90-second turns, not nine 53-second ones."""
    import time

    budget = Budget.of(480.0)
    budget.begin_turn(9)
    share = budget.turn_hard_at - time.monotonic()

    assert 90 <= share <= 110, f"{share:.0f}s is not a usable share"


def test_a_generous_budget_still_divides_evenly():
    """The floor is a floor, not a target -- it must not flatten a budget that
    could give every turn more than the minimum."""
    budget = Budget(max_model_calls=120)
    budget.begin_turn(4)

    assert budget.turn_max_calls == 30


def test_the_floor_does_not_resurrect_an_exhausted_run():
    budget = Budget(max_model_calls=5)
    while not budget.spent():
        budget.spend()

    budget.begin_turn(3)
    assert budget.spent(), "the floor handed out budget the run did not have"
