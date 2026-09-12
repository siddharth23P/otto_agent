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
