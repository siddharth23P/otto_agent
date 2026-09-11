"""Offline coverage for agent/eval/claw_bench.py.

The Claw-Eval checkout is not a dependency of this repo, so nothing here
imports it. What IS testable without it is everything the harness decides for
itself before any of their code runs: when the clock cuts a run short, which
tasks need a container, which need a key this machine does not have, and what
the agent is actually asked to do.

The two-stage deadline gets the most attention because getting it wrong is
invisible in a passing run and expensive in a failing one -- a task killed
mid-command reports nothing, where one told to wrap up writes its finding
down.
"""
import time
from types import SimpleNamespace

import pytest

from agent.eval import claw_bench as cb


def _task(**kw):
    """The handful of TaskDefinition fields this module reads, as a stub."""
    env = SimpleNamespace(
        timeout_seconds=kw.pop("timeout_seconds", 300),
        max_turns=kw.pop("max_turns", 20),
        mock_today=kw.pop("mock_today", None),
        fixtures=kw.pop("fixtures", []),
    )
    return SimpleNamespace(
        task_id=kw.pop("task_id", "T001_example"),
        environment=env,
        services=kw.pop("services", []),
        sandbox_files=kw.pop("sandbox_files", []),
        sandbox_grader_files=kw.pop("sandbox_grader_files", []),
        env_snapshot_files=kw.pop("env_snapshot_files", []),
        env_snapshot_commands=kw.pop("env_snapshot_commands", []),
        prompt=SimpleNamespace(text=kw.pop("prompt", "Do the thing."), attachments=[]),
        **kw,
    )


# --------------------------------------------------------------------------
# The deadline
# --------------------------------------------------------------------------

def test_a_fresh_deadline_lets_a_tool_run():
    assert cb.Deadline.of(300).check() is None


def test_past_the_wrap_up_point_a_tool_refuses_instead_of_running():
    now = time.monotonic()
    d = cb.Deadline(hard_at=now + 60, wrap_up_at=now - 1)
    result = d.check()
    assert result is not None
    stdout, stderr, code = result
    assert code == 1
    assert "final answer now" in stderr


def test_past_the_hard_point_it_refuses_rather_than_raising():
    """It used to raise. agent/pipeline/budget.py now owns ending a run, and it
    ends it by answering -- raising here unwinds the graph out from under the
    budget and loses the work. Seen on T026: a 50-second model call let the
    clock pass between the loop's own budget check and the tool dispatch right
    after it, and the whole run went down."""
    now = time.monotonic()
    d = cb.Deadline(hard_at=now - 1, wrap_up_at=now - 2)
    stdout, stderr, code = d.check()
    assert code == 1
    assert "answer now" in stderr


def test_the_wrap_up_stage_comes_before_the_hard_stage():
    """If they coincided there would be no chance to answer -- the whole
    point of the first stage."""
    d = cb.Deadline.of(300)
    assert d.wrap_up_at < d.hard_at


def test_a_short_budget_still_leaves_a_usable_window():
    """A budget under the safety margin must not produce a deadline already
    in the past, which would fail every task before it started."""
    d = cb.Deadline.of(1)
    assert d.hard_at > time.monotonic()


# --------------------------------------------------------------------------
# What the environment can and cannot run
# --------------------------------------------------------------------------

def test_a_task_with_container_fixtures_needs_one():
    assert cb.needs_container(_task(sandbox_files=["fixtures/config.json"]))


def test_a_task_graded_from_container_screenshots_needs_one():
    assert cb.needs_container(_task(env_snapshot_files=["/workspace/frames/*.png"]))


def test_a_pure_service_task_does_not_need_a_container():
    assert not cb.needs_container(_task(services=[SimpleNamespace(name="gmail")]))


def test_a_service_reaching_the_real_internet_is_reported_when_its_key_is_absent(monkeypatch):
    monkeypatch.delenv("SERP_DEV_KEY", raising=False)
    task = _task(services=[SimpleNamespace(name="web_real")])
    assert cb.missing_service_keys(task) == ["SERP_DEV_KEY"]


def test_nothing_is_reported_once_the_key_is_there(monkeypatch):
    monkeypatch.setenv("SERP_DEV_KEY", "k")
    task = _task(services=[SimpleNamespace(name="web_real")])
    assert cb.missing_service_keys(task) == []


def test_a_fixture_backed_service_needs_no_key():
    assert cb.missing_service_keys(_task(services=[SimpleNamespace(name="gmail")])) == []


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

def test_the_prompt_starts_with_the_task_text():
    assert cb.build_prompt(_task(prompt="Book the room."), in_container=False).startswith("Book the room.")


def test_a_container_task_is_told_where_its_tools_act():
    """Otto's file tools reach the container through a bound CommandRunner;
    an agent that thinks it is on its own machine looks in the wrong place."""
    prompt = cb.build_prompt(_task(sandbox_files=["a"]), in_container=True)
    assert "/workspace" in prompt and "container" in prompt


def test_a_non_container_task_is_not_told_it_is_in_one():
    assert "container" not in cb.build_prompt(_task(), in_container=False)


def test_a_mocked_date_reaches_the_agent():
    """Half these tasks are about scheduling; an agent using the real date
    answers a different question from the one the grader asks."""
    assert "2025-03-14" in cb.build_prompt(_task(mock_today="2025-03-14"), in_container=False)


# --------------------------------------------------------------------------
# Missing checkout
# --------------------------------------------------------------------------

def test_a_missing_checkout_is_a_sentence_not_a_traceback(monkeypatch):
    monkeypatch.delenv("CLAW_EVAL_ROOT", raising=False)
    with pytest.raises(cb.ClawEvalUnavailable) as exc:
        cb.claw_root(None)
    assert "--claw-root" in str(exc.value)


def test_a_path_that_is_not_a_checkout_says_so(tmp_path):
    with pytest.raises(cb.ClawEvalUnavailable) as exc:
        cb.claw_root(tmp_path)
    assert "does not look like" in str(exc.value)


# --------------------------------------------------------------------------
# Tool results
# --------------------------------------------------------------------------

def test_a_short_tool_result_is_untouched():
    assert cb._clip("hello") == "hello"


def test_a_long_tool_result_keeps_both_ends():
    """Mock services return whole inboxes. Keeping only the head loses the
    record the agent was reaching for as often as not."""
    text = "A" * 4000 + "MIDDLE" + "Z" * 4000
    clipped = cb._clip(text)
    assert clipped.startswith("A") and clipped.endswith("Z")
    assert "characters omitted" in clipped
    assert len(clipped) < len(text)


def test_the_final_answer_falls_back_to_a_specialists_candidate():
    """A run stopped on its deadline mid-review still did the work; reporting
    nothing for it would blame the agent for the clock."""
    assert cb._final_text({"final_output": None, "output": "the number is 42"}) == "the number is 42"


def test_a_finished_run_reports_its_own_final_output():
    assert cb._final_text({"final_output": "done", "output": "draft"}) == "done"


def test_no_state_at_all_is_an_empty_answer_not_a_crash():
    assert cb._final_text(None) == ""


def test_a_task_runs_for_its_own_budget_by_default():
    assert cb.task_budget(_task(timeout_seconds=900)) == 900.0


def test_a_budget_cap_lowers_a_generous_task_budget():
    """Tasks here allow 120 to 900 seconds and the hard ones use all of it,
    so a sweep at full budget is hours. The cap is what makes a sample
    affordable -- and it lowers scores, which is why it is reported."""
    assert cb.task_budget(_task(timeout_seconds=900), 120) == 120.0


def test_a_cap_above_the_task_budget_does_not_raise_it():
    """A ceiling, never a floor -- a 120s task given --max-seconds 600 must
    still stop at 120, or the run stops matching the benchmark."""
    assert cb.task_budget(_task(timeout_seconds=120), 600) == 120.0


# --------------------------------------------------------------------------
# Which task tools change something
# --------------------------------------------------------------------------
#
# Claw-Eval's specs do not say -- every endpoint is a POST -- so the name is
# all there is. The bias is deliberate: over-gating a read costs one model
# call, under-gating a send is T026.

def test_sending_and_creating_are_treated_as_irreversible():
    for name in ("gmail_send_message", "calendar_create_event", "crm_update_contact",
                 "finance_transfer", "todo_delete_task", "ticket_assign"):
        assert cb.tool_mutates(name), f"{name} was not gated"


def test_reading_is_not_gated():
    for name in ("gmail_list_messages", "gmail_get_message", "web_search",
                 "finance_list_transactions", "contacts_search", "kb_get_article"):
        assert not cb.tool_mutates(name), f"{name} was gated needlessly"


def test_an_unrecognised_tool_is_gated():
    """The safe direction. A benchmark tool nobody anticipated is treated as
    irreversible until someone says otherwise."""
    assert cb.tool_mutates("frobnicate_widget")


def test_a_write_verb_wins_over_a_read_verb_in_the_same_name():
    """`get_or_create` creates."""
    assert cb.tool_mutates("get_or_create_record")
