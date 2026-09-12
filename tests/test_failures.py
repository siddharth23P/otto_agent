"""Coverage for the failure taxonomy and the tool-cost tally
(agent/eval/failures.py).

Two questions a score cannot answer, both read off `actions` -- the lossy
one-line-per-tool-call record every run already keeps, prefixed with the mode
that made the call. No model call anywhere in here, which is the constraint
that makes the whole thing applicable to a batch that already finished.

What is tested is the READING, not the taxonomy's choice of names: that a
line the parser cannot understand is dropped rather than guessed at, that a
cascade is distinguished from a retry, and that the per-seat split survives
being summed across a batch.
"""
import pytest

from agent.eval import failures
from agent.eval.failures import (
    ERROR_CASCADE, FIRST_CALL_FAILED, NO_ACTIONS, ZERO_WRITE,
    classify, distribution, parse_actions, summarise, tool_cost,
    total_tool_cost,
)


def _ok(seat, tool, target="x"):
    return f"{seat}: {tool} {target} -> ok: fine"


def _bad(seat, tool, target="x"):
    return f"{seat}: {tool} {target} -> FAILED (exit 1): boom"


# --------------------------------------------------------------------------
# Parsing the record
# --------------------------------------------------------------------------

def test_a_real_action_line_parses_into_seat_tool_target_and_outcome():
    line = "solve: write_file TUI_IMPROVEMENTS.md -> ok: wrote it (162 lines)"

    action, = parse_actions([line])

    assert (action.seat, action.tool, action.failed) == ("solve", "write_file", False)
    assert action.target == "TUI_IMPROVEMENTS.md"


def test_a_failing_line_parses_as_failed():
    line = "solve: execute_bash rustc main.rs -> FAILED (exit 1): error[E0433]"

    action, = parse_actions([line])

    assert action.failed is True
    assert action.tool == "execute_bash"


def test_a_delegated_seat_keeps_its_own_label():
    # A subtask's spend is not the parent seat's, and collapsing them would
    # hide the one thing the per-seat split is for.
    action, = parse_actions(["solve(delegated): read_file x -> ok: 1"])

    assert action.seat == "solve(delegated)"


def test_a_line_with_no_seat_prefix_still_yields_its_tool():
    action, = parse_actions(["read_file x -> ok: 1"])

    assert action.tool == "read_file"
    assert action.seat == "unknown"


@pytest.mark.parametrize("line", [
    "",
    "otto has an answer",
    "evaluator rejected agent: 0/1 criteria met",
    "switched to plan mode",
    "compacted 3 older tool result(s)",
])
def test_a_line_this_cannot_read_is_dropped_not_guessed_at(line):
    # These are real board lines. A parser that guessed a tool out of one
    # would corrupt every count downstream, and silently.
    assert parse_actions([line]) == []


def test_parsing_survives_being_handed_nothing():
    assert parse_actions(None) == []
    assert parse_actions([]) == []


# --------------------------------------------------------------------------
# Failure kinds
# --------------------------------------------------------------------------

def test_a_run_with_no_tool_calls_is_tagged_as_such():
    # Distinct from zero_write and worth its own name: a run that answered
    # without touching anything either did not need to, or never got going.
    assert classify([]) == [NO_ACTIONS]
    assert classify(None) == [NO_ACTIONS]


def test_a_run_that_never_edited_anything_is_a_zero_write():
    actions = [_ok("solve", "list_files", "."), _ok("solve", "read_file", "a.py")]

    assert ZERO_WRITE in classify(actions)


def test_a_run_that_wrote_a_file_is_not_a_zero_write():
    actions = [_ok("solve", "read_file", "a.py"), _ok("solve", "write_file", "a.py")]

    assert ZERO_WRITE not in classify(actions)


def test_a_failed_write_does_not_count_as_having_written():
    actions = [_ok("solve", "read_file"), _bad("solve", "write_file")]

    assert ZERO_WRITE in classify(actions)


def test_running_a_shell_command_is_not_counted_as_an_edit():
    # execute_bash can write, but it is also how everything gets CHECKED, so
    # counting it as a write would mean no run ever had zero writes -- and
    # the tag would be dead.
    assert ZERO_WRITE in classify([_ok("solve", "execute_bash", "pytest")])


def test_a_failing_first_call_is_tagged():
    assert FIRST_CALL_FAILED in classify([_bad("solve", "read_file"),
                                          _ok("solve", "read_file")])


def test_a_working_first_call_is_not_tagged_even_if_later_ones_fail():
    tags = classify([_ok("solve", "read_file"), _bad("solve", "execute_bash")])

    assert FIRST_CALL_FAILED not in tags


def test_two_failures_in_a_row_are_a_retry_not_a_cascade():
    actions = [_bad("solve", "execute_bash")] * 2

    assert ERROR_CASCADE not in classify(actions)


def test_three_failures_in_a_row_are_a_cascade():
    # The signature of a run that stopped reading its own tool results.
    actions = [_bad("solve", "execute_bash")] * failures.CASCADE_LENGTH

    assert ERROR_CASCADE in classify(actions)


def test_failures_spread_out_are_not_a_cascade():
    actions = [
        _bad("solve", "execute_bash"), _ok("solve", "read_file"),
        _bad("solve", "execute_bash"), _ok("solve", "read_file"),
        _bad("solve", "execute_bash"),
    ]

    assert ERROR_CASCADE not in classify(actions)


def test_tags_are_not_mutually_exclusive():
    # The overlap is the useful part: a run whose opening move failed, which
    # then failed twice more and changed nothing, is three facts.
    actions = [_bad("solve", "execute_bash")] * 3

    assert set(classify(actions)) == {FIRST_CALL_FAILED, ZERO_WRITE, ERROR_CASCADE}


def test_a_distribution_counts_each_tag_across_runs():
    runs = [
        [_ok("solve", "write_file")],                       # nothing wrong
        [_ok("solve", "read_file")],                        # zero_write
        [_bad("solve", "execute_bash")] * 3,                # all three
    ]

    counts = distribution(runs)

    assert counts[ZERO_WRITE] == 2
    assert counts[ERROR_CASCADE] == 1
    assert NO_ACTIONS not in counts


# --------------------------------------------------------------------------
# What the tools cost
# --------------------------------------------------------------------------

def test_tool_calls_are_counted_by_tool_and_by_seat():
    actions = [
        _ok("solve", "read_file"), _ok("solve", "read_file"),
        _ok("plan", "read_file"), _bad("solve", "execute_bash"),
    ]

    cost = tool_cost(actions)

    assert cost.calls == 4
    assert cost.by_tool == {"read_file": 3, "execute_bash": 1}
    assert cost.by_seat == {"solve": 3, "plan": 1}


def test_failures_are_counted_separately_from_calls():
    # A tool called a lot that works is a different cost from one called a
    # lot that does not.
    actions = [_ok("solve", "execute_bash"), _bad("solve", "execute_bash")]

    cost = tool_cost(actions)

    assert cost.by_tool["execute_bash"] == 2
    assert cost.failures_by_tool == {"execute_bash": 1}
    assert cost.failures == 1


def test_the_busiest_tool_comes_first():
    actions = [_ok("solve", "read_file")] * 3 + [_ok("solve", "write_file")]

    assert list(tool_cost(actions).by_tool) == ["read_file", "write_file"]


def test_a_batch_sums_and_re_sorts_rather_than_keeping_arrival_order():
    runs = [
        [_ok("solve", "write_file")],
        [_ok("plan", "read_file")] * 5,
    ]

    total = total_tool_cost(runs)

    assert total.calls == 6
    assert list(total.by_tool) == ["read_file", "write_file"]
    assert total.by_seat == {"plan": 5, "solve": 1}


def test_a_batch_of_nothing_totals_to_nothing():
    total = total_tool_cost([None, []])

    assert total.calls == 0
    assert total.to_dict()["by_tool"] == {}


# --------------------------------------------------------------------------
# The batch summary
# --------------------------------------------------------------------------

def test_a_summary_names_which_runs_carry_each_tag():
    # A distribution nobody can trace back to a trace is a statistic rather
    # than a lead.
    result = summarise({
        "T001": [_ok("solve", "write_file")],
        "T002": [_ok("solve", "read_file")],
        "T003": [],
    })

    assert result["runs_by_kind"][ZERO_WRITE] == ["T002"]
    assert result["runs_by_kind"][NO_ACTIONS] == ["T003"]
    assert result["failures_by_kind"][ZERO_WRITE] == 1


def test_a_summary_carries_the_tool_cost_too():
    result = summarise({"T001": [_ok("solve", "read_file")]})

    assert result["tool_cost"]["calls"] == 1


def test_the_kinds_are_ordered_most_common_first():
    result = summarise({
        "a": [_ok("solve", "read_file")],
        "b": [_ok("solve", "read_file")],
        "c": [],
    })

    assert list(result["failures_by_kind"])[0] == ZERO_WRITE
