"""Routing that learns from what it observed, and the restraints on it.

Milestone 4's routing item. Logging verified per-(task, model) outcomes and
routing on them is worth about +15.3% relative, and it is the one form of
adaptation that costs no model calls -- the evidence is a by-product of runs
that were happening anyway.

Every threshold here exists to stop the router acting on noise. On a benchmark
whose own scores swing 0.36 between identical runs, a chain reordered from
three samples is a chain reordered by a coin.
"""
import pytest

from agent.router import outcomes as o
from agent.router.mapping import Candidate, Task


@pytest.fixture
def seat_log(tmp_path):
    with o.bind_log(tmp_path / "outcomes.db"):
        yield


def _fill(task, model_id, *, runs, approved, calls_each=10):
    for i in range(runs):
        o.record(task, model_id, approved=i < approved, calls=calls_each)


def _ids(candidates):
    return [c.spec for c in candidates]


A = Candidate(spec="vendor:model-a")
B = Candidate(spec="vendor:model-b")


# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

def test_nothing_recorded_reads_back_as_nothing(seat_log):
    assert o.records("reason") == []
    assert o.preference("reason") == {}


def test_runs_accumulate_into_one_row(seat_log):
    _fill("reason", "model-a", runs=5, approved=3)

    (row,) = o.records("reason")
    assert (row.runs, row.approved) == (5, 3)
    assert row.approval_rate == pytest.approx(0.6)


def test_calls_are_recorded_beside_the_verdict(seat_log):
    """Two models that succeed equally often are not equally good, and the
    difference has been measured as large as 31x on one task."""
    _fill("reason", "model-a", runs=2, approved=2, calls_each=40)

    assert o.records("reason")[0].calls_per_run == pytest.approx(40)


def test_a_row_under_the_sample_floor_is_not_offered_for_routing(seat_log):
    """`preference` leaves it out entirely rather than returning it with a
    caveat -- a caller that has to remember to check is one that will forget."""
    _fill("reason", "model-a", runs=o.MIN_SAMPLES - 1, approved=o.MIN_SAMPLES - 1)

    assert o.records("reason")[0].runs == o.MIN_SAMPLES - 1
    assert o.preference("reason") == {}


def test_the_log_can_be_turned_off(tmp_path):
    """A measurement that holds routing fixed has to be able to say so, or the
    thing being measured changes underneath it."""
    with o.bind_log(None):
        o.record("reason", "model-a", approved=True)
        assert o.records("reason") == []


def test_read_only_routes_on_the_log_without_adding_to_it(tmp_path):
    with o.bind_log(tmp_path / "outcomes.db"):
        _fill("reason", "model-a", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)
        with o.read_only():
            o.record("reason", "model-a", approved=False)
            assert "model-a" in o.preference("reason"), "read path was closed too"
        assert o.records("reason")[0].runs == o.MIN_SAMPLES


# --------------------------------------------------------------------------
# Reordering
# --------------------------------------------------------------------------

def test_an_empty_log_leaves_the_declared_order_exactly_as_written(seat_log):
    """A fresh install, a test, and a route nobody has exercised all behave
    the way mapping.py reads."""
    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [A.spec, B.spec]


def test_a_clearly_better_second_candidate_overtakes(seat_log):
    _fill("reason", "model-a", runs=o.MIN_SAMPLES, approved=2)
    _fill("reason", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)

    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [B.spec, A.spec]


def test_a_difference_inside_the_margin_does_not_move_anything(seat_log):
    """At the sample floor one run is worth 8 points, so a small lead is not
    a lead."""
    _fill("reason", "model-a", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES - 1)
    _fill("reason", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)

    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [A.spec, B.spec]


def test_the_same_result_for_measurably_fewer_calls_wins(seat_log):
    _fill("reason", "model-a", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES, calls_each=40)
    _fill("reason", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES, calls_each=12)

    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [B.spec, A.spec]


def test_an_unmeasured_candidate_is_never_unseated_by_a_measured_one(seat_log):
    """"We measured B at 100%" is not evidence that A is worse -- A has no
    number at all. The declared order encodes a person's judgment, and
    unseating it needs evidence on both sides."""
    _fill("reason", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)

    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [A.spec, B.spec]


def test_a_query_candidate_keeps_its_place(seat_log):
    """The log is keyed on a model. A capability query is not one yet."""
    query = Candidate(provider="vendor")
    _fill("reason", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)

    order = o.reorder("reason", [query, B], o.spec_id)
    assert order[0] is query


def test_another_tasks_evidence_does_not_reorder_this_one(seat_log):
    """Seats are separate. A model that judges well is not thereby a better
    solver, and the table is keyed on both."""
    _fill("evaluate", "model-b", runs=o.MIN_SAMPLES, approved=o.MIN_SAMPLES)
    _fill("evaluate", "model-a", runs=o.MIN_SAMPLES, approved=0)

    assert _ids(o.reorder("reason", [A, B], o.spec_id)) == [A.spec, B.spec]


def test_the_router_actually_consults_the_log(monkeypatch, seat_log):
    """Not a unit of the sorter. The real Router.resolve, so a future change
    that stops consulting the log fails here rather than quietly going back to
    declared order."""
    from agent.router import router as rr

    asked: list = []

    def spy(task, candidates, *args, **kwargs):
        asked.append(task)
        return list(candidates)

    monkeypatch.setattr(rr.seat_outcomes, "reorder", spy)
    try:
        rr.Router().resolve(Task.REASON)
    except Exception:
        pass  # no credentials here; what matters is that it asked

    assert asked == [Task.REASON.value]


# --------------------------------------------------------------------------
# Saying WHY the chain is in the order it is in
# --------------------------------------------------------------------------

def test_a_candidate_chosen_on_evidence_is_not_reported_as_degraded():
    """`fell_back` used to be `index > 0`, which was the same question until
    the chain could be reordered. After it could, the router's best-evidenced
    choice was displayed as a failure -- "degraded" in `otto route` and in the
    TUI -- because nothing ahead of it had actually been skipped."""
    from agent.router.router import RoutingDecision

    on_evidence = RoutingDecision(
        task=Task.REASON, provider="v", model=None, endpoint=None,
        params={}, index=1, skipped=(),
    )
    assert on_evidence.chosen_on_evidence
    assert not on_evidence.fell_back


def test_a_real_fallback_is_still_reported_as_one():
    from agent.router.router import RoutingDecision, Skip

    degraded = RoutingDecision(
        task=Task.REASON, provider="v", model=None, endpoint=None,
        params={}, index=1, skipped=(Skip(0, "vendor:model-a", "no key"),),
    )
    assert degraded.fell_back
    assert not degraded.chosen_on_evidence
