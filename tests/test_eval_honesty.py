"""The three places an eval was able to report a number that meant nothing.

Each of these is a REPORTING rule rather than a measurement change, and each
was a real reading somebody could have quoted:

  * `otto eval-memory` at its default budgets never compacted LoCoMo's
    shorter conversations, so retrieval was never asked a question. It
    printed `recalled 0%` beside `answerable 100%` and a token ratio of
    1.000, which reads as a perfect score and is an empty one.

  * `otto eval-claw` printed "mean score" for a single run per task, when
    identical code and configuration have scored 0.36 apart on one task.
    Much of an optimisation branch was read that way.

  * `otto eval-swe` printed one resolve rate over a sample that was part
    native and part emulated. django is 231 of the 500 instances and had no
    arm64 build in every instance tried, so about half the benchmark runs
    several times slower on Apple silicon.
"""
import pytest

from agent.cli import eval_claw
from agent.eval import memory_bench, swe_bench


# --------------------------------------------------------------------------
# eval-memory: refuse to report when nothing was compacted
# --------------------------------------------------------------------------

@pytest.mark.parametrize("final,raw,exercised", [
    (1000, 1000, False),   # the reported failure: ratio 1.000
    (995, 1000, False),    # within token-counting slop of "nothing happened"
    (69, 1000, True),      # the ratio a forced-compaction run produced
    (0, 0, False),         # no conversations at all
])
def test_whether_recall_was_exercised_follows_the_compression_ratio(final, raw, exercised):
    assert memory_bench._recall_was_exercised(final, raw) is exercised


def test_the_threshold_is_below_one_not_equal_to_it():
    # Token counting is approximate; a view can come back a hair under raw
    # without a single compaction having happened.
    assert 0.9 < memory_bench.NO_COMPACTION_RATIO < 1.0


# --------------------------------------------------------------------------
# eval-claw: a single run is not evidence
# --------------------------------------------------------------------------

class _Outcome:
    def __init__(self, trials):
        self.trials = list(trials)
        self.task_score = sum(trials) / len(trials)
        self.completion = self.robustness = self.communication = self.task_score
        self.passed = self.task_score > 0.5
        self.error = ""
        self.wall_time_s = 100.0
        self.model_calls = 10


def test_a_single_run_per_task_is_not_marked_as_evidence():
    summary = eval_claw._summary([_Outcome([0.86]), _Outcome([0.60])])

    assert summary["trials"] == 1
    assert summary["is_evidence"] is False


def test_enough_trials_is_marked_as_evidence():
    trials = eval_claw.MIN_TRIALS_FOR_EVIDENCE
    summary = eval_claw._summary([_Outcome([0.86] * trials)])

    assert summary["trials"] == trials
    assert summary["is_evidence"] is True


def test_the_trial_count_is_always_reported_even_for_one_run():
    # Carried in the JSON, not inferred from whether a reliability key
    # happens to exist -- the JSON is what gets pasted into a comparison
    # months later.
    assert "trials" in eval_claw._summary([_Outcome([0.5])])


def test_the_evidence_threshold_is_more_than_one_trial():
    # A threshold of 1 would make every run evidence and the flag pointless.
    assert eval_claw.MIN_TRIALS_FOR_EVIDENCE > 1


def test_the_observed_spread_is_recorded_so_the_warning_can_cite_it():
    # The reason a reader is told to distrust a single run is a measurement,
    # not an opinion: T093 scored 0.86 / 0.60 / 0.96 on identical code.
    assert eval_claw.OBSERVED_SINGLE_RUN_SPREAD == pytest.approx(0.36)


def test_an_empty_run_reports_no_tasks_rather_than_dividing_by_zero():
    assert eval_claw._summary([]) == {"tasks": 0}


# --------------------------------------------------------------------------
# eval-swe: emulated instances get a wider budget, and the report says so
# --------------------------------------------------------------------------

def test_an_emulated_instance_gets_a_wider_budget_than_a_native_one():
    # A budget tuned for native turns "several times slower" into a recorded
    # agent failure, on roughly half the benchmark.
    assert swe_bench.EMULATION_TIME_FACTOR > 1.0


def test_the_manifest_probe_is_cached_because_each_miss_is_a_round_trip():
    # resolve_image is now asked twice per instance -- once to size the
    # deadline, once by start_container -- and each miss is a `docker
    # manifest inspect`. The cache is on the IMAGE NAME rather than on
    # resolve_image, which would capture host_arch() in the key.
    assert hasattr(swe_bench._manifest_exists, "cache_clear")
    assert not hasattr(swe_bench.resolve_image, "cache_clear")


def test_resolve_image_still_reports_emulation_after_the_cache_was_added(monkeypatch):
    swe_bench._manifest_exists.cache_clear()
    monkeypatch.setattr(swe_bench, "host_arch", lambda: "arm64")
    monkeypatch.setattr(swe_bench, "_manifest_exists", lambda image: False)

    image, emulated = swe_bench.resolve_image("django__django-10097")

    assert emulated is True
    assert ".x86_64." in image
