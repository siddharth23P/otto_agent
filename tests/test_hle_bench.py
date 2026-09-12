"""Coverage for agent/eval/hle_bench.py -- the parts that must be right before
a single paid question is scored: judge parsing that fails closed, image
filtering that is reported rather than silent, deterministic sampling, and a
gated-dataset error that says what to do.

No network and no model: the two answer paths and the judge are the expensive
bits and are injected at their call sites, so everything here runs offline.
"""
import pytest

from agent.eval import hle_bench as hle


# ---- judging --------------------------------------------------------------


def test_a_clear_verdict_parses():
    assert hle.parse_judgement("extracted_final_answer: 42\ncorrect: yes") == ("42", True)
    assert hle.parse_judgement("extracted_final_answer: None\ncorrect: no") == ("None", False)


def test_an_unreadable_verdict_scores_wrong_not_right():
    """Fails closed, for the same reason the pipeline's own approval parse
    does: a judge that never rendered a verdict is not evidence the answer was
    correct."""
    for reply in ["", "the judge rambled", "correct: maybe", "CORRECT ANSWER!"]:
        _, correct = hle.parse_judgement(reply)
        assert correct is False, reply


def test_verdict_parsing_is_case_insensitive_and_ignores_surrounding_prose():
    _, correct = hle.parse_judgement("Reasoning: the units match.\nCorrect: YES\n")
    assert correct is True


# ---- sampling -------------------------------------------------------------


def test_image_questions_are_skipped_and_counted():
    """Otto's chat path is text-only, so scoring it on questions whose content
    it cannot see would measure the wrong thing -- but an image-heavy category
    dropping out silently would look like a difficulty change."""
    rows = [{"image": "", "id": "a"}, {"image": "http://x/y.png", "id": "b"}, {"image": "", "id": "c"}]

    kept, skipped = hle.sample_questions(rows, limit=None)

    assert [r["id"] for r in kept] == ["a", "c"]
    assert skipped == 1


def test_sampling_is_deterministic_so_two_modes_score_the_same_questions():
    rows = [{"image": "", "id": str(i)} for i in range(50)]

    first, _ = hle.sample_questions(rows, limit=10, seed=7)
    second, _ = hle.sample_questions(rows, limit=10, seed=7)
    different, _ = hle.sample_questions(rows, limit=10, seed=8)

    assert [r["id"] for r in first] == [r["id"] for r in second]
    assert [r["id"] for r in first] != [r["id"] for r in different]


def test_a_limit_larger_than_the_dataset_keeps_everything():
    rows = [{"image": "", "id": str(i)} for i in range(3)]

    kept, _ = hle.sample_questions(rows, limit=99)

    assert len(kept) == 3


# ---- the gate -------------------------------------------------------------


def test_a_missing_token_explains_the_gate_rather_than_failing_obscurely(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with pytest.raises(hle.DatasetGated) as excinfo:
        hle.download_hle(tmp_path / "absent.parquet")

    assert "huggingface.co/datasets/cais/hle" in str(excinfo.value)


def test_an_already_cached_parquet_needs_no_token(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cached = tmp_path / "hle.parquet"
    cached.write_bytes(b"not really a parquet, but present")

    assert hle.download_hle(cached) == cached


# ---- scoring --------------------------------------------------------------


def test_a_question_that_raises_is_scored_wrong_not_fatal(monkeypatch):
    """One bad question must not end a sweep that has already been paid for."""
    def _boom(question):
        raise RuntimeError("provider fell over")

    monkeypatch.setattr(hle, "_answer_raw", _boom)
    monkeypatch.setattr(hle, "judge", lambda q, r, a: ("", False))

    [result] = hle.run_hle([{"id": "1", "question": "q", "answer": "a"}], mode="raw")

    assert result.correct is False
    assert "provider fell over" in result.response


def test_the_summary_reports_calls_per_question_alongside_accuracy(monkeypatch):
    """The cost side is the point of the agent-vs-raw comparison: accuracy
    alone would hide a graph that spends thirty calls to match one."""
    monkeypatch.setattr(hle, "_answer_raw", lambda q: ("Answer: 42", 7))
    monkeypatch.setattr(hle, "judge", lambda q, r, a: ("42", a == "42"))

    results = hle.run_hle([
        {"id": "1", "category": "Math", "question": "q", "answer": "42"},
        {"id": "2", "category": "Math", "question": "q", "answer": "not 42"},
    ], mode="raw")
    report = hle.summarise(results)

    assert report["accuracy"] == 0.5
    assert report["llm_calls_per_question"] == 7.0
    assert report["by_category"]["Math"]["n"] == 2
