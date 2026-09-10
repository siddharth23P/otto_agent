"""Coverage for agent/eval/memory_bench.py -- entirely offline, using small
inline fake LoCoMo-shaped data rather than the real 2.8MB locomo10.json (this
repo's established offline-vs-live test split; the real download is exercised
manually, not in the test suite -- see the module's own docstring and
`download_locomo()`). Every conversation here is deliberately tiny, so the
"production budget" tests below rely on the SAME budgets memory_bench.py
defaults to (X_BUDGET/Y_BUDGET, imported from agent.memory.queue) being large
relative to a handful of short turns -- exactly the real finding a smoke run
against actual LoCoMo data produced (nothing compacts at production scale).
The stress-test tests instead pass tiny explicit x_budget/y_budget so
compaction is forced deterministically, without needing a large fixture.
"""
from __future__ import annotations

from agent.eval.memory_bench import (
    BenchmarkReport,
    CATEGORY_NAMES,
    ConversationResult,
    QAResult,
    _canned_summarize,
    _is_reachable,
    _is_verbatim,
    _normalize,
    _session_keys,
    run_benchmark,
    run_one_conversation,
)
from agent.memory.queue import TieredQueue
from agent.memory.store import MemoryStore


def _fake_sample(sample_id="test-1", *, qa=None):
    """A minimal but shape-correct LoCoMo sample: two short sessions, four
    dialogue turns total, one QA item per scored category plus one
    adversarial item (to confirm it's excluded from scoring)."""
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Alice",
            "speaker_b": "Bob",
            "session_1": [
                {"speaker": "Alice", "dia_id": "D1:1", "text": "I adopted a cat named Whiskers."},
                {"speaker": "Bob", "dia_id": "D1:2", "text": "That's great, what color is she?"},
            ],
            "session_2": [
                {"speaker": "Alice", "dia_id": "D2:1", "text": "Whiskers is orange and white."},
                {"speaker": "Bob", "dia_id": "D2:2", "text": "Cute! Did you get her from a shelter?"},
            ],
        },
        "qa": qa if qa is not None else [
            {"question": "What is the cat's name?", "answer": "Whiskers", "evidence": ["D1:1"], "category": 1},
            {"question": "When did Alice describe the cat's color?", "answer": "session 2",
             "evidence": ["D2:1"], "category": 2},
            {"question": "What color is the cat Alice adopted?", "answer": "orange and white",
             "evidence": ["D1:1", "D2:1"], "category": 3},
            {"question": "What pet does Alice have?", "answer": "a cat", "evidence": ["D1:1"], "category": 4},
            {"question": "What breed is the dog?", "evidence": ["D1:1"], "category": 5,
             "adversarial_answer": "a labrador"},
        ],
    }


# ---- _session_keys ---------------------------------------------------

def test_session_keys_sorts_numerically_not_lexically():
    conversation = {"session_2": [], "session_10": [], "session_1": [], "speaker_a": "x"}
    assert _session_keys(conversation) == ["session_1", "session_2", "session_10"]


# ---- _canned_summarize -------------------------------------------------

def test_canned_summarize_returns_empty_for_a_prompt_with_no_numbered_items():
    assert _canned_summarize("nothing numbered here") == ""


def test_canned_summarize_covers_every_item_in_groups_of_four_with_citations():
    prompt = "\n".join(f"[{i}] item text {i}" for i in range(1, 7))
    reply = _canned_summarize(prompt)
    lines = reply.splitlines()
    assert len(lines) == 2  # 6 items, group_size=4 -> groups of 4 and 2
    assert "[sources: 1,2,3,4]" in lines[0]
    assert "[sources: 5,6]" in lines[1]


# ---- QAResult.answerable ------------------------------------------------

def test_qa_result_answerable_true_when_visible_verbatim():
    r = QAResult(question="q", category=1, evidence_ids=["D1:1"], stored=True,
                 visible_verbatim=True, recalled=False)
    assert r.answerable is True


def test_qa_result_answerable_true_when_recalled_but_not_verbatim():
    r = QAResult(question="q", category=1, evidence_ids=["D1:1"], stored=True,
                 visible_verbatim=False, recalled=True)
    assert r.answerable is True


def test_qa_result_answerable_false_when_neither():
    r = QAResult(question="q", category=1, evidence_ids=["D1:1"], stored=True,
                 visible_verbatim=False, recalled=False)
    assert r.answerable is False


def test_qa_result_evidence_and_recalled_text_default_to_empty_string():
    # Callers that construct a QAResult directly without them (every test
    # above this one) must still get a valid instance -- these two fields
    # exist purely for --show-items debug output, not for `answerable`.
    r = QAResult(question="q", category=1, evidence_ids=["D1:1"], stored=True,
                 visible_verbatim=True, recalled=False)
    assert r.evidence_text == ""
    assert r.recalled_text == ""


# ---- _is_verbatim / _is_reachable --------------------------------------

def test_is_verbatim_true_for_text_still_in_x(tmp_path):
    store = MemoryStore(tmp_path / "s.db")
    try:
        queue = TieredQueue("k", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)
        queue.append("hello")
        assert _is_verbatim(queue, "hello") is True
        assert _is_verbatim(queue, "never appended") is False
    finally:
        store.close()


def test_is_reachable_true_for_compacted_but_permanently_stored_text(tmp_path):
    store = MemoryStore(tmp_path / "s.db")
    try:
        # x_budget=0 forces every append straight into Y; y_budget=0 forces
        # an immediate compaction on the very next append.
        queue = TieredQueue("k", store, summarize=_canned_summarize, x_budget=0, y_budget=0)
        queue.append("first item")
        queue.append("second item")

        assert _is_verbatim(queue, "first item") is False  # compacted away
        assert _is_reachable(store, queue, "first item") is True  # but permanent
    finally:
        store.close()


# ---- _normalize ----------------------------------------------------------

def test_normalize_collapses_whitespace_and_lowercases():
    assert _normalize("  Whiskers   is\norange ") == "whiskers is orange"


# ---- run_one_conversation, production-sized budgets ---------------------

def test_run_one_conversation_at_production_budget_never_compacts():
    result = run_one_conversation(_fake_sample(), summarize=_canned_summarize)

    assert result.turn_count == 4
    assert result.raw_tokens > 0
    # Nothing compacted -- current_view() is still just the raw turns (plus
    # a little "RECENT:" formatting overhead, so not an exact 1.0 ratio).
    assert result.compression_ratio < 1.5

    scored = result.scored_results()
    assert len(scored) == 4  # adversarial (category 5) excluded
    assert all(r.stored for r in scored)
    assert all(r.visible_verbatim for r in scored)  # still all in X
    assert all(r.answerable for r in scored)
    assert all(not r.recalled for r in scored)  # nothing compacted to recall from


def test_run_one_conversation_populates_evidence_text_and_recalled_text():
    # These two fields are what lets a person eyeball whether a `recalled`
    # verdict is a real semantic-search find or a coincidental substring
    # match -- both must actually be populated, not left at their default
    # empty string, on every scored result.
    result = run_one_conversation(_fake_sample(), summarize=_canned_summarize)

    for r in result.scored_results():
        assert r.evidence_text  # non-empty -- the raw cited turn(s)
        assert r.recalled_text  # non-empty -- recall() always returns
        # something, even just "(nothing has been compacted away yet...)"
    # Spot-check one item's evidence_text actually IS the cited turn text.
    single_hop = next(r for r in result.scored_results() if r.category == 1)
    assert "Whiskers" in single_hop.evidence_text


def test_run_one_conversation_skips_qa_items_with_no_evidence_in_range():
    sample = _fake_sample(qa=[
        {"question": "orphan evidence", "evidence": ["D9:9"], "category": 1},
        {"question": "no evidence key at all", "category": 1},
    ])
    result = run_one_conversation(sample, summarize=_canned_summarize)
    assert result.qa_results == []


def test_run_one_conversation_max_turns_truncates_and_skips_later_evidence():
    result = run_one_conversation(_fake_sample(), summarize=_canned_summarize, max_turns=2)

    assert result.turn_count == 2
    # Only D1:1/D1:2 exist now -- QA items citing D2:* evidence are skipped.
    remaining_evidence = {e for r in result.qa_results for e in r.evidence_ids}
    assert remaining_evidence <= {"D1:1", "D1:2"}


# ---- run_one_conversation, stress-test (forced compaction) --------------

def test_run_one_conversation_stress_budget_forces_compaction_and_still_stores_everything():
    result = run_one_conversation(
        _fake_sample(), summarize=_canned_summarize, x_budget=1, y_budget=1,
    )

    scored = result.scored_results()
    assert len(scored) == 4
    # The engine-correctness guarantee holds even under aggressive
    # compaction: nothing cited is ever actually lost.
    assert all(r.stored for r in scored)
    # But it's no longer all sitting verbatim in the live view -- the whole
    # point of the stress-test budget is to force real compaction.
    assert not all(r.visible_verbatim for r in scored)
    assert result.compression_ratio < 1.0


# ---- run_benchmark / BenchmarkReport -------------------------------------

def test_run_benchmark_aggregates_multiple_conversations():
    report = run_benchmark([_fake_sample("a"), _fake_sample("b")])

    assert report.live is False
    assert len(report.conversations) == 2
    assert {c.sample_id for c in report.conversations} == {"a", "b"}


def test_benchmark_report_summary_excludes_adversarial_and_breaks_down_by_category():
    report = run_benchmark([_fake_sample()])
    summary = report.summary()

    assert summary["conversation_count"] == 1
    assert summary["overall"]["n"] == 4  # 5 QA items minus 1 adversarial
    assert set(summary["by_category"]) == {
        CATEGORY_NAMES[1], CATEGORY_NAMES[2], CATEGORY_NAMES[3], CATEGORY_NAMES[4],
    }
    for rates in summary["by_category"].values():
        assert rates["n"] == 1
    assert summary["overall"]["answerable_coverage"] == 1.0
    assert summary["overall"]["visible_verbatim_coverage"] == 1.0


def test_benchmark_report_summary_handles_zero_scored_results_without_dividing_by_zero():
    report = run_benchmark([_fake_sample(qa=[])])
    summary = report.summary()

    assert summary["overall"]["n"] == 0
    assert summary["overall"]["store_coverage"] is None
    assert summary["overall"]["answerable_coverage"] is None


def test_benchmark_report_to_dict_has_per_conversation_and_summary_sections():
    report = run_benchmark([_fake_sample()])
    d = report.to_dict()

    assert d["live"] is False
    assert len(d["conversations"]) == 1
    conv = d["conversations"][0]
    assert conv["sample_id"] == "test-1"
    assert conv["turn_count"] == 4
    assert len(conv["qa"]) == 5  # to_dict() includes ALL qa items, adversarial too
    assert "summary" in d


def test_conversation_result_compression_ratio_is_zero_for_an_empty_conversation():
    result = ConversationResult(
        sample_id="empty", turn_count=0, raw_tokens=0, final_view_tokens=0, qa_results=[],
    )
    assert result.compression_ratio == 0.0
