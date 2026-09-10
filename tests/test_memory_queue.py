"""Coverage for agent/memory/queue.py's TieredQueue -- the X/Y tiered
short-term-memory engine (2026-09-10 design call). Token counting is
monkeypatched to plain `len` throughout so budget math is deterministic on
short test strings, and the local embedding model is monkeypatched to
always raise EmbeddingUnavailable (autouse) so tests never touch the real
network-dependent model -- one test overrides that per-scenario to prove
the "embedding available" path also works.

Several tests reach into TieredQueue's own `_x`/`_y_raw`/`_y_bullets` and
call `_compact_y_if_full()` directly rather than driving everything through
`append()`, specifically for the multi-generation citation-carry-forward
scenarios -- those are about the compaction step's own invariants, and
white-boxing them keeps the arithmetic legible instead of reverse-engineering
budget numbers that happen to trigger the right sequence of overflows.
"""
import numpy as np
import pytest

import agent.memory.queue as q
from agent.memory.embeddings import EmbeddingUnavailable
from agent.memory.hashing import content_hash
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _deterministic_tokens(monkeypatch):
    monkeypatch.setattr(q, "count_tokens", len)


@pytest.fixture(autouse=True)
def _no_real_embeddings(monkeypatch):
    def _raise(texts):
        raise EmbeddingUnavailable("no embeddings in tests")

    monkeypatch.setattr(q, "embed", _raise)


# ---- append() / X-overflow-to-Y ----------------------------------------


def test_append_ignores_empty_text(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=10, y_budget=10)

    tq.append("")
    tq.append(None)

    assert not tq.has_content


def test_append_stays_in_x_until_budget_exceeded(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=10, y_budget=100)

    tq.append("12345")  # 5 tokens
    tq.append("123")  # +3 = 8, still <= 10

    assert tq._y_raw == []
    assert tq._x == ["12345", "123"]


def test_x_overflow_moves_everything_to_y_in_one_step(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=5, y_budget=100)

    tq.append("12345")  # 5 tokens, stays (== budget)
    tq.append("1")  # now 6 > 5 -> whole of X moves to Y at once

    assert tq._x == []
    assert tq._y_raw == ["12345", "1"]
    assert tq._y_bullets == []


# ---- compaction: first generation ---------------------------------------


def test_full_compaction_flushes_raw_chunks_and_produces_cited_bullets(store):
    def _summarize(prompt):
        assert "[1] 12345" in prompt
        assert "[2] 1" in prompt
        assert "[3] abcdef" in prompt
        return "First summary [sources: 1,2]\nSecond summary [sources: 3]"

    tq = q.TieredQueue("history", store, summarize=_summarize, x_budget=5, y_budget=10)

    tq.append("12345")
    tq.append("1")
    tq.append("abcdef")  # pushes y_tokens from 6 to 12 > y_budget(10) -> compacts

    h1, h2, h3 = content_hash("12345"), content_hash("1"), content_hash("abcdef")
    assert store.get_chunks([h1, h2, h3]) == {h1: "12345", h2: "1", h3: "abcdef"}

    bullets = store.current_bullets("history")
    assert [b.text for b in bullets] == ["First summary", "Second summary"]
    assert bullets[0].hash_refs == sorted([h1, h2])
    assert bullets[1].hash_refs == [h3]
    assert bullets[0].embedding is None  # embeddings unavailable in this test

    assert tq._y_raw == []
    assert [b.text for b in tq._y_bullets] == ["First summary", "Second summary"]


def test_compaction_stores_embedding_when_the_model_is_available(store, monkeypatch):
    monkeypatch.setattr(q, "embed", lambda texts: [np.array([1.0, 0.0])])

    tq = q.TieredQueue("history", store, summarize=lambda p: "summary [sources: 1]", x_budget=1000, y_budget=1)
    tq._y_raw = ["raw"]
    tq._compact_y_if_full()

    [bullet] = store.current_bullets("history")
    assert bullet.embedding is not None
    assert np.allclose(bullet.embedding, [1.0, 0.0])


def test_unparseable_summary_falls_back_to_one_bullet_covering_everything(store):
    tq = q.TieredQueue(
        "history", store, summarize=lambda p: "not a bullet list at all", x_budget=1000, y_budget=5,
    )
    tq._y_raw = ["raw one", "raw two"]

    tq._compact_y_if_full()

    h1, h2 = content_hash("raw one"), content_hash("raw two")
    assert len(tq._y_bullets) == 1
    assert "unparsed summary of 2 items" in tq._y_bullets[0].text
    assert tq._y_bullets[0].hash_refs == sorted([h1, h2])


# ---- compaction: second generation carries hash_refs forward ------------


def test_second_compaction_carries_prior_bullet_hash_refs_forward_unchanged(store):
    calls = []

    def _summarize(prompt):
        calls.append(prompt)
        if len(calls) == 1:
            return "first summary [sources: 1,2]"
        return "combined [sources: 1,2]"

    tq = q.TieredQueue("history", store, summarize=_summarize, x_budget=1000, y_budget=5)

    # generation 1: two raw items compact into one bullet.
    tq._y_raw = ["raw one", "raw two"]
    tq._compact_y_if_full()  # y_tokens = 14 > 5

    h1, h2 = content_hash("raw one"), content_hash("raw two")
    assert [b.text for b in tq._y_bullets] == ["first summary"]
    assert tq._y_bullets[0].hash_refs == sorted([h1, h2])

    # generation 2: the prior bullet (cited as item 1) plus a new raw item
    # (item 2) compact again.
    tq._y_raw = ["raw three"]
    tq._compact_y_if_full()  # y_tokens = len("first summary") + len("raw three") > 5

    h3 = content_hash("raw three")
    assert [b.text for b in tq._y_bullets] == ["combined"]
    assert tq._y_bullets[0].hash_refs == sorted([h1, h2, h3])

    # the first bullet's own TEXT is never itself hashed as a leaf chunk --
    # only the raw content it transitively cites is real, permanent storage.
    assert store.get_chunks([content_hash("first summary")]) == {}
    assert store.get_chunks([h1, h2, h3]) == {h1: "raw one", h2: "raw two", h3: "raw three"}

    # generation 1's bullet is superseded, not deleted -- current_bullets()
    # only surfaces the live generation.
    assert {b.text for b in store.current_bullets("history")} == {"combined"}


# ---- current_view() ------------------------------------------------------


def test_current_view_shows_only_recent_when_nothing_has_overflowed(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq.append("hello")

    assert tq.current_view() == "RECENT:\nhello"


def test_recent_items_returns_x_as_a_plain_list(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq.append("first")
    tq.append("second")

    assert tq.recent_items == ["first", "second"]
    # a plain, independent copy -- mutating it must not touch the queue
    tq.recent_items.append("third")
    assert tq.recent_items == ["first", "second"]


def test_earlier_view_is_empty_when_only_x_has_content(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq.append("recent stuff")

    assert tq.earlier_view() == ""


def test_earlier_view_matches_current_view_minus_recent(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq._y_bullets = [q.NewBullet(text="a summary", hash_refs=["h1"])]
    tq._y_raw = ["not yet summarized text"]
    tq._x = ["newest text"]

    assert tq.earlier_view() == "EARLIER (summarized):\n- a summary\n\nEARLIER (not yet summarized):\nnot yet summarized text"
    assert tq.current_view() == tq.earlier_view() + "\n\nRECENT:\nnewest text"


def test_current_view_includes_y_raw_not_yet_summarized(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq._y_raw = ["earlier stuff"]
    tq._x = ["recent stuff"]

    view = tq.current_view()

    assert "EARLIER (not yet summarized):\nearlier stuff" in view
    assert "RECENT:\nrecent stuff" in view
    assert view.index("EARLIER") < view.index("RECENT")


def test_current_view_orders_bullets_before_raw_before_recent(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq._y_bullets = [q.NewBullet(text="a summary", hash_refs=["h1"])]
    tq._y_raw = ["not yet summarized text"]
    tq._x = ["newest text"]

    view = tq.current_view()

    assert (
        view.index("EARLIER (summarized)")
        < view.index("EARLIER (not yet summarized)")
        < view.index("RECENT")
    )
    assert "- a summary" in view
    assert "not yet summarized text" in view
    assert "newest text" in view


# ---- has_content -----------------------------------------------------------


def test_has_content_false_when_empty(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=10, y_budget=10)

    assert not tq.has_content


def test_has_content_true_with_only_x(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq.append("something")

    assert tq.has_content


def test_has_content_true_with_only_y_raw(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq._y_raw = ["stuff"]

    assert tq.has_content


def test_has_content_true_with_only_y_bullets(store):
    tq = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=1000, y_budget=1000)
    tq._y_bullets = [q.NewBullet(text="x", hash_refs=[])]

    assert tq.has_content


# ---- _parse_bullets() ------------------------------------------------------


def test_parse_bullets_drops_lines_without_a_sources_tag():
    parsed = q._parse_bullets("a bullet with no tag\nreal one [sources: 1]", item_count=1)

    assert parsed == [("real one", [1])]


def test_parse_bullets_drops_out_of_range_indices():
    parsed = q._parse_bullets("x [sources: 1,5]", item_count=2)

    assert parsed == [("x", [1])]


def test_parse_bullets_drops_lines_whose_indices_are_all_out_of_range():
    parsed = q._parse_bullets("x [sources: 9]", item_count=2)

    assert parsed == []


def test_parse_bullets_strips_leading_bullet_markers():
    parsed = q._parse_bullets("- x [sources: 1]\n* y [sources: 2]", item_count=2)

    assert parsed == [("x", [1]), ("y", [2])]


# ---- budget constants -------------------------------------------------------


def test_budget_constants_match_the_40_percent_of_260k_spec():
    assert q.MERCURY_2_5_CONTEXT_WINDOW == 260_000
    assert q.TOTAL_BUDGET == 104_000
    assert q.TOTAL_BUDGET == int(260_000 * 0.40)
    assert q.X_BUDGET + q.Y_BUDGET == q.TOTAL_BUDGET
