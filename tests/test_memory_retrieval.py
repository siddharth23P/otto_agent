"""Coverage for agent/memory/retrieval.py's recall() -- two-stage search: rank
a `kind`'s CURRENT bullets to narrow the store to a candidate hash list, then
rank the RAW chunks in that list and return only the best few, each with its
immediate neighbours. Both stages fall back to unranked-most-recent whenever
the local embedding model can't be reached (nothing has an embedding stored,
or the live query-embedding call itself fails).

`embed_query` is what gets monkeypatched throughout rather than `embed`:
recall() embeds the QUERY behind BGE's search instruction, never as plain
stored text (agent/memory/embeddings.py's own note on the asymmetry).
"""
import numpy as np
import pytest

import agent.memory.retrieval as retr
from agent.memory.embeddings import EmbeddingUnavailable
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    yield s
    s.close()


def test_recall_with_nothing_compacted_yet_says_so(store):
    result = retr.recall(store, "history", "anything")

    assert "nothing has been compacted away yet" in result


def test_recall_ranks_by_cosine_similarity_to_the_query(store, monkeypatch):
    store.add_bullet("history", 1, "about cats", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "about dogs", ["h2"], np.array([0.0, 1.0]))
    store.add_chunk("history", "h1", "cats chunk")
    store.add_chunk("history", "h2", "dogs chunk")

    # the query vector matches "about cats" exactly, "about dogs" not at all
    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "tell me about cats", top_k=1)

    assert "about cats" in result
    assert "cats chunk" in result
    assert "about dogs" not in result


def test_recall_falls_back_to_recent_unranked_when_no_bullet_has_an_embedding(store, monkeypatch):
    store.add_bullet("history", 1, "older", ["h1"], None)
    store.add_bullet("history", 2, "newer", ["h2"], None)
    store.add_chunk("history", "h1", "older chunk")
    store.add_chunk("history", "h2", "newer chunk")

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "anything", top_k=1, neighbour_window=0)

    assert "- newer" in result  # most recent of the (unranked) bullets
    assert "- older" not in result


def test_recall_falls_back_to_recent_unranked_when_query_embedding_fails(store, monkeypatch):
    store.add_bullet("history", 1, "older", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 2, "newer", ["h2"], np.array([0.0, 1.0]))
    store.add_chunk("history", "h1", "older chunk")
    store.add_chunk("history", "h2", "newer chunk")

    def _raise(query):
        raise EmbeddingUnavailable("model unavailable right now")

    monkeypatch.setattr(retr, "embed_query", _raise)

    result = retr.recall(store, "history", "anything", top_k=1, neighbour_window=0)

    assert "- newer" in result
    assert "- older" not in result


def test_recall_only_pulls_chunks_some_live_bullet_actually_cites(store, monkeypatch):
    store.add_bullet("history", 1, "bullet one", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "bullet two", ["h2"], np.array([0.0, 1.0]))
    store.add_chunk("history", "h1", "chunk one")
    store.add_chunk("history", "h2", "chunk two")
    store.add_chunk("history", "h3", "an unrelated chunk no bullet cites")

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "match bullet one", top_k=2, neighbour_window=0)

    assert "chunk one" in result
    assert "chunk two" in result
    assert "an unrelated chunk no bullet cites" not in result


def test_recall_searches_every_live_bullets_chunks_not_just_the_matched_ones(store, monkeypatch):
    """Compaction does not produce comparable bullets -- one accumulated bullet
    can cite hundreds of chunks while its neighbours cite two. Narrowing to the
    best-matching bullets first scored 5% on a live replay where searching all
    of their hashes scored 90%; bullets summarize, they do not index."""
    store.add_chunk("history", "h1", "a turn about sailing", np.array([1.0, 0.0]))
    store.add_chunk("history", "h2", "a turn about baking", np.array([0.0, 1.0]))
    # the bullet whose summary matches the query cites the WRONG chunk
    store.add_bullet("history", 1, "summary that matches the query", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "summary that does not", ["h2"], np.array([-1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([0.0, 1.0]))

    result = retr.recall(store, "history", "tell me about baking", top_k=1,
                         max_chunks=1, neighbour_window=0)

    assert "a turn about baking" in result


# ---- stage 2: the cap, the chunk ranking, and neighbour expansion --------


def test_recall_caps_how_many_chunks_one_matched_bullet_expands_to(store, monkeypatch):
    """A bullet accumulates citations forward, generation over generation, so
    a late one cites nearly everything -- uncapped, a single match returned the
    whole conversation (the module docstring's 38,033 characters)."""
    hashes = [f"h{i}" for i in range(20)]
    for i, h in enumerate(hashes):
        store.add_chunk("history", h, f"raw item {i}")
    store.add_bullet("history", 1, "one bullet citing everything", hashes, np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "anything", max_chunks=3, neighbour_window=0)

    assert result.count("  > ") == 3


def test_recall_ranks_the_raw_chunks_not_just_the_bullet(store, monkeypatch):
    store.add_chunk("history", "h1", "about sailing", np.array([1.0, 0.0]))
    store.add_chunk("history", "h2", "about baking", np.array([0.0, 1.0]))
    store.add_bullet("history", 1, "a bullet covering both", ["h1", "h2"], np.array([1.0, 1.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([0.0, 1.0]))

    result = retr.recall(store, "history", "tell me about baking", max_chunks=1, neighbour_window=0)

    assert "about baking" in result
    assert "about sailing" not in result


def test_recall_returns_the_turns_either_side_of_a_hit(store, monkeypatch):
    """The largest single retrieval win measured on the LoCoMo replay: the
    answer to a question about a turn is routinely in the NEXT turn."""
    store.add_chunk("history", "h0", "you: what did you do yesterday")
    store.add_chunk("history", "h1", "otto: I joined the group")
    store.add_chunk("history", "h2", "otto: it was on the 3rd of May")
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "when did you join", neighbour_window=1)

    assert "it was on the 3rd of May" in result  # never cited by any bullet
    assert "what did you do yesterday" in result


def test_recall_neighbours_are_ordered_oldest_first(store, monkeypatch):
    for i in range(5):
        store.add_chunk("history", f"h{i}", f"turn {i}")
    store.add_bullet("history", 1, "a bullet", ["h2"], np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "anything", neighbour_window=1)

    assert result.index("turn 1") < result.index("turn 2") < result.index("turn 3")


def test_recall_stops_at_the_token_budget(store, monkeypatch):
    """The count caps are a proxy for size; this is the ceiling that actually
    holds when a session's turns are long (an Otto reply, not a chat line)."""
    for i in range(10):
        store.add_chunk("history", f"h{i}", f"turn {i} " + "word " * 200, np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "a bullet", [f"h{i}" for i in range(10)], np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    generous = retr.recall(store, "history", "anything", neighbour_window=0, token_budget=10_000)
    tight = retr.recall(store, "history", "anything", neighbour_window=0, token_budget=500)

    assert generous.count("  > ") == 10
    assert 0 < tight.count("  > ") < 10


def test_recall_always_returns_at_least_the_best_chunk(store, monkeypatch):
    """A budget smaller than the single best chunk still returns it -- an empty
    result would be strictly worse than an oversized one."""
    store.add_chunk("history", "h1", "word " * 500, np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "anything", neighbour_window=0, token_budget=1)

    assert result.count("  > ") == 1


def test_recall_falls_back_to_recent_chunks_when_none_have_an_embedding(store, monkeypatch):
    for i in range(6):
        store.add_chunk("history", f"h{i}", f"turn {i}")  # no embeddings stored
    store.add_bullet("history", 1, "a bullet", [f"h{i}" for i in range(6)], np.array([1.0, 0.0]))

    monkeypatch.setattr(retr, "embed_query", lambda q: np.array([1.0, 0.0]))

    result = retr.recall(store, "history", "anything", max_chunks=2, neighbour_window=0)

    assert "turn 5" in result and "turn 4" in result
    assert "turn 0" not in result
