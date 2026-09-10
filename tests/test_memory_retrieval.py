"""Coverage for agent/memory/retrieval.py's recall() -- semantic search over
a `kind`'s CURRENT bullets, with graceful fallback to unranked recent
bullets whenever the local embedding model can't be reached (no bullets
have an embedding at all, or the live query-embedding call itself fails).
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
    monkeypatch.setattr(retr, "embed", lambda texts: [np.array([1.0, 0.0])])

    result = retr.recall(store, "history", "tell me about cats", top_k=1)

    assert "about cats" in result
    assert "cats chunk" in result
    assert "about dogs" not in result


def test_recall_falls_back_to_recent_unranked_when_no_bullet_has_an_embedding(store, monkeypatch):
    store.add_bullet("history", 1, "older", ["h1"], None)
    store.add_bullet("history", 2, "newer", ["h2"], None)
    store.add_chunk("history", "h1", "older chunk")
    store.add_chunk("history", "h2", "newer chunk")

    def _boom(texts):
        raise AssertionError("embed() should never be called with no embeddable bullets")

    monkeypatch.setattr(retr, "embed", _boom)

    result = retr.recall(store, "history", "anything", top_k=1)

    assert "newer" in result  # most recent of the (unranked) bullets
    assert "older" not in result


def test_recall_falls_back_to_recent_unranked_when_query_embedding_fails(store, monkeypatch):
    store.add_bullet("history", 1, "older", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 2, "newer", ["h2"], np.array([0.0, 1.0]))
    store.add_chunk("history", "h1", "older chunk")
    store.add_chunk("history", "h2", "newer chunk")

    def _raise(texts):
        raise EmbeddingUnavailable("model unavailable right now")

    monkeypatch.setattr(retr, "embed", _raise)

    result = retr.recall(store, "history", "anything", top_k=1)

    assert "newer" in result
    assert "older" not in result


def test_recall_only_pulls_chunks_the_picked_bullets_actually_cite(store, monkeypatch):
    store.add_bullet("history", 1, "bullet one", ["h1"], np.array([1.0, 0.0]))
    store.add_bullet("history", 1, "bullet two", ["h2"], np.array([0.0, 1.0]))
    store.add_chunk("history", "h1", "chunk one")
    store.add_chunk("history", "h2", "chunk two")
    store.add_chunk("history", "h3", "an unrelated chunk no bullet cites")

    monkeypatch.setattr(retr, "embed", lambda texts: [np.array([1.0, 0.0])])

    result = retr.recall(store, "history", "match bullet one", top_k=2)

    assert "chunk one" in result
    assert "chunk two" in result
    assert "an unrelated chunk no bullet cites" not in result
