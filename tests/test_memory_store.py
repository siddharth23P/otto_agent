"""Coverage for agent/memory/store.py's MemoryStore -- the SQLite-backed,
content-addressed chunk table plus per-`kind` current-bullets table that
agent/memory/queue.py reads and writes directly (no LangGraph state
channels involved).
"""
import numpy as np
import pytest

import agent.memory.store as store_module
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    yield s
    s.close()


def test_add_and_get_chunk_round_trips(store):
    store.add_chunk("history", "hash1", "some raw text")

    assert store.get_chunks(["hash1"]) == {"hash1": "some raw text"}


def test_get_chunks_omits_unknown_hashes(store):
    store.add_chunk("history", "hash1", "some raw text")

    result = store.get_chunks(["hash1", "does-not-exist"])

    assert result == {"hash1": "some raw text"}


def test_get_chunks_of_empty_list_is_empty_dict(store):
    assert store.get_chunks([]) == {}


def test_add_chunk_is_idempotent_on_same_hash(store):
    store.add_chunk("history", "hash1", "first")
    store.add_chunk("history", "hash1", "second")  # INSERT OR IGNORE

    assert store.get_chunks(["hash1"]) == {"hash1": "first"}


def test_add_and_get_bullet_round_trips_including_embedding(store):
    embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    store.add_bullet("history", 1, "summary text", ["h1", "h2"], embedding)

    [bullet] = store.current_bullets("history")
    assert bullet.text == "summary text"
    assert bullet.hash_refs == ["h1", "h2"]
    assert bullet.generation == 1
    assert bullet.kind == "history"
    assert np.allclose(bullet.embedding, embedding)


def test_bullet_without_embedding_round_trips_as_none(store):
    store.add_bullet("history", 1, "summary text", ["h1"], None)

    [bullet] = store.current_bullets("history")
    assert bullet.embedding is None


def test_supersede_bullets_excludes_them_from_current_bullets(store):
    store.add_bullet("history", 1, "old summary", ["h1"], None)
    store.add_bullet("history", 2, "new summary", ["h1", "h2"], None)

    store.supersede_bullets("history", before_generation=2)

    [bullet] = store.current_bullets("history")
    assert bullet.text == "new summary"
    assert bullet.generation == 2


def test_supersede_bullets_is_scoped_to_generations_strictly_before(store):
    store.add_bullet("history", 1, "gen1", ["h1"], None)
    store.add_bullet("history", 2, "gen2", ["h2"], None)
    store.add_bullet("history", 3, "gen3", ["h3"], None)

    store.supersede_bullets("history", before_generation=2)

    texts = {b.text for b in store.current_bullets("history")}
    assert texts == {"gen2", "gen3"}


def test_kind_namespaces_are_independent(store):
    store.add_bullet("history", 1, "history summary", ["h1"], None)
    store.add_bullet("context", 1, "context summary", ["h2"], None)

    history_texts = {b.text for b in store.current_bullets("history")}
    context_texts = {b.text for b in store.current_bullets("context")}

    assert history_texts == {"history summary"}
    assert context_texts == {"context summary"}


def test_for_session_builds_a_path_under_db_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DB_DIR", tmp_path / "otto-memory")

    store = MemoryStore.for_session("some-session-id")
    try:
        assert store.path == tmp_path / "otto-memory" / "some-session-id.db"
    finally:
        store.close()
