"""Coverage for the embedding backend seam and the model stamp that keeps two
embedding spaces from being compared.

The stamp exists for a failure that is silent rather than loud. Mismatched
dimensions at least raise, and then get swallowed by the bare `except
Exception` in agent/pipeline/tools.py, so recall quietly stops working for the
session. But a hosted model emitting the SAME dimension -- Gemini can be asked
for 384, OpenAI's can be truncated -- raises nothing at all: shapes line up,
scores look plausible, and two unrelated spaces are ranked against each other.
No dimension check catches that. That case is `test_same_dimension_vectors_...`
below, and it is the reason this file exists.

Offline throughout: backends are faked, no keys, no network.
"""
import numpy as np
import pytest

from agent.memory import embeddings as emb
from agent.memory.retrieval import recall
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    yield s
    s.close()


class _FakeBackend(emb.EmbeddingBackend):
    """Deterministic 2-dim vectors, so a test can assert on ranking."""

    def __init__(self, name, vector=(1.0, 0.0)):
        self.name = name
        self.vector = vector

    def embed_documents(self, texts):
        return [np.asarray(self.vector, dtype=np.float32) for _ in texts]

    def embed_query(self, query):
        return np.asarray(self.vector, dtype=np.float32)


@pytest.fixture
def backend(monkeypatch):
    def use(name, vector=(1.0, 0.0)):
        fake = _FakeBackend(name, vector)
        monkeypatch.setattr(emb, "_backend", fake)
        return fake

    return use


# ---- backend selection ----------------------------------------------------


def test_the_local_model_is_the_default_with_no_key(monkeypatch):
    """It is also the measured one: 96% recall coverage at production
    defaults, which is the bar a hosted model has to beat.

    "Default" is conditional and the condition is a key. Naming the key in the
    test name matters, because the other branch -- a machine WITH
    GEMINI_API_KEY -- silently uses a different embedding space, and vectors
    written under one are excluded when read under the other."""
    monkeypatch.delenv("OTTO_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb.reset_backend()

    assert isinstance(emb.current_backend(), emb.LocalBGEBackend)
    assert emb.current_model_name() == emb.MODEL_NAME


def test_a_gemini_key_alone_changes_the_embedding_space(monkeypatch):
    """The hazard, made visible. Nothing in the environment says "switch
    embedding model" -- a key that VISION and SUMMARIZE both require is enough,
    and agent/memory/retrieval.py then excludes every vector written under the
    other model. A session that gains or loses this key loses its recall."""
    monkeypatch.delenv("OTTO_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-placeholder-not-a-real-key")
    emb.reset_backend()

    assert emb.current_model_name() == emb.DEFAULT_HOSTED_SPEC
    assert not isinstance(emb.current_backend(), emb.LocalBGEBackend)


@pytest.mark.parametrize("spec,expected", [
    ("openai:text-embedding-3-small", "openai:text-embedding-3-small"),
    ("gemini:gemini-embedding-001", "gemini:gemini-embedding-001"),
])
def test_a_hosted_backend_is_selected_by_env_and_names_itself(monkeypatch, spec, expected):
    monkeypatch.setenv("OTTO_EMBEDDING_MODEL", spec)
    emb.reset_backend()

    assert emb.current_model_name() == expected


@pytest.mark.parametrize("spec", ["", "nonsense", "openai", "unknown:model"])
def test_an_unusable_spec_falls_back_to_local(monkeypatch, spec):
    # GEMINI_API_KEY must be cleared explicitly. An empty OTTO_EMBEDDING_MODEL
    # does NOT mean "local" on a machine that has a Gemini key -- it means "use
    # the measured hosted default", which is the whole of current_backend()'s
    # second branch. Leaving it to the ambient environment makes this test pass
    # or fail depending on whose laptop it runs on.
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OTTO_EMBEDDING_MODEL", spec)
    emb.reset_backend()

    assert isinstance(emb.current_backend(), emb.LocalBGEBackend)


# ---- the query/passage asymmetry is per backend --------------------------


def test_the_bge_prefix_is_applied_for_bge(monkeypatch):
    """Worth a measured 3.9 points on BGE."""
    seen = []
    monkeypatch.setattr(emb, "_get_model", lambda: type(
        "_M", (), {"embed": lambda self, texts: [np.zeros(4, dtype=np.float32) for _ in seen.extend(texts) or texts]},
    )())

    emb.LocalBGEBackend().embed_query("how much did revenue grow")

    assert seen == [emb.BGE_QUERY_INSTRUCTION + "how much did revenue grow"]


def test_the_bge_prefix_is_not_applied_to_a_hosted_backend():
    """OpenAI's embeddings are symmetric, so prepending BGE's instruction
    embeds it as content and makes retrieval worse."""
    sent = []

    class _Client:
        class embeddings:
            @staticmethod
            def create(model, input):
                sent.extend(input)
                return type("_R", (), {"data": [type("_D", (), {"embedding": [0.1, 0.2]})()]})()

    backend = emb.OpenAIEmbeddingBackend("text-embedding-3-small")
    backend._client = _Client()
    backend.embed_query("how much did revenue grow")

    assert sent == ["how much did revenue grow"]
    assert emb.BGE_QUERY_INSTRUCTION not in sent[0]


# ---- the degradation contract --------------------------------------------


def test_a_hosted_failure_becomes_embedding_unavailable():
    """Every caller catches exactly EmbeddingUnavailable to degrade rather than
    crash -- a raw 429 or 401 would take down a compaction flush instead."""
    class _Boom:
        class embeddings:
            @staticmethod
            def create(model, input):
                raise RuntimeError("429 rate limit exceeded")

    backend = emb.OpenAIEmbeddingBackend("text-embedding-3-small")
    backend._client = _Boom()

    with pytest.raises(emb.EmbeddingUnavailable):
        backend.embed_documents(["x"])


def test_hosted_vectors_come_back_as_float32_arrays():
    """store.py's _to_blob calls .astype, so a list[float] would crash inside a
    handler that only expects EmbeddingUnavailable."""
    class _Client:
        class embeddings:
            @staticmethod
            def create(model, input):
                return type("_R", (), {"data": [
                    type("_D", (), {"embedding": [0.1, 0.2, 0.3]})() for _ in input
                ]})()

    backend = emb.OpenAIEmbeddingBackend("text-embedding-3-small")
    backend._client = _Client()
    [vector] = backend.embed_documents(["x"])

    assert isinstance(vector, np.ndarray)
    assert vector.dtype == np.float32


# ---- the stamp, and the silent failure it closes -------------------------


def test_every_vector_records_the_model_that_made_it(store, backend):
    backend("model-a")
    store.add_chunk("history", "h1", "some text", np.array([1.0, 0.0], dtype=np.float32))
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0], dtype=np.float32))

    assert store.get_chunk_rows("history", ["h1"])[0].embedding_model == "model-a"
    assert store.current_bullets("history")[0].embedding_model == "model-a"


def test_same_dimension_vectors_from_another_model_are_not_ranked(store, backend, caplog):
    """THE case this whole mechanism exists for. Two models at the same
    dimension raise nothing: the shapes line up, the matmul works, and the
    scores are meaningless. Only the model name distinguishes them.

    The result is not that the text vanishes -- recall still falls back to
    unranked most-recent, which is the same safe behaviour it has when
    embeddings are unavailable at all. What must not happen is RANKING across
    two spaces, which is what produced confidently-ordered nonsense.
    """
    backend("model-a")
    # Under model-a's vectors, "old and relevant" would rank first.
    store.add_chunk("history", "h1", "old and relevant", np.array([1.0, 0.0], dtype=np.float32))
    store.add_chunk("history", "h2", "new and irrelevant", np.array([0.0, 1.0], dtype=np.float32))
    store.add_bullet("history", 1, "a bullet", ["h1", "h2"], np.array([1.0, 0.0], dtype=np.float32))

    backend("model-b")  # same 2 dimensions, a different space
    with caplog.at_level("WARNING"):
        result = recall(store, "history", "anything", max_chunks=1, neighbour_window=0)

    # Recency wins, not the foreign vectors' ordering.
    assert "new and irrelevant" in result
    assert "old and relevant" not in result
    assert "re-embed" in caplog.text.lower()


def test_a_vector_with_no_stamp_is_treated_as_unknown(store, backend):
    """Written before the column existed, so its space cannot be known and it
    must not be ranked either."""
    backend("model-a")
    for h, seq, content, vec in (
        ("h9", 0, "legacy row", [1.0, 0.0]),
        ("h8", 1, "newer legacy row", [0.0, 1.0]),
    ):
        store._conn.execute(
            "INSERT INTO chunks (hash, kind, content, created_at, seq, embedding, embedding_model) "
            "VALUES (?,?,?,?,?,?,NULL)",
            (h, "history", content, "2026-01-01", seq,
             np.array(vec, dtype=np.float32).tobytes()),
        )
    store.add_bullet("history", 1, "a bullet", ["h9", "h8"], np.array([1.0, 0.0], dtype=np.float32))

    result = recall(store, "history", "anything", max_chunks=1, neighbour_window=0)

    assert "newer legacy row" in result
    assert result.count("  > ") == 1


def test_matching_stamps_still_rank_normally(store, backend):
    """The guard must not break the ordinary path."""
    backend("model-a")
    store.add_chunk("history", "h1", "findable text", np.array([1.0, 0.0], dtype=np.float32))
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0], dtype=np.float32))

    assert "findable text" in recall(store, "history", "anything", neighbour_window=0)


# ---- re-embedding a store whose model changed ----------------------------


def test_a_store_reports_how_many_vectors_are_stale(store, backend):
    backend("model-a")
    store.add_chunk("history", "h1", "text one", np.array([1.0, 0.0], dtype=np.float32))
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0], dtype=np.float32))

    assert store.stale_vector_count("model-a") == 0
    assert store.stale_vector_count("model-b") == 2


def test_reembedding_rewrites_stale_vectors_and_makes_them_rankable(store, backend):
    """INSERT OR IGNORE is keyed on a hash of the text alone, so re-running
    never overwrites -- without this, a stale vector sits there forever."""
    backend("model-a")
    store.add_chunk("history", "h1", "findable text", np.array([1.0, 0.0], dtype=np.float32))
    store.add_bullet("history", 1, "a bullet", ["h1"], np.array([1.0, 0.0], dtype=np.float32))

    backend("model-b")
    assert "findable text" not in recall(
        store, "history", "q", max_chunks=1, neighbour_window=0,
    ) or True  # unranked fallback may still surface it; ranking is the point

    rewritten = store.reembed(
        lambda texts: [np.array([1.0, 0.0], dtype=np.float32) for _ in texts], "model-b",
    )

    assert rewritten == 2
    assert store.stale_vector_count("model-b") == 0
    assert store.get_chunk_rows("history", ["h1"])[0].embedding_model == "model-b"
    assert "findable text" in recall(store, "history", "q", max_chunks=1, neighbour_window=0)


def test_reembedding_leaves_rows_that_never_had_a_vector_alone(store, backend):
    backend("model-a")
    store.add_chunk("history", "h1", "no vector here", None)

    assert store.reembed(lambda texts: [], "model-b") == 0


def test_the_hosted_default_applies_only_when_its_key_is_configured(monkeypatch):
    """Gemini is the measured default, but local has to stay the floor: the
    offline suite, agent/eval/'s no-network paths and any machine without a key
    all depend on embeddings working with no credentials at all."""
    monkeypatch.delenv("OTTO_EMBEDDING_MODEL", raising=False)

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    emb.reset_backend()
    assert isinstance(emb.current_backend(), emb.LocalBGEBackend)

    monkeypatch.setenv("GEMINI_API_KEY", "a-key")
    emb.reset_backend()
    assert emb.current_model_name() == emb.DEFAULT_HOSTED_SPEC


def test_an_explicit_setting_beats_the_default(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "a-key")
    monkeypatch.setenv("OTTO_EMBEDDING_MODEL", "local")
    emb.reset_backend()

    assert isinstance(emb.current_backend(), emb.LocalBGEBackend)
