"""Coverage for agent/memory/embeddings.py -- the local fastembed wrapper.
Fakes fastembed entirely (via sys.modules) rather than requiring the real
model/network, since this sandbox's egress proxy blocks Hugging Face Hub
(see the module's own docstring) and the real model shouldn't be a test
dependency regardless.
"""
import sys

import numpy as np
import pytest

import agent.memory.embeddings as emb


@pytest.fixture(autouse=True)
def _reset_model_cache():
    """Every test starts from a clean lazy-load state -- these module
    globals are exactly what _get_model() caches across calls."""
    emb._model = None
    emb._model_load_failed = False
    yield
    emb._model = None
    emb._model_load_failed = False


def test_embed_empty_list_short_circuits_without_loading_a_model(monkeypatch):
    def _boom():
        raise AssertionError("should never be called for an empty input")

    monkeypatch.setattr(emb, "_get_model", _boom)

    assert emb.embed([]) == []


def test_embed_returns_one_vector_per_text_in_order(monkeypatch):
    class _FakeModel:
        def embed(self, texts):
            return [np.array([float(len(t)), 0.0, 0.0]) for t in texts]

    monkeypatch.setattr(emb, "_get_model", lambda: _FakeModel())

    vectors = emb.embed(["ab", "abcd"])

    assert len(vectors) == 2
    assert vectors[0][0] == 2.0
    assert vectors[1][0] == 4.0


def test_get_model_wraps_a_loading_failure_as_embedding_unavailable(monkeypatch):
    class _FakeFastembedModule:
        class TextEmbedding:
            def __init__(self, model_name):
                raise RuntimeError("no network")

    monkeypatch.setitem(sys.modules, "fastembed", _FakeFastembedModule())

    with pytest.raises(emb.EmbeddingUnavailable):
        emb._get_model()


def test_get_model_caches_failure_and_does_not_retry(monkeypatch):
    attempts = []

    class _FakeFastembedModule:
        class TextEmbedding:
            def __init__(self, model_name):
                attempts.append(model_name)
                raise RuntimeError("no network")

    monkeypatch.setitem(sys.modules, "fastembed", _FakeFastembedModule())

    with pytest.raises(emb.EmbeddingUnavailable):
        emb._get_model()
    with pytest.raises(emb.EmbeddingUnavailable):
        emb._get_model()

    assert len(attempts) == 1


def test_embed_wraps_a_run_time_failure_as_embedding_unavailable(monkeypatch):
    class _FakeModel:
        def embed(self, texts):
            raise RuntimeError("onnxruntime blew up")

    monkeypatch.setattr(emb, "_get_model", lambda: _FakeModel())

    with pytest.raises(emb.EmbeddingUnavailable):
        emb.embed(["hello"])


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])

    assert emb.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])

    assert emb.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_does_not_crash():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 2.0])

    assert emb.cosine_similarity(a, b) == 0.0
