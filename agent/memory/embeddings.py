"""A local, offline embedding model for semantic recall over agent/memory/
store.py's current bullets (agent/memory/retrieval.py) -- 2026-09-10
design call.

Inception has no embeddings endpoint at all (checked their docs directly:
chat, FIM, edit, and model-listing are the only REST families they
document -- no `/v1/embeddings`, no mention of one being planned). This
repo also made a deliberate, documented call to go Inception-only for
every LLM call (agent/router/mapping.py's own module docstring: "going all
in with Mercury," every other vendor provider removed). Reaching for
fastembed here doesn't reopen that decision -- an embedding isn't an LLM
call, and fastembed runs a small (~50MB of dependencies, no torch) ONNX
model entirely on-device, not a second vendor's hosted API.

Lazily constructed and genuinely optional at runtime: a process that never
calls recall_memory (agent/pipeline/tools.py -- not yet wired up, see the
project's design doc) never loads the model at all. Its FIRST use needs
network access once, to download the model from Hugging Face Hub (cached
under the platform default afterward, one-time, per machine) -- built and
tested in a sandboxed dev environment whose egress allowlist blocks that
host, so this deliberately fails toward EmbeddingUnavailable rather than
crashing whatever called it; agent/memory/retrieval.py's own fallback
(module docstring there) is what keeps a caller usable either way. A live
`otto chat`/`otto tui` run, with ordinary internet access, downloads the
model once and caches it from then on.
"""
from __future__ import annotations

import threading
from typing import Iterable

import numpy as np

#: BAAI/bge-small-en-v1.5 -- 384-dim, a well-established small/fast default
#: for local semantic search; fastembed ships it as one of its built-in
#: supported models (no custom ONNX export needed).
MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingUnavailable(Exception):
    """The local embedding model couldn't be loaded or run -- most often:
    no network on its first-ever use, to download the model once."""


_lock = threading.Lock()
_model = None
_model_load_failed = False


def _get_model():
    global _model, _model_load_failed
    if _model is not None:
        return _model
    with _lock:
        if _model is not None:
            return _model
        if _model_load_failed:
            raise EmbeddingUnavailable("embedding model previously failed to load")
        try:
            from fastembed import TextEmbedding
            _model = TextEmbedding(model_name=MODEL_NAME)
        except Exception as exc:  # network, disk, corrupt cache, missing dep, ...
            _model_load_failed = True
            raise EmbeddingUnavailable(str(exc)) from exc
        return _model


def embed(texts: Iterable[str]) -> list[np.ndarray]:
    """Embed each of `texts`, in order. Raises EmbeddingUnavailable rather
    than returning a partial or zeroed-out result if the model can't be
    loaded or run -- callers (agent/memory/queue.py, retrieval.py) decide
    for themselves what "no embedding available" should fall back to."""
    texts = list(texts)
    if not texts:
        return []
    model = _get_model()
    try:
        return [np.asarray(vec, dtype=np.float32) for vec in model.embed(texts)]
    except EmbeddingUnavailable:
        raise
    except Exception as exc:
        raise EmbeddingUnavailable(str(exc)) from exc


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
