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

That local model is still the default, and the 2026-09-11 measurement says it
should probably stay one: `otto eval-memory` scores retrieval at 96% recall
coverage on it, and the wins recorded in agent/memory/retrieval.py came from
retrieval STRUCTURE rather than embedding quality -- neighbour expansion was
worth 14 points, not narrowing on bullets first was worth 85, while the only
clean embedding-quality delta on record is this file's own query prefix at 3.9.
A hosted embedder is now selectable so that claim can be tested rather than
assumed; OTTO_EMBEDDING_MODEL names one as "provider:model".

Deliberately NOT routed through agent/router. agent/memory/__init__.py and
agent/eval/memory_bench.py both promise this package imports nothing from
agent.pipeline or agent.router, and the benchmark's offline mode depends on it.
There is no fallback chain to express here and no chat model to build, so the
routing machinery would buy nothing and cost a documented property.

Two contracts every backend must honour, because the callers depend on them:
embeddings come back as float32 numpy arrays (store.py's `_to_blob` calls
`.astype`), and every failure is an EmbeddingUnavailable (queue.py,
retrieval.py and rag.py each catch exactly that, and nothing else, to degrade
instead of crashing).
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


class EmbeddingBackend:
    """One way of turning text into vectors.

    `name` is stamped into every vector this backend produces
    (agent/memory/store.py's `embedding_model`) so a store can refuse to
    compare vectors from two different embedding spaces. It must change
    whenever the numbers would change.
    """

    name: str = ""

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        raise NotImplementedError


class LocalBGEBackend(EmbeddingBackend):
    """fastembed's BAAI/bge-small-en-v1.5, on-device. The default."""

    name = MODEL_NAME

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        model = _get_model()
        try:
            return [np.asarray(v, dtype=np.float32) for v in model.embed(texts)]
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(str(exc)) from exc

    def embed_query(self, query: str) -> np.ndarray:
        # BGE is trained asymmetrically -- see BGE_QUERY_INSTRUCTION below.
        # This is why the prefix belongs to a backend and not to this module:
        # OpenAI's embeddings are symmetric and Gemini signals the difference
        # with a task_type instead, so prepending it there embeds the
        # instruction as content and makes retrieval worse.
        [vector] = self.embed_documents([BGE_QUERY_INSTRUCTION + query])
        return vector


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI's hosted embeddings. Symmetric -- no query/passage distinction."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = f"openai:{model_id}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import openai

                self._client = openai.OpenAI()
            except Exception as exc:
                raise EmbeddingUnavailable(f"openai embeddings unavailable: {exc}") from exc
        return self._client

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        try:
            reply = self._get_client().embeddings.create(model=self.model_id, input=texts)
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            # 429, 401, timeout, per-batch token limits -- none of them are
            # EmbeddingUnavailable on their own, and every caller catches only
            # that, so an untranslated one crashes a compaction flush.
            raise EmbeddingUnavailable(f"openai embeddings failed: {exc}") from exc
        return [np.asarray(d.embedding, dtype=np.float32) for d in reply.data]

    def embed_query(self, query: str) -> np.ndarray:
        [vector] = self.embed_documents([query])
        return vector


class GeminiEmbeddingBackend(EmbeddingBackend):
    """Gemini's hosted embeddings. Signals query-vs-passage with `task_type`
    rather than with a prefix."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.name = f"gemini:{model_id}"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import os

                from google import genai

                self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            except Exception as exc:
                raise EmbeddingUnavailable(f"gemini embeddings unavailable: {exc}") from exc
        return self._client

    def _embed(self, texts: list[str], task_type: str) -> list[np.ndarray]:
        try:
            from google.genai import types

            reply = self._get_client().models.embed_content(
                model=self.model_id,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
        except EmbeddingUnavailable:
            raise
        except Exception as exc:
            raise EmbeddingUnavailable(f"gemini embeddings failed: {exc}") from exc
        return [np.asarray(e.values, dtype=np.float32) for e in reply.embeddings]

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> np.ndarray:
        [vector] = self._embed([query], "RETRIEVAL_QUERY")
        return vector


_BACKENDS = {"openai": OpenAIEmbeddingBackend, "gemini": GeminiEmbeddingBackend}
_backend: EmbeddingBackend | None = None


def current_backend() -> EmbeddingBackend:
    """The backend this process embeds with.

    Chosen once from OTTO_EMBEDDING_MODEL ("provider:model", e.g.
    "openai:text-embedding-3-small"); anything unset or unrecognised means the
    local model, which is the measured default.
    """
    global _backend
    if _backend is None:
        import os

        spec = os.environ.get("OTTO_EMBEDDING_MODEL", "").strip()
        provider, _, model_id = spec.partition(":")
        factory = _BACKENDS.get(provider)
        _backend = factory(model_id) if factory and model_id else LocalBGEBackend()
    return _backend


def current_model_name() -> str:
    """What to stamp on a vector produced right now -- see EmbeddingBackend."""
    return current_backend().name


def reset_backend() -> None:
    """Forget the selected backend, so a changed environment takes effect.
    For tests and for `otto` commands that switch models mid-process."""
    global _backend
    _backend = None


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
    return current_backend().embed_documents(texts)


#: BAAI/bge-* models are trained asymmetrically: a stored passage is embedded
#: as-is, but a QUERY is meant to arrive behind this exact instruction (the
#: model card's own wording). We were embedding questions as plain passages,
#: which costs real accuracy for nothing -- measured on the LoCoMo replay
#: (agent/eval/memory_bench.py), adding it moved top-5 chunk retrieval from
#: 61.4% to 65.3% with no change in what gets returned. Documents must NOT
#: get the prefix; that is the whole point of the asymmetry, and why this is
#: a separate function rather than a flag on embed().
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def embed_query(query: str) -> np.ndarray:
    """Embed `query` as a SEARCH QUERY rather than as stored text -- what
    agent/memory/retrieval.py ranks with. Raises EmbeddingUnavailable on the
    same terms as embed(), which callers already handle."""
    return current_backend().embed_query(query)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)
