"""Retrieval over the files in the bound workspace, on the engine Otto already
has.

Nothing here is a new retrieval stack. `agent/memory/` already contains a
content-addressed chunk store with embeddings (`store.py`), a local embedding
model (`embeddings.py`), and a two-stage ranked search with neighbour
expansion and a token budget (`retrieval.py`'s `recall()`), all of which were
built and measured for conversation history. A corpus is the same problem with
a different source, so this indexes files into that store and asks `recall()`
the question.

The one seam worth explaining is the covering bullet. `recall()` searches
chunks reachable from the CURRENT generation's bullets -- that indirection is
what makes compaction lossless for conversation memory. An indexed corpus has
no compaction and therefore no bullets, so it would be invisible. Writing a
single bullet whose `hash_refs` name every indexed chunk makes the whole
corpus reachable and hands the real work to stage two, which is the part that
does ranked retrieval. It is a two-line adapter rather than a parallel engine.

Distinct from `recall_memory`, and the tool descriptions have to keep saying
so: that searches what this CONVERSATION said and later compacted away, this
searches what is written in the FILES. Two tools that sound alike cost
tool-loop turns, and turns are the real per-message cost.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from agent.memory.embeddings import EmbeddingUnavailable, embed
from agent.memory.hashing import content_hash
from agent.memory.store import MemoryStore
from agent.memory.tokens import count_tokens

#: Roughly one screen of text. Small enough that a hit is specific, large
#: enough to carry its own context -- and `recall()`'s neighbour expansion
#: pulls the surrounding chunks back anyway, so erring small is cheap here.
CHUNK_TOKENS = 200

#: Suffixes worth indexing. An allowlist rather than a blocklist: a corpus
#: walk that tries to embed a PNG or a compiled binary wastes the embedding
#: budget and pollutes every later search with noise.
TEXT_SUFFIXES = frozenset({
    ".py", ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".sh", ".bash", ".zsh", ".js", ".jsx", ".ts", ".tsx", ".html",
    ".css", ".sql", ".c", ".h", ".cpp", ".hpp", ".rs", ".go", ".java", ".rb",
    ".csv", ".tsv", ".xml", ".env", ".gitignore", ".dockerfile", ".conf",
})

#: Same noise directories `list_files` skips -- version control, build trees,
#: dependency caches. Never source, so skipping them loses nothing.
SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "node_modules",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", "dist", "build",
    ".eggs", ".idea", ".vscode", "target",
})

MAX_FILE_BYTES = 1024 * 1024


def split_into_chunks(text: str, chunk_tokens: int = CHUNK_TOKENS) -> list[str]:
    """`text` split on line boundaries into roughly `chunk_tokens`-sized pieces.

    Line boundaries rather than a fixed character count: a chunk that stops
    mid-statement embeds poorly and reads worse when it comes back.
    """
    chunks: list[str] = []
    current: list[str] = []
    budget = 0
    for line in text.splitlines():
        cost = count_tokens(line) or 1
        if current and budget + cost > chunk_tokens:
            chunks.append("\n".join(current))
            current, budget = [], 0
        current.append(line)
        budget += cost
    if current:
        chunks.append("\n".join(current))
    return [c for c in chunks if c.strip()]


def corpus_fingerprint(root: Path) -> str:
    """A cheap signature of the corpus, so an unchanged workspace is not
    re-embedded on every query. Names, sizes and mtimes only -- reading every
    file to hash it would cost as much as indexing."""
    parts = []
    for path in sorted(indexable_files(root)):
        stat = path.stat()
        parts.append(f"{path.relative_to(root)}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def indexable_files(root: Path) -> list[Path]:
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        found.append(path)
    return found


def index_corpus(store: MemoryStore, kind: str, root: Path) -> int:
    """Index every text file under `root` into `store`, and make it all
    reachable from one covering bullet. Returns the number of chunks written.
    """
    hashes: list[str] = []
    texts: list[str] = []
    for path in indexable_files(root):
        try:
            body = path.read_text(errors="replace")
        except OSError:
            continue
        label = str(path.relative_to(root))
        for chunk in split_into_chunks(body):
            texts.append(f"{label}\n{chunk}")

    for batch_start in range(0, len(texts), 64):
        batch = texts[batch_start:batch_start + 64]
        try:
            vectors = list(embed(batch))
        except EmbeddingUnavailable:
            vectors = [None] * len(batch)
        for text, vector in zip(batch, vectors):
            digest = content_hash(text)
            store.add_chunk(kind, digest, text, vector)
            hashes.append(digest)

    if hashes:
        # The covering bullet -- see the module docstring. Its text is what
        # stage one ranks, and there is only one of it, so the wording is
        # irrelevant; stage two does the real retrieval.
        store.add_bullet(kind, 1, f"the {len(hashes)} indexed chunks of {root.name}", hashes, None)
    return len(hashes)
