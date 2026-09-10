"""recall() -- what a future `recall_memory` tool (agent/pipeline/tools.py,
not yet wired up -- see the project's design doc) calls when the agent
decides it needs more than a bullet's own summary of something already
compacted away (agent/memory/queue.py). Embedding-based semantic search
over the CURRENT generation's bullets (agent/memory/store.py) -- the
2026-09-10 design call, verbatim: "using an embedding based semantic
search to retrieve only relevant chunks."
"""
from __future__ import annotations

from agent.memory.embeddings import EmbeddingUnavailable, cosine_similarity, embed
from agent.memory.store import MemoryStore

#: How many bullets' worth of underlying chunks to pull back per recall
#: call -- small on purpose: this becomes a fresh slice of an already-
#: budgeted prompt (agent/memory/queue.py's X_BUDGET/Y_BUDGET), not a
#: second unbounded dump.
DEFAULT_TOP_K = 3


def recall(store: MemoryStore, kind: str, query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """The relevant compacted-away material for `query`, as readable text
    (each matched bullet followed by the real chunk text it cites) -- or a
    plain "nothing to recall yet" message if this `kind` has never
    compacted anything.

    Falls back to the top_k MOST RECENT current bullets, unranked, if the
    local embedding model can't be reached (agent/memory/embeddings.py's
    EmbeddingUnavailable, or a bullet written before embeddings were
    available -- store.py's Bullet.embedding can be None) -- showing
    something imperfectly ranked beats showing nothing at all.
    """
    bullets = store.current_bullets(kind)
    if not bullets:
        return "(nothing has been compacted away yet -- there is nothing to recall)"

    embeddable = [b for b in bullets if b.embedding is not None]
    picked = bullets[-top_k:]  # default/fallback: most recent, unranked
    if embeddable:
        try:
            [query_vec] = embed([query])
        except EmbeddingUnavailable:
            pass
        else:
            scored = sorted(
                embeddable,
                key=lambda b: cosine_similarity(query_vec, b.embedding),
                reverse=True,
            )
            picked = scored[:top_k]

    all_hashes = sorted({h for b in picked for h in b.hash_refs})
    chunks = store.get_chunks(all_hashes)

    parts = []
    for bullet in picked:
        texts = [chunks[h] for h in bullet.hash_refs if h in chunks]
        block = f"- {bullet.text}"
        if texts:
            block += "\n" + "\n".join(f"  > {t}" for t in texts)
        parts.append(block)
    return "\n\n".join(parts)
