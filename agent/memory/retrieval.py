"""recall() -- what agent/pipeline/tools.py's `recall_memory` tool calls when
the agent decides it needs more than a bullet's own summary of something
already compacted away (agent/memory/queue.py). The 2026-09-10 design call,
verbatim: "using an embedding based semantic search to retrieve only relevant
chunks", and the spec's own description of the finder's job: "first pulling
all the hash and compiling the list then finding relevant information (not
all) and sending that as new X for agent to work on with."

That last clause is the shape of this file. Retrieval runs in two stages,
because the two things being searched are different in kind:

    1. COMPILE -- take the hash_refs of EVERY live bullet, which is the
       whole of what has been compacted away for this `kind` (agent/memory/
       queue.py guarantees the live generation's citations cover everything
       it ever flushed). This is the spec's "pulling all the hash and
       compiling the list", and it is deliberately not a filter.
    2. SELECT -- rank those RAW chunks by each chunk's own stored embedding
       and return only the best handful. The spec's "finding relevant
       information (not all)".

Stage 2 is not an optimisation, it is the whole point, and it used to be
missing: recall() expanded every matched bullet into the full text of every
hash it cited, uncapped. Because compaction accumulates citations forward
(queue.py), a late-generation bullet cites nearly everything, so a single
match returned nearly the entire conversation -- measured on the LoCoMo
replay (agent/eval/memory_bench.py): 38,033 characters for every one of 101
questions, with only two distinct bullets ever matching. That scores well on
a benchmark, for the same reason it would drown a real prompt: the answer is
in there because everything is in there. `max_chunks` is what makes "not
all" true.

Ranking the bullets to pick a candidate set FIRST was tried, and it is a
trap worth naming here so it doesn't get reintroduced as an efficiency win.
Compaction does not produce comparable bullets: on a real live replay the
store held seven, six of them citing one to three chunks each and one citing
237. Which bullet a question matched had almost nothing to do with where its
answer was, so narrowing to the best three bullets scored 5% where searching
every bullet's hashes scored 90%. Bullets summarize; they do not index.
`top_k` therefore only controls how many bullet summaries are printed as
context above the results -- it no longer gates what can be found.

Then each selected chunk is returned together with the chunks either side of
it (`neighbour_window`, agent/memory/store.py's `chunks_near`). A dialogue
turn is a poor retrieval unit on its own -- the answer to "when did she join
the group?" is routinely in the turn AFTER the one that names the group --
and on the same replay this was the single largest win available: top-5
retrieval went from 65.3% to 79.2% while still returning roughly 6% of what
the old uncapped dump did.

Every embedding step degrades rather than fails. If the local model can't be
reached (agent/memory/embeddings.py's EmbeddingUnavailable) or nothing has an
embedding stored, each stage falls back to "most recent", unranked -- showing
something imperfectly ordered beats showing nothing at all.
"""
from __future__ import annotations

import logging

import numpy as np

from agent.memory.embeddings import (
    EmbeddingUnavailable,
    cosine_similarity,
    current_model_name,
    embed_query,
)
from agent.memory.store import Bullet, Chunk, MemoryStore
from agent.memory.tokens import count_tokens

#: How many bullet summaries to print above the results, as context for what
#: stretch of the conversation they came from. Ranked, but see the module
#: docstring: this does NOT limit what stage 2 can find.
DEFAULT_TOP_K = 3

#: How many bullets a PROCEDURAL recall reads -- "what did I already try here",
#: asked mid-task while acting.
#:
#: One, not three, and that is measured rather than tidy. Across memory
#: substrates, retrieval depth pulls in opposite directions for the two jobs
#: this store does: accuracy on user-history question answering rises
#: monotonically with k, while task-time recall FALLS about 7 points going from
#: k=1 to k=5, because retrieved context starves attention from the thing the
#: agent is meant to be acting on. A separate ablation independently peaks at
#: k=1 and degrades from k>=2 (2608.15008, 2509.25140).
#:
#: Same store, two read policies. The old single `top_k` served the QA job and
#: quietly taxed the acting one.
PROCEDURAL_TOP_K = 1

#: How many RAW CHUNKS stage 2 actually returns, before neighbour expansion.
#: This is the cap that makes recall() a slice of a budgeted prompt (agent/
#: memory/queue.py's X_BUDGET/Y_BUDGET) rather than a second unbounded dump.
DEFAULT_MAX_CHUNKS = 20

#: How many chunks either side of each selected chunk to include (see the
#: module docstring). 2 means "the two turns before and the two after".
#:
#: Measured with `otto eval-memory --max-chunks/--neighbours` against a real
#: LIVE replay's store (250 turns, 249 chunks, a real summarizer's bullets),
#: scoring how often recall() surfaces a question's cited evidence and how
#: much text it returns to do it:
#:
#:     max_chunks  neighbours   coverage   chars returned
#:              5           0      65.3%              922
#:              5           1      79.2%            2,112
#:              5           2      84.2%            3,189
#:             10           2      90.1%            6,293
#:             15           2      92.1%            8,992
#:             20           2      96.0%           11,526
#:             40           2      98.0%           19,660
#:      (uncapped, before this)     99.0%           38,033
#:
#: Two things that curve says, and the reason these are separate knobs at
#: all: widening the window buys coverage more cheaply than raising the cap
#: (5/2 beats 10/1 for less text), and the last two points cost more than the
#: first eighty. 20/2 is ~2.9K tokens per call, under 3% of TOTAL_BUDGET --
#: affordable to spend on a tool result the agent explicitly asked for, which
#: the old uncapped 38,033 characters was not.
DEFAULT_NEIGHBOUR_WINDOW = 2

#: The hard ceiling on how much text one recall() returns, and the one that
#: actually protects a real session. Both knobs above are COUNTS of items,
#: which is only a proxy for size: LoCoMo's dialogue turns average ~150
#: characters, while an Otto turn is a person's whole message or a full
#: assistant reply and can be thousands, so the same 20 items that measured
#: at ~2.9K tokens above could be ten times that here. Chunks are taken in
#: rank order until the next would not fit, so a session of long turns
#: returns fewer, longer items instead of blowing the budget.
#:
#: ~3% of agent/memory/queue.py's TOTAL_BUDGET. recall() is meant to become a
#: fresh slice of X ("sending that as new X for agent to work on with" -- the
#: spec), so it has to leave room for X to still hold the conversation.
DEFAULT_TOKEN_BUDGET = 3_000

logger = logging.getLogger(__name__)

_NOTHING_YET = "(nothing has been compacted away yet -- there is nothing to recall)"


def _comparable(items: list, model: str) -> list:
    """Only the vectors produced by the model we are querying with.

    Comparing across embedding spaces is the failure this guards, and it is
    quieter than it sounds. Mismatched dimensions raise a ValueError from
    numpy that agent/pipeline/tools.py catches and turns into one line of
    stderr, so recall simply stops working for the rest of the session. Worse,
    a hosted model emitting the SAME dimension -- Gemini can be asked for 384,
    OpenAI's can be truncated -- raises nothing at all: the shapes line up, the
    scores look plausible, and two unrelated spaces get ranked against each
    other. Checking dimensions would not catch that; checking the model does.

    A vector with no stamp was written before the column existed. Its space is
    unknown, so it is not comparable either.
    """
    comparable = [i for i in items if i.embedding is not None and i.embedding_model == model]
    foreign = sum(
        1 for i in items
        if i.embedding is not None and i.embedding_model != model
    )
    if foreign and not comparable:
        # Everything stored was embedded by a different model, so ranking is
        # impossible and recall silently degrades to unranked most-recent.
        # That is the safe behaviour but an invisible one -- said out loud
        # here, because the fix is to re-embed the store and nothing else will
        # ever mention it.
        logger.warning(
            "%d stored vector(s) were embedded by a different model than %r -- "
            "they cannot be ranked against this query. Re-embed the store to "
            "use them again.",
            foreign, model,
        )
    return comparable


def _rank_bullets(bullets: list[Bullet], query_vec, top_k: int, model: str = "") -> list[Bullet]:
    embeddable = _comparable(bullets, model)
    if query_vec is None or not embeddable:
        return bullets[-top_k:]  # fallback: most recent, unranked
    return sorted(
        embeddable, key=lambda b: cosine_similarity(query_vec, b.embedding), reverse=True,
    )[:top_k]


def _rank_chunks(chunks: list[Chunk], query_vec, max_chunks: int, model: str = "") -> list[Chunk]:
    """Every compacted chunk is a candidate here, so this is the one step that
    scales with session length -- hence the single matrix product rather than
    a Python-level loop over cosine_similarity()."""
    embeddable = _comparable(chunks, model)
    if query_vec is None or not embeddable:
        return chunks[-max_chunks:]  # fallback: most recent, unranked
    matrix = np.stack([c.embedding for c in embeddable])
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vec) or 1.0)
    scores = (matrix @ query_vec) / np.where(norms == 0, 1.0, norms)
    return [embeddable[i] for i in np.argsort(-scores)[:max_chunks]]


def _select(
    store: MemoryStore, kind: str, ranked: list[Chunk], window: int, token_budget: int,
) -> list[Chunk]:
    """`ranked` best-first, each taken together with the chunks either side of
    it, until the next one would not fit in `token_budget` -- then returned in
    conversation order, oldest first, the only order the text reads correctly
    in. Spending the budget best-first is what makes it bind on the LEAST
    relevant material rather than on whatever happened to rank last.
    """
    seqs = [c.seq for c in ranked if c.seq is not None]
    nearby: dict[int, Chunk] = {}
    if seqs and window > 0:
        nearby = {c.seq: c for c in store.chunks_near(kind, seqs, window) if c.seq is not None}

    chosen: dict[str, Chunk] = {}
    spent = 0
    for chunk in ranked:
        # A chunk written before `seq` existed (agent/memory/store.py's
        # migration) can't be found by position -- it comes back on its own.
        group = [chunk] if chunk.seq is None else [
            nearby[s] for s in range(chunk.seq - window, chunk.seq + window + 1) if s in nearby
        ] or [chunk]
        addition = [c for c in group if c.hash not in chosen]
        cost = sum(count_tokens(c.content) for c in addition)
        if chosen and spent + cost > token_budget:
            break
        chosen.update({c.hash: c for c in addition})
        spent += cost
    return sorted(chosen.values(), key=lambda c: (c.seq is None, c.seq))


def recall(
    store: MemoryStore,
    kind: str,
    query: str,
    *,
    purpose: str = "recall",
    top_k: int | None = None,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    neighbour_window: int = DEFAULT_NEIGHBOUR_WINDOW,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """The relevant compacted-away material for `query`, as readable text: the
    bullets that best match, as context, then the handful of real raw items
    the search actually selected (each with its immediate neighbours). A plain
    "nothing to recall yet" message if this `kind` has never compacted
    anything.
    """
    # "acting" is a narrower read than "recall" on purpose -- see
    # PROCEDURAL_TOP_K. The default stays the broad one, so a caller that does
    # not say what it is for gets what it got before.
    if top_k is None:
        top_k = PROCEDURAL_TOP_K if purpose == "acting" else DEFAULT_TOP_K

    bullets = store.current_bullets(kind)
    if not bullets:
        return _NOTHING_YET

    try:
        query_vec = embed_query(query)
        query_model = current_model_name()
    except EmbeddingUnavailable:
        query_vec, query_model = None, ""

    # Every live bullet's hashes, not just the matched ones -- the module
    # docstring's stage 1, and why narrowing here was a trap.
    candidate_hashes = sorted({h for b in bullets for h in b.hash_refs})
    candidates = store.get_chunk_rows(kind, candidate_hashes)
    picked_bullets = _rank_bullets(bullets, query_vec, top_k, query_model)
    shown = _select(
        store, kind, _rank_chunks(candidates, query_vec, max_chunks, query_model),
        neighbour_window, token_budget,
    )

    parts = [f"- {b.text}" for b in picked_bullets]
    parts += [f"  > {c.content}" for c in shown]
    return "\n".join(parts)
