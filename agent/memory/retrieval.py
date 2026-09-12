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
import re

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

#: How far below the best match a chunk may score and still be worth
#: returning, as a fraction of that best score.
#:
#: This turns `max_chunks` from a quota into a ceiling: a query whose evidence
#: is genuinely spread across many chunks still gets them, and one whose answer
#: is in the first two stops there instead of padding the prompt to twenty.
#:
#: 0.85 rather than something tighter because these are cosine similarities on
#: a self-similar stream, where the spread between a decisive chunk and a
#: near-duplicate is small by construction -- a tight ratio would cut the
#: second hop of a multi-hop question, which is the case this is meant to help.
FALLOFF_RATIO = 0.85

#: Never return fewer than this, however sharp the falloff. A threshold that
#: could return one chunk would make an oddly-worded query worse than the fixed
#: cap it replaced.
MIN_CHUNKS = 3

#: How many chunks the second hop may add. Small: it is reaching for evidence
#: the question does not mention, which is exactly the material that is
#: valuable when it is right and noise when it is wrong. Three is enough for
#: the two-hop questions the benchmark actually contains.
MAX_SECOND_HOP = 3

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

NOTHING_COMPACTED = "(nothing has been compacted away yet -- there is nothing to recall)"

#: The two kinds a session stores under, named here because both the memory
#: layer and agent/pipeline/ have to agree on them and this is the lowest
#: module they share -- agent/pipeline/tools.py cannot import nodes.py, which
#: is what writes the second one.
#:
#: "history" is the CONVERSATION, compacted by a summariser into cited
#: bullets (agent/memory/queue.py). "context" is TOOL OUTPUT evicted from one
#: run's transcript, stored as chunks with no bullet layer at all: it was
#: already reduced to a line in `actions` when the call ran, and summarising
#: it again would pay a model call to abstract an abstraction.
HISTORY_KIND = "history"
EVICTED_KIND = "context"


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
        # `bullets[-0:]` is the WHOLE list, not none of it -- Python has no
        # negative zero. No caller passes 0 today, which is exactly why this
        # is worth pinning: the day one does, the unranked fallback would
        # quietly return everything instead of nothing.
        return bullets[-top_k:] if top_k else []
    return sorted(
        embeddable, key=lambda b: cosine_similarity(query_vec, b.embedding), reverse=True,
    )[:top_k]


#: A token worth matching literally: long enough not to be a common word, and
#: carrying a digit or a separator, which is what ids, dates, versions, paths
#: and hostnames have and ordinary prose does not. `AK-4417-QX`, `2027-01-31`,
#: `v2.14.0`, `agent/memory/retrieval.py`.
#:
#: Deliberately narrow in the same way agent/pipeline/tools.py's
#: `looks_like_a_literal` is, and for the same measured reason -- but not the
#: same pattern: that one requires the whole query to BE the token, because
#: grep either runs or does not. Here the literal pass is additive, so a token
#: anywhere inside a longer question can be used.
_RARE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{3,}")


def _literal_tokens(query: str) -> list[str]:
    return [t for t in _RARE_TOKEN.findall(query)
            if any(ch.isdigit() for ch in t) or any(ch in t for ch in "_/:.")]


def _rank_chunks(chunks: list[Chunk], query_vec, max_chunks: int, model: str = "",
                 *, second_hop: bool = False, query: str = "") -> list[Chunk]:
    """Every compacted chunk is a candidate here, so this is the one step that
    scales with session length -- hence the single matrix product rather than
    a Python-level loop over cosine_similarity().

    An exact-token pass runs first when the query carries one. This is the
    same finding agent/pipeline/tools.py's `rag` already acts on -- measured
    on Otto's own tree, grep found 4/4 exact identifiers and 0/4 conceptual
    questions, and the embedding index found 4/4 of both but took forty
    seconds to do the first. Memory recall never got the other half of that
    decision, and it shows on exactly the queries you would expect: a date.

    Additive, never a replacement. Literal hits go first and the semantic
    ranking fills whatever room is left, so a query with no rare token in it
    behaves exactly as before, and one that has a token can only gain.
    """
    embeddable = _comparable(chunks, model)
    if query_vec is None or not embeddable:
        # Same `[-0:]` trap as _rank_bullets above.
        return chunks[-max_chunks:] if max_chunks else []
    matrix = np.stack([c.embedding for c in embeddable])
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(query_vec) or 1.0)
    scores = (matrix @ query_vec) / np.where(norms == 0, 1.0, norms)
    order = np.argsort(-scores)[:max_chunks]
    keep = _before_the_falloff(scores[order])
    ranked = [embeddable[i] for i in order[:keep]]

    tokens = _literal_tokens(query)
    if not tokens:
        return ranked
    # Over every candidate, not just the embeddable ones. A literal match
    # needs no vector, so a chunk that was never embedded -- or was embedded
    # by a model this query cannot be ranked against -- is still findable by
    # the one thing that does not care: its text.
    #
    # Newest first among the exact hits: a literal appearing several times is
    # usually being updated, and the last word on it is the one that matters.
    hits = [c for c in reversed(chunks)
            if any(t.lower() in c.content.lower() for t in tokens)]
    if not hits:
        return ranked
    seen = {id(c) for c in hits}
    room = max(max_chunks - len(hits), 0) if max_chunks else 0
    return hits[:max_chunks or len(hits)] + [
        c for c in ranked if id(c) not in seen
    ][:room]


def _second_hop(candidates: list[Chunk], primary: list[Chunk], room: int) -> list[Chunk]:
    """Rank again against what the first pass FOUND, not against the query.

    This is the multi-hop case, and it is why that category is the weakest one
    the memory benchmark measures -- 82% against 91% overall. A question whose
    answer needs two pieces of evidence only matches the FIRST of them: the
    second matches the answer to the first hop, which does not appear in the
    question at all. Ranking against the query alone cannot reach it however
    many chunks it returns.

    Costs nothing. Every chunk already carries a stored embedding, so the
    second pass is one more matrix product against a vector already in memory
    -- no model call, no network, no query rewriting.

    Held to the same falloff bar as the first pass, and capped by whatever room
    the first pass left.

    NOTHING CALLS THIS. It was written for issue #22 and measured, and it did
    not work:

        multi-hop recall   82%  ->  82%     (no change, n=44)
        overall recall     91%  ->  91%
        text per query   2,318  ->  3,209   (+38%)

    So it costs more and finds nothing extra. The reasoning still looks right
    -- a multi-hop question's second piece of evidence really does match the
    first hop's ANSWER rather than the question -- which is why it stays here
    rather than being deleted. What it suggests is that the anchor is wrong:
    ranking against the top chunk's whole embedding finds chunks similar to
    that chunk, which in a self-similar stream is its neighbours, and those
    were already being returned by the neighbour window. A working version
    needs an anchor that represents what the question still LACKS, not what it
    already found.
    """
    seen = {c.hash for c in primary}
    pool = [c for c in candidates if c.hash not in seen and c.embedding is not None]
    if not pool or room <= 0:
        return []
    anchor = primary[0].embedding
    matrix = np.stack([c.embedding for c in pool])
    norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(anchor) or 1.0)
    scores = (matrix @ anchor) / np.where(norms == 0, 1.0, norms)
    order = np.argsort(-scores)[:room]
    keep = min(_before_the_falloff(scores[order]), MAX_SECOND_HOP)
    return [pool[i] for i in order[:keep]]


def _before_the_falloff(ranked_scores) -> int:
    """How many of the ranked chunks to keep, stopping where they stop helping.

    `max_chunks` is a fixed number and a fixed number is wrong in both
    directions: too shallow for a multi-hop question whose second piece of
    evidence ranks low, too noisy for a single-fact lookup where everything
    after the first match is filler. Agent memory is a bounded, highly
    self-similar stream, so the hard part is telling decisive evidence from
    near-duplicates rather than finding relevant text at all.

    The cheap version of "expand while the next piece reduces uncertainty",
    needing no extra model call: keep taking chunks while they score close to
    the best one, and stop at the first that does not. A run of near-equal
    scores is a genuine multi-hop spread and is kept; a cliff after the first
    is the single-fact case and the tail is dropped.

    Never returns fewer than MIN_CHUNKS. A threshold that can return one chunk
    would make a slightly-odd query worse than the fixed cap it replaced, and
    the floor costs almost nothing when the tail is cheap.
    """
    if len(ranked_scores) <= MIN_CHUNKS:
        return len(ranked_scores)
    best = float(ranked_scores[0])
    if best <= 0:
        return len(ranked_scores)  # nothing to measure a fall against
    floor = best * FALLOFF_RATIO
    for position, score in enumerate(ranked_scores):
        if float(score) < floor:
            return max(MIN_CHUNKS, position)
    return len(ranked_scores)


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


def recall_chunks(
    store: MemoryStore,
    kind: str,
    query: str,
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    neighbour_window: int = DEFAULT_NEIGHBOUR_WINDOW,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """Like recall(), for a store with no bullet layer at all.

    `recall()` finds its candidates by compiling the hash_refs of every live
    bullet, which is right for conversation history: something summarised it,
    and those summaries are both the index and useful context to print above
    the results. Some material has no summariser and needs none -- evicted tool
    output, for instance, which agent/pipeline/nodes.py has already reduced to
    a one-line record in `actions`. Summarising it again would be paying a
    model call to abstract something that was already abstracted once.

    So this ranks the chunks directly. Same ranking, same neighbour expansion,
    same token budget; no bullets in and none printed. Empty string when the
    kind holds nothing, rather than recall()'s "nothing compacted yet" -- the
    caller is combining this with other sources and an absence should read as
    an absence, not as a sentence.
    """
    hashes = store.chunk_hashes(kind)
    if not hashes:
        return ""

    try:
        query_vec = embed_query(query)
        query_model = current_model_name()
    except EmbeddingUnavailable:
        query_vec, query_model = None, ""

    candidates = store.get_chunk_rows(kind, hashes)
    shown = _select(
        store, kind,
        _rank_chunks(candidates, query_vec, max_chunks, query_model, query=query),
        neighbour_window, token_budget,
    )
    return "\n".join(f"  > {c.content}" for c in shown)


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
        return NOTHING_COMPACTED

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
        store, kind,
        _rank_chunks(candidates, query_vec, max_chunks, query_model, query=query),
        neighbour_window, token_budget,
    )

    parts = [f"- {b.text}" for b in picked_bullets]
    parts += [f"  > {c.content}" for c in shown]
    return "\n".join(parts)
