"""The X/Y tiered short-term-memory queue -- 2026-09-10 design call
(verbatim spec): "Otto short term memory is like a queue with X+Y space.
Once X gets filled memory moves to Y. When Y gets filled all memory in Y
is flushed and stored with a hash in a DB. Then we take all the memory in
Y and make a summarized bullet list and place it in the last slot of Y
with each bullet having hash of reference to DB. Once it gets filled again
we take all and do the same but we dont map any hash from before in this
list unless it's a bullet point. A bullet point can have multiple hash
associated with it."

Scope, confirmed with the person: UNIFIED -- one engine, two independent
uses. Conversation history growing without bound across turns (agent/
pipeline/run.py's `history` parameter, sixth refinement -- flagged there as
a known, deliberately out-of-scope gap) and in-turn context/board growth
across unbounded overseer rounds (agent/pipeline/nodes.py's module
docstring: "no cap on how many times the overseer may retry a task") both
get the SAME TieredQueue mechanism, as two separate instances (`kind`
namespaces them in the store -- agent/memory/store.py) rather than one
merged pool: mixing conversation turns and task-internal context into one
summary would make either harder to read cleanly. Graph/CLI-side wiring of
these two instances is tracked separately from this file -- see the
project's claude/otto-tiered-memory-design.md for what's built vs. what's
still pending. As of Phase 2 (agent/memory/wiring.py, agent/memory/
session.py): the `kind="history"` instance IS wired in -- agent/cli/
shell.py's Session owns one per session, seeded via `run_pipeline_stream`'s
new `memory_context` parameter. The `kind="context"` instance (in-turn
context/board growth, agent/pipeline/nodes.py) is still NOT wired -- it
needs a place to persist a live TieredQueue across multiple node calls
within one LangGraph run, which `history`'s per-Session object didn't have
to solve (a Session already lives for the whole CLI process); still open.

Two tiers, then permanent storage:

    X -- a small, purely in-memory holding buffer for the most recent raw
         items, verbatim, no DB round-trip. What a prompt reads as "the
         detailed, recent part."
    Y -- a larger in-memory buffer. Once X's own token budget is exceeded,
         X's ENTIRE current content moves to Y in one step (not a trickle)
         and X starts over empty.
    DB -- once Y's OWN token budget is exceeded (raw items plus whatever
          bullet list is already sitting in Y from an earlier round),
          everything currently in Y is compacted in one step:

            1. every RAW item in Y is flushed to the store as a `chunk`,
               keyed by its content hash (agent/memory/hashing.py) --
               permanent, content-addressed, never rewritten.
            2. the WHOLE of Y (its raw items AND any prior bullets, each
               shown to the summarizer as one numbered "item") is handed
               to the injected `summarize` callback, asked to return one
               bullet per line, each ending in a `[sources: N,N,...]` tag
               naming which item number(s) it draws from -- deliberately
               NOT hardwired to a specific model/prompt/provider here (see
               __init__ below), so this file needs no live LLM to test.
            3. each new bullet's `hash_refs` is built from exactly the
               items its citation names: a cited RAW item contributes its
               own single hash; a cited PRIOR BULLET contributes its own
               `hash_refs` UNCHANGED -- its text is never itself hashed as
               a new leaf chunk ("we dont map any hash from before in this
               list unless it's a bullet point" -- the spec, verbatim).
               This is what keeps every hash_refs entry resolving to real,
               permanently-stored raw content no matter how many
               compaction generations have happened, and why one bullet
               can end up covering several hashes ("a bullet point can
               have multiple hash associated with it") -- it accumulates
               everything transitively cited into it, generation over
               generation.
            3b. every item the reply did NOT cite gets its own extra
               bullet anyway (_uncited_bullets, below), so the new
               generation's hash_refs collectively cover everything this
               compaction flushed. The prompt asks for full coverage; this
               is what makes it true. Without it an omitted item stays in
               the chunk store but nothing live points at it, and recall()
               searches only the live generation -- and because a prior
               bullet is itself a numbered item carrying every hash behind
               it, one uncited bullet used to sever a whole chain at once.
            4. Y is replaced by just that new bullet list -- a handful of
               short, cited lines instead of everything they summarize --
               freeing almost all of Y's budget again. The store's OLDER
               generation of bullets for this `kind` is marked superseded
               (agent/memory/store.py), not deleted -- an audit trail, not
               live state.

`current_view()` is what a prompt actually reads: Y's current bullet list
(compact, older, cited) followed by X's raw items (verbatim, newest/most
detailed) -- NOT the DB, which is cold storage nothing reads directly.
agent/memory/retrieval.py's recall() is how an agent that needs more than
a bullet's own summary pulls the real cited text back on demand.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from agent.memory.embeddings import EmbeddingUnavailable, embed
from agent.memory.hashing import content_hash
from agent.memory.store import MemoryStore
from agent.memory.tokens import count_tokens

#: mercury-2.5 backs every conversational node in this graph (router/
#: planner/solver/summarizer/finder/evaluator -- agent/router/mapping.py's
#: TASK_ROUTES, checked directly: every one of them is pinned to
#: "inception:mercury-2.5"). Its context window is 260K tokens; the design
#: target, confirmed with the person, is staying at 40% of that (not the
#: smaller "12%" this file's own author first proposed) -- 104,000 tokens
#: total across BOTH tiers, leaving 60% of the window for the task prompt
#: itself, tool results, the plan, and everything else a node's own prompt
#: adds on top of "what memory shows it."
MERCURY_2_5_CONTEXT_WINDOW = 260_000
TARGET_FRACTION = 0.40
TOTAL_BUDGET = int(MERCURY_2_5_CONTEXT_WINDOW * TARGET_FRACTION)  # 104,000

#: Split roughly 1:3, X:Y -- X only needs to hold a handful of the most
#: recent items verbatim before handing off; Y needs to be the bulk of the
#: budget so compaction (an LLM call) doesn't fire on every single append.
#: Both are plain module constants specifically so they're easy to tune
#: from real usage later, per the person's own framing of the 104K figure
#: as "use this as a first pass."
X_BUDGET = 24_000
Y_BUDGET = TOTAL_BUDGET - X_BUDGET  # 80,000
assert X_BUDGET + Y_BUDGET == TOTAL_BUDGET


@dataclass
class NewBullet:
    text: str
    hash_refs: list[str]


_SOURCE_TAG = re.compile(r"\[sources:\s*([0-9,\s]+)\]\s*$", re.IGNORECASE)


def _build_summarize_prompt(items: list[str]) -> str:
    numbered = "\n".join(f"[{i + 1}] {item}" for i, item in enumerate(items))
    return (
        "Summarize the numbered items below into a short bullet list. "
        "Cover every item -- every item number from 1 to "
        f"{len(items)} must appear in at least one bullet's citation. "
        "Each bullet must end with exactly which item number(s) it draws "
        "from, in the exact form \"[sources: N,N,...]\" -- one bullet may "
        "cover several items. Reply with exactly one bullet per line, "
        "nothing else, no preamble, no markdown.\n\n" + numbered
    )


def _parse_bullets(text: str, item_count: int) -> list[tuple[str, list[int]]]:
    """(bullet text, 1-based item indices) pairs -- a line with no
    parseable `[sources: ...]` tag, or one that cites nothing actually in
    range, is DROPPED rather than guessed at (queue.py's own caller falls
    back to a single everything-bullet if this returns nothing at all --
    see _compact_y_if_full)."""
    parsed = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        match = _SOURCE_TAG.search(line)
        if not match:
            continue
        indices = sorted({int(n) for n in match.group(1).split(",") if n.strip().isdigit()})
        indices = [i for i in indices if 1 <= i <= item_count]
        if not indices:
            continue
        bullet_text = _SOURCE_TAG.sub("", line).strip()
        if bullet_text:
            parsed.append((bullet_text, indices))
    return parsed


#: Longest single-item excerpt kept as a carry-forward bullet's own text
#: (_uncited_bullets, below). Long enough that recall()'s embedding search
#: has real, distinguishing words to rank on -- a carried-forward turn about
#: someone's dog has to stay findable by the word "dog" -- but short enough
#: that a compaction whose summarizer cited almost nothing still frees most
#: of Y rather than re-filling it with what it just tried to compact.
CARRY_FORWARD_EXCERPT_CHARS = 160


def _excerpt(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= CARRY_FORWARD_EXCERPT_CHARS:
        return collapsed
    return collapsed[:CARRY_FORWARD_EXCERPT_CHARS].rstrip() + "..."


def _embed_each(texts: list[str]) -> list["np.ndarray | None"]:
    """One embedding per text, or a list of Nones if the local model can't be
    reached -- store.py treats a NULL embedding as "not rankable", not as an
    error (agent/memory/embeddings.py's EmbeddingUnavailable), so a compaction
    that happens while the model is unavailable still stores everything."""
    if not texts:
        return []
    try:
        return list(embed(texts))
    except EmbeddingUnavailable:
        return [None] * len(texts)


def _uncited_bullets(
    items: list[str], item_hashes: list[list[str]], covered: set[int], prior_bullet_count: int,
) -> list["NewBullet"]:
    """One extra bullet for every item the summarizer's reply never cited,
    so the new generation's `hash_refs` collectively cover EVERYTHING this
    compaction flushed.

    The prompt (_build_summarize_prompt) asks for full coverage, but nothing
    made the model deliver it, and an omission here is not proportional
    damage -- it is a cliff. Measured on LoCoMo at a stress budget (250
    turns, x=200/y=400, 19 generations): a summarizer citing ~70% of its
    items left only 3-4% of the 249 stored chunks reachable from the live
    bullet generation, and recall() found the right turn for 2-3% of
    questions instead of ~100%. The reason it is a cliff rather than a
    ~30% loss is that a PRIOR BULLET is itself one of the numbered items,
    carrying every hash it has accumulated over every generation before it;
    one compaction that summarizes it into prose without citing its number
    severs the whole accumulated chain at once. The raw text is still in the
    permanent chunk store either way (store_coverage stays 100%), but
    nothing in the live generation points at it any more, and recall()
    only ever searches the live generation (agent/memory/retrieval.py,
    agent/memory/store.py's `superseded`) -- so it becomes unreachable in
    practice by any path short of a direct hash lookup nothing performs.

    An uncited PRIOR BULLET is carried forward exactly as it was: its text
    is already a summary and its hash_refs already resolve, so re-deriving
    either would only lose fidelity. An uncited RAW item becomes a bullet
    holding its own (truncated) text, which keeps the distinguishing words
    recall() ranks on instead of hiding it inside a contentless catch-all.
    """
    carried = []
    for index in range(1, len(items) + 1):
        if index in covered:
            continue
        is_prior_bullet = index <= prior_bullet_count
        text = items[index - 1] if is_prior_bullet else _excerpt(items[index - 1])
        carried.append(NewBullet(text=text, hash_refs=list(item_hashes[index - 1])))
    return carried


#: Whether a compaction is allowed to summarise a PREVIOUS compaction's
#: bullets, or only the raw text that has newly overflowed.
#:
#: True is the original behaviour and every generation's bullets are then a
#: summary of the last generation's bullets, geometrically. The research says
#: not to: consolidating a model's own distillations and feeding them back
#: made one model fail 54% of problems it had previously solved, and
#: raw-episode retention doubled accuracy against forced consolidation.
#: Abstract once from raw; never re-abstract an abstraction.
#:
#: It is still the default, and that is now a measurement rather than
#: inertia. `otto eval-compaction` compares the two arms directly, and on
#: eight constraints planted through a conversation:
#:
#:                       120 turns   400 turns
#:     shipping             8/8         8/8
#:     abstract-once        8/8         8/8
#:     both, tight budget   8/8         8/8
#:
#: No difference at all. What was destroying constraints was not re-abstraction
#: but the FIRST compaction, and nothing a later generation does can recover
#: what generation one dropped. PROTECT_PREFIX below is what actually moved
#: the number; this knob moved nothing.
#:
#: The alternative also costs something real: with prior bullets carried
#: forward verbatim, the budget has to be freed some other way, and the only
#: way that does not re-abstract is to drop older bullets' TEXT while merging
#: their citations forward (`_shed_oldest`). Losing the oldest summaries
#: cleanly, rather than degrading all of them slowly. Kept, unused by default,
#: because it is the arm that made the comparison possible.
REABSTRACT = True

#: Items starting with this are carried into a compaction VERBATIM instead of
#: being summarised. Empty string disables it.
#:
#: Type-blind compaction is measured as destructive in a specific way: it is
#: the CONSTRAINTS that go. Constraint recall falls to 53% at 50% compression
#: and 24% at 10%, where a type-aware policy holds 100/95/80. And in a
#: conversation the type is visible from the speaker -- what the person said
#: is the requirement, what Otto said is a report of work, and a report can be
#: summarised without losing anything the next turn has to honour.
#:
#: agent/memory/wiring.py writes "you: " in front of every human turn, which
#: is what makes this a one-line rule rather than a classifier.
#:
#: Measured with `otto eval-compaction`, eight constraints planted through a
#: conversation, counting how many are still in `current_view()` afterwards:
#:
#:                        120 turns   400 turns
#:     type-blind            3/8         0/8
#:     type-blind, tight     0/8         0/8
#:     protected             8/8         8/8
#:     protected, tight      8/8         8/8
#:
#: Eight out of eight at every budget and every length tried, through 23
#: compaction rounds. Type-blind goes to nothing. This is the single largest
#: measured change to compaction in this file.
PROTECT_PREFIX = "you: "


class TieredQueue:
    """One X/Y buffer, for one `kind` ("history" or "context") of one
    session's MemoryStore. `summarize` is the only injected dependency --
    a plain `str -> str` callable (a real caller wires it to
    ROUTER.chat_model(Task.SUMMARIZE, ...), agent/pipeline/nodes.py's own
    summarizer role model; a test hands it a canned/fake function) -- kept
    that way so this file never imports agent.pipeline/agent.router and
    never needs a live model, network, or LangGraph to be exercised.
    """

    def __init__(
        self,
        kind: str,
        store: MemoryStore,
        summarize: Callable[[str], str],
        x_budget: int = X_BUDGET,
        y_budget: int = Y_BUDGET,
        reabstract: bool = REABSTRACT,
        protect: str = PROTECT_PREFIX,
    ) -> None:
        self.kind = kind
        self.store = store
        self.summarize = summarize
        self.x_budget = x_budget
        self.y_budget = y_budget
        self.reabstract = reabstract
        self.protect = protect
        self._x: list[str] = []
        self._y_raw: list[str] = []
        self._y_bullets: list[NewBullet] = []
        self._generation = 0

    # ---- writing -----------------------------------------------------

    def append(self, text: str) -> None:
        if not text:
            return
        self._x.append(text)
        if self._tokens(self._x) > self.x_budget:
            self._y_raw.extend(self._x)
            self._x = []
            self._compact_y_if_full()

    def _shed_oldest(self, bullets: list[NewBullet]) -> list[NewBullet]:
        """Bring `bullets` back inside the Y budget without rewriting any of them.

        Only reached when re-abstraction is off. The oldest bullets' TEXT is
        dropped and their citations are merged into the oldest survivor, so
        every raw chunk stays reachable through a live bullet -- which is what
        agent/memory/retrieval.py's stage one depends on -- while no surviving
        summary has been through a model twice.

        A clean loss of the oldest summaries, rather than a slow degradation
        of all of them. Which is better is the question the bench exists to
        answer; this is the half that does not re-abstract.
        """
        while len(bullets) > 1 and self._tokens([b.text for b in bullets]) > self.y_budget:
            dropped, keeper = bullets[0], bullets[1]
            bullets = [
                NewBullet(text=keeper.text,
                          hash_refs=sorted({*dropped.hash_refs, *keeper.hash_refs})),
                *bullets[2:],
            ]
        return bullets

    @staticmethod
    def _tokens(items: list[str]) -> int:
        return sum(count_tokens(t) for t in items)

    def _y_tokens(self) -> int:
        return self._tokens(self._y_raw) + self._tokens([b.text for b in self._y_bullets])

    def _compact_y_if_full(self) -> None:
        if self._y_tokens() <= self.y_budget:
            return
        self._generation += 1

        # Every current Y item, oldest first: prior bullets, then this
        # cycle's newly-overflowed raw text -- and each raw item's hash,
        # flushed to the store right away (permanent, regardless of
        # whether the summarizer's reply parses cleanly below).
        items: list[str] = []
        item_hashes: list[list[str]] = []
        # Prior bullets join the summariser's input ONLY when re-abstraction
        # is allowed. Otherwise they are carried forward untouched and the
        # budget is freed by _shed_oldest below instead.
        carried: list[NewBullet] = [] if self.reabstract else list(self._y_bullets)
        for bullet in (self._y_bullets if self.reabstract else ()):
            # A protected bullet stays protected. It kept its verbatim text
            # last time precisely so it would not be summarised, and feeding
            # it to the summariser on the NEXT round would undo that one
            # generation later -- which is exactly what happened the first
            # time this was written, and what the test now holds.
            if self.protect and bullet.text.startswith(self.protect):
                carried.append(bullet)
                continue
            items.append(bullet.text)
            item_hashes.append(bullet.hash_refs)
        # One batched embed() for the whole flush rather than one per item --
        # each chunk carries its own vector so retrieval.py can rank the RAW
        # text, not just the bullets summarizing it (agent/memory/store.py's
        # `chunks.embedding`).
        chunk_embeddings = _embed_each(self._y_raw)
        for text, embedding in zip(self._y_raw, chunk_embeddings):
            h = content_hash(text)
            self.store.add_chunk(self.kind, h, text, embedding)
            if self.protect and text.startswith(self.protect):
                # Never shown to the summariser. It becomes its own bullet,
                # word for word -- see PROTECT_PREFIX.
                carried.append(NewBullet(text=text, hash_refs=[h]))
                continue
            items.append(text)
            item_hashes.append([h])

        reply = self.summarize(_build_summarize_prompt(items))
        parsed = _parse_bullets(reply, len(items))
        if not parsed:
            # The summarizer's reply didn't parse at all -- fail toward
            # ONE bullet covering everything rather than silently losing
            # this cycle's content (same "fail closed toward keeping it"
            # spirit as agent/pipeline/nodes.py's own unparseable-reply
            # handling, _tool_loop's UNPARSEABLE_FEEDBACK retry).
            parsed = [(f"(unparsed summary of {len(items)} items)", list(range(1, len(items) + 1)))]

        new_bullets = [
            NewBullet(text=bullet_text, hash_refs=sorted({h for i in indices for h in item_hashes[i - 1]}))
            for bullet_text, indices in parsed
        ]
        # Nothing above guarantees the summarizer cited every item, and an
        # item no surviving bullet points at is one recall() can never find
        # again -- see _uncited_bullets for what that costs, measured.
        new_bullets += _uncited_bullets(
            items, item_hashes,
            covered={i for _, indices in parsed for i in indices},
            prior_bullet_count=len(self._y_bullets) if self.reabstract else 0,
        )
        new_bullets = carried + new_bullets

        if not self.reabstract:
            new_bullets = self._shed_oldest(new_bullets)

        for bullet, embedding in zip(new_bullets, _embed_each([b.text for b in new_bullets])):
            self.store.add_bullet(self.kind, self._generation, bullet.text, bullet.hash_refs, embedding)

        self.store.supersede_bullets(self.kind, before_generation=self._generation)

        self._y_raw = []
        self._y_bullets = new_bullets

    # ---- reading -------------------------------------------------------

    @property
    def recent_items(self) -> list[str]:
        """X's current raw items, oldest to newest, verbatim, exactly as
        appended -- unlike `earlier_view()`/`current_view()` (below) this
        is a plain list, not pre-formatted text, so a caller that needs to
        rebuild something item-shaped from it (agent/memory/wiring.py's
        reconstruction of real Human/AIMessage objects for the graph, one
        per recent turn) doesn't have to re-parse a rendered blob to get
        back what it just put in.
        """
        return list(self._x)

    def earlier_view(self) -> str:
        """Everything OLDER than X -- Y's bullets (compacted) followed by
        Y's own raw overflow that hasn't triggered a compaction yet (still
        verbatim -- Y filling up doesn't erase anything, only Y's OWN
        budget being exceeded does). Empty string if neither has anything
        yet. `current_view()` is this plus X; a caller that wants to keep
        "recent, verbatim" (X) and "older, likely-compacted" (this) in two
        separate places in its own prompt -- rather than one combined blob
        -- calls this directly instead of re-splitting current_view()'s
        output back apart.
        """
        parts = []
        if self._y_bullets:
            parts.append("EARLIER (summarized):\n" + "\n".join(f"- {b.text}" for b in self._y_bullets))
        if self._y_raw:
            parts.append("EARLIER (not yet summarized):\n" + "\n\n".join(self._y_raw))
        return "\n\n".join(parts)

    def current_view(self) -> str:
        """What a prompt actually reads, oldest to newest: `earlier_view()`
        (Y's bullets, then Y's own not-yet-compacted raw overflow) followed
        by X's raw items (newest, verbatim). Never the DB directly.
        """
        parts = []
        earlier = self.earlier_view()
        if earlier:
            parts.append(earlier)
        if self._x:
            parts.append("RECENT:\n" + "\n\n".join(self._x))
        return "\n\n".join(parts)

    @property
    def has_content(self) -> bool:
        return bool(self._x or self._y_bullets or self._y_raw)
