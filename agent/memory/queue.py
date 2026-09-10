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
project's claude/otto-tiered-memory-design.md for what's built (this
engine) vs. what's still pending.

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
    ) -> None:
        self.kind = kind
        self.store = store
        self.summarize = summarize
        self.x_budget = x_budget
        self.y_budget = y_budget
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
        for bullet in self._y_bullets:
            items.append(bullet.text)
            item_hashes.append(bullet.hash_refs)
        for text in self._y_raw:
            h = content_hash(text)
            self.store.add_chunk(self.kind, h, text)
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

        for bullet in new_bullets:
            embedding = None
            try:
                [embedding] = embed([bullet.text])
            except EmbeddingUnavailable:
                embedding = None  # store.py's Bullet.embedding=None is a
                                   # valid state -- retrieval.py falls back.
            self.store.add_bullet(self.kind, self._generation, bullet.text, bullet.hash_refs, embedding)

        self.store.supersede_bullets(self.kind, before_generation=self._generation)

        self._y_raw = []
        self._y_bullets = new_bullets

    # ---- reading -------------------------------------------------------

    def current_view(self) -> str:
        """What a prompt actually reads, oldest to newest: Y's bullets
        (compacted), then Y's own raw overflow that hasn't triggered a
        compaction YET (still verbatim -- Y filling up doesn't erase
        anything, only Y's OWN budget being exceeded does), then X's raw
        items (newest, verbatim). Never the DB directly.
        """
        parts = []
        if self._y_bullets:
            parts.append("EARLIER (summarized):\n" + "\n".join(f"- {b.text}" for b in self._y_bullets))
        if self._y_raw:
            parts.append("EARLIER (not yet summarized):\n" + "\n\n".join(self._y_raw))
        if self._x:
            parts.append("RECENT:\n" + "\n\n".join(self._x))
        return "\n\n".join(parts)

    @property
    def has_content(self) -> bool:
        return bool(self._x or self._y_bullets or self._y_raw)
