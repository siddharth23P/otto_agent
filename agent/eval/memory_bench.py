"""Memory retrieval benchmark: LoCoMo (snap-research/locomo, arXiv:2402.17753,
"Evaluating Very Long-Term Conversational Memory of LLM Agents") against
agent/memory/'s tiered queue -- how much of a long conversation Otto's
short-term memory can actually recall, not how well Otto answers open-ended
questions (a full agent QA eval is a much larger, LLM-per-question
undertaking -- see the module docstring's "Two summarizer/embedding
backends" section for the one opt-in LLM call this script DOES make).

LoCoMo's own data (github.com/snap-research/locomo, data/locomo10.json,
2.8MB, no auth/license gate): 10 synthetic-but-long two-person conversations
spanning multiple sessions across real dates weeks apart -- a few hundred to
several thousand dialogue turns each, closer to how a real long-running otto
chat session could actually grow than any conversation this repo's own
tests script by hand. Each conversation ships its own question set (`qa`),
each item citing which dialogue turn id(s) (`evidence`, e.g. "D3:7") the
answer actually depends on, and a `category`: 1=single-hop, 2=temporal,
3=multi-hop, 4=open-domain, 5=adversarial (an intentionally unanswerable
question with a plausible-sounding wrong "expected" answer, included so a
system that always guesses SOMETHING scores worse, not better -- excluded
from the coverage/recall scoring below for that reason, not measured here).

What this script measures, per conversation:
  1. Every dialogue turn is appended to a `TieredQueue`, in order, across
     every session -- one continuous history, exactly the cross-turn
     compaction path agent/memory/wiring.py wires into a real Session.
     Everything ends up either still verbatim (X, or Y's own not-yet-
     compacted raw overflow) or compacted into the permanent, hash-
     addressed chunk store -- never discarded (agent/memory/store.py's
     whole design).
  2. STORE COVERAGE: for every non-adversarial QA item, is ALL of its cited
     evidence still reachable somewhere (verbatim, or as a permanent
     chunk)? Should be ~100% by construction -- this is a correctness
     check on the ENGINE, not a retrieval-quality one; a miss here would
     be a real bug (lost information), not just an imperfect answer.
  3. RECALL COVERAGE: does `agent.memory.retrieval.recall()`, given ONLY
     the question text, actually surface at least one cited evidence
     turn's raw text in its top-k result? This is the real "smart zone of
     context" test -- whether semantic search finds the needle without
     reading everything back.
  4. Final context-budget stats: current_view()'s token count at the end
     of the whole conversation vs. the conversation's own raw token total
     -- the compression ratio agent/memory/queue.py's X/Y budgets buy.

Two summarizer/embedding backends, chosen with run_benchmark()'s `live`
flag (CLI: --live, default off): offline (default) uses a deterministic,
no-network, no-LLM canned summarizer (_canned_summarize, below) that groups
items into fixed-size, correctly-cited bullets -- so the ENGINE's own
mechanics (citation propagation, nothing-ever-lost, budget math) are
exercised and scored even with no INCEPTION_API_KEY and no embedding-model
network access (this repo's own sandboxed dev environment has neither --
see agent/memory/embeddings.py's own docstring). `live=True` routes
through the real agent.memory.wiring.summarize_for_memory (Task.SUMMARIZE)
and the real fastembed model instead -- meaningfully better summaries and
ranked (not just most-recent-fallback) recall, but needs both a working
INCEPTION_API_KEY and outbound network to Hugging Face Hub; imported lazily
(inside run_benchmark(), only when live=True) so this module stays
importable/runnable offline with zero agent.pipeline/agent.router
dependency, same decoupling agent/memory/'s own __init__.py promises for
the engine itself.
"""
from __future__ import annotations

import dataclasses
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable

from agent.memory.hashing import content_hash
from agent.memory.queue import TieredQueue
from agent.memory.retrieval import recall
from agent.memory.store import MemoryStore
from agent.memory.tokens import count_tokens

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "locomo10.json"

CATEGORY_NAMES = {
    1: "single-hop", 2: "temporal", 3: "multi-hop", 4: "open-domain", 5: "adversarial",
}
#: Excluded from coverage/recall scoring -- module docstring. Kept out of
#: CATEGORY_NAMES' iteration order in reports rather than out of the dict
#: itself, so an unrecognised/new category still prints under a legible name.
_SCORED_CATEGORIES = (1, 2, 3, 4)

#: agent/memory/queue.py's own numbered-item marker ("[1] text", "[2] text",
#: ...) -- what _canned_summarize below parses back out of the compaction
#: prompt it's handed, instead of importing _build_summarize_prompt's
#: internals (this file has no dependency on agent.memory.queue beyond its
#: public TieredQueue/NewBullet-shaped API).
_ITEM_MARKER = re.compile(r"^\[(\d+)\] ", flags=re.MULTILINE)


def download_locomo(path: Path = DEFAULT_CACHE, *, force: bool = False) -> Path:
    """Fetch data/locomo10.json (snap-research/locomo) into `path`, once --
    a plain HTTPS GET of a ~2.8MB JSON file, no auth, no license gate, no
    pip package needed. Re-downloads only if `force` or the file is
    missing; callers that already have a copy (CI, an offline machine)
    can just point `path` at it instead.
    """
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(LOCOMO_URL, timeout=30) as resp:  # nosec B310 -- fixed https URL
        path.write_bytes(resp.read())
    return path


def load_locomo(path: Path = DEFAULT_CACHE) -> list[dict]:
    return json.loads(path.read_text())


def _canned_summarize(prompt: str) -> str:
    """Deterministic, offline, no-LLM stand-in for a real summarizer --
    module docstring's "Two summarizer/embedding backends". Groups
    agent/memory/queue.py's numbered "[N] text" items into fixed-size
    chunks, one bullet per chunk, each citing EXACTLY the item numbers it
    covers -- valid input to queue.py's own `_parse_bullets()`, so a
    compaction runs its real citation-propagation logic end to end, with
    only the bullet's own prose replaced by a placeholder (a real model's
    job, not this benchmark's).
    """
    item_numbers = [int(n) for n in _ITEM_MARKER.findall(prompt)]
    if not item_numbers:
        return ""
    group_size = 4
    lines = []
    for start in range(0, len(item_numbers), group_size):
        group = item_numbers[start:start + group_size]
        cites = ",".join(str(n) for n in group)
        lines.append(f"covers items {group[0]}-{group[-1]} of this stretch [sources: {cites}]")
    return "\n".join(lines)


@dataclasses.dataclass
class QAResult:
    question: str
    category: int
    evidence_ids: list[str]
    #: Every evidence turn's raw text is reachable somewhere (verbatim or
    #: as a permanent chunk) -- an engine-correctness check, not scored
    #: retrieval quality; see module docstring point 2.
    stored: bool
    #: Every evidence turn's raw text is STILL verbatim in the live
    #: current_view() (X, or Y's not-yet-compacted raw overflow) -- no
    #: recall() needed at all, a prompt reading current_view() already has
    #: it. False means at least one evidence turn has been compacted away,
    #: which is exactly when recall() (`recalled`, below) is the thing
    #: that's actually supposed to find it.
    visible_verbatim: bool
    #: recall() actually surfaced at least one evidence turn's raw text in
    #: its top-k result, given only the question -- module docstring
    #: point 3, the real "did semantic search find the needle" signal.
    #: Only meaningful (and only scored in the summary) when
    #: `visible_verbatim` is False -- see `answerable`.
    recalled: bool
    #: The exact evidence text(s) `recalled` was computed from, joined --
    #: NOT what got shown to the summarizer (which sees "speaker: text" per
    #: item, one at a time); this is one or more full turns concatenated,
    #: for a human comparing this result against `recalled_text` below to
    #: see it in full. Kept even when `visible_verbatim` is True, so a
    #: still-verbatim item's evidence is inspectable the same way.
    evidence_text: str = ""
    #: recall()'s raw return value for this question -- always computed
    #: (recall() is a local embedding search, not an LLM call, so this
    #: costs nothing extra even for a `visible_verbatim` item). When
    #: nothing has been compacted yet for this `kind`, recall() itself
    #: returns a fixed "(nothing has been compacted away yet...)" message,
    #: never an empty string, so a blank value here would mean this field
    #: wasn't populated, not that recall() found nothing.  The point of
    #: keeping this at all: `recalled` is an exact-substring match against
    #: this text (module docstring point 3), not a fuzzy/semantic one, so
    #: seeing the actual text answers "did it find the RIGHT thing, or does
    #: this conversation just repeat similar phrasing often enough that a
    #: substring match is easy to satisfy by accident" -- read the two
    #: side by side (CLI: `otto eval-memory --show-items failures`) rather
    #: than trusting the boolean alone when a conversation has a lot of
    #: near-duplicate turns.
    recalled_text: str = ""

    @property
    def answerable(self) -> bool:
        """Could something reading ONLY current_view() + one recall() call
        actually reach this evidence right now -- the metric that
        combines both retrieval paths into one "is this still findable"
        signal, regardless of which path it went through."""
        return self.visible_verbatim or self.recalled


@dataclasses.dataclass
class ConversationResult:
    sample_id: str
    turn_count: int
    raw_tokens: int
    final_view_tokens: int
    qa_results: list[QAResult]

    @property
    def compression_ratio(self) -> float:
        return self.final_view_tokens / self.raw_tokens if self.raw_tokens else 0.0

    def scored_results(self) -> list[QAResult]:
        return [r for r in self.qa_results if r.category in _SCORED_CATEGORIES]


def _session_keys(conversation: dict) -> list[str]:
    keys = [k for k in conversation if re.fullmatch(r"session_\d+", k)]
    return sorted(keys, key=lambda k: int(k.split("_")[1]))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_verbatim(queue: TieredQueue, text: str) -> bool:
    """True if `text` is still sitting in the queue's own live view
    UN-compacted -- X's recent items, or Y's own raw overflow that hasn't
    triggered a compaction yet. A prompt that just reads current_view()
    (no recall() call at all) already has this text; see QAResult.
    visible_verbatim.
    """
    return text in queue.recent_items or text in queue._y_raw


def _is_reachable(store: MemoryStore, queue: TieredQueue, text: str) -> bool:
    """True if `text` (a raw dialogue turn, already appended to `queue`) is
    still recoverable -- verbatim in the queue's own live view (X, or Y's
    own not-yet-compacted raw overflow), or permanently stored as a chunk
    (compacted away, but agent/memory/store.py's chunks are never
    deleted). False would mean the engine actually LOST the information --
    a real bug, not just an imperfect answer (module docstring point 2).
    """
    if _is_verbatim(queue, text):
        return True
    return bool(store.get_chunks([content_hash(text)]))


def run_one_conversation(
    sample: dict,
    *,
    summarize: Callable[[str], str],
    top_k: int = 5,
    store_path: Path | None = None,
    max_turns: int | None = None,
    x_budget: int | None = None,
    y_budget: int | None = None,
) -> ConversationResult:
    """Replay one LoCoMo conversation through a fresh TieredQueue (own
    MemoryStore, own `kind`, so nothing leaks between conversations run in
    the same process) and score every QA item against it. `max_turns`
    truncates the conversation (from the start) for a fast smoke run --
    QA items whose evidence falls after the cutoff are simply skipped
    (their evidence never got appended, so scoring them would be
    meaningless either way).

    `x_budget`/`y_budget` default to agent/memory/queue.py's own real
    production constants (X_BUDGET=24,000/Y_BUDGET=80,000) when left
    None -- but every LoCoMo conversation (11K-24K raw tokens, measured)
    fits inside those without ever filling Y, so compaction never fires
    and `recall()` never has anything compacted to search (module
    docstring point 3 goes untested at those settings). Passing smaller
    values here is a deliberate "stress test" mode: it forces real
    compaction+recall to actually run, at the cost of no longer matching
    Otto's real per-turn budget.
    """
    conversation = sample["conversation"]
    path = store_path or Path(f"/tmp/otto-memory-bench-{sample['sample_id']}.db")
    if path.exists():
        path.unlink()
    store = MemoryStore(path)
    kind = f"locomo-{sample['sample_id']}"
    queue_kwargs = {}
    if x_budget is not None:
        queue_kwargs["x_budget"] = x_budget
    if y_budget is not None:
        queue_kwargs["y_budget"] = y_budget
    queue = TieredQueue(kind, store, summarize=summarize, **queue_kwargs)

    turn_by_id: dict[str, str] = {}
    raw_tokens = 0
    turn_count = 0
    for session_key in _session_keys(conversation):
        for turn in conversation[session_key]:
            if max_turns is not None and turn_count >= max_turns:
                break
            text = f"{turn['speaker']}: {turn['text']}"
            turn_by_id[turn["dia_id"]] = text
            queue.append(text)
            raw_tokens += count_tokens(text)
            turn_count += 1
        if max_turns is not None and turn_count >= max_turns:
            break

    qa_results = []
    for qa in sample.get("qa", []):
        evidence_ids = [e for e in qa.get("evidence") or [] if e in turn_by_id]
        if not evidence_ids:
            continue  # no evidence, or it fell past max_turns -- nothing to score
        evidence_texts = [turn_by_id[e] for e in evidence_ids]

        stored = all(_is_reachable(store, queue, text) for text in evidence_texts)
        visible_verbatim = all(_is_verbatim(queue, text) for text in evidence_texts)
        recalled_text = recall(store, kind, qa["question"], top_k=top_k)
        recalled = any(_normalize(t) in _normalize(recalled_text) for t in evidence_texts)

        qa_results.append(QAResult(
            question=qa["question"], category=qa.get("category", 0),
            evidence_ids=evidence_ids, stored=stored,
            visible_verbatim=visible_verbatim, recalled=recalled,
            evidence_text="\n".join(evidence_texts), recalled_text=recalled_text,
        ))

    final_view_tokens = count_tokens(queue.current_view())
    store.close()
    path.unlink(missing_ok=True)

    return ConversationResult(
        sample_id=sample["sample_id"], turn_count=turn_count, raw_tokens=raw_tokens,
        final_view_tokens=final_view_tokens, qa_results=qa_results,
    )


@dataclasses.dataclass
class BenchmarkReport:
    live: bool
    conversations: list[ConversationResult]

    def to_dict(self) -> dict:
        return {
            "live": self.live,
            "conversations": [
                {
                    "sample_id": c.sample_id,
                    "turn_count": c.turn_count,
                    "raw_tokens": c.raw_tokens,
                    "final_view_tokens": c.final_view_tokens,
                    "compression_ratio": c.compression_ratio,
                    "qa": [dataclasses.asdict(r) for r in c.qa_results],
                }
                for c in self.conversations
            ],
            "summary": self.summary(),
        }

    def summary(self) -> dict:
        by_category: dict[int, list[QAResult]] = {c: [] for c in _SCORED_CATEGORIES}
        for conv in self.conversations:
            for r in conv.scored_results():
                by_category.setdefault(r.category, []).append(r)

        def _rates(results: list[QAResult]) -> dict:
            n = len(results)
            return {
                "n": n,
                "store_coverage": (sum(r.stored for r in results) / n) if n else None,
                #: Still verbatim in current_view() -- no recall() call
                #: needed at all. High at Otto's real production budget
                #: (module docstring point 3's caveat): most LoCoMo
                #: conversations never fill Y, so this is where nearly
                #: all the coverage comes from at those settings.
                "visible_verbatim_coverage": (
                    sum(r.visible_verbatim for r in results) / n
                ) if n else None,
                #: recall() actually surfaced it, given only the
                #: question -- the real semantic-search signal. Only
                #: non-zero once something has actually been compacted
                #: away (a smaller x_budget/y_budget "stress test" run,
                #: or a conversation long enough at production settings).
                "recall_coverage": (sum(r.recalled for r in results) / n) if n else None,
                #: visible_verbatim OR recalled -- "is this still
                #: findable at all, through either path" (QAResult.
                #: answerable). The one number that answers the actual
                #: question this benchmark asks.
                "answerable_coverage": (
                    sum(r.answerable for r in results) / n
                ) if n else None,
            }

        all_scored = [r for conv in self.conversations for r in conv.scored_results()]
        raw_total = sum(c.raw_tokens for c in self.conversations)
        final_total = sum(c.final_view_tokens for c in self.conversations)
        return {
            "overall": _rates(all_scored),
            "by_category": {
                CATEGORY_NAMES.get(cat, str(cat)): _rates(results)
                for cat, results in sorted(by_category.items())
            },
            "overall_compression_ratio": (final_total / raw_total) if raw_total else None,
            "conversation_count": len(self.conversations),
        }


def run_benchmark(
    samples: list[dict],
    *,
    live: bool = False,
    top_k: int = 5,
    max_turns: int | None = None,
    x_budget: int | None = None,
    y_budget: int | None = None,
) -> BenchmarkReport:
    """`live=True` swaps in the real Task.SUMMARIZE-backed summarizer
    (agent.memory.wiring.summarize_for_memory, needs INCEPTION_API_KEY) in
    place of `_canned_summarize` -- imported lazily, here, so the offline
    default never touches agent.pipeline/agent.router at all (module
    docstring). There is no separate embeddings switch: agent/memory/
    queue.py and retrieval.py already always try the real local embedding
    model first and fall back to EmbeddingUnavailable-driven unranked
    recall automatically (agent/memory/embeddings.py) -- `live` changes
    summary quality, not whether embeddings get attempted.

    `x_budget`/`y_budget`, left None, default to agent/memory/queue.py's
    own production constants inside run_one_conversation() -- pass smaller
    values for a "stress test" run that actually forces compaction (see
    run_one_conversation()'s own docstring for why that matters).
    """
    if live:
        from agent.memory.wiring import summarize_for_memory
        summarize = summarize_for_memory
    else:
        summarize = _canned_summarize

    results = [
        run_one_conversation(
            sample, summarize=summarize, top_k=top_k, max_turns=max_turns,
            x_budget=x_budget, y_budget=y_budget,
        )
        for sample in samples
    ]
    return BenchmarkReport(live=live, conversations=results)
