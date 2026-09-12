"""A bank of lessons Otto distils from its own finished runs, and reads back
at the start of the next one. This is the self-evolution mechanism, and it is
deliberately the smallest shape the evidence supports.

WHY THIS SHAPE AND NOT A BIGGER ONE. Automatic harness evolution, benchmarked
against plain repeated sampling under matched feedback and inference budgets,
does not consistently win and generalises poorly -- one evolved system showed
a 31.7-point gap between its own proxy metric and held-out tasks. Across five
methods, three frontier models and three regimes, the measured gains were
+1.37% isolated, +0.75% sequential, +0.90% interleaved, and NEGATIVE in all
three for the strongest model tested. So the version of self-evolution worth
building is the one with the best-attested numbers and the least machinery:
distilling short reusable lessons from a run's own trajectory, +4.6 to +8.3pp
with about 1.4x fewer steps.

FOUR RULES, EACH FROM A MEASURED FAILURE:

  1. LEARN FROM FAILURES TOO. A bank built only from successes throws away
     the half of the signal that says what not to do -- and in this design a
     failed run is where the sharpest lesson usually is.

  2. FEW, AND SHORT. `MAX_PER_RUN` is 3. Injecting ten skill-like items every
     turn, with the model deciding when they apply, scored 16.4 points BELOW
     a variant that injected none and tracked verified state instead, while
     costing more per success. More retrieved text is not more help.

  3. READ ONE. `TOP_K` is 1. Task-time procedural recall peaks at k=1 and
     loses about 7 points by k=5, because retrieved context starves attention
     from the thing being acted on. The same store's question-answering read
     path rises with k; these are different jobs and they get different
     numbers (agent/memory/retrieval.py's PROCEDURAL_TOP_K says the same).

  4. ABSTRACT ONCE, FROM RAW. Consolidating a model's own correct solutions
     into memory and feeding them back made it fail 54% of the problems it had
     previously solved; raw-episode retention doubled accuracy against forced
     consolidation. So a lesson is distilled from a TRAJECTORY and never from
     another lesson -- `distil` is never shown the bank. What existing lessons
     are used for is adjudication at WRITE time, which is a comparison, not a
     re-abstraction: a near-duplicate is dropped rather than stacked, because
     retrieved evidence changes a stored memory only 3.3% of the time when
     that decision is left to read time (8.7% -> 68.0% validity when moved to
     the write).

WHERE IT LIVES. Its own SQLite file, not a session store: a lesson that only
survives the session that learned it is not a lesson. Same `MemoryStore`
schema, `kind="lesson"`, ranked by the same embeddings as everything else,
and degrading the same way -- if the embedding backend is unreachable the
bank falls back to most-recent rather than failing the run.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path

from agent.memory.embeddings import (
    EmbeddingUnavailable,
    cosine_similarity,
    current_model_name,
    embed,
    embed_query,
)
from agent.memory.hashing import content_hash
from agent.memory.store import DB_DIR, MemoryStore

log = logging.getLogger(__name__)

#: The `kind` every lesson is stored under, so one bank file could later hold
#: other cross-run material without the two ranking against each other.
KIND = "lesson"

#: How many lessons one finished run may contribute. Three, per rule 2 above.
#: A run that wants to write ten has not learned ten things, it has summarised
#: itself.
MAX_PER_RUN = 3

#: How many lessons a starting run reads. One, per rule 3.
TOP_K = 1

#: Above this cosine similarity to a lesson already in the bank, a new one is
#: treated as the same lesson said again and dropped. Chosen to be tolerant of
#: rewording and intolerant of genuine repetition -- the failure being
#: prevented is a bank that fills with fifty phrasings of "read the file
#: before editing it", which then crowds out everything else at k=1.
DUPLICATE_SIMILARITY = 0.93

#: A lesson under this similarity to the task at hand is not offered at all.
#: Recalling an irrelevant lesson is worse than recalling none: it is the
#: always-on skill injection that measured 16.4 points down.
MIN_RELEVANCE = 0.35

#: Ceiling on one lesson's stored text. A lesson that needs more than this is
#: a transcript, and transcripts are what the raw store is for.
MAX_LESSON_CHARS = 400


@dataclass(frozen=True)
class Lesson:
    """One transferable thing, in three parts.

    Split rather than free text because the parts do different jobs: `cue` is
    what gets matched against the next task, `action` is what gets acted on,
    and `outcome` says whether this is a path to take or avoid. A single prose
    blob would have to be re-read to tell those apart every time.
    """

    #: When this applies -- the situation, not the specific task.
    cue: str
    #: What to do about it.
    action: str
    #: "worked" or "failed". A lesson from a failed run is still a lesson.
    outcome: str = "worked"

    def rendered(self) -> str:
        """The stored form, which `_parse` has to be able to read back.

        The parts are trimmed to fit the budget JOINTLY, rather than the
        assembled string being sliced at the end. Slicing the whole thing cut
        the trailing `[outcome]` off any lesson over the limit, and `_parse`
        requires that suffix -- so a long lesson was written to the bank and
        could never be read out of it again. A silent, permanent slot loss
        with no error anywhere, measured on a 400-character pair.
        """
        frame = len("When : [] ") + len(self.outcome)
        room = max(0, MAX_LESSON_CHARS - frame)
        cue, action = self.cue, self.action
        if len(cue) + len(action) > room:
            # Halve the budget between them, then give whatever one of them
            # does not need to the other -- a short cue should not force a
            # long action to be cut.
            share = room // 2
            if len(cue) <= share:
                action = action[: room - len(cue)]
            elif len(action) <= share:
                cue = cue[: room - len(action)]
            else:
                cue, action = cue[:share], action[: room - share]
        return f"When {cue}: {action} [{self.outcome}]"


#: "nobody has said" -- distinct from an explicit None, which means OFF.
_UNSET = object()

_BANK: ContextVar = ContextVar("otto_lesson_bank", default=_UNSET)

#: Whether a run may WRITE to the bank. Separate from whether it may read,
#: because the held-out measurement needs exactly that combination: read
#: everything the development runs learned, contribute nothing back. Collapsing
#: the two into one switch would mean the honest number could only be produced
#: by a run that learns nothing at all -- which measures the wrong thing, since
#: what is being tested is whether the lessons TRANSFER.
_WRITES: ContextVar[bool] = ContextVar("otto_lesson_writes", default=True)


def bank_path() -> Path:
    return DB_DIR / "lessons.db"


@contextmanager
def bind_bank(store: MemoryStore | None):
    """Point the bank somewhere else for the duration -- a test's tmp_path, or
    a benchmark that must not write into the real one. Same seam shape as
    agent/pipeline/execution.py's command runner, for the same reason: the
    thing being swapped is a side effect on the world.

    `None` means OFF, not "use the default". A held-out measurement has to be
    able to say "learn nothing from this", or the held-out set stops being
    held out after the first run against it.
    """
    token = _BANK.set(store)
    try:
        yield
    finally:
        _BANK.reset(token)


@contextmanager
def read_only():
    """Lessons may be read, none may be written, for the duration.

    This is what makes a held-out set held out. The loop reads what earlier
    runs learned and its results feed nothing back, so the measured number
    answers "do these lessons transfer" rather than "did the loop find
    something that works on the tasks it was tuned on" -- a distinction one
    evolved system got 31.7 points wrong.
    """
    token = _WRITES.set(False)
    try:
        yield
    finally:
        _WRITES.reset(token)


def _bank() -> MemoryStore | None:
    bound = _BANK.get()
    if bound is not _UNSET:
        return bound
    try:
        return MemoryStore(bank_path())
    except (OSError, ValueError) as exc:  # a bank that cannot open is not a failed run
        log.warning("lesson bank unavailable: %s", exc)
        return None


def learning_enabled() -> bool:
    """Whether there is anywhere to write. Checked BEFORE the distilling call
    is made, not after: spending a model call to produce lessons that are then
    thrown away is the worst of both."""
    return _WRITES.get() and _bank() is not None


def _embed_one(text: str):
    try:
        return embed([text])[0], current_model_name()
    except (EmbeddingUnavailable, OSError, ValueError, RuntimeError) as exc:
        log.debug("lesson not embedded: %s", exc)
        return None, None


# --------------------------------------------------------------------------
# Reading: one lesson, only if it is actually about this
# --------------------------------------------------------------------------

def recall_lessons(task: str, *, top_k: int = TOP_K) -> list[Lesson]:
    """The most relevant stored lessons for `task`, at most `top_k` of them.

    Returns nothing -- not the most recent, not a default -- when the bank is
    empty or when nothing clears `MIN_RELEVANCE`. "No lesson" is a correct and
    common answer, and an agent given a lesson about something else is worse
    off than one given silence.
    """
    store = _bank()
    if store is None or not task.strip():
        return []

    rows = store.get_chunk_rows(KIND, store.chunk_hashes(KIND))
    if not rows:
        return []

    try:
        query = embed_query(task)
        model = current_model_name()
    except Exception:
        # No ranking available. Most-recent is the wrong answer here: an
        # unranked lesson is an irrelevant lesson with extra steps.
        return []

    scored = [
        (cosine_similarity(query, row.embedding), row)
        for row in rows
        if row.embedding is not None and row.embedding_model == model
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        _parse(row.content)
        for score, row in scored[:top_k]
        if score >= MIN_RELEVANCE and _parse(row.content) is not None
    ]


def all_lessons() -> list[Lesson]:
    """Everything in the bank, oldest first. For inspection and for tests --
    the agent reads through `recall_lessons`, which is ranked and capped."""
    store = _bank()
    if store is None:
        return []
    parsed = [_parse(row.content) for row in store.get_chunk_rows(KIND, store.chunk_hashes(KIND))]
    return [lesson for lesson in parsed if lesson is not None]


# --------------------------------------------------------------------------
# Writing: at most three, adjudicated against what is already there
# --------------------------------------------------------------------------

def record_lessons(lessons, *, max_per_run: int = MAX_PER_RUN) -> list[Lesson]:
    """Store `lessons`, dropping near-duplicates, and return what was kept.

    The adjudication happens HERE rather than at read time, which is the whole
    of the 8.7% -> 68.0% memory-validity result: deciding at read time whether
    a retrieved memory still applies is a decision that, measured, almost never
    gets made.
    """
    store = _bank()
    if store is None or not _WRITES.get():
        return []

    existing = [
        (row.embedding, row.embedding_model)
        for row in store.get_chunk_rows(KIND, store.chunk_hashes(KIND))
    ]
    kept: list[Lesson] = []
    for lesson in list(lessons)[:max_per_run]:
        text = lesson.rendered()
        vector, model = _embed_one(text)
        if _is_duplicate(vector, model, existing):
            continue
        store.add_chunk(KIND, content_hash(text), text, vector, model)
        existing.append((vector, model))
        kept.append(lesson)
    return kept


def _is_duplicate(vector, model, existing) -> bool:
    """Near-duplicate against anything already banked.

    With no embedding available this says False: the store's own hash dedup
    still catches an exact repeat, and refusing to learn anything whenever the
    embedding backend is down would make the bank stop growing silently.
    """
    if vector is None:
        return False
    return any(
        other is not None and other_model == model
        and cosine_similarity(vector, other) >= DUPLICATE_SIMILARITY
        for other, other_model in existing
    )


# --------------------------------------------------------------------------
# The stored form
# --------------------------------------------------------------------------

_RENDERED = re.compile(r"^When (?P<cue>.+?): (?P<action>.+?) \[(?P<outcome>\w+)\]$", re.S)


def _parse(text: str) -> Lesson | None:
    match = _RENDERED.match(text.strip())
    if not match:
        return None
    return Lesson(**match.groupdict())


def parse_distilled(reply: str, *, outcome_default: str = "worked") -> list[Lesson]:
    """Lessons out of a distiller's reply, which is asked for as JSON.

    Tolerant on purpose: a reply wrapped in a fence, or with prose around the
    array, still yields its lessons. A distillation that fails to parse costs
    a run's learning silently, and this is the cheapest place to be generous.
    """
    body = reply.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", body, re.S)
    if fenced:
        body = fenced.group(1).strip()
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        items = json.loads(body[start:end + 1])
    except (json.JSONDecodeError, ValueError):
        return []

    lessons: list[Lesson] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        cue = str(item.get("cue") or "").strip()
        action = str(item.get("action") or "").strip()
        if not cue or not action:
            continue
        outcome = str(item.get("outcome") or outcome_default).strip().lower()
        lessons.append(Lesson(
            cue=cue[:MAX_LESSON_CHARS], action=action[:MAX_LESSON_CHARS],
            outcome="failed" if outcome.startswith("fail") else "worked",
        ))
    return lessons[:MAX_PER_RUN]


def as_json(lessons) -> str:
    return json.dumps([asdict(lesson) for lesson in lessons], indent=2)
