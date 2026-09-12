"""What each seat's model actually achieved, recorded so the chain can be
ordered by evidence instead of by the order somebody typed it in.

Milestone 4's last piece. Logging verified per-(task, model) outcomes and
routing on them is worth about +15.3% relative on its own, and it is the one
form of adaptation that costs no model calls: the evidence is a by-product of
runs that were happening anyway.

WHAT IS RECORDED, AND WHY SO LITTLE. A run's approval cannot be attributed to
one model when the agent changed modes mid-run -- three models touched the
answer and only one verdict came back. So a multi-mode run records NOTHING.
That throws away real data, and it is still the right call: a log that
mis-attributes is worse than a smaller honest one, because the whole purpose
of the log is to be trusted enough to change routing.

WHAT IT CANNOT LEARN. The evaluator's own seat. Every run uses it, and nothing
in a run says whether the judge was RIGHT -- crediting the judge with the
verdict it issued would be a system marking its own homework. Those seats stay
where mapping.py pins them until something outside a run says otherwise.

THE ORDERING IS CONSERVATIVE ON PURPOSE. A candidate only moves when it has
`MIN_SAMPLES` runs behind it, and it is ordered on approval rate with cost as
the tiebreak. Below that threshold the declared order in mapping.py stands
exactly as written, so a fresh install, a test, and a route nobody has
exercised all behave the way they read.
"""
from __future__ import annotations

import logging
import random
import sqlite3
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Where the log lives. Beside the memory stores rather than in the repo: it is
#: a record of what this installation has observed, not a fact about the code.
DB_DIR = Path.home() / ".otto"

#: How many runs a (task, model) pair needs before its number is allowed to
#: move anything. Small samples on a benchmark whose own scores swing 0.36
#: between identical runs are noise, and acting on noise is how a router
#: convinces itself a coin is weighted.
MIN_SAMPLES = 12

#: How often the chain is handed to the candidate we know LEAST about instead
#: of the one the evidence prefers.
#:
#: Without this the ordering freezes permanently, and the first version of
#: this file had exactly that bug. Once a candidate is demoted it stops being
#: resolved, so it stops accruing runs, so its record stays frozen at the
#: twelve samples that demoted it -- measured: 200 further runs left the
#: loser on 12. Twelve runs then decide a seat forever, and a model that
#: later improves (a newer pin, a provider that fixed something, or simply
#: twelve unlucky draws) never gets a second chance.
#:
#: One in ten, and only while the log is writable: exploring without
#: recording what happened costs a worse answer and learns nothing, which is
#: also what keeps a held-out measurement reproducible.
EXPLORATION_RATE = 0.10

#: How much better one candidate's approval rate has to be before it overtakes
#: a candidate declared above it. A margin, not a strict comparison, because at
#: MIN_SAMPLES a difference of one run is a difference of 8 points.
MIN_MARGIN = 0.10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seat_outcomes (
    task        TEXT NOT NULL,
    model_id    TEXT NOT NULL,
    runs        INTEGER NOT NULL DEFAULT 0,
    approved    INTEGER NOT NULL DEFAULT 0,
    calls       INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (task, model_id)
);
"""


@dataclass(frozen=True)
class SeatRecord:
    task: str
    model_id: str
    runs: int
    approved: int
    calls: int

    @property
    def approval_rate(self) -> float:
        return self.approved / self.runs if self.runs else 0.0

    @property
    def calls_per_run(self) -> float:
        return self.calls / self.runs if self.runs else 0.0

    @property
    def trusted(self) -> bool:
        return self.runs >= MIN_SAMPLES


_UNSET = object()
_LOG: ContextVar = ContextVar("otto_seat_outcomes", default=_UNSET)


def log_path() -> Path:
    return DB_DIR / "outcomes.db"


@contextmanager
def bind_log(path: Path | str | None):
    """Point the log somewhere else, or turn it off with None.

    Off, not "use the default" -- same reason agent/memory/lessons.py draws
    that distinction. A measurement that is supposed to hold routing fixed has
    to be able to say so, or the thing being measured changes underneath it.
    """
    token = _LOG.set(None if path is None else Path(path))
    try:
        yield
    finally:
        _LOG.reset(token)


#: Whether a run may WRITE what it observed. Separate from whether it may
#: read, because a held-out measurement needs exactly that combination: route
#: on everything the development runs observed, contribute nothing back. Same
#: split, and the same reason, as agent/memory/lessons.py's.
_WRITES: ContextVar[bool] = ContextVar("otto_seat_writes", default=True)


@contextmanager
def read_only():
    """Routing may read the log, nothing may be added to it."""
    token = _WRITES.set(False)
    try:
        yield
    finally:
        _WRITES.reset(token)


def _connect() -> sqlite3.Connection | None:
    bound = _LOG.get()
    if bound is None:
        return None
    path = log_path() if bound is _UNSET else bound
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path), check_same_thread=False)
        conn.executescript(_SCHEMA)
        return conn
    except sqlite3.Error as exc:
        # Routing must never stop because a log could not be opened.
        log.warning("seat outcome log unavailable: %s", exc)
        return None


def record(task: str, model_id: str, *, approved: bool, calls: int = 0) -> None:
    """One finished, single-mode run against one seat."""
    if not _WRITES.get():
        return
    conn = _connect()
    if conn is None or not model_id:
        return
    try:
        conn.execute(
            "INSERT INTO seat_outcomes (task, model_id, runs, approved, calls) "
            "VALUES (?, ?, 1, ?, ?) "
            "ON CONFLICT(task, model_id) DO UPDATE SET "
            "  runs = runs + 1, approved = approved + ?, calls = calls + ?",
            (str(task), model_id, int(approved), int(calls),
             int(approved), int(calls)),
        )
        conn.commit()
    except sqlite3.Error as exc:
        log.warning("could not record seat outcome: %s", exc)
    finally:
        conn.close()


def records(task: str | None = None) -> list[SeatRecord]:
    """Everything observed, best approval rate first within each task."""
    conn = _connect()
    if conn is None:
        return []
    try:
        sql = "SELECT task, model_id, runs, approved, calls FROM seat_outcomes"
        args: tuple = ()
        if task is not None:
            sql += " WHERE task = ?"
            args = (str(task),)
        rows = conn.execute(sql, args).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    found = [SeatRecord(*row) for row in rows]
    found.sort(key=lambda r: (r.task, -r.approval_rate, r.calls_per_run))
    return found


def preference(task: str) -> dict[str, SeatRecord]:
    """Trusted records for `task`, by model id. Untrusted ones are left out
    entirely rather than returned with a caveat -- a caller that has to
    remember to check `trusted` is a caller that will forget."""
    return {r.model_id: r for r in records(task) if r.trusted}


def spec_id(candidate) -> str:
    """The model id a candidate resolves to, when that is knowable without
    asking a provider. A capability query is not a model yet, so it gets "" --
    and `reorder` then leaves it exactly where it was declared, because the
    log is keyed on a model.

    Here rather than on Router because it is a fact about this log's key, and
    the router is one caller of it.
    """
    spec = getattr(candidate, "spec", None)
    if not spec:
        return ""
    _, _, model_id = spec.partition(":")
    return model_id


def reorder(task: str, candidates, model_id_of=spec_id) -> list:
    """`candidates`, best-evidenced first, declared order preserved otherwise.

    `model_id_of(candidate)` returns the model id a candidate would resolve to,
    or "" when that cannot be known without asking a provider -- an open query,
    for instance.

    ONLY TWO MEASURED CANDIDATES EVER TRADE PLACES. A candidate nobody has
    evidence about keeps its declared position, even against one with a good
    record, because "we measured A at 90%" is not evidence that B is worse than
    A -- B has no number at all. The declared order in mapping.py encodes a
    person's judgment, and unseating it needs evidence on both sides.

    Between two measured candidates the challenger has to be ahead by
    `MIN_MARGIN` to overtake. Inside that margin the incumbent keeps its place
    unless it is also spending more calls per run to get the same result, which
    is the one case where a tie is not really a tie.

    Adjacent swaps rather than a sort key: a chain is two or three candidates
    long, and "B overtakes A" is a comparison between two records, not a score
    each can be given on its own.
    """
    known = preference(task)
    if not known:
        return list(candidates)

    ordered = list(candidates)
    if explorer := _explore(ordered, known, model_id_of):
        ordered.remove(explorer)
        return [explorer, *ordered]

    for _ in range(len(ordered)):
        settled = True
        for i in range(len(ordered) - 1):
            incumbent = known.get(model_id_of(ordered[i]) or "")
            challenger = known.get(model_id_of(ordered[i + 1]) or "")
            if _overtakes(challenger, incumbent):
                ordered[i], ordered[i + 1] = ordered[i + 1], ordered[i]
                settled = False
        if settled:
            break
    return ordered


def _explore(candidates, known, model_id_of):
    """Occasionally, the trusted candidate with the fewest runs behind it.

    Exploitation alone is a ratchet: the winner keeps winning because only
    the winner is ever asked. This is the one in ten that keeps every
    measured candidate's record alive. It returns None the rest of the time,
    and always when the log cannot be written to -- an exploration nobody
    records is a worse answer bought for nothing.
    """
    if not _WRITES.get() or random.random() >= EXPLORATION_RATE:
        return None
    measured = [(known[mid], c) for c in candidates
                if (mid := model_id_of(c) or "") in known]
    if len(measured) < 2:
        return None  # nothing to explore between
    return min(measured, key=lambda pair: pair[0].runs)[1]


def _overtakes(challenger: SeatRecord | None, incumbent: SeatRecord | None) -> bool:
    """Whether the candidate declared SECOND should be tried first."""
    if challenger is None or incumbent is None:
        return False
    # `>=`, not `>`. The docstring says the challenger has to be ahead BY
    # MIN_MARGIN, and at exactly that margin a strict `>` sent it to the tie
    # branch below instead -- so "ahead by ten points" did not overtake while
    # "ahead by ten points and a hair" did. Nothing in the numbers justifies
    # that line falling between them.
    if challenger.approval_rate >= incumbent.approval_rate + MIN_MARGIN:
        return True
    if abs(challenger.approval_rate - incumbent.approval_rate) < MIN_MARGIN:
        # Same result, measurably cheaper. A fifth fewer calls per run, so
        # that ordinary run-to-run variation does not shuffle the chain.
        return challenger.calls_per_run < incumbent.calls_per_run * 0.8
    return False
