"""The session index: which `otto chat`/`otto tui` sessions exist, so one can
be listed, resumed, renamed or deleted (2026-09-13, "session manager").

Before this, a session was a `uuid4` minted in agent/cli/shell.py and a
SQLite file named after it under ~/.otto/memory/ -- and that was all. There
were three thousand of those files on the machine this was written on,
three short of all of them empty (a run opens its file before the first
turn, and every test and benchmark that ran the graph opened one), and the
three that held anything held only what compaction had retired: the live
tiers of the queue were never written, so even a session with real history
on disk could not be picked up again. `TieredQueue(restore=True)` fixes
the second half (agent/memory/queue.py); this file is the first: one small
table, ~/.otto/sessions.db, sibling of outcomes.db and routes.json, saying
which ids are sessions a person had, what they were about, and when.

A row is written by the first FINISHED turn (agent/cli/shell.py's
`Session.record_turn`), not by opening a session: a session that never got
an answer has nothing to come back to, and registering it would recreate
the clutter this exists to end. The title is the first thing the person
said, cut short, until they rename it.

Deliberately imports only agent.memory.store, like the rest of this
package's engine: the CLI (agent/cli/sessions.py, shell.py, tui.py) sits on
top of it, never the other way round.
"""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterator

from agent.config.home import otto_home
from agent.memory import store as store_module
from agent.memory.store import MemoryStore, session_db_path

#: Sibling of agent/router/outcomes.py's outcomes.db: per-installation state
#: about the person's own sessions, not one session's memory. Under
#: OTTO_HOME when that is set (agent/config/home.py).
DEFAULT_INDEX_PATH = otto_home() / "sessions.db"

#: Where the index is right now, or None for the default. A module global,
#: NOT a contextvar like agent/router/outcomes.py's log binding: the TUI
#: records a turn from a worker thread (agent/cli/tui.py's run_turn), and a
#: contextvar bound on the UI thread is invisible there -- the first version
#: of this file was a contextvar, and every TUI test that finished a turn
#: wrote a fake session into the developer's real index through exactly
#: that gap. Where the index lives is a fact about the process, not about
#: one task, so process-wide state is the honest representation.
_INDEX: Path | None = None

#: How much of a first message becomes a title. Long enough to tell two
#: sessions apart in a list, short enough for a 32-column sidebar to show
#: most of it.
TITLE_LENGTH = 60

#: What a session id a person's own otto mints looks like: `uuid4().hex`.
#: The hosts (agent/embed.py, agent/server/app.py) accept nothing else from a
#: caller; this module itself still takes the plain names tests and
#: benchmarks use ("abc", "eval-1a2b3c4d"), which agent/memory/store.py's
#: `session_db_path` keeps from ever naming a path.
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")
#: A reference a host accepts for `resolve`: "last" or a hex prefix of an id.
_REF_RE = re.compile(r"^[0-9a-f]{1,32}$")


class InvalidSessionId(ValueError, LookupError):
    """A session id or reference that is not one otto could have minted.

    Both a ValueError (it is a bad argument) and a LookupError (no session
    can match it), so a host that already turns LookupError into "no such
    session" keeps working, and one that wants to say "that is not an id"
    can catch this first."""


def valid_id(session_id: object) -> bool:
    """Whether `session_id` is a full id in the shape `uuid4().hex` gives."""
    return isinstance(session_id, str) and bool(SESSION_ID_RE.fullmatch(session_id))


def valid_ref(ref: object) -> bool:
    """Whether `ref` is something `resolve` may be asked for by a host:
    "last", or a lower-case hex prefix of an id."""
    return isinstance(ref, str) and (ref == "last" or bool(_REF_RE.fullmatch(ref)))


def check_id(session_id: object) -> str:
    """`session_id`, or InvalidSessionId -- the hosts' one gate."""
    if not valid_id(session_id):
        raise InvalidSessionId(f"{str(session_id)[:80]!r} is not a session id")
    return session_id  # type: ignore[return-value]


def check_ref(ref: object) -> str:
    if not valid_ref(ref):
        raise InvalidSessionId(f"{str(ref)[:80]!r} is not a session id, prefix or 'last'")
    return ref  # type: ignore[return-value]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    workspace TEXT,
    created_at TEXT NOT NULL,
    last_active_at TEXT NOT NULL,
    turns INTEGER NOT NULL DEFAULT 0
);
"""


@dataclass(frozen=True)
class SessionInfo:
    id: str
    title: str
    workspace: str | None
    created_at: str
    last_active_at: str
    turns: int

    @property
    def short_id(self) -> str:
        """Eight hex characters, the way git shows a commit: enough to be
        unique among any number of sessions a person will ever have, and
        short enough to type after `/resume`."""
        return self.id[:8]

    @property
    def label(self) -> str:
        return self.title or "(untitled)"


def index_path() -> Path:
    return _INDEX if _INDEX is not None else DEFAULT_INDEX_PATH


@contextmanager
def bind_index(path: Path | str) -> Iterator[None]:
    """Point the index somewhere else for the duration -- a test's tmp_path,
    the same seam shape as agent/router/outcomes.py's bind_log, for the same
    reason: the thing being swapped is a file in the person's home. Seen by
    every thread (see `_INDEX`); not re-entrant across threads, which no
    caller needs."""
    global _INDEX
    before = _INDEX
    _INDEX = Path(path)
    try:
        yield
    finally:
        _INDEX = before


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _count(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row(r: tuple) -> SessionInfo:
    return SessionInfo(id=r[0], title=r[1], workspace=r[2], created_at=r[3],
                       last_active_at=r[4], turns=int(r[5]))


_SELECT = "SELECT id, title, workspace, created_at, last_active_at, turns FROM sessions"


def title_from(text: str) -> str:
    """The first line of what the person said, whitespace collapsed, cut to
    TITLE_LENGTH with an ellipsis -- a title, not a transcript."""
    line = " ".join(text.strip().split())
    if len(line) <= TITLE_LENGTH:
        return line
    return line[: TITLE_LENGTH - 1].rstrip() + "…"


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def touch(session_id: str, *, title: str = "", workspace: Path | str | None,
          turns: int) -> SessionInfo:
    """Record that `session_id` just finished a turn: create its row if this
    was the first, else bump last_active/turns/workspace. `title` only ever
    fills an EMPTY title -- the first message names the session, a rename
    keeps its name, later messages change nothing."""
    now = _now()
    ws = str(workspace) if workspace is not None else None
    with _connect() as conn:
        existing = conn.execute(f"{_SELECT} WHERE id = ?", (session_id,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO sessions (id, title, workspace, created_at, last_active_at, turns) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, title, ws, now, now, turns),
            )
        else:
            conn.execute(
                "UPDATE sessions SET title = CASE WHEN title = '' THEN ? ELSE title END, "
                "workspace = ?, last_active_at = ?, turns = ? WHERE id = ?",
                (title, ws, now, turns, session_id),
            )
        return _row(conn.execute(f"{_SELECT} WHERE id = ?", (session_id,)).fetchone())


def rename(session_id: str, title: str, *, workspace: Path | str | None = None) -> SessionInfo:
    """Set the title. Creates the row if the session has not finished a turn
    yet -- a person who names a session means to keep it."""
    title = " ".join(title.split())
    now = _now()
    with _connect() as conn:
        existing = conn.execute(f"{_SELECT} WHERE id = ?", (session_id,)).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO sessions (id, title, workspace, created_at, last_active_at, turns) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (session_id, title, str(workspace) if workspace is not None else None, now, now),
            )
        else:
            conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        return _row(conn.execute(f"{_SELECT} WHERE id = ?", (session_id,)).fetchone())


#: Files in the memory directory that are not sessions and must never be
#: deleted or pruned as one.
_NOT_SESSIONS = frozenset({"lessons.db"})


def delete(session_id: str) -> bool:
    """Forget the session: its index row and its memory file. True if either
    existed. The file goes too because the row was the only thing that made
    it findable; without one it is the orphan `prune()` would remove next.

    Confined: only a file directly in the memory directory, and never the
    lesson bank that shares it -- checked before anything is removed, so a
    refused id leaves the row too. ValueError for anything else."""
    path = session_db_path(session_id)
    root = store_module.DB_DIR.resolve()
    if path.name in _NOT_SESSIONS or path.resolve().parent != root:
        raise ValueError(f"{session_id!r} does not name a session file")
    removed = False
    if not _absent():
        with _connect() as conn:
            removed = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount > 0
    if path.exists():
        try:
            path.unlink()
        except PermissionError as exc:
            # Windows: the file is open -- in this process (a caller that
            # forgot Session.close()) or in another otto. Say which file
            # rather than leak WinError 32; the row is already gone, so a
            # later prune() cannot find the file either -- it is the one
            # kind of orphan that is meant to be deleted by hand.
            raise OSError(f"{path} is still open, in this otto or another; close that session "
                          f"first") from exc
        removed = True
    return removed


@dataclass(frozen=True)
class PruneReport:
    #: Orphan files (no index row) that held nothing, now deleted.
    removed_files: int
    #: Index rows whose file was gone, now dropped.
    dropped_rows: int
    #: Orphan files that DO hold something. Left alone and counted, because
    #: deleting data nobody can name is not this function's call to make.
    kept_orphans: int

    def summary(self) -> str:
        parts = [f"removed {self.removed_files} empty file(s)",
                 f"dropped {self.dropped_rows} stale row(s)"]
        if self.kept_orphans:
            parts.append(f"kept {self.kept_orphans} unindexed file(s) that hold history")
        return ", ".join(parts)


def prune() -> PruneReport:
    """Clear what accumulates on its own: memory files no session row names
    and nothing was ever written to (every graph run before 2026-09-13's
    conftest fix left one, benchmarks still do), and rows whose file is gone.

    A file that is unindexed but not empty is kept and counted -- a session
    from before the index existed, most likely. It cannot be resumed (its
    live tiers were never written), but its compacted history is real and
    `recall_memory` could still be pointed at it by hand.
    """
    with _connect() as conn:
        known = {r[0] for r in conn.execute("SELECT id FROM sessions")}
    removed = kept = 0
    for path in sorted(store_module.DB_DIR.glob("*.db")):
        if path.name in _NOT_SESSIONS or path.stem in known:
            continue
        store = MemoryStore(path)
        try:
            empty = store.is_empty()
        finally:
            store.close()
        if empty:
            path.unlink()
            removed += 1
        else:
            kept += 1
    dropped = 0
    with _connect() as conn:
        for sid in known:
            try:
                gone = not session_db_path(sid).exists()
            except ValueError:
                gone = True  # a row no file could ever belong to
            if gone:
                conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))
                dropped += 1
    return PruneReport(removed_files=removed, dropped_rows=dropped, kept_orphans=kept)


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def _absent() -> bool:
    """Reads do not create the file: `otto sessions` on a fresh machine
    should say "nothing saved" and leave the home directory as it found
    it. Only a write (touch, rename) brings the index into being."""
    return not index_path().exists()


def get(session_id: str) -> SessionInfo | None:
    if _absent():
        return None
    with _connect() as conn:
        row = conn.execute(f"{_SELECT} WHERE id = ?", (session_id,)).fetchone()
    return _row(row) if row else None


def list_sessions(limit: int | None = None) -> list[SessionInfo]:
    """Newest activity first -- the one a person most likely wants back is
    the one they were just in."""
    if _absent():
        return []
    sql = f"{_SELECT} ORDER BY last_active_at DESC, created_at DESC"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with _connect() as conn:
        return [_row(r) for r in conn.execute(sql)]


def resolve(ref: str) -> SessionInfo:
    """`ref` is "last", a full id, or a unique prefix of one -- the git
    convention, so a person can type what the list shows. Raises LookupError
    saying which of the three failed; the CLI prints that message as is."""
    ref = ref.strip()
    if not ref:
        raise LookupError("no session given")
    if ref == "last":
        newest = list_sessions(limit=1)
        if not newest:
            raise LookupError("no saved sessions yet")
        return newest[0]
    rows: list[SessionInfo] = []
    if not _absent():
        with _connect() as conn:
            rows = [_row(r) for r in conn.execute(f"{_SELECT} WHERE id LIKE ?", (ref + "%",))]
    exact = [r for r in rows if r.id == ref]
    if exact:
        return exact[0]
    if not rows:
        raise LookupError(f"no session matches {ref!r}")
    if len(rows) > 1:
        shown = ", ".join(r.short_id for r in rows[:5])
        raise LookupError(f"{ref!r} is ambiguous: {shown}")
    return rows[0]


def describe_age(iso: str, now: datetime | None = None) -> str:
    """'just now', '5m ago', '3h ago', '2d ago' -- what a list wants next to
    each row, coarse on purpose: the question is which session, not when."""
    then = datetime.fromisoformat(iso)
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    seconds = max(0, int((now - then).total_seconds()))
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


# --------------------------------------------------------------------------
# Moving a session between machines
# --------------------------------------------------------------------------

#: The file format's version, so a later change to it can still read this.
EXPORT_VERSION = 1


def export_filename(info: SessionInfo, today: date | None = None) -> str:
    """`otto-session-<id>-<date>.json` -- the shape agent/cli/lessons.py's
    export uses, with the id so two exports do not collide. A name, never a
    path: `otto serve` hands it to a phone that decides where it goes."""
    return f"otto-session-{info.short_id}-{(today or date.today()):%Y-%m-%d}.json"


def export_session(session_id: str, path: Path | str) -> Path:
    """Write `export_payload(session_id)` to `path` as JSON, and return the
    path."""
    payload = export_payload(session_id)
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


def export_payload(session_id: str, *, include_workspace: bool = True) -> dict:
    """One session as plain data -- its index row and everything in its
    memory file that a restore reads: the live tiers, the current bullets,
    the retired chunks they cite. `include_workspace=False` leaves out the
    directory it worked in, which is a path on this computer and means
    nothing to whoever receives it over a socket.

    Vectors are not exported: they belong to whichever embedding model made
    them (agent/memory/store.py's `embedding_model`), which the machine
    importing this may not have. An imported chunk starts with no vector,
    which recall already treats as "not rankable", so the worst case is
    that `recall_memory` returns the most recent items rather than the best
    ones until they are re-embedded. Superseded bullets are left behind as
    well: they are an audit trail, not memory.
    """
    info = get(session_id)
    if info is None:
        raise LookupError(f"no session {session_id[:8]!r} to export")
    store = MemoryStore(session_db_path(session_id))
    try:
        chunks = store.get_chunk_rows("history", store.chunk_hashes("history"))
        bullets = store.current_bullets("history")
        pending = store.pending("history")
    finally:
        store.close()
    return {
        "version": EXPORT_VERSION,
        "session": {
            "id": info.id, "title": info.title,
            "workspace": info.workspace if include_workspace else None,
            "created_at": info.created_at, "last_active_at": info.last_active_at,
            "turns": info.turns,
        },
        "chunks": [{"hash": c.hash, "content": c.content} for c in chunks],
        "bullets": [{"generation": b.generation, "text": b.text, "hash_refs": b.hash_refs}
                    for b in bullets],
        "pending": [{"tier": tier, "text": text} for tier, text in pending],
    }


def import_session(path: Path | str) -> SessionInfo:
    """Read a file `export_session` wrote into a session of this machine's
    own and return it. Keeps the exported id when nothing here has it, so
    a session moved once and moved back is the same session; mints a new
    one otherwise, so importing never overwrites what is here -- and also
    when the exported id is not a uuid hex id, since a file can say anything
    and the id becomes a file name. Raises ValueError for a file that cannot
    be read or is not an export.
    """
    path = Path(path).expanduser()
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from exc
    return import_payload(payload, source=str(path))


def _checked_export(payload: object, source: str) -> dict:
    """`payload` once it has the shape `export_payload` writes, all of it,
    before anything is written: an import that failed half way would leave a
    memory file no row names. ValueError saying what is wrong otherwise."""
    if not isinstance(payload, dict) or not isinstance(payload.get("session"), dict) \
            or not isinstance(payload.get("pending"), list):
        raise ValueError(f"{source} is not an otto session export")
    version = payload.get("version", 1)
    if not isinstance(version, int) or isinstance(version, bool):
        raise ValueError(f"{source} is not an otto session export")
    if version > EXPORT_VERSION:
        raise ValueError(f"{source} was written by a newer otto (format {version})")

    def rows(key: str, fields: dict[str, type]) -> None:
        items = payload.get(key, [])
        if not isinstance(items, list) or not all(
                isinstance(item, dict) and all(isinstance(item.get(f), t) for f, t in fields.items())
                for item in items):
            raise ValueError(f"{source} has a malformed {key} list")

    rows("chunks", {"hash": str, "content": str})
    rows("bullets", {"text": str, "hash_refs": list})
    rows("pending", {"text": str})
    if any(item.get("tier", "x") not in ("x", "y") for item in payload["pending"]):
        raise ValueError(f"{source} has a malformed pending list")
    return payload


def import_payload(payload: object, *, keep_workspace: bool = True, source: str = "the import") -> SessionInfo:
    """`import_session` for data already in hand -- what `otto serve` receives.
    `keep_workspace=False` drops the directory the export names: a path from
    another machine, and one a client could otherwise choose, which a later
    resume would open as this session's workspace."""
    payload = _checked_export(payload, source)
    meta = payload["session"]
    session_id = str(meta.get("id") or "")
    if not valid_id(session_id) or get(session_id) is not None or session_db_path(session_id).exists():
        session_id = uuid.uuid4().hex
    store = MemoryStore(session_db_path(session_id))
    try:
        for chunk in payload.get("chunks", []):
            store.add_chunk("history", chunk["hash"], chunk["content"], None)
        for bullet in payload.get("bullets", []):
            try:
                generation = int(bullet.get("generation", 1))
            except (TypeError, ValueError):
                generation = 1
            store.add_bullet("history", generation, bullet["text"],
                             [str(ref) for ref in bullet.get("hash_refs", [])], None)
        for item in payload["pending"]:
            store.add_pending("history", item["text"], item.get("tier", "x"))
    finally:
        store.close()
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, workspace, created_at, last_active_at, turns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, str(meta.get("title") or "")[:200],
             (str(meta["workspace"]) if keep_workspace and meta.get("workspace") else None),
             str(meta.get("created_at") or now), now, _count(meta.get("turns"))),
        )
    return get(session_id)
