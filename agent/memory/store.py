"""SQLite-backed content-addressable store for agent/memory/queue.py's
compacted-away memory -- permanent leaf `chunks` (raw text, hash-keyed,
never rewritten) plus the CURRENT generation's `bullets` for each `kind`
("history" or "context" -- TieredQueue's "unified" design: one schema, two
independent namespaces, per the 2026-09-10 design call).

One SQLite file per session (~/.otto/memory/<session_id>.db) -- next to
the `~/.otto` convention agent/cli/output.py's own docstring already
establishes for internal Otto state (the ledger/profile), not
otto_output/, which is meant to be opened by a person. Session-owned:
summarizer/finder's own Python code is meant to read and write this
directly, not through LangGraph state channels -- every external turn's
own graph thread is disposable by design (agent/pipeline/run.py's
`_graph_thread_id`), so anything meant to survive past one turn has to
live somewhere that isn't thrown away with it.

`bullets.superseded` is how an old compaction generation's bullets get
retired without deleting them (kept for an audit trail -- cheap on disk,
and useful for understanding how a session's memory evolved) once a later
generation supersedes them: `current_bullets()` only ever returns the
live, non-superseded set for a `kind`, which is what a prompt or a
recall_memory search should ever see -- an older, now-redundant bullet
would just be noise stacked on top of the newer one that already covers
the same ground (and then some).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

#: Sibling to the ledger/profile convention agent/cli/output.py's own
#: docstring documents for `~/.otto` -- internal state, not a deliverable.
DB_DIR = Path.home() / ".otto" / "memory"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    hash TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    -- 0,1,2,... in the order this `kind` first flushed them, so a chunk's
    -- NEIGHBOURS are recoverable (chunks_near, below). A dialogue turn is a
    -- poor retrieval unit on its own: the answer to "when did she join the
    -- group?" is routinely in the turn AFTER the one that names it, so
    -- retrieval.py returns each hit with the turns either side of it.
    seq INTEGER,
    -- Which model produced `embedding`; see _CHUNK_MIGRATIONS above.
    embedding_model TEXT,
    -- The chunk's own embedding, written at flush time (agent/memory/
    -- queue.py) so retrieval.py can rank the raw text directly instead of
    -- only ranking the bullets that summarize it. NULL is a valid state,
    -- exactly as it is for a bullet -- the local model was unavailable
    -- when this chunk was flushed (agent/memory/embeddings.py).
    embedding BLOB
);
CREATE TABLE IF NOT EXISTS bullets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    generation INTEGER NOT NULL,
    text TEXT NOT NULL,
    hash_refs TEXT NOT NULL,
    embedding BLOB,
    embedding_model TEXT,
    superseded INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bullets_kind_superseded ON bullets(kind, superseded);
CREATE INDEX IF NOT EXISTS idx_chunks_kind_seq ON chunks(kind, seq);
"""

#: Columns added to `chunks` after the first release of this schema. SQLite's
#: CREATE TABLE IF NOT EXISTS leaves an existing table alone, so a session DB
#: written before they existed needs them added explicitly -- one ALTER each,
#: skipped when already present. Existing rows keep NULL for both, which both
#: readers already treat as "unknown", so an old DB degrades to the old
#: behaviour rather than erroring.
_CHUNK_MIGRATIONS = (
    ("seq", "INTEGER"),
    ("embedding", "BLOB"),
    ("embedding_model", "TEXT"),
)

#: `bullets` carries an embedding too, so it needs the same stamp -- and had no
#: migration of its own until now.
_BULLET_MIGRATIONS = (("embedding_model", "TEXT"),)

#: Which model produced a vector. Without it, two embedding spaces can be
#: compared and the storage layer cannot tell: `_from_blob` infers length from
#: the byte count, so a 384-dim and a 1536-dim vector are both just bytes.
#:
#: Mismatched dimensions at least raise -- and then get swallowed by the bare
#: `except Exception` in agent/pipeline/tools.py, leaving recall quietly dead
#: for the rest of the session. The worse case is a hosted model emitting the
#: SAME dimension (Gemini can be asked for 384, OpenAI's can be truncated):
#: nothing raises, the scores look plausible, and two unrelated spaces are
#: ranked against each other. A dimension check would not catch that. The model
#: name is the only thing that does.


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_blob(embedding: "np.ndarray | None") -> bytes | None:
    return embedding.astype(np.float32).tobytes() if embedding is not None else None


def _stamp(embedding: "np.ndarray | None", model: str | None) -> str | None:
    """Which model to record for this vector.

    Defaults to whatever the process is embedding with right now, so a caller
    that simply passes a vector cannot forget to say where it came from -- the
    failure that would reintroduce is silent, not loud.
    """
    if embedding is None:
        return None
    if model:
        return model
    from agent.memory.embeddings import current_model_name

    return current_model_name()


def _from_blob(blob: bytes | None) -> "np.ndarray | None":
    return np.frombuffer(blob, dtype=np.float32) if blob is not None else None


@dataclass(frozen=True)
class Bullet:
    id: int
    kind: str
    generation: int
    text: str
    #: Hashes into `chunks` -- always resolves to REAL, permanently stored
    #: raw content, no matter how many compaction generations produced
    #: this bullet (agent/memory/queue.py's citation-based carry-forward).
    hash_refs: list[str]
    #: None when this bullet was written without a usable embedding (the
    #: local model was unavailable at the time -- agent/memory/
    #: embeddings.py's EmbeddingUnavailable) -- excluded from a semantic
    #: search's ranking, not an error.
    embedding: np.ndarray | None
    #: Which model produced `embedding`; see Chunk.embedding_model.
    embedding_model: str | None = None


@dataclass(frozen=True)
class Chunk:
    """One permanently-stored raw item, with the position and embedding
    retrieval.py needs to rank it and to find what sat either side of it."""
    seq: int | None
    hash: str
    content: str
    embedding: np.ndarray | None
    #: Which model produced `embedding`. None for a vector written before this
    #: column existed -- treated as "unknown space", i.e. not comparable.
    embedding_model: str | None = None


class MemoryStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._migrate_chunks()
        self._conn.commit()

    def _migrate_chunks(self) -> None:
        for table, migrations in (("chunks", _CHUNK_MIGRATIONS), ("bullets", _BULLET_MIGRATIONS)):
            present = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for column, sql_type in migrations:
                if column not in present:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")

    @classmethod
    def for_session(cls, session_id: str) -> "MemoryStore":
        return cls(DB_DIR / f"{session_id}.db")

    def close(self) -> None:
        self._conn.close()

    # ---- chunks (permanent, content-addressed raw text) ----------------

    def add_chunk(
        self, kind: str, hash_: str, content: str, embedding: np.ndarray | None = None,
        embedding_model: str | None = None,
    ) -> None:
        """INSERT OR IGNORE -- a hash that's already stored is left exactly
        as it was (dedup, not overwrite; see hashing.py's own docstring), which
        also means it keeps the `seq` it was first given: a turn repeated
        verbatim much later stays one chunk, sitting where it first appeared.
        """
        self._conn.execute(
            "INSERT OR IGNORE INTO chunks "
            "(hash, kind, content, created_at, seq, embedding, embedding_model) "
            "VALUES (?, ?, ?, ?, (SELECT COALESCE(MAX(seq), -1) + 1 FROM chunks WHERE kind = ?), ?, ?)",
            (hash_, kind, content, _now(), kind, _to_blob(embedding),
             _stamp(embedding, embedding_model)),
        )
        self._conn.commit()

    def stale_vector_count(self, model: str) -> int:
        """How many stored vectors were made by a different model.

        Zero means nothing needs re-embedding. Anything else means recall is
        silently degraded to unranked-most-recent for those rows, because
        agent/memory/retrieval.py refuses to rank across embedding spaces.
        """
        rows = self._conn.execute(
            "SELECT (SELECT COUNT(*) FROM chunks "
            "        WHERE embedding IS NOT NULL AND IFNULL(embedding_model,'') != ?) "
            "     + (SELECT COUNT(*) FROM bullets "
            "        WHERE embedding IS NOT NULL AND IFNULL(embedding_model,'') != ?)",
            (model, model),
        ).fetchone()
        return int(rows[0] or 0)

    def reembed(self, embed_texts, model: str, *, batch: int = 64) -> int:
        """Re-embed every stored vector that a different model produced, and
        return how many were rewritten.

        This exists because `add_chunk` is INSERT OR IGNORE keyed on a hash of
        the TEXT alone: re-running against an existing store never overwrites,
        so a stale vector would otherwise sit there forever, unrankable and
        unmentioned.

        `embed_texts` is a `list[str] -> list[np.ndarray]` callable -- the
        caller's, not this module's, so the store keeps its promise of
        depending on nothing that needs a network or a key.
        """
        rewritten = 0
        for table, key in (("chunks", "hash"), ("bullets", "id")):
            column = "content" if table == "chunks" else "text"
            rows = self._conn.execute(
                f"SELECT {key}, {column} FROM {table} "
                f"WHERE embedding IS NOT NULL AND IFNULL(embedding_model,'') != ?",
                (model,),
            ).fetchall()
            for start in range(0, len(rows), batch):
                window = rows[start:start + batch]
                vectors = embed_texts([r[1] for r in window])
                for (identifier, _), vector in zip(window, vectors):
                    self._conn.execute(
                        f"UPDATE {table} SET embedding = ?, embedding_model = ? WHERE {key} = ?",
                        (_to_blob(vector), model, identifier),
                    )
                    rewritten += 1
            self._conn.commit()
        return rewritten

    def get_chunks(self, hashes: list[str]) -> dict[str, str]:
        if not hashes:
            return {}
        placeholders = ",".join("?" for _ in hashes)
        rows = self._conn.execute(
            f"SELECT hash, content FROM chunks WHERE hash IN ({placeholders})", hashes,
        ).fetchall()
        return dict(rows)

    def get_chunk_rows(self, kind: str, hashes: list[str]) -> list[Chunk]:
        """The full `Chunk` records for `hashes`, oldest first -- what
        retrieval.py ranks over once a bullet match has narrowed the store
        down to a candidate set ("first pulling all the hash and compiling
        the list then finding relevant information (not all)" -- the spec).
        """
        if not hashes:
            return []
        placeholders = ",".join("?" for _ in hashes)
        rows = self._conn.execute(
            f"SELECT seq, hash, content, embedding, embedding_model FROM chunks "
            f"WHERE kind = ? AND hash IN ({placeholders}) ORDER BY seq",
            [kind, *hashes],
        ).fetchall()
        return [Chunk(seq=r[0], hash=r[1], content=r[2], embedding=_from_blob(r[3]),
                      embedding_model=r[4]) for r in rows]

    def chunks_near(self, kind: str, seqs: list[int], window: int) -> list[Chunk]:
        """Every chunk of `kind` within `window` positions of any of `seqs`,
        oldest first, the hits themselves included and deduplicated -- the
        turns either side of a match, which is where a dialogue's answer
        very often actually sits (see the `seq` column's own note above).
        """
        wanted = {s + d for s in seqs for d in range(-window, window + 1) if s + d >= 0}
        if not wanted:
            return []
        placeholders = ",".join("?" for _ in wanted)
        rows = self._conn.execute(
            f"SELECT seq, hash, content, embedding, embedding_model FROM chunks "
            f"WHERE kind = ? AND seq IN ({placeholders}) ORDER BY seq",
            [kind, *sorted(wanted)],
        ).fetchall()
        return [Chunk(seq=r[0], hash=r[1], content=r[2], embedding=_from_blob(r[3]),
                      embedding_model=r[4]) for r in rows]

    # ---- bullets (the current, compacted-down summary of everything) ---

    def add_bullet(
        self, kind: str, generation: int, text: str,
        hash_refs: list[str], embedding: np.ndarray | None,
        embedding_model: str | None = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO bullets "
            "(kind, generation, text, hash_refs, embedding, embedding_model, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, generation, text, json.dumps(hash_refs), _to_blob(embedding),
             _stamp(embedding, embedding_model), _now()),
        )
        self._conn.commit()
        return cur.lastrowid

    def supersede_bullets(self, kind: str, before_generation: int) -> None:
        """Retire every bullet of `kind` from a generation strictly earlier
        than `before_generation` -- called once a fresh generation's own
        bullets have all been written, so the two never overlap mid-write.
        """
        self._conn.execute(
            "UPDATE bullets SET superseded = 1 WHERE kind = ? AND generation < ? AND superseded = 0",
            (kind, before_generation),
        )
        self._conn.commit()

    def current_bullets(self, kind: str) -> list[Bullet]:
        rows = self._conn.execute(
            "SELECT id, kind, generation, text, hash_refs, embedding, embedding_model FROM bullets "
            "WHERE kind = ? AND superseded = 0 ORDER BY id",
            (kind,),
        ).fetchall()
        return [
            Bullet(
                id=r[0], kind=r[1], generation=r[2], text=r[3],
                hash_refs=json.loads(r[4]),
                embedding=_from_blob(r[5]),
                embedding_model=r[6],
            )
            for r in rows
        ]
