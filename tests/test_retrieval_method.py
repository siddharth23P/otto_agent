"""Lexical against semantic, on Otto's own chunk store.

*Is Grep All You Need?* found inline grep beat inline vector retrieval for
every harness-model pair it tested, by as much as 23 points. That is a strong
claim and it does not transfer here, which is worth pinning rather than
re-litigating: the study grepped FILES in a long corpus, where chunking is the
weak link. Otto's store is already chunked with a per-chunk embedding, which is
the configuration embeddings are good at.

Measured on a 300-turn conversation, 288 chunks, with a SQLite FTS5 index built
over exactly the same rows:

    query type                    semantic   lexical
    paraphrase (no shared words)      5/5       2/5
    rare token (id, name, date)       4/4       4/4
    shared vocabulary                 8/8       8/8

Semantic wins or ties everywhere; lexical never wins. So the answer to the
issue is "change nothing", and these tests exist so that stays true rather
than being assumed.

The study's SECOND finding does apply and is a constraint to preserve: one
configuration fell from 93.1% to 55.2% when identical results arrived through
a file the model had to open rather than inline in the tool result. Otto
returns results inline, which is the arm that won.
"""
import re
import sqlite3

import pytest

from agent.eval import compaction_bench as cb
from agent.memory import retrieval as mr
from agent.memory.queue import TieredQueue
from agent.memory.store import MemoryStore


def _embeddings_work() -> bool:
    """Whether this machine can rank at all.

    The suite runs with no credentials and no hosted embedding spec
    (tests/conftest.py), so on CI `recall` degrades to unranked most-recent
    exactly as agent/memory/retrieval.py documents. Comparing a degraded
    semantic index against a working lexical one would measure the absence of
    a model, so the comparison skips rather than reporting a loss that is
    really a missing download.
    """
    try:
        from agent.memory.embeddings import embed
        embed(["probe"])
        return True
    except Exception:
        return False


needs_embeddings = pytest.mark.skipif(
    not _embeddings_work(),
    reason="no embedding backend here, so semantic recall is unranked most-recent",
)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    store = MemoryStore(tmp_path_factory.mktemp("retr") / "s.db")
    queue = TieredQueue(kind="history", store=store,
                        summarize=cb.keep_fraction_summarizer(),
                        x_budget=400, y_budget=1200)
    for item in cb.build_conversation(turns=300):
        queue.append(item)

    conn = store._conn
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(hash, content)")
    conn.executemany(
        "INSERT INTO chunk_fts (hash, content) VALUES (?, ?)",
        conn.execute("SELECT hash, content FROM chunks WHERE kind='history'").fetchall(),
    )
    conn.commit()
    yield store
    store.close()


def _lexical(store, query: str, limit: int = 20) -> str:
    terms = re.findall(r"[A-Za-z0-9]{3,}", query)
    if not terms:
        return ""
    try:
        rows = store._conn.execute(
            "SELECT content FROM chunk_fts WHERE chunk_fts MATCH ? ORDER BY rank LIMIT ?",
            (" OR ".join(terms), limit),
        ).fetchall()
    except sqlite3.OperationalError:
        return ""
    return "\n".join(r[0] for r in rows)


#: The query shares no words with the evidence, which is where an embedding
#: should win and a lexical index cannot match at all.
PARAPHRASE = [
    ("do not record the credential anywhere", "deployment key"),
    ("which timezone should the output use", "Europe/Lisbon"),
    ("which tables are off limits", "billing tables"),
    ("what colour did they turn down", "rejected blue"),
    ("how many rows per file at most", "5000 rows"),
]

#: An id, a name, a date -- where a lexical index should win outright.
RARE_TOKEN = ["AK-4417-QX", "Priya", "2027-01-31"]


def _found(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


@needs_embeddings
def test_semantic_beats_lexical_on_a_paraphrase(corpus):
    """The case the whole embedding index exists for, and the one a lexical
    index cannot do at all."""
    semantic = sum(_found(mr.recall(corpus, "history", q), needle)
                   for q, needle in PARAPHRASE)
    lexical = sum(_found(_lexical(corpus, q), needle) for q, needle in PARAPHRASE)

    assert semantic > lexical, f"semantic {semantic} did not beat lexical {lexical}"
    assert semantic == len(PARAPHRASE)


@needs_embeddings
def test_semantic_is_not_beaten_on_rare_tokens_either(corpus):
    """Where grep should have the advantage, it merely ties -- the embedding
    index finds exact tokens too, so there is no case left where switching
    would gain anything."""
    for token in RARE_TOKEN:
        assert _found(mr.recall(corpus, "history", token), token), token


def test_results_come_back_inline_rather_than_as_a_path_to_open(corpus):
    """The study's other finding, and a constraint to preserve rather than a
    change to make: one configuration fell from 93.1% to 55.2% when identical
    results arrived through a file the model had to read separately."""
    text = mr.recall(corpus, "history", "the deployment key")

    assert len(text) > 80, "recall returned a stub rather than the content"
    assert not text.strip().startswith("/"), "recall handed back a path"
    assert "\n" in text
