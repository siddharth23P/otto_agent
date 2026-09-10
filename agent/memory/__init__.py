"""Otto's tiered short-term-memory engine -- 2026-09-10 design call.

Keeps an unbounded stream of text (conversation history, or an in-turn
task's own gathered context/board -- see agent/memory/queue.py's module
docstring for the "unified" reasoning) inside a bounded, budgeted slice of
whichever model is actually reading it, while staying fully recoverable
rather than truncated away for good.

Five small modules, each independently testable without a live LLM, a
network connection, or a compiled LangGraph graph:

    tokens.py      -- approximate token counting, for budgeting.
    hashing.py     -- content-addressable hashing for the store.
    store.py       -- the SQLite-backed permanent chunk/bullet store.
    embeddings.py  -- a local, offline embedding model for semantic recall.
    queue.py       -- TieredQueue: the X/Y buffer + compaction mechanics.
    retrieval.py   -- recall(): what agent/pipeline/tools.py's
                      `recall_memory` tool (not yet wired up -- see the
                      design doc) actually calls.

This package is deliberately decoupled from agent.pipeline/agent.router:
TieredQueue takes a `summarize` callback rather than importing ROUTER
itself, so nothing here creates a circular import or needs a live model to
be unit-tested. Wiring it INTO the graph (agent/pipeline/nodes.py,
agent/cli/chat.py, agent/cli/tui.py) is tracked separately -- see
claude/otto-tiered-memory-design.md in the project for the full design,
including what's built here (the engine) vs. what's still pending
(graph/CLI integration).
"""
