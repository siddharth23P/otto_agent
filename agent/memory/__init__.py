"""Otto's tiered short-term-memory engine -- 2026-09-10 design call.

Keeps an unbounded stream of text (conversation history, or an in-turn
task's own gathered context/board -- see agent/memory/queue.py's module
docstring for the "unified" reasoning) inside a bounded, budgeted slice of
whichever model is actually reading it, while staying fully recoverable
rather than truncated away for good.

Six small modules make up the engine itself, each independently testable
without a live LLM, a network connection, or a compiled LangGraph graph:

    tokens.py      -- approximate token counting, for budgeting.
    hashing.py     -- content-addressable hashing for the store.
    store.py       -- the SQLite-backed permanent chunk/bullet store.
    embeddings.py  -- a local, offline embedding model for semantic recall.
    queue.py       -- TieredQueue: the X/Y buffer + compaction mechanics.
    retrieval.py   -- recall(): what agent/pipeline/tools.py's
                      `recall_memory` tool actually calls.

Those six are deliberately decoupled from agent.pipeline/agent.router:
TieredQueue takes a `summarize` callback rather than importing ROUTER
itself, so nothing there creates a circular import or needs a live model to
be unit-tested. Two more modules are the Phase 2 wiring, and ARE allowed to
cross that boundary, since bridging it is their whole job:

    session.py  -- a contextvars-based binding of "the current run's
                   MemoryStore", so agent/pipeline/tools.py's recall_memory
                   (a plain function of one string) can find it. Imports
                   only agent.memory.store -- no cycle with tools.py, which
                   is itself imported BY agent/pipeline/nodes.py.
    wiring.py   -- everything else: building a live TieredQueue's
                   `summarize` callback from ROUTER/Task.SUMMARIZE, and
                   converting between a TieredQueue's plain-text items and
                   real LangChain messages for agent/cli/shell.py's
                   `Session` and agent/pipeline/run.py.

See claude/otto-tiered-memory-design.md in the project for the full design.
As of Phase 2: cross-turn conversation history IS wired in (Session, via
wiring.py/session.py); in-turn context/board growth within a single graph
run is NOT yet -- it needs a place to persist a live TieredQueue across
several LangGraph node calls, which the wired half didn't have to solve.
"""
