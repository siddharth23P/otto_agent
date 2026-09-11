"""Glue between the standalone agent/memory/ engine (TieredQueue, MemoryStore)
and a live otto chat/tui session -- Phase 2 of claude/otto-tiered-memory-
design.md (Phase 1, the engine itself, is agent/memory/{tokens,hashing,
store,embeddings,queue,retrieval}.py, all of which stay free of any
dependency on agent.pipeline/agent.router -- that package's own __init__.py
docstring promise). This file is deliberately the one place in agent/memory/
that crosses that boundary, since bridging the two is its whole job; nothing
else in the package should ever import agent.pipeline or agent.router.

Kept out of agent/cli/shell.py too, even though shell.py's `Session` is this
module's only real caller: shell.py is meant to stay a thin CLI-layer file
(its own module docstring: "Session carries no d/llm/config/agents... there
is nothing left to size or fix"), and the reconstruction logic here (turning
a TieredQueue's items back into real LangChain messages, wiring a `summarize`
callback through the SAME Router/Task.SUMMARIZE path agent/pipeline/nodes.py's
own summarizer role uses) belongs with the memory engine's own concerns, not
the REPL's.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from agent.memory.queue import TieredQueue
from agent.memory.store import MemoryStore
from agent.pipeline.nodes import ROUTER, _call
from agent.router.mapping import Task

#: What agent/pipeline/nodes.py's own `_conversation_so_far()` uses to label
#: each prior message ("you"/"otto") -- matched here exactly so a queue item
#: written by `record_turn()` below round-trips back through `_to_message()`
#: into the same HumanMessage/AIMessage split it started as.
_HUMAN_PREFIX = "you: "
_AI_PREFIX = "otto: "

#: The one instruction handed to Task.SUMMARIZE ahead of a TieredQueue
#: compaction prompt (agent/memory/queue.py's `_build_summarize_prompt()`
#: builds everything AFTER this) -- kept short and generic on purpose: the
#: numbered-items/citation format is already fully spelled out by the
#: compaction prompt itself, this just frames what kind of content it is.
_SUMMARIZE_SYSTEM_PROMPT = (
    "You compress a session's conversation history so it keeps fitting in "
    "a model's context window without losing what it said. Follow the "
    "numbering and citation instructions in the message exactly."
)


def summarize_for_memory(prompt: str) -> str:
    """The `summarize: str -> str` callback a live TieredQueue needs
    (agent/memory/queue.py's TieredQueue.__init__) -- routed through
    Task.SUMMARIZE (the same model/route agent/pipeline/nodes.py's own
    summarizer role uses) via the same module-level ROUTER/._call every
    other graph call goes through, so a compaction counts against the
    same provider health/telemetry as everything else this session does.
    """
    llm = ROUTER.chat_model(Task.SUMMARIZE)
    return _call(llm, [SystemMessage(_SUMMARIZE_SYSTEM_PROMPT), HumanMessage(prompt)])


def new_history_queue(session_id: str) -> TieredQueue:
    """A fresh `kind="history"` TieredQueue backed by this session's own
    SQLite store (agent/memory/store.py's `~/.otto/memory/<session_id>.db`
    convention) -- one per otto chat/tui `Session` (agent/cli/shell.py),
    rebuilt on `/new`/"new session" exactly like the session id itself is.
    """
    store = MemoryStore.for_session(session_id)
    return TieredQueue(kind="history", store=store, summarize=summarize_for_memory)


def _speaker_prefix(message: BaseMessage) -> str:
    return _HUMAN_PREFIX if isinstance(message, HumanMessage) else _AI_PREFIX


def _content_text(message: BaseMessage) -> str:
    # Every message this module ever constructs (record_turn, below) is a
    # plain HumanMessage(str)/AIMessage(str) -- chat.py/tui.py never hand a
    # multi-block content list the way a raw provider response occasionally
    # does (agent/pipeline/nodes.py's own _content_text handles THAT case,
    # for LLM replies) -- so a plain str() is exactly right here, no need
    # to import or duplicate that block-flattening logic.
    return str(message.content)


def record_turn(queue: TieredQueue, human: BaseMessage, ai: BaseMessage | None) -> None:
    """Append one finished turn to `queue` -- the human side always, the AI
    side only if the turn actually produced something (a run that errored
    out or produced no final output has nothing worth remembering as
    "otto said"). Each message becomes its OWN queue item, not one glued-
    together exchange, so a later compaction's per-item citation lines up
    with one line = one turn's one message (agent/memory/queue.py's
    citation-based compaction reads more naturally that way, and a bullet
    citing "item 4" unambiguously means one specific message, not a pair).
    """
    queue.append(f"{_speaker_prefix(human)}{_content_text(human)}")
    if ai is not None:
        queue.append(f"{_speaker_prefix(ai)}{_content_text(ai)}")


def _to_message(item: str) -> BaseMessage:
    """The inverse of record_turn()'s formatting -- only ever applied to
    `queue.recent_items` (agent/memory/queue.py's X tier), which holds
    exactly what record_turn() put there, verbatim, nothing compaction-
    rewritten (compacted/summarized items never leave Y, so they never
    reach this function -- see history_for_graph() below for where THEY
    go instead). The trailing `else AIMessage` is defensive, not a real
    case this module's own writer (record_turn) can ever produce.
    """
    if item.startswith(_HUMAN_PREFIX):
        return HumanMessage(item[len(_HUMAN_PREFIX):])
    if item.startswith(_AI_PREFIX):
        return AIMessage(item[len(_AI_PREFIX):])
    return AIMessage(item)


def history_for_graph(queue: TieredQueue) -> tuple[list[BaseMessage], str]:
    """(bounded prior messages, compacted-memory context block) for
    agent/pipeline/run.py's `history`/`memory_context` parameters --
    replaces handing the graph an ever-growing raw list.

    X's own recent items become real Human/AIMessage objects (via
    `_to_message()`) -- agent/pipeline/nodes.py's `_conversation_so_far()`
    keeps reading `state["messages"]` exactly as it always has, seeing real
    typed messages for however many turns still fit verbatim, unchanged by
    any of this. Y's bullets and not-yet-summarized raw overflow (older
    than X) become `memory_context` instead: `_initial()` seeds
    `state["context"]` with it, which every prompt-builder in nodes.py
    already displays via its existing "CONTEXT GATHERED SO FAR:" section --
    no new prompt plumbing needed on that side at all.
    """
    messages = [_to_message(item) for item in queue.recent_items]
    earlier = queue.earlier_view()
    memory_context = f"EARLIER CONVERSATION (compacted -- older than what's shown above):\n{earlier}" if earlier else ""
    return messages, memory_context
