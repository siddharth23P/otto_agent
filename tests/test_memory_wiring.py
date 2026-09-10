"""Coverage for agent/memory/wiring.py -- the glue between a live otto
chat/tui Session and the standalone agent/memory/ engine. `ROUTER`/`_call`
(imported from agent.pipeline.nodes) are monkeypatched throughout so
nothing here needs a live model, a network connection, or an
INCEPTION_API_KEY -- same offline-test bar as every other agent/memory/
test file.
"""
from langchain_core.messages import AIMessage, HumanMessage

import agent.memory.store as store_module
import agent.memory.wiring as wiring
from agent.memory.queue import NewBullet, TieredQueue
from agent.memory.store import MemoryStore


def test_summarize_for_memory_calls_call_with_a_system_and_human_message(monkeypatch):
    captured = {}

    class _FakeLLM:
        pass

    fake_llm = _FakeLLM()

    class _FakeRouter:
        def chat_model(self, task, *, temperature):
            captured["task"] = task
            captured["temperature"] = temperature
            return fake_llm

    def _fake_call(llm, messages):
        captured["llm"] = llm
        captured["messages"] = messages
        return "the summary"

    monkeypatch.setattr(wiring, "ROUTER", _FakeRouter())
    monkeypatch.setattr(wiring, "_call", _fake_call)

    result = wiring.summarize_for_memory("summarize this")

    assert result == "the summary"
    assert captured["llm"] is fake_llm
    assert captured["temperature"] == 0.2
    assert [type(m).__name__ for m in captured["messages"]] == ["SystemMessage", "HumanMessage"]
    assert captured["messages"][1].content == "summarize this"


def test_new_history_queue_builds_a_history_kind_queue_backed_by_the_session(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "DB_DIR", tmp_path / "otto-memory")
    monkeypatch.setattr(wiring, "ROUTER", object())  # never called -- queue starts empty
    monkeypatch.setattr(wiring, "_call", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")))

    queue = wiring.new_history_queue("session-abc")

    assert isinstance(queue, TieredQueue)
    assert queue.kind == "history"
    assert queue.store.path == tmp_path / "otto-memory" / "session-abc.db"
    queue.store.close()


def test_record_turn_appends_human_and_ai_with_matching_prefixes(tmp_path):
    store = MemoryStore(tmp_path / "session.db")
    try:
        queue = TieredQueue("history", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)

        wiring.record_turn(queue, HumanMessage("hello there"), AIMessage("hi, how can I help?"))

        assert queue.recent_items == ["you: hello there", "otto: hi, how can I help?"]
    finally:
        store.close()


def test_record_turn_skips_the_ai_item_when_there_is_no_output(tmp_path):
    store = MemoryStore(tmp_path / "session.db")
    try:
        queue = TieredQueue("history", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)

        wiring.record_turn(queue, HumanMessage("hello"), None)

        assert queue.recent_items == ["you: hello"]
    finally:
        store.close()


def test_to_message_round_trips_a_human_item():
    message = wiring._to_message("you: what's N Queens?")

    assert isinstance(message, HumanMessage)
    assert message.content == "what's N Queens?"


def test_to_message_round_trips_an_ai_item():
    message = wiring._to_message("otto: it's a classic backtracking problem.")

    assert isinstance(message, AIMessage)
    assert message.content == "it's a classic backtracking problem."


def test_to_message_falls_back_to_ai_for_an_unprefixed_item():
    # Defensive only -- record_turn() (this module's one writer) never
    # produces an item without one of the two known prefixes.
    message = wiring._to_message("no prefix at all")

    assert isinstance(message, AIMessage)
    assert message.content == "no prefix at all"


def test_history_for_graph_reconstructs_recent_items_as_real_messages(tmp_path):
    store = MemoryStore(tmp_path / "session.db")
    try:
        queue = TieredQueue("history", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)
        wiring.record_turn(queue, HumanMessage("hi"), AIMessage("hello!"))
        wiring.record_turn(queue, HumanMessage("solve N queens"), AIMessage("here you go"))

        messages, memory_context = wiring.history_for_graph(queue)

        assert [type(m).__name__ for m in messages] == ["HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
        assert [m.content for m in messages] == ["hi", "hello!", "solve N queens", "here you go"]
        assert memory_context == ""
    finally:
        store.close()


def test_history_for_graph_puts_earlier_compacted_material_into_memory_context(tmp_path):
    store = MemoryStore(tmp_path / "session.db")
    try:
        queue = TieredQueue("history", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)
        queue._y_bullets = [NewBullet(text="earlier stuff happened", hash_refs=["h1"])]
        queue._x = ["you: recent message"]

        messages, memory_context = wiring.history_for_graph(queue)

        assert [m.content for m in messages] == ["recent message"]
        assert "EARLIER CONVERSATION (compacted" in memory_context
        assert "earlier stuff happened" in memory_context
    finally:
        store.close()


def test_history_for_graph_with_an_empty_queue_returns_empty_messages_and_context(tmp_path):
    store = MemoryStore(tmp_path / "session.db")
    try:
        queue = TieredQueue("history", store, summarize=lambda p: "", x_budget=10_000, y_budget=10_000)

        messages, memory_context = wiring.history_for_graph(queue)

        assert messages == []
        assert memory_context == ""
    finally:
        store.close()
