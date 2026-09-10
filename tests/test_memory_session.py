"""Coverage for agent/memory/session.py's bind_store()/current_store() --
the contextvars-based binding agent/pipeline/tools.py's recall_memory tool
reads, since a plain TOOL_DISPATCH function (one string in, one ToolResult
out) has no other way to see which session's MemoryStore a run is using.
"""
import pytest

from agent.memory.session import bind_store, current_store
from agent.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    yield s
    s.close()


def test_current_store_is_none_with_nothing_bound():
    assert current_store() is None


def test_bind_store_makes_current_store_return_it(store):
    with bind_store(store):
        assert current_store() is store
    assert current_store() is None


def test_bind_store_restores_the_previous_binding_on_exit(tmp_path):
    outer = MemoryStore(tmp_path / "outer.db")
    inner = MemoryStore(tmp_path / "inner.db")
    try:
        with bind_store(outer):
            assert current_store() is outer
            with bind_store(inner):
                assert current_store() is inner
            assert current_store() is outer
        assert current_store() is None
    finally:
        outer.close()
        inner.close()


def test_bind_store_restores_even_if_the_block_raises(store):
    with pytest.raises(ValueError):
        with bind_store(store):
            raise ValueError("boom")
    assert current_store() is None


def test_bind_store_of_none_is_a_valid_explicit_unbind(store):
    with bind_store(store):
        with bind_store(None):
            assert current_store() is None
        assert current_store() is store
