"""Langfuse is probed once and silenced when unreachable, and a run's
contextvar binders survive being unwound from a different context.

Both came from the same live symptom: a REPL flooded with "Transient error
HTTPConnectionPool(host='localhost', port=3000)" from a collector that was
not running, and a TUI that printed "ValueError: <Token ...> was created in
a different Context" after being quit while a turn was still streaming.
"""
from __future__ import annotations

import contextvars
import logging
import socket

import pytest

from agent.pipeline import tracing
from agent.pipeline.budget import Budget, bind_budget, current_budget
from agent.pipeline.execution import bind_command_runner, current_command_runner
from agent.pipeline.run import _config
from agent.pipeline.toolkit import bind_extra_tools, current_extra_tools
from agent.pipeline.usage import UsageLedger, bind_usage, current_usage
from agent.pipeline.workspace import bind_workspace, current_workspace
from agent.memory.session import bind_store, current_store


@pytest.fixture(autouse=True)
def _fresh_decision(monkeypatch):
    tracing.reset_for_tests()
    monkeypatch.delenv("LANGFUSE_TRACING_ENABLED", raising=False)
    yield
    tracing.reset_for_tests()


def _keys(monkeypatch, url="http://127.0.0.1:1"):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", url)


# --------------------------------------------------------------------------
# the probe
# --------------------------------------------------------------------------

def test_no_keys_means_no_tracing_and_no_probe(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

    def never(*a, **k):
        raise AssertionError("no keys, so nothing should be probed")

    monkeypatch.setattr(socket, "create_connection", never)
    assert tracing.langfuse_ready() is False
    assert tracing.callback_handler() is None
    with tracing.observe_run(session_id="s", name="n", input="i", tags=[]) as span:
        assert isinstance(span, tracing.NullSpan)
        assert span.trace_id is None


def test_unreachable_host_disables_tracing_with_one_warning(monkeypatch, caplog):
    _keys(monkeypatch)

    def refused(*a, **k):
        raise ConnectionRefusedError

    monkeypatch.setattr(socket, "create_connection", refused)
    with caplog.at_level(logging.WARNING, logger="agent.pipeline.tracing"):
        assert tracing.langfuse_ready() is False
        assert tracing.langfuse_ready() is False
        assert tracing.callback_handler() is None
    warnings = [r for r in caplog.records if "unreachable" in r.getMessage()]
    assert len(warnings) == 1, "decided once, warned once"
    assert "127.0.0.1:1" in warnings[0].getMessage()
    # Everything else in the process that builds a client later is quiet too.
    import os
    assert os.environ["LANGFUSE_TRACING_ENABLED"] == "false"


def test_explicit_disable_is_honoured_without_probing(monkeypatch):
    _keys(monkeypatch)
    monkeypatch.setenv("LANGFUSE_TRACING_ENABLED", "false")

    def never(*a, **k):
        raise AssertionError("explicitly off, so nothing should be probed")

    monkeypatch.setattr(socket, "create_connection", never)
    assert tracing.langfuse_ready() is False


def test_reachable_host_traces_through_the_sdk(monkeypatch):
    _keys(monkeypatch, url="https://langfuse.example.test")
    seen = {}

    class _Sock:
        def close(self):
            seen["closed"] = True

    def ok(addr, timeout):
        seen["addr"] = addr
        seen["timeout"] = timeout
        return _Sock()

    monkeypatch.setattr(socket, "create_connection", ok)
    assert tracing.langfuse_ready() is True
    assert seen["addr"] == ("langfuse.example.test", 443)
    assert seen["timeout"] == tracing.PROBE_TIMEOUT_S
    assert seen["closed"] is True

    # The SDK is only imported and used once the host answered.
    import langfuse

    class _Span:
        trace_id = "t-1"

        def update(self, **kw):
            seen["update"] = kw

        def score_trace(self, **kw):
            pass

    class _Client:
        def start_as_current_observation(self, **kw):
            seen["observation"] = kw
            from contextlib import contextmanager

            @contextmanager
            def cm():
                yield _Span()

            return cm()

    from contextlib import contextmanager

    @contextmanager
    def _propagate(**kw):
        seen["propagate"] = kw
        yield

    monkeypatch.setattr(langfuse, "get_client", lambda: _Client())
    monkeypatch.setattr(langfuse, "propagate_attributes", _propagate)
    with tracing.observe_run(session_id="s1", name="otto:pipeline", input="hi", tags=["pipeline"]) as span:
        assert span.trace_id == "t-1"
    assert seen["propagate"]["session_id"] == "s1"
    assert seen["observation"]["name"] == "otto:pipeline"


def test_config_without_a_handler_registers_no_callbacks():
    assert _config("thread", None)["callbacks"] == []
    handler = object()
    assert _config("thread", handler)["callbacks"] == [handler]


def test_probe_defaults_ports_by_scheme():
    calls = []

    class _Sock:
        def close(self):
            pass

    def ok(addr, timeout):
        calls.append(addr)
        return _Sock()

    import agent.pipeline.tracing as t
    original = socket.create_connection
    socket.create_connection = ok
    try:
        assert t._probe("http://localhost:3000") is True
        assert t._probe("https://cloud.langfuse.com") is True
        assert t._probe("http://plain.test") is True
    finally:
        socket.create_connection = original
    assert calls == [("localhost", 3000), ("cloud.langfuse.com", 443), ("plain.test", 80)]


# --------------------------------------------------------------------------
# unwinding from another context
# --------------------------------------------------------------------------

def _exit_elsewhere(cm) -> None:
    """Enter `cm` in one child context and exit it in another: what happens
    to a generator's `with` blocks when a thread other than the one that
    created them finalises it. The token cannot be reset there; the binder
    must not raise. Neither child touches this test's own context."""
    contextvars.copy_context().run(cm.__enter__)
    contextvars.copy_context().run(cm.__exit__, None, None, None)


def test_every_run_binder_survives_exit_from_another_context(tmp_path):
    budget = Budget(max_model_calls=3)
    ledger = UsageLedger()
    runner = object()
    tools = {"t": object()}

    _exit_elsewhere(bind_budget(budget))
    _exit_elsewhere(bind_usage(ledger))
    _exit_elsewhere(bind_store(None))
    _exit_elsewhere(bind_workspace(tmp_path))
    _exit_elsewhere(bind_command_runner(runner))
    _exit_elsewhere(bind_extra_tools(tools))

    # No binder raised on the foreign exit, and nothing leaked into the
    # context the test itself runs in.
    assert current_budget() is None
    assert current_usage() is None
    assert current_store() is None
    assert current_workspace() is None
    assert current_command_runner() is None
    assert current_extra_tools() == {}


def test_the_sdk_signature_is_recognised_and_nothing_else():
    assert tracing.closing_in_foreign_context(
        ValueError("<Token var=<ContextVar name='x' default=None at 0x1> at 0x2> was created in a different Context")
    )
    assert not tracing.closing_in_foreign_context(ValueError("bad value"))
    assert not tracing.closing_in_foreign_context(RuntimeError("different Context"))
