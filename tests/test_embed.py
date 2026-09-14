"""agent/embed.py: the surface an embedding host depends on.

Offline throughout: the pipeline is replaced by fake generators that yield
the same shapes run.py does, so what is tested is the contract -- bindings
on the calling thread, the ask/answer hand-off, cancellation, the events."""
from __future__ import annotations

import os
import threading

import pytest

from agent import embed
from agent.pipeline import run as pipeline
from agent.pipeline.progress import check_cancelled
from agent.pipeline.toolkit import ExtraTool, dispatch_table, render_note
from agent.pipeline.tools import ToolResult, reachable_tools


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    for name in embed.KEY_VARS:
        monkeypatch.setenv(name, "test-placeholder-not-a-real-key")
    # Naming a source scrubs ambient keys, so the placeholders are supplied
    # through it, as a host's keystore would.
    embed.configure(tmp_path / "home", env_file=tmp_path / "keys.env",
                    environ={name: "test-placeholder-not-a-real-key" for name in embed.KEY_VARS})
    return tmp_path


# --------------------------------------------------------------------------
# configure and keys
# --------------------------------------------------------------------------

def test_configure_sets_the_variables_and_creates_the_home(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    root = embed.configure(tmp_path / "state", env_file=tmp_path / "k.env")
    assert root.is_dir()
    assert os.environ["OTTO_HOME"] == str(root)
    assert os.environ["OTTO_ENV_FILE"] == str((tmp_path / "k.env").resolve())


def test_configure_is_idempotent_for_the_same_home_and_refuses_another(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    embed.configure(tmp_path / "a")
    embed.configure(tmp_path / "a")
    with pytest.raises(RuntimeError):
        embed.configure(tmp_path / "b")


def test_set_key_writes_the_env_file_and_only_ever_returns_the_masked_form(configured, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    shown = embed.set_key("OPENAI_API_KEY", "sk-verysecret9876")
    assert shown == "********9876"
    text = (configured / "keys.env").read_text()
    assert "OPENAI_API_KEY=" in text
    assert embed.key_status()["OPENAI_API_KEY"] == "********9876"
    assert "verysecret" not in str(embed.key_status())


def test_environ_mode_never_writes_a_file(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embed.configure(tmp_path / "home", environ={"OPENAI_API_KEY": "sk-fromkeystore1234"})
    assert os.environ["OPENAI_API_KEY"] == "sk-fromkeystore1234"
    assert embed.set_key("OPENAI_API_KEY", "sk-rotated5678") == "********5678"
    assert os.environ["OPENAI_API_KEY"] == "sk-rotated5678"
    assert embed.set_key("OPENAI_API_KEY", "") == "not set"
    assert "OPENAI_API_KEY" not in os.environ
    assert not list((tmp_path / "home").glob("*.env"))


def test_set_key_rejects_a_bad_name(configured):
    with pytest.raises(ValueError):
        embed.set_key("not a name", "x")


def test_key_vars_match_the_provider_classes():
    from agent.router.llm_provider import builtin_provider_names, provider_class

    assert set(embed.KEY_VARS) == {provider_class(n).env_var for n in builtin_provider_names()}


def test_version_and_ready(configured):
    info = embed.version()
    assert info["api"] == embed.API_VERSION and isinstance(info["api"], int)
    assert info["python"].count(".") == 2
    assert embed.ready()  # conftest's placeholder Inception key


# --------------------------------------------------------------------------
# a turn, driven by fakes with run.py's shapes
# --------------------------------------------------------------------------

def _phone_screen():
    seen = []
    return ExtraTool(name="phone_screen", description="Read the screen.",
                     call=lambda body: (seen.append(body), ToolResult("[1] ok", "", 0))[1],
                     mutates=False), seen


def test_a_turn_binds_tools_on_the_calling_thread_and_pauses_for_an_answer(configured, monkeypatch):
    tool, seen = _phone_screen()
    observed = {}

    def fake_run(text, **kwargs):
        # The bindings must be visible HERE, inside the stream, on this thread.
        observed["dispatch"] = "phone_screen" in dispatch_table()
        observed["note"] = render_note()
        observed["bash_reachable"] = "execute_bash" in reachable_tools()
        dispatch_table()["phone_screen"]("{}")
        yield {"agent": {"board": ["solve: phone_screen ok"], "output": None}}
        yield {"__ask__": {"question": "Which one?", "choices": ["a", "b"], "thread_id": "t1"}}

    def fake_resume(answer, **kwargs):
        observed["answer"] = answer
        yield {"__final__": {"final_output": "picked " + answer}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    monkeypatch.setattr(pipeline, "resume_pipeline_stream", fake_resume)

    runtime = embed.Runtime()
    handle = runtime.open_session()
    events = []

    def on_event(event):
        events.append(event)
        if event["type"] == "ask":
            # Answer from another thread, as a UI would.
            threading.Thread(target=handle.answer, args=(event["thread_id"], "b")).start()

    handle.run("find milk", events=on_event, tools=[tool], guidance="Look first.",
               disabled_tools={"execute_bash"})

    assert observed == {"dispatch": True, "note": observed["note"], "bash_reachable": False,
                        "answer": "b"}
    assert "phone_screen" in observed["note"] and "Look first." in observed["note"]
    assert seen == ["{}"]
    assert [e["type"] for e in events] == ["board", "ask", "final"]
    assert events[0]["lines"] == ["solve: phone_screen ok"]
    assert events[2]["text"] == "picked b"
    assert "total_tokens" in events[2]["usage"] or isinstance(events[2]["usage"], dict)
    assert handle.turns == 1
    assert any(row["id"] == handle.id for row in runtime.list_sessions())
    handle.close()


def test_answer_with_the_wrong_thread_is_refused(configured):
    handle = embed.Runtime().open_session()
    assert handle.answer("nope", "x") is False
    handle.close()


def test_cancel_releases_a_pending_question(configured, monkeypatch):
    def fake_run(text, **kwargs):
        yield {"__ask__": {"question": "?", "choices": [], "thread_id": "t"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    handle = embed.Runtime().open_session()
    events = []

    def on_event(event):
        events.append(event)
        if event["type"] == "ask":
            threading.Thread(target=handle.cancel).start()

    handle.run("q", events=on_event)
    assert events[-1] == {"type": "error", "code": "cancelled", "message": "stopped"}
    assert handle.turns == 0
    handle.close()


def test_cancel_stops_a_run_at_the_next_model_call(configured, monkeypatch):
    def fake_run(text, **kwargs):
        yield {"agent": {"board": ["thinking"]}}
        check_cancelled()  # what _call does before spending
        yield {"__final__": {"final_output": "should not arrive"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    handle = embed.Runtime().open_session()
    events = []

    def on_event(event):
        events.append(event)
        if event["type"] == "board":
            handle.cancel()

    handle.run("q", events=on_event)
    assert events[-1]["code"] == "cancelled"
    handle.close()


def test_a_provider_failure_is_an_event_not_an_exception(configured, monkeypatch):
    from agent.router.llm_provider.base import AuthError

    def fake_run(text, **kwargs):
        raise AuthError("Otto requires Inception")
        yield  # pragma: no cover

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("q", events=events.append)
    assert events == [{"type": "error", "code": "provider", "message": "Otto requires Inception"}]
    handle.close()


def test_one_turn_at_a_time(configured, monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_run(text, **kwargs):
        started.set()
        release.wait(5)
        yield {"__final__": {"final_output": "ok"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    handle = embed.Runtime().open_session()
    worker = threading.Thread(target=handle.run, args=("a",), kwargs={"events": lambda e: None})
    worker.start()
    started.wait(5)
    with pytest.raises(RuntimeError):
        handle.run("b", events=lambda e: None)
    release.set()
    worker.join(5)
    assert handle.turns == 1
    handle.close()


def test_progress_is_forwarded_as_plain_data(configured, monkeypatch):
    from agent.pipeline.progress import report

    def fake_run(text, **kwargs):
        report("tool", "phone_act", detail={"target": "phone_act:{}"})
        yield {"__final__": {"final_output": "ok"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("q", events=events.append)
    progress = [e for e in events if e["type"] == "progress"]
    assert progress and progress[0]["kind"] == "tool" and progress[0]["text"] == "phone_act"
    assert progress[0]["detail"] == {"target": "phone_act:{}"}
    handle.close()


def test_sessions_can_be_listed_resumed_and_deleted(configured, monkeypatch):
    def fake_run(text, **kwargs):
        yield {"__final__": {"final_output": "hello back"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    runtime = embed.Runtime()
    first = runtime.open_session()
    first.run("hello", events=lambda e: None)
    sid = first.id
    first.close()

    rows = runtime.list_sessions()
    assert rows[0]["id"] == sid and rows[0]["turns"] == 1 and rows[0]["title"]
    transcript = runtime.transcript("last")
    assert transcript["id"] == sid
    assert [m["role"] for m in transcript["messages"]] == ["you", "otto"]
    resumed = runtime.open_session(sid[:8])
    assert resumed.id == sid and resumed.turns == 1
    resumed.close()
    assert runtime.delete_session(sid) is True
    assert not runtime.list_sessions()


def test_configure_keeps_its_env_file_on_a_repeat_call(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    embed.configure(tmp_path / "home", env_file=tmp_path / "keys.env")
    embed.configure(tmp_path / "home")  # the docstring's safe repeat
    embed.set_key("OPENAI_API_KEY", "sk-persisted1234")
    assert "OPENAI_API_KEY=" in (tmp_path / "keys.env").read_text()


def test_a_named_key_source_wins_over_the_ambient_environment(tmp_path, monkeypatch):
    """A key the shell exported must not stand in for one the host did not
    provide, and the env file's value beats the shell's for the same name."""
    monkeypatch.setattr(embed, "_configured", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient-0000")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ambient-1111")
    (tmp_path / "keys.env").write_text("OPENAI_API_KEY=sk-fromfile-2222\n")
    embed.configure(tmp_path / "home", env_file=tmp_path / "keys.env")
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert os.environ["OPENAI_API_KEY"] == "sk-fromfile-2222"
    assert embed.key_status()["ANTHROPIC_API_KEY"] == "not set"


def test_configure_without_a_source_keeps_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ambient-0000")
    embed.configure(tmp_path / "home")
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ambient-0000"
