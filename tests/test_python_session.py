"""One persistent Python interpreter per run -- agent/pipeline/python_session.py
and the branch of execute_python that uses it (issue #1).

Every execute_python call used to be a standalone script in a fresh process.
An agent that loaded a dataset or built an index in one call had to rebuild
it in the next, from the transcript or through disk: work compounded as text,
never as live state. Prime Agent (arXiv:2608.23552) calls the persistent
interpreter the layer that makes long-horizon work compositional at all.

What a persistent process gives up is the property the fresh-process model
had for free: nothing survived a call. So most of this file is about the
things that used to be wiped by process exit and now have to be handled on
purpose -- isolation between runs, a hung call that SIGINT cannot stop, a
snippet that forges the framing, a memory ceiling, and the session being torn
down with the run rather than outliving it. Written before the module
existed, so the three design forks (what a delegated subagent gets, what the
container path does, what a forced restart tells the model) were settled
here first and the code made to match.

Offline throughout: the interpreter is a real child process, but nothing
here reaches a provider or the network.
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

import agent.pipeline.tools as pt
from agent.pipeline.execution import bind_command_runner
from agent.pipeline.python_session import (
    LocalPythonSession,
    PYTHON_SESSION_ENV,
    RESET_NOTE,
    SessionResult,
    SessionUnavailable,
    UNAVAILABLE_NOTE,
    bind_python_session,
    current_python_session,
    default_python_session,
    fresh_python_session,
    python_session,
    python_session_note,
)
from agent.pipeline.workspace import bind_workspace

#: How long a call that is meant to finish instantly may take. Generous,
#: because CI runners are slow and the first call in a session spawns a
#: process; a hang is measured in the tens of seconds anyway.
FAST_S = 30.0


@pytest.fixture
def session():
    with python_session() as live:
        yield live


# --------------------------------------------------------------------------
# 1-3: the feature, and what a failing call does to it
# --------------------------------------------------------------------------

def test_state_persists_from_one_call_to_the_next(session):
    """The whole point: build something once, use it in the next call."""
    first = pt.execute_python("import json\ndata = {'n': 41}\nprint('built')")
    assert first.ok, first.stderr
    second = pt.execute_python("print(json.dumps({'n': data['n'] + 1}))")
    assert second.ok, second.stderr
    assert second.stdout.strip() == '{"n": 42}'


def test_a_function_defined_earlier_is_callable_later(session):
    pt.execute_python("def double(x):\n    return 2 * x")
    assert pt.execute_python("print(double(21))").stdout.strip() == "42"


def test_nothing_bound_is_still_a_fresh_process_per_call():
    """The regression guard for every existing caller: with no session bound,
    execute_python is byte-for-byte the old behaviour."""
    assert current_python_session() is None
    assert pt.execute_python("x = 1").ok
    later = pt.execute_python("print(x)")
    assert not later.ok
    assert "NameError" in later.stderr


def test_an_exception_fails_that_call_only_and_the_session_survives(session):
    pt.execute_python("keep = 'still here'")
    broken = pt.execute_python("raise ValueError('boom')")
    assert broken.returncode == 1
    assert "ValueError: boom" in broken.stderr
    assert not broken.timed_out
    assert RESET_NOTE not in broken.stderr
    after = pt.execute_python("print(keep)")
    assert after.ok, after.stderr
    assert after.stdout.strip() == "still here"


def test_the_traceback_names_the_snippet_not_the_shim(session):
    """The model reads the traceback to fix its code. Frames from the
    interpreter's own plumbing are noise it cannot act on."""
    broken = pt.execute_python("def f():\n    return 1 / 0\nf()")
    assert "ZeroDivisionError" in broken.stderr
    assert "shim" not in broken.stderr.lower()
    assert 'File "<session>", line 2, in f' in broken.stderr


def test_a_syntax_error_is_a_failed_call_not_a_dead_session(session):
    broken = pt.execute_python("def (:\n")
    assert broken.returncode == 1
    assert "SyntaxError" in broken.stderr
    assert pt.execute_python("print('fine')").stdout.strip() == "fine"


def test_sys_exit_reports_the_code_like_a_script_would(session):
    """Parity with the fresh-process path, where `sys.exit(3)` was exit 3."""
    result = pt.execute_python("import sys\nsys.exit(3)")
    assert result.returncode == 3
    assert not result.timed_out
    assert pt.execute_python("print('alive')").stdout.strip() == "alive"


def test_input_does_not_hang_the_session(session):
    """A snippet that calls input() with nobody at the keyboard used to get
    EOF instantly. It must not now block on a pipe forever."""
    started = time.monotonic()
    result = pt.execute_python("name = input('who? ')")
    assert time.monotonic() - started < FAST_S
    assert "EOFError" in result.stderr


# --------------------------------------------------------------------------
# 4-7: isolation -- the central risk
# --------------------------------------------------------------------------

def test_two_concurrent_runs_never_see_each_others_variables():
    """Two threads, each with its own session, each defining the same name.
    A module-level singleton would fail this."""
    seen: dict[str, object] = {}
    go = threading.Barrier(2)

    def run(label: str) -> None:
        with python_session():
            pt.execute_python(f"who = {label!r}")
            go.wait(timeout=FAST_S)
            seen[label] = pt.execute_python("print(who)").stdout.strip()
            seen[f"{label}-other"] = pt.execute_python("print(other)")

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=FAST_S * 2)
    assert seen["a"] == "a" and seen["b"] == "b"
    for label in ("a", "b"):
        assert "NameError" in seen[f"{label}-other"].stderr


def test_the_binding_is_gone_once_the_block_exits():
    """Mirrors test_workspace_session.py's "restores previous binding on exit"."""
    outer = LocalPythonSession()
    try:
        with bind_python_session(outer):
            assert current_python_session() is outer
            with python_session() as inner:
                assert current_python_session() is inner
            assert current_python_session() is outer
        assert current_python_session() is None
    finally:
        outer.close()


def test_a_delegated_subagent_gets_its_own_session_never_the_parents():
    """The design fork the issue asked to settle before any code existed.

    A delegate runs with none of the parent's conversation (nodes.py's
    DELEGATE_CONTRACT); a child that could reach the parent's live objects
    would be reaching state it cannot know exists, and a child that poisoned
    the interpreter would corrupt the parent's remaining calls after it
    returned. So: its own session, torn down when it returns. Explicit, not
    an accident of contextvar inheritance.
    """
    with python_session() as parent:
        pt.execute_python("secret = 'parent'")
        with fresh_python_session() as child:
            assert child is not None and child is not parent
            assert current_python_session() is child
            leaked = pt.execute_python("print(secret)")
            assert "NameError" in leaked.stderr
            pt.execute_python("secret = 'child'")
        assert current_python_session() is parent
        assert not child.alive
        assert pt.execute_python("print(secret)").stdout.strip() == "parent"


def test_fresh_python_session_binds_nothing_when_nothing_is_bound():
    """A run without a session (the container path, or the feature switched
    off) delegates the same way it always did."""
    with fresh_python_session() as child:
        assert child is None
        assert current_python_session() is None


def test_delegate_runs_the_child_in_a_different_session(monkeypatch):
    """The wiring in nodes.py, not just the helper: _delegate opens a fresh
    session around the child's loop."""
    from agent.pipeline import nodes as pn

    inside: list = []

    def fake_loop(state, messages, **kwargs):
        inside.append(current_python_session())
        return "child answer", "final", "find"

    monkeypatch.setattr(pn, "_agent_loop", fake_loop)
    with python_session() as parent:
        result = pn._delegate({}, "find\nlook something up", actions=[], parent_mode="solve")
    assert result.ok, result.stderr
    assert inside and inside[0] is not None and inside[0] is not parent
    assert not inside[0].alive


def test_a_poisoned_builtin_lasts_one_call_not_the_run(session):
    """The shim needs the builtins to read the next request at all: a
    snippet that rebound `len` broke `json.loads`, and every call after it
    timed out with no output. So the builtins are put back after each call.
    Module-level state a snippet changes (its own globals, `os.system`) is
    the model's to manage; the plumbing's is not."""
    same_call = pt.execute_python(
        "import builtins\nbuiltins.len = lambda x: -1\nprint(len('abc'))"
    )
    assert same_call.stdout.strip() == "-1"
    next_call = pt.execute_python("print(len('abc'))")
    assert next_call.ok, next_call.stderr
    assert next_call.stdout.strip() == "3"
    # A name ADDED to builtins goes the same way.
    pt.execute_python("import builtins\nbuiltins.helper = 1")
    assert "NameError" in pt.execute_python("print(helper)").stderr


def test_back_to_back_runs_do_not_leak_a_poisoned_builtin(monkeypatch):
    """Two golden-eval tasks in sequence, the way agent/eval/runner.py runs
    them: what task A does to its interpreter never reaches task B. Each
    run_pipeline() call binds and closes its own session."""
    from agent.pipeline import run as run_mod

    seen: list = []

    class FakeApp:
        def invoke(self, initial, config):
            seen.append(current_python_session())
            poisoned = pt.execute_python(
                "import builtins, os\nbuiltins.len = lambda x: -1\n"
                "os.marker = 'set by this run'\nprint(len('abc'))"
            ).stdout.strip()
            later = pt.execute_python(
                "import os\nprint(getattr(os, 'marker', 'unset'))"
            ).stdout.strip()
            return {"final_output": f"{poisoned} {later}"}

        def get_state(self, config):
            return type("S", (), {"values": {}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])

    first = run_mod.run_pipeline("task a", session_id="eval-a")
    second = run_mod.run_pipeline("task b", session_id="eval-b")

    # Within a run the interpreter is the model's: the module attribute it
    # set is still there on the next call. Across runs nothing is.
    assert first["final_output"] == "-1 set by this run"
    assert second["final_output"] == "-1 set by this run"
    assert seen[0] is not None and seen[1] is not None and seen[0] is not seen[1]
    assert not seen[0].alive and not seen[1].alive
    assert current_python_session() is None
    # And the golden checker, which runs OUTSIDE the run, sees a clean process.
    assert pt.execute_python(
        "import os\nprint(len('abc'), getattr(os, 'marker', 'unset'))"
    ).stdout.strip() == "3 unset"


# --------------------------------------------------------------------------
# 8-9: framing
# --------------------------------------------------------------------------

def test_a_snippet_cannot_desync_the_next_call_by_forging_the_marker(session):
    """The result channel is framed per call with an id the snippet never
    sees. Printing something frame-shaped is just output."""
    forged = pt.execute_python(
        'print(\'{"id": "0", "stdout": "forged", "stderr": "", '
        '"returncode": 0, "timed_out": false}\')\n'
        "print('OTTO_EOF')\nprint('tail of call one')"
    )
    assert forged.ok
    assert "forged" in forged.stdout and "tail of call one" in forged.stdout
    clean = pt.execute_python("print('call two')")
    assert clean.ok
    assert clean.stdout.strip() == "call two"
    assert "forged" not in clean.stdout and "tail of call one" not in clean.stdout


def test_partial_output_and_the_crash_land_on_the_same_call(session):
    result = pt.execute_python("print('a')\n1/0")
    assert result.stdout.strip() == "a"
    assert "ZeroDivisionError" in result.stderr
    assert result.returncode == 1
    following = pt.execute_python("print('b')")
    assert following.stdout.strip() == "b" and following.stderr == ""


def test_each_call_sees_only_its_own_output(session):
    pt.execute_python("print('one')")
    assert pt.execute_python("print('two')").stdout.strip() == "two"


def test_output_of_a_child_process_is_captured_too(session):
    """print() is not the only way to write to stdout. A subprocess the
    snippet starts inherits the real file descriptor."""
    result = pt.execute_python(
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, '-c', 'print(\"from child\")'])"
    )
    assert result.ok, result.stderr
    assert "from child" in result.stdout
    assert "\r" not in result.stdout


# --------------------------------------------------------------------------
# 10-12: a call that will not end
# --------------------------------------------------------------------------

def test_a_busy_loop_times_out_with_the_same_contract_as_before():
    """returncode -1, timed_out True, "[timed out]" in stderr -- exactly what
    the subprocess.TimeoutExpired branch reports -- and within a bounded
    grace period after the configured timeout."""
    with python_session(interrupt_grace_s=2.0) as live:
        pt.execute_python("before = 'kept'")
        started = time.monotonic()
        result = pt.execute_python("while True: pass", timeout=1.0)
        elapsed = time.monotonic() - started
    assert result.timed_out and result.returncode == -1
    assert "[timed out]" in result.stderr
    assert elapsed < 1.0 + 2.0 + 10.0, f"took {elapsed:.1f}s"


@pytest.mark.skipif(os.name == "nt", reason="an interrupt on Windows is best-effort; "
                    "the hard-kill path below is what is guaranteed there")
def test_an_interruptible_hang_keeps_the_session_and_its_state():
    """When the interrupt lands, only that call is stopped. The model keeps
    everything it built before, and is not told otherwise."""
    with python_session(interrupt_grace_s=5.0):
        pt.execute_python("before = 'kept'")
        result = pt.execute_python("import time\nwhile True: time.sleep(0.01)", timeout=1.0)
        assert result.timed_out
        assert RESET_NOTE not in result.stderr
        after = pt.execute_python("print(before)")
    assert after.ok, after.stderr
    assert after.stdout.strip() == "kept"


def test_a_hang_the_interrupt_cannot_stop_is_hard_killed_and_replaced():
    """A snippet that ignores the interrupt (the stand-in for a C call that
    never checks for one). The session is killed, and the NEXT call on the
    same handle succeeds against a fresh interpreter instead of hanging."""
    with python_session(interrupt_grace_s=1.0) as live:
        pt.execute_python("before = 'gone'")
        pid = live.pid
        started = time.monotonic()
        result = pt.execute_python(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            + ("signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n" if os.name == "nt" else "")
            + "while True: time.sleep(0.01)",
            timeout=1.0,
        )
        elapsed = time.monotonic() - started
        assert result.timed_out and result.returncode == -1
        assert elapsed < FAST_S
        after = pt.execute_python("print('fresh')")
        assert after.ok, after.stderr
        assert after.stdout.strip() == "fresh"
        assert live.pid != pid
        assert "NameError" in pt.execute_python("print(before)").stderr


def test_a_forced_restart_is_said_in_the_tool_result():
    """The model must not assume its variables survived. Mirrors the
    "[timed out]" annotation: the fact lands in stderr where it reads it."""
    with python_session(interrupt_grace_s=1.0):
        result = pt.execute_python(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            + ("signal.signal(signal.SIGBREAK, signal.SIG_IGN)\n" if os.name == "nt" else "")
            + "while True: time.sleep(0.01)",
            timeout=1.0,
        )
    assert "[timed out]" in result.stderr
    assert RESET_NOTE in result.stderr


def test_a_snippet_that_kills_the_interpreter_is_reported_as_a_reset(session):
    """os._exit, a segfault, an OOM kill: the child is simply gone. That is a
    failed call that says the session restarted, not a hang and not a crash
    of the run."""
    pt.execute_python("x = 1")
    result = pt.execute_python("import os\nos._exit(7)")
    assert not result.ok
    assert RESET_NOTE in result.stderr
    after = pt.execute_python("print('back')")
    assert after.ok and after.stdout.strip() == "back"
    assert "NameError" in pt.execute_python("print(x)").stderr


# --------------------------------------------------------------------------
# 13-14: ceilings
# --------------------------------------------------------------------------

@pytest.mark.skipif(os.name != "posix" or sys.platform == "darwin",
                    reason="RLIMIT_AS is enforced on Linux; macOS ignores it "
                           "and Windows has no equivalent here")
def test_the_memory_ceiling_is_a_clean_failure_not_an_oom_kill():
    limit = 512 * 1024 * 1024
    with python_session(memory_limit_bytes=limit):
        pt.execute_python("keep = 'still here'")
        result = pt.execute_python(f"blob = bytearray({limit * 2})")
        assert result.returncode == 1
        assert "MemoryError" in result.stderr
        assert RESET_NOTE not in result.stderr
        assert pt.execute_python("print(keep)").stdout.strip() == "still here"


def test_a_max_calls_ceiling_evicts_and_recreates_the_session():
    with python_session(max_calls=2) as live:
        pt.execute_python("x = 1")
        assert pt.execute_python("print(x)").stdout.strip() == "1"
        first_pid = live.pid
        third = pt.execute_python("print(x)")
    assert "NameError" in third.stderr
    assert RESET_NOTE in third.stderr
    assert live.pid != first_pid or not live.alive


def test_an_idle_timeout_evicts_and_recreates_the_session():
    with python_session(idle_timeout_s=0.3):
        pt.execute_python("x = 1")
        time.sleep(0.6)
        result = pt.execute_python("print(x)")
    assert "NameError" in result.stderr
    assert RESET_NOTE in result.stderr


# --------------------------------------------------------------------------
# 15-16: lifecycle
# --------------------------------------------------------------------------

def test_the_session_is_torn_down_with_its_block():
    with python_session() as live:
        pt.execute_python("x = 1")
        pid = live.pid
        private = live.private_dir
        assert live.alive and pid is not None
        assert private is not None and private.is_dir()
    assert not live.alive
    assert not private.exists()
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_a_session_starts_no_process_until_it_is_used():
    """A greeting costs two model calls and no execute_python. It must not
    also cost an interpreter."""
    with python_session() as live:
        assert not live.alive and live.pid is None


def test_a_throwaway_session_gets_a_private_directory_deleted_with_it(tmp_path):
    """No workspace bound: the snippet's cwd is private to this session,
    never shared with another, and gone when the session is."""
    dirs = []
    for _ in range(2):
        with python_session() as live:
            cwd = pt.execute_python("import os\nprint(os.getcwd())").stdout.strip()
            dirs.append(cwd)
            assert os.path.isdir(cwd)
            pt.execute_python("open('scratch.txt', 'w').write('x')")
        assert not os.path.exists(cwd)
    assert dirs[0] != dirs[1]


def test_a_session_runs_in_the_bound_workspace(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    return 41 + 1\n")
    with bind_workspace(tmp_path), python_session():
        cwd = pt.execute_python("import os\nprint(os.getcwd())").stdout.strip()
        assert os.path.realpath(cwd) == os.path.realpath(str(tmp_path))
        assert pt.execute_python("import m\nprint(m.f())").stdout.strip() == "42"
        pt.execute_python("print('hello')")
    # `__pycache__` is Python importing m, as it always was; nothing of the
    # session itself (no shim, no capture file, no snippet) may be here.
    left = sorted(p.name for p in tmp_path.iterdir() if p.name != "__pycache__")
    assert left == ["m.py"], f"the session left something in the workspace: {left}"


def test_closing_twice_is_harmless():
    live = LocalPythonSession()
    with bind_python_session(live):
        pt.execute_python("x = 1")
    live.close()
    live.close()
    assert not live.alive


# --------------------------------------------------------------------------
# 17: the container path
# --------------------------------------------------------------------------

@pytest.mark.skipif(os.name == "nt", reason="exercises the container path by "
                    "running POSIX commands on the host shell")
def test_a_bound_command_runner_wins_and_the_result_says_state_will_not_persist(tmp_path):
    """The decision: no persistent interpreter inside a container yet. A
    session bound alongside a runner is NOT used -- the snippet must land in
    the container, not on the host -- and the fallback is observable rather
    than silent."""
    import subprocess

    commands: list[str] = []

    def runner(command, timeout):
        commands.append(command)
        proc = subprocess.run(command, shell=True, cwd=tmp_path,
                              capture_output=True, text=True, timeout=timeout)
        return proc.stdout, proc.stderr, proc.returncode

    with bind_command_runner(runner), python_session() as live:
        first = pt.execute_python("x = 1")
        second = pt.execute_python("print(x)")
        assert not live.alive
    assert len(commands) == 2 and "python3 -" in commands[0]
    assert first.ok
    assert "NameError" in second.stderr
    assert "does not persist" in first.stderr


def test_no_session_is_offered_for_a_run_inside_a_container():
    """run.py asks default_python_session() what to bind. With a runner bound
    the answer is nothing, so the prompt claims no persistence it cannot
    deliver."""
    with bind_command_runner(lambda command, timeout: ("", "", 0)):
        assert default_python_session() is None
        assert python_session_note() == ""


# --------------------------------------------------------------------------
# Wiring: run.py binds one, the switch, and what the model is told
# --------------------------------------------------------------------------

def test_run_pipeline_stream_binds_a_session_for_the_graph(monkeypatch):
    """Same shape as test_workspace_session.py's binding test: a contextvar
    set on the CLI's thread is invisible to a generator consumed on a worker
    thread, so the binding has to happen where the graph runs."""
    from agent.pipeline import run as run_mod

    seen: list = []

    class FakeApp:
        def stream(self, *args, **kwargs):
            seen.append(current_python_session())
            return iter(())

        def get_state(self, config):
            return type("S", (), {"values": {"final_output": "done"}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])
    monkeypatch.delenv(PYTHON_SESSION_ENV, raising=False)

    list(run_mod.run_pipeline_stream("hi", session_id="s1"))

    assert len(seen) == 1 and isinstance(seen[0], LocalPythonSession)
    assert not seen[0].alive       # never used, never started
    assert current_python_session() is None


def test_the_env_var_switches_the_feature_off(monkeypatch):
    """The Claw-Eval comparison the issue proposes needs both arms runnable
    from the same checkout."""
    monkeypatch.setenv(PYTHON_SESSION_ENV, "0")
    assert default_python_session() is None
    monkeypatch.setenv(PYTHON_SESSION_ENV, "off")
    assert default_python_session() is None
    monkeypatch.setenv(PYTHON_SESSION_ENV, "1")
    live = default_python_session()
    assert isinstance(live, LocalPythonSession)
    live.close()
    monkeypatch.delenv(PYTHON_SESSION_ENV, raising=False)
    live = default_python_session()
    assert isinstance(live, LocalPythonSession)
    live.close()


def test_run_pipeline_binds_nothing_when_switched_off(monkeypatch):
    from agent.pipeline import run as run_mod

    seen: list = []

    class FakeApp:
        def invoke(self, initial, config):
            seen.append(current_python_session())
            return {"final_output": "done"}

        def get_state(self, config):
            return type("S", (), {"values": {}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])
    monkeypatch.setenv(PYTHON_SESSION_ENV, "0")
    run_mod.run_pipeline("hi", session_id="s1")
    assert seen == [None]


def test_the_prompt_note_is_empty_with_no_session():
    assert python_session_note() == ""


def test_the_prompt_note_says_state_persists_and_what_a_reset_means(session):
    """A model not told that state persists rebuilds everything every call,
    which is the cost the feature exists to remove. And one not told what a
    restart looks like would trust variables that are gone."""
    note = python_session_note()
    assert "execute_python" in note
    assert "persist" in note
    assert "restart" in note
    # Stable within a run: nodes.py rebuilds the prompt on resume and the
    # stored transcript assumes it is byte-identical.
    assert python_session_note() == note


def test_the_note_reaches_the_agents_conversation(monkeypatch):
    from langchain_core.messages import AIMessageChunk, HumanMessage

    from agent.pipeline import nodes as pn

    class Scripted:
        def __init__(self):
            self.seen = []

        def stream(self, messages):
            self.seen.append(list(messages))
            yield AIMessageChunk(content="FINAL:\ndone")

    fake = Scripted()
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["the task is done"]))
    state = {
        "messages": [HumanMessage("count the rows")], "board": [], "node": None,
        "feedback": "", "output": None, "context": "", "node_error": None,
        "pending_question": None, "pending_choices": None, "asking_role": None,
        "final_output": None, "actions": [], "transcript": None, "mode": None,
        "mode_log": [], "model_calls": 0, "rejections": 0,
    }
    with python_session():
        pn.agent(state)
    assert any(python_session_note() in str(m.content) for m in fake.seen[0])


# --------------------------------------------------------------------------
# An interpreter that will not start
# --------------------------------------------------------------------------

class _Unstartable:
    """A session whose interpreter never comes up."""

    def run(self, code, timeout):
        raise SessionUnavailable("no interpreter here")

    def close(self):
        pass

    def fresh(self):
        return _Unstartable()


def test_an_interpreter_that_cannot_start_falls_back_to_a_fresh_process():
    """Review nit on the first cut: the fallback reused the container note,
    which told the model something about a container that did not exist.
    The note names what actually happened."""
    with python_session(_Unstartable()):
        result = pt.execute_python("print('ran anyway')")
    assert result.ok, result.stderr
    assert result.stdout.strip() == "ran anyway"
    assert UNAVAILABLE_NOTE in result.stderr
    assert "container" not in result.stderr


def test_a_child_that_dies_before_its_first_request_is_unavailable_not_a_crash(monkeypatch):
    """The retry after a broken pipe is guarded too: a second failure is
    SessionUnavailable, the one exception execute_python falls back on,
    never a bare OSError out of the tool loop."""
    live = LocalPythonSession()
    calls = {"n": 0}
    real_send = live._send

    def broken_send(request):
        calls["n"] += 1
        raise BrokenPipeError("stdin closed")

    monkeypatch.setattr(live, "_send", broken_send)
    try:
        with pytest.raises(SessionUnavailable):
            live.run("print(1)", timeout=FAST_S)
        assert calls["n"] == 2
        assert not live.alive
        # And through the tool: a fresh process, with the note.
        with bind_python_session(live):
            result = pt.execute_python("print('fresh')")
        assert result.stdout.strip() == "fresh"
        assert UNAVAILABLE_NOTE in result.stderr
    finally:
        monkeypatch.setattr(live, "_send", real_send)
        live.close()


# --------------------------------------------------------------------------
# The session object itself
# --------------------------------------------------------------------------

def test_run_returns_a_session_result_with_the_reset_flag():
    with python_session() as live:
        result = live.run("print('hi')", timeout=FAST_S)
    assert isinstance(result, SessionResult)
    assert (result.stdout.strip(), result.returncode, result.timed_out, result.reset) == \
        ("hi", 0, False, False)


def test_long_output_is_bounded_but_keeps_both_ends(session):
    result = pt.execute_python("for i in range(200000): print(i)")
    assert result.ok, result.stderr
    assert result.stdout.startswith("0\n")
    assert result.stdout.rstrip().endswith("199999")
    assert len(result.stdout) <= pt._TAIL + 100


def test_line_endings_come_back_as_newlines_on_every_platform(session):
    """CI on Windows: every line came back `\\r\\n`. The fresh-process path
    decoded with universal newlines (`subprocess.run(text=True)`), so the
    session must too -- the model reads the same text on every OS. Raw
    bytes here so the case is exercised on Linux as well."""
    result = pt.execute_python(
        "import sys\nsys.stdout.buffer.write(b'a\\r\\nb\\r\\n')\nprint('c')"
    )
    assert result.ok, result.stderr
    assert result.stdout == "a\nb\nc\n"


def test_a_snippet_that_breaks_stdout_does_not_break_the_next_call(session):
    """Namespace poisoning is the model's own problem within a run, but the
    plumbing that carries results back must not be poisonable by a snippet
    that reassigns or closes sys.stdout."""
    pt.execute_python("import sys\nsys.stdout = None")
    assert pt.execute_python("print('after')").stdout.strip() == "after"
    pt.execute_python("import sys\nsys.stdout.close()")
    assert pt.execute_python("print('again')").stdout.strip() == "again"
