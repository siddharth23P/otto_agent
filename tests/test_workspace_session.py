"""A workspace reaching an interactive session, end to end.

tests/test_workspace_tools.py already covers what a file tool does once a
workspace IS bound. Nothing covered whether an `otto chat`/`otto tui` turn
could ever get one bound, and until 2026-09-12 the answer was no: only
agent/eval/'s harnesses called bind_workspace, so every file tool in every
interactive turn refused with "no workspace is bound for this run" and the
agent was left guessing paths. These pin the chain that closes that --
CLI flag, Session, run.py's entry points, the contextvar the tools read, and
the prompt note that tells the model where it landed.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.cli.shell import (
    Session, describe_workspace, resolve_workspace, set_workspace,
)
from agent.pipeline import tools as tools_mod
from agent.pipeline.tools import (
    COMMAND_TIMEOUT_ENV, THROWAWAY_TIMEOUT_S, WORKSPACE_TIMEOUT_S,
    default_timeout, read_file, write_file,
)
from agent.pipeline.workspace import bind_workspace, current_workspace, workspace_note


class FakeCtx:
    pass


# --------------------------------------------------------------------------
# Resolving the flags
# --------------------------------------------------------------------------

def test_no_flags_means_the_current_directory(tmp_path, monkeypatch):
    """The 2026-09-12 design call: a session works on the repo you launched it
    in, the way every other developer tool does."""
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace(None, False) == tmp_path.resolve()


def test_an_explicit_path_wins_over_the_current_directory(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(tmp_path)
    assert resolve_workspace(other, False) == other.resolve()


def test_no_workspace_beats_an_explicit_path(tmp_path):
    """The two together are a contradiction, and it resolves towards LESS
    access -- the only direction that cannot surprise anyone."""
    assert resolve_workspace(tmp_path, True) is None


def test_describe_says_which_state_a_session_is_in(tmp_path):
    assert str(tmp_path) in describe_workspace(tmp_path)
    assert "file tools are off" in describe_workspace(None)


# --------------------------------------------------------------------------
# Changing it mid-session
# --------------------------------------------------------------------------

def test_set_workspace_takes_a_real_directory(tmp_path):
    session = Session(ctx=FakeCtx())
    assert set_workspace(session, str(tmp_path)) is None
    assert session.workspace == tmp_path.resolve()


def test_set_workspace_refuses_a_path_that_is_not_there(tmp_path):
    """Rejected rather than created, unlike bind_workspace: a harness wants a
    fresh scratch dir, a person at a prompt has typo'd."""
    session = Session(ctx=FakeCtx())
    missing = tmp_path / "projcts"
    problem = set_workspace(session, str(missing))
    assert problem and "does not exist" in problem
    assert session.workspace is None
    assert not missing.exists()


def test_set_workspace_refuses_a_file(tmp_path):
    session = Session(ctx=FakeCtx())
    target = tmp_path / "a.txt"
    target.write_text("x")
    problem = set_workspace(session, str(target))
    assert problem and "not a directory" in problem
    assert session.workspace is None


def test_a_new_session_keeps_the_workspace(tmp_path):
    """"Clear history, start fresh" is about the conversation. Someone who
    opened otto on a repo and cleared the chat is still on that repo."""
    session = Session(ctx=FakeCtx(), workspace=tmp_path)
    before = session.session_id
    session.reset()
    assert session.session_id != before
    assert session.workspace == tmp_path


def test_workspace_arg_is_what_run_py_wants(tmp_path):
    assert Session(ctx=FakeCtx(), workspace=tmp_path).workspace_arg() == str(tmp_path)
    assert Session(ctx=FakeCtx()).workspace_arg() is None


# --------------------------------------------------------------------------
# run.py actually binding it
# --------------------------------------------------------------------------

def test_run_pipeline_stream_binds_the_workspace_for_the_graph(tmp_path, monkeypatch):
    """The whole point of moving the bind inside run.py: a contextvar set on
    the CLI's thread is invisible to a generator consumed on a worker thread,
    so the binding has to happen where the graph runs."""
    from agent.pipeline import run as run_mod

    seen: list[Path | None] = []

    class FakeApp:
        def stream(self, *args, **kwargs):
            seen.append(current_workspace())
            return iter(())

        def get_state(self, config):
            return type("S", (), {"values": {"final_output": "done"}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])

    events = list(run_mod.run_pipeline_stream(
        "hi", session_id="s1", workspace=str(tmp_path),
    ))

    assert seen == [tmp_path.resolve()]
    assert events[-1]["__final__"]["final_output"] == "done"
    # And it is unbound again once the generator is done.
    assert current_workspace() is None


def test_run_pipeline_stream_without_a_workspace_binds_nothing(monkeypatch):
    from agent.pipeline import run as run_mod

    seen: list[Path | None] = []

    class FakeApp:
        def stream(self, *args, **kwargs):
            seen.append(current_workspace())
            return iter(())

        def get_state(self, config):
            return type("S", (), {"values": {}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])
    list(run_mod.run_pipeline_stream("hi", session_id="s1"))
    assert seen == [None]


def test_a_harness_binding_its_own_workspace_still_wins(tmp_path, monkeypatch):
    """agent/eval/ wraps its own `with bind_workspace(scratch)` around the
    call and passes no workspace= argument. That has to keep working."""
    from agent.pipeline import run as run_mod

    seen: list[Path | None] = []

    class FakeApp:
        def stream(self, *args, **kwargs):
            seen.append(current_workspace())
            return iter(())

        def get_state(self, config):
            return type("S", (), {"values": {}})()

    monkeypatch.setattr(run_mod, "app", FakeApp())
    monkeypatch.setattr(run_mod.ROUTER, "prewarm", lambda: [])
    with bind_workspace(tmp_path):
        list(run_mod.run_pipeline_stream("hi", session_id="s1"))
    assert seen == [tmp_path.resolve()]


# --------------------------------------------------------------------------
# What the model is told
# --------------------------------------------------------------------------

def test_the_prompt_note_is_empty_with_no_workspace():
    assert workspace_note() == ""


def test_the_prompt_note_names_the_root_and_what_is_in_it(tmp_path):
    """A model that is not told where it is invents a path -- the observed
    failure was `/workspace/otto_ui.py` then `/tmp/otto_ui.py`."""
    (tmp_path / "pyproject.toml").write_text("[project]")
    (tmp_path / "src").mkdir()
    (tmp_path / ".hidden").write_text("x")
    with bind_workspace(tmp_path):
        note = workspace_note()
    assert str(tmp_path.resolve()) in note
    assert "pyproject.toml" in note
    assert "src/" in note
    assert ".hidden" not in note  # dotfiles are noise, not orientation


def test_the_prompt_note_does_not_dump_a_huge_directory(tmp_path):
    for i in range(80):
        (tmp_path / f"f{i:03d}.txt").write_text("x")
    with bind_workspace(tmp_path):
        note = workspace_note()
    assert "and 56 more" in note


# --------------------------------------------------------------------------
# The command timeout
# --------------------------------------------------------------------------

def test_timeout_is_short_without_a_workspace(monkeypatch):
    monkeypatch.delenv(COMMAND_TIMEOUT_ENV, raising=False)
    assert default_timeout() == THROWAWAY_TIMEOUT_S


def test_timeout_is_long_with_one(tmp_path, monkeypatch):
    """`pytest` on a real repository does not finish in ten seconds, and what
    the agent reads back from a cut-off build is `[timed out]`, which is
    indistinguishable from a suite that hung."""
    monkeypatch.delenv(COMMAND_TIMEOUT_ENV, raising=False)
    with bind_workspace(tmp_path):
        assert default_timeout() == WORKSPACE_TIMEOUT_S


def test_the_env_var_overrides_both(tmp_path, monkeypatch):
    monkeypatch.setenv(COMMAND_TIMEOUT_ENV, "300")
    assert default_timeout() == 300.0
    with bind_workspace(tmp_path):
        assert default_timeout() == 300.0


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "   "])
def test_a_broken_env_var_is_ignored_not_fatal(bad, monkeypatch):
    """A typo in an env var must not be why a run dies, and treating "abc" as
    zero would time every command out instantly."""
    monkeypatch.setenv(COMMAND_TIMEOUT_ENV, bad)
    assert default_timeout() == THROWAWAY_TIMEOUT_S


# --------------------------------------------------------------------------
# The thing the user actually asked for
# --------------------------------------------------------------------------

def test_otto_can_read_edit_and_verify_a_real_codebase(tmp_path):
    """The end-to-end shape of "work on an already implemented codebase":
    read a file that was already there, change it, and run something that
    sees the change. Every step of this returned "no workspace is bound"
    before this work."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (tmp_path / "test_calc.py").write_text(
        "from calc import add\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n"
    )

    with bind_workspace(tmp_path):
        original = read_file("calc.py")
        assert original.ok and "a - b" in original.stdout

        written = write_file("calc.py\ndef add(a, b):\n    return a + b\n")
        assert written.ok, written.stderr

        proof = tools_mod.execute_bash("python -m pytest test_calc.py -q")
        assert proof.ok, f"{proof.stdout}\n{proof.stderr}"

    assert (tmp_path / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
