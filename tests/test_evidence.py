"""The verification-evidence ledger: a structural check that costs no model
call in the ordinary case.

Otto already spends two calls at the end of a run asking a model whether the
answer holds. That is the right instrument for "is this correct" and the wrong
one for "was anything checked at all", because it asks a model to notice an
absence in its own work. The absence is in the action record for free.

This is a ledger and a policy, never a runner and never a judge. It executes
nothing and it cannot block a run.
"""
import uuid

from langchain_core.messages import AIMessageChunk

from agent.memory.lessons import bind_bank
from agent.pipeline import evidence as ev
from agent.pipeline import nodes as pn
from agent.pipeline.budget import Budget, bind_budget
from agent.pipeline.run import _initial


# --------------------------------------------------------------------------
# What counts as a check
# --------------------------------------------------------------------------

def test_a_test_suite_that_passed_is_evidence():
    for command in ("pytest -q", "npm test", "go test ./...", "cargo test",
                    "make test", "uv run pytest tests/", "mypy agent/"):
        assert ev.is_check("execute_bash", command, 0), command


def test_a_check_that_failed_is_not_evidence():
    """A run that ends on a red suite has proved nothing, and treating the
    attempt as the proof is the exact confusion this file removes."""
    assert not ev.is_check("execute_bash", "pytest -q", 1)


def test_running_the_code_counts_in_any_language():
    """The general rule that carries everything the word list misses. A list
    of build-system names cannot keep up; "you ran it and it did not explode"
    holds everywhere."""
    assert ev.is_check("execute_python", "import app; app.main()", 0)


def test_looking_at_something_is_not_proving_it():
    for tool, body in (("read_file", "app.py"), ("list_files", "."),
                       ("write_file", "app.py\nx"), ("recall_memory", "app")):
        assert not ev.is_check(tool, body, 0), tool


def test_an_unrelated_shell_command_is_not_a_check():
    for command in ("ls -la", "cat app.py", "git status", "echo hello"):
        assert not ev.is_check("execute_bash", command, 0), command


# --------------------------------------------------------------------------
# What counts as needing one
# --------------------------------------------------------------------------

def test_editing_code_leaves_something_to_prove():
    ledger = ev.Ledger()
    ledger.record("write_file", "app.py\nprint(1)", 0)

    assert ledger.needs_check
    assert ledger.unproven == ["app.py"]


def test_prose_never_asks_to_be_verified():
    """A README has no runtime behaviour, and demanding a command for one
    teaches the agent to run something meaningless to satisfy the guard."""
    ledger = ev.Ledger()
    for path in ("README.md", "docs/design.rst", "notes.txt", "LICENSE",
                 "CHANGELOG", "data/rows.csv"):
        ledger.record("write_file", f"{path}\nwords", 0)

    assert not ledger.needs_check, ledger.unproven


def test_a_passing_check_clears_everything_before_it():
    ledger = ev.Ledger()
    ledger.record("write_file", "a.py\nx", 0)
    ledger.record("edit_file", "b.py\nx", 0)
    ledger.record("execute_bash", "pytest -q", 0)

    assert not ledger.needs_check


def test_editing_again_after_a_check_needs_another():
    """The whole point. What was proved before the edit says nothing about
    the file after it."""
    ledger = ev.Ledger()
    ledger.record("write_file", "a.py\nx", 0)
    ledger.record("execute_bash", "pytest -q", 0)
    ledger.record("write_file", "a.py\ny", 0)

    assert ledger.needs_check


def test_a_failed_edit_leaves_nothing_to_prove():
    ledger = ev.Ledger()
    ledger.record("write_file", "a.py\nx", 1)

    assert not ledger.needs_check


def test_the_note_names_the_files():
    """"Did you verify?" is answerable with "yes" by a model that did not.
    "app.py has changed and nothing has run since" is not."""
    ledger = ev.Ledger()
    ledger.record("write_file", "app.py\nx", 0)
    note = ev.render_note(ledger)

    assert "app.py" in note
    assert "nothing has run since" in note


def test_a_large_refactor_does_not_reprint_itself():
    ledger = ev.Ledger()
    for i in range(40):
        ledger.record("write_file", f"mod{i}.py\nx", 0)
    note = ev.render_note(ledger)

    assert "other file(s)" in note
    assert len(note) < 500


def test_the_note_accepts_that_there_may_be_nothing_to_run():
    """Enforcement with no way to say no gets routed around rather than
    obeyed: one system blocked 94% of non-compliant actions while its safe
    success rate stayed under 5%, because the agent hallucinated past it."""
    ledger = ev.Ledger()
    ledger.record("write_file", "a.py\nx", 0)

    assert "nothing to run" in ev.render_note(ledger)


# --------------------------------------------------------------------------
# In the real loop
# --------------------------------------------------------------------------

class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        reply = self._replies.pop(0) if self._replies else "FINAL:\nAPPROVE: yes\nWHY: ok"
        yield AIMessageChunk(content=reply)


def _run(monkeypatch, replies):
    model = _Scripted(replies)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: model)
    config = {
        "configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
        "recursion_limit": pn._RECURSION_SAFETY_NET,
    }
    with bind_budget(Budget(max_model_calls=40)), bind_bank(None):
        final = pn.app.invoke(_initial("write the thing"), config)
    return model, final


def test_finishing_right_after_an_edit_costs_one_exchange(monkeypatch, tmp_path):
    """The one case worth an extra exchange. Reply 1 is always the rubric --
    the criteria are written from the task before any attempt exists."""
    from agent.pipeline.workspace import bind_workspace

    with bind_workspace(str(tmp_path)):
        _, final = _run(monkeypatch, [
            "- app.py exists and runs",                       # 1, the rubric
            "ACTION: write_file\nCODE:\napp.py\nprint(1)",    # 2, an edit
            "FINAL:\nwrote it",                                # 3, HELD
            "ACTION: execute_python\nCODE:\nprint(1)",        # 4, the proof
            "FINAL:\nwrote it and ran it",                     # 5
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "wrote it and ran it", (
        "the first answer was accepted with nothing having been run"
    )


def test_it_asks_once_and_then_takes_the_answer(monkeypatch, tmp_path):
    """A guard that keeps asking is a guard the model learns to answer rather
    than act on -- and one that never lets go cannot be told there is nothing
    to run."""
    from agent.pipeline.workspace import bind_workspace

    with bind_workspace(str(tmp_path)):
        _, final = _run(monkeypatch, [
            "- the file is written",                          # 1, the rubric
            "ACTION: write_file\nCODE:\napp.py\nprint(1)",    # 2
            "FINAL:\nwrote it",                                # 3, held once
            "FINAL:\nnothing to run, it is a fixture",         # 4, taken
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "nothing to run, it is a fixture"


def test_a_run_that_changed_no_code_is_never_held(monkeypatch):
    """The ordinary case, and why this costs no model call: a run that
    answered a question pays nothing for the guard."""
    model, final = _run(monkeypatch, [
        "- the count is right",                               # 1, the rubric
        "ACTION: execute_bash\nCODE:\nls",                    # 2
        "FINAL:\nthere are three files",                       # 3
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
    ])

    assert final["final_output"] == "there are three files"
    assert model.calls == 4, "the guard charged for a run that changed nothing"
