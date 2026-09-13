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


# --------------------------------------------------------------------------
# Pages: a test of the logic is not a load of the page
# --------------------------------------------------------------------------
#
# The chess game that motivated this passed its own move-generation tests and
# `node --check` with a stray token in the script that threw before the board
# was rendered. The ledger now keeps a second list for files whose runtime is
# a browser, cleared only by a browser loading something.

def test_a_page_load_that_came_back_clean_is_evidence():
    """`browse` fails the call when a workspace page throws (tools.py), so a
    clean return is "you ran it and it did not explode" for a page."""
    assert ev.is_check("browse", "open index.html", 0)
    assert not ev.is_check("browse", "open index.html", 1)


def test_a_walkthrough_that_passed_is_evidence_and_settles_the_page():
    """`exercise` fails at the first step that does not hold and on any page
    error, so a clean return is the strongest thing the ledger can see."""
    assert ev.is_check("exercise", "open index.html\nclick New game", 0)
    assert not ev.is_check("exercise", "open index.html\nclick New game", 1)
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("exercise", "open index.html\nexpect Chess", 0)

    assert not ledger.needs_render and not ledger.needs_check


def test_the_render_note_asks_for_a_walkthrough():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)

    assert "exercise" in ev.render_note(ledger, browser=True)


def test_a_test_suite_does_not_prove_a_page_draws():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("execute_python", "import re; assert re.search('p', open('index.html').read())", 0)

    assert not ledger.needs_check, "the script did run"
    assert ledger.needs_render, "but nothing loaded the page"
    assert ledger.unrendered == ["index.html"]


def test_loading_the_page_settles_it():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("edit_file", "app.js\n---OLD---\na\n---NEW---\nb", 0)
    ledger.record("browse", "open index.html", 0)

    assert not ledger.needs_render
    assert not ledger.needs_check, "a load is a run"


def test_a_page_that_threw_settles_nothing():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("browse", "open index.html", 1)

    assert ledger.needs_render


def test_only_files_a_browser_runs_are_pages():
    for path in ("index.html", "site/page.htm", "app.js", "style.css",
                 "src/App.tsx", "Card.vue", "x.svelte"):
        assert ev.is_web(path), path
    for path in ("app.py", "main.go", "README.md", "Makefile", "data.json"):
        assert not ev.is_web(path), path


def test_the_render_note_names_the_command_and_only_when_a_browser_is_reachable():
    """A note that names a tool the run cannot call teaches it to lie."""
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("execute_python", "print(1)", 0)

    with_browser = ev.render_note(ledger, browser=True)
    assert "index.html" in with_browser
    assert "exercise" in with_browser
    assert "look" in with_browser
    assert "not something a browser shows" in with_browser, "a way to say no"

    assert "exercise" not in ev.render_note(ledger, browser=False)


def test_a_changed_page_outranks_a_changed_file_when_both_are_open():
    """The load IS a run, so the more specific ask is the one to make."""
    ledger = ev.Ledger()
    ledger.record("write_file", "app.py\nx", 0)
    ledger.record("write_file", "index.html\n<p>", 0)

    assert "exercise" in ev.render_note(ledger, browser=True)
    assert "nothing has run since" in ev.render_note(ledger, browser=False)


def test_finishing_after_editing_a_page_is_held_until_a_browser_loads_it(monkeypatch, tmp_path):
    """In the real loop, with a local browser at hand: the Python check clears
    the first hold and not the second, and the answer is taken only after a
    clean load."""
    from agent.pipeline import browsing
    from agent.pipeline.workspace import bind_workspace

    monkeypatch.setattr(browsing, "_LOCAL", "/some/python")
    monkeypatch.setattr(
        browsing, "run_local",
        lambda op, argument, limits, *, workspace, timeout=90.0:
            ("url: file:///x/index.html\ntitle: t\ntext:\n  drawn", "", 0),
    )
    with bind_workspace(str(tmp_path)):
        _, final = _run(monkeypatch, [
            "- index.html loads and draws",                          # 1, rubric
            "ACTION: write_file\nCODE:\nindex.html\n<p>hi</p>",      # 2, a page
            "ACTION: execute_python\nCODE:\nprint(1)",               # 3, a run
            "FINAL:\nwrote it",                                       # 4, HELD
            "ACTION: browse\nCODE:\nopen index.html",                 # 5, the load
            "FINAL:\nwrote it and it loads",                          # 6
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "wrote it and it loads", (
        "the answer was accepted with the page never loaded"
    )


def test_without_a_browser_a_page_is_just_a_file(monkeypatch, tmp_path):
    """No local interpreter and no container: no hold names the browser. The
    one hold there is asks for the thing to be USED (`exercise` reaches a
    shell with any workspace), and "nothing a person runs" is taken."""
    from agent.pipeline.workspace import bind_workspace

    with bind_workspace(str(tmp_path)):
        model, final = _run(monkeypatch, [
            "- index.html exists",                                    # 1, rubric
            "ACTION: write_file\nCODE:\nindex.html\n<p>hi</p>",      # 2
            "ACTION: execute_python\nCODE:\nprint(1)",               # 3, clears the run
            "FINAL:\nwrote it",                                       # 4, HELD once
            "FINAL:\nwrote it; nothing a person runs here",           # 5, taken
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "wrote it; nothing a person runs here"


# --------------------------------------------------------------------------
# Using it: a test the agent wrote is not a person using the thing
# --------------------------------------------------------------------------
#
# Measured live, five runs in a row -- a CLI, an API, a curses app -- each
# wrote its own harness, passed it, finished without once using the thing
# the way a person would, and the judge approved every one.

def test_a_passing_harness_clears_the_run_but_not_the_use():
    ledger = ev.Ledger()
    ledger.record("write_file", "wordfreq.py\nprint(1)", 0)
    ledger.record("write_file", "run_tests.sh\nset -e", 0)
    ledger.record("execute_bash", "pytest -q", 0)

    assert not ledger.needs_check
    assert ledger.needs_use
    assert ledger.unused == ["wordfreq.py", "run_tests.sh"]


def test_a_walkthrough_of_any_kind_settles_the_use():
    for first in ("run python3 wordfreq.py f.txt\nexit = 0", "tty python3 counter.py\nexpect Count",
                  "serve python3 api.py\nrequest GET http://127.0.0.1:8790/health\nstatus = 200"):
        ledger = ev.Ledger()
        ledger.record("write_file", "thing.py\nprint(1)", 0)
        ledger.record("exercise", first, 0)
        assert not ledger.needs_use, first
        assert not ledger.needs_check, first


def test_tests_and_prose_are_not_things_a_person_uses():
    ledger = ev.Ledger()
    for path in ("tests/test_x.py", "test_interactive.py", "x_test.go", "spec/a.spec.ts",
                 "conftest.py", "README.md"):
        ledger.record("write_file", f"{path}\nx", 0)

    assert not ledger.needs_use, ledger.unused


def test_the_use_note_only_when_exercise_is_reachable():
    ledger = ev.Ledger()
    ledger.record("write_file", "counter.py\nx", 0)
    ledger.record("execute_bash", "python3 counter.py --test", 0)

    note = ev.render_note(ledger, exercise=True)
    assert "counter.py" in note and "exercise" in note and "tty" in note
    assert "nothing a person runs" in note, "a way to say no"
    assert "exercise" not in ev.render_note(ledger, exercise=False)


def test_a_page_nobody_loaded_outranks_code_nobody_used():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)

    assert "browser has loaded" in ev.render_note(ledger, browser=True, exercise=True)
    assert "used it the way a person would" in ev.render_note(ledger, browser=False, exercise=True)


def test_finishing_on_a_passing_harness_is_held_until_the_thing_is_used(monkeypatch, tmp_path):
    """In the real loop, with a workspace (so `exercise` is reachable): the
    harness clears the first hold and not the second; the answer is taken
    only after a walkthrough."""
    from agent.pipeline import walkthrough
    from agent.pipeline.workspace import bind_workspace

    monkeypatch.setattr(walkthrough, "LocalShell", lambda cwd: _FakeShell())
    with bind_workspace(str(tmp_path)):
        _, final = _run(monkeypatch, [
            "- tool.py prints the count",                              # 1, rubric
            "ACTION: write_file\nCODE:\ntool.py\nprint('3 items')",   # 2
            "ACTION: execute_bash\nCODE:\npython3 tool.py",           # 3, a run
            "FINAL:\nwrote it",                                        # 4, HELD
            "ACTION: exercise\nCODE:\nrun python3 tool.py\nexpect 3 items\nexit = 0",  # 5
            "FINAL:\nwrote it and used it",                            # 6
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "wrote it and used it", (
        "the answer was accepted on the agent's own harness alone"
    )


class _FakeShell:
    def run(self, command, timeout=0):
        return "3 items\n", 0

    def close(self):
        pass


def test_a_launch_only_walkthrough_does_not_count_as_using_it():
    """One step proves the thing starts. Live, a model launched its AppKit
    app with `mac ./counter` alone and the judge took that for the buttons
    working."""
    ledger = ev.Ledger()
    ledger.record("write_file", "counter.swift\nimport Cocoa", 0)
    ledger.record("exercise", "mac ./counter", 0)

    assert ledger.needs_use and ledger.launch_only == "mac ./counter"
    note = ev.render_note(ledger, exercise=True)
    assert "only launched it" in note and "test mode" in note

    ledger.record("exercise", "mac ./counter\nclick Add one\nexpect Count: 1", 0)
    assert not ledger.needs_use and not ledger.launch_only


def test_a_launch_only_walkthrough_is_held_once_more(monkeypatch, tmp_path):
    """In the loop: the first hold asks for use, a launch-only walkthrough
    answers it, and the second hold asks for the steps after the launch."""
    from agent.pipeline import walkthrough
    from agent.pipeline.workspace import bind_workspace

    monkeypatch.setattr(walkthrough, "LocalShell", lambda cwd: _FakeShell())
    with bind_workspace(str(tmp_path)):
        _, final = _run(monkeypatch, [
            "- tool.py prints the count",                              # 1, rubric
            "ACTION: write_file\nCODE:\ntool.py\nprint('3 items')",   # 2
            "FINAL:\nwrote it",                                        # 3, HELD: nothing used
            "ACTION: exercise\nCODE:\nrun python3 tool.py",           # 4, launch only
            "FINAL:\nwrote it and launched it",                        # 5, HELD: only launched
            "ACTION: exercise\nCODE:\nrun python3 tool.py\nexpect 3 items",  # 6
            "FINAL:\nwrote it and used it",                            # 7
            "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok",
        ])

    assert final["final_output"] == "wrote it and used it"
