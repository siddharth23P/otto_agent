"""Coverage for walkthroughs beyond the browser: a served app, a command
line, a program in a terminal.

One tool, `exercise`, whose first step says what kind of thing is being
used. The shape of the report is the same for every kind -- a summary, then
every step and what happened to it -- and it is written by code, so the judge
reads it as evidence (nodes.py's `_walkthrough_block`). The shell backend
runs where `execute_bash` runs and needs nothing installed; the page and
terminal backends live in the optional interpreter Playwright does.
"""
import os
import sys
import textwrap

import pytest

import agent.pipeline.tools as pt
from agent.pipeline import evidence as ev
from agent.pipeline import browsing, walkthrough
from agent.pipeline.execution import bind_command_runner
from agent.pipeline.workspace import bind_workspace
from conftest import posix_only

_REAL_BROWSER_PYTHON = os.environ.get("OTTO_BROWSER_PYTHON", "")


# --------------------------------------------------------------------------
# The first step picks the kind
# --------------------------------------------------------------------------

def test_the_first_step_picks_the_backend():
    assert walkthrough.parse_walk("open a.html\nclick Go").backend == "page"
    assert walkthrough.parse_walk("serve npm run dev\nopen http://127.0.0.1:5173/").backend == "page"
    assert walkthrough.parse_walk("serve uvicorn app\nrequest GET http://127.0.0.1:8000/").backend == "shell"
    assert walkthrough.parse_walk("run ls\nexpect a").backend == "shell"
    assert walkthrough.parse_walk("tty python3 app.py\nexpect Name").backend == "tty"


def test_steps_from_another_kind_are_refused_by_name():
    """Short and specific: the kind's own steps, nothing else. The full help
    here sent a model back with the same `exit` in a page walkthrough three
    times running -- measured live."""
    why = walkthrough.parse_walk("open a.html\nrun ls")
    assert why.startswith("`run` is not a page step")
    assert "click, type, press, expect, count" in why and "serve" not in why
    # `exit = 0` on a page is accepted as "it threw nothing" -- models kept
    # writing it to mean "and that is the end" -- and only that form.
    assert walkthrough.parse_walk("open a.html\nexit = 0").backend == "page"
    assert "no exit code" in walkthrough.parse_walk("open a.html\nexit = 2")
    assert "not a shell step" in walkthrough.parse_walk("run ls\nclick Go")
    assert "not a tty step" in walkthrough.parse_walk("tty app\ncount css .x = 1")
    assert "cannot start" in walkthrough.parse_walk("click Go")


def test_a_page_walkthrough_may_request_its_own_server_too():
    """A served app has a page AND an API, and a model asked for both in one
    walkthrough -- three times, and was refused each time. Now it can."""
    walk = walkthrough.parse_walk(
        "serve python3 serve.py\nopen http://127.0.0.1:8765/\nclick Add\n"
        "request GET http://127.0.0.1:8765/api/todos\nstatus = 200\nexpect milk"
    )
    assert walk.backend == "page"
    for verb in ("request", "status"):
        assert f'verb == "{verb}"' in browsing.DRIVER


def test_a_request_may_carry_a_json_body():
    assert walkthrough.split_request('POST http://127.0.0.1:1/items {"name": "x"}') == (
        "POST", "http://127.0.0.1:1/items", '{"name": "x"}')
    assert walkthrough.split_request("get http://127.0.0.1:1/") == ("GET", "http://127.0.0.1:1/", "")
    assert "json body" in walkthrough.parse_walk("run ls\nrequest /items")


def test_a_server_needs_something_to_do_with_it():
    assert "say what to do" in walkthrough.parse_walk("serve npm run dev")
    assert "loopback" in walkthrough.parse_walk("serve x\nopen https://example.com/")
    assert "loopback" in walkthrough.parse_walk("serve x\nrequest GET https://example.com/")
    assert walkthrough.serve_port(walkthrough.parse_walk("serve x\nopen http://localhost:5173/app")) == 5173
    assert walkthrough.serve_port(walkthrough.parse_walk("serve x\nrequest GET http://127.0.0.1:8000/h")) == 8000


def test_written_forms_are_checked_before_anything_runs():
    assert "exit = N" in walkthrough.parse_walk("run ls\nexit zero")
    assert "status = N" in walkthrough.parse_walk("run ls\nstatus ok")
    assert "count <selector> = N" in walkthrough.parse_walk("open a.html\ncount .piece 32")
    assert "request GET <url>" in walkthrough.parse_walk("run ls\nrequest /health")
    assert "milliseconds" in walkthrough.parse_walk("tty app\nwait soon")
    assert "only the first" in walkthrough.parse_walk("run ls\nserve x")
    assert "only the first" in walkthrough.parse_walk("tty a\ntty b")
    assert "too many" in walkthrough.parse_walk("run ls\n" + "expect x\n" * walkthrough.MAX_WALK_STEPS)
    assert "empty" in walkthrough.parse_walk("  \n ")


def test_steps_without_an_argument_are_the_ones_that_take_none():
    walk = walkthrough.parse_walk("open a.html\npress Enter\nchanged\nwait 100")
    assert [s.verb for s in walk.steps] == ["open", "press", "changed", "wait"]
    assert "needs something" in walkthrough.parse_walk("open a.html\nclick")
    assert walkthrough.parse_walk("tty app\nscreen").backend == "tty"


# --------------------------------------------------------------------------
# The shell backend, for real: it needs nothing installed
# --------------------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path):
    with bind_workspace(tmp_path) as ws:
        yield ws


def test_a_command_line_tool_is_used_and_its_output_checked(workspace):
    (workspace / "tool.py").write_text(textwrap.dedent("""
        import sys
        if "--bogus" in sys.argv:
            print("unknown option", file=sys.stderr); sys.exit(2)
        n = int(sys.argv[sys.argv.index("--count") + 1])
        print(f"{n} items")
    """))
    result = pt.exercise(textwrap.dedent(f"""
        run {sys.executable} tool.py --count 3
        expect 3 items
        expect not 4 items
        exit = 0
        run {sys.executable} tool.py --bogus
        exit = 2
        expect unknown option
    """))

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("7/7 steps passed")
    assert "-> ok (exit 0): 3 items" in result.stdout


@posix_only
def test_a_crash_cannot_be_walked_past_unless_expected(workspace):
    result = pt.exercise(f"run {sys.executable} -c 'raise SystemExit(3)'\nexpect anything")

    assert result.returncode == 1
    assert "exited 3" in result.stderr
    assert "add `exit = 3` after this step" in result.stdout


@posix_only
def test_an_expected_exit_may_come_after_the_message_check(workspace):
    """`expect <message>` then `exit = 2` means the same as the other order.
    Insisting on the order sent a model back ten times running."""
    result = pt.exercise(
        f"run {sys.executable} -c 'import sys; print(\"no such file\", file=sys.stderr); sys.exit(2)'\n"
        "expect no such file\nexit = 2"
    )

    assert result.returncode == 0, result.stderr


def test_a_posted_body_reaches_the_server(workspace):
    (workspace / "srv.py").write_text(textwrap.dedent("""
        import json, sys
        from http.server import BaseHTTPRequestHandler, HTTPServer
        items = []
        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass
            def do_GET(self):
                body = json.dumps(items).encode()
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(body)
            def do_POST(self):
                n = int(self.headers.get("Content-Length", 0))
                items.append(json.loads(self.rfile.read(n)))
                self.send_response(201); self.end_headers(); self.wfile.write(b"created")
        HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
    """))
    port = 8700 + os.getpid() % 200
    result = pt.exercise(textwrap.dedent(f"""
        serve {sys.executable} srv.py {port}
        request POST http://127.0.0.1:{port}/items {{"name": "milk"}}
        status = 201
        expect created
        request GET http://127.0.0.1:{port}/items
        expect milk
    """))

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith("6/6 steps passed")


def test_a_wrong_expectation_names_what_was_printed(workspace):
    result = pt.exercise("run echo hello world\nexpect goodbye")

    assert result.returncode == 1
    assert "not in the last output" in result.stderr
    assert "hello world" in result.stderr


def test_a_served_api_is_requested_and_stopped(workspace):
    """A server started for the walkthrough, requests against it, and the
    server gone when the report is written."""
    (workspace / "index.html").write_text("<p>served ok</p>")
    port = 8600 + os.getpid() % 300
    result = pt.exercise(textwrap.dedent(f"""
        serve {sys.executable} -m http.server {port} --bind 127.0.0.1
        request GET http://127.0.0.1:{port}/index.html
        status = 200
        expect served ok
        request GET http://127.0.0.1:{port}/missing.html
        status = 404
    """))

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("6/6 steps passed")
    assert f"port {port} is answering" in result.stdout
    import socket
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


def test_a_server_that_never_comes_up_says_so_with_its_output(workspace, monkeypatch):
    monkeypatch.setattr(walkthrough, "SERVE_TIMEOUT_S", 2.0)
    result = pt.exercise(textwrap.dedent(f"""
        serve {sys.executable} -c "print('boom, no port'); raise SystemExit(1)"
        request GET http://127.0.0.1:8999/
    """))

    assert result.returncode == 1
    assert "exited before opening port" in result.stderr
    assert "boom, no port" in result.stderr


def test_in_a_container_commands_go_through_the_runner():
    seen = []

    def runner(command, timeout):
        seen.append(command)
        return ("3 items\n", "", 0)

    with bind_command_runner(runner):
        result = pt.exercise("run mytool --count 3\nexpect 3 items\nexit = 0")

    assert result.returncode == 0
    assert seen == ["mytool --count 3"]


def test_in_a_container_a_server_is_detached_and_killed():
    seen = []

    def runner(command, timeout):
        seen.append(command)
        if command.startswith("nohup"):
            return ("4242\n", "", 0)
        if command.startswith("kill -0"):
            return ("", "", 0)
        if command.startswith("curl"):
            return ('{"ok":true}\n__OTTO_STATUS__200', "", 0)
        return ("", "", 0)

    with bind_command_runner(runner), \
            pytest.MonkeyPatch.context() as mp:
        mp.setattr(walkthrough, "_wait_for_port", lambda port, shell, seconds=0: None)
        result = pt.exercise(
            "serve uvicorn app:app --port 8000\nrequest GET http://127.0.0.1:8000/health\n"
            "status = 200\nexpect ok"
        )

    assert result.returncode == 0, result.stderr
    assert any(c.startswith("nohup sh -c") for c in seen)
    assert any(c.startswith("kill") and "4242" in c for c in seen), "the server was left running"


# --------------------------------------------------------------------------
# The terminal backend
# --------------------------------------------------------------------------

def test_without_a_terminal_emulator_it_names_the_fix(workspace):
    result = pt.exercise("tty python3 app.py\nexpect Name?")

    assert result.returncode == 1
    assert "pyte" in result.stderr


def test_the_terminal_driver_knows_every_tty_step():
    for verb in walkthrough.TTY_STEPS:
        assert f'verb == "{verb}"' in walkthrough.PTY_DRIVER, verb


def test_otto_itself_needs_no_terminal_emulator():
    with pytest.raises(ImportError):
        __import__("pyte")


@pytest.mark.skipif(not _REAL_BROWSER_PYTHON, reason="set OTTO_BROWSER_PYTHON to an interpreter with pyte")
def test_for_real_a_program_in_a_terminal_is_driven_and_its_screen_read(workspace, monkeypatch):
    monkeypatch.setattr(walkthrough, "_TTY", _REAL_BROWSER_PYTHON)
    (workspace / "ask.py").write_text(
        'name = input("Name? ")\nprint(f"Hello, {name}!")\nraise SystemExit(0)\n'
    )
    result = pt.exercise(textwrap.dedent(f"""
        tty {sys.executable} ask.py
        expect Name?
        type otto
        press Enter
        expect Hello, otto!
        screen
        exit = 0
    """))

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith("7/7 steps passed")
    assert "| Hello, otto!" in result.stdout, "the screen is in the report"

    wrong = pt.exercise(f"tty {sys.executable} ask.py\nexpect Password?")
    assert wrong.returncode == 1
    assert "not on screen" in wrong.stderr


@pytest.mark.skipif(not _REAL_BROWSER_PYTHON, reason="set OTTO_BROWSER_PYTHON to an interpreter with pyte")
def test_for_real_a_typed_escape_is_the_key_it_names(workspace, monkeypatch):
    """`type \\x1b[A` is the up arrow. Left literal, a model taught its own
    curses app to read a backslash as a key -- live."""
    monkeypatch.setattr(walkthrough, "_TTY", _REAL_BROWSER_PYTHON)
    (workspace / "keys.py").write_text(textwrap.dedent("""
        import curses
        def main(s):
            s.addstr(0, 0, "ready"); s.refresh()
            k = s.getch()
            s.addstr(1, 0, "UP" if k == curses.KEY_UP else f"other {k}"); s.refresh()
            s.getch()
        curses.wrapper(main)
    """))
    result = pt.exercise(
        f"tty {sys.executable} keys.py\nexpect ready\ntype \\x1b[A\nexpect UP\npress q"
    )

    assert result.returncode == 0, result.stderr + result.stdout


# --------------------------------------------------------------------------
# A served app in the browser, for real
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _REAL_BROWSER_PYTHON, reason="set OTTO_BROWSER_PYTHON to an interpreter with Playwright")
def test_for_real_an_app_is_served_and_used_in_the_browser(workspace, monkeypatch):
    monkeypatch.setattr(browsing, "_LOCAL", _REAL_BROWSER_PYTHON)
    (workspace / "counter.html").write_text(
        "<!doctype html><title>Counter</title><p id=n>Count: 0</p><button id=add>Add one</button>"
        "<script>let c=0;document.getElementById('add').onclick=()=>{c++;"
        "document.getElementById('n').textContent='Count: '+c};</script>"
    )
    port = 8900 + os.getpid() % 90
    result = pt.exercise(textwrap.dedent(f"""
        serve {sys.executable} -m http.server {port} --bind 127.0.0.1
        open http://127.0.0.1:{port}/counter.html
        expect Count: 0
        click Add one
        changed
        expect Count: 1
    """))

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith("6/6 steps passed")
    import socket
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1).close()


# --------------------------------------------------------------------------
# What the ledger makes of each kind
# --------------------------------------------------------------------------

def test_only_a_page_walkthrough_settles_a_page():
    ledger = ev.Ledger()
    ledger.record("write_file", "index.html\n<p>", 0)
    ledger.record("write_file", "tool.py\nprint(1)", 0)
    ledger.record("exercise", "run python3 tool.py\nexit = 0", 0)

    assert not ledger.needs_check, "a shell walkthrough is a run"
    assert ledger.needs_render, "but it loaded no page"

    ledger.record("exercise", "serve python3 -m http.server\nopen http://127.0.0.1:8000/", 0)
    assert not ledger.needs_render


@pytest.mark.skipif(not _REAL_BROWSER_PYTHON, reason="set OTTO_BROWSER_PYTHON to an interpreter with pyte")
def test_for_real_press_up_reaches_a_curses_app(workspace, monkeypatch):
    """curses switches the terminal to application cursor mode and expects
    ESC O A for Up. The driver follows the switch, as a real terminal does."""
    monkeypatch.setattr(walkthrough, "_TTY", _REAL_BROWSER_PYTHON)
    (workspace / "counter.py").write_text(textwrap.dedent("""
        import curses
        def main(s):
            n = 0
            while True:
                s.clear(); s.addstr(0, 0, f"Count: {n}"); s.refresh()
                k = s.getch()
                if k == curses.KEY_UP: n += 1
                elif k == curses.KEY_DOWN: n -= 1
                elif k == ord("q"): return
        curses.wrapper(main)
    """))
    result = pt.exercise(textwrap.dedent(f"""
        tty {sys.executable} counter.py
        expect Count: 0
        press Up
        press Up
        expect Count: 2
        press Down
        expect Count: 1
        press q
        exit = 0
    """))

    assert result.returncode == 0, result.stderr + result.stdout
    assert result.stdout.startswith("9/9 steps passed")


def test_a_launch_only_walkthrough_says_so_in_its_summary(workspace):
    """Live, a model launched its app with one step and the judge took
    "1/1 steps passed" for the buttons working."""
    result = pt.exercise("run echo started")
    assert result.returncode == 0
    assert "launch only" in result.stdout.splitlines()[0]
    assert "launch only" not in pt.exercise("run echo started\nexpect started").stdout
