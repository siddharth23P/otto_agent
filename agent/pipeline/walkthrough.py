"""Using what was built, step by step, and reporting each step as the machine
saw it -- for a page, a served app, a command, or a program in a terminal.

One tool, `exercise` (agent/pipeline/tools.py), and the FIRST step picks how
the rest run:

    open index.html            a page in the workspace, in a real browser
    serve npm run dev          a server started for the walkthrough, then a page
    open http://127.0.0.1:5173/     on it or requests against it
    run mytool --count 3       commands in a shell, with what they printed
    tty python3 app.py         a program in a pseudo-terminal, with its screen

Whatever the backend, the shape of the answer is the same: a summary line,
then every step and what happened to it, stopping at the first that did not
hold. It is written by code -- what the browser found, what the command
printed, what was on the terminal -- so the judge can read it as evidence and
not as the model's account of itself (nodes.py's `_walkthrough_block`), and
a clean return is a check in agent/pipeline/evidence.py's sense: a command
that would have failed if the thing did not work.

WHY NOT ONE TOOL PER KIND. The agent prompt has a measured ceiling
(tests/test_prompt_tool_sync.py), and four names with four vocabularies would
not fit under it, nor would a model keep them apart. One name, one shape,
and a first line that says which world the steps live in.

WHERE EACH ONE RUNS. The shell backend runs where `execute_bash` does --
here, or in the bound container -- and needs nothing installed. The browser
backend is agent/pipeline/browsing.py's driver. The terminal backend is a
driver script of its own (PTY_DRIVER below) that needs a terminal emulator
(`pyte`) in whichever interpreter `OTTO_BROWSER_PYTHON` names, the same
optional interpreter Playwright lives in. Otto itself still ships neither.

A DESKTOP APP HAS NO LOCAL PATH, deliberately: a tool that clicks a real
desktop clicks whatever its owner has open (agent/pipeline/screen.py). In a
container, `look_act` is that tool.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

#: The steps each backend understands. Small vocabularies on purpose: one a
#: model uses correctly beats a faithful reproduction of a keyboard.
PAGE_STEPS = ("open", "serve", "click", "type", "press", "expect", "count", "wait", "changed",
              "request", "status", "exit")
SHELL_STEPS = ("run", "serve", "request", "expect", "exit", "status", "wait")
TTY_STEPS = ("tty", "type", "press", "expect", "wait", "screen", "exit")
#: An app on a device or a desktop -- agent/pipeline/native.py. The first
#: step names the platform; the rest are the same for all five.
DEVICE_KINDS = ("android", "ios", "mac", "linux", "windows")
DEVICE_STEPS = ("click", "type", "press", "expect", "wait", "screen", "changed")

#: How many steps one walkthrough may hold. A hundred-step script is a test
#: suite, and there are better tools for one of those.
MAX_WALK_STEPS = 40

#: How long a `serve`d command gets to open its port.
SERVE_TIMEOUT_S = 60.0

WALK_HELP = (
    "one step per line. The FIRST says what kind of thing this is: "
    "`open <workspace path>` for a page; `serve <command>` then "
    "`open http://127.0.0.1:PORT/` for an app on a server it starts; "
    "`run <command>` for a command line; `tty <command>` for a program in a "
    "terminal; `android <apk> <package>` or `android <package>`, `ios <.app or "
    "bundle id>`, `mac <app name or .app>`, `linux <command>`, `windows "
    "<command>` for an app on a device or a desktop. Then, for a page: `click <text>` or `click css <selector>`, "
    "`type <label> = <value>`, `press <key>`, `expect [not] <text>`, "
    "`count <selector> = N`, `changed` (the screen differs from before the "
    "last step), `wait <ms>`. For a command line: `run <command>`, "
    "`expect [not] <text>` (in what it printed), `exit = N`, `request GET "
    "<url>`, `status = N`. For a terminal: `type <text>`, `press <key>`, "
    "`expect [not] <text>` (on screen), `screen` (put the screen in the "
    "report), `exit = N`, `wait <ms>`"
)

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1", "0.0.0.0")


@dataclass(frozen=True)
class Step:
    verb: str
    argument: str
    line: str


@dataclass(frozen=True)
class Walk:
    #: "page", "shell" or "tty".
    backend: str
    steps: tuple[Step, ...]

    @property
    def lines(self) -> list[str]:
        return [s.line for s in self.steps]


def _is_loopback_http(url: str) -> bool:
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return False
    host = (parts.hostname or "").lower()
    return parts.scheme == "http" and (host in LOOPBACK_HOSTS or f"[{host}]" in LOOPBACK_HOSTS)


def unquote(argument: str) -> str:
    """`expect "[]"` means the two characters, not four. A model quotes what
    it expects the way it would in code, and the quotes then have to be on
    the screen -- measured live, four failed steps in one run for that."""
    for mark in ('"', "'"):
        if len(argument) >= 2 and argument.startswith(mark) and argument.endswith(mark):
            inner = argument[1:-1]
            return inner.replace('\\"', '"') if mark == '"' else inner
    for prefix in ("not ",):
        if argument.lower().startswith(prefix):
            rest = argument[len(prefix):].strip()
            stripped = unquote(rest)
            if stripped != rest:
                return argument[:len(prefix)] + stripped
    return argument


def parse_walk(body: str) -> Walk | str:
    """The walkthrough, or why the script is not one.

    Checked here, before any browser or shell starts: a typo in a verb should
    cost a tool result that names the vocabulary, not a browser launch.
    """
    steps: list[Step] = []
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        verb, _, rest = line.partition(" ")
        steps.append(Step(verb.lower(), unquote(rest.strip()), line))
    if not steps:
        return f"an empty walkthrough -- {WALK_HELP}"
    if len(steps) > MAX_WALK_STEPS:
        return f"{len(steps)} steps is too many for one walkthrough -- at most {MAX_WALK_STEPS}"

    first = steps[0]
    if first.verb == "serve":
        if len(steps) < 2 or steps[1].verb not in ("open", "request", "run"):
            return ("after `serve <command>`, say what to do with the server: "
                    "`open http://127.0.0.1:PORT/` for its page, or `request GET "
                    "http://127.0.0.1:PORT/...` for its API")
        backend = "page" if steps[1].verb == "open" else "shell"
    elif first.verb == "open":
        backend = "page"
    elif first.verb == "run":
        backend = "shell"
    elif first.verb == "tty":
        backend = "tty"
    elif first.verb in DEVICE_KINDS:
        backend = first.verb
    else:
        return (f"{first.verb!r} cannot start a walkthrough -- the first step is "
                f"`open`, `serve`, `run`, `tty`, or one of {', '.join(DEVICE_KINDS)}. {WALK_HELP}")

    allowed = {"page": PAGE_STEPS, "shell": SHELL_STEPS, "tty": TTY_STEPS}.get(
        backend, (backend, *DEVICE_STEPS))
    for n, step in enumerate(steps):
        if step.verb in DEVICE_KINDS and n != 0:
            # "build it, then use it" in one script, three times in one live
            # run. The build is a shell matter; the walkthrough starts at
            # the launch.
            return (f"`{step.verb}` starts a walkthrough of its own: build with execute_bash "
                    f"(or a `run` walkthrough) first, then a separate `exercise` whose first "
                    f"line is `{step.verb} {step.argument or '<app>'}`")
        if step.verb not in allowed:
            # Short and specific. The full help here sent a model back with
            # the same `exit` in a page walkthrough three times running.
            others = ", ".join(v for v in allowed if v not in ("open", "serve", "run", "tty", *DEVICE_KINDS))
            article = "an" if backend[0] in "aeiou" else "a"
            return (f"`{step.verb}` is not {article} {backend} step. After the first line, "
                    f"{article} {backend} walkthrough's steps are: {others}")
        if step.verb in ("serve", "tty") and n != 0:
            return f"`{step.verb}` is the first step and only the first"
        if step.verb == "open" and n > 1:
            return "one page per walkthrough -- `open` comes first, or right after `serve`"
        if step.verb == "open" and n == 1 and first.verb != "serve":
            return "one page per walkthrough -- `open` is the first step and only the first"
        if step.verb not in ("wait", "changed", "screen") and not step.argument:
            return f"`{step.verb}` needs something after it -- {WALK_HELP}"
        if step.verb == "wait" and not step.argument.isdigit():
            return "`wait` takes a number of milliseconds"
        if step.verb in ("exit", "status"):
            _, _, want = step.argument.partition("=")
            if not want.strip().lstrip("-").isdigit():
                return f"`{step.verb}` is written `{step.verb} = N`"
            if step.verb == "exit" and backend == "page" and want.strip() != "0":
                # A page has no exit code. `exit = 0` is accepted as "the
                # page threw nothing", because models keep writing it to
                # mean "and that is the end" -- three runs in a row.
                return "a page has no exit code; `exit = 0` means it threw nothing, and is the only form"
        if step.verb == "count" and "=" not in step.argument:
            return "`count` is written `count <selector> = N`"
        if step.verb == "request":
            method, url, _ = split_request(step.argument)
            if not url or "://" not in url:
                return "`request` is written `request GET <url>`, or `request POST <url> <json body>`"
            if not _is_loopback_http(url):
                # The agent's own server is the point of this backend. A
                # remote API is `execute_bash` and curl, where check_url and
                # the third-party framing already apply.
                return "`request` reaches the server this walkthrough started: a loopback http URL"
    if first.verb == "open" and "://" in first.argument:
        return ("`open` at the start takes a workspace path. To open a URL, start a "
                "server for it: `serve <command>` then `open http://127.0.0.1:PORT/`. "
                "A live site is `browse` and `browse_act`")
    if first.verb == "serve" and steps[1].verb == "open" and not _is_loopback_http(steps[1].argument):
        return ("after `serve`, `open` takes the loopback URL the server answers on, "
                "like `open http://127.0.0.1:5173/`")
    return Walk(backend, tuple(steps))


def split_request(argument: str) -> tuple[str, str, str]:
    """`(method, url, body)` from a request step: `POST <url> {"name": "x"}`.
    Everything after the URL is the body, sent as JSON."""
    method, _, rest = argument.partition(" ")
    url, _, body = rest.strip().partition(" ")
    return method.strip().upper(), url.strip(), body.strip()


def serve_port(walk: Walk) -> int | None:
    """The port a `serve` walkthrough waits for, read off the step after it."""
    if walk.steps[0].verb != "serve" or len(walk.steps) < 2:
        return None
    argument = walk.steps[1].argument
    url = split_request(argument)[1] if walk.steps[1].verb == "request" else argument
    try:
        return urlsplit(url.strip()).port or 80
    except ValueError:
        return None


def failed_step(report: str) -> str:
    """The line of a walkthrough report that failed, or "" if none did."""
    for line in report.splitlines():
        if "-> FAILED: " in line:
            return line.strip()
    return ""


def summary_line(steps: list[str], report: list[str], failed: bool) -> str:
    """The first line of every report: the one the action record keeps."""
    passed = len(report) - (1 if failed else 0)
    shown = "; ".join(s[:40] for s in steps[:12]) + (" ..." if len(steps) > 12 else "")
    # A walkthrough with only the opening step proved the thing starts and
    # nothing else, and the judge should see that at a glance: live, a model
    # launched its AppKit app with `mac ./counter` and no step after it, and
    # the judge took "1/1 steps passed" for the buttons working.
    weak = " (launch only -- nothing was clicked, typed or checked)" if len(steps) == 1 else ""
    return f"{passed}/{len(steps)} steps passed{weak}: {shown}"


# --------------------------------------------------------------------------
# The shell backend: commands, what they printed, and a server if asked
# --------------------------------------------------------------------------

#: How long one `run` step may take. Generous: a walkthrough of a CLI may
#: build something on the way.
RUN_TIMEOUT_S = 120.0


class LocalShell:
    """Commands run here, in the workspace. A server started with `serve`
    runs in its own process group so it can be stopped with everything it
    spawned, and its output is kept for the report if it never comes up."""

    def __init__(self, cwd: str):
        self.cwd = cwd
        self.server: subprocess.Popen | None = None
        self.server_log = None

    def run(self, command: str, timeout: float = RUN_TIMEOUT_S) -> tuple[str, int]:
        try:
            proc = subprocess.run(
                command, shell=True, cwd=self.cwd, capture_output=True, text=True,
                errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            # A command that never returns is usually a server or a windowed
            # app -- measured live, a model `run` its own AppKit binary, waited
            # the whole two minutes, then bolted a self-test mode onto the app
            # rather than reach for the kind that opens windows.
            return (f"[timed out after {int(timeout)}s -- if this starts a server, use "
                    "`serve <cmd>`; if it opens a window, use `mac|linux|windows <cmd>`]"), -1
        return (proc.stdout + proc.stderr), proc.returncode

    def serve(self, command: str) -> None:
        self.server_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self.server = subprocess.Popen(
            command, shell=True, cwd=self.cwd, stdout=self.server_log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
        )

    def server_alive(self) -> bool:
        return self.server is not None and self.server.poll() is None

    def server_output(self) -> str:
        if self.server_log is None:
            return ""
        self.server_log.seek(0)
        return self.server_log.read()[-2000:]

    def request(self, method: str, url: str, body: str = "", timeout: float = 15.0) -> tuple[int, str]:
        import urllib.error
        import urllib.request

        data = body.encode("utf-8") if body else None
        headers = {"Content-Type": "application/json"} if body else {}
        req = urllib.request.Request(url, method=method.upper(), data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read(65536).decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read(65536).decode("utf-8", "replace")

    def close(self) -> None:
        if self.server is not None and self.server.poll() is None:
            stop_process_group(self.server)
        if self.server_log is not None:
            self.server_log.close()


class RemoteShell:
    """The same, through a bound command runner (a container). A server is
    detached with nohup and its pid kept, because the runner is one-shot:
    a command that has not exited cannot return."""

    def __init__(self, runner):
        self.runner = runner
        self.pid = ""
        self.log = "/tmp/otto-walk-server.log"

    def run(self, command: str, timeout: float = RUN_TIMEOUT_S) -> tuple[str, int]:
        out, err, code = self.runner(command, timeout)
        return out + err, code

    def serve(self, command: str) -> None:
        out, _, _ = self.runner(
            f"nohup sh -c {_quote(command)} > {self.log} 2>&1 < /dev/null & echo $!", 10.0,
        )
        self.pid = out.strip().splitlines()[-1] if out.strip() else ""

    def server_alive(self) -> bool:
        if not self.pid:
            return False
        _, _, code = self.runner(f"kill -0 {self.pid}", 5.0)
        return code == 0

    def server_output(self) -> str:
        out, _, _ = self.runner(f"tail -c 2000 {self.log} 2>/dev/null", 5.0)
        return out

    def request(self, method: str, url: str, body: str = "", timeout: float = 15.0) -> tuple[int, str]:
        payload = f" -H 'Content-Type: application/json' -d {_quote(body)}" if body else ""
        out, err, code = self.runner(
            f"curl -s -X {method.upper()}{payload} -w '\\n__OTTO_STATUS__%{{http_code}}' {_quote(url)}",
            timeout,
        )
        if code != 0:
            raise RuntimeError((err or out).strip()[:200] or f"curl exited {code}")
        body, _, status = out.rpartition("\n__OTTO_STATUS__")
        return int(status.strip() or 0), body

    def close(self) -> None:
        if self.pid:
            self.runner(f"kill -TERM -- -{self.pid} 2>/dev/null || kill {self.pid} 2>/dev/null", 5.0)


def stop_process_group(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Stop a process and everything it spawned. A process group on POSIX;
    Windows has no `killpg`, so there it is the process itself."""
    if os.name == "nt":
        # terminate() stops the shell and leaves what it spawned listening;
        # taskkill /T takes the tree. Measured on CI, where the served port
        # was still answering after the walkthrough.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
        return
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _quote(text: str) -> str:
    import shlex

    return shlex.quote(text)


def _wait_for_port(port: int, shell, seconds: float = SERVE_TIMEOUT_S) -> None:
    import socket

    deadline = time.time() + seconds
    while time.time() < deadline:
        if not shell.server_alive():
            raise RuntimeError(
                f"the server exited before opening port {port}; its output ended: "
                + shell.server_output().strip()[-400:]
            )
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.25)
    raise RuntimeError(
        f"port {port} did not open within {int(seconds)}s; the server's output ended: "
        + shell.server_output().strip()[-400:]
    )


def run_shell(walk: Walk, shell, *, wait_for_port=None) -> tuple[str, bool]:
    """A shell walkthrough, to completion or the first failing step.

    Returns `(report, failed)`. The report's first line is the summary; each
    step follows with what it produced. A `run` whose command exits non-zero
    fails the step unless the very next step is `exit = N`, so a walkthrough
    that means to test error handling says so and one that does not cannot
    walk past a crash.
    """
    # Looked up at call time, not bound at definition, so a test can stand
    # in a port wait without a real server.
    wait_for_port = wait_for_port or _wait_for_port
    steps = walk.steps
    report: list[str] = []
    failed = False
    last_output, last_exit, last_status = "", 0, None
    def expects_exit(after: int) -> bool:
        """Whether an `exit = N` follows this command before the next one.
        Anywhere before it, not only the next line: a model that writes
        `expect <message>` first and `exit = 2` after it means the same
        thing, and was sent back ten times running for the order."""
        for later in steps[after:]:
            if later.verb in ("run", "request", "serve"):
                return False
            if later.verb == "exit":
                return True
        return False

    try:
        for n, step in enumerate(steps, 1):
            try:
                if step.verb == "serve":
                    shell.serve(step.argument)
                    port = serve_port(walk)
                    if port is not None:
                        wait_for_port(port, shell)
                    report.append(f"{n}. {step.line} -> ok (port {port} is answering)")
                    continue
                if step.verb == "run":
                    last_output, last_exit = shell.run(step.argument)
                    head = next((l for l in last_output.splitlines() if l.strip()), "")
                    if last_exit != 0 and not expects_exit(n):
                        raise RuntimeError(
                            f"exited {last_exit} -- {head[:160]!r}; add `exit = {last_exit}` "
                            "after this step if that is expected"
                        )
                    report.append(f"{n}. {step.line} -> ok (exit {last_exit}): {head[:120]}")
                    continue
                if step.verb == "request":
                    method, url, body = split_request(step.argument)
                    last_status, last_output = shell.request(method, url, body)
                    head = next((l for l in last_output.splitlines() if l.strip()), "")
                    report.append(f"{n}. {step.line} -> ok ({last_status}): {head[:120]}")
                    continue
                if step.verb == "expect":
                    absent = step.argument.lower().startswith("not ")
                    what = step.argument[4:].strip() if absent else step.argument
                    present = what in last_output
                    if present == absent:
                        seen = last_output.strip().replace("\n", " | ")[:200]
                        raise RuntimeError(
                            ("still in" if absent else "not in") + f" the last output: {seen!r}"
                        )
                    report.append(f"{n}. {step.line} -> ok")
                    continue
                if step.verb == "exit":
                    want = int(step.argument.partition("=")[2])
                    if last_exit != want:
                        raise RuntimeError(f"the last command exited {last_exit}")
                    report.append(f"{n}. {step.line} -> ok")
                    continue
                if step.verb == "status":
                    want = int(step.argument.partition("=")[2])
                    if last_status != want:
                        raise RuntimeError(f"the last request returned {last_status}")
                    report.append(f"{n}. {step.line} -> ok")
                    continue
                if step.verb == "wait":
                    time.sleep(min(int(step.argument), 5000) / 1000)
                    report.append(f"{n}. {step.line} -> ok")
                    continue
                raise RuntimeError("not a step this knows")
            except Exception as exc:  # one step's failure is the report's, not the tool's
                reason = (str(exc).strip().splitlines() or ["failed"])[0][:300]
                report.append(f"{n}. {step.line} -> FAILED: {reason}")
                failed = True
                break
    finally:
        shell.close()
    lines = [summary_line(walk.lines, report, failed), *("  " + r for r in report)]
    return "\n".join(lines) + "\n", failed


# --------------------------------------------------------------------------
# The terminal backend: a program in a pseudo-terminal, and its screen
# --------------------------------------------------------------------------

#: Runs in whichever interpreter has `pyte`. Argument 1 is the script, 2 the
#: limits (cols, rows). Prints the summary, the steps, then `screen:` and the
#: final screen; exits 6 when a step failed and 3 when pyte is missing.
PTY_DRIVER = r'''
import fcntl, json, os, pty, select, signal, struct, subprocess, sys, termios, time
try:
    import pyte
except ImportError:
    print("this interpreter has no pyte installed", file=sys.stderr); sys.exit(3)
script, limits = sys.argv[1], json.loads(sys.argv[2])
cols, rows = int(limits.get("cols", 100)), int(limits.get("rows", 30))
KEYS = {"enter": "\r", "return": "\r", "tab": "\t", "escape": "\x1b", "esc": "\x1b",
        "backspace": "\x7f", "space": " ", "up": "\x1b[A", "down": "\x1b[B",
        "right": "\x1b[C", "left": "\x1b[D", "home": "\x1b[H", "end": "\x1b[F",
        "pageup": "\x1b[5~", "pagedown": "\x1b[6~", "delete": "\x1b[3~", "insert": "\x1b[2~",
        "f1": "\x1bOP", "f2": "\x1bOQ", "f3": "\x1bOR", "f4": "\x1bOS", "f5": "\x1b[15~",
        "f6": "\x1b[17~", "f7": "\x1b[18~", "f8": "\x1b[19~", "f9": "\x1b[20~",
        "f10": "\x1b[21~", "f11": "\x1b[23~", "f12": "\x1b[24~"}
def key_bytes(name):
    n = name.strip().lower()
    if n.startswith("ctrl+") and len(n) == 6:
        return chr(ord(n[5]) & 0x1f).encode()
    if n.startswith("alt+") and len(n) == 5:
        return b"\x1b" + n[4].encode()
    if n in KEYS:
        return KEYS[n].encode()
    if len(name) == 1:
        return name.encode()
    raise Exception("not a key this knows: " + name)
screen = pyte.Screen(cols, rows)
stream = pyte.ByteStream(screen)
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))
env = dict(os.environ, TERM="xterm-256color", COLUMNS=str(cols), LINES=str(rows))
proc = None
# Application cursor mode (DECCKM). curses' keypad(True) switches the
# terminal into it, and then expects an arrow as ESC O A, not ESC [ A -- a
# real terminal follows the switch, so this does too. Sent the wrong form,
# curses reads a bare ESC and the arrow never arrives; measured live.
state = {"app": False}
CURSOR = {"\x1b[A", "\x1b[B", "\x1b[C", "\x1b[D", "\x1b[H", "\x1b[F"}
def send(text):
    if state["app"] and text in CURSOR:
        text = "\x1bO" + text[-1]
    os.write(master, text.encode("latin-1", "replace"))
def pump(seconds):
    end = time.time() + seconds
    while True:
        left = end - time.time()
        if left <= 0:
            return
        r, _, _ = select.select([master], [], [], min(left, 0.05))
        if not r:
            continue
        try:
            data = os.read(master, 65536)
        except OSError:
            return
        if not data:
            return
        if b"\x1b[?1h" in data:
            state["app"] = True
        if b"\x1b[?1l" in data:
            state["app"] = False
        stream.feed(data)
def text():
    return "\n".join(line.rstrip() for line in screen.display).rstrip()
def shown():
    """The screen for the report, with a run of empty rows folded to one --
    a full-screen app is mostly empty rows between a header and a footer."""
    out, blank = [], False
    for line in text().splitlines():
        if line.strip():
            out.append(line); blank = False
        elif not blank:
            out.append("..."); blank = True
    return out[:rows]
def until(pred, seconds):
    end = time.time() + seconds
    while time.time() < end:
        pump(0.1)
        if pred():
            return True
    return pred()
steps = [l.strip() for l in script.splitlines() if l.strip()]
report, failed = [], ""
for n, line in enumerate(steps, 1):
    verb, _, rest = line.partition(" ")
    verb, rest = verb.lower(), rest.strip()
    try:
        if verb == "tty":
            proc = subprocess.Popen(rest, shell=True, stdin=slave, stdout=slave, stderr=slave,
                                    env=env, preexec_fn=os.setsid, close_fds=True)
            pump(0.6)
        elif proc is None:
            raise Exception("no program is running -- `tty <command>` comes first")
        elif verb == "type":
            # `type \x1b[A` means the arrow, not six characters. Decoding
            # here is what keeps a model from teaching its own app to read
            # a literal backslash as a key -- which one did, live.
            text_out = rest
            if "\\" in rest:
                text_out = rest.encode("latin-1", "backslashreplace").decode("unicode_escape")
            send(text_out)
            pump(0.2)
        elif verb == "press":
            send(key_bytes(rest).decode("latin-1"))
            pump(0.3)
        elif verb == "wait":
            pump(min(int(rest), 5000) / 1000)
        elif verb == "expect":
            absent = rest.lower().startswith("not ")
            what = rest[4:].strip() if absent else rest
            ok = until(lambda: (what in text()) != absent, 5.0)
            if not ok:
                raise Exception(("still on screen" if absent else "not on screen") + " after 5s")
        elif verb == "screen":
            pump(0.2)
            report.append(f"{n}. {line} -> ok")
            report.extend("   | " + l for l in shown())
            continue
        elif verb == "exit":
            want = int(rest.partition("=")[2])
            until(lambda: proc.poll() is not None, 10.0)
            if proc.poll() is None:
                raise Exception("still running after 10s")
            if proc.returncode != want:
                raise Exception(f"exited {proc.returncode}")
        else:
            raise Exception("not a step this knows")
        report.append(f"{n}. {line} -> ok")
    except Exception as exc:
        reason = (str(exc).strip().splitlines() or ["failed"])[0][:200]
        report.append(f"{n}. {line} -> FAILED: {reason}")
        failed = f"step {n}"
        break
pump(0.2)
final = text()
if proc is not None and proc.poll() is None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        until(lambda: proc.poll() is not None, 2.0)
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
passed = len([r for r in report if " -> ok" in r])
listed = "; ".join(s[:40] for s in steps[:12]) + (" ..." if len(steps) > 12 else "")
print(f"{passed}/{len(steps)} steps passed: {listed}")
for r in report:
    print("  " + r)
print("screen:")
for l in shown():
    print("  " + l)
sys.exit(6 if failed else 0)
'''

#: Sentinel for "not probed yet". `None` means probed and nothing found.
_UNPROBED = object()
_TTY: object = _UNPROBED

NO_TTY_DRIVER = (
    "no interpreter with a terminal emulator was found. To let otto drive a "
    "program in a terminal, install `pyte` in the Python OTTO_BROWSER_PYTHON "
    "names (`pip install pyte`)"
)


def tty_interpreter() -> str | None:
    """A Python here that can import `pyte`, or None. Probed once per process,
    like browsing.local_interpreter, and for the same reason."""
    global _TTY
    if _TTY is not _UNPROBED:
        return _TTY  # type: ignore[return-value]
    named = os.environ.get("OTTO_BROWSER_PYTHON", "").strip()
    found = None
    for candidate in [c for c in (named, sys.executable) if c]:
        try:
            probe = subprocess.run([candidate, "-c", "import pyte"], capture_output=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            found = candidate
            break
    _TTY = found
    return found


def forget_tty_interpreter() -> None:
    global _TTY
    _TTY = _UNPROBED


def run_tty_local(script: str, limits: str, *, cwd: str, timeout: float = 120.0) -> tuple[str, str, int]:
    """The terminal driver, run here."""
    interpreter = tty_interpreter()
    if interpreter is None:
        return "", NO_TTY_DRIVER, 3
    try:
        proc = subprocess.run(
            [interpreter, "-c", PTY_DRIVER, script, limits],
            capture_output=True, text=True, errors="replace", timeout=timeout, cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        return "", f"the program did not finish the walkthrough within {int(timeout)}s", -1
    except OSError as exc:
        return "", f"could not start the terminal driver: {exc}", 1
    return proc.stdout, proc.stderr, proc.returncode
