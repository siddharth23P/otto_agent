"""Driving a browser, through the container Otto is already working in.

Otto has had no browser at all. Adding one raised a question with an unexpected
answer: where does it run? Not here -- a browser is a 300MB dependency and a
process to supervise, and `agent/pipeline/` deliberately holds no vendor SDKs.
It runs in the SAME container the file and shell tools already reach through
`agent/pipeline/execution.py`'s command runner. Otto gains no dependency; it
gains a driver script it ships into a machine that already has Chromium.

Two things shape the design, and both are measured.

THE CODE PATH WINS, SO THIS IS THE FALLBACK. Agents that route each subtask to
code or GUI and PREFER code reach 60.76% on OSWorld in 10.15 steps against ~15
for GUI-only, a 32% reduction; a hybrid action space is worth +22% relative.
On the web the same shape holds harder: API plus browser scores 38.9% on
WebArena against browsing alone, **+24.0 absolute points**. GUI-only chains are
described as brittle and prone to cascading failure. So `execute_bash` and an
HTTP client remain the first answer whenever a site has one, and this is for
when it does not.

THE OBSERVATION IS THE LEVER. Refining only the observation and action space --
no planner, no critic, no tree search, no examples -- beat every scaffolding
trick tried against it: +9.8 points over the previous state of the art, +29.4%
relative. So what comes back from a page here is a short structured digest --
title, url, headings, links, form fields, visible text, clipped -- and never
raw DOM or a full text dump. A page is a few hundred characters to this agent,
not a few hundred thousand.

The operations are high-level for the same reason: a small vocabulary a model
can use correctly beats a faithful reproduction of a mouse. Reverse-engineering
a site's latent functionality into named operations is measured as raising
success while cutting steps.

HONEST LIMITATION. Each call drives a fresh page and restores cookies and
storage from disk, because a one-shot command cannot hold a live browser open
between calls. Navigation, reading, forms and login flows work. In-page
JavaScript state that never touches storage does not survive between calls, and
the digest says so rather than pretending. A task that needs a genuinely live
session wants a browser server in the container, which is a later job.

AND LOCALLY, WHEN THERE IS NO CONTAINER. An ordinary `otto chat` or `otto tui`
turn binds a workspace and no container, and until now that meant no browser at
all -- which is how a "fully playable chess game" shipped with a board that
never drew: a stray token in the script threw at load, `node --check` passed
(it is a valid identifier), the run's own tests exercised the move logic in
isolation, and the judge, with nothing that could load a page, approved it.
Three rounds of "the board is blank / the pieces are transparent" later, the
lesson otto distilled for itself was "load the actual output in a browser". So
the same DRIVER now also runs HERE, through whichever Python interpreter has
Playwright: `OTTO_BROWSER_PYTHON` if set, otherwise otto's own. Otto still
ships no browser dependency (tests/test_browsing.py holds that); a person who
wants otto to see pages installs Playwright once and points the variable at it.

What the driver reports changed with it: a page's UNCAUGHT ERRORS and
console errors now come back in the digest, and for a page in the workspace --
the agent's own work -- an error fails the call. That is what makes `browse
open index.html` a check in agent/pipeline/evidence.py's sense: a command that
would fail if the page were broken.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import tempfile
from pathlib import Path

#: Ceiling on the digest of one page. Generous enough for a dense article's
#: headings and links, far below a real DOM.
MAX_DIGEST_CHARS = 4000

#: How many of each kind of thing a digest names. A page with 400 links is a
#: navigation index, and listing all of them is how an observation stops being
#: an observation.
MAX_LINKS = 30
MAX_FIELDS = 20
MAX_HEADINGS = 15

#: What the agent may ask for. Deliberately small: a vocabulary a model uses
#: correctly beats a faithful reproduction of a mouse.
READ_OPS = ("open", "read", "find", "back")
ACT_OPS = ("click", "type", "submit")


#: Schemes `open` will follow. Everything else -- `file:`, `data:`, `ftp:`,
#: `chrome:` -- is refused before the browser sees it.
#:
#: `file:///etc/passwd` through a browser is a file read with extra steps, and
#: the digest comes back to the model as ordinary tool output.
ALLOWED_SCHEMES = ("http", "https")

#: Networks `open` will not reach.
#:
#: This is the SSRF half, and it matters more here than in an ordinary HTTP
#: client because of WHO chooses the URL. The agent picks it, and the agent
#: reads web pages -- so a page it has already opened can tell it to open
#: something else, which is indirect prompt injection with a network request
#: on the end of it. `169.254.169.254` is the cloud metadata endpoint on every
#: major provider and hands out credentials to anything that asks.
#:
#: Link-local, loopback and the three private ranges, plus the IPv6 forms.
#: Blocked by parsed address rather than by string matching, and the host is
#: NORMALISED first -- browsers accept `http://2130706433/` and
#: `http://0177.0.0.1/` as 127.0.0.1, and `ipaddress` does not, so without
#: that step both walk straight past this list. Measured on the first draft of
#: exactly this function.
BLOCKED_NETWORKS = (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "169.254.0.0/16", "0.0.0.0/8", "100.64.0.0/10",
    "::1/128", "fc00::/7", "fe80::/10",
)


#: Names that mean loopback or metadata without being addresses. Short on
#: purpose: this is not an attempt at a name blocklist, which cannot work,
#: only a refusal of the handful anybody actually types.
BLOCKED_HOSTS = frozenset({
    "localhost", "ip6-localhost", "ip6-loopback",
    "metadata.google.internal", "metadata.goog", "instance-data",
})


def _as_address(host: str):
    """`host` as an IP address, accepting the forms a browser accepts.

    `ipaddress` takes dotted quads and IPv6. A browser also takes a bare
    integer (`2130706433`) and octal or hex octets (`0177.0.0.1`,
    `0x7f.0.0.1`), all of which mean 127.0.0.1 -- so a check that only asks
    `ipaddress` lets every one of them through. Returns None when the host is
    a name rather than any form of address.
    """
    import ipaddress

    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    if host.isdigit():  # bare integer form
        try:
            return ipaddress.ip_address(int(host))
        except ValueError:
            return None
    parts = host.split(".")
    if len(parts) == 4:  # octal or hex octets
        try:
            octets = [int(p, 0) if p.lower().startswith("0x")
                      else int(p, 8) if p.startswith("0") and p != "0"
                      else int(p) for p in parts]
        except ValueError:
            return None
        if all(0 <= o <= 255 for o in octets):
            return ipaddress.ip_address(".".join(str(o) for o in octets))
    return None


def check_url(url: str) -> str:
    """Why this URL must not be opened, or "" if it may be.

    Refused BEFORE the driver script is built, so a rejected URL never reaches
    the container at all.

    Hostnames that are not literal addresses are allowed through: resolving
    them here would be a check against a different answer from the one the
    container's own resolver gives, and a name that resolves to a blocked
    address from inside the container is a DNS-rebinding problem this cannot
    honestly solve from out here. What it does stop is the direct form, which
    is the one that actually gets used.
    """
    import ipaddress
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        return f"that is not a URL ({exc})"

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        return (f"{parts.scheme or 'that'}: is not a scheme this opens -- "
                f"only {' and '.join(ALLOWED_SCHEMES)}")
    host = (parts.hostname or "").strip("[]")
    if not host:
        return "that URL names no host"
    if host.lower() in BLOCKED_HOSTS or host.lower().endswith(".localhost"):
        return f"{host} is a loopback or metadata name this will not open"

    address = _as_address(host)
    if address is None:
        return ""  # a name, not a literal address -- see the docstring
    # An IPv4 address wrapped in IPv6 (`::ffff:127.0.0.1`) is the v4 address.
    if getattr(address, "ipv4_mapped", None) is not None:
        address = address.ipv4_mapped
    for network in BLOCKED_NETWORKS:
        if address in ipaddress.ip_network(network):
            return (f"{host} is on a network this will not open ({network}) -- "
                    "loopback, private and cloud-metadata addresses are refused")
    return ""


def parse_op(body: str, allowed: tuple[str, ...]) -> tuple[str, str] | str:
    """`(operation, argument)` from a CODE: body, or why it is not one."""
    head, _, rest = body.strip().partition("\n")
    op, _, inline = head.strip().partition(" ")
    op = op.strip().lower()
    if not op:
        return f"say what to do: one of {', '.join(allowed)}"
    if op not in allowed:
        return f"{op!r} is not one of {', '.join(allowed)}"
    argument = (inline.strip() + ("\n" + rest if rest else "")).strip()
    return op, argument


#: Runs in the container. One operation, then a digest of whatever page the
#: browser is looking at afterwards.
#:
#: Session continuity is cookies and storage on disk rather than a live
#: process, so a login survives between calls and an un-persisted in-page
#: state does not -- see this module's docstring.
DRIVER = r'''
import json, os, sys
BASE = os.environ.get("OTTO_BROWSER_STATE_DIR") or "/tmp"
os.makedirs(BASE, exist_ok=True)
STATE = os.path.join(BASE, "otto-browser-state.json")
LAST = os.path.join(BASE, "otto-browser-url.txt")
op, arg = sys.argv[1], sys.argv[2]
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("this container has no playwright installed", file=sys.stderr); sys.exit(3)

def digest(page, limits):
    out = ["url: " + page.url, "title: " + (page.title() or "")]
    def grab(selector, cap, label, fmt):
        rows = []
        for el in page.query_selector_all(selector)[:cap]:
            try:
                row = fmt(el)
            except Exception:
                continue
            if row:
                rows.append(row)
        if rows:
            out.append(label + ":")
            out.extend("  " + r for r in rows)
    grab("h1,h2,h3", limits["headings"], "headings",
         lambda el: (el.inner_text() or "").strip()[:120])
    grab("a[href]", limits["links"], "links",
         lambda el: ((el.inner_text() or "").strip()[:80] + " -> " + (el.get_attribute("href") or "")[:120]))
    grab("input,textarea,select,button", limits["fields"], "fields and buttons",
         lambda el: (el.get_attribute("name") or el.get_attribute("aria-label")
                     or el.get_attribute("placeholder") or (el.inner_text() or "").strip()[:60]
                     or el.get_attribute("type") or "")[:80])
    body = (page.inner_text("body") or "").strip()
    out.append("text:")
    out.append("  " + body[:limits["chars"]].replace("\n", "\n  "))
    return "\n".join(out)

def find(page, spec):
    spec = spec.strip()
    if spec.lower().startswith("css "):
        return page.locator(spec[4:].strip())
    return page.get_by_text(spec, exact=False)

server = {"proc": None, "log": None}

def serve(command):
    """Start the app's own server for this walkthrough, in its own process
    group so it can be stopped with everything it spawned."""
    import subprocess, tempfile
    server["log"] = tempfile.TemporaryFile(mode="w+", errors="replace")
    server["proc"] = subprocess.Popen(command, shell=True, stdout=server["log"],
                                      stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                      start_new_session=True)

def server_output():
    if server["log"] is None:
        return ""
    server["log"].seek(0)
    return server["log"].read()[-400:].strip()

def wait_port(port, seconds):
    import socket, time
    end = time.time() + seconds
    while time.time() < end:
        if server["proc"].poll() is not None:
            raise Exception(f"the server exited ({server['proc'].returncode}) before opening port {port}; its output ended: " + server_output())
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.25)
    raise Exception(f"port {port} did not open within {int(seconds)}s; the server's output ended: " + server_output())

def stop_server():
    import signal
    proc = server["proc"]
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except Exception:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass

def walk(page, script, limits):
    """Every step of a walkthrough, in this one page. Stops at the first
    step that fails: the ones after it would be acting on a page in a state
    nobody asked for."""
    from urllib.parse import urlsplit
    steps = [l.strip() for l in script.splitlines() if l.strip()]
    report, failed = [], ""
    frames = []
    response = None  # (status, text) of the last request, until the next page step
    for n, line in enumerate(steps, 1):
        verb, _, rest = line.partition(" ")
        verb, rest = verb.lower(), rest.strip()
        try:
            if verb == "exit":
                # `exit = 0` on a page: it threw nothing so far.
                if errors:
                    raise Exception("the page threw: " + errors[0][:160])
                report.append(f"{n}. {line} -> ok (the page threw nothing)")
                continue
            if verb == "request":
                method, _, tail = rest.partition(" ")
                url, _, body = tail.strip().partition(" ")
                kwargs = {"method": method.upper()}
                if body.strip():
                    kwargs["data"] = body.strip()
                    kwargs["headers"] = {"Content-Type": "application/json"}
                got = page.request.fetch(url.strip(), **kwargs)
                response = (got.status, got.text())
                head = next((l for l in response[1].splitlines() if l.strip()), "")
                report.append(f"{n}. {line} -> ok ({response[0]}): {head[:120]}")
                continue
            if verb == "status":
                if response is None:
                    raise Exception("no request has been made yet")
                want = int(rest.partition("=")[2])
                if response[0] != want:
                    raise Exception(f"the last request returned {response[0]}")
                report.append(f"{n}. {line} -> ok")
                continue
            if verb == "expect" and response is not None:
                absent = rest.lower().startswith("not ")
                what = rest[4:].strip() if absent else rest
                if (what in response[1]) == absent:
                    seen = response[1].strip().replace("\n", " | ")[:200]
                    raise Exception(("still in" if absent else "not in") + f" the last response: {seen!r}")
                report.append(f"{n}. {line} -> ok")
                continue
            response = None
            if verb == "serve":
                serve(rest)
                nxt = steps[n].partition(" ")[2].strip() if n < len(steps) else ""
                port = urlsplit(nxt).port or 80
                wait_port(port, limits.get("serve_timeout", 60))
                report.append(f"{n}. {line} -> ok (port {port} is answering)")
                continue
            if verb == "open":
                url = rest if "://" in rest else "file://" + os.path.abspath(rest)
                page.goto(url, wait_until="load", timeout=30000)
            elif verb == "changed":
                # The frame after the most recent ACTION -- a click, a key, a
                # typed text -- against the one before it. Not the previous
                # step: an `expect` in between changes nothing, and a model
                # writes `click`, `expect`, `changed` in that order. For a
                # canvas -- a game with no DOM to count -- this is the honest
                # cheap signal that a key did something.
                if len(frames) < 2:
                    raise Exception("nothing to compare yet -- `changed` follows a click, type or press")
                if frames[-1] == frames[-2]:
                    raise Exception("the screen is identical to before the last click, type or press")
                report.append(f"{n}. {line} -> ok")
                continue
            elif verb == "click":
                find(page, rest).first.click(timeout=5000)
                page.wait_for_timeout(150)
            elif verb == "type":
                field, _, value = rest.partition("=")
                page.get_by_label(field.strip()).first.fill(value.strip(), timeout=5000)
            elif verb == "press":
                page.keyboard.press(rest)
                page.wait_for_timeout(150)
            elif verb == "wait":
                page.wait_for_timeout(min(int(rest), 5000))
            elif verb == "expect":
                absent = rest.lower().startswith("not ")
                what = rest[4:] if absent else rest
                find(page, what).first.wait_for(
                    state="hidden" if absent else "visible", timeout=3000)
            elif verb == "count":
                sel, _, want = rest.rpartition("=")
                sel = sel.strip()
                sel = sel[4:].strip() if sel.lower().startswith("css ") else sel
                got = page.locator(sel).count()
                if got != int(want):
                    raise Exception(f"found {got}, expected {want.strip()}")
            else:
                raise Exception("not a step this knows")
            if verb in ("open", "click", "type", "press"):
                frames.append(page.screenshot(full_page=False))
            report.append(f"{n}. {line} -> ok")
        except Exception as exc:
            reason = (str(exc).strip().splitlines() or ["failed"])[0][:200]
            report.append(f"{n}. {line} -> FAILED: {reason}")
            failed = f"step {n} `{line}` -- {reason}"
            break
    return steps, report, failed

limits = json.loads(sys.argv[3])
# What the page THREW, kept apart from what it shows. An uncaught exception or
# a console.error is the single cheapest signal that a page is broken, and a
# digest that only carries the visible text hides it completely: a script that
# dies before it draws leaves a page that reads as merely empty.
errors = []
walked = None
with sync_playwright() as p:
    browser = p.chromium.launch(args=["--no-sandbox"])
    ctx = browser.new_context(storage_state=STATE if os.path.exists(STATE) else None)
    page = ctx.new_page()
    page.on("pageerror", lambda exc: errors.append("uncaught " + str(exc).strip().splitlines()[0][:300]))
    page.on("console", lambda msg: errors.append("console.error " + msg.text[:300])
            if msg.type == "error" else None)
    start = open(LAST).read().strip() if os.path.exists(LAST) else ""
    try:
        if op == "walk":
            walked = walk(page, arg, limits)
        elif op == "open":
            page.goto(arg, wait_until="domcontentloaded", timeout=30000)
        else:
            if not start:
                print("no page is open yet -- use `open <url>` first", file=sys.stderr); sys.exit(4)
            page.goto(start, wait_until="domcontentloaded", timeout=30000)
            if op == "click":
                page.get_by_text(arg, exact=False).first.click(timeout=15000)
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            elif op == "type":
                field, _, value = arg.partition("=")
                page.get_by_label(field.strip()).first.fill(value.strip(), timeout=15000)
            elif op == "submit":
                page.keyboard.press("Enter")
                page.wait_for_load_state("domcontentloaded", timeout=15000)
            elif op == "find":
                hits = page.get_by_text(arg, exact=False).all()[:10]
                print("matches: " + str(len(hits)))
                for h in hits:
                    print("  " + (h.inner_text() or "").strip()[:120])
            elif op == "back":
                page.go_back(wait_until="domcontentloaded", timeout=15000)
    except Exception as exc:
        stop_server()
        print(str(exc)[:400], file=sys.stderr); sys.exit(5)
    # A beat for whatever the page scheduled -- a setTimeout, a fetch, a
    # first frame -- so an error raised just after load is seen too.
    page.wait_for_timeout(250)
    open(LAST, "w").write(page.url)
    ctx.storage_state(path=STATE)
    if limits.get("screenshot"):
        try:
            page.screenshot(path=limits["screenshot"], full_page=False)
        except Exception as exc:
            errors.append("screenshot failed: " + str(exc)[:200])
    if walked is not None:
        # The summary FIRST: it is the one line the action record keeps.
        steps, report, failed = walked
        passed = len(report) - (1 if failed else 0)
        weak = " (launch only -- nothing was clicked, typed or checked)" if len(steps) == 1 else ""
        print(f"{passed}/{len(steps)} steps passed{weak}: "
              + "; ".join(s[:40] for s in steps[:12]) + (" ..." if len(steps) > 12 else ""))
        for line in report:
            print("  " + line)
    print(digest(page, limits))
    if errors:
        print("page errors:")
        for line in errors[:limits.get("errors", 10)]:
            print("  " + line)
    browser.close()
    stop_server()
    if walked is not None and walked[2]:
        sys.exit(6)
'''

# Walkthroughs -- `exercise`, one step per line in ONE page -- are parsed and
# shaped in agent/pipeline/walkthrough.py; the browser half of them is the
# `walk` op of DRIVER above.

#: The line the driver prints above whatever the page threw. Spelled out in
#: the driver too, since that text is a script and not this module;
#: tests/test_browsing.py holds the two the same.
ERRORS_HEADING = "page errors:"

#: How many of a page's errors the digest carries. A page that throws on
#: every frame would otherwise be the whole observation.
MAX_ERRORS = 10


def page_errors(digest: str) -> list[str]:
    """What the page threw, out of a driver digest. Empty for a clean page."""
    lines = digest.splitlines()
    try:
        start = lines.index(ERRORS_HEADING)
    except ValueError:
        return []
    found = []
    for line in lines[start + 1:]:
        if not line.startswith("  "):
            break
        found.append(line.strip())
    return found


# --------------------------------------------------------------------------
# Running the driver here, when no container is bound
# --------------------------------------------------------------------------

#: Sentinel for "not probed yet". `None` means probed and nothing found.
_UNPROBED = object()
_LOCAL: object = _UNPROBED

#: What a turn is told when it has a workspace, no container, and no local
#: interpreter with Playwright. Names the fix rather than only the absence.
NO_LOCAL_BROWSER = (
    "no container is bound for this run and no local browser was found. To "
    "let otto load pages here, install Playwright in some Python and point "
    "OTTO_BROWSER_PYTHON at it: `pip install playwright && playwright install "
    "chromium-headless-shell`"
)


def _candidates() -> list[str]:
    """Interpreters that might have Playwright, most deliberate first."""
    named = os.environ.get("OTTO_BROWSER_PYTHON", "").strip()
    return [c for c in (named, sys.executable) if c]


def local_interpreter() -> str | None:
    """A Python interpreter on this machine that can import Playwright, or
    None. Probed once per process -- a subprocess per candidate, and the
    answer does not change mid-run -- so `reachable_tools()` can ask on every
    prompt composition for free.
    """
    global _LOCAL
    if _LOCAL is not _UNPROBED:
        return _LOCAL  # type: ignore[return-value]
    found = None
    for candidate in _candidates():
        try:
            probe = subprocess.run(
                [candidate, "-c", "import playwright.sync_api"],
                capture_output=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if probe.returncode == 0:
            found = candidate
            break
    _LOCAL = found
    return found


def forget_local_interpreter() -> None:
    """Drop the cached probe -- for tests, and for a setup screen that just
    installed something."""
    global _LOCAL
    _LOCAL = _UNPROBED


def state_dir(workspace: Path | str) -> Path:
    """Where the local driver keeps cookies, the last URL and the last
    screenshot for this workspace. Outside the workspace on purpose: a
    `.otto-browser/` appearing in the repository under edit would show up in
    `git status` and in the agent's own next listing.
    """
    key = hashlib.sha1(str(Path(workspace).resolve()).encode("utf-8", "replace")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"otto-browser-{key}"


def last_screenshot(workspace: Path | str) -> Path:
    """The screenshot the last local `browse` call left, whether or not it
    exists yet."""
    return state_dir(workspace) / "otto-browser-last.png"


def run_local(op: str, argument: str, limits: str, *, workspace: Path | str,
              timeout: float = 90.0) -> tuple[str, str, int]:
    """The driver, run here. Same contract as a command runner's return:
    `(stdout, stderr, returncode)`."""
    interpreter = local_interpreter()
    if interpreter is None:
        return "", NO_LOCAL_BROWSER, 3
    where = state_dir(workspace)
    where.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OTTO_BROWSER_STATE_DIR": str(where)}
    try:
        proc = subprocess.run(
            [interpreter, "-c", DRIVER, op, argument, limits],
            capture_output=True, text=True, errors="replace", timeout=timeout, env=env,
            # In the workspace, so a `serve npm run dev` step starts the
            # app that is actually there.
            cwd=str(workspace),
        )
    except subprocess.TimeoutExpired:
        return "", f"the browser did not finish `{op}` within {int(timeout)}s", -1
    except OSError as exc:
        return "", f"could not start the browser: {exc}", 1
    return proc.stdout, proc.stderr, proc.returncode
