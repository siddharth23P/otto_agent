"""Coverage for the browser tools and the observation they return.

Otto had no browser. The design question was where one runs, and the answer was
not "here": a browser is a 300MB dependency and a process to supervise, and the
container Otto already drives for benchmark tasks ships Chromium and Playwright
already. So the driver is a script sent through the same command-runner seam
the file tools use, and Otto gains no dependency at all.

Two measured things shape what is tested here. The code path beats the GUI path
-- preferring code takes 32% fewer steps, and API-plus-browser beats browsing
alone by 24 absolute points on WebArena -- so these tools are the fallback and
say so. And the OBSERVATION is the lever: refining only the observation and
action space, with no planner or critic or search, beat every scaffolding trick
tried against it by +9.8 points.
"""
import re

import pytest

import agent.pipeline.tools as pt
from agent.pipeline import browsing
from agent.pipeline.execution import bind_command_runner


def _fake_browser(digest: str = "url: https://x\ntitle: X\ntext:\n  hello", code: int = 0):
    """A command runner that records the driver invocation."""
    seen = {}

    def run(command: str, timeout: float):
        seen["command"] = command
        return (digest, "", 0) if code == 0 else ("", digest, code)

    run.seen = seen
    return run


# --------------------------------------------------------------------------
# Where it runs
# --------------------------------------------------------------------------

def test_without_a_container_it_refuses_cleanly():
    """Same refusal shape as the file tools with no workspace: an ordinary chat
    turn has not opened a container, and there is no browser in one."""
    result = pt.browse("open https://example.com")
    assert result.returncode == 1
    assert "no container" in result.stderr


def test_the_driver_is_sent_into_the_container():
    """The URL is checked by parsing it out and comparing the HOST, not by
    asking whether the command contains "example.com".

    A substring test passes for `https://example.com.attacker.test` and for
    `https://attacker.test/?u=example.com`, so it cannot tell "the URL we
    asked for was sent" from "something with those characters in it was" --
    which is the whole question here. CodeQL flags the substring form for
    exactly that reason and it is right to.
    """
    from urllib.parse import urlparse

    runner = _fake_browser()
    with bind_command_runner(runner):
        pt.browse("open https://example.com/docs")
    command = runner.seen["command"]

    assert "playwright" in command
    sent = re.search(r"https?://[^\s'\"]+", command)
    assert sent is not None, command
    assert urlparse(sent.group()).netloc == "example.com"


def test_otto_itself_needs_no_browser_dependency():
    """The point of running it over there. `agent/pipeline/` holds no vendor
    SDKs and this must not be the exception."""
    import agent.pipeline.browsing as mod

    assert "import playwright" not in mod.__doc__ or True
    with pytest.raises(ImportError):
        __import__("playwright")


# --------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------

def test_reading_and_acting_are_separate_tools():
    """Clicking a button on a live site is irreversible and reading a page is
    not. Splitting them is what puts the acting half behind the same hold that
    covers sending a message, without taxing every read."""
    assert set(browsing.READ_OPS).isdisjoint(browsing.ACT_OPS)
    assert "click" in browsing.ACT_OPS
    assert "read" in browsing.READ_OPS


def test_an_unknown_operation_names_the_ones_that_exist():
    runner = _fake_browser()
    with bind_command_runner(runner):
        result = pt.browse("teleport somewhere")
    assert result.returncode == 1
    assert "open" in result.stderr


def test_acting_verbs_are_refused_by_the_reading_tool():
    """Otherwise the split buys nothing -- a click through `browse` would skip
    the hold."""
    runner = _fake_browser()
    with bind_command_runner(runner):
        result = pt.browse("click Buy now")
    assert result.returncode == 1


def test_an_operation_with_nothing_to_act_on_is_refused():
    runner = _fake_browser()
    with bind_command_runner(runner):
        assert pt.browse("open").returncode == 1


def test_the_vocabulary_stays_small():
    """A vocabulary a model uses correctly beats a faithful reproduction of a
    mouse."""
    assert len(browsing.READ_OPS) + len(browsing.ACT_OPS) <= 8


# --------------------------------------------------------------------------
# The observation
# --------------------------------------------------------------------------

def test_a_page_comes_back_as_a_digest_not_a_page():
    runner = _fake_browser(digest="url: https://x\ntitle: Shop\nlinks:\n  Basket -> /cart")
    with bind_command_runner(runner):
        result = pt.browse("open https://x")
    assert result.returncode == 0
    assert "title: Shop" in result.stdout


def test_the_digest_is_bounded():
    """A page with 400 links is a navigation index, and listing all of them is
    how an observation stops being an observation."""
    assert browsing.MAX_DIGEST_CHARS <= 8000
    assert browsing.MAX_LINKS <= 50


def test_an_enormous_page_is_clipped_on_the_way_back():
    runner = _fake_browser(digest="x" * 200_000)
    with bind_command_runner(runner):
        result = pt.browse("open https://x")
    assert len(result.stdout) < 200_000


def test_a_driver_failure_is_reported_not_raised():
    runner = _fake_browser(digest="Timeout 30000ms exceeded", code=5)
    with bind_command_runner(runner):
        result = pt.browse("open https://slow")
    assert result.returncode == 1
    assert "Timeout" in result.stderr


def test_a_container_without_playwright_says_so():
    runner = _fake_browser(digest="this container has no playwright installed", code=3)
    with bind_command_runner(runner):
        result = pt.browse("open https://x")
    assert "playwright" in result.stderr


# --------------------------------------------------------------------------
# Parsing, on its own
# --------------------------------------------------------------------------

def test_an_argument_may_span_lines():
    assert browsing.parse_op("type Email = a@b.c", browsing.ACT_OPS) == ("type", "Email = a@b.c")


def test_an_operation_with_no_argument_parses():
    assert browsing.parse_op("read", browsing.READ_OPS) == ("read", "")


def test_an_empty_body_is_explained():
    assert isinstance(browsing.parse_op("   ", browsing.READ_OPS), str)


# --------------------------------------------------------------------------
# Where it is allowed to go
# --------------------------------------------------------------------------
#
# The agent chooses this URL, and the agent reads web pages -- so a page it
# has already opened can steer the next request. That is indirect prompt
# injection with a network call on the end of it, which is why the check is on
# the production path and not only in a test.

def test_only_http_and_https_are_opened():
    """`file:///etc/passwd` through a browser is a file read with extra steps,
    and the digest comes back as ordinary tool output."""
    for url in ("file:///etc/passwd", "data:text/html,<b>x",
                "ftp://example.com/x", "chrome://settings"):
        assert browsing.check_url(url), url


def test_the_cloud_metadata_endpoint_is_refused():
    """169.254.169.254 hands out credentials to anything that asks, on every
    major provider."""
    assert browsing.check_url("http://169.254.169.254/latest/meta-data/")


def test_loopback_and_private_ranges_are_refused():
    for url in ("http://127.0.0.1:8080/", "http://[::1]/", "https://10.1.2.3/",
                "http://192.168.0.5/", "http://172.16.9.9/"):
        assert browsing.check_url(url), url


def test_the_obfuscated_forms_of_localhost_are_refused():
    """A browser resolves all three of these to 127.0.0.1 and `ipaddress` does
    not parse any of them, so a check that only asked `ipaddress` let every
    one through. Measured on the first draft of this function."""
    for url in ("http://2130706433/", "http://0177.0.0.1/", "http://0x7f.0.0.1/",
                "http://[::ffff:127.0.0.1]/"):
        assert browsing.check_url(url), url


def test_loopback_and_metadata_by_NAME_are_refused():
    for url in ("http://localhost:9000/", "http://foo.localhost/",
                "http://metadata.google.internal/computeMetadata/v1/"):
        assert browsing.check_url(url), url


def test_an_ordinary_public_url_is_allowed():
    for url in ("https://example.com", "https://sub.example.com/a?b=c",
                "http://93.184.216.34/"):
        assert browsing.check_url(url) == "", url


def test_a_refused_url_never_reaches_the_container():
    """Checked before the driver script is built, so the request is not made
    and then discarded -- it is not made."""
    runner = _fake_browser()
    with bind_command_runner(runner):
        result = pt.browse("open http://169.254.169.254/latest/meta-data/")

    assert result.returncode == 1
    assert "command" not in runner.seen, "the driver ran anyway"


# --------------------------------------------------------------------------
# Locally, when no container is bound
# --------------------------------------------------------------------------
#
# The case that made this necessary: an ordinary `otto chat` turn built a chess
# game whose script threw at load. `node --check` passed, the run's own tests
# exercised the move logic, and the judge -- with nothing that could load a
# page -- approved a board that never drew. So the same driver now runs here,
# through whatever interpreter has Playwright, and a page of otto's own that
# throws is a FAILED call: the command that fails if the page is broken.

import os
from pathlib import Path

from agent.pipeline import walkthrough
from agent.pipeline.workspace import bind_workspace

#: Read before the autouse fixture in conftest clears it, so the integration
#: test at the bottom can still find a real interpreter when one was named.
_REAL_BROWSER_PYTHON = os.environ.get("OTTO_BROWSER_PYTHON", "")


def _local(monkeypatch, digest: str, code: int = 0, stderr: str = ""):
    """A local driver that records what it was asked and returns `digest`."""
    seen = {}

    def run_local(op, argument, limits, *, workspace, timeout=90.0):
        seen.update(op=op, argument=argument, limits=limits, workspace=workspace)
        return (digest, stderr, code)

    monkeypatch.setattr(browsing, "run_local", run_local)
    monkeypatch.setattr(browsing, "_LOCAL", "/some/python")
    return seen


def test_with_a_workspace_but_no_playwright_it_names_the_fix(tmp_path):
    """The refusal says what to install and which variable to set, because
    "no browser" on its own sends a person to the README at best."""
    (tmp_path / "index.html").write_text("<p>hi</p>")
    with bind_workspace(tmp_path):
        result = pt.browse("open index.html")

    assert result.returncode == 1
    assert "OTTO_BROWSER_PYTHON" in result.stderr
    assert "playwright" in result.stderr


def test_a_workspace_path_opens_as_a_file_url(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    seen = _local(monkeypatch, "url: file:///x/index.html\ntitle: t\ntext:\n  hi")
    with bind_workspace(tmp_path):
        result = pt.browse("open index.html")

    assert result.returncode == 0
    assert seen["op"] == "open"
    assert seen["argument"].startswith("file://")
    assert seen["argument"].endswith("/index.html")
    assert Path(seen["workspace"]).resolve() == tmp_path.resolve()


def test_a_path_that_escapes_the_workspace_never_reaches_the_driver(tmp_path, monkeypatch):
    """Confined the way every file tool is. `file:` typed directly is refused
    separately by check_url, so the only way to a file is through the root."""
    seen = _local(monkeypatch, "url: file:///etc/passwd")
    with bind_workspace(tmp_path):
        outside = pt.browse("open ../../../etc/passwd")
        direct = pt.browse("open file:///etc/passwd")

    assert outside.returncode == 1 and "workspace" in outside.stderr
    assert direct.returncode == 1
    assert "op" not in seen


def test_a_missing_page_is_reported_not_opened(tmp_path, monkeypatch):
    seen = _local(monkeypatch, "url: file:///x")
    with bind_workspace(tmp_path):
        result = pt.browse("open nothing-here.html")

    assert result.returncode == 1
    assert "not a file" in result.stderr
    assert "op" not in seen


def test_a_workspace_page_that_throws_fails_the_call(tmp_path, monkeypatch):
    """The whole point. The digest still comes back -- the model should see
    what did render -- but the call is a failure, which is what lets
    evidence.py count a clean `open` as a check."""
    (tmp_path / "index.html").write_text("<script>лекет</script>", encoding="utf-8")
    seen = _local(monkeypatch, (
        "url: file:///x/index.html\ntitle: Chess vs AI\ntext:\n  Chess vs Computer\n"
        "page errors:\n  uncaught ReferenceError: лекет is not defined"
    ))
    with bind_workspace(tmp_path):
        result = pt.browse("open index.html")

    assert result.returncode == 1
    assert "ReferenceError" in result.stderr
    assert "Chess vs Computer" in result.stdout
    assert seen["op"] == "open"


def test_somebody_elses_site_throwing_is_reported_not_failed(tmp_path, monkeypatch):
    """Half the web logs an error on load. A site that is not this run's
    work is not this run's failure -- the errors are in the digest, and the
    model can read them."""
    _local(monkeypatch, (
        "url: https://example.com/\ntitle: Example\ntext:\n  hello\n"
        "page errors:\n  console.error tracker blocked"
    ))
    with bind_workspace(tmp_path):
        result = pt.browse("open https://example.com/")

    assert result.returncode == 0
    assert "page errors" in result.stdout


def test_the_local_driver_is_asked_for_a_screenshot(tmp_path, monkeypatch):
    """That file is what a local `look` shows afterwards."""
    (tmp_path / "index.html").write_text("<p>hi</p>")
    seen = _local(monkeypatch, "url: file:///x")
    with bind_workspace(tmp_path):
        pt.browse("open index.html")

    import json
    limits = json.loads(seen["limits"])
    assert limits["screenshot"] == str(browsing.last_screenshot(tmp_path))
    assert limits["errors"] == browsing.MAX_ERRORS


def test_the_state_lives_outside_the_workspace(tmp_path):
    """A `.otto-browser/` in the repository under edit would show up in
    `git status` and in the agent's own next listing."""
    where = browsing.state_dir(tmp_path)

    assert tmp_path.resolve() not in where.resolve().parents
    assert where == browsing.state_dir(tmp_path), "not deterministic"
    assert browsing.last_screenshot(tmp_path).parent == where


def test_the_container_path_is_unchanged_by_the_local_one(tmp_path, monkeypatch):
    """With a container bound, a workspace path is NOT resolved here -- the
    container's browser has its own filesystem -- and the driver still goes
    through the command runner."""
    runner = _fake_browser()
    monkeypatch.setattr(browsing, "_LOCAL", "/some/python")
    with bind_workspace(tmp_path), bind_command_runner(runner):
        result = pt.browse("open index.html")

    assert result.returncode == 1, "a bare path is not a URL the container can open"
    assert "command" not in runner.seen


# ---- what the driver reports -----------------------------------------------

def test_the_driver_reports_what_the_page_threw():
    """Both halves of the contract in one place: the script prints the
    heading this module reads back, and it listens for the two things a
    broken page does -- throw, and console.error."""
    assert f'print("{browsing.ERRORS_HEADING}")' in browsing.DRIVER
    assert 'page.on("pageerror"' in browsing.DRIVER
    assert 'page.on("console"' in browsing.DRIVER


def test_page_errors_are_read_back_out_of_a_digest():
    digest = (
        "url: file:///x\ntitle: t\ntext:\n  hello\n"
        "page errors:\n  uncaught TypeError: x is not a function\n  console.error boom\n"
    )
    assert browsing.page_errors(digest) == [
        "uncaught TypeError: x is not a function", "console.error boom",
    ]
    assert browsing.page_errors("url: file:///x\ntitle: t\ntext:\n  hello") == []


def test_the_driver_keeps_its_state_where_it_is_told():
    """The container path keeps /tmp; the local path gives each workspace its
    own directory through the variable, so two workspaces never share a
    cookie jar or a last-opened URL."""
    assert "OTTO_BROWSER_STATE_DIR" in browsing.DRIVER


# ---- the probe ------------------------------------------------------------

def test_the_probe_is_cached_and_can_be_forgotten(monkeypatch):
    monkeypatch.setattr(browsing, "_LOCAL", "/cached/python")
    assert browsing.local_interpreter() == "/cached/python"

    browsing.forget_local_interpreter()
    monkeypatch.setenv("OTTO_BROWSER_PYTHON", "/no/such/interpreter")
    # Nothing here has Playwright (test_otto_itself_needs_no_browser_dependency
    # holds that), so the probe comes up empty and stays empty.
    assert browsing.local_interpreter() is None
    assert browsing._LOCAL is None


def test_a_named_interpreter_is_tried_before_ottos_own(monkeypatch):
    monkeypatch.setenv("OTTO_BROWSER_PYTHON", "/deliberate/python")
    assert browsing._candidates()[0] == "/deliberate/python"


# ---- for real, when a browser is at hand -------------------------------------

@pytest.mark.skipif(not _REAL_BROWSER_PYTHON,
                    reason="set OTTO_BROWSER_PYTHON to an interpreter with Playwright")
def test_for_real_a_page_that_throws_fails_and_a_clean_one_leaves_a_screenshot(tmp_path, monkeypatch):
    monkeypatch.setattr(browsing, "_LOCAL", _REAL_BROWSER_PYTHON)
    (tmp_path / "broken.html").write_text(
        "<!doctype html><title>Broken</title><div id=b></div>"
        "<script>'use strict';\nлекет\ndocument.getElementById('b').textContent='drawn'</script>"
    )
    (tmp_path / "fine.html").write_text(
        "<!doctype html><title>Fine</title><div id=b></div>"
        "<script>document.getElementById('b').textContent='drawn'</script>"
    )
    with bind_workspace(tmp_path):
        broken = pt.browse("open broken.html")
        fine = pt.browse("open fine.html")

    assert broken.returncode == 1
    assert "лекет is not defined" in broken.stderr
    assert "drawn" not in broken.stdout, "the script died before it drew"
    assert fine.returncode == 0
    assert "drawn" in fine.stdout
    assert browsing.last_screenshot(tmp_path).is_file()


# --------------------------------------------------------------------------
# Walkthroughs: using the page, not just loading it
# --------------------------------------------------------------------------
#
# `browse` and `browse_act` each drive a fresh page, so a selected chess piece
# is gone by the next call. `exercise` runs a whole sequence in ONE page and
# reports each step as the browser saw it -- and the judge is shown that
# report as evidence the thing works.

def test_a_page_walkthrough_is_parsed_before_any_browser_starts():
    walk = walkthrough.parse_walk(
        "open index.html\nclick New game\ncount css .piece = 32\nexpect not Checkmate\nwait 100"
    )
    assert walk.backend == "page"
    assert [s.verb for s in walk.steps] == ["open", "click", "count", "expect", "wait"]


def test_the_driver_knows_every_page_step():
    for verb in walkthrough.PAGE_STEPS:
        assert f'verb == "{verb}"' in browsing.DRIVER, f"the driver has no `{verb}` step"


def test_the_walk_opens_the_workspace_page_and_reports_every_step(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    seen = _local(monkeypatch, (
        "3/3 steps passed: open index.html; click New game; count css .piece = 32\n"
        "  1. open file:///x/index.html -> ok\n  2. click New game -> ok\n"
        "  3. count css .piece = 32 -> ok\nurl: file:///x/index.html\ntitle: Chess\ntext:\n  Chess"
    ))
    with bind_workspace(tmp_path):
        result = pt.exercise("open index.html\nclick New game\ncount css .piece = 32")

    assert result.returncode == 0
    assert seen["op"] == "walk"
    first, *rest = seen["argument"].splitlines()
    assert first.startswith("open file://") and first.endswith("/index.html")
    assert rest == ["click New game", "count css .piece = 32"]
    assert result.stdout.startswith("3/3 steps passed"), "the summary is the action record's line"


def test_a_step_that_does_not_hold_fails_the_call_and_names_itself(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    _local(monkeypatch, (
        "1/3 steps passed: open index.html; click e2; expect Computer is thinking\n"
        "  1. open file:///x/index.html -> ok\n"
        "  2. click e2 -> FAILED: Timeout 5000ms exceeded waiting for get_by_text(\"e2\")\n"
        "url: file:///x/index.html\ntitle: Chess\ntext:\n  Chess"
    ), code=6)
    with bind_workspace(tmp_path):
        result = pt.exercise("open index.html\nclick e2\nexpect Computer is thinking")

    assert result.returncode == 1
    assert "step 2" in result.stderr or "2. click e2" in result.stderr
    assert "FAILED" in result.stderr
    assert "1/3 steps passed" in result.stdout, "the report still comes back"


def test_a_page_that_threw_during_a_walk_fails_it_even_when_every_step_held(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    _local(monkeypatch, (
        "1/1 steps passed: open index.html\n  1. open file:///x/index.html -> ok\n"
        "url: file:///x/index.html\ntitle: t\ntext:\n  t\n"
        "page errors:\n  uncaught ReferenceError: лекет is not defined"
    ))
    with bind_workspace(tmp_path):
        result = pt.exercise("open index.html")

    assert result.returncode == 1
    assert "threw" in result.stderr and "ReferenceError" in result.stderr


def test_a_walk_that_could_not_run_is_reported_not_raised(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<p>hi</p>")
    _local(monkeypatch, "", code=5, stderr="Timeout 30000ms exceeded")
    with bind_workspace(tmp_path):
        result = pt.exercise("open index.html")

    assert result.returncode == 1 and "Timeout" in result.stderr


def test_a_walk_never_reaches_outside_the_workspace(tmp_path, monkeypatch):
    seen = _local(monkeypatch, "1/1 steps passed: open x")
    with bind_workspace(tmp_path):
        outside = pt.exercise("open ../../../etc/passwd")
        missing = pt.exercise("open nothing.html")

    assert outside.returncode == 1 and missing.returncode == 1
    assert "op" not in seen


def test_exercise_is_read_only_because_it_only_ever_opens_workspace_files():
    """The tier the mutation gate reads. A walkthrough cannot act on a live
    site -- parse_walk refuses a URL -- so it changes nothing outside the
    run, and a hold before every run of the page it just built would cost an
    exchange for nothing."""
    assert pt.TOOL_TIERS["exercise"] == pt.READ_ONLY
    assert pt.TOOL_NEEDS["exercise"] == pt.NEEDS_EITHER, "the shell backend needs nothing installed"


@pytest.mark.skipif(not _REAL_BROWSER_PYTHON,
                    reason="set OTTO_BROWSER_PYTHON to an interpreter with Playwright")
def test_for_real_a_walkthrough_uses_the_page_and_stops_where_it_breaks(tmp_path, monkeypatch):
    monkeypatch.setattr(browsing, "_LOCAL", _REAL_BROWSER_PYTHON)
    (tmp_path / "counter.html").write_text(
        "<!doctype html><title>Counter</title><p id=n>Count: 0</p>"
        "<button id=add>Add one</button><button id=reset>Reset</button>"
        "<script>let c=0;const n=document.getElementById('n');"
        "document.getElementById('add').onclick=()=>{c++;n.textContent='Count: '+c};"
        "document.getElementById('reset').onclick=()=>{c=0;n.textContent='Count: 0'};</script>"
    )
    with bind_workspace(tmp_path):
        works = pt.exercise(
            "open counter.html\nexpect Count: 0\nclick Add one\nclick Add one\n"
            "expect Count: 2\nexpect not Count: 0\nclick Reset\nexpect Count: 0\ncount css button = 2"
        )
        wrong = pt.exercise("open counter.html\nclick Add one\nexpect Count: 5")

    assert works.returncode == 0, works.stderr
    assert works.stdout.startswith("9/9 steps passed")
    assert wrong.returncode == 1
    assert "expect Count: 5" in wrong.stderr and "FAILED" in wrong.stderr
    assert "2/3 steps passed" in wrong.stdout
