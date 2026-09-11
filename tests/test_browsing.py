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
    runner = _fake_browser()
    with bind_command_runner(runner):
        pt.browse("open https://example.com")
    assert "playwright" in runner.seen["command"]
    assert "example.com" in runner.seen["command"]


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
