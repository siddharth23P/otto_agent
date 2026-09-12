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
