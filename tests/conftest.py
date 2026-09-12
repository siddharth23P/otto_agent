"""Make the suite collectable without credentials.

`agent/pipeline/nodes.py` builds `ROUTER = Router()` at module scope, and
`Router.__init__` raises AuthError when INCEPTION_API_KEY is absent. So a bare
`uv run pytest` on a clean checkout does not fail a test -- it fails COLLECTION,
on 13 modules at once, before anything runs. That is a hard stop for CI and for
anyone new to the repo, and it is why every command in this project's history
has been written as `INCEPTION_API_KEY=fake-key uv run pytest`.

The fix belongs here rather than in the modules. Import-time construction of the
router is a deliberate choice (agent/cli/main.py loads .env before importing
anything, precisely so that constructor sees real keys), and a test suite should
not be the reason to unpick it.

Placeholders, not real keys: every test in this suite is offline and fakes the
model, so what these need to do is satisfy a constructor, not authenticate. A
real key in the environment is left alone -- `setdefault` -- because a developer
running the live benchmarks from the same shell should not have them silently
swapped for fakes.
"""
import os

#: Every provider agent/router/llm_provider registers. Inception is the only one
#: whose absence breaks collection today (it is Router.REQUIRED); the rest are
#: here so that a test which asks whether a provider is configured gets the same
#: answer on every machine, rather than depending on the developer's .env.
_PLACEHOLDER_KEYS = (
    "INCEPTION_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)

for _name in _PLACEHOLDER_KEYS:
    os.environ.setdefault(_name, "test-placeholder-not-a-real-key")

#: Keep the embedding backend local and deterministic. agent/memory/embeddings.py
#: switches to hosted Gemini the moment GEMINI_API_KEY is present -- which the
#: line above guarantees -- and a suite that then tried to embed would reach the
#: network with a fake key. An explicit empty value means "no hosted spec", which
#: falls through to the local model.
os.environ.setdefault("OTTO_EMBEDDING_MODEL", "")


#: Never let one test's embedding backend become every later test's.
#:
#: agent/memory/embeddings.py caches the backend in a module global and builds
#: it once, on first use. `reset_backend()` clears it, and the tests that
#: exercise backend selection call that -- but nothing puts it back, and
#: monkeypatch restoring GEMINI_API_KEY or OTTO_EMBEDDING_MODEL does not
#: rebuild a backend already chosen under the patched values.
#:
#: So a test that deliberately selected the local model left every later test
#: in the session running on it. That is how
#: test_semantic_is_not_beaten_on_rare_tokens_either failed only in the full
#: suite and never on its own: alone it ran on hosted Gemini and recalled all
#: three rare tokens, and in the suite it ran on leaked BAAI/bge-small and
#: missed the date. A test whose result depends on which other tests ran
#: before it is not testing what it says it is.
import pytest  # noqa: E402

from agent.memory import embeddings as _embeddings  # noqa: E402


@pytest.fixture(autouse=True)
def _no_leaked_embedding_backend():
    """Snapshot and restore, rather than reset. Resetting would make the next
    test that embeds rebuild the backend from scratch -- for the local model
    that is loading weights, on every test in the suite."""
    before = _embeddings._backend
    yield
    _embeddings._backend = before


#: Never let a test read from or write to the developer's real lesson bank.
#:
#: agent/memory/lessons.py opens ~/.otto/memory/lessons.db on first use, so a
#: test that runs the graph without binding one would create it -- and a test
#: that finished a run would TEACH it. Both are wrong: a suite that writes into
#: the thing the agent learns from is a suite that changes the agent's behaviour
#: by being run. Tests about the bank bind their own with `bind_bank`.
from agent.memory import lessons as _lessons  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_lesson_bank(request):
    if "bank" in getattr(request, "fixturenames", ()):
        yield  # that test binds its own
        return
    with _lessons.bind_bank(None):
        yield


#: And never the developer's real seat-outcome log, for the same reason.
#: agent/router/outcomes.py reorders routing from it, so a suite that wrote
#: into it would eventually change which model answers -- a test run silently
#: reconfiguring the agent it is testing.
from agent.router import outcomes as _outcomes  # noqa: E402


@pytest.fixture(autouse=True)
def _no_real_outcome_log(request, tmp_path_factory):
    if "seat_log" in getattr(request, "fixturenames", ()):
        yield  # that test binds its own
        return
    with _outcomes.bind_log(tmp_path_factory.mktemp("seats") / "outcomes.db"):
        yield


#: Tests that can only run where the host shell is POSIX.
#:
#: Otto's remote branch -- every tool's container path -- generates POSIX
#: commands, because the far end is always a Linux container: `find` with
#: `-prune`, `base64`, heredocs, `sh -c`. Several tests exercise that branch
#: without Docker by binding a command runner that runs those commands on the
#: HOST shell, which is exactly the right trick on Linux and macOS and is
#: cmd.exe on Windows.
#:
#: So these are skipped on Windows rather than made to pass there: what they
#: test is the behaviour of a Linux container, and a version of them that
#: cmd.exe could satisfy would be testing something Otto never does.
posix_only = pytest.mark.skipif(
    os.name == "nt",
    reason="exercises the container path by running POSIX commands on the host shell",
)

#: Filesystem behaviour that genuinely differs on Windows -- symlink creation
#: needs a privilege, and an over-long path fails with a different error at a
#: different layer. The production code handles both; only the assertions here
#: are platform-specific.
posix_filesystem = pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX filesystem semantics (symlinks without privilege, path-length errors)",
)
