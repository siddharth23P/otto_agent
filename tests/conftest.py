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


#: Never let a test read from or write to the developer's real lesson bank.
#:
#: agent/memory/lessons.py opens ~/.otto/memory/lessons.db on first use, so a
#: test that runs the graph without binding one would create it -- and a test
#: that finished a run would TEACH it. Both are wrong: a suite that writes into
#: the thing the agent learns from is a suite that changes the agent's behaviour
#: by being run. Tests about the bank bind their own with `bind_bank`.
import pytest  # noqa: E402

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
