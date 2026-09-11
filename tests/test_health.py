"""Temporarily unwell models and providers.

agent/router/llm_provider/retired.py handles the permanent case: a model the
vendor will never serve again. This is the common one -- a rate limit, a
five-hundred, a timeout -- where the honest answer is "not now" and Otto's
answer used to be to resolve the same candidate again on the very next call.

Two layers because the failures differ in kind, and a hard rule that neither
may leave a task with no route at all.
"""
import pytest

from agent.router.health import (
    BREAKER_THRESHOLD, MAX_COOLDOWN_S, Health, bind_health, note_failure,
)


@pytest.fixture
def health():
    with bind_health(Health()) as h:
        yield h


class _Status(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"status {status}")
        self.status_code = status
        self.response = type("R", (), {"headers": headers or {}})()


# --------------------------------------------------------------------------
# Which layer a failure lands in
# --------------------------------------------------------------------------

def test_a_rate_limit_cools_one_model_not_the_vendor(health):
    """Quota is metered per model. Locking the vendor over a 429 on the cheap
    model would take out the seats that were fine."""
    note_failure(_Status(429), provider="openai", model_id="gpt-5-mini")

    assert health.cooling("openai", "gpt-5-mini")
    assert health.cooling("openai", "gpt-5") == ""


def test_one_five_hundred_is_a_blip(health):
    """The next call retries it. A breaker that trips on the first failure is
    a breaker that trips on noise."""
    note_failure(_Status(503), provider="anthropic", model_id="haiku")

    assert health.cooling("anthropic", "haiku") == ""


def test_a_run_of_them_is_an_outage(health):
    """Re-trying every candidate of a dead vendor on every call for the rest
    of a run is how a fifteen-minute outage costs a whole budget."""
    for _ in range(BREAKER_THRESHOLD):
        note_failure(_Status(503), provider="anthropic", model_id="haiku")

    assert "provider anthropic" in health.cooling("anthropic", "haiku")
    assert "provider anthropic" in health.cooling("anthropic", "sonnet"), (
        "the breaker is about the wire, so it covers every model on it"
    )


def test_a_timeout_with_no_status_still_counts(health):
    class APITimeoutError(Exception):
        pass

    for _ in range(BREAKER_THRESHOLD):
        note_failure(APITimeoutError("took too long"), provider="gemini", model_id="flash")

    assert health.cooling("gemini", "flash")


def test_a_bad_key_is_nobody_s_health_problem(health):
    """401 means the config is wrong. Cooling the provider would hide that
    behind a delay instead of reporting it."""
    note_failure(_Status(401), provider="openai", model_id="gpt-5-mini")
    note_failure(_Status(404), provider="openai", model_id="gpt-5-mini")

    assert health.cooling("openai", "gpt-5-mini") == ""


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------

def test_a_success_clears_both_layers(health):
    """One answered request settles "is this vendor reachable", whichever
    model answered it."""
    for _ in range(BREAKER_THRESHOLD):
        note_failure(_Status(500), provider="openai", model_id="gpt-5-mini")
    health.note_rate_limit("openai", "gpt-5-mini")

    health.note_success("openai", "gpt-5-mini")

    assert health.cooling("openai", "gpt-5-mini") == ""
    assert health.snapshot() == {"providers": {}, "models": {}}


def test_the_window_doubles_each_time_it_is_re_entered(health):
    """A vendor that is genuinely down gets asked less and less often, rather
    than on a fixed heartbeat for the rest of the run."""
    health.note_rate_limit("openai", "m")
    first = health.cooling_until("openai", "m")
    health.note_rate_limit("openai", "m")
    second = health.cooling_until("openai", "m")

    assert second > first


def test_the_doubling_stops_somewhere(health):
    """Past the cap a provider is asked once every few minutes -- often
    enough to notice a recovery, rare enough to cost nothing."""
    for _ in range(20):
        health.note_rate_limit("openai", "m")

    assert health.snapshot()["models"]["openai:m"] <= MAX_COOLDOWN_S


def test_the_vendor_s_own_retry_after_wins_over_a_guess(health):
    """A 429 carrying Retry-After: 2, waited out for twenty seconds, is
    nineteen seconds of nothing."""
    note_failure(_Status(429, {"retry-after": "2"}),
                 provider="openai", model_id="gpt-5-mini")

    assert health.snapshot()["models"]["openai:gpt-5-mini"] <= 2.0


# --------------------------------------------------------------------------
# Never into nothing
# --------------------------------------------------------------------------

def _router(monkeypatch, chain):
    """The real Router.resolve over a two-candidate chain, with the catalogue
    and the key check stubbed. Not a hand-rolled sorter: the point of these
    two is that `resolve` itself honours cooldowns and knows when not to."""
    from agent.router import router as rr
    from agent.router.mapping import Task

    monkeypatch.setattr(
        rr.Router, "_select",
        lambda self, pool, c: type(
            "M", (), {"id": rr.seat_outcomes.spec_id(c), "provider": "v"},
        )(),
    )
    monkeypatch.setitem(rr.TASK_ROUTES, Task.CHAT_FAST, chain)
    router = rr.Router.__new__(rr.Router)
    router.catalogue = type("C", (), {"models": lambda self, p: []})()
    router.strict = False
    router._configured = ("v",)
    return router, Task.CHAT_FAST


def test_a_cooling_candidate_is_skipped_while_another_is_available(monkeypatch, health):
    from agent.router.mapping import Candidate

    router, task = _router(monkeypatch, (Candidate(spec="v:a"), Candidate(spec="v:b")))
    health.note_rate_limit("v", "a")

    assert router.resolve(task).model.id == "b"


def test_when_everything_is_cooling_it_routes_anyway(monkeypatch, health):
    """A circuit breaker that leaves a task with no route has turned a slow
    provider into a broken agent, which is worse than the problem. Cooldowns
    are a preference, never a prohibition."""
    from agent.router.mapping import Candidate

    router, task = _router(monkeypatch, (Candidate(spec="v:a"), Candidate(spec="v:b")))
    health.note_rate_limit("v", "a")
    health.note_rate_limit("v", "b")

    assert router.resolve(task).model.id == "a", "the breaker left the task with no route"


def test_a_real_failure_still_raises_rather_than_being_retried_blind(monkeypatch, health):
    """The second pass is only for candidates skipped as UNWELL. A missing key
    or a model that does not exist must still surface as no viable route."""
    from agent.router.mapping import Candidate
    from agent.router.router import NoViableRoute

    router, task = _router(monkeypatch, (Candidate(spec="nope:a"),))

    with pytest.raises(NoViableRoute):
        router.resolve(task)


def test_binding_a_fresh_instance_actually_reaches_the_router(monkeypatch, health):
    """It did not, at first. Both callers did `from ... import HEALTH`, which
    binds the NAME at import, so `bind_health` swapped a module global nobody
    was reading and every test silently measured the process-wide instance."""
    from agent.router import health as module
    from agent.router import router as rr

    assert rr.provider_health.HEALTH is module.HEALTH is health
