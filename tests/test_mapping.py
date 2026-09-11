"""Tests for the routing table and its validator.

These never re-implement `validate()`. A test that contains the same rules as
the code it checks agrees with that code no matter what either of them does --
including when both are wrong. Every case here hardcodes a table that is known
to be broken (or known to be fine) and asserts on the outcome.

Nothing here needs an API key, a network connection, or a vendor SDK.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent.router.llm_provider.base import Capability
from agent.router.mapping import (
    TASK_ROUTES,
    Candidate,
    Endpoint,
    MappingError,
    Preference,
    Task,
    validate,
)

CHAT = frozenset({Capability.CHAT})


def table(**overrides: tuple[Candidate, ...]) -> dict[Task, tuple[Candidate, ...]]:
    """The real table with one or more chains swapped out.

    Always start from a valid table. A hand-built one breaks several rules at
    once -- a case meant to test "empty chain" also trips "no route defined"
    for every task left out -- and then you cannot tell which check fired.
    """
    return {**TASK_ROUTES, **overrides}


def problems_from(routes) -> str:
    """Run validate() expecting failure, return the message."""
    with pytest.raises(MappingError) as exc:
        validate(routes)
    return str(exc.value)


# ---------------------------------------------------------------------------
# The real table
# ---------------------------------------------------------------------------


def test_real_table_is_valid():
    """Regression guard. Fails the moment a route is pasted in wrong."""
    validate()


def test_every_task_has_a_route():
    """Stated directly, not just as a side effect of validate()."""
    assert set(TASK_ROUTES) == set(Task)


def test_every_chain_is_a_non_empty_tuple():
    for task, chain in TASK_ROUTES.items():
        assert isinstance(chain, tuple), f"{task} is {type(chain).__name__}"
        assert chain, f"{task} is empty"


def test_inception_is_the_floor_of_every_chat_chain():
    """The cost policy's structural half: no chat task can fail outright."""
    chat_tasks = [
        t for t, chain in TASK_ROUTES.items()
        if all(c.endpoint is Endpoint.CHAT for c in chain)
    ]
    assert chat_tasks, "expected at least one chat-only task"
    for task in chat_tasks:
        specs = [c.spec for c in TASK_ROUTES[task] if c.spec]
        assert any(s.startswith("inception:") for s in specs), (
            f"{task} has no pinned Inception candidate, so it can raise "
            f"NoViableRoute when Inception itself is unreachable"
        )


def test_every_route_is_inception_only():
    """Otto is Inception-only (2026-09-09) -- every candidate in every chain
    is now a pinned Inception spec. This is the property that made the
    anthropic/openai/gemini provider modules and their TASK_ROUTES entries
    deletable rather than just unused: nothing left in the table can resolve
    against them."""
    for task, chain in TASK_ROUTES.items():
        for i, c in enumerate(chain):
            assert c.provider_name == "inception", f"{task.name}[{i}]: {c.provider_name!r}"


def test_chat_chains_are_pinned_to_mercury_2_5():
    """The point of today's change: not just "Inception", the CURRENT
    generation. A route still pinned to mercury-2 would keep working (both
    ids are live), but silently miss the quality Mercury 2.5 exists for."""
    chat_tasks = [
        t for t, chain in TASK_ROUTES.items()
        if all(c.endpoint is Endpoint.CHAT for c in chain)
    ]
    for task in chat_tasks:
        specs = [c.spec for c in TASK_ROUTES[task]]
        assert specs == ["inception:mercury-2.5"], f"{task}: {specs}"


# ---------------------------------------------------------------------------
# Tables that must be rejected
# ---------------------------------------------------------------------------

REJECTED = [
    # --- table level ---
    pytest.param({Task.CHAT_FAST: {1, 2}}, "not tuple", id="chain-is-a-set"),
    pytest.param({Task.CHAT_FAST: [Candidate(spec="inception:mercury-2", requires=CHAT)]},
                 "not tuple", id="chain-is-a-list"),
    pytest.param({Task.CHAT_FAST: ()}, "empty", id="chain-is-empty"),
    pytest.param({Task.CHAT_FAST: ("just a string",)},
                 "not a Candidate", id="element-is-not-a-candidate"),

    # --- spec / provider ---
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:m", provider="openai",
                                             requires=CHAT),)},
                 "not both", id="spec-and-provider-both-set"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="mercury-2", requires=CHAT),)},
                 "provider:model", id="spec-has-no-colon"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="a:b:c", requires=CHAT),)},
                 "more than one", id="spec-has-two-colons"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="mistral:large", requires=CHAT),)},
                 "unknown provider", id="pinned-vendor-not-registered"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:", requires=CHAT),)},
                 "empty model id", id="spec-has-no-model"),
    pytest.param({Task.CHAT_FAST: (Candidate(provider="mistral", requires=CHAT),)},
                 "unknown provider", id="scoped-vendor-not-registered"),
    pytest.param({Task.CHAT_FAST: (Candidate(requires=frozenset()),)},
                 "must set requires", id="query-matches-everything"),

    # --- requires ---
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2",
                                             requires={Capability.CHAT}),)},
                 "not frozenset", id="requires-is-a-mutable-set"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2",
                                             requires=frozenset({"chatt"})),)},
                 "non-Capability", id="requires-holds-a-raw-string"),
    pytest.param({Task.CODE_COMPLETE: (Candidate(spec="inception:mercury-edit-2",
                                                 requires=CHAT,
                                                 endpoint=Endpoint.FIM),)},
                 "needs", id="endpoint-capability-not-required"),

    # --- endpoints only Inception implements ---
    # These exercise the inception-only branch for real again. While openai
    # and anthropic were unregistered they failed one check earlier, on
    # "unknown provider", and the comment here used to note the lost
    # coverage; registering those vendors (2026-09-11) restores it.
    pytest.param({Task.CODE_COMPLETE: (Candidate(spec="openai:gpt-4o",
                                                 requires=frozenset({Capability.FIM}),
                                                 endpoint=Endpoint.FIM),)},
                 "inception-only", id="fim-pinned-to-a-non-inception-vendor"),
    pytest.param({Task.CODE_EDIT: (Candidate(provider="anthropic",
                                             requires=frozenset({Capability.EDIT}),
                                             endpoint=Endpoint.EDIT),)},
                 "inception-only", id="edit-scoped-to-a-non-inception-vendor"),
    pytest.param({Task.CODE_COMPLETE: (Candidate(requires=frozenset({Capability.FIM}),
                                                 endpoint=Endpoint.FIM),)},
                 "open query", id="fim-as-an-open-query"),

    # --- scalars ---
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2", requires=CHAT,
                                             min_context=True),)},
                 "must be an int", id="min-context-is-a-bool"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2", requires=CHAT,
                                             min_context=0),)},
                 "must be positive", id="min-context-is-zero"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2", requires=CHAT,
                                             min_context=-5),)},
                 "must be positive", id="min-context-is-negative"),
    pytest.param({Task.CHAT_FAST: (Candidate(provider="inception", requires=CHAT,
                                             name_contains="   "),)},
                 "non-empty string", id="name-contains-is-blank"),
    pytest.param({Task.CHAT_FAST: (Candidate(provider="inception", requires=CHAT,
                                             prefer="cheap"),)},
                 "not a Preference", id="prefer-is-a-raw-string"),

    # --- params legality, per endpoint ---
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2", requires=CHAT,
                                             params={"stop": ["\n"], "tools": []}),)},
                 "not accepted", id="invoke-time-options-in-chat-params"),
    pytest.param({Task.CODE_COMPLETE: (Candidate(spec="inception:mercury-edit-2",
                                                 requires=frozenset({Capability.FIM}),
                                                 endpoint=Endpoint.FIM,
                                                 params={"temperature": 0.7}),)},
                 "not accepted", id="temperature-on-fim"),
    pytest.param({Task.CODE_EDIT: (Candidate(spec="inception:mercury-edit-2",
                                             requires=frozenset({Capability.EDIT}),
                                             endpoint=Endpoint.EDIT,
                                             params={"diffusing": True}),)},
                 "not accepted", id="chat-only-param-on-edit"),
    pytest.param({Task.CODE_COMPLETE: (Candidate(spec="inception:mercury-edit-2",
                                                 requires=frozenset({Capability.FIM}),
                                                 endpoint=Endpoint.FIM,
                                                 params={"suffix": "x"}),)},
                 "not accepted", id="param-collides-with-positional-arg"),

    # --- chain level ---
    pytest.param({Task.REASON: (Candidate(requires=CHAT),
                                Candidate(spec="inception:mercury-2", requires=CHAT))},
                 "must be last", id="open-query-shadows-later-pins"),
    pytest.param({Task.REASON: (Candidate(requires=CHAT),
                                Candidate(requires=frozenset({Capability.TOOLS,
                                                              Capability.CHAT})))},
                 "at most one", id="two-open-queries"),
]


@pytest.mark.parametrize("overrides, fragment", REJECTED)
def test_rejects(overrides, fragment):
    message = problems_from(table(**overrides))
    assert fragment in message, message
    # The message must also point at the offending task, or it is useless
    # for finding the line.
    task = next(iter(overrides))
    assert task.name in message, message


def test_rejects_a_missing_task():
    """Iterating Task rather than the dict is what catches this."""
    incomplete = {t: c for t, c in TASK_ROUTES.items() if t is not Task.PLAN}
    message = problems_from(incomplete)
    assert "no route defined" in message
    assert "PLAN" in message


def test_reports_every_problem_at_once():
    """Accumulate-then-raise, not fail-fast.

    Fail-fast means one typo fixed per run. With three broken chains the
    message must name all three.
    """
    message = problems_from(table(**{
        Task.CHAT_FAST: (),
        Task.REASON: {1, 2},
        Task.SUMMARIZE: ("nope",),
    }))
    for task in ("CHAT_FAST", "REASON", "SUMMARIZE"):
        assert task in message, message
    assert len(message.strip().splitlines()) >= 4  # header + three problems


# ---------------------------------------------------------------------------
# Tables that must be accepted
# ---------------------------------------------------------------------------
#
# The half people skip. An over-strict validator is as broken as a permissive
# one, and only these catch it.

ACCEPTED = [
    # Several provider-scoped queries in one chain is fine even with a single
    # known vendor -- nothing in the validator requires them to be distinct
    # vendors, only that at most one is an *open* (vendor-less) query.
    pytest.param({Task.REASON: (Candidate(provider="inception", name_contains="mercury-2.5",
                                          requires=CHAT),
                                Candidate(provider="inception", name_contains="mercury-2",
                                          requires=CHAT),
                                Candidate(provider="inception", name_contains="edit",
                                          requires=CHAT))},
                 id="several-scoped-queries-in-one-chain"),
    pytest.param({Task.REASON: (Candidate(spec="inception:mercury-2.5", requires=CHAT),
                                Candidate(requires=CHAT))},
                 id="open-query-in-last-position"),
    pytest.param({Task.CODE_COMPLETE: (Candidate(provider="inception",
                                                 requires=frozenset({Capability.FIM}),
                                                 endpoint=Endpoint.FIM),)},
                 id="fim-scoped-to-inception"),
    pytest.param({Task.CHAT_FAST: (Candidate(spec="inception:mercury-2.5", requires=CHAT),)},
                 id="pin-with-no-params-or-bounds"),
    pytest.param({Task.CHAT_FAST: (Candidate(provider="inception", name_contains="mercury",
                                             requires=CHAT, min_context=1,
                                             prefer=Preference.LARGEST_CONTEXT),)},
                 id="every-optional-field-set"),
]


@pytest.mark.parametrize("overrides", ACCEPTED)
def test_accepts(overrides):
    validate(table(**overrides))


# ---------------------------------------------------------------------------
# Properties the whole build order depends on
# ---------------------------------------------------------------------------


def test_known_providers_matches_the_registry():
    """`mapping.py` hardcodes vendor names because it cannot import the registry.

    The dependency points from the impure module to the pure one, so this is
    where the two are reconciled. Without it, renaming a provider silently
    turns every route naming it into a skip.
    """
    from agent.router.llm_provider import provider_names
    from agent.router.mapping import KNOWN_PROVIDERS

    assert KNOWN_PROVIDERS == set(provider_names())


def test_import_needs_no_credentials():
    """The property Phases 1 and 2 are sequenced around.

    Run in a subprocess with every *_API_KEY stripped: importing the module
    runs validate(), and neither may reach for an environment. A subprocess
    rather than importlib.reload() -- reloading swaps the Task enum class out
    from under every other test in the session.
    """
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [sys.executable, "-c", "import agent.router.mapping"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_candidate_cannot_go_in_a_set():
    """Documents why chains are tuples.

    `params` makes a frozen Candidate unhashable, so a `{...}` literal fails
    loudly instead of silently scrambling the fallback order. If someone
    "fixes" the unhashability, this test is the warning.
    """
    with pytest.raises(TypeError, match="unhashable"):
        {Candidate(spec="inception:mercury-2", requires=CHAT, params={"temperature": 0.2})}


#: Params that switch a model into a thinking/reasoning mode.
THINKING_PARAMS = frozenset({"thinking", "reasoning_effort", "include_thoughts"})


def test_thinking_routes_require_a_reasoning_model():
    """A route that configures thinking must ask for a model that can think.

    Otherwise the candidate resolves to whatever matched, the vendor rejects the
    parameter, and you get a 400 only when that candidate finally wins -- which
    may be weeks after the route was written.

    Inception is exempt: `reasoning_effort` is a parameter of its chat endpoint
    itself, not a per-model capability, and it publishes no reasoning flag.
    """
    for task, chain in TASK_ROUTES.items():
        for i, candidate in enumerate(chain):
            if candidate.provider_name == "inception":
                continue
            if set(candidate.params) & THINKING_PARAMS:
                assert Capability.REASONING in candidate.requires, (
                    f"{task.name}[{i}] enables thinking but does not require REASONING"
                )


def test_diffusing_is_inception_only():
    """`diffusing` selects the replacing renderer in chat.py. On another vendor
    it would be an unknown parameter *and* the wrong renderer."""
    for task, chain in TASK_ROUTES.items():
        for i, candidate in enumerate(chain):
            if candidate.params.get("diffusing"):
                assert candidate.provider_name == "inception", f"{task.name}[{i}]"
