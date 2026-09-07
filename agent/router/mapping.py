"""Static routing intent.

Imports only `Capability` from the provider base -- no registry, no vendor SDKs.
That is what lets routing be tested with an empty .env and no network.
"""

from typing import Any, Mapping

from .llm_provider.base import Capability
from enum import StrEnum
from dataclasses import dataclass, field


class Task(StrEnum):
    """What a swarm node is doing. Named by intent, never by model or vendor."""

    CHAT_FAST = "chat_fast"
    REASON = "reason"
    PLAN = "plan"
    SUMMARIZE = "summarize"
    CODE_COMPLETE = "code_complete"
    CODE_EDIT = "code_edit"


class Endpoint(StrEnum):
    """Which provider method a candidate is calling.

    Not derivable from `requires`: a capability says what a model can do, an
    endpoint says which function the router invokes. They differ in signature
    and in accepted parameters, so the router has to be told, not left to infer.

        CHAT -> provider.chat_model(model_id, **params) -> BaseChatModel
        FIM  -> provider.fim(model_id, prefix, suffix, **params) -> str
        EDIT -> provider.code_edit(model_id, code_to_edit, **params) -> str
    """

    CHAT = "chat"
    FIM = "fim"
    EDIT = "edit"


class Preference(StrEnum):
    """How to choose when a query matches several models.

    Context window is a proxy for tier, not a price. It is the only comparable
    number every provider publishes -- `ModelInfo.raw` carries pricing for
    Inception but nothing else, so a real cost-aware rule needs a price field on
    `ModelInfo` first. Until then this is an honest approximation, and pinning
    is the answer wherever the choice actually costs money.
    """

    #: Biggest window that qualifies -- proxy for "most capable".
    LARGEST_CONTEXT = "largest_context"
    #: Smallest window that still clears `min_context` -- proxy for "cheap tier".
    SMALLEST_CONTEXT = "smallest_context"


@dataclass(frozen=True)
class Candidate:
    """One proposal in a chain. Three forms, in decreasing determinism:

        pinned          spec="inception:mercury-2"   exact model
        provider query  provider="anthropic"         best match within one vendor
        open query      neither                      best match anywhere

    Pin when you need reproducibility. Query when the vendor churns model ids
    faster than you want to edit this file -- capabilities outlive ids.
    """

    spec: str | None = None
    #: Scopes a query to one vendor. Mutually exclusive with `spec`.
    provider: str | None = None
    requires: frozenset[Capability] = frozenset()
    endpoint: Endpoint = Endpoint.CHAT
    min_context: int | None = None
    #: Substring the model id must contain, matched case-insensitively.
    #: Vendors keep tier names stable across versions -- "flash", "haiku",
    #: "mini" have outlived several generations of version numbers -- so this
    #: expresses "the cheap tier" more reliably than a pin or a context bound.
    name_contains: str | None = None
    #: Tiebreak for queries. Ignored for a pin, which matches exactly one model.
    prefer: Preference = Preference.SMALLEST_CONTEXT
    params: Mapping[str, Any] = field(default_factory=dict)

    @property
    def provider_name(self) -> str | None:
        """Vendor this candidate resolves against, or None for an open query."""
        if self.spec is not None:
            head, sep, _ = self.spec.partition(":")
            return head if sep else None
        return self.provider

    @property
    def is_query(self) -> bool:
        return self.spec is None


# Chains are ordered: first viable candidate wins, the rest are the fallback path.
# Must be tuples -- a set literal both scrambles that order and fails to build,
# since `params` makes a frozen Candidate unhashable.
#
# `params` are bound per route and must be valid for that candidate's endpoint.
# The three Inception endpoints accept disjoint parameter sets:
#
#   chat  constructor kwargs for ChatInception -- temperature, max_tokens,
#         diffusing, realtime, reasoning_effort, reasoning_summary
#   fim   call kwargs -- max_tokens, stop, top_p, top_k, frequency_penalty,
#         presence_penalty, repetition_penalty   (no temperature)
#   edit  call kwargs -- max_tokens, temperature, top_p, presence_penalty
#
# Note the asymmetry: CHAT params reach a constructor, FIM/EDIT params reach a
# call. `stop` and `tools` are chat *invoke*-time options, not constructor
# fields, so they do not belong in a CHAT route's params.

TASK_ROUTES: dict[Task, tuple[Candidate, ...]] = {
    # COST POLICY. This is a leaderless swarm: every node routes independently,
    # so a flagship model in a chain is not one expensive call, it is one per
    # node per turn. Cheap tier only -- Mercury, Flash, Haiku, Mini.
    #
    # Tier is expressed with `name_contains`, not pins and not context bounds.
    # Pins go stale silently. Context fails outright as a cost proxy: every
    # Anthropic model has the same 200k window, so no context rule can tell
    # Haiku from Opus. Vendor tier names are the one stable, comparable signal.
    #
    # Mercury is pinned (ids verified against Inception's docs) and leads every
    # chat chain: diffusion decoding is the fastest and cheapest option here.
    # The others exist for a revoked or rate-limited Inception key.

    Task.CHAT_FAST: (
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.2, "reasoning_effort": "instant", "diffusing": True},
        ),
        Candidate(
            provider="gemini",
            name_contains="flash",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.2},
        ),
        Candidate(
            provider="anthropic",
            name_contains="haiku",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.2},
        ),
        Candidate(
            provider="openai",
            name_contains="mini",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.2},
        ),
    ),

    # Cheap models that can still think. Mercury's reasoning_effort is a dial on
    # a cheap model rather than a switch to an expensive one, so it stays first.
    Task.REASON: (
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.7, "reasoning_effort": "medium", "diffusing": True},
        ),
        Candidate(
            provider="anthropic",
            name_contains="haiku",
            requires=frozenset({Capability.CHAT, Capability.TOOLS, Capability.REASONING}),
            params={"temperature": 0.7},
        ),
        Candidate(
            provider="gemini",
            name_contains="flash",
            requires=frozenset({Capability.CHAT, Capability.TOOLS}),
            params={"temperature": 0.7},
        ),
    ),

    # Long-context planning. Flash leads on window size alone -- it is the only
    # cheap-tier model with a seven-figure context.
    Task.PLAN: (
        Candidate(
            provider="gemini",
            name_contains="flash",
            requires=frozenset({Capability.CHAT, Capability.TOOLS}),
            min_context=900_000,
            params={"temperature": 0.4},
        ),
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.4, "reasoning_effort": "high", "diffusing": True},
        ),
        Candidate(
            provider="anthropic",
            name_contains="haiku",
            requires=frozenset({Capability.CHAT, Capability.TOOLS}),
            min_context=150_000,
            params={"temperature": 0.4},
        ),
    ),

    # Wide and uncreative. min_context is the floor; the cheap tier supplies it.
    Task.SUMMARIZE: (
        Candidate(
            provider="gemini",
            name_contains="flash",
            requires=frozenset({Capability.CHAT}),
            min_context=500_000,
            params={"temperature": 0.1},
        ),
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.1, "reasoning_effort": "instant", "diffusing": True},
        ),
        Candidate(
            provider="openai",
            name_contains="mini",
            requires=frozenset({Capability.CHAT}),
            min_context=100_000,
            params={"temperature": 0.1},
        ),
    ),

    # No fallback exists for either of these -- no other vendor implements the
    # endpoints. `validate()` enforces that, so nobody can add one that looks
    # like coverage.
    Task.CODE_COMPLETE: (
        Candidate(
            spec="inception:mercury-edit-2",
            requires=frozenset({Capability.FIM}),
            endpoint=Endpoint.FIM,
            params={"max_tokens": 256, "presence_penalty": 1.5},
        ),
    ),
    Task.CODE_EDIT: (
        Candidate(
            spec="inception:mercury-edit-2",
            requires=frozenset({Capability.EDIT}),
            endpoint=Endpoint.EDIT,
            params={"temperature": 0.4, "max_tokens": 1024},
        ),
    ),
}

KNOWN_PROVIDERS = frozenset({"inception", "openai", "anthropic", "gemini"})
INCEPTION_ONLY_ENDPOINTS = frozenset({Endpoint.FIM, Endpoint.EDIT})
ENDPOINT_CAPABILITY: dict[Endpoint, Capability] = {
    Endpoint.CHAT: Capability.CHAT,
    Endpoint.FIM: Capability.FIM,
    Endpoint.EDIT: Capability.EDIT,
}

#: Deliberately named for the vendor it describes. These are Inception's
#: parameter sets, not a general truth -- when a second FIM provider appears,
#: this name is what tells you the table needs rethinking.
#:
#: `model`, `prompt`, `suffix` and `messages` are absent on purpose: the provider
#: wrapper supplies them positionally, so a route listing one would pass it twice
#: and raise "got multiple values for argument".
INCEPTION_PARAMS_BY_ENDPOINT: dict[Endpoint, frozenset[str]] = {
    Endpoint.CHAT: frozenset({
        "temperature", "max_tokens", "diffusing", "realtime",
        "reasoning_effort", "reasoning_summary", "model_kwargs",
    }),
    Endpoint.FIM: frozenset({
        "max_tokens", "stop", "top_p", "top_k",
        "frequency_penalty", "presence_penalty", "repetition_penalty",
    }),
    Endpoint.EDIT: frozenset({
        "max_tokens", "temperature", "top_p", "presence_penalty",
    }),
}


class MappingError(Exception): ...


def validate(routes: Mapping[Task, tuple[Candidate, ...]] = TASK_ROUTES) -> None:
    """Check the routing table's shape. Collects every problem, then raises once.

    Shape only. Whether a model *exists* needs a live catalogue, so that check
    belongs to the Router -- same split as `is_configured()` vs `check()`.
    """
    problems: list[str] = []

    for task in Task:
        chain = routes.get(task)
        if chain is None:
            problems.append(f"{task!r}: no route defined")
            continue
        if not isinstance(chain, tuple):
            problems.append(f"{task!r}: chain is {type(chain).__name__}, not tuple")
            continue
        if not chain:
            problems.append(f"{task!r}: chain is empty")
            continue

        for i, c in enumerate(chain):
            where = f"{task!r}[{i}]"

            if not isinstance(c, Candidate):
                problems.append(f"{where}: {type(c).__name__}, not a Candidate")
                continue

            if c.name_contains is not None and (
                not isinstance(c.name_contains, str) or not c.name_contains.strip()
            ):
                problems.append(
                    f"{where}: name_contains must be a non-empty string, "
                    f"got {c.name_contains!r}"
                )

            if not isinstance(c.prefer, Preference):
                problems.append(f"{where}: prefer {c.prefer!r} is not a Preference")

            if not isinstance(c.endpoint, Endpoint):
                problems.append(f"{where}: endpoint {c.endpoint!r} is not an Endpoint")
                continue

            # --- spec / provider: pinned, provider-scoped query, or open query ---
            provider: str | None = None
            if c.spec is not None and c.provider is not None:
                problems.append(
                    f"{where}: set spec or provider, not both "
                    f"(spec={c.spec!r}, provider={c.provider!r})"
                )
            if c.spec is None:
                if not c.requires:
                    problems.append(
                        f"{where}: query candidate must set requires, "
                        f"or it matches every model"
                    )
                if c.provider is not None:
                    if c.provider in KNOWN_PROVIDERS:
                        provider = c.provider
                    else:
                        problems.append(
                            f"{where}: unknown provider {c.provider!r}; "
                            f"known: {sorted(KNOWN_PROVIDERS)}"
                        )
            else:
                provider, sep, model = c.spec.partition(":")
                if not sep:
                    problems.append(f"{where}: spec {c.spec!r} is not 'provider:model'")
                    provider = None
                elif ":" in model:
                    problems.append(f"{where}: spec {c.spec!r} has more than one ':'")
                    provider = None
                elif provider not in KNOWN_PROVIDERS:
                    problems.append(
                        f"{where}: unknown provider {provider!r}; "
                        f"known: {sorted(KNOWN_PROVIDERS)}"
                    )
                    provider = None
                elif not model:
                    problems.append(f"{where}: spec {c.spec!r} has an empty model id")

            # --- requires ---
            if not isinstance(c.requires, frozenset):
                problems.append(
                    f"{where}: requires is {type(c.requires).__name__}, not frozenset"
                )
            else:
                loose = sorted(r for r in c.requires if not isinstance(r, Capability))
                if loose:
                    problems.append(f"{where}: requires holds non-Capability values {loose}")

            need = ENDPOINT_CAPABILITY[c.endpoint]
            if need not in c.requires:
                problems.append(
                    f"{where}: endpoint {c.endpoint.value!r} needs "
                    f"{need.value!r} in requires"
                )

            # --- endpoints only Inception implements ---
            if c.endpoint in INCEPTION_ONLY_ENDPOINTS:
                if c.spec is None and c.provider is None:
                    problems.append(
                        f"{where}: endpoint {c.endpoint.value!r} cannot be an open "
                        f"query -- only inception implements it"
                    )
                elif provider is not None and provider != "inception":
                    problems.append(
                        f"{where}: endpoint {c.endpoint.value!r} is inception-only, "
                        f"got provider {provider!r}"
                    )

            # --- min_context ---
            if c.min_context is not None:
                # bool is a subclass of int, so isinstance alone lets True through.
                if not isinstance(c.min_context, int) or isinstance(c.min_context, bool):
                    problems.append(
                        f"{where}: min_context must be an int, got {c.min_context!r}"
                    )
                elif c.min_context <= 0:
                    problems.append(
                        f"{where}: min_context must be positive, got {c.min_context}"
                    )

            # --- params, checked only where we actually know the parameter set ---
            if provider == "inception":
                illegal = sorted(set(c.params) - INCEPTION_PARAMS_BY_ENDPOINT[c.endpoint])
                if illegal:
                    problems.append(
                        f"{where}: params {illegal} not accepted by inception "
                        f"{c.endpoint.value!r}"
                    )

        # --- chain level: a query candidate shadows every pin after it ---
        # Only an *open* query shadows what follows -- it can match any vendor.
        # A provider-scoped query only ever competes within its own vendor, so
        # several of them in one chain is the normal, intended shape.
        open_queries = [
            i for i, c in enumerate(chain)
            if isinstance(c, Candidate) and c.spec is None and c.provider is None
        ]
        if len(open_queries) > 1:
            problems.append(
                f"{task!r}: {len(open_queries)} open query candidates; at most one allowed"
            )
        elif open_queries and open_queries[0] != len(chain) - 1:
            problems.append(
                f"{task!r}: open query at index {open_queries[0]} shadows every "
                f"candidate after it; it must be last"
            )

    if problems:
        raise MappingError("invalid TASK_ROUTES:\n  " + "\n  ".join(problems))


validate()
