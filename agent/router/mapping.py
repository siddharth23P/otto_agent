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


@dataclass(frozen=True)
class Candidate:
    spec: str | None
    requires: frozenset[Capability]
    endpoint: Endpoint = Endpoint.CHAT
    min_context: int | None = None
    params: Mapping[str, Any] = field(default_factory=dict)


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
    Task.CHAT_FAST: (
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            endpoint=Endpoint.CHAT,
            params={"temperature": 0.2, "reasoning_effort": "instant", "diffusing": True},
        ),
    ),
    Task.REASON: (
        Candidate(
            spec="inception:mercury-2",
            requires=frozenset({Capability.CHAT}),
            endpoint=Endpoint.CHAT,
            params={"temperature": 0.7, "reasoning_effort": "medium", "diffusing": True},
        ),
    ),
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
            params={"temperature": 0.2, "max_tokens": 1024},
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

            if not isinstance(c.endpoint, Endpoint):
                problems.append(f"{where}: endpoint {c.endpoint!r} is not an Endpoint")
                continue

            # --- spec: None means "query the catalogue", anything else must parse ---
            provider: str | None = None
            if c.spec is None:
                if not c.requires:
                    problems.append(
                        f"{where}: query candidate (spec=None) must set requires, "
                        f"or it matches every model"
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
                if c.spec is None:
                    problems.append(
                        f"{where}: endpoint {c.endpoint.value!r} cannot be a query "
                        f"candidate -- only inception implements it"
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
        queries = [i for i, c in enumerate(chain)
                   if isinstance(c, Candidate) and c.spec is None]
        if len(queries) > 1:
            problems.append(
                f"{task!r}: {len(queries)} query candidates; at most one allowed"
            )
        elif queries and queries[0] != len(chain) - 1:
            problems.append(
                f"{task!r}: query candidate at index {queries[0]} shadows the pinned "
                f"candidates after it; it must be last"
            )

    if problems:
        raise MappingError("invalid TASK_ROUTES:\n  " + "\n  ".join(problems))


validate()
