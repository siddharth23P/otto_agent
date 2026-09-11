"""Static routing intent.

Imports only `Capability` from the provider base -- no registry, no vendor SDKs.
That is what lets routing be tested with an empty .env and no network.
"""

from typing import Any, Mapping

from agent.router.llm_provider.base import Capability
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

    Nothing in today's `TASK_ROUTES` exercises this -- every candidate is now
    a pinned Inception spec, and a pin resolves to its exact model id
    (`Router._select`), never a tiebreak. Kept for the day a query candidate
    (e.g. "whichever Mercury variant is smallest") earns its place again --
    removing the mechanism to save a few lines now would just mean rebuilding
    it later under time pressure.
    """

    #: Biggest window that qualifies -- proxy for "most capable".
    LARGEST_CONTEXT = "largest_context"
    #: Smallest window that still clears `min_context` -- proxy for "cheap tier".
    SMALLEST_CONTEXT = "smallest_context"


@dataclass(frozen=True)
class Candidate:
    """One proposal in a chain. Three forms, in decreasing determinism:

        pinned          spec="inception:mercury-2.5"   exact model
        provider query  provider="inception"           best match within one vendor
        open query      neither                        best match anywhere

    Pin when you need reproducibility. Query when the vendor churns model ids
    faster than you want to edit this file -- capabilities outlive ids. Every
    route in `TASK_ROUTES` today is a pin: Inception is the only vendor, and
    its two chat-capable generations (mercury-2, mercury-2.5) are different
    enough in quality that "whichever one exists" is never what a route wants.
    """

    spec: str | None = None
    #: Scopes a query to one vendor. Mutually exclusive with `spec`.
    provider: str | None = None
    requires: frozenset[Capability] = frozenset()
    endpoint: Endpoint = Endpoint.CHAT
    min_context: int | None = None
    #: Substring the model id must contain, matched case-insensitively.
    #: Ignored on a pinned candidate -- see `Router._select`.
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


# Chains are ordered: first viable candidate wins, the rest are the fallback
# path. Must be tuples -- a set literal both scrambles that order and fails to
# build, since `params` makes a frozen Candidate unhashable.
#
# Otto is Inception-only (2026-09-09: "going all in with Mercury" -- the
# multi-vendor fallback chains from Phases 2-7 are gone, along with the
# anthropic/openai/gemini provider modules themselves). Every chain below is
# a single pinned candidate; there is no secondary vendor left to fall back
# to, so `NoViableRoute` now means exactly one thing: Inception itself is
# unreachable or the pinned id has gone stale.
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
#
# `diffusing=True` streams the *denoising effect*: each chunk is a full
# snapshot of the whole answer, progressively less noisy, NOT the next slice
# of text. A consumer that appends chunks concatenates every refinement step
# and produces garbage -- the renderer must REPLACE the previous frame.
# `chat.py` selects its renderer on this flag.

TASK_ROUTES: dict[Task, tuple[Candidate, ...]] = {
    # `reasoning_effort` is the dial every chat route tunes, cheapest to most
    # thoughtful: "instant" for a swarm node's fast turnaround (classify,
    # review), "medium"/"high" for the calls that actually have to think
    # (reconcile, plan) -- one model, four settings, not four models.
    Task.CHAT_FAST: (
        Candidate(
            spec="inception:mercury-2.5",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.2, "reasoning_effort": "instant", "diffusing": True},
        ),
    ),

    Task.REASON: (
        Candidate(
            spec="inception:mercury-2.5",
            requires=frozenset({Capability.CHAT}),
            # max_tokens set explicitly, generous: this route authors whole
            # code files and reconciles (stitches) them, and an unset
            # max_tokens fell back to whatever Inception defaults to
            # unspecified -- too low for a real script, and a diffusing call
            # cut off at length is not a clean prefix, it's an unconverged
            # snapshot (code_nodes.py's _call() retries on that finish_reason,
            # this just makes hitting it in the first place rarer).
            #
            # 2026-09-09 flipped diffusing False and bumped reasoning_effort
            # to "high" while chasing malformed tool-calling replies -- a bare
            # fragment with neither "ACTION:" nor "FINAL:" in it -- and said
            # explicitly: "Revert if it doesn't measurably help."
            #
            # 2026-09-10, measured: "high" is the cause, not the cure. Six
            # trials per setting on a real solver-shaped conversation,
            # counting replies with any content at all, and replies that were
            # well-formed:
            #
            #     diffusing  effort    non-empty  well-formed   avg
            #       False    high         2/6         2/6      13.8s
            #       True     high         1/6         1/6      10.5s
            #       False    medium       6/6         6/6       6.3s
            #       True     medium       6/6         6/6       9.7s
            #       False    (unset)      6/6         6/6      10.0s
            #       True     (unset)      6/6         6/6       7.2s
            #
            # At "high" the stream yields two chunks, no content and no
            # finish_reason. That empty reply IS the malformed fragment the
            # experiment was chasing -- _parse_worker_reply can only call it
            # unparseable -- so asking for more deliberation made the very bug
            # it was aimed at four times worse. It also wrecked the agent's
            # ratio of thinking to doing: on a hard task the solver spent 39
            # of its 40 tool-loop turns on empty replies and issued 2 real
            # commands in six minutes.
            #
            # "medium" it is: every reply usable, and the fastest of the
            # working settings. `diffusing` is left where the experiment put
            # it -- nothing here shows it doing harm, and changing two things
            # at once is how this became hard to attribute in the first place.
            params={"temperature": 0.7, "reasoning_effort": "medium",
                    "diffusing": False, "max_tokens": 8192},
        ),
    ),

    Task.PLAN: (
        Candidate(
            spec="inception:mercury-2.5",
            requires=frozenset({Capability.CHAT}),
            # "medium", not "high", for the reason spelled out on Task.REASON
            # above: at "high" this model returns an empty stream most of the
            # time. Measured on REASON's prompt shape rather than this one,
            # but it is the same model and endpoint, and 1-2 usable replies
            # out of 6 is not a risk worth carrying here either.
            params={"temperature": 0.4, "reasoning_effort": "medium",
                    "diffusing": True, "max_tokens": 4096},
        ),
    ),

    Task.SUMMARIZE: (
        Candidate(
            spec="inception:mercury-2.5",
            requires=frozenset({Capability.CHAT}),
            params={"temperature": 0.1, "reasoning_effort": "instant", "diffusing": True},
        ),
    ),

    # No fallback exists for either of these -- no other vendor implements the
    # endpoints. `validate()` enforces that, so nobody can add one that looks
    # like coverage. Unchanged by the Mercury 2.5 switch: Inception's docs
    # list no mercury-edit-2.5, only mercury-edit-2, for FIM/edit.
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

KNOWN_PROVIDERS = frozenset({"inception", "anthropic", "openai", "gemini"})
INCEPTION_ONLY_ENDPOINTS = frozenset({Endpoint.FIM, Endpoint.EDIT})
ENDPOINT_CAPABILITY: dict[Endpoint, Capability] = {
    Endpoint.CHAT: Capability.CHAT,
    Endpoint.FIM: Capability.FIM,
    Endpoint.EDIT: Capability.EDIT,
}

#: Deliberately named for the vendor it describes. These are Inception's
#: parameter sets, not a general truth -- see PARAMS_BY_PROVIDER below, which
#: is the table that grew when the second vendor actually arrived.
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


#: What each vendor's chat model actually accepts, so a route carrying the
#: wrong kwarg fails at import rather than on the first live call.
#:
#: This matters more than it looks. Every CHAT route here used to carry
#: `reasoning_effort` and `diffusing`, which are `ChatInception` constructor
#: arguments -- handing either to `ChatOpenAI` or `ChatAnthropic` raises. The
#: old validation ran only `if provider == "inception"`, so a non-Inception
#: route's params were completely unchecked and the mistake would surface as a
#: TypeError deep inside a node, mid-run.
#:
#: A provider absent from this table is not validated, which is the honest
#: default for a vendor whose parameter set nobody here has written down yet.
PARAMS_BY_PROVIDER: dict[str, dict[Endpoint, frozenset[str]]] = {
    "inception": INCEPTION_PARAMS_BY_ENDPOINT,
    # LangChain's ChatAnthropic / ChatOpenAI / ChatGoogleGenerativeAI share
    # this much; anything vendor-specific goes through `model_kwargs`.
    "anthropic": {Endpoint.CHAT: frozenset({
        "temperature", "max_tokens", "top_p", "top_k", "timeout",
        "stop", "model_kwargs", "thinking",
    })},
    "openai": {Endpoint.CHAT: frozenset({
        "temperature", "max_tokens", "max_completion_tokens", "top_p",
        "timeout", "stop", "model_kwargs", "reasoning_effort",
    })},
    "gemini": {Endpoint.CHAT: frozenset({
        "temperature", "max_tokens", "max_output_tokens", "top_p", "top_k",
        "timeout", "model_kwargs",
    })},
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
            known = PARAMS_BY_PROVIDER.get(provider or "", {}).get(c.endpoint)
            if known is not None:
                illegal = sorted(set(c.params) - known)
                if illegal:
                    problems.append(
                        f"{where}: params {illegal} not accepted by {provider} "
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
