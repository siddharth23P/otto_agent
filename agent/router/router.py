import contextvars
import weakref
import logging
from concurrent.futures import ThreadPoolExecutor

from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from contextlib import contextmanager
from dataclasses import dataclass

from langchain.chat_models import BaseChatModel

from agent.router.llm_provider import reset as registry_reset
from agent.router.llm_provider import get_provider
from agent.router.llm_provider import provider_class, provider_names
from agent.router.llm_provider.temperature import apply_to_params
from agent.router.llm_provider.base import AuthError, Capability, CapabilityNotSupported, ModelInfo, ProviderError
from agent.router import outcomes as seat_outcomes
from agent.router import health as provider_health
from agent.router import overrides as route_overrides
from agent.router.mapping import TASK_ROUTES, Candidate, Endpoint, Preference, Task

_log = logging.getLogger(__name__)

@dataclass(frozen=True,slots=True)
class Skip:
    index: int      # position in the chain — tells you which line of mapping.py
    target: str     # "inception:mercury-2.5"
    reason: str     # "INCEPTION_API_KEY not set"
    
    def __str__(self) -> str:
        return f"[{self.index}] {self.target}: {self.reason}"

def render(c: Candidate) -> str:
    """One-line label for a candidate, for logs and Skip records."""
    if c.spec:
        return c.spec
    tier = f"*{c.name_contains}*" if c.name_contains else "*"
    return f"{c.provider or 'any'}:{tier}"

@dataclass(frozen=True,slots=True)
class RoutingDecision:
    task: Task
    provider: str               # "inception"
    model: ModelInfo            # the real catalogue entry, not just an id
    endpoint: Endpoint          # which provider method to call
    params: Mapping[str, Any]   # from the winning candidate
    index: int                  # which candidate won
    skipped: tuple[Skip, ...]   # everything tried before it

    @property
    def fell_back(self) -> bool:
        """Something ahead of this candidate was TRIED and could not serve.

        Not `index > 0`. Since agent/router/outcomes.py reorders the chain
        from observed results, a later candidate can be chosen first on
        purpose -- and reporting that as "degraded", which is what every
        reader of this does, would call the router's best-evidenced choice a
        failure. Degradation is about candidates that were skipped, so ask
        that."""
        return bool(self.skipped)

    @property
    def chosen_on_evidence(self) -> bool:
        """Picked ahead of a candidate declared above it, nothing having
        failed. The reason `otto route` can explain an order that does not
        match mapping.py."""
        return self.index > 0 and not self.skipped
    
class NoViableRoute(ProviderError):
    def __init__(self, task: Task, skipped: tuple[Skip, ...]):
        self.task, self.skipped = task, skipped
        detail = "\n  ".join(str(s) for s in skipped) or "chain was empty"
        super().__init__(f"no viable model for {task.value}:\n  {detail}")
        
@runtime_checkable
class Catalogue(Protocol):
    def is_configured(self, provider: str) -> bool: ...
    def models(self, provider: str) -> list[ModelInfo]: ...
    def reset(self) -> None: ...
    
    
class RegistryCatalogue:
    def is_configured(self, provider: str) -> bool:
        # A vendor whose SDK is not installed is not configured, rather than
        # a broken router: langchain_openai needs tiktoken, langchain_anthropic
        # needs jiter, and an embedded Otto (agent/embed.py) may ship without
        # one of them. `health_report` names the import failure; here it only
        # has to not take the other providers down with it.
        try:
            cls = provider_class(provider)
        except ImportError as exc:
            # Only the adapter import is inside the try: is_configured()
            # reads the environment and must not have a bug of its own
            # read as "vendor not installed".
            _log.warning("provider %s is unavailable: %s", provider, exc)
            return False
        return cls.is_configured()
    def models(self, provider: str) -> list[ModelInfo]:
        return get_provider(provider).list_models()
    def reset(self) -> None:
        registry_reset()

@dataclass
class FakeCatalogue:
    data: dict[str, list[ModelInfo]]
    def is_configured(self, provider): return provider in self.data
    def models(self, provider):        return self.data[provider]
    def reset(self) -> None: pass
    
def _decision_metadata(d: "RoutingDecision") -> dict[str, Any]:
    """The routing facts worth filtering a trace by, in one place.

    Kept next to the router rather than in the CLI so a LangGraph node calling
    fim()/code_edit() directly gets the same fields a REPL turn does.
    """
    return {
        "otto_task": d.task.value,
        "otto_provider": d.provider,
        "otto_endpoint": d.endpoint.value,
        "otto_candidate": d.index,
        "otto_fell_back": d.fell_back,
        "otto_context_window": d.model.context_window,
        "otto_skipped": [str(s) for s in d.skipped],
    }


@contextmanager
def _observe(name: str, *, model: str, input: Any,
             model_parameters: Mapping[str, Any],
             metadata: Mapping[str, Any] | None = None):
    """Wrap a raw-SDK call in a Langfuse generation, or do nothing.

    `fim()` and `code_edit()` bypass LangChain, and the Langfuse callback hooks
    the runnable interface -- so without this they are invisible in a trace that
    otherwise shows every chat call.

    Deliberately best-effort: the router must not stop routing because tracing
    is unavailable, so an absent package or an unconfigured client yields None
    and the call proceeds untraced.
    """
    try:
        from langfuse import get_client

        with get_client().start_as_current_observation(
            as_type="generation",
            name=name,
            model=model,
            input=input,
            model_parameters=dict(model_parameters),
            metadata=dict(metadata or {}),
        ) as generation:
            yield generation
        return
    except Exception:                     # not installed, or no keys
        pass
    yield None


def select_candidate(pool: Sequence[ModelInfo], c: Candidate) -> ModelInfo | None:
    """The model in `pool` that `c` names, or None.

    Module-level rather than a Router method so agent/router/automap.py can
    ask the same question of a detected pool without a router.
    """
    if c.spec is not None:
        # A pin names one exact model, not a tier to search within -- once
        # a vendor has two generations live at once (mercury-2 alongside
        # mercury-2.5, say), matching by capability/name_contains/context
        # like a query would could silently resolve a "pinned" candidate
        # to the WRONG generation depending on which one sorts first under
        # `prefer`. Match the id from the spec directly instead; `requires`
        # still gates it, so a pin to a model that lost a capability fails
        # loudly ("no model matched") rather than serving it anyway.
        _, _, model_id = c.spec.partition(":")
        return next(
            (m for m in pool if m.id == model_id and c.requires <= m.capabilities),
            None,
        )
    matches = [m for m in pool
               if c.requires <= m.capabilities
               and (c.name_contains is None
                    or c.name_contains.lower() in m.id.lower())
               and (c.min_context is None
                    or (m.context_window or 0) >= c.min_context)]
    if not matches:
        return None
    biggest = c.prefer is Preference.LARGEST_CONTEXT
    return sorted(matches,
                  key=lambda m: (m.context_window or 0, m.id),
                  reverse=biggest)[0]


class Router:
    """Every configured provider is usable (2026-09-11).

    This was a two-vendor router until now: `_usable()` admitted `REQUIRED` and
    exactly one `secondary`, the first configured member of `OPTIONAL`, and
    anything else with a valid key was skipped as "not the selected secondary".
    That single-secondary rule existed to give Phase 8's plain hive
    (`agent/graph/nodes.py`, `agent/graph/run.py`) one alternate vendor for
    seat diversity -- and `agent/graph/` has since been removed. The only
    readers left were two display rows in `agent/cli/doctor.py`.

    It has to go, because the routing table now names four vendors at once:
    Anthropic judges and plans, OpenAI solves, Gemini summarises and reads
    images, Inception keeps chat-fast plus the FIM/edit endpoints no other
    vendor here serves. Under the old rule three of those four would silently
    never be reached.

    `REQUIRED` stays. Inception is still the one provider Otto cannot start
    without -- it alone serves `Endpoint.FIM`/`Endpoint.EDIT`
    (`mapping.py`'s `INCEPTION_ONLY_ENDPOINTS`), and a missing key there is a
    broken install rather than a degraded one. Every other vendor is optional
    in the real sense: configure it and its routes resolve, leave it out and
    its routes are skipped with a legible reason.
    """

    REQUIRED = "inception"

    #: Every router constructed in this process, so a key added at runtime
    #: (the TUI's setup screen, agent/router/reload.py) can re-snapshot ALL
    #: of them -- agent/pipeline/nodes.py holds one as a module global that
    #: four other modules bound by name, so rebinding is not an option and
    #: mutating in place is the only reload that reaches everyone.
    _LIVE: "weakref.WeakSet[Router]" = weakref.WeakSet()

    def _snapshot(self) -> None:
        # No longer raises when Inception is missing (2026-09-12): construction
        # happens at import time in nodes.py, and a raise there meant `otto
        # tui` could not open on a machine with no keys -- which is exactly the
        # machine that needs the setup screen. The requirement still holds,
        # enforced by `require_ready()` at the first resolve and at the
        # pipeline's entry points, with the same message as before.
        self._configured = tuple(p for p in provider_names() if self.catalogue.is_configured(p))

    def __init__(self, catalogue: Catalogue | None = None, *, strict: bool = False):
        self.catalogue = catalogue or RegistryCatalogue()
        self.strict = strict
        self._snapshot()
        Router._LIVE.add(self)

    def reset(self):
        self.catalogue.reset()
        self._snapshot()

    @classmethod
    def reset_all(cls) -> None:
        """Re-snapshot every live router after keys or providers changed."""
        for router in list(cls._LIVE):
            router.reset()

    def ready(self) -> bool:
        """Whether the one provider otto cannot run without is configured."""
        return self.REQUIRED in self._configured

    def require_ready(self) -> None:
        if not self.ready():
            raise AuthError("Otto requires Inception. Set INCEPTION_API_KEY in .env")

    def _usable(self, provider: str) -> bool:
        return provider in self._configured

    def usable(self) -> tuple[str, ...]:
        return tuple(p for p in self._configured if self._usable(p))

    def prewarm(self) -> dict[str, str]:
        """Fetch every configured provider's catalogue, all at once.

        They used to be fetched one after another, so the first turn of a
        process waited for the sum of four vendors' model-list round trips
        rather than the slowest one. Each is independent and cached after
        (get_provider's lru_cache, each provider's own list cache). Same
        result, same failures reported."""
        names = self.usable()
        failures: dict[str, str] = {}
        if not names:
            return failures

        def fetch(name: str) -> str | None:
            try:
                self.catalogue.models(name)
            except ProviderError as exc:
                return str(exc)
            return None

        with ThreadPoolExecutor(max_workers=len(names), thread_name_prefix="otto-prewarm") as pool:
            # One context copy per task: a Context cannot be entered by two
            # threads at once.
            futures = [pool.submit(contextvars.copy_context().run, fetch, name) for name in names]
            for name, future in zip(names, futures):
                if (problem := future.result()) is not None:
                    failures[name] = problem
        return failures
    
    def _match(self, c: Candidate, only: str | None = None, *,
               honour_cooldowns: bool = True) -> ModelInfo | str:
        provider = c.provider_name
        if only is not None and provider != only:
            return f"{provider} is not the pinned vendor"
        if provider is None:
            return "open queries not supported"
        if provider not in self._configured:
            return f"{provider.upper()}_API_KEY not set"
        if not self._usable(provider):
            return f"{provider} is not the selected secondary"
        
        pool = self.catalogue.models(provider)
        model = self._select(pool, c)
        if model is None:
            return "no model matched"
        # Last, and only when `cooling` is being honoured -- a candidate that
        # is merely unwell is still the right answer when the alternative is
        # no answer, so `resolve` re-runs the chain ignoring this if every
        # candidate was skipped for it.
        # Through the MODULE, not a name bound at import: `bind_health`
        # swaps the module global, and a captured reference would
        # silently keep consulting the one this process started with.
        if honour_cooldowns and (why := provider_health.HEALTH.cooling(provider, model.id)):
            return why
        return model

    def _select(self, pool, c: Candidate) -> ModelInfo | None:
        return select_candidate(pool, c)

    def resolve(self, task: Task, *, only: str | None = None) -> RoutingDecision:
        """The first viable candidate, preferring one that is not cooling.

        Two passes, because a circuit breaker that leaves a task with no route
        has turned a slow provider into a broken agent. The first pass skips
        anything agent/router/health.py says is unwell; if that leaves nothing,
        the second takes the chain as it stands. Cooldowns are a preference,
        never a prohibition.
        """
        self.require_ready()
        try:
            return self._resolve(task, only=only, honour_cooldowns=True)
        except NoViableRoute as exc:
            if not any("cooling" in s.reason for s in exc.skipped):
                raise
            # Everything viable was merely unwell. Ask again anyway.
            return self._resolve(task, only=only, honour_cooldowns=False)

    def _resolve(self, task: Task, *, only: str | None,
                 honour_cooldowns: bool) -> RoutingDecision:
        skips: list[Skip] = []
        # A seat a host bound for this run (overrides.bind_seats -- the phone
        # host's fast judge) leads the live chain; otherwise the live chain.
        declared = route_overrides.bound_chain(task) or TASK_ROUTES[task]
        # Declared order, re-ordered by what this installation has actually
        # observed each seat's model achieve. A no-op until a (task, model)
        # pair has enough runs behind it to be trusted, which is most of the
        # time -- see agent/router/outcomes.py for why the bar is where it is.
        # A pin the person set (agent/router/overrides.py) is never reordered
        # or explored away: "use this model" means this model, and an
        # evidence-based swap behind their back would make the pin a lie.
        # A bound seat is the host's explicit choice and is held the same way.
        if declared and (route_overrides.is_pin(task, declared[0])
                         or route_overrides.is_bound(task, declared[0])):
            chain = list(declared)
        else:
            chain = seat_outcomes.reorder(task.value, declared)
        # `index` stays an index into the DECLARED chain, never into the tried
        # order. Everything that reads it -- the tree `otto route`
        # prints -- is asking "which candidate in mapping.py is this", and an
        # index into a list the reader cannot see would answer a different
        # question while looking like the same one.
        position = {id(c): i for i, c in enumerate(declared)}
        for tried, c in enumerate(chain):
            i = position[id(c)]
            outcome = self._match(c, only=only,
                                  honour_cooldowns=honour_cooldowns)
            if isinstance(outcome, str):
                skips.append(Skip(i, render(c), outcome))
                continue

            if self.strict and tried > 0:
                raise RoutingDegraded(task, tuple(skips))

            return RoutingDecision(
                task=task, provider=outcome.provider, model=outcome,
                endpoint=c.endpoint, params=dict(c.params),
                index=i, skipped=tuple(skips),
            )
        raise NoViableRoute(task, tuple(skips))
    
    def chain(self, task: Task) -> list[RoutingDecision]:
        """Every candidate this installation could use for `task`, in the order to try them: the
        ones not cooling first, as `resolve` would pick them, then the cooling ones. For a caller
        that moves down the chain when a call fails (agent/pipeline/vision.py), where `resolve`
        only knows what was already known to be unwell."""
        self.require_ready()
        declared = route_overrides.bound_chain(task) or TASK_ROUTES[task]
        ready, cooling = [], []
        for i, c in enumerate(declared):
            outcome = self._match(c, honour_cooldowns=False)
            if isinstance(outcome, str):
                continue
            decision = RoutingDecision(task=task, provider=outcome.provider, model=outcome,
                                       endpoint=c.endpoint, params=dict(c.params), index=i, skipped=())
            (cooling if provider_health.HEALTH.cooling(outcome.provider, outcome.id) else ready).append(decision)
        return ready + cooling

    def model_for(self, d: RoutingDecision, **overrides) -> BaseChatModel:
        if d.endpoint is not Endpoint.CHAT:
            raise CapabilityNotSupported(f"{d.task.value} routes to {d.endpoint.value}, not chat")
        provider = get_provider(d.provider)
        # Temperature is decided per MODEL, not per call site and not per
        # vendor: Inception silently resets anything under 0.5 to 1.0, and
        # OpenAI's reasoning models reject any value at all. See
        # agent/router/llm_provider/temperature.py.
        params = apply_to_params(d.provider, d.model.id, {**d.params, **overrides}, d.model)
        llm = provider.chat_model(d.model.id, **params)
        # Stamped so a failure downstream can name the vendor this came from:
        # agent/pipeline/nodes.py's _call needs it to record a retirement
        # against the right catalogue. Defensive, because not every chat class
        # tolerates an unknown attribute.
        try:
            object.__setattr__(llm, "_otto_provider", d.provider)
        except Exception:  # pragma: no cover -- slotted/frozen model classes
            pass
        return llm
    
    def chat_model(self, task: Task, *, only: str | None = None, **overrides) -> BaseChatModel:
        return self.model_for(self.resolve(task,only=only), **overrides)
    
    def fim(self, prefix: str, suffix: str = "", *,task: Task = Task.CODE_COMPLETE, **overrides) -> str:
        d = self.resolve(task)
        if d.endpoint is not Endpoint.FIM:
            raise CapabilityNotSupported(f"{task.value} is not a FIM route")
        provider = get_provider(d.provider)
        provider.require(Capability.FIM)
        params = {**d.params, **overrides}
        with _observe("inception.fim", model=d.model.id, input=prefix,
                      model_parameters=params,
                      metadata=_decision_metadata(d)) as generation:
            result = provider.fim(d.model.id, prefix, suffix, **params)
            if generation is not None:
                generation.update(output=result.text, usage_details=result.usage)
            # The router's own contract stays `str`: a graph node wants the
            # completion, not a usage envelope. The envelope exists so the span
            # above can be costed.
            return result.text
    
    def code_edit(self, code_to_edit: str, *, current_file: str = "",
                  recently_viewed: Sequence[str] = (), edit_history: Sequence[str] = (),
                  task: Task = Task.CODE_EDIT, **overrides) -> str:
        d = self.resolve(task)
        if d.endpoint is not Endpoint.EDIT:
            raise CapabilityNotSupported(f"{task.value} is not an edit route")
        provider = get_provider(d.provider)
        provider.require(Capability.EDIT)
        params = {**d.params, **overrides}
        with _observe("inception.code_edit", model=d.model.id, input=code_to_edit,
                      model_parameters=params,
                      metadata=_decision_metadata(d)) as generation:
            result = provider.code_edit(
                d.model.id, code_to_edit,
                current_file=current_file,
                recently_viewed=recently_viewed,
                edit_history=edit_history,
                **params)
            if generation is not None:
                generation.update(output=result.text, usage_details=result.usage)
            return result.text
        

class RoutingDegraded(ProviderError):
    def __init__(self, task: Task, skipped: tuple[Skip, ...]):
        self.task, self.skipped = task, skipped
        detail = "\n  ".join(str(s) for s in skipped)
        super().__init__(
            f"{task.value} fell back past its preferred candidate:\n  {detail}")
