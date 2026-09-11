from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from contextlib import contextmanager
from dataclasses import dataclass

from langchain.chat_models import BaseChatModel

from agent.router.llm_provider import reset as registry_reset
from agent.router.llm_provider import get_provider, provider_class, provider_names
from agent.router.llm_provider.base import AuthError, Capability, CapabilityNotSupported, ModelInfo, ProviderError
from agent.router.mapping import TASK_ROUTES, Candidate, Endpoint, Preference, Task

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
        return self.index > 0
    
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
        return provider_class(provider).is_configured()
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


class Router:
    """Every configured provider is usable (2026-09-11).

    This was a two-vendor router until now: `_usable()` admitted `REQUIRED` and
    exactly one `secondary`, the first configured member of `OPTIONAL`, and
    anything else with a valid key was skipped as "not the selected secondary".
    That single-secondary rule existed to give Phase 8's plain hive
    (`agent/graph/nodes.py`, `agent/graph/run.py`) one alternate vendor for
    seat diversity -- and `agent/graph/` is now an empty package. The only
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

    def _snapshot(self) -> None:
        self._configured = tuple(p for p in provider_names() if self.catalogue.is_configured(p))
        if self.REQUIRED not in self._configured:
            raise AuthError("Otto requires Inception. Set INCEPTION_API_KEY in .env")

    def __init__(self, catalogue: Catalogue | None = None, *, strict: bool = False):
        self.catalogue = catalogue or RegistryCatalogue()
        self.strict = strict
        self._snapshot()
        
    def reset(self):
        self.catalogue.reset()
        self._snapshot()

    def _usable(self, provider: str) -> bool:
        return provider in self._configured

    def usable(self) -> tuple[str, ...]:
        return tuple(p for p in self._configured if self._usable(p))

    def prewarm(self) -> dict[str, str]:
        failures: dict[str, str] = {}
        for name in self.usable():
            try:
                self.catalogue.models(name)
            except ProviderError as exc:
                failures[name] = str(exc)
        return failures
    
    def _match(self, c: Candidate, only: str | None = None) -> ModelInfo | str:
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
        return model

    def _select(self, pool, c: Candidate) -> ModelInfo | None:
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
        
    def resolve(self, task: Task, *, only: str | None = None) -> RoutingDecision:
        skips: list[Skip] = []
        for i, c in enumerate(TASK_ROUTES[task]):
            outcome = self._match(c, only=only)
            if isinstance(outcome, str):
                skips.append(Skip(i, render(c), outcome))
                continue
            
            if self.strict and i > 0:
                raise RoutingDegraded(task, tuple(skips))
            
            return RoutingDecision(
                task=task, provider=outcome.provider, model=outcome,
                endpoint=c.endpoint, params=dict(c.params),
                index=i, skipped=tuple(skips),
            )
        raise NoViableRoute(task, tuple(skips))
    
    def model_for(self, d: RoutingDecision, **overrides) -> BaseChatModel:
        if d.endpoint is not Endpoint.CHAT:
            raise CapabilityNotSupported(f"{d.task.value} routes to {d.endpoint.value}, not chat")
        provider = get_provider(d.provider)
        return provider.chat_model(d.model.id, **{**d.params, **overrides})
    
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
