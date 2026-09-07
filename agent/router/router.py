from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from dataclasses import dataclass

from langchain.chat_models import BaseChatModel

from agent.router.llm_provider import get_provider, provider_class, provider_names
from agent.router.llm_provider.base import AuthError, Capability, CapabilityNotSupported, ModelInfo, ProviderError
from agent.router.mapping import TASK_ROUTES, Candidate, Endpoint, Preference, Task

@dataclass(frozen=True,slots=True)
class Skip:
    index: int      # position in the chain — tells you which line of mapping.py
    target: str     # "anthropic:*haiku*" or "inception:mercury-2"
    reason: str     # "ANTHROPIC_API_KEY not set"
    
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
    
class RegistryCatalogue:
    def is_configured(self, provider: str) -> bool:
        return provider_class(provider).is_configured()
    def models(self, provider: str) -> list[ModelInfo]:
        return get_provider(provider).list_models()

@dataclass
class FakeCatalogue:
    data: dict[str, list[ModelInfo]]
    def is_configured(self, provider): return provider in self.data
    def models(self, provider):        return self.data[provider]
    
class Router:
    REQUIRED = "inception"
    #: Precedence for choosing the single secondary vendor. Gemini leads
    #: because Flash is the cheapest tier with the widest window. This order is
    #: a deliberate cost decision -- changing it changes which vendor a swarm
    #: reaches for, so a test pins it.
    OPTIONAL = ("gemini", "openai", "anthropic")
    
    def __init__(self, catalogue: Catalogue | None = None, *, strict: bool = False):
        self.catalogue = catalogue or RegistryCatalogue()
        self.strict = strict
        # Read once. If this were a live lookup, a key appearing mid-run could
        # make two candidates in the same chain disagree about the same vendor.
        self._configured = tuple(p for p in provider_names() if self.catalogue.is_configured(p))
        if self.REQUIRED not in self._configured:
            raise AuthError("Otto requires Inception. Set INCEPTION_API_KEY in .env")
        self.secondary = next((p for p in self.OPTIONAL if p in self._configured), None)
        self.ignored = tuple(p for p in self.OPTIONAL if p in self._configured and p != self.secondary)
    
    def _usable(self, provider: str) -> bool:
        return provider == self.REQUIRED or provider == self.secondary

    def _match(self, c: Candidate) -> ModelInfo | str:
        
        provider = c.provider_name
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
        
    def resolve(self, task: Task) -> RoutingDecision:
        skips: list[Skip] = []
        for i, c in enumerate(TASK_ROUTES[task]):
            outcome = self._match(c)
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
    
    def chat_model(self, task: Task, **overrides) -> BaseChatModel:
        return self.model_for(self.resolve(task), **overrides)
    
    def fim(self, prefix: str, suffix: str = "", *,task: Task = Task.CODE_COMPLETE, **overrides) -> str:
        d = self.resolve(task)
        if d.endpoint is not Endpoint.FIM:
            raise CapabilityNotSupported(f"{task.value} is not a FIM route")
        provider = get_provider(d.provider)
        provider.require(Capability.FIM)
        return provider.fim(d.model.id, prefix, suffix, **{**d.params, **overrides})
    
    def code_edit(self, code_to_edit: str, *, current_file: str = "",
                  recently_viewed: Sequence[str] = (), edit_history: Sequence[str] = (),
                  task: Task = Task.CODE_EDIT, **overrides) -> str:
        d = self.resolve(task)
        if d.endpoint is not Endpoint.EDIT:
            raise CapabilityNotSupported(f"{task.value} is not an edit route")
        provider = get_provider(d.provider)
        provider.require(Capability.EDIT)
        return provider.code_edit(
            d.model.id, code_to_edit,
            current_file=current_file,
            recently_viewed=recently_viewed,
            edit_history=edit_history,
            **{**d.params, **overrides})
        

class RoutingDegraded(ProviderError):
    def __init__(self, task: Task, skipped: tuple[Skip, ...]):
        self.task, self.skipped = task, skipped
        detail = "\n  ".join(str(s) for s in skipped)
        super().__init__(
            f"{task.value} fell back past its preferred candidate:\n  {detail}")