from typing import Any, Mapping, Protocol, runtime_checkable

from dataclasses import dataclass, field

from agent.router.llm_provider import get_provider, provider_class
from agent.router.llm_provider.base import ModelInfo, ProviderError
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
    def __init__(self, catalogue: Catalogue | None = None):
        self.catalogue = catalogue or RegistryCatalogue()
        
    def _match(self, c: Candidate) -> ModelInfo | str:
        provider = c.provider_name
        if provider is None:
            return "open queries not supported"
        if not self.catalogue.is_configured(provider):
            return f"{provider.upper()}_API_KEY not set"
        
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
            return RoutingDecision(
                task=task, provider=outcome.provider, model=outcome,
                endpoint=c.endpoint, params=dict(c.params),
                index=i, skipped=tuple(skips),
            )
        raise NoViableRoute(task, tuple(skips))