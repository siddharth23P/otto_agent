"""A proposed model for every task, from the models this machine can reach.

Pure: hand it the detected pool (`all_models()`, or a test's list) and the
routing table, get back one `Proposal` per Task with a one-line reason. The
setup screen shows the proposals beside the current resolution and turns the
accepted ones into pins (agent/router/overrides.py); it never applies them
on its own.

The order of preference is deliberate and narrow:

  1. The shipped chain, in its declared order. mapping.py's pins are measured
     choices, so if the head's provider is configured and the model is in the
     catalogue, that is the proposal and no pin is needed (`source="shipped"`).
     A later candidate that resolves is `"fallback"`.
  2. Only if nothing in the chain resolves: the pool, restricted to the one
     provider a task is bound to (WEB -> anthropic, FIM/EDIT -> inception;
     agent/router/overrides.py's PROVIDER_ONLY), then filtered by the chain's
     own `requires` tiers head-first -- REASON asks for REASONING before it
     settles for CHAT, VISION never accepts a model that cannot see -- and
     sorted by context window: largest for the seats that think, smallest for
     the seats that classify and condense.
  3. Otherwise None, with the reason a person can act on.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from agent.router import overrides
from agent.router.llm_provider.base import Capability, ModelInfo
from agent.router.mapping import TASK_ROUTES, Candidate, Task
from agent.router.router import select_candidate

Source = Literal["shipped", "fallback", "pool", "none"]

#: Seats that want the most capable model that qualifies; every other task
#: takes the smallest context that does, as a proxy for the cheap tier.
LARGEST: frozenset[Task] = frozenset({Task.REASON, Task.PLAN, Task.EVALUATE, Task.VISION, Task.WEB})


@dataclass(frozen=True)
class Proposal:
    task: Task
    model: ModelInfo | None
    source: Source
    reason: str

    @property
    def spec(self) -> str | None:
        return self.model.spec if self.model is not None else None

    @property
    def needs_pin(self) -> bool:
        """A shipped head that resolves needs nothing; anything else the
        person accepts has to be written down as a pin to take effect."""
        return self.model is not None and self.source != "shipped"


def _tiers(chain: Sequence[Candidate]) -> list[frozenset[Capability]]:
    seen: list[frozenset[Capability]] = []
    for c in chain:
        if c.requires not in seen:
            seen.append(c.requires)
    return seen


def propose_one(task: Task, chain: Sequence[Candidate],
                by_provider: Mapping[str, Sequence[ModelInfo]]) -> Proposal:
    for i, c in enumerate(chain):
        provider = c.provider_name
        if provider is None or provider not in by_provider:
            continue
        model = select_candidate(by_provider[provider], c)
        if model is not None:
            if i == 0:
                return Proposal(task, model, "shipped",
                                f"shipped choice {model.spec} is configured and listed")
            return Proposal(task, model, "fallback",
                            f"shipped choice unavailable; candidate {i} {model.spec} is")

    bound = overrides.PROVIDER_ONLY.get(task)
    if bound:
        provider, why = bound
        pool: list[ModelInfo] = list(by_provider.get(provider, ()))
        if not pool:
            return Proposal(task, None, "none", f"needs {provider} ({why}), which is not configured")
    else:
        pool = [m for models in by_provider.values() for m in models]

    for tier in _tiers(chain):
        matches = [m for m in pool if tier <= m.capabilities]
        if matches:
            matches.sort(key=lambda m: (m.context_window or 0, m.id), reverse=task in LARGEST)
            model = matches[0]
            names = "+".join(sorted(c.value for c in tier)) or "any"
            size = "largest" if task in LARGEST else "smallest"
            return Proposal(task, model, "pool",
                            f"no shipped candidate configured; {size}-context {names} model")
    needed = "/".join("+".join(sorted(c.value for c in t)) for t in _tiers(chain))
    return Proposal(task, None, "none", f"no configured model offers {needed}")


def propose(pool: Sequence[ModelInfo],
            routes: Mapping[Task, tuple[Candidate, ...]] | None = None) -> dict[Task, Proposal]:
    """One proposal per Task, against the SHIPPED chains -- a pin already in
    the live table must not propose itself back."""
    by_provider: dict[str, list[ModelInfo]] = defaultdict(list)
    for m in pool:
        by_provider[m.provider].append(m)
    return {
        task: propose_one(task, overrides.shipped(task) if routes is None else routes[task], by_provider)
        for task in Task
    }
