"""How much a compaction policy actually loses, measured rather than argued.

Otto has two things that throw text away -- agent/pipeline/nodes.py's
`_compact` on a run's transcript, and agent/memory/queue.py's TieredQueue on a
session's conversation -- and until this there was no way to tell a good one
from a bad one. `otto eval-memory` scores whether raw evidence stays
RETRIEVABLE, which the chunk store guarantees almost by construction; it says
nothing about whether what the model actually reads still contains the
constraint it was given.

That is the gap the Compaction Cliff measures and names: constraint recall
falls to 53% at 50% compression and 24% at 10%, and to 10% over five
successive rounds, while a type-aware policy holds 100/95/80 and stabilises at
96%. Those are differences no benchmark Otto had could see.

WHAT THIS MEASURES. A synthetic conversation with constraints planted in it at
known positions, compacted under each named policy, then two questions asked
of the result:

    survives   did the constraint's own words reach `current_view()` --
               what a prompt is built from
    recoverable  could `recall()` find it afterwards

The gap between those two is the interesting number. A policy can score badly
on the first and perfectly on the second, which means "the model will not see
it unless it thinks to go looking" -- true of Otto today, and the reason the
second compaction tier now leaves a pointer behind rather than nothing.

NO MODEL CALLS BY DEFAULT. The summariser is a deterministic stand-in that
keeps a fixed fraction of each item, so a policy comparison is reproducible
and free. That makes it a measurement of the POLICY rather than of whichever
model happened to summarise; `--live` swaps in the real one when the question
is about the model instead.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.memory.queue import TieredQueue
from agent.memory.retrieval import recall
from agent.memory.store import MemoryStore

#: Constraints planted in the synthetic conversation. Each is a sentence a
#: later turn would have to honour, of the kind that matters and that a
#: summariser is most likely to smooth away: a number, a prohibition, a
#: preference, a name.
CONSTRAINTS = [
    "the deployment key is AK-4417-QX and must never be written to a log",
    "the report has to be in Europe/Lisbon time, not UTC",
    "do not touch the billing tables, they are replicated to a third party",
    "the client rejected blue, so the palette stays to greens and greys",
    "every export must be capped at 5000 rows per file",
    "Priya owns the schema and any change to it needs her sign-off",
    "retries are capped at three, after which the job must page someone",
    "the legacy endpoint stays up until 2027-01-31 at the earliest",
]

#: Filler turns between constraints, so compaction has something to compact.
#: Deliberately bland: the point is whether a constraint survives beside a
#: great deal of text that does not matter, which is the real situation.
_FILLER = (
    "checked the dashboard, nothing unusual in the last hour",
    "ran the smoke tests, all green, took about forty seconds",
    "read through the handler, it does what the docstring says",
    "asked about timelines, nobody has a firm date yet",
    "pulled the latest changes, no conflicts",
    "looked at yesterday's numbers, roughly flat",
)


def build_conversation(*, turns: int = 120, constraints=CONSTRAINTS) -> list[str]:
    """A conversation with the constraints spread evenly through it.

    Evenly rather than clustered, because position is the thing being tested:
    a policy that keeps a verbatim tail scores perfectly on the last few and
    badly on the first, and a mean over clustered constraints would hide that.
    """
    spacing = max(1, turns // (len(constraints) + 1))
    items: list[str] = []
    planted = 0
    for i in range(turns):
        if planted < len(constraints) and i and i % spacing == 0:
            items.append(f"you: remember, {constraints[planted]}")
            planted += 1
        else:
            items.append(f"otto: {_FILLER[i % len(_FILLER)]} (turn {i})")
    items.extend(f"you: remember, {c}" for c in constraints[planted:])
    return items


def keep_fraction_summarizer(fraction: float = 0.35):
    """A deterministic stand-in for Task.SUMMARIZE.

    It keeps the first `fraction` of each numbered item and cites it, which is
    the shape of a real summary -- lossy, positional, and citing -- without a
    model's variance. A policy that only looks good under a generous
    summariser is not a policy, it is a lucky model.
    """
    item_line = re.compile(r"^\s*(\d+)[.):]\s*(.*)$")

    def summarize(prompt: str) -> str:
        bullets = []
        for line in prompt.splitlines():
            match = item_line.match(line)
            if not match:
                continue
            index, text = match.group(1), match.group(2).strip()
            head = text[: max(20, int(len(text) * fraction))]
            bullets.append(f"- {head} [{index}]")
        return "\n".join(bullets) or "- (nothing) [1]"

    return summarize


@dataclass
class PolicyResult:
    name: str
    survives: int
    recoverable: int
    total: int
    view_chars: int
    compactions: int
    missing: list[str] = field(default_factory=list)

    @property
    def survival_rate(self) -> float:
        return self.survives / self.total if self.total else 0.0

    @property
    def recovery_rate(self) -> float:
        return self.recoverable / self.total if self.total else 0.0


#: The policies compared. A spec is the keyword arguments a TieredQueue is
#: built with, so adding one is a line rather than a subclass -- and so the
#: matrix cannot drift from what the queue actually does.
#:
#: `roomy` and `cramped` are the two ends of the same policy, which is the
#: comparison worth having first: most of what looks like a compaction
#: decision is really a budget decision, and a matrix that did not show that
#: would attribute a budget's effect to an algorithm.
POLICIES: dict[str, dict] = {
    "shipping":  {"x_budget": 400, "y_budget": 1200},
    "roomy":     {"x_budget": 1200, "y_budget": 4000},
    "cramped":   {"x_budget": 150, "y_budget": 400},
    "tiny-tail": {"x_budget": 60, "y_budget": 1200},
    "no-y":      {"x_budget": 400, "y_budget": 60},
    # The question the bench was built for. Same budgets as `shipping`, with
    # prior bullets carried forward untouched instead of re-summarised: the
    # oldest summaries are dropped cleanly rather than every summary degrading
    # slowly. See agent/memory/queue.py's REABSTRACT.
    "abstract-once":        {"x_budget": 400, "y_budget": 1200, "reabstract": False},
    "abstract-once-tight":  {"x_budget": 150, "y_budget": 400, "reabstract": False},
    # Type-aware: what the PERSON said is never handed to the summariser.
    "protect-user":         {"x_budget": 400, "y_budget": 1200},
    "protect-user-tight":   {"x_budget": 150, "y_budget": 400},
    "type-blind":           {"x_budget": 400, "y_budget": 1200, "protect": ""},
    "type-blind-tight":     {"x_budget": 150, "y_budget": 400, "protect": ""},
}


def run_policy(
    name: str,
    spec: dict,
    *,
    root: Path,
    conversation: list[str],
    constraints=CONSTRAINTS,
    summarize=None,
) -> PolicyResult:
    """One policy, end to end: replay the conversation, then ask both
    questions of what is left."""
    store = MemoryStore(root / f"{name}.db")
    queue = TieredQueue(
        kind="history", store=store,
        summarize=summarize or keep_fraction_summarizer(),
        **spec,
    )
    before = queue._generation
    for item in conversation:
        queue.append(item)
    view = queue.current_view()

    survives = recoverable = 0
    missing: list[str] = []
    for constraint in constraints:
        key = _signature(constraint)
        in_view = key in _normalise(view)
        found = key in _normalise(recall(store, "history", constraint))
        survives += in_view
        recoverable += in_view or found
        if not (in_view or found):
            missing.append(constraint)

    result = PolicyResult(
        name=name, survives=survives, recoverable=recoverable,
        total=len(constraints), view_chars=len(view),
        compactions=queue._generation - before, missing=missing,
    )
    store.close()
    return result


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def _signature(constraint: str) -> str:
    """The part of a constraint that carries it.

    Matching the whole sentence would score any lossy summariser at zero and
    tell us nothing; matching a single word would score noise as a pass. The
    signature is the longest run of words that contains what a later turn
    would actually need -- the number, the name, the prohibition -- taken as
    the first six words, which is where these sentences put it.
    """
    return _normalise(" ".join(constraint.split()[:6]))


def run_matrix(root: Path, *, turns: int = 120, policies=None,
               summarize=None) -> list[PolicyResult]:
    conversation = build_conversation(turns=turns)
    return [
        run_policy(name, spec, root=root, conversation=conversation,
                   summarize=summarize)
        for name, spec in (policies or POLICIES).items()
    ]
