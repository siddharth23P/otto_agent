"""What KIND of failure a run was, and what its tools cost, read off the
record a run already keeps.

Two issues, one input. A run is recorded as a score, so a regression says
something got worse without saying what broke; and Otto went from six tools
to seventeen with the menu stated on every call, with nothing counting what
that costs which seat. Both questions are answerable from `AgentState`'s
`actions` -- one lossy line per tool call, already prefixed with the mode
that made it ("solve: execute_bash rustc main.rs -> FAILED (exit 1): ...").

NO MODEL CALL, and that is the constraint that shapes everything here. A
classifier that asked a model what went wrong would cost a call per failure,
would need its own evaluation, and would be the sort of thing that quietly
stops being run. Every tag below is decided by counting and string matching
over lines that were written anyway, so it can be applied to a batch that has
already finished, retroactively, for free.

WHAT THE TAGS ARE FOR. Recuris (arXiv:2608.24876) localised an injected fault
64.8% of the time from a structured trace against 13.0% from the task outcome
alone. The point is not the taxonomy's elegance, it is that "score went down"
is not a repairable statement and "38% of failures changed no files at all"
is. The six names come from that paper's own reported failure modes; three of
them are decidable from the action record and three are not, and this module
implements the three rather than approximating six.

WHAT IS DELIBERATELY NOT HERE. `hallucinated completion` needs the final
answer's CLAIMS checked against what ran, which is a judgment; `missed reads`
needs to know what should have been read; `omitted writes` needs to know what
should have been written. Each of those is a model call or a task-specific
oracle. Guessing at them from the action record would put three
made-up-looking numbers beside three real ones, and a taxonomy nobody trusts
is worse than three tags somebody acts on.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

#: `actions` lines look like "<seat>: <tool> <target> -> ok: ..." or
#: "<seat>: <tool> <target> -> FAILED (exit 1): ...". The seat prefix is added
#: by nodes.py's `_agent_loop`; a line without one is a delegated subtask's
#: ("solve(delegated): ...") or came from a caller that did not prefix, and is
#: still usable for the tool half.
_ACTION = re.compile(
    r"^(?:(?P<seat>[a-z_]+(?:\(delegated\))?)\s*:\s*)?"
    r"(?P<tool>[a-z_]+)"
    r"(?:\s+(?P<target>.*?))?"
    r"\s*->\s*(?P<outcome>ok|FAILED)\b",
)

#: Tools that change something outside the conversation. Kept here rather than
#: imported from agent/pipeline/tools.py's TOOL_TIERS on purpose: this module
#: reads RECORDED runs, including ones from a revision whose tier table said
#: something else, and a classifier that changed its answer when the registry
#: changed would make two batches incomparable.
WRITE_TOOLS = frozenset({
    "write_file", "edit_file", "apply_patch", "execute_bash", "execute_python",
})

#: Tools whose whole job is to change a file. `execute_bash` can write too,
#: but it is also how everything gets checked, so counting it as a write would
#: mean no run ever had zero writes.
EDIT_TOOLS = frozenset({"write_file", "edit_file", "apply_patch"})

#: How many failures in a row count as a cascade rather than a bad call.
#:
#: Two is a retry; three is a run that stopped reading what came back. Matches
#: nodes.py's own instinct -- its REPEATED_FAILURE_NOTE fires on the SECOND
#: failure against the same target, which is a narrower thing than this.
CASCADE_LENGTH = 3


@dataclass(frozen=True)
class Action:
    """One parsed line of the action record."""

    seat: str
    tool: str
    target: str
    failed: bool


def parse_actions(actions: Iterable[str]) -> list[Action]:
    """The action record as structure. Unparseable lines are DROPPED rather
    than guessed at -- a line this cannot read is a line whose tool and
    outcome are unknown, and inventing either would corrupt every count
    downstream."""
    parsed: list[Action] = []
    for line in actions or ():
        match = _ACTION.match(str(line).strip())
        if match is None:
            continue
        parsed.append(Action(
            seat=(match.group("seat") or "unknown"),
            tool=match.group("tool"),
            target=(match.group("target") or "").strip(),
            failed=match.group("outcome") == "FAILED",
        ))
    return parsed


# --------------------------------------------------------------------------
# Failure kinds
# --------------------------------------------------------------------------

#: Tag: the run made no successful edit at all.
#:
#: Recuris reports "zero-write episodes" as its own mode for a reason -- a run
#: that never changed anything failed before it started, and that is a
#: completely different repair from one that changed the wrong thing. It is
#: also the cheapest tag here: it needs no target, no ordering and no oracle.
ZERO_WRITE = "zero_write"

#: Tag: CASCADE_LENGTH or more consecutive failing calls.
#:
#: The signature of a run that stopped reading its own tool results, which is
#: the failure `_DIAGNOSTIC_HABITS` rule 2 is written against. Tagging it says
#: whether that guidance is landing.
ERROR_CASCADE = "error_cascade"

#: Tag: the run's FIRST tool call failed.
#:
#: Recuris's "wrong first tool call" needs the task's intended target to
#: decide properly, which is not available here. What IS decidable is that the
#: opening move did not work -- a weaker claim, named for what it actually
#: measures rather than for the paper's mode.
FIRST_CALL_FAILED = "first_call_failed"

#: Tag: no tool calls were recorded at all.
#:
#: Distinct from ZERO_WRITE and worth its own name: a run that answered
#: without touching anything either did not need to, or never got going. Both
#: readings matter and neither is visible in a score.
NO_ACTIONS = "no_actions"


def classify(actions: Sequence[str] | None) -> list[str]:
    """Every tag that applies to this run, in a stable order.

    Tags are NOT mutually exclusive: a run whose first call failed and which
    then failed three more times and changed nothing carries three, and the
    overlap is the useful part. Forcing one tag per run would need a priority
    order this module has no evidence for.
    """
    parsed = parse_actions(actions)
    if not parsed:
        return [NO_ACTIONS]

    tags: list[str] = []
    if parsed[0].failed:
        tags.append(FIRST_CALL_FAILED)
    if not any(a.tool in EDIT_TOOLS and not a.failed for a in parsed):
        tags.append(ZERO_WRITE)

    run = 0
    for action in parsed:
        run = run + 1 if action.failed else 0
        if run >= CASCADE_LENGTH:
            tags.append(ERROR_CASCADE)
            break
    return tags


def distribution(runs: Iterable[Sequence[str] | None]) -> dict[str, int]:
    """How many of these runs carry each tag. What the issue asked for: the
    distribution says where the next change should go."""
    counts: Counter[str] = Counter()
    for actions in runs:
        counts.update(classify(actions))
    return dict(counts.most_common())


# --------------------------------------------------------------------------
# What the tools cost
# --------------------------------------------------------------------------

@dataclass
class ToolCost:
    """Tool calls for one run, broken down the two ways the issue asks for.

    `budget.py` already counts model calls and `usage.py` now counts tokens
    per model. This is the third axis: seventeen tools with the menu stated on
    every call, and nothing saying which seat spends them. Two papers put the
    question the same way -- tool overuse as a measurable cost, and model
    strength deciding whether an agent manages tool context at all -- and
    Otto routes across four vendors and several tiers, so the per-SEAT split
    is the half that answers it.

    Nothing here argues for removing a tool. It argues for having the number
    before the menu grows again.
    """

    by_tool: dict[str, int] = field(default_factory=dict)
    by_seat: dict[str, int] = field(default_factory=dict)
    #: Failures, the same two ways -- a tool that is called a lot and works is
    #: a different cost from one called a lot that does not.
    failures_by_tool: dict[str, int] = field(default_factory=dict)
    failures_by_seat: dict[str, int] = field(default_factory=dict)

    @property
    def calls(self) -> int:
        return sum(self.by_tool.values())

    @property
    def failures(self) -> int:
        return sum(self.failures_by_tool.values())

    def to_dict(self) -> dict:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "by_tool": self.by_tool,
            "by_seat": self.by_seat,
            "failures_by_tool": self.failures_by_tool,
            "failures_by_seat": self.failures_by_seat,
        }


def tool_cost(actions: Sequence[str] | None) -> ToolCost:
    """One run's tool spend, by tool and by seat."""
    by_tool: Counter[str] = Counter()
    by_seat: Counter[str] = Counter()
    failed_tool: Counter[str] = Counter()
    failed_seat: Counter[str] = Counter()
    for action in parse_actions(actions):
        by_tool[action.tool] += 1
        by_seat[action.seat] += 1
        if action.failed:
            failed_tool[action.tool] += 1
            failed_seat[action.seat] += 1
    return ToolCost(
        by_tool=dict(by_tool.most_common()),
        by_seat=dict(by_seat.most_common()),
        failures_by_tool=dict(failed_tool.most_common()),
        failures_by_seat=dict(failed_seat.most_common()),
    )


def total_tool_cost(runs: Iterable[Sequence[str] | None]) -> ToolCost:
    """The same, summed over a batch -- what to read off a Claw-Eval run."""
    total = ToolCost()
    for actions in runs:
        one = tool_cost(actions)
        for source, target in (
            (one.by_tool, total.by_tool),
            (one.by_seat, total.by_seat),
            (one.failures_by_tool, total.failures_by_tool),
            (one.failures_by_seat, total.failures_by_seat),
        ):
            for key, count in source.items():
                target[key] = target.get(key, 0) + count
    # Re-sorted so the busiest is first however the runs arrived.
    total.by_tool = dict(sorted(total.by_tool.items(), key=lambda kv: -kv[1]))
    total.by_seat = dict(sorted(total.by_seat.items(), key=lambda kv: -kv[1]))
    return total


def summarise(runs: Mapping[str, Sequence[str] | None]) -> dict:
    """Both answers for a batch, keyed by run id where a tag applies.

    `failures_by_kind` is the distribution; `runs_by_kind` names WHICH runs,
    because a distribution nobody can trace back to a trace is a statistic
    rather than a lead.
    """
    runs_by_kind: dict[str, list[str]] = {}
    for run_id, actions in runs.items():
        for tag in classify(actions):
            runs_by_kind.setdefault(tag, []).append(run_id)
    return {
        "failures_by_kind": {k: len(v) for k, v in sorted(
            runs_by_kind.items(), key=lambda kv: -len(kv[1]))},
        "runs_by_kind": runs_by_kind,
        "tool_cost": total_tool_cost(runs.values()).to_dict(),
    }
