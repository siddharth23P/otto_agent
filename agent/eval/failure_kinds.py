"""What KIND of failure a run had, read from the record it already keeps.

A score says a run got worse. It does not say what broke, and a regression
nobody can attribute is a regression nobody can scope a repair to. Measured:
an injected fault was localised 64.8% of the time from a structured trace and
13.0% of the time from the task outcome alone -- attribution is most of the
difference between a repair that is targeted and reversible and one that is
speculative.

NO MODEL CALL, AND NOTHING NEW RECORDED. Every signal here is already in
`actions` (one line per tool call, written by `_summarise_action` when the call
ran), in the checklist, and in the answer. This is a reading of what a run
already wrote down, which is what keeps it free enough to run on every task.

DELIBERATELY NOT A DIAGNOSIS. These are the shapes a failure takes, not its
cause -- `no_writes` says the run changed nothing, not why. They are for
COMPARING runs: a batch that shifts from `answered_without_evidence` to
`error_cascade` between two revisions has told you where to look, which is the
whole point.

A run can carry several. They are not mutually exclusive and forcing a single
label would throw away the combination, which is usually the informative part.
"""
from __future__ import annotations

import re

#: Tools whose success means something in the environment changed.
_WRITING = ("write_file", "edit_file", "predict_edit", "execute_bash",
            "execute_python", "browse_act", "look_act")

#: Tools that only look.
_READING = ("read_file", "list_files", "rag", "recall_memory", "code_map",
            "web_search", "browse", "look", "view_image")

_FAILED = re.compile(r"->\s*FAILED", re.I)
_OK = re.compile(r"->\s*ok", re.I)

#: How many failing calls in a row count as a cascade rather than a setback.
#: Three, for the same reason the provider breaker uses three: one is noise,
#: two is a coincidence.
CASCADE_RUN = 3


def _tool_of(line: str) -> str:
    return line.split(":", 1)[-1].strip().split(" ", 1)[0] if line else ""


def classify(
    *,
    actions: list[str] | None,
    checklist: list[dict] | None = None,
    answer: str = "",
    resolved: bool | None = None,
) -> list[str]:
    """Every failure shape this run shows, in a stable order.

    `resolved` is the harness's own verdict when it has one -- a benchmark
    knows whether the task passed and Otto's evaluator only knows whether it
    approved. When it is None the classification is about the SHAPE of the run
    rather than about success, which is still worth recording.
    """
    # None means NOBODY RECORDED the actions; [] means the run took none.
    # Collapsing the two would tag every run from a report written before this
    # existed as having done nothing, which is the confident-and-wrong kind of
    # wrong -- caught by reading this back against reports already on disk.
    unrecorded = actions is None
    actions = list(actions or [])
    kinds: list[str] = []

    tools = [_tool_of(a) for a in actions]
    failed = [bool(_FAILED.search(a)) for a in actions]
    wrote = [t in _WRITING and not f for t, f in zip(tools, failed)]

    if unrecorded:
        pass  # nothing about the action record can be claimed
    elif not actions:
        kinds.append("no_actions")
    elif not any(wrote):
        # Recuris' "zero-write episode": the run looked at things and changed
        # nothing. Only a failure when the task wanted something changed,
        # which is why it is reported rather than judged here.
        kinds.append("no_writes")

    if not unrecorded and actions and all(t in _READING for t in tools):
        kinds.append("read_only")

    run = 0
    for f in failed:
        run = run + 1 if f else 0
        if run >= CASCADE_RUN:
            kinds.append("error_cascade")
            break

    if failed and failed[0]:
        # The first tool call is where a run commits to an approach, and
        # getting it wrong is measured as its own mode rather than as part of
        # the cascade that may follow.
        kinds.append("wrong_first_call")

    open_items = [i for i in (checklist or []) if i.get("status") == "pending"]
    blocked = [i for i in (checklist or []) if i.get("status") == "blocked"]
    if blocked:
        kinds.append("blocked_by_environment")

    if answer.strip() and open_items and resolved is False:
        # "Hallucinated completion": it answered, confidently, with criteria
        # still open. The single most expensive shape, because it reads as
        # success everywhere except the grader.
        kinds.append("answered_with_criteria_open")
    if not answer.strip():
        kinds.append("no_answer")

    return kinds


#: One line per kind, for a report a person reads.
DESCRIPTIONS = {
    "no_actions": "took no action at all",
    "no_writes": "changed nothing in the environment",
    "read_only": "only read, never acted",
    "error_cascade": f"{CASCADE_RUN}+ failing calls in a row",
    "wrong_first_call": "its first tool call failed",
    "blocked_by_environment": "something outside its control stopped it",
    "answered_with_criteria_open": "answered with criteria still unmet",
    "no_answer": "produced no answer",
}


def summarise(per_run: list[list[str]]) -> dict[str, int]:
    """How often each kind appeared across a batch, most common first.

    This is the number a comparison is actually read from: two revisions with
    the same mean score and different failure profiles are two different
    systems, and the score alone cannot say so.
    """
    counts: dict[str, int] = {}
    for kinds in per_run:
        for kind in kinds:
            counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


# --------------------------------------------------------------------------
# What the tools actually cost
# --------------------------------------------------------------------------
#
# Otto had six tools and has seventeen, and the menu is restated on every call
# -- 27% of the system prompt before a single word of the task. Two separate
# results say that is worth a number rather than a shrug: tool OVERUSE is a
# real cost (training the knowledge boundary cut tool use 24% while raising
# performance 37%), and whether an agent manages its tool context at all
# depends on model strength (reasoning models 90-94%, medium models 0-60%).
# Otto routes across four vendors and several tiers, so both land here.
#
# Nothing in this argues for removing a tool. It argues for knowing the cost
# before the menu grows again.

def tool_usage(actions: list[str] | None) -> dict[str, int]:
    """How many times each tool was called, most used first.

    Read off the action record rather than newly recorded, for the same reason
    the failure kinds are: it is already written, and a measurement that costs
    nothing gets run on every task instead of on the ones somebody remembered.
    """
    counts: dict[str, int] = {}
    for line in actions or []:
        tool = _tool_of(line)
        if tool:
            counts[tool] = counts.get(tool, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def usage_by_seat(actions: list[str] | None) -> dict[str, int]:
    """Calls per MODE, which is per model -- the breakdown that says whether a
    weaker seat is flailing with a menu it cannot hold."""
    counts: dict[str, int] = {}
    for line in actions or []:
        mode, sep, _ = (line or "").partition(":")
        if sep and mode.strip():
            key = mode.strip().split()[0]
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def merge_usage(per_run: list[dict[str, int]]) -> dict[str, int]:
    """Totals across a batch, most used first."""
    total: dict[str, int] = {}
    for counts in per_run:
        for key, n in counts.items():
            total[key] = total.get(key, 0) + n
    return dict(sorted(total.items(), key=lambda kv: (-kv[1], kv[0])))
