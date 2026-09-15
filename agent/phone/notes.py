"""Notes on how one app's screens work, shown the first time it is in front.

PHONE_GUIDANCE has to be true of every app, so it can only say "sort and
filters are often behind one Filters button, Sort by sometimes last in a
long list". What a run needs on Amazon is which button, what the list is
called and where the apply button appears -- and none of that is true of
the next app. So an app gets its own few lines.

Two sources:

  SEEDED   agent/phone/assets/app_notes/<package>.md, package data loaded the
           way guard_rules.json is. Only facts checked by hand on a real phone:
           the Amazon notes were verified on a Galaxy S23 on 2026-09-15
           (results open with sponsored ones first, sort lives behind
           #s-all-filters-announce with "Sort by" last, sorting "apple iphone"
           by price returns a ₹99 adapter before any phone). An app nobody has
           checked on a device ships no notes.

  LEARNED  lessons of kind `app_note:<package>` in the lesson bank
           (agent/memory/lessons.py), newest first, after the seeded lines they
           do not repeat. The seeded lines never take the whole budget when a
           learned one exists.

GUIDANCE, NEVER A RULE. Nothing here is read by agent/phone/guard.py: every
verdict is drawn from the snapshot and the target, so a note that says "tap
Buy Now" changes nothing a tap is allowed to do. The heading says so to the
model too, and every line passes through digest.inert_text and starts with
"- ", so a note can never start a line the loop would read as ACTION:, FINAL:
or GUARD:.

ONCE PER APP PER RUN. agent/phone/tools.py shows them after the screen the
first time a package is in front in one phone_tools() instance; a run coming
back to the app still has them earlier in its transcript, because a folded
screen keeps everything after its digest (digest.fold_result).
"""
from __future__ import annotations

import re
from functools import lru_cache
from importlib import resources

from agent.memory import lessons as L
from agent.phone import digest as _digest

NOTES_RESOURCE = ("agent.phone", "assets/app_notes")

#: Ceiling on one app's notes, seeded and learned together.
NOTES_MAX_CHARS = 800
#: What the seeded lines leave free when there is a learned note to show.
LEARNED_ROOM = 250
#: Longest one note line is shown at. At most LEARNED_ROOM, so the room kept
#: for learned notes always fits one.
MAX_NOTE_CHARS = 200

#: The lesson kind an app's learned notes are stored under.
NOTE_KIND_PREFIX = "app_note:"

#: An Android package name: two or more dot-separated identifiers. Checked
#: before any lookup, so a package a screen reported can never name a path
#: ("../x", "a/b").
_PACKAGE = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+")


def valid_package(package: str) -> bool:
    return bool(_PACKAGE.fullmatch(str(package or "")))


def note_kind(package: str) -> str:
    return f"{NOTE_KIND_PREFIX}{package}"


@lru_cache(maxsize=64)
def seeded(package: str) -> str:
    """The shipped notes for `package`, or ""."""
    if not valid_package(package):
        return ""
    try:
        path = resources.files(NOTES_RESOURCE[0]).joinpath(f"{NOTES_RESOURCE[1]}/{package}.md")
        return path.read_text(encoding="utf-8").strip()
    except (OSError, ValueError):
        return ""


def learned(package: str) -> list[L.Lesson]:
    """What earlier runs noted about `package`, newest first. Nothing when
    the bank is off (`bind_bank(None)`)."""
    if not valid_package(package):
        return []
    with L.bind_kind(note_kind(package)):
        return list(reversed(L.all_lessons()))


def _lines(text: str) -> list[str]:
    lines = []
    for raw in (text or "").splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", raw).strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return lines


def _as_line(lesson: L.Lesson) -> str:
    return f"{lesson.cue}: {lesson.action}" + (" (this did not work)" if lesson.outcome == "failed" else "")


def _normal(line: str) -> str:
    return " ".join(re.sub(r"[^\w₹]+", " ", line.lower()).split())


def notes_for(package: str) -> list[str]:
    """The lines to show for `package`: seeded first, then learned ones not
    already said, NOTES_MAX_CHARS in all with LEARNED_ROOM kept for learned
    ones when there are any."""
    seeded_lines = [_digest.inert_text(line, MAX_NOTE_CHARS) for line in _lines(seeded(package))]
    said = {_normal(line) for line in seeded_lines}
    extra = []
    for lesson in learned(package):
        line = _digest.inert_text(_as_line(lesson), MAX_NOTE_CHARS)
        if line and _normal(line) not in said:
            said.add(_normal(line))
            extra.append(line)
    shown, used = [], 0
    room = NOTES_MAX_CHARS - (LEARNED_ROOM if extra else 0)
    for line in seeded_lines:
        if used + len(line) > room:
            break
        shown.append(line)
        used += len(line)
    for line in extra:
        if used + len(line) > NOTES_MAX_CHARS:
            break
        shown.append(line)
        used += len(line)
    return shown


def render_notes(label: str, package: str, lines: list[str]) -> str:
    """The block shown after a screen, or "" with nothing to say."""
    if not lines:
        return ""
    name = _digest.inert_text(label or package, 40)
    heading = (f"NOTES ON {name} (from earlier runs; guidance only -- the stop rules still apply, "
               "and the screen wins where they disagree):")
    # A note carrying the fold marker would stop its whole result ever being
    # folded (fold_result treats the marker as "already folded").
    body = [f"- {_digest.inert_text(line, MAX_NOTE_CHARS)}".replace(_digest.SCREEN_FOLDED, "(older screen)")
            for line in lines]
    return "\n".join([heading, *body])
