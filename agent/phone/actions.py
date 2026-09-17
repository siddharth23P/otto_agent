"""The phone's premapped actions as a dictionary the agent searches.

The app (otto_android actions/ActionCatalog.kt) offers any number of actions -- set an alarm, write an
SMS, share a file -- each with typed fields. None of them is written into the prompt: the agent calls
`phone_find` with what it wants to do, reads the matching entries in full, then `phone_action` runs one
by name. The prompt stays the same size however many actions the phone has.

An entry, as the bridge sends it:
    {"name": "alarm.set", "summary": "set an alarm in the clock app", "effect": "change|read|confirm",
     "group": "clock", "keywords": ["wake", "alarm clock"],
     "params": [{"name": "hour", "type": "integer", "required": true, "doc": "0-23", "min": 0, "max": 23,
                 "choices": [...]}]}
"""
from __future__ import annotations

import re

#: Most entries one search returns in full.
MAX_FOUND = 5

_WORD = re.compile(r"[a-z0-9]+")
#: Words that say nothing about which action is meant.
_FILLER = frozenset("a an the to for of on in my me please set up do can you with and or".split())


def clean_menu(menu: object) -> list[dict]:
    """The entries that have a name, in the phone's order; anything else is dropped."""
    if not isinstance(menu, list):
        return []
    return [a for a in menu if isinstance(a, dict) and isinstance(a.get("name"), str) and a["name"]]


def _words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _FILLER]


def _stem(word: str) -> str:
    """Crude, but enough for "alarms"/"alarm", "sharing"/"share", "texted"/"text"."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def search(menu: list[dict], query: str, limit: int = MAX_FOUND) -> list[dict]:
    """The entries that best match `query`: a word in the name or the keywords counts most, then the
    summary, then the group and the fields; ties keep the phone's order. An exact name wins outright."""
    query = query.strip().lower()
    for entry in menu:
        if entry["name"].lower() == query:
            return [entry]
    wanted = {_stem(w) for w in _words(query)}
    if not wanted:
        return []
    scored = []
    for order, entry in enumerate(menu):
        fields = entry.get("params") or []
        places = (
            (3, entry["name"]),
            (3, " ".join(map(str, entry.get("keywords") or []))),
            (2, str(entry.get("summary") or "")),
            (1, str(entry.get("group") or "")),
            (1, " ".join(f"{p.get('name', '')} {p.get('doc', '')} {' '.join(map(str, p.get('choices') or []))}"
                         for p in fields if isinstance(p, dict))),
        )
        score = 0
        for weight, text in places:
            have = {_stem(w) for w in _words(text)}
            score += weight * len(wanted & have)
        if score:
            scored.append((-score, order, entry))
    scored.sort(key=lambda s: s[:2])
    # A match far weaker than the best is noise ("call dad" also names a volume stream).
    floor = -scored[0][0] / 3 if scored else 0
    return [entry for score, _, entry in scored[:limit] if -score >= floor]


def describe(entry: dict) -> str:
    """One entry in full: how to call it, and what each field takes.

        alarm.set -- set an alarm in the clock app
          hour: integer 0..23, required
          days: list of mon|tue|..., optional -- repeat on these days
    """
    you = " (the person finishes it: the phone is handed to them)" if entry.get("effect") == "confirm" else ""
    lines = [f"{entry['name']} -- {entry.get('summary', '')}{you}".rstrip(" -")]
    for p in entry.get("params") or []:
        if not isinstance(p, dict):
            continue
        kind = {"string": "text", "string_list": "list of text"}.get(str(p.get("type")), str(p.get("type", "any")))
        if p.get("choices"):
            kind = ("list of " if p.get("type") == "string_list" else "") + "|".join(map(str, p["choices"]))
        if p.get("min") is not None or p.get("max") is not None:
            kind += f" {_num(p.get('min'))}..{_num(p.get('max'))}"
        need = "required" if p.get("required", True) else "optional"
        doc = f" -- {p['doc']}" if p.get("doc") else ""
        lines.append(f"  {p.get('name')}: {kind}, {need}{doc}")
    if len(lines) == 1:
        lines.append("  (no fields)")
    return "\n".join(lines)


def _num(value: object) -> str:
    if value is None:
        return ""
    return str(int(value)) if isinstance(value, float) and value.is_integer() else str(value)


def near(menu: list[dict], name: str) -> list[str]:
    """Names to suggest for a name the phone does not have."""
    return [e["name"] for e in search(menu, name.replace(".", " ").replace("_", " "), limit=3)]
