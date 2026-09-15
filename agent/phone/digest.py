"""The screen as words the model can act on.

A snapshot is what the phone's accessibility tree looks like once the host
has walked it (the Android app's TreeWalker):

    {"snapshot_id": "s12", "app": {"package": "...", "label": "Blinkit"},
     "screen": {"w": 1080, "h": 2400}, "keyboard": false, "secure": false,
     "nodes": [{"i": 3, "t": "Add to cart", "d": "", "r": "button",
                "b": [900, 1700, 1060, 1780], "c": true, "e": false,
                "s": false, "p": false, "f": false, "k": null,
                "v": "add-to-cart-button"}, ...]}

    i index   t text   d content description   r role   b bounds [l,t,r,b]
    c clickable   e editable   s scrollable   p password field
    f focused   k checked (true/false, null when not checkable)
    v resource id, package prefix dropped ("" or absent when none)

A web page hands one product to accessibility several times over (a link, a
heading, a text) and a price as a summary plus each of its pieces. The digest
shows a label once: a node whose label an earlier, enclosing node already
shows is folded into that node's line (`shown_nodes`). It is still on the
screen and still a target by its text. An element whose label says it is
sponsored carries the flag `ad`, and a label that is only a tracking link
reads `(link)`.

An actionable element shows its id after a `#`. A web page's form buttons
often all read "Submit" and say what they do only in their id, so a target
may name the id ("#add-to-cart-button", or its words: "add to cart").

The digest is one line per node, bounded, with every piece of screen text
made inert the same way agent/pipeline/toolkit.py makes a tool description
inert: control characters collapsed, length capped, and the line begins with
the index in brackets so nothing an app displays can start a line the
framework would read as its own. A password field's text is never shown.
"""
from __future__ import annotations

import re
from typing import Any

from agent.phone import guard

#: How many nodes a digest shows before it says how many it left out.
MAX_NODES = 120
#: Ceiling on the whole digest.
MAX_CHARS = 6000
#: Longest a single label is shown at.
MAX_LABEL = 80

#: A label that says the element is a paid placement.
_SPONSORED = re.compile(r"\bsponsored\b", re.IGNORECASE)
#: The "Sponsored Ad - " a listing puts before the product's own name.
_SPONSORED_PREFIX = re.compile(r"^\s*sponsored(\s+ad)?\s*[-\u2013\u2014:\u00b7|]\s*", re.IGNORECASE)
#: A label that is a URL or a tracking query, not words: nothing to read out.
_LINK = re.compile(r"^(?:ref=|click\?|https?://|www\.)\S*$|^\S*[?&]\S*=\S*$", re.IGNORECASE)

_CONTROL = re.compile(r"[\x00-\x1f\x7f  ]+")


def inert_text(text: Any, limit: int = MAX_LABEL) -> str:
    """One line, control characters gone, bounded."""
    collapsed = " ".join(_CONTROL.sub(" ", str(text or "")).split())
    if len(collapsed) > limit:
        collapsed = collapsed[: limit - 1].rstrip() + "…"
    return collapsed


def label_of(node: dict) -> str:
    """What a person reads on the node: its text, else its description."""
    return str(node.get("t") or node.get("d") or "").strip()


def is_ad(node: dict) -> bool:
    """Whether the element's label says it is sponsored."""
    return bool(_SPONSORED.search(label_of(node)))


def view_id_of(node: dict) -> str:
    """The node's resource id, "" when it has none."""
    return str(node.get("v") or "").strip()


def centre_of(node: dict) -> tuple[int, int] | None:
    bounds = node.get("b")
    if not (isinstance(bounds, (list, tuple)) and len(bounds) == 4):
        return None
    try:
        left, top, right, bottom = (int(v) for v in bounds)
    except (TypeError, ValueError):
        return None
    return (left + right) // 2, (top + bottom) // 2


def _flags(node: dict) -> str:
    bits = []
    if node.get("c"):
        bits.append("clickable")
    if node.get("e"):
        bits.append("editable")
    if node.get("s"):
        bits.append("scrollable")
    if node.get("f"):
        bits.append("focused")
    if node.get("k") is True:
        bits.append("checked")
    elif node.get("k") is False:
        bits.append("unchecked")
    return " ".join(bits)


def render_line(node: dict) -> str:
    index = node.get("i", "?")
    role = inert_text(node.get("r") or "", 24) or "view"
    centre = centre_of(node)
    where = f" @{centre[0]},{centre[1]}" if centre else ""
    flags = _flags(node)
    if node.get("p"):
        return f"[{index}] password field (text hidden){' ' + flags if flags else ''}{where}"
    raw = label_of(node)
    if raw and _LINK.match(raw):
        shown = "(link)"
    else:
        label = inert_text(_SPONSORED_PREFIX.sub("", raw) if is_ad(node) else raw)
        shown = f'"{label}"' if label else "(no text)"
    if is_ad(node):
        flags = f"{flags} ad".strip()
    ident = view_id_of(node)
    tag = f" #{inert_text(ident, 48)}" if ident and (node.get("c") or node.get("e") or node.get("s")) else ""
    return f"[{index}] {shown} {role}{' ' + flags if flags else ''}{tag}{where}"


def _words(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _within(inner: dict, outer: dict) -> bool:
    """Whether `inner`'s bounds lie inside `outer`'s, give or take 2px."""
    try:
        il, it, ir, ib = (int(v) for v in inner.get("b"))
        ol, ot, orr, ob = (int(v) for v in outer.get("b"))
    except (TypeError, ValueError):
        return False
    return ol - 2 <= il and ot - 2 <= it and ir <= orr + 2 and ib <= ob + 2


def shown_nodes(nodes: list[dict]) -> list[dict]:
    """The nodes a digest gives a line: all of them, except one whose label,
    as whole words, an earlier node enclosing it already shows. A clickable
    node folds only into a clickable one, so what can be tapped keeps a line.
    A password field, an editable or scrollable node and a checkable one
    (its state is on its own line) always show."""
    shown: list[tuple[dict, str]] = []
    for node in nodes:
        label = _words(label_of(node))
        foldable = label and not (node.get("p") or node.get("e") or node.get("s") or node.get("k") is not None)
        if foldable and any(
            _within(node, prior) and f" {label} " in f" {prior_label} " and (not node.get("c") or prior.get("c"))
            for prior, prior_label in shown
        ):
            continue
        shown.append((node, label))
    return [node for node, _ in shown]


def render_digest(snapshot: dict, *, max_nodes: int = MAX_NODES, max_chars: int = MAX_CHARS) -> str:
    """The whole screen, bounded. Never raises on a partial snapshot."""
    app = snapshot.get("app") or {}
    screen = snapshot.get("screen") or {}
    head = (
        f"app: {inert_text(app.get('label') or '(unknown)', 40)} "
        f"({inert_text(app.get('package') or '?', 80)})"
        f"  screen {screen.get('w', '?')}x{screen.get('h', '?')}"
        f"  keyboard: {'shown' if snapshot.get('keyboard') else 'hidden'}"
        f"  snapshot: {inert_text(snapshot.get('snapshot_id') or '?', 24)}"
    )
    if snapshot.get("secure"):
        head += "  (secure window: screenshots are blocked here)"
    nodes = [n for n in (snapshot.get("nodes") or []) if isinstance(n, dict)]
    shown = shown_nodes(nodes)
    lines = [head]
    for node in shown[:max_nodes]:
        lines.append(render_line(node))
    if len(shown) > max_nodes:
        lines.append(f"(+{len(shown) - max_nodes} more nodes not shown; scroll to see them)")
    if not nodes:
        lines.append("(no readable nodes -- a drawn or web view; try phone_look with a question)")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def find_node(snapshot: dict, text: str, *, clickable_only: bool = False) -> tuple[int | None, list[int]]:
    """`(index, candidates)`: the one node whose label or id matches `text`,
    or None with the indices that matched too loosely. Tiers, first match
    wins: exact label, exact id (its words, or "#the-id" as shown), label
    substring, id-words substring. Within a tier one node wins and several
    are ambiguous: the caller shows the candidates rather than guessing. A
    label always outranks an id, so naming what is written keeps working.
    "[12]", the number the digest shows, names that element whatever its
    text: a list with no label (Amazon's filter categories) has no other
    name, and a scroll needs one when a page holds several lists."""
    want = " ".join(str(text or "").lower().split())
    if not want:
        return None, []
    numbered = re.fullmatch(r"\[(\d+)\]", want)
    if numbered:
        wanted = int(numbered.group(1))
        for node in snapshot.get("nodes") or []:
            if (isinstance(node, dict) and node.get("i") == wanted and not node.get("p")
                    and (node.get("c") or not clickable_only)):
                return wanted, []
        return None, []
    bare = want[1:].strip() if want.startswith("#") else want
    want_words = guard.id_words(bare)
    tiers: list[list[int]] = [[], [], [], []]
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict) or node.get("p"):
            continue
        if clickable_only and not node.get("c"):
            continue
        if "i" not in node:
            continue
        label = " ".join(label_of(node).lower().split())
        ident = view_id_of(node)
        words = guard.id_words(ident)
        if not label and not words:
            continue
        if label and label == want:
            tiers[0].append(int(node["i"]))
        elif words and (ident.lower() == bare or words == want_words):
            tiers[1].append(int(node["i"]))
        elif label and want in label:
            tiers[2].append(int(node["i"]))
        elif words and want_words and want_words in words:
            tiers[3].append(int(node["i"]))
    for found in tiers:
        if len(found) == 1:
            return found[0], []
        if found:
            return None, found
    return None, []


def node_at(snapshot: dict, x: int, y: int) -> dict | None:
    """The smallest node whose bounds contain the point, or None. A tap by
    coordinates is a tap on whatever is drawn there, and the guard has to
    judge that element as it would one named by its text."""
    best, best_area = None, None
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        b = node.get("b")
        if not (isinstance(b, (list, tuple)) and len(b) == 4):
            continue
        try:
            left, top, right, bottom = (int(v) for v in b)
        except (TypeError, ValueError):
            continue
        if left <= x <= right and top <= y <= bottom:
            area = max(1, right - left) * max(1, bottom - top)
            if best_area is None or area < best_area:
                best, best_area = node, area
    return best


def node_by_index(snapshot: dict, index: int) -> dict | None:
    for node in snapshot.get("nodes") or []:
        if isinstance(node, dict) and node.get("i") == index:
            return node
    return None
