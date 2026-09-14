"""The screen as words the model can act on.

A snapshot is what the phone's accessibility tree looks like once the host
has walked it (the Android app's TreeWalker):

    {"snapshot_id": "s12", "app": {"package": "...", "label": "Blinkit"},
     "screen": {"w": 1080, "h": 2400}, "keyboard": false, "secure": false,
     "nodes": [{"i": 3, "t": "Add to cart", "d": "", "r": "button",
                "b": [900, 1700, 1060, 1780], "c": true, "e": false,
                "s": false, "p": false, "f": false, "k": null}, ...]}

    i index   t text   d content description   r role   b bounds [l,t,r,b]
    c clickable   e editable   s scrollable   p password field
    f focused   k checked (true/false, null when not checkable)

The digest is one line per node, bounded, with every piece of screen text
made inert the same way agent/pipeline/toolkit.py makes a tool description
inert: control characters collapsed, length capped, and the line begins with
the index in brackets so nothing an app displays can start a line the
framework would read as its own. A password field's text is never shown.
"""
from __future__ import annotations

import re
from typing import Any

#: How many nodes a digest shows before it says how many it left out.
MAX_NODES = 120
#: Ceiling on the whole digest.
MAX_CHARS = 6000
#: Longest a single label is shown at.
MAX_LABEL = 80

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
    label = inert_text(label_of(node))
    shown = f'"{label}"' if label else "(no text)"
    return f"[{index}] {shown} {role}{' ' + flags if flags else ''}{where}"


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
    lines = [head]
    for node in nodes[:max_nodes]:
        lines.append(render_line(node))
    if len(nodes) > max_nodes:
        lines.append(f"(+{len(nodes) - max_nodes} more nodes not shown; scroll to see them)")
    if not nodes:
        lines.append("(no readable nodes -- a drawn or web view; try phone_look with a question)")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def find_node(snapshot: dict, text: str, *, clickable_only: bool = False) -> tuple[int | None, list[int]]:
    """`(index, candidates)`: the one node whose label matches `text`, or None
    with the indices that partially matched. An exact label wins over a
    substring; a unique substring wins; anything else is ambiguous and the
    caller shows the candidates rather than guessing."""
    want = " ".join(str(text or "").lower().split())
    if not want:
        return None, []
    exact: list[int] = []
    partial: list[int] = []
    for node in snapshot.get("nodes") or []:
        if not isinstance(node, dict) or node.get("p"):
            continue
        if clickable_only and not node.get("c"):
            continue
        label = " ".join(label_of(node).lower().split())
        if not label or "i" not in node:
            continue
        if label == want:
            exact.append(int(node["i"]))
        elif want in label:
            partial.append(int(node["i"]))
    if len(exact) == 1:
        return exact[0], []
    if exact:
        return None, exact
    if len(partial) == 1:
        return partial[0], []
    return None, partial


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
