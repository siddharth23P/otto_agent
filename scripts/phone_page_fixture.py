"""Turn a phone capture into a page fixture for the money guard's corpus.

    python scripts/phone_page_fixture.py SRC OUT.json [--package P] [--label L] [--scrub WORD ...]

SRC is either the app's own capture (a debug build answers
`adb shell am broadcast -a dev.otto.phone.DUMP_TREE --es name NAME` with
files/dumps/NAME.json, `{"snapshot": ..., "windows": ...}`) or a uiautomator
dump (*.xml), replayed the way the app's TreeWalker keeps nodes. OUT is the
snapshot as the app sends it, scrubbed: a checkout capture carries a name, an
address, a phone number, a card's last digits and order ids, and none of that
belongs in a repository. `--scrub` adds words (the person's own names) to
replace; tests/test_phone_page_corpus.py scans every fixture for what this
misses. The phone guard's `page.kind` is dropped: the fixture is what both
guards judge, not what one of them already said.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

#: Mirrors the app's TreeWalker: an off-screen id is kept when one of its words is one of these.
PAGE_ID_WORDS = {"buy", "cart", "checkout", "order", "pay", "payment", "place", "submit", "ptc", "purchase"}
MAX_NODES, MAX_ID, MAX_HINT, MAX_OFFSCREEN_IDS, MAX_OFFSCREEN_ID = 400, 120, 60, 40, 60

SCRUBS = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "person@example.com"),
    (re.compile(r"\b\d{3}-\d{7}-\d{7}\b"), "000-0000000-0000000"),              # an Amazon order id
    (re.compile(r"(?:\+91[\s-]?)?\b[6-9]\d{9}\b"), "9000000000"),               # an Indian mobile number
    (re.compile(r"(?:[•*xX]\s?){2,}\s*\d{4}\b"), "•••• 0000"),                  # a masked card
    (re.compile(r"(?i)\b(ending (?:in|with))\s*\d{4}\b"), r"\1 0000"),
    (re.compile(r"\b\d{6}\b"), "000000"),                                        # a PIN code
)


def id_words(view_id: str) -> list[str]:
    text = view_id.split(":id/", 1)[-1]
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).lower()
    return [w for w in re.split(r"[^a-z0-9]+", text) if w]


def role_of(cls: str, editable: bool) -> str:
    s = cls.rsplit(".", 1)[-1].lower()
    if editable or "edittext" in s:
        return "edit-field"
    for key, name in (("button", "button"), ("checkbox", "checkbox"), ("switch", "switch"), ("toggle", "switch"),
                      ("radio", "radio"), ("image", "image")):
        if key in s:
            return name
    if any(k in s for k in ("recycler", "listview", "scrollview", "viewpager")):
        return "list"
    if "webview" in s:
        return "web"
    if "textview" in s or "text" in s:
        return "text"
    if "tab" in s:
        return "tab"
    if "seekbar" in s or "slider" in s:
        return "slider"
    return "view"


def from_uiautomator(path: Path, package: str, label: str) -> dict:
    root = ET.parse(path).getroot()
    nodes: list[dict] = []
    offscreen: list[str] = []
    width = height = 0
    for n in root.iter("node"):
        b = list(map(int, re.findall(r"-?\d+", n.get("bounds", "[0,0][0,0]"))))
        width, height = max(width, b[2]), max(height, b[3])
        text, desc = n.get("text", "").strip(), n.get("content-desc", "").strip()
        vid = n.get("resource-id", "").split(":id/")[-1]
        editable = n.get("class", "").endswith("EditText")
        clickable, scrollable = n.get("clickable") == "true", n.get("scrollable") == "true"
        sized = b[2] > b[0] and b[3] > b[1]
        if not sized and vid and len(offscreen) < MAX_OFFSCREEN_IDS and vid[:MAX_OFFSCREEN_ID] not in offscreen \
                and PAGE_ID_WORDS & set(id_words(vid)):
            offscreen.append(vid[:MAX_OFFSCREEN_ID])
        if not (sized and (text or desc or clickable or editable or scrollable)) or len(nodes) >= MAX_NODES:
            continue
        node = {"i": len(nodes) + 1, "t": "" if n.get("password") == "true" else text, "d": desc,
                "r": role_of(n.get("class", ""), editable), "b": b, "c": clickable, "e": editable, "s": scrollable,
                "p": n.get("password") == "true", "f": n.get("focused") == "true",
                "k": (n.get("checked") == "true") if n.get("checkable") == "true" else None, "v": vid[:MAX_ID]}
        if editable and n.get("hint", "").strip():
            node["h"] = n.get("hint", "").strip()[:MAX_HINT]
        if n.get("heading") == "true":
            node["g"] = True
        nodes.append(node)
    return {"snapshot_id": path.stem, "app": {"package": package, "label": label},
            "screen": {"w": width, "h": height}, "keyboard": False, "secure": False, "settled": True,
            "page": {"seq": 0, "activity": "", "offscreen_ids": offscreen}, "nodes": nodes}


#: A masked card's dots, drawn as a node of their own; the digits follow in the next node.
SEARCHY = re.compile(r"(?i)\bsearch\b")
MASK = re.compile(r"^\s*(?:[•*xX]\s?){2,}\s*$")


def scrub_text(text: str, words: list[str]) -> str:
    for pattern, replacement in SCRUBS:
        text = pattern.sub(replacement, text)
    for word in words:
        text = re.sub(re.escape(word), "Person", text, flags=re.IGNORECASE)
    return text


def scrub(snapshot: dict, name: str, words: list[str]) -> dict:
    snapshot = json.loads(json.dumps(snapshot))
    snapshot["snapshot_id"] = name
    page = snapshot.get("page")
    if isinstance(page, dict):
        page.pop("kind", None)
    nodes = [n for n in snapshot.get("nodes") or [] if isinstance(n, dict)]
    for i, node in enumerate(nodes):
        for key in ("t", "d", "h"):
            if isinstance(node.get(key), str):
                node[key] = scrub_text(node[key], words)
        # "••••" and "7005" are two nodes on Amazon's payment page: a card's last digits after its mask.
        if i and MASK.match(str(nodes[i - 1].get("t") or "")) and re.fullmatch(r"\s*\d{4}\s*", str(node.get("t") or "")):
            node["t"] = "0000"
        if node.get("e") and node.get("t") and node["t"] != node.get("h") and not SEARCHY.search(node["t"]):
            # What was typed into a field is the person's. An empty field shows its hint as its text, and a
            # search box's hint is what says it is one, so those stay.
            node["t"] = "typed text"
    return snapshot


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("src", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--package", default="in.amazon.mShop.android.shopping")
    parser.add_argument("--label", default="Amazon")
    parser.add_argument("--scrub", nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.src.suffix == ".xml":
        snapshot = from_uiautomator(args.src, args.package, args.label)
    else:
        data = json.loads(args.src.read_text(encoding="utf-8"))
        snapshot = data.get("snapshot", data)
    fixture = scrub(snapshot, args.out.stem, args.scrub)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(fixture.get('nodes') or [])} nodes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
