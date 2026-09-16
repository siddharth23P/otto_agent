"""The money guard's rules, and the verdicts the Python side draws from them.

The rules are data (assets/guard_rules.json) so that the phone, which
ENFORCES them in Kotlin, and this package, which pre-checks with them and
tests them, read one file. The app vendors a copy and its CI asserts the two
are byte-identical; a change here fails the app's bump until it copies it.
Both sides also judge one corpus of real pages (agent/phone/assets/
guard_pages.json) and must reach the same numbers on every one.

A PAGE IS JUDGED WHOLE, NOT BY A WORD ON IT. The first guard matched words:
"buy" anywhere on a screen made it a payment step, so Amazon's results
("Buy for ₹71,549 with HDFC Bank credit card" on every listing), its home
screen (a tile called "Pay") and its cart ("Proceed to checkout" under a
total) all stopped shopping runs that were nowhere near paying (2026-09-15/
16, a Galaxy S23). Now a page is one of `KINDS`:

  secure    a hard signal, any one: a payment or banking app in front
            (`package_verdict`), a payment screen by its window's class (the
            UPI PIN pad runs inside the shop's own app), a window that
            refused a screenshot as protected, or a field that asks for a
            secret -- a password, a card number, a CVV, an OTP, a PIN (not a
            postal PIN code) -- by its label, its hint, its input type or the
            short caption over it. Nothing is described or touched.

  payment   the page scores as one (`page_scores`): a final pay control
            ("Place your order", "Pay ₹499"), a list of payment methods to
            choose from, masked card numbers, a payment or order-review
            title, a total with an amount under a wide button -- and against
            it, a grid of priced products or a product page. The person pays
            here: any tap, typing or Enter hands the phone over. Amazon's
            checkout opens on its payment methods (captured 2026-09-16), so a
            shopping run is handed over there.

  checkout  an address or delivery step: a checkout title and an address
            choice. Forward buttons need phone_commit.

  cart      a checkout entry ("Proceed to checkout") with a total and cart
            rows (quantity steppers, Delete, Save for later). Entering
            checkout needs phone_commit.

  none      everything else.

A page's class is held while the same window is scrolled (`classify_page`'s
memory, keyed by the app's window changes): a review page scrolled past its
total is still a review page.

WHAT A CONTROL'S OWN WORDS STILL DECIDE, ON ANY PAGE. `control_verdict`
reads a short label -- a sentence is content, not a button -- and the
element's resource id (a web form's buttons can all read "Submit"; Amazon's
Buy Now says what it is only in `buy-now-button`). An explicit pay control
("Pay", "Buy Now", "Place order", "Slide to pay") is never tapped, whatever
the page; off a payment page it is declined without handing over, so a run
can still add something to a cart. A commit control ("Send", "Delete") needs
phone_commit. Labels are read after NFKC folding, invisible characters
removed and look-alike letters from other scripts folded (`normal`), so
"Pаy now" with a Cyrillic а is still "pay now". A control labelled in a
language the lists do not carry is the limit that remains; the page class,
the mutation gate and the person's hand-over stand behind it.

Java-compatible regex only (the app compiles the same strings with
java.util.regex): no lookbehind, no possessive quantifiers, no named groups.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable

#: Characters a screen can hide inside a word: zero-width joiners and
#: spaces, soft hyphens, bidi marks, word joiners. Stripped before any
#: match, so "Pay​now" is "paynow".
_INVISIBLE = re.compile("[​-‏⁠-⁤­﻿‪-‮⁦-⁩]")


#: Letters from other scripts that draw the same as a Latin one -- the
#: Cyrillic and Greek look-alikes a label can be spelled with so that
#: "Pаy now" (Cyrillic а) reads as "Pay now" to a person and as nothing to a
#: word list. NFKC does not fold across scripts, so this table does, after
#: lower-casing (the capitals lower-case to these). Not exhaustive; the
#: rest of Unicode's confusables table is a long tail.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ѵ": "v", "ԍ": "g",
    "ӏ": "l", "к": "k", "т": "t", "м": "m", "в": "b", "н": "h", "ь": "b", "ѡ": "w",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "υ": "u", "τ": "t",
    "ε": "e", "β": "b", "χ": "x", "γ": "y", "ς": "s",
})

#: The page classes, least strict first.
KINDS = ("none", "cart", "checkout", "payment", "secure")


def normal(text: str) -> str:
    """NFKC-folded, invisible characters removed, lower-cased, look-alike
    letters from other scripts folded to Latin, one space between words: the
    form every verdict matches against. The Kotlin side applies the same
    steps."""
    folded = unicodedata.normalize("NFKC", str(text or ""))
    lowered = _INVISIBLE.sub("", folded).lower().translate(_CONFUSABLES)
    return " ".join(lowered.split())


def id_words(view_id: str) -> str:
    """An element's resource id as words, in `normal` form: the app's
    package prefix dropped ("com.app:id/buyNow" is "buyNow"), camelCase and
    -_./: split. "buy-now-button" and "buyNowButton" both read "buy now
    button". The Kotlin side splits the same way."""
    text = str(view_id or "")
    if ":id/" in text:
        text = text.split(":id/", 1)[1]
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return normal(re.sub(r"[-_./:#]+", " ", text))


RULES_RESOURCE = ("agent.phone", "assets/guard_rules.json")
RULES_VERSION = 3


def rules_text() -> str:
    """The rules file verbatim -- what the app compares its copy against."""
    return resources.files(RULES_RESOURCE[0]).joinpath(RULES_RESOURCE[1]).read_text(encoding="utf-8")


class GuardRulesError(RuntimeError):
    """The rules file is missing, not JSON, of another version, or missing a
    section. Raised by every verdict until it is fixed: a guard with no rules
    refuses, it does not guess."""


#: Every section and the JSON type it must have.
_REQUIRED_SECTIONS: dict[str, type] = {
    "denied_packages": list, "denied_names": list, "package_words": dict, "package_word_exceptions": list,
    "secure_activities": list, "sensitive_field_patterns": list, "sensitive_screen_patterns": list,
    "pay_controls": dict, "entry_controls": dict, "forward_controls": dict, "commit_words": list,
    "commit_max_words": int, "control_max_words": int, "page": dict, "settings_pages": list,
}


@lru_cache(maxsize=1)
def rules() -> dict:
    try:
        data = json.loads(rules_text())
    except (OSError, ValueError) as exc:
        raise GuardRulesError(f"guard_rules.json is unreadable ({exc}); reinstall otto-cli-agent") from exc
    if not isinstance(data, dict):
        raise GuardRulesError("guard_rules.json is not an object; reinstall otto-cli-agent")
    missing = [k for k, kind in _REQUIRED_SECTIONS.items() if not isinstance(data.get(k), kind)]
    if missing:
        raise GuardRulesError(f"guard_rules.json lacks {', '.join(missing)}; reinstall otto-cli-agent")
    if data.get("version") != RULES_VERSION:
        raise GuardRulesError(f"guard_rules.json is version {data.get('version')!r}; this otto reads "
                              f"{RULES_VERSION}; reinstall otto-cli-agent")
    return data


def _word_re(phrase: str) -> re.Pattern:
    """A phrase as a whole-word pattern over `normal` text: word edges only
    where the phrase starts or ends with a letter or digit, so "pay ₹"
    matches "pay ₹499"."""
    phrase = normal(phrase)
    head = r"(^|\W)" if phrase[:1].isalnum() else ""
    tail = r"($|\W)" if phrase[-1:].isalnum() else ""
    return re.compile(head + re.escape(phrase) + tail)


@lru_cache(maxsize=1)
def _compiled() -> dict:
    data = rules()
    try:
        page = data["page"]
        words = data["package_words"]
        return {
            "denied": frozenset(normal(p) for p in data["denied_packages"]),
            "names": tuple(_word_re(n) for n in data["denied_names"]),
            "whole": frozenset(normal(w) for w in words["whole"]),
            "affix": tuple(normal(w) for w in words["affix"]),
            "exceptions": tuple(normal(w) for w in data["package_word_exceptions"]),
            "activities": tuple(re.compile(p) for p in data["secure_activities"]),
            "field": tuple(re.compile(p, re.IGNORECASE) for p in data["sensitive_field_patterns"]),
            "screen": tuple(re.compile(p, re.IGNORECASE) for p in data["sensitive_screen_patterns"]),
            "pay_exact": frozenset(normal(w) for w in data["pay_controls"]["exact"]),
            "pay_phrases": tuple(_word_re(w) for w in data["pay_controls"]["phrases"]),
            "pay_squashed": frozenset(normal(w).replace(" ", "") for w in data["pay_controls"]["phrases"]
                                      if " " in normal(w) and normal(w).replace(" ", "").isalpha()),
            "pay_ids": tuple(_word_re(w) for w in data["pay_controls"]["ids"]),
            "entry_exact": frozenset(normal(w) for w in data["entry_controls"]["exact"]),
            "entry_phrases": tuple(_word_re(w) for w in data["entry_controls"]["phrases"]),
            "entry_ids": tuple(_word_re(w) for w in data["entry_controls"]["ids"]),
            "forward_exact": frozenset(normal(w) for w in data["forward_controls"]["exact"]),
            "forward_prefixes": tuple(normal(w) for w in data["forward_controls"]["prefixes"]),
            "forward_ids": tuple(_word_re(w) for w in data["forward_controls"]["ids"]),
            "commit": tuple(normal(w) for w in data["commit_words"]),
            "commit_ids": tuple(_word_re(w) for w in data["commit_words"]),
            "commit_max": int(data["commit_max_words"]),
            "control_max": int(data["control_max_words"]),
            "pages": tuple(data["settings_pages"]),
            "payment_min": int(page["payment_min"]),
            "checkout_min": int(page["checkout_min"]),
            "cart_min": int(page["cart_min"]),
            "weights": {k: int(v) for k, v in page["weights"].items()},
            "final_pay": tuple(_word_re(w) for w in page["final_pay_phrases"]),
            "amount": re.compile(page["amount"], re.IGNORECASE),
            "total": re.compile(page["total"], re.IGNORECASE),
            "payment_titles": tuple(normal(w) for w in page["payment_titles"]),
            "methods": tuple(re.compile(p, re.IGNORECASE) for p in page["payment_methods"]),
            "masked": re.compile(page["masked_card"], re.IGNORECASE),
            "checkout_titles": tuple(normal(w) for w in page["checkout_titles"]),
            "address_choices": tuple(_word_re(w) for w in page["address_choices"]),
            "address_ids": tuple(_word_re(w) for w in page["address_ids"]),
            "cart_titles": tuple(normal(w) for w in page["cart_titles"]),
            "cart_structure": tuple(re.compile(p, re.IGNORECASE) for p in page["cart_structure"]),
            "cart_ids": tuple(re.compile(p, re.IGNORECASE) for p in page["cart_ids"]),
            "product_ids": tuple(w.lower() for w in page["product_ids"]),
            "add_to_cart": re.compile(page["add_to_cart"], re.IGNORECASE),
            "top_region": float(page["top_region"]),
            "wide_control": float(page["wide_control"]),
            "title_max": int(page["title_max_words"]),
            "row_band": int(page["row_band"]),
            "row_container_max": int(page["row_container_max"]),
        }
    except (TypeError, AttributeError, KeyError, ValueError, re.error) as exc:
        raise GuardRulesError(f"guard_rules.json has a malformed entry ({exc}); reinstall otto-cli-agent") from exc


def settings_pages() -> tuple[str, ...]:
    return _compiled()["pages"]


# --------------------------------------------------------------------------
# Apps
# --------------------------------------------------------------------------

def _tokens(text: str) -> list[str]:
    """A package or a label as words: split at camelCase, then at anything
    that is not a letter (dots, underscores, digits, spaces)."""
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", unicodedata.normalize("NFKC", str(text or "")))
    return [t for t in re.split(r"[^a-z]+", normal(spaced)) if t]


def package_verdict(package: str, label: str = "") -> str:
    """Why the model may not act in this app, or "". A denied package or a
    denied app's name decides alone. A money word decides as a word of the
    package or the label: a `whole` word must be one ("upi", "emi", "lend" --
    not the middle of "gemini" or "calendar"); an `affix` word may also begin
    or end one ("superbank", "paytm", "hdfcbank")."""
    c = _compiled()
    pkg = normal(package)
    if pkg in c["denied"]:
        return f"{package} is a payment or banking app"
    shown = normal(label)
    if shown and any(p.search(shown) for p in c["names"]):
        return f"{label} is a payment or banking app"
    source = f"{package} {label or ''}"
    for exc in c["exceptions"]:
        source = re.sub(re.escape(exc), " ", source, flags=re.IGNORECASE)
    for token in _tokens(source):
        if token in c["whole"]:
            return f"{label or package} looks money-related ({token!r})"
        for word in c["affix"]:
            if token == word or (token.startswith(word) or token.endswith(word)):
                return f"{label or package} looks money-related ({word!r})"
    return ""


# --------------------------------------------------------------------------
# Secrets
# --------------------------------------------------------------------------

def sensitive_text(texts: Iterable[str]) -> list[str]:
    """Which strings name a secret outright ("UPI PIN", "Enter your OTP",
    "CVV"): what a screenshot, a look's description or a learned note must
    not carry. A chat that mentions an OTP matches; whether its screen may be
    acted on is `sensitive_field`'s question, not this one."""
    found = []
    for text in texts:
        s = normal(text)
        if s and any(p.search(s) for p in _compiled()["screen"]):
            found.append(str(text or "")[:60])
    return found


#: Longest a caption above a field may be to count as its label.
MAX_LABEL_CHARS = 32
#: How many one-character fields in a row read as an OTP's boxes.
OTP_BOXES = 4


def _node_text(node: dict) -> str:
    return str(node.get("t") or node.get("d") or "")


def sensitive_field(nodes: list, index: int) -> str:
    """What `nodes[index]` asks for when it is a field asking for a secret,
    or "": a password field (by its flag or its input type), or a field whose
    own label or hint, or the short caption just before it, names one. A
    chat's message box under a line that mentions an OTP is under a sentence,
    not a caption."""
    node = nodes[index]
    if not isinstance(node, dict) or not node.get("e"):
        return ""
    if node.get("p") or node.get("n") in ("pw", "numpw"):
        return "a password field"
    field = _compiled()["field"]
    for own in (_node_text(node), str(node.get("h") or "")):
        s = normal(own)
        if s and any(p.search(s) for p in field):
            return f"a field asking for {own[:40]!r}"
    for prev in nodes[max(0, index - 2):index]:
        if not isinstance(prev, dict) or prev.get("e") or prev.get("p"):
            continue
        caption = normal(_node_text(prev))
        if caption and len(caption) <= MAX_LABEL_CHARS and any(p.search(caption) for p in field):
            return f"a field under {_node_text(prev)[:40]!r}"
    return ""


def _otp_boxes(nodes: list) -> bool:
    run = 0
    for node in nodes:
        if isinstance(node, dict) and node.get("e") and node.get("m") == 1:
            run += 1
            if run >= OTP_BOXES:
                return True
        else:
            run = 0
    return False


def _page(snapshot: dict) -> dict:
    page = snapshot.get("page")
    return page if isinstance(page, dict) else {}


def secure_reason(snapshot: dict) -> str:
    """Why nothing on this screen may be described or touched, or "": a
    payment or banking app, a payment screen by its window's class, a window
    that refused a screenshot, or a field asking for a secret."""
    app = snapshot.get("app") or {}
    why = package_verdict(str(app.get("package") or ""), str(app.get("label") or ""))
    if why:
        return why
    activity = str(_page(snapshot).get("activity") or "").lower()
    if activity and any(p.search(activity) for p in _compiled()["activities"]):
        return "this is a payment screen (its window is a payment component) -- the person takes over here"
    if snapshot.get("secure"):
        return "this window is protected (secure content) -- the person takes over here"
    nodes = [n for n in (snapshot.get("nodes") or []) if isinstance(n, dict)]
    for i in range(len(nodes)):
        if asks := sensitive_field(nodes, i):
            return f"this looks like a payment or sign-in screen ({asks}) -- the person takes over here"
    if _otp_boxes(nodes):
        return "this looks like a payment or sign-in screen (a code's digit boxes) -- the person takes over here"
    return ""


def snapshot_verdict(snapshot: dict) -> str:
    """Why this screen is not even described: `secure_reason`."""
    return secure_reason(snapshot)


# --------------------------------------------------------------------------
# Controls
# --------------------------------------------------------------------------

_SEPARATORS = re.compile(r"[·|•+\-–—:,!.?&]+")


def strip_amounts(text: str) -> str:
    """A label in `normal` form with its amounts, bracketed counts and
    separators taken out: "Buy · ₹1,299" is "buy", "Proceed to Buy (1 item)"
    is "proceed to buy"."""
    s = _compiled()["amount"].sub(" ", normal(text))
    s = re.sub(r"\([^)]*\)", " ", s)
    s = _SEPARATORS.sub(" ", s)
    return " ".join(s.split())


def _label_verdict(label: str, role: str = "") -> str:
    c = _compiled()
    text = normal(label)
    if not text:
        return ""
    stripped = strip_amounts(text)
    words = stripped.split()
    if len(words) <= c["control_max"]:
        # Only a short label is a control's own words; a sentence that
        # mentions buying is content.
        if stripped in c["pay_exact"] or any(p.search(text) or p.search(stripped) for p in c["pay_phrases"]):
            return "pay"
        # "PayNow", and "Pay​now" once its zero-width space is gone.
        if any(w in c["pay_squashed"] for w in words):
            return "pay"
        if stripped in c["entry_exact"] or any(p.search(stripped) for p in c["entry_phrases"]):
            return "entry"
    for word in c["commit"]:
        if (stripped == word or stripped.startswith(word + " ")) and (len(words) <= c["commit_max"] or role == "button"):
            return "commit"
    if len(words) <= c["control_max"] and (
            stripped in c["forward_exact"] or any(stripped == p or stripped.startswith(p + " ") for p in c["forward_prefixes"])):
        return "forward"
    return ""


def _id_verdict(view_id: str) -> str:
    c = _compiled()
    words = id_words(view_id)
    if not words:
        return ""
    if any(p.search(words) for p in c["pay_ids"]):
        return "pay"
    if any(p.search(words) for p in c["entry_ids"]):
        return "entry"
    if any(p.search(words) for p in c["commit_ids"]):
        return "commit"
    if any(p.search(words) for p in c["forward_ids"]):
        return "forward"
    return ""


#: Control verdicts, strictest first.
CONTROL_ORDER = ("pay", "entry", "commit", "forward")


def control_verdict(label: str, view_id: str = "", role: str = "") -> str:
    """What tapping this control does, by its own short label and its id:
    'pay' (never tapped), 'entry' (enters checkout), 'commit' (cannot be
    taken back), 'forward' (a Continue: what it does depends on the page),
    or ''. The stricter of the label's and the id's verdicts."""
    verdicts = (_label_verdict(label, role), _id_verdict(view_id))
    for kind in CONTROL_ORDER:
        if kind in verdicts:
            return kind
    return ""


def mentions_pay_control(text: str) -> bool:
    """Whether a text names a pay control anywhere in it ("tap Buy Now"):
    what a learned note must not point at."""
    c = _compiled()
    s = normal(text)
    return bool(s) and (any(p.search(s) for p in c["pay_phrases"]) or strip_amounts(s) in c["pay_exact"])


def looks_like_payment(text: str) -> bool:
    """Whether a description of a screen -- a look's answer, all a blind tap
    has to go on -- describes a payment step: a secret, a pay control, a
    checkout or payment page, or a total with an amount."""
    c = _compiled()
    s = normal(text)
    if not s:
        return False
    return bool(sensitive_text([s]) or mentions_pay_control(s) or re.search(r"(^|\W)(checkout|payment|pay)($|\W)", s)
                or (c["total"].search(s) and c["amount"].search(s)))


def search_focused(nodes) -> bool:
    """Whether the field being typed in is a search box: a focused, editable,
    non-password node whose label (an empty field shows its hint), hint or
    resource id says "search". Enter there runs a search, whatever surrounds
    it (2026-09-16)."""
    for n in nodes or ():
        if not isinstance(n, dict) or not n.get("e") or not n.get("f") or n.get("p"):
            continue
        label = normal(f"{_node_text(n)} {n.get('h') or ''}")
        if re.search(r"(^|\W)search($|\W)", label) or " search " in f" {id_words(str(n.get('v') or ''))} ":
            return True
    return False


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Control:
    """Something a tap could act on: a clickable element, a text inside a
    clickable container with no words of its own (a tile), or the id of an
    element scrolled out of view (`bounds` None)."""
    label: str
    view_id: str
    role: str
    bounds: tuple[int, int, int, int] | None

    @property
    def verdict(self) -> str:
        return control_verdict(self.label, self.view_id, self.role)

    def named(self) -> str:
        tag = f" #{self.view_id[:60]}" if self.view_id else ""
        return f"{self.label[:60]!r}{tag}" if self.label.strip() else tag.strip()


def _bounds(node: dict) -> tuple[int, int, int, int] | None:
    b = node.get("b")
    if isinstance(b, (list, tuple)) and len(b) == 4 and all(isinstance(v, (int, float)) for v in b):
        return int(b[0]), int(b[1]), int(b[2]), int(b[3])
    return None


def _inside(inner: tuple, outer: tuple) -> bool:
    return outer[0] <= inner[0] and outer[1] <= inner[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def offscreen_ids(snapshot: dict) -> list[str]:
    return [v for v in _page(snapshot).get("offscreen_ids") or [] if isinstance(v, str)]


#: How many texts inside a labelless clickable container are its labels.
MAX_TILE_TEXTS = 3


def controls(snapshot: dict) -> list[Control]:
    """Every control on the page: the clickable elements, the texts of
    clickable containers that have none of their own, then the ids off
    screen."""
    nodes = [n for n in (snapshot.get("nodes") or []) if isinstance(n, dict)]
    found: list[Control] = []
    for node in nodes:
        if not node.get("c"):
            continue
        box = _bounds(node)
        label = _node_text(node)
        view_id, role = str(node.get("v") or ""), str(node.get("r") or "")
        found.append(Control(label, view_id, role, box))
        if not label.strip() and box is not None:
            inner = [n for n in nodes if not n.get("c") and _node_text(n).strip()
                     and (b := _bounds(n)) is not None and _inside(b, box)]
            found.extend(Control(_node_text(n), view_id, role, _bounds(n)) for n in inner[:MAX_TILE_TEXTS])
    found.extend(Control("", view_id, "", None) for view_id in offscreen_ids(snapshot))
    return found


def _screen_size(snapshot: dict, nodes: list) -> tuple[int, int]:
    screen = snapshot.get("screen") or {}
    w, h = screen.get("w"), screen.get("h")
    boxes = [b for n in nodes if (b := _bounds(n)) is not None]
    if not isinstance(w, int) or w <= 0:
        w = max((b[2] for b in boxes), default=1)
    if not isinstance(h, int) or h <= 0:
        h = max((b[3] for b in boxes), default=1)
    return max(w, 1), max(h, 1)


def _title_match(text: str, titles: tuple[str, ...]) -> bool:
    s = strip_amounts(text)
    return any(s == t or (" " in t and s.startswith(t + " ")) for t in titles)


def page_features(snapshot: dict) -> dict[str, bool]:
    """The evidence about what this page is, each yes or no. The words, the
    weights and the thresholds are the rules' `page` section."""
    c = _compiled()
    nodes = [n for n in (snapshot.get("nodes") or []) if isinstance(n, dict)]
    width, height = _screen_size(snapshot, nodes)
    found = controls(snapshot)
    on_screen = [k for k in found if k.bounds is not None and k.label.strip()]

    final_pay = any(len(strip_amounts(k.label).split()) <= c["control_max"]
                    and any(p.search(normal(k.label)) for p in c["final_pay"]) for k in on_screen)
    bare_pay = any(strip_amounts(k.label) in ("pay", "buy", "purchase") for k in on_screen)

    # A list to choose from: checkable elements, each with a payment method in its row.
    rows = 0
    for node in nodes:
        box = _bounds(node)
        if box is None or (node.get("k") is None and node.get("r") != "radio"):
            continue
        holders = [b for n in nodes if (b := _bounds(n)) is not None and b != box and _inside(box, b)
                   and b[3] - b[1] <= c["row_container_max"]]
        if holders:
            row = min(holders, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            top, bottom = row[1], row[3]
        else:
            top, bottom = box[1] - c["row_band"], box[3] + c["row_band"]
        beside = " ".join(normal(_node_text(n)) for n in nodes
                          if (b := _bounds(n)) is not None and b[1] < bottom and b[3] > top and _node_text(n).strip())
        if any(p.search(beside) for p in c["methods"]):
            rows += 1
    method_list = rows >= 2
    masked = any(c["masked"].search(normal(_node_text(n))) for n in nodes if _node_text(n).strip())

    top_limit = height * c["top_region"]
    titles = [n for n in nodes if _node_text(n).strip() and (
        n.get("g") or ((b := _bounds(n)) is not None and b[1] < top_limit
                       and len(strip_amounts(_node_text(n)).split()) <= c["title_max"]))]
    payment_title = any(_title_match(_node_text(n), c["payment_titles"]) for n in titles)
    checkout_title = any(_title_match(_node_text(n), c["checkout_titles"]) for n in titles)
    cart_title = any(_title_match(_node_text(n), c["cart_titles"]) for n in titles)

    totals = []
    for n in nodes:
        s, box = normal(_node_text(n)), _bounds(n)
        if not s or box is None or not c["total"].search(s):
            continue
        middle = (box[1] + box[3]) / 2
        if c["amount"].search(s) or any(
                m is not n and c["amount"].search(normal(_node_text(m))) and (mb := _bounds(m)) is not None
                and abs((mb[1] + mb[3]) / 2 - middle) <= 40 for m in nodes):
            totals.append(box)
    primary_after_total = any(
        n.get("c") and (b := _bounds(n)) is not None and b[2] - b[0] >= width * c["wide_control"]
        and any(t[3] - 10 <= b[1] <= t[3] + 400 for t in totals)
        for n in nodes)

    priced = sum(1 for n in nodes if n.get("c") and c["amount"].search(normal(_node_text(n))))
    adds = sum(1 for k in on_screen if c["add_to_cart"].search(normal(k.label)))
    listing = priced >= 3 or adds >= 4
    ids = [str(n.get("v") or "") for n in nodes if n.get("v")] + offscreen_ids(snapshot)
    product_page = any(p in v.lower() for v in ids for p in c["product_ids"])
    address_choice = (any(p.search(normal(k.label)) for k in on_screen for p in c["address_choices"])
                      or any(p.search(id_words(v)) for v in ids for p in c["address_ids"]))
    entry = any(k.verdict == "entry" for k in found)
    cart_structure = (any(p.search(normal(k.label)) for k in on_screen for p in c["cart_structure"])
                      or any(p.search(v) for v in ids for p in c["cart_ids"]))
    return {
        "final_pay_control": final_pay, "bare_pay_control": bare_pay, "payment_method_list": method_list,
        "masked_card": masked, "payment_title": payment_title, "total_with_amount": bool(totals),
        "primary_after_total": primary_after_total, "listing": listing, "product_page": product_page,
        "checkout_title": checkout_title, "address_choice": address_choice, "entry_control": entry,
        "cart_structure": cart_structure, "cart_title": cart_title, "no_clickable_amounts": priced == 0,
    }


def page_scores(features: dict[str, bool]) -> tuple[int, int, int]:
    """(payment, checkout, cart) from `page_features` and the rules' weights."""
    w = _compiled()["weights"]
    f = {k: int(bool(v)) for k, v in features.items()}
    payment = (w["final_pay_control"] * f["final_pay_control"] + w["bare_pay_control"] * f["bare_pay_control"]
               + w["payment_method_list"] * f["payment_method_list"] + w["masked_card"] * f["masked_card"]
               + w["payment_title"] * f["payment_title"] + w["total_with_amount"] * f["total_with_amount"]
               + w["primary_after_total"] * f["primary_after_total"])
    if not f["final_pay_control"]:
        # A Buy Now sheet opens over its product page, whose ids stay in the tree: the
        # evidence against speaks only when no final pay control does.
        payment += w["listing"] * f["listing"] + w["product_page"] * f["product_page"]
    checkout = (w["checkout_title"] * f["checkout_title"] + w["address_choice"] * f["address_choice"]
                + w["checkout_total"] * f["total_with_amount"] + w["checkout_listing"] * f["listing"])
    cart = (w["entry_control"] * f["entry_control"] + w["cart_total"] * f["total_with_amount"]
            + w["cart_structure"] * f["cart_structure"] + w["cart_title"] * f["cart_title"]
            + w["no_clickable_amounts"] * f["no_clickable_amounts"] + w["cart_listing"] * f["listing"])
    return payment, checkout, cart


@dataclass(frozen=True)
class PageVerdict:
    kind: str
    reason: str
    payment: int
    checkout: int
    cart: int
    features: frozenset[str]
    #: What the next judgement of the same window starts from: (key, kind), or None.
    memory: tuple | None = None


def page_key(snapshot: dict) -> tuple | None:
    """Which window a snapshot is of: the app, the class it named for its
    window, and how many window changes it has made. None from a phone that
    does not say (an app built before pages were judged): nothing is held."""
    page = _page(snapshot)
    if not isinstance(page.get("seq"), int):
        return None
    app = snapshot.get("app") or {}
    return str(app.get("package") or ""), str(page.get("activity") or ""), page["seq"]


def stricter(a: str, b: str) -> str:
    return a if KINDS.index(a) >= KINDS.index(b) else b


_REASONS = (
    ("payment_method_list", "payment methods to choose from"), ("final_pay_control", "a final pay button"),
    ("masked_card", "saved cards"), ("payment_title", "a payment or order-review title"),
    ("bare_pay_control", "a Pay button"), ("total_with_amount", "a total"), ("checkout_title", "a checkout title"),
    ("address_choice", "an address to choose"), ("entry_control", "a checkout button"), ("cart_structure", "cart rows"),
)


def classify_page(snapshot: dict, memory: tuple | None = None) -> PageVerdict:
    """What this page is (`KINDS`), judged whole, and the memory the next
    judgement of the same window starts from. A page stays at least as strict
    as it was judged on the same window -- scrolling does not change what a
    page is -- until it shows it is something else: a grid of priced
    products, a product page, or, from a payment page, a cart whose payment
    score is well under the line. The phone's own `page.kind`, when it sent
    one, is a floor."""
    why = secure_reason(snapshot)
    features = page_features(snapshot)
    payment, checkout, cart = page_scores(features)
    on = frozenset(k for k, v in features.items() if v)
    floor = str(_page(snapshot).get("kind") or "")
    if why or floor == "secure":
        return PageVerdict("secure", why or "the phone judged this screen protected -- the person takes over here",
                           payment, checkout, cart, on)
    c = _compiled()
    now = ("payment" if payment >= c["payment_min"] else "checkout" if checkout >= c["checkout_min"]
           else "cart" if cart >= c["cart_min"] else "none")
    key = page_key(snapshot)
    kind = now
    if memory and key is not None and memory[0] == key:
        held = memory[1]
        if features["listing"] or (features["product_page"] and not features["final_pay_control"]):
            kind = now
        elif held == "payment" and now == "cart" and payment <= c["payment_min"] - 2:
            # Back from a payment page to a cart the same window shows (Amazon's checkout and cart
            # share one WebView): a cart that is clearly one.
            kind = now
        else:
            kind = stricter(now, held)
    if floor in KINDS:
        kind = stricter(kind, floor)
    shown = [text for feature, text in _REASONS if feature in on][:3]
    reason = ", ".join(shown) if kind == now and shown else ("as it was judged before it scrolled" if kind != now else "")
    return PageVerdict(kind, reason, payment, checkout, cart, on,
                       (key, kind) if key is not None and kind in ("cart", "checkout", "payment") else None)


def enter_verdict(snapshot: dict, kind: str) -> str:
    """What pressing Enter on this page may do: '' (press it), 'decline' (it
    could submit a pay or checkout control; tap the one you mean) or
    'handover' (a payment page). A focused search box submits a search."""
    if kind in ("secure", "payment"):
        return "handover"
    if search_focused(snapshot.get("nodes") or []):
        return ""
    if kind in ("cart", "checkout"):
        return "decline"
    return "decline" if any(k.verdict in ("pay", "entry") for k in controls(snapshot)) else ""


def enter_blocker(snapshot: dict) -> str:
    """The first pay or checkout control on the page, as a refusal names it, or ""."""
    for k in controls(snapshot):
        if k.verdict in ("pay", "entry"):
            return k.named()
    return ""


def page_record(snapshot: dict, memory: tuple | None = None) -> tuple[dict[str, Any], tuple | None]:
    """A page as the corpus (agent/phone/assets/guard_pages.json) records it -- its class, its
    three scores, its evidence, what Enter does and every control's verdict, in order -- and the
    memory for the next page of a sequence. The app's PageCorpusTest builds the same record."""
    verdict = classify_page(snapshot, memory)
    record = {
        "kind": verdict.kind, "payment": verdict.payment, "checkout": verdict.checkout, "cart": verdict.cart,
        "features": sorted(verdict.features), "enter": enter_verdict(snapshot, verdict.kind) or "press",
        "controls": [[k.label, k.view_id, k.role, k.verdict] for k in controls(snapshot) if k.verdict],
    }
    return record, verdict.memory
