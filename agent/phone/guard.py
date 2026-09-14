"""The money guard's rules, and the verdicts the Python side draws from them.

The rules are data (assets/guard_rules.json) so that the phone, which
ENFORCES them in Kotlin, and this package, which pre-checks with them and
tests them, read one file. The app vendors a copy and its CI asserts the two
are byte-identical; a change here fails the app's bump until it copies it.

Three verdicts, three kinds of evidence:

  package_verdict   the app in front. A package on the denied list is a
                    payment or banking app by name; a package or label with
                    a money word in it is one by description. Single signal
                    either way: the model may not act there, and phone_screen
                    does not even describe it.

  screen_verdict    what is on the screen. A sensitive pattern -- "UPI PIN",
                    "OTP", "CVV", "Pay ₹499" -- counts ONLY with a second
                    signal: an input field that ASKS for it (its own label or
                    hint matches, or it is a password field), a money-word
                    package, or a secure (FLAG_SECURE) window. A chat that
                    mentions an OTP next to its message box, or a news story
                    about banks, is neither; the first version of this rule
                    counted any input field and stopped the agent on both.

  target_verdict    the thing about to be tapped. A pay word ("Pay now",
                    "Place order", and the bare "Pay", "Buy", "Checkout") is
                    never tappable, whatever tool asks. A forward word
                    ("Continue", "Next", "Confirm") is a pay word when the
                    screen shows a checkout signal (a total, "payment", a
                    card field): that is what the last button of a checkout
                    is usually called. A commit word ("Send", "Delete", and
                    "Checkout": reaching the payment page is the person's
                    call, once) is tappable only through phone_commit, which
                    the mutation gate holds once.

WHAT THE LABEL VERDICT CANNOT SEE. `target_verdict` reads the words on the
control: its text, or its content description when it has none (an icon
button with a "Send" description is covered). A control with neither, or one
labelled in a language the word lists do not carry, is an ordinary tap to
this side. The phone shares the same lists and the same limit; the mutation
gate and the person's own hand-over are what stand behind it, and the lists
are meant to grow (guard_rules.json is data).

`denied_names` are the labels of the denied apps, for an install asked for by
name before any package is known, and for a listing whose package the phone
cannot read. `package_word_exceptions` are substrings removed before the money words are
looked for, each guarding one word: "payload" and "paypal.shopping" hide a
"pay" that is not money.

Java-compatible regex only (the app compiles the same strings with
java.util.regex): no lookbehind, no possessive quantifiers, no named groups.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from importlib import resources
from typing import Iterable

#: Characters a screen can hide inside a word: zero-width joiners and
#: spaces, soft hyphens, bidi marks, word joiners. Stripped before any
#: match, so "Pay\u200bnow" is "pay now".
_INVISIBLE = re.compile("[\u200b-\u200f\u2060-\u2064\u00ad\ufeff\u202a-\u202e\u2066-\u2069]")


#: Letters from other scripts that draw the same as a Latin one -- the
#: Cyrillic and Greek look-alikes a label can be spelled with so that
#: "Pаy now" (Cyrillic а) reads as "Pay now" to a person and as nothing to a
#: word list. NFKC does not fold across scripts, so this table does, after
#: lower-casing (the capitals lower-case to these). Not exhaustive; the
#: rest of Unicode's confusables table is a long tail, and a label spelled
#: in a script the lists do not carry is the limit the module docstring
#: names.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i",
    "ј": "j", "ѕ": "s", "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ѵ": "v", "ԍ": "g",
    "ӏ": "l", "к": "k", "т": "t", "м": "m", "в": "b", "н": "h", "ь": "b", "ѡ": "w",
    "α": "a", "ο": "o", "ρ": "p", "ν": "v", "ι": "i", "κ": "k", "υ": "u", "τ": "t",
    "ε": "e", "β": "b", "χ": "x", "γ": "y", "ς": "s",
})


def normal(text: str) -> str:
    """NFKC-folded, invisible characters removed, lower-cased, look-alike
    letters from other scripts folded to Latin, one space between words: the
    form every verdict matches against. The Kotlin side applies the same
    steps."""
    folded = unicodedata.normalize("NFKC", str(text or ""))
    lowered = _INVISIBLE.sub("", folded).lower().translate(_CONFUSABLES)
    return " ".join(lowered.split())

RULES_RESOURCE = ("agent.phone", "assets/guard_rules.json")


def rules_text() -> str:
    """The rules file verbatim -- what the app compares its copy against."""
    return resources.files(RULES_RESOURCE[0]).joinpath(RULES_RESOURCE[1]).read_text(encoding="utf-8")


class GuardRulesError(RuntimeError):
    """The rules file is missing, not JSON, or missing a section. Raised by
    every verdict until it is fixed: a guard with no rules refuses, it does
    not guess."""


@lru_cache(maxsize=1)
def rules() -> dict:
    try:
        data = json.loads(rules_text())
    except (OSError, ValueError) as exc:
        raise GuardRulesError(f"guard_rules.json is unreadable ({exc}); reinstall otto-cli-agent") from exc
    missing = [k for k in _REQUIRED_SECTIONS if not isinstance(data.get(k), list)]
    if not isinstance(data, dict) or missing:
        raise GuardRulesError(f"guard_rules.json lacks {', '.join(missing) or 'its sections'}; reinstall otto-cli-agent")
    return data


_REQUIRED_SECTIONS = ("denied_packages", "package_words", "sensitive_patterns", "pay_words",
                      "commit_words", "settings_pages")


@lru_cache(maxsize=1)
def _compiled() -> dict:
    data = rules()
    try:
        return {
            "denied": frozenset(p.lower() for p in data["denied_packages"]),
            "names": tuple(re.compile(r"(^|\W)" + re.escape(n.lower()) + r"($|\W)") for n in data.get("denied_names", ())),
            "words": tuple(w.lower() for w in data["package_words"]),
            "exceptions": tuple(w.lower() for w in data.get("package_word_exceptions", ())),
            "sensitive": tuple(re.compile(p, re.IGNORECASE) for p in data["sensitive_patterns"]),
            "pay": tuple(w.lower() for w in data["pay_words"]),
            "forward": tuple(w.lower() for w in data.get("forward_words", ())),
            "checkout": tuple(re.compile(p, re.IGNORECASE) for p in data.get("checkout_signals", ())),
            "commit": tuple(w.lower() for w in data["commit_words"]),
            "pages": tuple(data["settings_pages"]),
        }
    except (TypeError, AttributeError, re.error) as exc:
        raise GuardRulesError(f"guard_rules.json has a malformed entry ({exc}); reinstall otto-cli-agent") from exc


def settings_pages() -> tuple[str, ...]:
    return _compiled()["pages"]


def _has_word(haystack: str, words: Iterable[str], exceptions: Iterable[str] = ()) -> str:
    text = normal(haystack)
    for exc in exceptions:
        text = text.replace(exc, " ")
    for word in words:
        if word in text:
            return word
    return ""


def package_verdict(package: str, label: str = "") -> str:
    """Why the model may not act in this app, or ""."""
    c = _compiled()
    pkg = normal(package)
    if pkg in c["denied"]:
        return f"{package} is a payment or banking app"
    shown = normal(label)
    if shown and any(p.search(shown) for p in c["names"]):
        return f"{label} is a payment or banking app"
    hit = _has_word(f"{pkg} {label or ''}", c["words"], c["exceptions"])
    if hit:
        return f"{label or package} looks money-related ({hit!r})"
    return ""


def sensitive_matches(texts: Iterable[str]) -> list[str]:
    """Which screen strings match a sensitive pattern."""
    found = []
    for text in texts:
        s = normal(text)
        if any(p.search(s) for p in _compiled()["sensitive"]):
            found.append(str(text or "")[:60])
    return found


def screen_verdict(texts: Iterable[str], *, package: str = "", label: str = "",
                   editable: bool = False, secure: bool = False) -> str:
    """Why the model may not act on this screen, or "". Two signals required
    for a text match; a denied package needs none (package_verdict).
    `editable` means an input field that asks for the secret, not merely
    the presence of one -- see `snapshot_verdict`."""
    matches = sensitive_matches(texts)
    if not matches:
        return ""
    reasons = []
    if editable:
        reasons.append("an input field")
    if package_verdict(package, label):
        reasons.append("a money-related app")
    if secure:
        reasons.append("a secure window")
    if not reasons:
        return ""
    return (f"this looks like a payment or sign-in screen ({matches[0]!r} with "
            f"{', '.join(reasons)}) -- the person takes over here")


def _whole(word: str, text: str) -> bool:
    return re.search(r"(^|\W)" + re.escape(word) + r"($|\W)", text) is not None


def checkout_context(texts: Iterable[str]) -> str:
    """The first screen string that says this is a checkout -- a total, an
    order summary, "payment", a card field -- or "". What turns a forward
    word into a pay word."""
    c = _compiled()
    for text in texts:
        s = normal(text)
        if s and any(p.search(s) for p in c["checkout"]):
            return str(text or "")[:60]
    return ""


def target_verdict(label: str, texts: Iterable[str] = ()) -> str:
    """'pay' for a button the model may never tap, 'commit' for one only
    phone_commit may tap, '' for anything else. `texts` are the other
    strings on the screen: a forward word ("Continue") is a pay word only
    when one of them is a checkout signal."""
    c = _compiled()
    text = normal(label)
    if not text:
        return ""
    # A phrase matches as a substring, and also with every space removed: a
    # zero-width character hidden inside "Pay now" normalises to "paynow",
    # and "PayNow" is how some apps spell it anyway. A single word matches
    # whole ("Pay", "Pay ₹499", not "Payload"). A pay word may over-match;
    # it only ever refuses.
    squashed = text.replace(" ", "")
    for word in c["pay"]:
        if " " in word:
            if word in text or word.replace(" ", "") in squashed:
                return "pay"
        elif _whole(word, text):
            return "pay"
    if any(_whole(word, text) or (" " in word and word in text) for word in c["forward"]):
        if checkout_context(texts):
            return "pay"
    for word in c["commit"]:
        # Whole words: "Send" is a commit, "Sending…" is a status line.
        if _whole(word, text):
            return "commit"
    return ""


#: Longest a caption above a field may be to count as its label.
MAX_LABEL_CHARS = 32


def _labelled_sensitively(nodes: list, index: int) -> bool:
    """Whether the text node just before `nodes[index]` reads like a form
    label asking for a secret: short, and a sensitive match."""
    for prev in nodes[max(0, index - 2):index]:
        if prev.get("e") or prev.get("p"):
            continue
        label = normal(str(prev.get("t") or prev.get("d") or ""))
        if label and len(label) <= MAX_LABEL_CHARS and sensitive_matches([label]):
            return True
    return False


def snapshot_verdict(snapshot: dict) -> str:
    """package_verdict then screen_verdict over one snapshot."""
    app = snapshot.get("app") or {}
    why = package_verdict(str(app.get("package") or ""), str(app.get("label") or ""))
    if why:
        return why
    nodes = [n for n in (snapshot.get("nodes") or []) if isinstance(n, dict)]
    texts = [str(n.get("t") or n.get("d") or "") for n in nodes]
    # The input-field signal is the field itself asking: a password field,
    # an editable node whose own label/hint is the sensitive text, or a
    # field whose hint is generic ("Enter code") under a short label that
    # is not ("Enter your OTP"). Short, because a form label is a few words
    # and a chat line that happens to mention an OTP is a sentence -- the
    # length is what keeps the benign corpus benign.
    asks = any(n.get("p") for n in nodes) or any(
        n.get("e") and (sensitive_matches([str(n.get("t") or n.get("d") or "")])
                        or _labelled_sensitively(nodes, i))
        for i, n in enumerate(nodes)
    )
    return screen_verdict(texts, package=str(app.get("package") or ""),
                          label=str(app.get("label") or ""), editable=asks,
                          secure=bool(snapshot.get("secure")))
