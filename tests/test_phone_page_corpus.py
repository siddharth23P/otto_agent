"""The page corpus both money guards judge.

tests/fixtures/phone_pages/ holds pages as the app sends them: Amazon's home, search, results,
product, cart, upsell and payment pages captured on a Galaxy S23 (2026-09-15/16, scrubbed by
scripts/phone_page_fixture.py), and synthetic pages for what was not captured (an order review, an
address step, OTP and PIN forms). agent/phone/assets/guard_pages.json records each one's class, its
three scores, its evidence, what Enter does and every control's verdict. This file and the app's
PageCorpusTest judge every page and must reach exactly those records: that is what keeps the
Python and the Kotlin guard the same guard, beyond sharing one rules file. The manifest is written
by scripts/phone_page_manifest.py and reviewed as a diff.
"""
from __future__ import annotations

import hashlib
import json
import re
from importlib import resources
from pathlib import Path

import pytest

from agent.phone import guard

FIXTURES = Path(__file__).parent / "fixtures" / "phone_pages"


def _manifest() -> dict:
    return json.loads(resources.files("agent.phone").joinpath("assets/guard_pages.json").read_text(encoding="utf-8"))


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


MANIFEST = _manifest()


def test_the_manifest_is_of_these_rules_and_lists_every_fixture_by_its_bytes():
    assert MANIFEST["version"] == guard.RULES_VERSION
    on_disk = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in FIXTURES.glob("*.json")}
    assert on_disk == {name: page["sha256"] for name, page in MANIFEST["pages"].items()}


@pytest.mark.parametrize("name", sorted(MANIFEST["pages"]))
def test_every_page_is_judged_as_recorded(name):
    record, _ = guard.page_record(_load(name))
    expected = {k: v for k, v in MANIFEST["pages"][name].items() if k != "sha256"}
    assert record == expected


@pytest.mark.parametrize("sequence", MANIFEST["sequences"], ids=lambda s: s["name"])
def test_every_sequence_is_judged_as_recorded(sequence):
    memory, kinds = None, []
    for name in sequence["pages"]:
        record, memory = guard.page_record(_load(name), memory)
        kinds.append(record["kind"])
    assert kinds == sequence["kinds"]


def test_the_corpus_has_every_class_of_page():
    assert {page["kind"] for page in MANIFEST["pages"].values()} == set(guard.KINDS)


def test_every_score_that_decides_a_page_is_clear_of_its_line():
    """A page scored one point under a threshold is a page one rule change from another class. Each
    score that decides -- payment always, checkout when the page is not a payment page, cart when it
    is neither -- is at or over its line, or at least two under it."""
    rules = guard.rules()["page"]
    for name, page in MANIFEST["pages"].items():
        if page["kind"] == "secure":
            continue
        for score, line in (("payment", rules["payment_min"]), ("checkout", rules["checkout_min"]), ("cart", rules["cart_min"])):
            assert page[score] >= line or page[score] <= line - 2, (name, score, page[score])
            if page[score] >= line:
                break


def test_the_real_pages_are_what_a_shopping_run_meets():
    pages = MANIFEST["pages"]
    assert pages["amazon_cart_pens.json"]["kind"] == "cart"
    assert pages["amazon_checkout_payment.json"]["kind"] == "payment"
    for name in ("amazon_home.json", "amazon_results_offers.json", "amazon_product_buybox.json", "amazon_checkout_upsell.json"):
        assert pages[name]["kind"] == "none", name
    assert pages["amazon_results_offers.json"]["enter"] == "press"
    assert ["Proceed to checkout", "", "button", "entry"] in pages["amazon_cart_pens.json"]["controls"]


#: What a scrubbed fixture must not carry: a mobile number, an order id, a card's last digits, a
#: PIN code, an email address.
PERSONAL = [
    re.compile(r"(?<!\d)[6-9]\d{9}(?!\d)"),
    re.compile(r"\b\d{3}-\d{7}-\d{7}\b"),
    re.compile(r"[•*]{2,}\s*(?!0000)\d{4}\b"),
    re.compile(r"(?<![\w₹,.])(?!000000)\d{6}(?![\w,.])"),
    re.compile(r"[\w.+-]+@(?!example\.com)[\w-]+\.[\w.]+"),
]


def test_no_fixture_carries_personal_data():
    for path in FIXTURES.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        nodes = data.get("nodes") or []
        for i, n in enumerate(nodes):
            for key in ("t", "d", "h"):
                text = str(n.get(key) or "")
                for pattern in PERSONAL:
                    assert not pattern.search(text), (path.name, n.get("i"), text)
            previous = str(nodes[i - 1].get("t") or "") if i else ""
            if re.fullmatch(r"\s*[•*]{2,}\s*", previous):
                assert str(n.get("t") or "") in ("0000", ""), (path.name, n.get("i"))
