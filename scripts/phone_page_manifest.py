"""Write agent/phone/assets/guard_pages.json: every page in tests/fixtures/phone_pages/ as the
money guard judges it now.

    python scripts/phone_page_manifest.py

The manifest is what both guards must agree with -- otto's tests/test_phone_page_corpus.py and the
app's PageCorpusTest both judge every fixture and compare with it -- so rewriting it is a decision,
not a chore: read the diff. A page whose class changed, or a score that moved next to its
threshold, is a change to what Otto may do on a real screen.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.phone import guard  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "phone_pages"
MANIFEST = ROOT / "agent" / "phone" / "assets" / "guard_pages.json"

#: Pages judged one after another with the memory of the one before, as a run would see them.
SEQUENCES = {
    "a payment page scrolled stays a payment page": ["amazon_checkout_payment", "amazon_checkout_payment_scrolled"],
    "an order review scrolled past its total is still one, until a product grid shows": [
        "amazon_order_review", "amazon_order_review_scrolled", "amazon_results_after_review"],
}


def main() -> int:
    pages = {}
    for path in sorted(FIXTURES.glob("*.json")):
        data = path.read_bytes()
        record, _ = guard.page_record(json.loads(data))
        pages[path.name] = {"sha256": hashlib.sha256(data).hexdigest(), **record}
    sequences = []
    for name, steps in SEQUENCES.items():
        memory, kinds = None, []
        for step in steps:
            record, memory = guard.page_record(json.loads((FIXTURES / f"{step}.json").read_text(encoding="utf-8")), memory)
            kinds.append(record["kind"])
        sequences.append({"name": name, "pages": [f"{step}.json" for step in steps], "kinds": kinds})
    manifest = {"version": guard.RULES_VERSION, "pages": pages, "sequences": sequences}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for name, record in pages.items():
        print(f"{name:40} {record['kind']:9} P={record['payment']:3} K={record['checkout']:3} C={record['cart']:3} "
              f"enter={record['enter']}")
    for sequence in sequences:
        print(f"{sequence['name']}: {sequence['kinds']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
