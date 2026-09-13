"""Run one or more golden items through the real pipeline and print
everything a score hides: the board (one line per event), the action record
(one line per tool call), the checklist the judge held the answer to, the
final answer, and the golden checker's own verdict.

Run with:
    uv run --env-file .env python debug_pipeline.py
Target different item(s):
    DEBUG_GOLDEN_ID=code_02 uv run --env-file .env python debug_pipeline.py
    DEBUG_GOLDEN_ID=code_01,code_04,nphard_ksp_01,nphard_tsp_02 uv run --env-file .env python debug_pipeline.py
"""
import json
import os

from agent.eval.runner import check_output, GoldenItem
from agent.pipeline.run import run_pipeline

ITEM_IDS = [x.strip() for x in os.environ.get("DEBUG_GOLDEN_ID", "nphard_tsp_02").split(",") if x.strip()]

for item_id in ITEM_IDS:
    item = GoldenItem(**json.load(open(f"agent/eval/golden/{item_id}.json")))
    print(f"===== {item_id} =====")
    final = run_pipeline(item.prompt, session_id=f"debug-{item_id}")

    print(f"-- model calls: {final.get('model_calls')} -- mode swaps: {final.get('mode_swaps')} --")

    print("-- board (one line per event) --")
    for line in final.get("board") or []:
        print(f"  {line}")

    print("-- actions (one line per tool call) --")
    for line in final.get("actions") or []:
        print(f"  {line}")

    print("-- checklist (what the judge held the answer to) --")
    for item_ in final.get("checklist") or []:
        print(f"  [{item_.get('status', '?')}] {item_.get('text', item_)}")

    print("-- final output --")
    print(f"  {final.get('final_output')!r}")
    passed, evidence = check_output(item, (final.get("final_output") or "").strip())
    print(f"  golden checker: {'PASS' if passed else 'FAIL'} -- {evidence}")
    print()
