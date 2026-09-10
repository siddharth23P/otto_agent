"""Run one or more golden items through the router/planner/solver/
summarizer/finder/evaluator graph and print everything: the board (a trace
of router dispatches, each specialist's answer, and evaluator verdicts),
the final output, and the golden checker's own verdict.

Replaces debug_pipeline_agents3.py (2026-09-10), which inspected the
retired swarm pipeline's subtask_specs/subtask_results/eval_verdicts --
none of which exist in this graph's AgentState (agent/pipeline/state.py).
There is no `agents` parameter anymore either: run_pipeline() takes just
(text, session_id) -- see agent/pipeline/run.py's module docstring for why.

Run with: uv run --env-file .env python debug_pipeline.py
Target different item(s): DEBUG_GOLDEN_ID=code_02 uv run --env-file .env python debug_pipeline.py
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

    print(f"-- dispatch rounds: {final['round']} -- last node: {final['node']} --")

    print(f"-- board (trace of what happened at each step) --")
    for line in final["board"]:
        print(f"  {line}")

    print(f"-- pending output at exit (may equal final_output, or be the last-rejected attempt) --")
    print(f"  {final['output']!r}")

    print(f"-- final output --")
    print(f"  {final['final_output']!r}")
    passed, evidence = check_output(item, (final["final_output"] or "").strip())
    print(f"  golden checker: {'PASS' if passed else 'FAIL'} -- {evidence}")
    print()
