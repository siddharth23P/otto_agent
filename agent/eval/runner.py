"""Golden-dataset eval harness: run the real pipeline on each golden item,
check its final answer with a real checker (execution for code, a
find-the-recomputed-value check for math), and report pass/fail per item.
This is the harness `otto eval` drives -- see agent/cli/eval.py.

Distinct from evaluator()'s runtime verification inside the pipeline itself
(agent/pipeline/nodes.py): that has no ground truth to check against (a live
user task has no known-correct answer), only whether the evaluator's own
judgment (backed by the same tool access every role node has) approves it.
A golden item DOES have a known-correct answer (the checker), so this
harness can and does check for actual correctness, not just internal
consistency -- that is the whole point of keeping a golden set: it is the
one place Otto's output gets compared against something verified in
advance, which is what makes it useful for debugging and measuring real
accuracy rather than just "did it not crash."

A checker never requires exact-string agreement -- it runs/inspects the
candidate's behavior (code: call the functions it defines; math: search the
text for the accepted numeric value). That is what "multiple correct
answers" means here: any implementation that behaves correctly passes, and
any answer text that states the right number passes, regardless of exact
wording or algorithm choice.

No `agents` anywhere in this module (2026-09-10): the router/planner/solver/
summarizer/finder/evaluator graph that replaced the swarm pipeline has
nothing to size -- see agent/pipeline/run.py's module docstring.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from agent.pipeline.run import run_pipeline
from agent.pipeline.tools import execute_python

GOLDEN_DIR = Path(__file__).parent / "golden"


@dataclass(frozen=True)
class GoldenItem:
    id: str
    domain: str
    prompt: str
    #: Python source. For domain == "code" it runs directly after the
    #: candidate's own code (so it can call whatever the candidate defined).
    #: For any other domain the candidate's raw text is injected as the
    #: string `CANDIDATE_OUTPUT` for the checker to inspect -- see
    #: _build_script(). Must print "OK" (or anything) and exit 0 on success;
    #: a failed assert (or any exception) means the item failed.
    checker: str


@dataclass(frozen=True)
class EvalResult:
    item_id: str
    domain: str
    passed: bool
    evidence: str
    seconds: float


def load_golden(directory: Path = GOLDEN_DIR) -> list[GoldenItem]:
    items = []
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text())
        items.append(GoldenItem(**data))
    return items


def _build_script(domain: str, candidate_output: str, checker: str) -> str:
    if domain == "code":
        return candidate_output + "\n\n" + checker
    # math (or any other non-code domain): candidate text isn't executable,
    # so it's injected as a string literal (repr() sidesteps any quoting/
    # escaping issue with arbitrary model output) that the checker inspects.
    return f"CANDIDATE_OUTPUT = {candidate_output!r}\n\n" + checker


def check_output(item: GoldenItem, candidate_output: str) -> tuple[bool, str]:
    script = _build_script(item.domain, candidate_output, item.checker)
    result = execute_python(script, timeout=15.0)
    if result.timed_out:
        return False, "checker timed out"
    if result.returncode != 0:
        return False, f"checker failed:\n{result.stderr}"
    return True, result.stdout.strip() or "OK"


def run_item(item: GoldenItem, *, session_id: str) -> EvalResult:
    start = time.monotonic()
    final = run_pipeline(item.prompt, session_id=session_id)
    output = (final.get("final_output") or "").strip()
    passed, evidence = check_output(item, output)
    return EvalResult(
        item_id=item.id, domain=item.domain, passed=passed, evidence=evidence,
        seconds=time.monotonic() - start,
    )


def run_golden(*, only_domain: str | None = None) -> list[EvalResult]:
    items = load_golden()
    if only_domain:
        items = [i for i in items if i.domain == only_domain]
    return [
        run_item(item, session_id=f"eval-{uuid.uuid4().hex[:8]}")
        for item in items
    ]
