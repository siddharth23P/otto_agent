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


#: Injected ahead of every non-code checker, so the rule below lives in ONE
#: place instead of being restated in twenty JSON strings.
#:
#: `answer_is` exists because the old checkers passed on
#:
#:     nums = [float(x) for x in re.findall(...)]
#:     assert any(abs(n - 1275.0) < 1e-6 for n in nums)
#:
#: which accepts the right number appearing ANYWHERE in the candidate's text.
#: A model that enumerates possibilities and concludes the wrong one still
#: passed, as long as the right value went by at some point -- and the harder
#: the problem, the more likely the model reasons out loud and the weaker that
#: check becomes. It was not being exploited when the issue was filed; the
#: point is that it could not have caught it.
#:
#: TWO tests, and the second is the one that matters. The value has to be
#: there, AND no `decoy` may be -- a decoy being the answer a specific wrong
#: method actually produces (first-fit needing 5 bins where 4 suffice, greedy
#: taking 4 sets where 3 do, the target read straight off a subset-sum
#: question). Naming the wrong answer is what turns "did the number appear"
#: into "did it reach the right one", and every decoy here already existed as
#: the `wrong` half of a pair in tests/test_eval_runner.py.
#:
#: Deliberately NOT "score the last number": an answer that closes with a
#: unit, a citation, or a restated question breaks that, and the failure is
#: silent -- a correct answer marked wrong.
_MATH_PREAMBLE = """
import re as _re


def _numbers(text):
    return [float(x) for x in _re.findall(r"-?\\d+\\.?\\d*", text)]


def answer_is(expected, *, decoys=(), tol=1e-6, text=None):
    '''Assert the candidate reached `expected` and did not also offer a decoy.

    A decoy is what a NAMED wrong method produces on this instance, not an
    arbitrary number -- so its presence means the candidate showed its
    working and landed somewhere else, or hedged between two answers. Either
    way it has not answered the question.
    '''
    body = CANDIDATE_OUTPUT if text is None else text
    nums = _numbers(body)
    assert any(abs(n - expected) <= tol for n in nums), (
        f"{expected} not found in {nums}"
    )
    for decoy in decoys:
        if abs(decoy - expected) <= tol:
            continue   # a decoy equal to the answer would reject every pass
        assert not any(abs(n - decoy) <= tol for n in nums), (
            f"{expected} was present but so was the wrong answer {decoy} -- "
            f"a candidate that offers both has not answered. numbers: {nums}"
        )
"""


def _build_script(domain: str, candidate_output: str, checker: str) -> str:
    if domain == "code":
        return candidate_output + "\n\n" + checker
    # math (or any other non-code domain): candidate text isn't executable,
    # so it's injected as a string literal (repr() sidesteps any quoting/
    # escaping issue with arbitrary model output) that the checker inspects,
    # followed by _MATH_PREAMBLE's `answer_is` -- the rule about what counts
    # as an answer lives there, once, rather than in twenty JSON strings.
    return (
        f"CANDIDATE_OUTPUT = {candidate_output!r}\n"
        + _MATH_PREAMBLE
        + "\n"
        + checker
    )


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
