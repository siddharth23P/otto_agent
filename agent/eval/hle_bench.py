"""Humanity's Last Exam (cais/hle, Center for AI Safety) against Otto.

HLE is 2,500 expert-written questions with short, unambiguous answers -- no
shell, no container, no tools. It measures what the model knows and can reason
out, which makes it the opposite of agent/eval/terminal_bench.py: nothing here
exercises the workspace, the container seam or the tool loop.

That contrast is the reason to run it in two modes on the SAME sample.
`raw` asks the model directly, once, which is what a published HLE number
measures. `agent` runs Otto's whole graph -- router, planner, solver,
evaluator -- over the identical question. The published figure for this model
family is around 17%; what nobody knows is whether wrapping it in a
multi-agent graph with a verifying evaluator moves that up, leaves it alone,
or costs accuracy by giving a confident wrong answer a second reviewer who is
no better informed. Answering that is worth more here than reproducing 17%.

THE DATASET IS GATED. `cais/hle` requires accepting its terms on Hugging Face
and an access token; an unauthenticated fetch of even the README returns 401.
Set HF_TOKEN (or pass --hf-token) after accepting at
huggingface.co/datasets/cais/hle. Nothing in this module tries to work around
that gate.

Grading follows the official harness (github.com/centerforaisafety/hle):
the same answer-format instruction to the model, and a judge shown the
question, the response and the correct answer, returning a strict yes/no. The
judge here is the same Inception model everything else uses rather than the
paper's o3-mini, so these numbers are comparable ACROSS THE TWO MODES BELOW
and only loosely against a published leaderboard.

Image questions are skipped. Otto's chat path is text-only, so scoring it on
questions whose content it cannot see would measure the wrong thing -- the
count of skipped items is reported so the sample is never silently reshaped.
"""
from __future__ import annotations

import dataclasses
import json
import os
import random
import re
from pathlib import Path
from typing import Iterable

#: The answer-format instruction from the official run_model_predictions.py,
#: verbatim -- so a response is shaped the way the official judge expects to
#: read it, whichever mode produced it.
ANSWER_FORMAT_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

#: The official judge instruction, plus the strict output shape. The paper's
#: harness gets this as a structured response; asked of a plain chat model it
#: has to be spelled out, and the parse below fails CLOSED -- an unreadable
#: verdict is scored wrong, never right.
JUDGE_PROMPT = (
    "Judge whether the following [response] to [question] is correct or not "
    "based on the precise and unambiguous [correct_answer] below.\n\n"
    "[question]: {question}\n\n"
    "[response]: {response}\n\n"
    "[correct_answer]: {correct_answer}\n\n"
    "Reply with exactly two lines and nothing else:\n"
    "extracted_final_answer: <the final answer from [response], or None>\n"
    "correct: <yes or no>"
)

DEFAULT_CACHE = Path(__file__).resolve().parent / "data" / "hle-test.parquet"
HLE_PARQUET_URL = (
    "https://huggingface.co/datasets/cais/hle/resolve/main/data/test-00000-of-00001.parquet"
)


class DatasetGated(RuntimeError):
    """The HLE parquet could not be fetched without an accepted-terms token."""


def download_hle(path: Path = DEFAULT_CACHE, *, token: str | None = None) -> Path:
    """Fetch the HLE test parquet, once, using a Hugging Face token.

    `cais/hle` is gated: you must accept its terms on the dataset page, then
    supply a token with read access. Without one every path -- parquet, README,
    the datasets library -- returns 401, so this raises DatasetGated with what
    to do rather than failing somewhere less legible later.
    """
    if path.exists():
        return path
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise DatasetGated(
            "cais/hle is a gated dataset. Accept its terms at "
            "https://huggingface.co/datasets/cais/hle, create a read token at "
            "https://huggingface.co/settings/tokens, and set HF_TOKEN."
        )
    import urllib.error
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        HLE_PARQUET_URL, headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:  # nosec B310
            path.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise DatasetGated(
            f"Hugging Face refused the download ({exc.code}). The token must "
            "belong to an account that has accepted the terms for cais/hle."
        ) from exc
    return path


def load_hle(path: Path = DEFAULT_CACHE) -> list[dict]:
    import pandas  # only needed on the read path, and pulled in by fastembed already

    return pandas.read_parquet(path).to_dict("records")


def _is_text_only(row: dict) -> bool:
    return not (row.get("image") or "").strip()


def sample_questions(
    rows: Iterable[dict], *, limit: int | None, seed: int = 0,
) -> tuple[list[dict], int]:
    """(sampled text-only questions, how many image questions were skipped).

    The skipped count is returned rather than logged so a caller reports the
    sample it actually scored -- an image-heavy category silently dropping out
    would otherwise look like a change in difficulty.
    """
    rows = list(rows)
    text_only = [r for r in rows if _is_text_only(r)]
    skipped = len(rows) - len(text_only)
    if limit is not None and limit < len(text_only):
        text_only = random.Random(seed).sample(text_only, limit)
    return text_only, skipped


@dataclasses.dataclass
class QuestionResult:
    id: str
    category: str
    question: str
    correct_answer: str
    response: str
    extracted: str
    correct: bool
    #: How many LLM calls this question cost. 1 in `raw` mode; in `agent` mode
    #: the whole graph runs, which is the number this benchmark is really
    #: comparing against its accuracy.
    llm_calls: int


_CORRECT_LINE = re.compile(r"^correct:\s*(yes|no)\b", re.IGNORECASE | re.MULTILINE)
_EXTRACTED_LINE = re.compile(r"^extracted_final_answer:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


def parse_judgement(reply: str) -> tuple[str, bool]:
    """(extracted answer, correct) from a judge reply.

    Fails closed: a reply with no readable `correct:` line scores WRONG. A
    judge that did not render a verdict is not evidence the answer was right,
    the same reasoning agent/pipeline/nodes.py's _parse_approval uses.
    """
    verdict = _CORRECT_LINE.search(reply)
    extracted = _EXTRACTED_LINE.search(reply)
    return (
        (extracted.group(1).strip() if extracted else ""),
        bool(verdict) and verdict.group(1).lower() == "yes",
    )


def _answer_raw(question: str) -> tuple[str, int]:
    """One direct model call -- what a published HLE number measures."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from agent.pipeline.nodes import ROUTER, _call
    from agent.router.mapping import Task

    llm = ROUTER.chat_model(Task.REASON)
    reply = _call(llm, [SystemMessage(ANSWER_FORMAT_PROMPT), HumanMessage(question)])
    return reply, 1


def _answer_with_agent(question: str) -> tuple[str, int]:
    """Otto's whole graph over the same question.

    The counter wraps _call rather than reading usage off the provider,
    because what matters for the comparison is how many model round-trips the
    graph spends to answer one question that `raw` answers in one.
    """
    import uuid

    import agent.pipeline.nodes as nodes
    from agent.pipeline.run import run_pipeline

    calls = 0
    original = nodes._call

    def counting(llm, messages):
        nonlocal calls
        calls += 1
        return original(llm, messages)

    nodes._call = counting
    try:
        state = run_pipeline(
            f"{question}\n\n{ANSWER_FORMAT_PROMPT}",
            session_id=f"hle-{uuid.uuid4().hex[:12]}",
        )
    finally:
        nodes._call = original
    return str(state.get("output") or ""), calls


def judge(question: str, response: str, correct_answer: str) -> tuple[str, bool]:
    """Grade one response against the official judge instruction."""
    from langchain_core.messages import HumanMessage

    from agent.pipeline.nodes import ROUTER, _call
    from agent.router.mapping import Task

    # Task.EVALUATE, not REASON: this is a judge, and the two seats are on
    # different vendors now. Leaving it on REASON would silently grade HLE with
    # the solver's model while the graph's own evaluator used another.
    llm = ROUTER.chat_model(Task.EVALUATE)
    reply = _call(llm, [HumanMessage(JUDGE_PROMPT.format(
        question=question, response=response, correct_answer=correct_answer,
    ))])
    return parse_judgement(reply)


def run_hle(
    rows: list[dict], *, mode: str = "raw", on_result=None,
) -> list[QuestionResult]:
    """Answer and grade each of `rows`. `mode` is "raw" or "agent"."""
    answer = {"raw": _answer_raw, "agent": _answer_with_agent}[mode]
    results = []
    for row in rows:
        question = str(row["question"])
        try:
            response, calls = answer(question)
        except Exception as exc:  # one bad question must not end the sweep
            response, calls = f"({type(exc).__name__}: {exc})", 0
        extracted, correct = judge(question, response, str(row["answer"]))
        result = QuestionResult(
            id=str(row.get("id", "")), category=str(row.get("category", "")),
            question=question, correct_answer=str(row["answer"]),
            response=response, extracted=extracted, correct=correct, llm_calls=calls,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results


def summarise(results: list[QuestionResult]) -> dict:
    n = len(results)
    by_category: dict[str, list[QuestionResult]] = {}
    for r in results:
        by_category.setdefault(r.category or "(uncategorised)", []).append(r)
    return {
        "n": n,
        "accuracy": (sum(r.correct for r in results) / n) if n else None,
        "llm_calls_total": sum(r.llm_calls for r in results),
        "llm_calls_per_question": (sum(r.llm_calls for r in results) / n) if n else None,
        "by_category": {
            name: {"n": len(rs), "accuracy": sum(r.correct for r in rs) / len(rs)}
            for name, rs in sorted(by_category.items())
        },
    }
