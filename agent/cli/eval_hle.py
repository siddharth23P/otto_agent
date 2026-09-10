"""`otto eval-hle` -- Humanity's Last Exam against Otto (agent/eval/hle_bench.py).

Two modes on the same sample, because the interesting number is not HLE's
published figure for this model but what Otto's graph does to it: `raw` asks
the model once, `agent` runs the whole router/planner/solver/evaluator graph
over the identical question and reports how many model calls that cost.

The dataset is gated -- accept the terms at huggingface.co/datasets/cais/hle
and set HF_TOKEN. Start with a small --limit: `agent` mode spends a full graph
run per question.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.eval.hle_bench import (
    DEFAULT_CACHE,
    DatasetGated,
    download_hle,
    load_hle,
    run_hle,
    sample_questions,
    summarise,
)


def eval_hle_cmd(
    mode: Annotated[
        str, typer.Option(help="'raw' (one model call per question) or 'agent' (the whole graph)."),
    ] = "raw",
    limit: Annotated[
        Optional[int], typer.Option(help="Score only this many questions (sampled deterministically)."),
    ] = 25,
    seed: Annotated[int, typer.Option(help="Sampling seed, so two modes score the SAME questions.")] = 0,
    data_path: Annotated[
        Optional[Path], typer.Option(help=f"Cached HLE parquet (default: {DEFAULT_CACHE}).")
    ] = None,
    hf_token: Annotated[
        Optional[str], typer.Option(help="Hugging Face token; defaults to $HF_TOKEN.")
    ] = None,
    show_items: Annotated[
        bool, typer.Option("--show-items", help="Print every question, answer and verdict."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
) -> None:
    if mode not in {"raw", "agent"}:
        err(f"mode must be 'raw' or 'agent', not {mode!r}")
        raise typer.Exit(2)

    path = data_path or DEFAULT_CACHE
    try:
        download_hle(path, token=hf_token)
    except DatasetGated as exc:
        err(str(exc))
        raise typer.Exit(1)

    rows, skipped = sample_questions(load_hle(path), limit=limit, seed=seed)
    out(f"scoring {len(rows)} text-only question(s) in {mode} mode "
        f"({skipped} image question(s) skipped -- Otto's chat path is text-only)")

    def _progress(r):
        mark = "correct" if r.correct else "wrong  "
        out(f"  {mark}  {r.llm_calls:>3} call(s)  {r.question[:72]}")

    results = run_hle(rows, mode=mode, on_result=None if as_json else _progress)
    report = summarise(results)

    if as_json:
        out(json.dumps({"mode": mode, "skipped_image_questions": skipped,
                        "summary": report,
                        "items": [r.__dict__ for r in results]}, indent=2))
        return

    out("")
    out(f"accuracy: {report['accuracy']:.1%} of {report['n']}"
        f"   model calls: {report['llm_calls_total']} "
        f"({report['llm_calls_per_question']:.1f} per question)")
    for name, stats in report["by_category"].items():
        out(f"  {name:<34} {stats['accuracy']:>6.1%}  (n={stats['n']})")

    if show_items:
        out("")
        for r in results:
            out(f"[{'correct' if r.correct else 'wrong'}] {r.question[:110]}")
            out(f"   expected: {r.correct_answer[:110]}")
            out(f"   extracted: {r.extracted[:110]}")
