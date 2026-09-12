"""`otto eval-compaction` -- what each compaction policy actually loses.

The instrument Otto did not have. `otto eval-memory` scores whether raw
evidence stays RETRIEVABLE, which the chunk store nearly guarantees; this
scores whether the constraint is still in what the prompt reads. Those are
different questions and only the second one distinguishes compaction policies.

Offline and free by default: the summariser is a deterministic stand-in, so a
policy comparison measures the POLICY rather than whichever model happened to
summarise that day.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.eval.compaction_bench import POLICIES, run_matrix


def eval_compaction_cmd(
    turns: Annotated[
        int, typer.Option(help="How long a conversation to replay. Longer means "
                               "more compaction rounds, which is where policies diverge."),
    ] = 120,
    policy: Annotated[
        Optional[str],
        typer.Option(help="Run one policy instead of the whole matrix."),
    ] = None,
    live: Annotated[
        bool,
        typer.Option("--live", help="Use the real Task.SUMMARIZE model instead of the "
                                    "deterministic stand-in. Answers a question about "
                                    "the MODEL, not about the policy, and costs calls."),
    ] = False,
    out_dir: Annotated[
        Optional[Path], typer.Option(help="Where to put the stores (default: a temp dir)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    chosen = POLICIES
    if policy:
        if policy not in POLICIES:
            err.print(f"no policy {policy!r} -- have {', '.join(POLICIES)}")
            raise typer.Exit(2)
        chosen = {policy: POLICIES[policy]}

    summarize = None
    if live:
        from agent.memory.wiring import summarize_for_memory

        summarize = summarize_for_memory
        err.print("[warn]--live[/] spends model calls, one per compaction round per policy")

    root = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="otto-compaction-"))
    root.mkdir(parents=True, exist_ok=True)
    err.print(f"{len(chosen)} policy/policies over {turns} turns -> {root}")

    results = run_matrix(root, turns=turns, policies=chosen, summarize=summarize)

    if as_json:
        out.print_json(data={
            "turns": turns, "live": live,
            "policies": [
                {
                    "name": r.name, "survives": r.survives, "recoverable": r.recoverable,
                    "total": r.total, "survival_rate": round(r.survival_rate, 3),
                    "recovery_rate": round(r.recovery_rate, 3),
                    "view_chars": r.view_chars, "compactions": r.compactions,
                    "missing": r.missing,
                }
                for r in results
            ],
        })
        return

    table = Table(box=box.SIMPLE, pad_edge=False)
    table.add_column("policy")
    table.add_column("in the view", justify="right")
    table.add_column("recoverable", justify="right")
    table.add_column("view chars", justify="right")
    table.add_column("rounds", justify="right")
    for r in results:
        # The number that matters is the first: what the model will actually
        # read without being told to go looking.
        style = "ok" if r.survival_rate >= 0.75 else "warn" if r.survival_rate else "bad"
        table.add_row(
            r.name,
            f"[{style}]{r.survives}/{r.total}[/]",
            f"{r.recoverable}/{r.total}",
            f"{r.view_chars:,}",
            str(r.compactions),
        )
    out.print(table)
    out.print(
        "'in the view' is what a prompt is built from; 'recoverable' is what "
        "recall_memory could still find. The gap between them is text the "
        "model will not see unless it thinks to go looking."
    )
