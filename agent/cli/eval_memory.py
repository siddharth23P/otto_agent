"""`otto eval-memory`: run agent/eval/memory_bench.py's LoCoMo-based memory
benchmark and report coverage. See that module's own docstring for what's
actually measured (store/visible-verbatim/recall/answerable coverage per QA
category, plus a compression ratio) and why the two `--x-budget`/`--y-budget`
options exist: real LoCoMo conversations are small enough that Otto's real
production budget (agent/memory/queue.py's X_BUDGET/Y_BUDGET) never triggers
compaction at all, so a run at the defaults mostly reports "everything is
still verbatim, nothing needed compacting" -- correct, but it doesn't
exercise the compaction+recall code path. Passing smaller values here forces
that path to actually run, at the cost of no longer matching Otto's real
per-turn budget.

No Langfuse tracking here (unlike `otto eval`, agent/cli/eval.py) -- this
benchmark doesn't run the pipeline or the router at all in its default,
offline mode (`--live` opts into one real LLM call per compaction, via
agent.memory.wiring.summarize_for_memory, agent/eval/memory_bench.py's own
module docstring), so there's no per-item trace worth syncing there.

`--show-items failures|all` prints each scored QA item's evidence text next
to recall()'s raw output -- worth reaching for whenever a coverage number
looks suspicious, since `recalled` is an exact-substring match (module
docstring point 3), not a fuzzy one: reading the two side by side is how
you tell a real semantic-search find from a coincidental match. It also
surfaces a real gap in the OFFLINE (`--no-live`, default) summarizer:
_canned_summarize groups raw turns into fixed-size blocks with placeholder
bullet text, and recall() returns a matched bullet's ENTIRE underlying raw
text -- so offline, "recalled" mostly tests "did the right block rank in
top-k," not "did it find the right sentence." `--live`'s real, topic-
scoped bullet summaries are the stricter version of this same test.
"""
from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich import box
from rich.table import Table

from agent.cli.ui import err, out
from agent.eval.memory_bench import BenchmarkReport, DEFAULT_CACHE, download_locomo, load_locomo, run_benchmark


def eval_memory_cmd(
    live: Annotated[
        bool,
        typer.Option(
            "--live/--no-live",
            help="Use the real Task.SUMMARIZE model + local embeddings (needs INCEPTION_API_KEY) "
                 "instead of the deterministic offline canned summarizer.",
        ),
    ] = False,
    samples: Annotated[
        Optional[int], typer.Option(help="Only run the first N of LoCoMo's 10 conversations.")
    ] = None,
    max_turns: Annotated[
        Optional[int], typer.Option(help="Truncate each conversation to its first N turns (fast smoke run).")
    ] = None,
    x_budget: Annotated[
        Optional[int],
        typer.Option(help="Override TieredQueue's X budget (tokens) -- smaller forces real compaction."),
    ] = None,
    y_budget: Annotated[
        Optional[int],
        typer.Option(help="Override TieredQueue's Y budget (tokens) -- smaller forces real compaction."),
    ] = None,
    top_k: Annotated[int, typer.Option(help="How many bullets recall() considers per question.")] = 5,
    data_path: Annotated[
        Optional[Path], typer.Option(help="Path to a cached locomo10.json (default: agent/eval/data/locomo10.json).")
    ] = None,
    force_download: Annotated[
        bool, typer.Option("--force-download", help="Re-download locomo10.json even if a cached copy exists.")
    ] = False,
    json: Annotated[
        bool, typer.Option("--json", help="Print the full report as JSON instead of a table.")
    ] = False,
    show_items: Annotated[
        str,
        typer.Option(
            help="Print per-question detail after the table: 'none' (default), 'failures' "
                 "(answerable=False items only -- the interesting/debug case), or 'all'. Each "
                 "item shows its evidence text and recall()'s raw output side by side, since "
                 "`recalled` is an exact-substring match, not a fuzzy one -- useful for checking "
                 "whether a pass/miss is real or an artifact of the conversation repeating similar "
                 "phrasing. Ignored when --json is set (the JSON report already includes both).",
        ),
    ] = "none",
) -> None:
    """Score agent/memory/'s tiered queue against the LoCoMo long-conversation-memory dataset."""
    if show_items not in ("none", "failures", "all"):
        err.print("[bad]--show-items must be 'none', 'failures', or 'all'[/]")
        raise typer.Exit(2)

    path = data_path or DEFAULT_CACHE
    with err.status(f"fetching LoCoMo dataset ({path})…"):
        try:
            download_locomo(path, force=force_download)
        except Exception as exc:
            err.print(f"[bad]could not download the LoCoMo dataset: {exc}[/]")
            err.print("[muted]pass --data-path to point at an already-downloaded locomo10.json[/]")
            raise typer.Exit(1) from exc
        all_samples = load_locomo(path)

    if samples is not None:
        all_samples = all_samples[:samples]
    if not all_samples:
        err.print("[bad]no LoCoMo samples to run[/]")
        raise typer.Exit(1)

    with err.status(f"running {len(all_samples)} conversation(s), live={live}…"):
        report = run_benchmark(
            all_samples, live=live, top_k=top_k, max_turns=max_turns,
            x_budget=x_budget, y_budget=y_budget,
        )

    if json:
        out.print(json_module.dumps(report.to_dict(), indent=2))
        return

    summary = report.summary()
    t = Table(box=box.SIMPLE, header_style="muted")
    t.add_column("category", style="spec")
    t.add_column("n", justify="right")
    t.add_column("store", justify="right")
    t.add_column("verbatim", justify="right")
    t.add_column("recalled", justify="right")
    t.add_column("answerable", justify="right")

    def _pct(value: Optional[float]) -> str:
        return "-" if value is None else f"{value * 100:.0f}%"

    for name, rates in summary["by_category"].items():
        t.add_row(
            name, str(rates["n"]), _pct(rates["store_coverage"]),
            _pct(rates["visible_verbatim_coverage"]), _pct(rates["recall_coverage"]),
            _pct(rates["answerable_coverage"]),
        )
    overall = summary["overall"]
    t.add_row(
        "[chosen]overall[/]", str(overall["n"]), _pct(overall["store_coverage"]),
        _pct(overall["visible_verbatim_coverage"]), _pct(overall["recall_coverage"]),
        _pct(overall["answerable_coverage"]),
    )
    out.print(t)

    ratio = summary["overall_compression_ratio"]
    ratio_text = "-" if ratio is None else f"{ratio:.3f}"
    out.print(
        f"[muted]{summary['conversation_count']} conversation(s) -- "
        f"final_view/raw token ratio: {ratio_text}[/]"
    )

    if show_items != "none":
        _print_items(report, show_items)

    if overall["n"] and overall["store_coverage"] is not None and overall["store_coverage"] < 1.0:
        err.print("[bad]store_coverage < 100% -- the engine lost cited evidence; this is a real bug[/]")
        raise typer.Exit(1)


def _snippet(text: str, limit: int = 280) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "…"


def _print_items(report: BenchmarkReport, which: str) -> None:
    """Per-question detail: for every scored QA item matching `which`, the
    exact evidence text next to recall()'s raw output -- the pair
    `recalled` was computed from (an exact substring check, module
    docstring point 3), so reading them side by side is how you tell a
    real find from a coincidental match in a conversation that repeats
    similar phrasing often.
    """
    out.print("")
    shown = 0
    for conv in report.conversations:
        for r in conv.scored_results():
            if which == "failures" and r.answerable:
                continue
            shown += 1
            verdict = "[ok]answerable[/]" if r.answerable else "[bad]MISS[/]"
            via = "verbatim" if r.visible_verbatim else ("recalled" if r.recalled else "neither")
            out.print(f"[spec]{conv.sample_id}[/] [muted]({via})[/] {verdict}")
            out.print(f"  Q: {r.question}")
            out.print(f"  evidence: {_snippet(r.evidence_text)}")
            out.print(f"  recalled: {_snippet(r.recalled_text)}")
            out.print("")
    if shown == 0:
        out.print("[muted](nothing matched --show-items filter)[/]")
