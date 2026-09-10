"""`otto eval`: run the golden dataset through the real pipeline and report
pass/fail per item. See agent/eval/runner.py for what "checked" means, and
agent/eval/golden/ for the items themselves.

By default this tracks the run as a Langfuse Dataset Run (agent/eval/
langfuse_sync.py): the golden set is synced to a Langfuse Dataset once,
then every `otto eval` invocation becomes one named, comparable run
against it in the Langfuse UI -- every item's trace, its pass/fail score,
and the run as a whole. Pass --no-experiment for the old local-only path
(no Langfuse project needed -- useful offline or in CI without Langfuse
creds), which prints a plain pass/fail table and nothing else.

No --agents option (2026-09-10): the router/planner/solver/summarizer/
finder/evaluator graph that replaced the swarm pipeline has nothing to
size -- see agent/pipeline/run.py's module docstring.
"""
from typing import Annotated, Optional

import typer
from rich import box
from rich.table import Table

from agent.cli.ui import err, out
from agent.eval.runner import run_golden


def eval_cmd(
    domain: Annotated[Optional[str], typer.Option(help="Only 'code' or 'math'.")] = None,
    experiment: Annotated[
        bool,
        typer.Option(
            "--experiment/--no-experiment",
            help="Track this run as a Langfuse Dataset Run (default) vs. a local-only table.",
        ),
    ] = True,
    run_name: Annotated[
        Optional[str], typer.Option(help="Name for the Langfuse experiment run (--experiment only).")
    ] = None,
) -> None:
    """Run the golden dataset through the pipeline and report pass/fail."""
    if domain is not None and domain not in ("code", "math"):
        err.print("[bad]--domain must be 'code' or 'math'[/]")
        raise typer.Exit(2)

    if experiment:
        _run_as_langfuse_experiment(domain=domain, run_name=run_name)
        return

    with err.status("running golden set…"):
        results = run_golden(only_domain=domain)

    if not results:
        out.print("[muted]no golden items matched[/]")
        return

    t = Table(box=box.SIMPLE, header_style="muted")
    t.add_column("id", style="spec")
    t.add_column("domain", style="muted")
    t.add_column("result")
    t.add_column("seconds", justify="right")
    t.add_column("evidence", style="muted")
    passed = 0
    for r in results:
        mark = "[ok]pass[/]" if r.passed else "[bad]fail[/]"
        passed += r.passed
        evidence = r.evidence if len(r.evidence) <= 80 else r.evidence[:80] + "…"
        t.add_row(r.item_id, r.domain, mark, f"{r.seconds:.1f}", evidence.replace("\n", " "))
    out.print(t)
    out.print(f"[muted]{passed}/{len(results)} passed[/]")

    _print_full_failures((r.item_id, r.evidence) for r in results if not r.passed)
    if passed < len(results):
        raise typer.Exit(1)


def _print_full_failures(failures) -> None:
    """The overview table truncates evidence to 80 chars for readability --
    fine for a scan, useless for actually debugging a failure (it cuts a
    checker's stderr right where the real exception would start). Print
    each failing item's full evidence separately so that's never the
    reason you have to go dig through the Langfuse UI.
    """
    failures = list(failures)
    if not failures:
        return
    out.print("\n[bad]failures, in full:[/]")
    for item_id, evidence in failures:
        out.print(f"[spec]{item_id}[/]")
        out.print(evidence)
        out.print("")


def _run_as_langfuse_experiment(*, domain: Optional[str], run_name: Optional[str]) -> None:
    from agent.eval.langfuse_sync import run_golden_experiment

    with err.status("syncing golden set to langfuse and running experiment…"):
        try:
            result = run_golden_experiment(only_domain=domain, run_name=run_name)
        except Exception as exc:
            err.print(f"[bad]langfuse experiment failed: {exc}[/]")
            err.print("[muted]is LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY set? falling back to --no-experiment still works offline.[/]")
            raise typer.Exit(1) from exc

    t = Table(box=box.SIMPLE, header_style="muted")
    t.add_column("id", style="spec")
    t.add_column("result")
    t.add_column("evidence", style="muted")
    all_passed = True
    for item_result in result.item_results:
        golden_id = getattr(item_result.item, "id", "?")
        ev = next((e for e in item_result.evaluations if e.name == "golden_pass"), None)
        item_passed = bool(ev.value) if ev is not None else False
        all_passed &= item_passed
        mark = "[ok]pass[/]" if item_passed else "[bad]fail[/]"
        comment = (ev.comment or "") if ev is not None else "no golden_pass score"
        comment = comment if len(comment) <= 80 else comment[:80] + "…"
        t.add_row(golden_id, mark, comment.replace("\n", " "))
    out.print(t)

    n = len(result.item_results)
    n_passed = sum(
        1 for r in result.item_results
        if any(e.name == "golden_pass" and e.value for e in r.evaluations)
    )
    out.print(f"[muted]{n_passed}/{n} passed[/]")
    if result.dataset_run_url:
        out.print(f"[muted]langfuse dataset run: {result.dataset_run_url}[/]")

    if not all_passed:
        raise typer.Exit(1)
