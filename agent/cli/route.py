from typing import Annotated
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
from rich import box
import typer

from agent.cli.ui import err, out
from agent.router.mapping import TASK_ROUTES, Task
from agent.router.outcomes import MIN_SAMPLES, records
from agent.router.router import NoViableRoute, render

def route(ctx: typer.Context,
        task: Annotated[Task, typer.Argument(help="Task to resolve.")]
        ) -> None:
    """Show how a task resolves, and what it skipped to get there."""
    try:
        d = ctx.obj.router.resolve(task)
    except NoViableRoute as exc:
        out.print(_chain(task, exc.skipped, None))
        err.print("[bad]no viable model[/]")
        raise typer.Exit(1)
    out.print(_chain(task, d.skipped, d))
    out.print(_summary(d))
    _print_evidence(task)


def _print_evidence(task: Task) -> None:
    """What this installation has observed each candidate achieve.

    Printed under the chain because it is the reason the chain may not be in
    the order mapping.py declares it. A route whose order came from evidence
    and does not say so is a route nobody can debug.
    """
    seen = records(task.value)
    if not seen:
        return
    t = Table(box=box.SIMPLE, pad_edge=False)
    t.add_column("model"); t.add_column("runs", justify="right")
    t.add_column("approved", justify="right"); t.add_column("calls/run", justify="right")
    for r in seen:
        mark = "" if r.trusted else f" [muted](under {MIN_SAMPLES}, not used)[/]"
        t.add_row(r.model_id + mark, str(r.runs),
                  f"{r.approval_rate:.0%}", f"{r.calls_per_run:.1f}")
    out.print(t)

def _chain(task, skipped, d) -> Tree:
    tree = Tree(f"[bold]{task.value}[/]")
    chosen = d.index if d else None
    for i, c in enumerate(TASK_ROUTES[task]):
        if chosen is not None and i == chosen:
            tree.add(f"[ok]✓[/] [chosen][{i}] {d.provider}:{d.model.id}[/]")
        elif i < len(skipped):
            sk = skipped[i]
            tree.add(f"[bad]✗[/] [muted][{i}] {sk.target} — {sk.reason}[/]")
        else:
            tree.add(f"[muted]·  [{i}] {render(c)} (not reached)[/]")
    return tree

def _summary(d) -> Panel:
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="muted")
    t.add_column()
    t.add_row("model", f"[spec]{d.provider}:{d.model.id}[/]")
    t.add_row("endpoint", d.endpoint.value)
    t.add_row("context", f"{d.model.context_window:,}" if d.model.context_window else "—")
    t.add_row("params", ", ".join(f"{k}={v}" for k, v in d.params.items()) or "—")
    if d.chosen_on_evidence:
        t.add_row("state", f"[ok]chosen on observed results[/] — candidate {d.index}")
    elif d.fell_back:
        t.add_row("state", f"[warn]degraded — candidate {d.index}, {len(d.skipped)} skipped[/]")
    return Panel(t, box=box.ROUNDED, border_style="muted", expand=False)
