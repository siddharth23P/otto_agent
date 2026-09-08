from typing import Annotated
from rich.tree import Tree
from rich.panel import Panel
from rich.table import Table
from rich import box
import typer

from agent.cli.ui import err, out
from agent.router.mapping import TASK_ROUTES, Task
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
    if d.fell_back:
        t.add_row("state", f"[warn]degraded — candidate {d.index}, {len(d.skipped)} skipped[/]")
    return Panel(t, box=box.ROUNDED, border_style="muted", expand=False)
