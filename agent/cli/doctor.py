import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from agent.cli.ui import err, out
from agent.router.llm_provider import health_report
from agent.router.llm_provider.base import HealthReport, ProviderStatus
from agent.router.router import Router

STYLE = {ProviderStatus.OK: "ok", ProviderStatus.NO_KEY: "muted",
         ProviderStatus.AUTH_FAILED: "bad",
         ProviderStatus.UNREACHABLE: "warn", ProviderStatus.ERROR: "bad"}


def health_table(reports: list[HealthReport]) -> Table:
    t = Table(box=box.SIMPLE, header_style="muted")
    t.add_column("provider")
    t.add_column("status")
    t.add_column("models", justify="right")
    t.add_column("detail", style="muted", overflow="fold")
    for r in reports:
        t.add_row(
            r.provider,
            f"[{STYLE[r.status]}]{r.status.value}[/]",
            str(r.model_count or ""),
            r.detail
        )
    return t


def router_view(router: Router) -> Panel:
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="muted")
    t.add_column()
    state = "[ok]configured[/]" if router.ready() else "[bad]missing -- set INCEPTION_API_KEY (otto tui -> Setup)[/]"
    t.add_row("required", f"{router.REQUIRED}  {state}")
    # Every configured provider is usable now -- there is no single "secondary"
    # seat any more (agent/router/router.py's own docstring for why).
    optional = tuple(p for p in router.usable() if p != router.REQUIRED)
    t.add_row("also configured", ", ".join(optional) if optional else "[muted]none[/]")
    return Panel(t, box=box.ROUNDED, border_style="muted", expand=False)


def doctor(ctx: typer.Context) -> None:
    """Check every provider with a real call, and show the routing policy."""

    with err.status("contacting providers"):
        reports = health_report()

    out.print(health_table(reports))

    # The table above is the diagnosis; the panel below is the conclusion.
    # A Router constructs without Inception now (agent/router/router.py), so
    # the missing-required case is a row in that panel and a non-zero exit,
    # not an exception that replaces the finding the table built toward.
    router = ctx.obj.router
    out.print(router_view(router))
    if not router.ready():
        raise typer.Exit(2)
