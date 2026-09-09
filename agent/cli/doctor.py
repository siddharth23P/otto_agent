import typer
from rich.table import Table
from rich.panel import Panel
from rich import box

from agent.cli.ui import err, out
from agent.router.llm_provider import health_report
from agent.router.llm_provider.base import AuthError, HealthReport, ProviderStatus
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
    t.add_row("required", router.REQUIRED)
    t.add_row("secondary", router.secondary or "[muted]none[/]")
    if router.ignored:
        t.add_row("ignored", f"{', '.join(router.ignored)} "
                   f"[muted](configured, but not the selected secondary)[/]")
    return Panel(t, box=box.ROUNDED, border_style="muted", expand=False)


def doctor(ctx: typer.Context) -> None:
    """Check every provider with a real call, and show the routing policy."""

    with err.status("contacting providers"):
        reports = health_report()

    out.print(health_table(reports))

    # The table above is the diagnosis. If the required provider is missing,
    # say so as the report's conclusion -- do not let the exception from
    # `ctx.obj.router` replace the very finding the table was building toward.
    try:
        router = ctx.obj.router
    except AuthError as exc:
        out.print(f"[bad]router unavailable[/] {exc}")
        raise typer.Exit(2) from None

    out.print(router_view(router))
