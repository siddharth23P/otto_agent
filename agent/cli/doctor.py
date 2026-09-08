import typer
from rich.table import Table
from rich import box

from agent.cli.ui import err, out
from agent.router.llm_provider import health_report
from agent.router.llm_provider.base import AuthError, ProviderStatus

STYLE = {ProviderStatus.OK: "ok", ProviderStatus.NO_KEY: "muted",
         ProviderStatus.AUTH_FAILED: "bad",
         ProviderStatus.UNREACHABLE: "warn", ProviderStatus.ERROR: "bad"}

def doctor(ctx: typer.Context) -> None:
    """Check every provider with a real call, and show the routing policy."""

    with err.status("contacting providers"):
        reports = health_report()
        
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
    out.print(t)
    
    # The table above is the diagnosis. If the required provider is missing,
    # say so as the report's conclusion -- do not let the exception from
    # `ctx.obj.router` replace the very finding the table was building toward.
    try:
        router = ctx.obj.router
    except AuthError as exc:
        out.print(f"[bad]router unavailable[/] {exc}")
        raise typer.Exit(2) from None

    out.print(f"[muted]required[/]  {router.REQUIRED}")
    out.print(f"[muted]secondary[/] {router.secondary or '[muted]none[/]'}")
    if router.ignored:
        out.print(f"[warn]ignored[/]   {', '.join(router.ignored)} "
                  f"[muted](configured, but not the selected secondary)[/]")