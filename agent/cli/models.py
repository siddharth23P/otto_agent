import json as json_lib
from typing import Annotated, Optional

import typer
from rich import box
from rich.table import Table

from agent.cli.ui import err, out
from agent.router.llm_provider import all_models, provider_names
from agent.router.llm_provider.base import Capability, ModelInfo


def models_table(found: list[ModelInfo]) -> Table:
    t = Table(box=box.SIMPLE, header_style="muted")
    t.add_column("provider", style="muted")
    t.add_column("model", style="spec")
    t.add_column("context", justify="right")
    t.add_column("max out", justify="right")
    t.add_column("capabilities", style="muted")
    for m in found:
        t.add_row(
            m.provider, m.id,
            f"{m.context_window:,}" if m.context_window else "—",
            f"{m.max_output_tokens:,}" if m.max_output_tokens else "—",
            " ".join(sorted(c.value for c in m.capabilities)),
        )
    return t


def models(
    capability: Annotated[Optional[Capability], typer.Option(help="Only models with this capability.")] = None,
    provider: Annotated[Optional[str], typer.Option(help="Only this vendor.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
) -> None:
    """Every model across every configured provider."""
    
    if provider is not None and provider not in provider_names():
        raise typer.BadParameter(
            f"unknown provider {provider!r}; known: {', '.join(provider_names())}",
            param_hint="--provider",
        )
        
    with err.status("fetching catalogues…"):
        found = all_models(capability)
        
    if provider:
        found = [m for m in found if m.provider == provider]
    found.sort(key=lambda m: (m.provider, m.id))
    
    if as_json:
        typer.echo(json_lib.dumps([
            {"provider": m.provider, "id": m.id,
             "context_window": m.context_window,
             "max_output_tokens": m.max_output_tokens,
             "capabilities": sorted(c.value for c in m.capabilities)}
            for m in found
        ], indent=2))
        return
    
    if not found:
        out.print("[muted]no models matched[/]")
        return
    
    out.print(models_table(found))
