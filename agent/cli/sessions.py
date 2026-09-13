"""`otto sessions` -- list, delete, rename and prune saved sessions.

The index itself is agent/memory/sessions.py; this is the table both the
REPL's `/sessions` and this command print, plus the flags that change the
index from outside a session. Resuming is `otto chat --resume` / `otto tui
--resume` (and `/resume` inside either), since resuming means opening one.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.memory import sessions as session_index
from agent.memory.sessions import SessionInfo


def sessions_table(rows: list[SessionInfo], current: str | None = None) -> Table:
    """Primitive Rich styles only, like agent/cli/lessons.py's table, so it
    renders the same in the REPL and inside a Textual Static."""
    table = Table(box=box.SIMPLE, show_edge=False, pad_edge=False)
    table.add_column("id", style="bold", no_wrap=True)
    table.add_column("title", overflow="fold", ratio=1)
    table.add_column("turns", justify="right")
    table.add_column("workspace", style="dim", overflow="fold")
    table.add_column("last active", style="dim", no_wrap=True)
    for r in rows:
        mark = " ◂" if r.id == current else ""
        workspace = r.workspace.rsplit("/", 1)[-1] if r.workspace else "off"
        table.add_row(r.short_id + mark, r.label, str(r.turns), workspace,
                      session_index.describe_age(r.last_active_at))
    return table


def default_export_path(info: SessionInfo, directory: Path | None = None) -> Path:
    """`otto-session-<id>-<date>.json`, in `directory` (the current one by
    default; the TUI proposes the home directory) -- the shape agent/cli/
    lessons.py's export uses, with the id so two exports do not collide."""
    return (directory or Path.cwd()) / f"otto-session-{info.short_id}-{date.today():%Y-%m-%d}.json"


def sessions_cmd(
    delete: Annotated[
        Optional[str],
        typer.Option("--delete", help="Forget this session (id, prefix, or last): its row and its memory file."),
    ] = None,
    rename: Annotated[
        Optional[str],
        typer.Option("--rename", help="Session to retitle (id, prefix, or last); needs --title."),
    ] = None,
    title: Annotated[Optional[str], typer.Option("--title", help="The new title, with --rename.")] = None,
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Delete memory files no session owns and nothing was written to, "
                                     "and rows whose file is gone."),
    ] = False,
    export: Annotated[
        Optional[str],
        typer.Option("--export", help="Session to write out as JSON (id, prefix, or last); --to says where."),
    ] = None,
    to: Annotated[Optional[Path], typer.Option("--to", help="With --export: the file to write.")] = None,
    import_from: Annotated[
        Optional[Path],
        typer.Option("--import", help="Read a session export into this machine's sessions."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="How many to list.")] = 30,
) -> None:
    if export is not None:
        info = _resolve_or_exit(export)
        written = session_index.export_session(info.id, to or default_export_path(info))
        # soft_wrap: a path someone will copy must not be broken across lines
        # by the console width -- Rich would split it mid-name in a narrow
        # terminal, and did in CI.
        err.print(f"exported {info.short_id} · {info.label} to {written}", soft_wrap=True)
        return
    if import_from is not None:
        try:
            info = session_index.import_session(import_from)
        except ValueError as exc:
            err.print(str(exc))
            raise typer.Exit(1)
        err.print(f"imported {info.short_id} · {info.label} · {info.turns} turn(s); "
                  f"otto chat --resume {info.short_id}")
        return
    if delete is not None:
        info = _resolve_or_exit(delete)
        try:
            session_index.delete(info.id)
        except OSError as exc:
            err.print(str(exc))
            raise typer.Exit(1)
        err.print(f"deleted {info.short_id} · {info.label}")
        return
    if rename is not None:
        if not title:
            err.print("--rename needs --title")
            raise typer.Exit(2)
        info = _resolve_or_exit(rename)
        session_index.rename(info.id, title)
        err.print(f"{info.short_id} is now called {title!r}")
        return
    if prune:
        report = session_index.prune()
        err.print(report.summary())
        return
    rows = session_index.list_sessions(limit=limit)
    if not rows:
        out.print("no saved sessions yet -- a session is saved once a turn finishes")
        return
    out.print(sessions_table(rows))
    out.print("otto chat --resume <id>  ·  otto tui --resume <id>  ·  last is the newest")


def _resolve_or_exit(ref: str) -> SessionInfo:
    try:
        return session_index.resolve(ref)
    except LookupError as exc:
        err.print(str(exc))
        raise typer.Exit(1)
