"""`otto lessons` -- read, move, and if need be empty, what Otto has learned.

A self-improving system whose learned state cannot be read is not a system
anyone should trust, and this is the cheapest possible remedy: the bank is a
few short lines of text, so print them.

There is a second reason it exists. Evolution that is not working is a real
and documented outcome -- across five methods and three frontier models the
measured gains were around +1%, and negative in every regime for the
strongest model. Being able to look at the bank and say "these lessons are
rubbish" is how that gets noticed, and `--clear` is how it gets undone.

`--export` / `--import` (2026-09-12) move a bank between machines as JSON,
through the same duplicate adjudication a run's own lessons face. The TUI's
"Export lessons…" / "Import lessons…" palette entries call the same two
functions (agent/memory/lessons.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.memory.lessons import (
    Lesson, all_lessons, bank_path, bind_bank, clear_bank, export_lessons, import_lessons,
)
from agent.memory.store import MemoryStore


def lessons_table(lessons: list[Lesson], path: Path | None = None) -> Table:
    """The bank as a table. Primitive Rich styles only, so it renders the same
    in the REPL (whose THEME maps ok/warn to exactly these) and inside a
    Textual Static, which never sees that THEME."""
    title = f"{len(lessons)} lesson(s)" + (f" in {path}" if path else "")
    t = Table(box=box.SIMPLE, header_style="dim", title=title, title_justify="left")
    t.add_column("outcome")
    t.add_column("when")
    t.add_column("do", overflow="fold")
    for lesson in lessons:
        mark = Text("worked", style="bold green") if lesson.outcome == "worked" else Text("failed", style="yellow")
        t.add_row(mark, lesson.cue, lesson.action)
    return t


def lessons_cmd(
    bank: Annotated[
        Optional[Path],
        typer.Option(help="Which bank to read (default: ~/.otto/memory/lessons.db)."),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Delete every lesson. Not undoable."),
    ] = False,
    export: Annotated[
        Optional[Path],
        typer.Option("--export", help="Write the bank to this JSON file."),
    ] = None,
    import_from: Annotated[
        Optional[Path],
        typer.Option("--import", help="Merge lessons from this JSON file into the bank."),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option("--replace", help="With --import: empty the bank first."),
    ] = False,
) -> None:
    path = bank or bank_path()
    if import_from is None and not path.exists():
        out.print(f"no lesson bank at {path} -- nothing learned yet")
        return

    store = MemoryStore(path)
    with bind_bank(store):
        if clear:
            removed = clear_bank()
            err.print(f"cleared {removed} lesson(s) from {path}")
            return
        if export is not None:
            n = export_lessons(export)
            err.print(f"exported {n} lesson(s) to {export}")
            return
        if import_from is not None:
            report = import_lessons(import_from, replace=replace)
            err.print(f"imported from {import_from}: {report.summary()}")
            return

        learned = all_lessons()
        if not learned:
            out.print(f"{path} is empty")
            return
        out.print(lessons_table(learned, path))
