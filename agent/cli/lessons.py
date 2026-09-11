"""`otto lessons` -- read, and if need be empty, what Otto has learned.

A self-improving system whose learned state cannot be read is not a system
anyone should trust, and this is the cheapest possible remedy: the bank is a
few short lines of text, so print them.

There is a second reason it exists. Evolution that is not working is a real
and documented outcome -- across five methods and three frontier models the
measured gains were around +1%, and negative in every regime for the
strongest model. Being able to look at the bank and say "these lessons are
rubbish" is how that gets noticed, and `--clear` is how it gets undone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.memory.lessons import KIND, all_lessons, bank_path, bind_bank
from agent.memory.store import MemoryStore


def lessons_cmd(
    bank: Annotated[
        Optional[Path],
        typer.Option(help="Which bank to read (default: ~/.otto/memory/lessons.db)."),
    ] = None,
    clear: Annotated[
        bool,
        typer.Option("--clear", help="Delete every lesson. Not undoable."),
    ] = False,
) -> None:
    path = bank or bank_path()
    if not path.exists():
        out.print(f"no lesson bank at {path} -- nothing learned yet")
        return

    store = MemoryStore(path)
    with bind_bank(store):
        if clear:
            # Straight to the table: these are chunks like any other, and the
            # store has no delete because nothing else in Otto ever needed one.
            store._conn.execute("DELETE FROM chunks WHERE kind = ?", (KIND,))
            store._conn.commit()
            err.print(f"cleared {path}")
            return

        learned = all_lessons()
        if not learned:
            out.print(f"{path} is empty")
            return
        out.print(f"{len(learned)} lesson(s) in {path}\n")
        for lesson in learned:
            marker = "[ok]worked[/]" if lesson.outcome == "worked" else "[warn]failed[/]"
            out.print(f"  {marker}  when {lesson.cue}")
            out.print(f"          {lesson.action}")
