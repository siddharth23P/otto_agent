"""`otto lessons` rendering, shared with the TUI's "Show lessons"."""
from __future__ import annotations

import io

from rich.console import Console

from agent.cli.lessons import lessons_table
from agent.memory.lessons import Lesson


def test_lessons_table_shows_both_outcomes_and_the_count(tmp_path):
    table = lessons_table([Lesson("tests fail on import", "check the venv"),
                           Lesson("the file is huge", "read it in slices", "failed")], tmp_path / "l.db")
    buf = io.StringIO()
    Console(file=buf, width=100, force_terminal=False).print(table)
    text = buf.getvalue()
    assert "2 lesson(s)" in text and "l.db" in text
    assert "worked" in text and "failed" in text
    assert "check the venv" in text and "the file is huge" in text
