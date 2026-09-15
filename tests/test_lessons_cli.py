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


def test_phone_lessons_are_listed_only_with_phone(tmp_path, monkeypatch):
    """`otto lessons --phone` reads the lessons phone runs keep apart."""
    import agent.cli.lessons as cli
    from agent.memory import lessons as L
    from agent.memory.store import MemoryStore

    def no_embeddings(texts):
        raise L.EmbeddingUnavailable("off in this test")

    monkeypatch.setattr(L, "embed", no_embeddings)
    path = tmp_path / "lessons.db"
    store = MemoryStore(path)
    with L.bind_bank(store):
        L.record_lessons([Lesson("a test fails on import", "check the venv")])
    store.close()

    def listed(**kw) -> str:
        buf = io.StringIO()
        monkeypatch.setattr(cli, "out", Console(file=buf, width=120, force_terminal=False))
        cli.lessons_cmd(bank=path, **kw)
        return buf.getvalue()

    assert "has no phone lessons yet" in listed(phone=True)
    store = MemoryStore(path)
    with L.bind_bank(store), L.bind_kind(L.PHONE_KIND):
        L.record_lessons([Lesson("a results list is unsorted", "open sort")])
    store.close()
    phone = listed(phone=True)
    assert "open sort" in phone and "check the venv" not in phone
    workspace = listed()
    assert "check the venv" in workspace and "open sort" not in workspace
