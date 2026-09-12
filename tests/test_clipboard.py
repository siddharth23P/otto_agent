"""agent/cli/clipboard.py: what a copied selection contains and where it goes."""
from __future__ import annotations

import subprocess

from rich.console import Group
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from agent.cli import clipboard


DOCTOR_TABLE = (
    "  provider    status   models   detail                        \n"
    " ──────────────────────────────────────────────────────────── \n"
    "  inception   ok            3                                  \n"
    "╭──────────────────────────╮\n"
    "│ required   inception  ok │\n"
    "│ also       none          │\n"
    "╰──────────────────────────╯\n"
)


def test_clean_drops_rules_and_edge_bars_but_keeps_the_words():
    cleaned = clipboard.clean(DOCTOR_TABLE)
    assert "─" not in cleaned and "│" not in cleaned and "╭" not in cleaned
    assert "provider    status   models   detail" in cleaned
    assert "required   inception  ok" in cleaned
    assert cleaned.splitlines()[-1] == "also       none"


def test_clean_turns_a_column_bar_into_spaces_and_trims_blank_edges():
    assert clipboard.clean("\n\n a │ b │ c \n\n") == " a  b  c", "indentation kept: it matters in code"
    assert clipboard.clean("") == ""
    assert clipboard.clean("plain text\nstays") == "plain text\nstays"


class FakeApp:
    def __init__(self, fail=False):
        self.copied = []
        self.fail = fail

    def copy_to_clipboard(self, text):
        if self.fail:
            raise RuntimeError("no driver")
        self.copied.append(text)


def test_copy_uses_osc52_and_the_native_command(monkeypatch):
    calls = []
    monkeypatch.setattr(clipboard, "native_command", lambda: ["/usr/bin/pbcopy"])
    monkeypatch.setattr(clipboard.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw["input"])))
    app = FakeApp()
    how = clipboard.copy(app, "hello")
    assert app.copied == ["hello"]
    assert calls == [(["/usr/bin/pbcopy"], b"hello")]
    assert how == "via OSC 52 + pbcopy"


def test_copy_says_so_when_only_osc52_is_available(monkeypatch):
    monkeypatch.setattr(clipboard, "native_command", lambda: None)
    how = clipboard.copy(FakeApp(), "hello")
    assert how.startswith("via OSC 52") and "Terminal.app" in how


def test_copy_survives_a_failing_native_command(monkeypatch):
    def boom(cmd, **kw):
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(clipboard, "native_command", lambda: ["/usr/bin/xclip", "-selection", "clipboard"])
    monkeypatch.setattr(clipboard.subprocess, "run", boom)
    assert clipboard.copy(FakeApp(), "x").startswith("via OSC 52")
    monkeypatch.setattr(clipboard, "native_command", lambda: None)
    assert clipboard.copy(FakeApp(fail=True), "x") == "no clipboard route worked"


def test_plain_text_of_renders_rich_content_without_markup():
    assert clipboard.plain_text_of("[bold]you[/] hi", 40) == "you hi"
    assert clipboard.plain_text_of(Text("plain"), 40) == "plain"
    group = Group(Text("● final"), Markdown("The **answer** is `42`."))
    rendered = clipboard.plain_text_of(group, 60)
    assert "● final" in rendered and "answer" in rendered and "**" not in rendered
    t = Table.grid()
    t.add_column(); t.add_row("a cell")
    assert "a cell" in clipboard.plain_text_of(t, 20)
