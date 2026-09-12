"""Copying text out of the TUI (2026-09-13, design call: "one can copy
anything from the UI except borders").

Textual already lets a person drag-select across widgets and press ctrl+c;
what it hands back and where it puts it are the two halves this module owns.

  * `clean()` -- the selection as a person means it. CSS borders are drawn by
    the compositor and never appear in selected text, but a Rich table's own
    rules and column bars do. Lines made only of box-drawing are dropped and
    edge bars are trimmed, so a selection over the doctor table comes out as
    the words in it.
  * `copy()` -- two routes at once. OSC 52 (what `App.copy_to_clipboard`
    writes) works in most terminals and not in macOS Terminal.app; a native
    command (`pbcopy`, `wl-copy`, `xclip`, `xsel`) works wherever one is
    installed. Both are tried, and the return value says which landed, so the
    toast can be honest about it.
"""
from __future__ import annotations

import re
import shutil
import subprocess

__all__ = ["clean", "copy", "native_command", "plain_text_of"]


def plain_text_of(content, width: int) -> str:
    """What a Static is showing, as the plain text a person would read off
    the screen -- rendered at the widget's width so line breaks match what a
    selection's coordinates point at. Markup strings and Text are taken as
    they are; a Rich renderable (Markdown, Table, Group) is rendered."""
    import io

    from rich.console import Console
    from rich.text import Text

    if isinstance(content, str):
        return Text.from_markup(content).plain
    if isinstance(content, Text):
        return content.plain
    buf = io.StringIO()
    Console(file=buf, width=max(8, width), force_terminal=False, no_color=True,
            highlight=False, legacy_windows=False).print(content, end="")
    return buf.getvalue()

_BOX = "─━│┃┄┅┆┇┈┉┊┋┌┍┎┏┐┑┒┓└┕┖┗┘┙┚┛├┝┞┟┠┡┢┣┤┥┦┧┨┩┪┫┬┭┮┯┰┱┲┳┴┵┶┷┸┹┺┻┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋╌╍╎╏═║╒╓╔╕╖╗╘╙╚╛╜╝╞╟╠╡╢╣╤╥╦╧╨╩╪╫╬╭╮╯╰╱╲╳╴╵╶╷╸╹╺╻╼╽╾╿"
_ONLY_BOX = re.compile(rf"^[\s{re.escape(_BOX)}]*$")
_EDGE_BARS = re.compile(rf"^[{re.escape(_BOX)}]\s?|\s?[{re.escape(_BOX)}]$")
_INNER_BAR = re.compile(r"\s[│┃║]\s")


def clean(text: str) -> str:
    """Selected text without any box-drawing: rule-only lines go, edge bars
    go, and a column bar between cells becomes two spaces."""
    kept: list[str] = []
    for line in (text or "").splitlines():
        if _ONLY_BOX.match(line):
            continue
        line = _EDGE_BARS.sub("", line)
        line = _INNER_BAR.sub("  ", line)
        kept.append(line.rstrip())
    while kept and not kept[0].strip():
        kept.pop(0)
    while kept and not kept[-1].strip():
        kept.pop()
    return "\n".join(kept)


_NATIVE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pbcopy", ()),
    ("wl-copy", ()),
    ("xclip", ("-selection", "clipboard")),
    ("xsel", ("--clipboard", "--input")),
)


def native_command() -> list[str] | None:
    """The first clipboard command installed here, or None."""
    for name, args in _NATIVE:
        path = shutil.which(name)
        if path:
            return [path, *args]
    return None


def copy(app, text: str) -> str:
    """Put `text` on the clipboard by every route available. Returns a short
    description of what worked, for the toast."""
    routes: list[str] = []
    try:
        app.copy_to_clipboard(text)
        routes.append("OSC 52")
    except Exception:
        pass
    command = native_command()
    if command:
        try:
            subprocess.run(command, input=text.encode("utf-8"), check=True, timeout=3,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            routes.append(command[0].rsplit("/", 1)[-1])
        except (OSError, subprocess.SubprocessError):
            pass
    if not routes:
        return "no clipboard route worked"
    note = "" if len(routes) > 1 or routes != ["OSC 52"] else " (macOS Terminal.app ignores this; iTerm, Ghostty, kitty, WezTerm honour it)"
    return "via " + " + ".join(routes) + note
