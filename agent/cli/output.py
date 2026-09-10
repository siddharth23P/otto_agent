"""Where a turn's final answer lands on disk, not just on screen.

Copying a multi-line Rich panel out of a live terminal mangles box-drawing
borders and wrapped lines -- see 13.4's own bug hunt, where a "garbled" paste
turned out to be a terminal copy artifact, not corrupted model output.
Writing the same text to a plain file sidesteps that entirely: chat.py and
tui.py both call this from the same place they already hold the final
CodeTask, right after rendering the panel.
"""

from __future__ import annotations

from pathlib import Path

#: Relative to wherever `otto chat`/`otto tui` is launched from -- next to
#: the project it's working on, not buried in ~/.otto with the ledger/profile
#: (10.1/10.2), since this is meant to be opened by a person, not read back
#: by Otto itself.
OUTPUT_DIR = Path.cwd() / "otto_output"

#: classify()'s LANGUAGE line is free text in plain English ("python",
#: "javascript"), not an enum -- mapped here to the extension a person would
#: actually expect to open it with. Deliberately not exhaustive: an
#: unrecognised language falls back to itself, sanitised, as the extension
#: (see `_extension`) rather than silently becoming .txt just because this
#: table hasn't met it yet.
_EXTENSIONS: dict[str, str] = {
    "python": "py", "javascript": "js", "typescript": "ts",
    "bash": "sh", "shell": "sh", "sh": "sh",
    "sql": "sql", "java": "java", "c": "c", "c++": "cpp", "cpp": "cpp",
    "go": "go", "rust": "rs", "ruby": "rb", "php": "php",
    "html": "html", "css": "css", "json": "json",
    "yaml": "yaml", "yml": "yaml", "markdown": "md",
}


def _extension(language: str | None) -> str:
    if not language:
        return "md"  # no code language: a chat reply, a story, a summary
    key = language.strip().lower()
    if key in _EXTENSIONS:
        return _EXTENSIONS[key]
    safe = "".join(c for c in key if c.isalnum())
    return safe or "txt"


def save_final(session_id: str, turn: int, text: str, language: str | None) -> Path:
    """Write one turn's final answer to its own file under OUTPUT_DIR,
    creating the directory on first use. Returns the path so the caller can
    tell the person where to find it."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{session_id}-{turn:03d}.{_extension(language)}"
    path.write_text(text)
    return path
