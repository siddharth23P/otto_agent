"""ASCII art and animation frame tables for the TUI -- data, not widgets.

Everything in here is a constant or a pure function, and nothing imports
Textual: a test can assert on a frame table without a terminal, and tui.py
can decide WHEN to show a frame while this module only says WHAT the frames
are. That split is also what keeps the art out of the way -- `set_interval`
callbacks in tui.py step through these tables; nothing here schedules anything.

Two rules every animation follows, written down once:

  * Motion is optional, content is not. `OTTO_NO_ANIMATION=1` (and any
    headless run, which is what the test suite is) skips straight to the
    last frame of every table. The last frame is therefore always the
    resting, complete rendering -- `reveal_frames()` ends on the wordmark,
    `SPARKLE_FRAMES` ends on the plain green "final" panel -- so a static
    terminal sees exactly what an animated one settles on.
  * No dependencies. The wordmark is drawn by hand rather than through
    pyfiglet, the spinner is ten braille characters, and the meter is two
    block glyphs. All of it fits an 80-column terminal with the 34-column
    token panel open (`MAX_WORDMARK_WIDTH` is checked by a test).
"""
from __future__ import annotations

import math
import os
import re

from rich.text import Text

#: Set to anything non-empty to stop every animation. Glyphs stay.
NO_ANIMATION_ENV = "OTTO_NO_ANIMATION"

#: The wordmark: "otto" in three rows of half-blocks. Small on purpose -- it
#: sits at the top of the sidebar and the setup screen, the way Crush places
#: its own mark, rather than filling the first seven rows of every session.
WORDMARK_SMALL: tuple[str, ...] = (
    "▄▀▀▄ ▀█▀ ▀█▀ ▄▀▀▄",
    "█  █  █   █  █  █",
    "▀▄▄▀  ▀   ▀  ▀▄▄▀",
)

WORDMARK = WORDMARK_SMALL
TAGLINE = "otto · one agent, one evaluator"

#: The wordmark is drawn in the terminal's own bold foreground, not a colour.
#: A cyan block-letter banner reads as a splash screen; bold on the default
#: background reads as a heading, which is what it is. Only the states that
#: mean something (a budget running out, a stop requested, an answer landing)
#: get a colour.
WORDMARK_STYLE = "bold"
TAGLINE_STYLE = "dim"

#: The sidebar is 32 columns with 1 of padding a side; the mark must fit.
MAX_WORDMARK_WIDTH = 28

#: The banner reveals left to right over REVEAL_STEPS frames, one every
#: REVEAL_EVERY seconds -- 0.6s in total, long enough to be seen and short
#: enough that nobody waits for it.
REVEAL_STEPS = 8
REVEAL_EVERY = 0.075

#: Braille spinner for the status line. Ten frames read as continuous motion
#: at five frames a second; a frozen one still says "stuck", which is the
#: point of having one at all.
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

#: (phase-text prefix, primitive Rich style). Matched by prefix on the phase
#: text the progress seam reports, first match wins. Primitive style words
#: only -- these land in a Static, which never sees agent/cli/ui.py's THEME.
PHASE_STYLES: tuple[tuple[str, str], ...] = (
    ("checking", "yellow"),
    ("judging", "yellow"),
    ("stopping", "bold red"),
)
DEFAULT_PHASE_STYLE = ""

#: One glyph per mode (agent/pipeline/modes.py), shown in the thinking
#: block's title when the agent switches.
MODE_GLYPHS: dict[str, str] = {
    "solve": "◆",
    "plan": "☰",
    "summarize": "≣",
    "find": "⌕",
}

#: The two shapes a board line takes when it names a mode (agent/pipeline/
#: nodes.py `_emit`): "escalated to plan mode -- reason" / "switched to find
#: mode", and the per-action prefix "solve: ...".
MODE_SWITCH = re.compile(r"^(?:escalated|switched|de-escalated) to (\w+) mode")
MODE_PREFIX = re.compile(r"^(solve|plan|summarize|find): ")

#: (glyph, style) for the bullet in front of "final" as an answer lands. The
#: LAST frame is the resting one -- the same ● every other header line uses.
SPARKLE_FRAMES: tuple[tuple[str, str], ...] = (
    ("✦", "bold green"),
    ("✧", "green"),
    ("·", "green"),
    ("●", "green"),
)
SPARKLE_EVERY = 0.12

#: The bullets a header line starts with. One vocabulary, used everywhere:
#: what the person said, what otto answered, what it is doing in between.
BULLET_ANSWER = "●"
BULLET_STREAMING = "○"
BULLET_THINKING = "◇"
BULLET_THOUGHT = "◆"
PROMPT_GLYPH = "›"

#: The thinking block's title sweeps once as it folds shut.
SWEEP_FRAMES: tuple[str, ...] = ("▸▹▹", "▹▸▹", "▹▹▸")
SWEEP_EVERY = 0.08

#: What an empty transcript shows instead of nothing. Text, no mascot: the
#: first one drawn here was a face, and a face is a taste nobody asked for.
EMPTY_STATE: tuple[str, ...] = (
    "type a message below to start.",
    "ctrl+p  commands       f2  setup       ctrl+t  hide the sidebar",
    "ctrl+y  copy answer    esc stop turn   ctrl+n  new session",
    "drag to select any text; ctrl+c copies it with no borders in it.",
)

#: Width of the per-turn budget meter in the status line.
METER_WIDTH = 12
METER_FULL = "▰"
METER_EMPTY = "▱"


def reveal_frames(lines: tuple[str, ...] | list[str], steps: int = REVEAL_STEPS) -> list[tuple[str, ...]]:
    """`steps + 1` frames that wipe `lines` in from the left.

    Frame 0 is blank, frame `steps` is `lines` exactly, and every row of every
    frame is the same width as the widest input row, so a Static showing them
    in turn never changes size.
    """
    rows = tuple(lines)
    width = max((len(r) for r in rows), default=0)
    padded = tuple(r.ljust(width) for r in rows)
    frames: list[tuple[str, ...]] = []
    for k in range(steps + 1):
        keep = math.ceil(width * k / steps) if steps else width
        frames.append(tuple(r[:keep] + " " * (width - keep) for r in padded))
    if frames:
        frames[-1] = padded
    return frames


def meter(used: int, total: int | None, *, width: int = METER_WIDTH,
          warn_at: float = 0.8) -> Text:
    """`used` of `total` as a bar: green while there is plenty, yellow as it
    fills, bold red from `warn_at` on -- the same fraction at which the run
    itself is told to wrap up (agent/pipeline/budget.py). Empty when there is
    no ceiling to measure against."""
    if not total or total <= 0:
        return Text("")
    fraction = min(max(used / total, 0.0), 1.0)
    filled = round(fraction * width)
    style = "green" if fraction < 0.6 else "yellow" if fraction < warn_at else "bold red"
    return Text(METER_FULL * filled + METER_EMPTY * (width - filled), style=style)


def phase_style(phase: str) -> str:
    lowered = (phase or "").lower()
    for prefix, style in PHASE_STYLES:
        if lowered.startswith(prefix):
            return style
    return DEFAULT_PHASE_STYLE


def mode_from_board_line(line: str) -> str | None:
    """The mode a board line announces, or None. Only names in MODE_GLYPHS."""
    text = (line or "").strip()
    match = MODE_SWITCH.match(text) or MODE_PREFIX.match(text)
    if not match:
        return None
    name = match.group(1).lower()
    return name if name in MODE_GLYPHS else None


def animations_enabled(app) -> bool:
    """Whether to step through frames or jump to the last one.

    Headless is the test suite (`App.run_test()` sets `is_headless`), and the
    env flag is the person. `animation_level` is NOT consulted: it stays
    "full" under run_test, so it cannot tell the two apart.
    """
    if os.environ.get(NO_ANIMATION_ENV, "").strip():
        return False
    return not bool(getattr(app, "is_headless", False))


def section(title: str, width: int = 28) -> Text:
    """A dim rule with a label in it -- the sidebar's section header, the way
    Crush and OpenCode label a sidebar without boxing it."""
    label = f"─ {title} "
    return Text(label + "─" * max(0, width - len(label)), style="dim")


#: Board-line rewrites: the ASCII the pipeline writes -> the glyph a person
#: reads. Applied by agent/cli/shell.py's render_update for both front ends.
_ARROW = re.compile(r"\s+->\s+")
_OUTCOME_OK = re.compile(r"\b(ok|approved|passed|done)\b$")
_OUTCOME_BAD = re.compile(r"\b(failed|error|rejected|refused|timed out)\b$")


def decorate_board_line(line: str) -> str:
    """Markup for one board line: a mode prefix becomes its glyph, `->` becomes
    an arrow, and a trailing outcome word is coloured. Returns Rich markup;
    the caller wraps it in whatever muted style it already used."""
    text = str(line or "")
    prefixed = MODE_PREFIX.match(text)
    if prefixed:
        text = f"{MODE_GLYPHS[prefixed.group(1)]} {text[prefixed.end():]}"
    switched = MODE_SWITCH.match(text)
    if switched and switched.group(1) in MODE_GLYPHS:
        text = f"{MODE_GLYPHS[switched.group(1)]} {text}"
    text = _ARROW.sub(" → ", text)
    if _OUTCOME_OK.search(text):
        text = _OUTCOME_OK.sub(lambda m: f"[green]{m.group(1)}[/]", text)
    elif _OUTCOME_BAD.search(text):
        text = _OUTCOME_BAD.sub(lambda m: f"[red]{m.group(1)}[/]", text)
    return text


# --------------------------------------------------------------------------
# Motion (2026-09-13, design call: "add some cool animations as well from
# awesometui.com"). What the field does that reads well: Mistral Vibe walks a
# braille snake while it thinks, OpenCode runs one bright dot along a row of
# dim ones, Crush shimmers its mark, and every streaming client blinks a
# caret. Each is a pure frame generator here; tui.py owns the timers.
# --------------------------------------------------------------------------

#: A braille cell is 2 dots wide and 4 dots tall; dot (x, y) within a cell
#: sets this bit. Two cells side by side give a 4x4 canvas -- the snake's map.
_BRAILLE_DOT_BITS = ((0x01, 0x08), (0x02, 0x10), (0x04, 0x20), (0x40, 0x80))
BRAILLE_BLANK = "\u2800"


def render_braille(points, width: int = 4, height: int = 4) -> str:
    """Draw `points` ((x, y) dot coordinates) on a width x height dot canvas
    as braille characters -- one character per 2x4 block."""
    cells = [[0] * ((width + 1) // 2) for _ in range((height + 3) // 4)]
    for x, y in points:
        x, y = int(x), int(y)
        if 0 <= x < width and 0 <= y < height:
            cells[y // 4][x // 2] |= _BRAILLE_DOT_BITS[y % 4][x % 2]
    return "\n".join("".join(chr(0x2800 + bits) for bits in row) for row in cells)


class Snake:
    """A three-dot snake walking a 4x4 braille canvas -- two characters wide.
    Deterministic for a given seed, so a test can assert a frame."""

    WIDTH = 4
    HEIGHT = 4
    LENGTH = 3

    def __init__(self, seed: int | None = None) -> None:
        import random as _random

        self._rng = _random.Random(seed)
        self._body: list[tuple[int, int]] = [(1, 0), (0, 0), (0, 1)]

    def _free(self, x: int, y: int) -> bool:
        return 0 <= x < self.WIDTH and 0 <= y < self.HEIGHT and (x, y) not in self._body

    def step(self) -> str:
        hx, hy = self._body[0]
        px, py = self._body[1]
        dx, dy = hx - px, hy - py
        options = [(x, y) for x, y in ((dx, dy), (-dy, dx), (dy, -dx)) if self._free(hx + x, hy + y)]
        if not options:
            options = [(x, y) for x, y in ((1, 0), (-1, 0), (0, 1), (0, -1)) if self._free(hx + x, hy + y)]
        if not options:
            self._body = [(1, 0), (0, 0), (0, 1)]
            return self.frame()
        # Keep going straight most of the time; a snake that turns every step
        # reads as noise rather than motion.
        if (dx, dy) in options and self._rng.random() < 0.7:
            mx, my = dx, dy
        else:
            mx, my = self._rng.choice(options)
        self._body = [(hx + mx, hy + my)] + self._body[: self.LENGTH - 1]
        return self.frame()

    def frame(self) -> str:
        return render_braille(self._body, self.WIDTH, self.HEIGHT)


DOT_SWEEP_WIDTH = 8


def dot_sweep(i: int, width: int = DOT_SWEEP_WIDTH) -> Text:
    """OpenCode's working indicator: a row of dim dots with one bright dot
    travelling along it and back."""
    span = max(1, width - 1)
    pos = i % (2 * span)
    if pos > span:
        pos = 2 * span - pos
    text = Text()
    for k in range(width):
        text.append("●" if k == pos else "·", style="bold" if k == pos else "dim")
    return text


CARET = "▌"
CARET_EVERY = 0.5

#: Odometer: numbers roll from the old value to the new one over this many
#: frames rather than jumping -- the sidebar's totals and the top bar's spend.
TWEEN_STEPS = 6
TWEEN_EVERY = 0.05


def tween(start: float, end: float, steps: int = TWEEN_STEPS) -> list[float]:
    """`steps` values easing out from `start` to `end`; the last is exactly
    `end`, so a display that shows the final frame shows the truth."""
    if steps <= 1 or start == end:
        return [end]
    out = []
    for k in range(1, steps + 1):
        t = k / steps
        eased = 1 - (1 - t) ** 3
        out.append(start + (end - start) * eased)
    out[-1] = end
    return out


SHIMMER_BAND = 3
SHIMMER_EVERY = 0.04


def shimmer_frames(lines: tuple[str, ...] | list[str], base_style: str = "bold",
                   band: int = SHIMMER_BAND) -> list[Text]:
    """A bright band sweeping left to right across `lines`; the last frame is
    the plain `base_style` rendering. Monochrome on purpose: the band is
    `reverse`, which reads on any theme and adds no colour."""
    rows = tuple(lines)
    width = max((len(r) for r in rows), default=0)
    frames: list[Text] = []
    for head in range(-band, width + 1):
        text = Text()
        for r_i, row in enumerate(rows):
            for c_i, ch in enumerate(row):
                lit = head - band < c_i <= head and not ch.isspace()
                text.append(ch, style=f"{base_style} reverse" if lit else base_style)
            if r_i < len(rows) - 1:
                text.append("\n")
        frames.append(text)
    frames.append(Text("\n".join(rows), style=base_style))
    return frames


SPARK_GLYPHS = "▁▂▃▄▅▆▇█"


def sparkline(values, width: int = 12) -> str:
    """The last `width` values as a bar per value, scaled to the largest."""
    tail = [max(0, int(v)) for v in list(values)[-width:]]
    if not tail:
        return ""
    peak = max(tail) or 1
    return "".join(SPARK_GLYPHS[min(len(SPARK_GLYPHS) - 1, round(v / peak * (len(SPARK_GLYPHS) - 1)))] for v in tail)


def art_text(lines: tuple[str, ...] | list[str], style: str = "") -> Text:
    """Rows as one Rich Text. A Text, not a markup string: box-drawing and
    braille contain nothing Rich would misread, but a bare str handed to
    `Static.update()` on this Textual version is parsed as markup anyway, and
    a Text is the same object either way."""
    return Text("\n".join(lines), style=style)
