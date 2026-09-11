"""Looking at a screen and clicking on it, in the container Otto is working in.

Same seam as the browser, and for a stronger reason. A screen-control tool
pointed at a real machine can click anything its owner happens to have open;
pointed at a container it can click the things somebody deliberately put in the
container. That difference is why this drives `docker/otto-desktop` through
`agent/pipeline/execution.py`'s command runner rather than a host, and why Otto
ships no screen-control dependency of its own.

WHAT COMES BACK IS WORDS, NOT PIXELS. A capture is routed through
agent/pipeline/vision.py exactly as `view_image` is, so the conversation
receives a description. That keeps the text-only invariant vision.py argues for
at length, and it is why the reading tool takes a QUESTION -- "which window is
in front?", "where is the Save button?" -- rather than handing back an image. A
narrow question answered twice is close to being able to look; a generic
caption is worth little.

THIS IS THE FALLBACK. Agents that route each subtask to code or GUI and prefer
CODE reach 60.76% on OSWorld in 10.15 steps against ~15 for GUI-only, a 32%
reduction, and a hybrid action space is worth +22% relative; GUI-only chains are
described as brittle and prone to cascading failure. The desktop image is built
on the one Otto already drives precisely so the shell is still there -- a
desktop with no shell on it would force the worse path.

HONEST ABOUT GROUNDING. Knowing WHERE to click is the hard part and none of
this solves it: the agent reads a description and names coordinates, which is
what every GUI agent does and why the measured scores are what they are --
grounding is one of the two failure modes named in the benchmark that defined
the task. Expect look, act, look again.
"""
from __future__ import annotations

#: Ceiling on a capture handed to the vision model. A 1280x800 PNG is well
#: under this; the cap is here so a giant virtual screen fails loudly rather
#: than silently costing a fortune in image tokens.
MAX_CAPTURE_BYTES = 8 * 1024 * 1024

#: What the agent may ask the screen to do. Two operations, because a small
#: vocabulary a model uses correctly beats a faithful reproduction of a mouse.
ACT_OPS = ("click", "type")

#: Grab the root window as a PNG on stdout, base64'd so it survives a shell.
#: ImageMagick's `import` is already on the desktop image.
#: `-w0` because base64 wraps at 76 columns by default and strict decoding
#: rejects the newlines -- which is exactly how this failed the first time.
CAPTURE = (
    'import -display "$DISPLAY" -window root png:- 2>/dev/null | base64 -w0'
)


def click_command(x: int, y: int) -> str:
    """Move the pointer and click, via xdotool."""
    return f'xdotool mousemove {int(x)} {int(y)} click 1'


def type_command(text: str) -> str:
    """Type wherever focus is. `--clearmodifiers` because a stuck modifier from
    an earlier action turns ordinary text into a stream of shortcuts."""
    quoted = text.replace("'", "'\\''")
    return f"xdotool type --clearmodifiers --delay 12 '{quoted}'"


def parse_act(body: str) -> tuple[str, str] | str:
    """`(operation, argument)` from a CODE: body, or why it is not one."""
    op, _, argument = body.strip().partition(" ")
    op = op.strip().lower()
    if op not in ACT_OPS:
        return f"say `click X Y` or `type <text>` -- {op!r} is neither"
    return op, argument.strip()


def parse_point(argument: str) -> tuple[int, int] | str:
    """A click target, or why it is not one."""
    parts = argument.replace(",", " ").split()
    if len(parts) != 2 or not all(p.lstrip("-").isdigit() for p in parts):
        return "click needs two numbers: `click X Y`"
    return int(parts[0]), int(parts[1])
