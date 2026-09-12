"""The full-screen TUI: `otto tui`.

A second front-end over the same headless run_pipeline_stream() the REPL
(chat.py) drives -- nothing about the graph or its entrypoints changes for
this to exist, per 9.9's own boundary ("keep run() headless: it yields, it
never prints"). `otto chat` stays exactly as built -- this is additive, not
a replacement.

Every "command" that used to be a typed `/word` is a `SystemCommand` in
Textual's own command palette (ctrl+p), which is already a fuzzy-searchable,
arrow-key-navigable menu -- no custom dropdown UI needed. The one place free
text is unavoidable is the message box itself (there's no way to turn an
open-ended task into a menu) and the /score dialog's numeric value.

No agent-count picker and no swarm sidebar (2026-09-10): the router/
planner/solver/summarizer/finder/evaluator graph that replaced the swarm
pipeline dispatches exactly one specialist per round, so there is nothing
left to size or animate as a fan-out -- see agent/pipeline/nodes.py's
module docstring for the design discussion.

Thinking/result split (2026-09-10, design call: "we need to keep thinking
and result separate in TUI and show user only result with thinking as
collapsable thing"): `#transcript` stopped being one flat RichLog. Every
graph update from render_update() (shell.py) -- router's dispatch line,
each specialist's board line and output preview, evaluator's verdict line
-- is "thinking": the process, not the answer. That now goes into a fresh
RichLog created FOR THAT TURN, wrapped in a `Collapsible` -- open while
the turn runs, shut once it ends (see "The thinking log runs EXPANDED"
below for why that is not the collapsed-from-birth it started as), so it
never crowds the finished transcript but is one click away.
Everything the user should see without expanding anything -- their own
message, the final answer, where it got saved, an error -- is "result":
mounted straight into `#transcript` (now a VerticalScroll of stacked
widgets, not a single log) as its own `Static`, always visible.

Two widget types, on purpose, not one: `RichLog.write()` is what already
renders agent.cli.ui.THEME's custom style names (muted/spec/chosen/ok/bad/
warn) correctly -- see `on_mount`'s theme push below, unchanged from
before this split -- so every render_update() call keeps going through a
RichLog. `Static`, not RichLog, holds the one-shot result blocks: RichLog
is a fixed/flexible-height scrolling *viewport* (checked live against this
Textual version -- it fills available space rather than sizing to its
content), which looks wrong for a single block stacked among others in a
VerticalScroll; `Static` sizes to its renderable's actual height. Nothing
mounted as `Static` here uses a THEME-custom name (only "bold"/"dim"/
"red"/"green"/"yellow" -- primitive Rich style words, not aliases), so it
never needs the theme push RichLog does.

Copying cleanly (2026-09-10, same day, follow-up design call: "user cannot
copy text from model output from tui cleanly"): a mouse-selected Panel is
never clean -- box-drawing borders, wrapped lines, and Markdown decoration
are all part of what gets selected, the same terminal-copy problem
output.py's own docstring already documents for `save_final`. `ctrl+y` /
"Copy last answer" (command palette) sidesteps selection entirely: it
sends the raw `final_output` string -- the exact text `save_final` already
writes to disk, no Panel/Markdown around it -- to the system clipboard via
Textual's `copy_to_clipboard` (OSC 52), which most terminals honor without
any drag-select at all. The one gap OSC 52 itself has (Textual's own
docstring on it): it does not work on macOS Terminal.app. `action_copy_last`
says so in its own message rather than leaving a silent no-op, and the
saved-file path (already printed after every turn) is the fallback either
way.

A screen that moves (2026-09-12, design call: "optimize latency and steps,
and improve the tui"): until this, the entire middle of a turn was blank.
The graph streams one update per NODE, `agent` is a node, and a run spends
every model call and every tool call inside it -- measured across the twenty
golden items, 828 seconds of wall time with 96% of it inside model requests
and not one graph update in the middle of any of them. The longest item ran
131 seconds against an unchanging screen, and the honest reading of "it is
taking too long" is partly that nothing said otherwise.

`#status`, below the transcript and above the input, is now live for the
length of a turn: what the run is doing, which model is answering, how many
model requests it has spent, what tool it is inside, and the elapsed clock.
It is fed by `agent/pipeline/progress.py` -- a contextvar sink bound around
the stream, in the same idiom as `bind_budget`/`bind_workspace` -- not by a
new graph channel, because none of it is state.

The answer streams too. Every `partial` update carries the WHOLE reply so
far (a diffusing route's chunks are refinements, not slices), so once one
contains `FINAL:` the text after it is mounted and replaced in place, and
the answer appears as it is written rather than 30 to 130 seconds later in
one block. `_STREAM_EVERY` throttles what crosses to the UI thread: chunks
arrive far faster than a person can read, and marshalling every one of them
spends the UI thread on frames nobody sees.

And a turn can be stopped. `escape` sets the cancel Event the same seam
binds; `_call` checks it before spending, so a stop lands within one model
call and never pays for another. That is the difference between waiting out
a 28-minute run started by accident and pressing a key.

Mid-run questions (2026-09-10, same day, nodes.py's seventh refinement,
design call: "widget with multi choice + text bar" for how the TUI should
ask a question back): `AskUserModal`, below, is that widget -- an
`OptionList` for the "multi choice" half (only mounted when the run
actually offered choices) plus an `Input` for the "text bar" half, always
present, so a free-text answer is always possible even when choices are
offered too. `run_turn` runs on a worker THREAD (`@work(thread=True, ...)`
-- it always has), and a modal can only be pushed/answered from the UI's
own thread/event loop, so `_ask_user_blocking` bridges the two with a
plain `threading.Event`: `call_from_thread` schedules the modal on the UI
thread and the worker thread blocks on the event (NOT the UI thread --
the app stays fully responsive, redraws and all, while a turn is paused
waiting on a person) until the modal dismisses it.

One turn at a time, enforced here rather than by the worker decorator
(2026-09-12, bug report: "in otto tui it's acting weird and slow", with a
screenshot showing the same message posted twice, a `thinking…` block for
the second, and then a green `final` panel answering the FIRST). `run_turn`
carried `@work(..., exclusive=True, group="turn")` and that reads like it
serialises turns. It does not. Textual's `Worker.cancel()` (worker.py)
cancels the asyncio task wrapping `loop.run_in_executor(...)`; the
executor thread underneath it is not interruptible and runs to
completion. Verified against Textual 8.2.8: two `exclusive=True` thread
workers started 0.15s apart both ran all their steps. So a second Enter
while a turn was in flight started a SECOND pipeline run -- two sets of
model calls billed at once, two interleaved streams of
`call_from_thread` mounts into one transcript (which is how an older
run's `final` lands under a newer run's `thinking…`), and two threads
racing on `session.turn`, `session.trace_id` and `_last_output`. The
guard is now `_turn_running`, flipped on the UI thread inside
`on_input_submitted` before the worker is even started, so there is no
window to double-submit into; the message box is disabled for the
duration, which is also the only thing that ever told the user a turn was
still going. `exclusive=True` is gone, since keeping it would keep
implying a cancellation that cannot happen.

Modal input must not escape into a new turn (2026-09-12, same report, same
duplicated-line symptom). `Input.Submitted` bubbles the whole way up the
DOM -- Input, to the modal Screen, to the App -- and a handler on a
ModalScreen does not stop that by dismissing. Verified against Textual
8.2.8: an App-level `on_input_submitted` still fires for text typed into a
pushed ModalScreen. Untreated, answering an `AskUserModal` question posted
`you <answer>` twice AND launched a whole fresh turn on the answer text
alongside the paused run it was meant to resume, and submitting in
`ScoreDialog` ran the pipeline on the literal string "0.8 nice". Both
modals now call `event.stop()`, and `on_input_submitted` below also checks
`event.input.id` -- belt and braces, because the failure mode of getting
this wrong is silent and expensive.

The thinking log runs EXPANDED and collapses when the turn ends
(2026-09-12, same report, the "slow" half). `RichLog.write()` defers every
write until the widget's size is known (rich_log.py: `if not
self._size_known: self._deferred_renders.append(...)`), and a `Collapsible`
that starts collapsed never lays its contents out, so the widget's size is
never known. Verified against Textual 8.2.8: 200 writes into a collapsed
one left `lines == 0` and `len(_deferred_renders) == 200`, all of it
rendering in a single synchronous burst on the UI thread the moment
someone expanded it. So the old arrangement showed NOTHING while a turn
ran -- for a pipeline turn that is minutes of a still screen with no
indication anything is happening, which is most of what "slow" meant here
-- and then paid for all of it at once. Mounting expanded makes each board
line render as it arrives; collapsing at the end keeps the resting
transcript result-first, which is what this split was for.

The workspace is shown, not discovered (2026-09-12, design call: "we need
filesystem management so we can use it to write code and work on already
implemented codebases"). `otto tui` now opens on the directory it was launched
in and hands it to `run_pipeline_stream` on every turn -- as an ARGUMENT, not
by binding the contextvar here, because `run_turn` consumes the stream on a
worker thread and a contextvar set on the UI thread is not visible there (see
agent/pipeline/run.py for the binding's new home). Which directory otto is
pointed at decides what every answer this session can be, so `on_mount` says
it in the transcript rather than leaving it to be discovered when a file tool
refuses. `WorkspacePrompt` (below) and the "Workspace…" palette entry change
it mid-session; both refuse while a turn is running, because a turn has
already handed its workspace to the pipeline and changing it then would take
effect NEXT turn while looking like it took effect on this one.

`.thinking-log` is `height: auto` with a `max-height`, not a fixed
`height: 12` (2026-09-12, same screenshot: a two-line log drawn as a
twelve-row box, ten of them blank). RichLog is a ScrollView and this
module's own docstring above says a ScrollView fills rather than sizes to
content -- true for `height: auto` alone, but `auto` capped by
`max-height` measures correctly in Textual 8.2.8 (checked live: 3 lines
gave a height of 3, 43 lines gave 16).

Split, and dressed (2026-09-12, design call: "let's plan qol stuff we can add
to tui", then "add cool animations and ascii arts", "go big"). This module
kept the app and the turn; the modals moved to agent/cli/modals.py, the token
panel to agent/cli/usage_panel.py, the setup wizard lives in
agent/cli/setup_screen.py, and every frame table is data in agent/cli/art.py.
The animation rules, once: every timer is created on the UI thread by a
UI-thread method (`set_interval`), workers never touch one; the sparkle and
the sweep start only after the last streamed `partial`, so they never compete
with the answer for the UI thread; `OTTO_NO_ANIMATION=1` and any headless run
(the test suite) jump every table to its last frame, which is by construction
the plain resting rendering. The palette is still the only command surface --
a leading slash in the message box is a message, not a command (asked, and
kept that way).

Redesigned against the field (2026-09-13, design call: "look into
awesometui.com and make our tui look as cool as them"). The terminal agents
that list well there -- OpenCode (the site's developer-tool award), Charm's
Crush, Gemini CLI, and Mistral Vibe, which is built on this exact Textual
release -- share a visual language, and this module now speaks it:

  * the terminal's own background shows through (`Screen { background:
    transparent }`, as Vibe does), so otto sits in the terminal rather than
    painting over it, and Textual's theme picker (already in the palette as
    "Change theme") restyles everything through the theme variables every
    rule below uses;
  * no boxes around messages. What the person said carries a thick accent
    rule on its left; the answer carries a success-coloured one with a
    single header line (`● final · model · 0:12`) above the Markdown, the
    shape OpenCode and Crush both use;
  * one slim top bar -- what this session is about on the left, what it
    has spent on the right -- instead of a title header, and a `›` prompt
    with a dim hint line under it saying which seat answers next;
  * a right sidebar with dim ruled section headers (session, routing,
    usage) rather than a single boxed token panel, toggled by the same key;
  * the thinking trace stays a collapsible, but its board lines are drawn
    with glyphs and arrows (`◆ execute_bash → ok`) and it closes as
    "thought for 0:12 · 5 steps · 3 calls".

Setup opens itself when nothing is configured. A Router now constructs on a
keyless machine (agent/router/router.py `ready()`), which is what lets the app
reach `on_mount` and push the setup screen instead of dying at import; the
first turn on such a machine fails at `run_pipeline_stream`'s own entry check
with a message that points back at f2.
"""

from __future__ import annotations

import os
import threading
import time
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Annotated, Iterable, Optional

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.text import Text
from rich.console import Group
from rich.table import Table
from textual import work
from textual.actions import SkipAction
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.selection import Selection
from textual.widgets import Collapsible, Footer, Input, RichLog, Static

from agent.cli import art, clipboard
from agent.cli.art import SPINNER, animations_enabled
from agent.cli.chat import NO_WORKSPACE_HELP, WORKSPACE_HELP
from agent.cli.context import AppContext
from agent.cli.lessons import lessons_table
from agent.cli.modals import (  # noqa: F401 -- re-exported for tests and callers
    AskUserModal, ModelPinDialog, PathPicker, ScoreDialog, TaskPicker, WorkspacePrompt,
)
from agent.cli.output import save_final
from agent.cli.setup_screen import SetupBackend, SetupScreen, pin_options
from agent.cli.shell import (
    Session, describe_workspace, render_update, resolve_workspace, set_workspace,
)
from agent.cli.ui import THEME
from agent.cli.usage_panel import UsagePanel, _ID_PREFIXES, _short_model, _thousands  # noqa: F401
from agent.memory.lessons import all_lessons, bank_path, export_lessons, import_lessons
from agent.pipeline.budget import WRAP_UP_FRACTION, default_budget
from agent.pipeline.pricing import PRICES_AS_OF, format_cost  # noqa: F401 -- re-exported
from agent.pipeline.progress import Cancelled, Progress, bind_progress
from agent.pipeline.run import resume_pipeline_stream, run_pipeline_stream
from agent.pipeline.usage import UsageLedger
from agent.router import outcomes as seat_outcomes
from agent.router import overrides as route_overrides
from agent.router import setup as provider_setup
from agent.router.automap import propose
from agent.router.llm_provider import all_models, provider_class, provider_names
from agent.router.llm_provider.base import AuthError
from agent.router.mapping import TASK_ROUTES, Task
from agent.router.reload import reload_everything
from agent.router.router import NoViableRoute


#: Seconds between answer frames that actually cross to the UI thread.
#: Chunks arrive an order of magnitude faster than anyone reads, and
#: marshalling every one of them spends the UI thread drawing frames nobody
#: sees. A tenth of a second still reads as continuous typing.
_STREAM_EVERY = 0.1

#: Frames for the status-line spinner (agent/cli/art.py). Kept under this
#: name for the tests that read it.
_SPINNER = SPINNER


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


#: Where the chosen theme is remembered between launches. Beside the other
#: per-installation state, and only written by a real terminal session --
#: never by the test suite's headless apps.
UI_STATE_PATH = seat_outcomes.DB_DIR / "ui.json"
THEME_ENV = "OTTO_THEME"


def _saved_theme() -> str | None:
    import json

    name = os.environ.get(THEME_ENV, "").strip()
    if name:
        return name
    try:
        return json.loads(UI_STATE_PATH.read_text()).get("theme") or None
    except (OSError, ValueError, AttributeError):
        return None


def configured_providers() -> tuple[str, ...]:
    """Providers with a key in the environment. No network -- this decides
    whether to open setup on launch, and a launch must not wait on a vendor."""
    found = []
    for name in provider_names():
        try:
            if provider_class(name).is_configured():
                found.append(name)
        except Exception:  # an SDK that failed to import is "not configured"
            continue
    return tuple(found)


class CopyableStatic(Static):
    """A Static whose text can be drag-selected whatever it holds.

    Textual's own `Widget.get_selection` hands back text only when the
    widget's visual is Text or Content; a Static holding Markdown, a Table
    or a Group renders through a RichVisual and yields nothing, so the
    answer -- the one thing most worth copying -- could not be selected.
    This renders the content to plain text at the current width instead.
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        text = clipboard.plain_text_of(self.content, self.content_size.width or self.size.width or 80)
        if not text:
            return None
        return selection.extract(text), "\n"


class SelectableRichLog(RichLog):
    """The thinking log, drag-selectable line by line. RichLog keeps its
    rendered lines as Strips, which already are the text on screen."""

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        lines = [strip.text for strip in self.lines]
        if not lines:
            return None
        return selection.extract("\n".join(lines)), "\n"


class OttoScreen(Screen):
    """The default screen, with ctrl+c copying the selection through every
    clipboard route (agent/cli/clipboard.py) rather than OSC 52 alone. With
    nothing selected the key falls through to Textual's own "ctrl+q quits"
    notice, as before."""

    def action_copy_text(self) -> None:
        if not self.app.action_copy_selection():
            raise SkipAction()


class Banner(Static):
    """The wordmark, revealed left to right on launch (agent/cli/art.py).
    Lives at the top of the sidebar -- one mark per screen, where Crush puts
    its own -- not in the transcript, where it would sit in every session's
    scrollback. Headless or `OTTO_NO_ANIMATION`: the last frame, at once."""

    DEFAULT_CSS = "Banner { height: auto; margin-bottom: 1; }"

    def __init__(self, lines: tuple[str, ...] = art.WORDMARK_SMALL, tagline: str = "") -> None:
        super().__init__(art.art_text(lines, art.WORDMARK_STYLE), id="banner")
        self._frames = art.reveal_frames(lines)
        self._tagline = tagline
        self._i = 0
        self._timer = None

    def on_mount(self) -> None:
        if animations_enabled(self.app):
            self._show(0)
            self._timer = self.set_interval(art.REVEAL_EVERY, self._advance)
        else:
            self._show(len(self._frames) - 1)

    def _advance(self) -> None:
        self._i += 1
        self._show(min(self._i, len(self._frames) - 1))
        if self._i >= len(self._frames) - 1 and self._timer is not None:
            self._timer.stop()

    def _show(self, i: int) -> None:
        text = art.art_text(self._frames[i], art.WORDMARK_STYLE)
        if i == len(self._frames) - 1 and self._tagline:
            text.append("\n" + self._tagline, style=art.TAGLINE_STYLE)
        self.update(text)

    def shimmer(self) -> None:
        """One bright band across the mark -- the "done" signal when a turn
        lands (Crush's mark does this). Settles on the plain wordmark."""
        if not animations_enabled(self.app):
            return
        frames = art.shimmer_frames(self._frames[-1], art.WORDMARK_STYLE)
        state = {"i": 0, "timer": None}

        def step() -> None:
            i = state["i"]
            self.update(frames[min(i, len(frames) - 1)])
            state["i"] = i + 1
            if i >= len(frames) - 1 and state["timer"] is not None:
                state["timer"].stop()

        state["timer"] = self.set_interval(art.SHIMMER_EVERY, step)


class Sidebar(Vertical):
    """The right-hand column: the small wordmark, then dim ruled sections
    (session, routing, usage). Sections, not a boxed panel -- Crush's and
    OpenCode's sidebars label with a rule and let the terminal background
    through, and it reads as part of the screen rather than a widget on it.
    The usage section is the UsagePanel it always was, so everything that
    queried `#usage` still finds it."""

    DEFAULT_CSS = """
    Sidebar { width: 32; height: 1fr; padding: 0 1; border-left: solid $panel-lighten-2; }
    Sidebar > Static { height: auto; }
    Sidebar .side-brand { margin-bottom: 1; }
    Sidebar .side-rule { margin-top: 1; }
    Sidebar UsagePanel { width: 1fr; height: auto; border: none; padding: 0; }
    """

    def __init__(self, ledger: UsageLedger) -> None:
        super().__init__(id="sidebar")
        self._ledger = ledger

    def compose(self) -> ComposeResult:
        yield Banner(art.WORDMARK_SMALL)
        yield Static(art.section("session"), classes="side-rule")
        yield CopyableStatic(Text(""), id="side-session")
        yield Static(art.section("routing"), classes="side-rule")
        yield CopyableStatic(Text(""), id="side-routing")
        yield Static(art.section("usage"), classes="side-rule")
        yield UsagePanel(self._ledger)

    @staticmethod
    def _rows(pairs: list[tuple[str, str]]) -> Table:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="dim", no_wrap=True)
        grid.add_column(overflow="fold")
        for key, value in pairs:
            grid.add_row(key, value)
        return grid

    def refresh_session(self, *, workspace: Path | None, turns: int, mode: str, model: str,
                        turn_tokens: list[int] = ()) -> None:
        glyph = art.MODE_GLYPHS.get(mode, "")
        rows = [
            ("workspace", workspace.name if workspace else "off"),
            ("turns", str(turns)),
            ("mode", f"{glyph} {mode}".strip() if mode else "—"),
            ("model", _short_model(model) if model else "—"),
        ]
        if turn_tokens:
            # Tokens per turn as bars, btop-style: the shape of the session
            # at a glance, without a number per turn.
            rows.append(("per turn", art.sparkline(turn_tokens)))
        self.query_one("#side-session", Static).update(self._rows(rows))

    def refresh_routing(self, *, pins: dict, providers: tuple[str, ...]) -> None:
        pinned = ", ".join(sorted(t.value for t in pins)) if pins else "none"
        self.query_one("#side-routing", Static).update(self._rows([
            ("pins", pinned),
            ("providers", ", ".join(providers) if providers else "none — f2"),
        ]))


class OttoApp(App):
    TITLE = "otto"
    BINDINGS = [
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+y", "copy_last", "Copy last answer"),
        ("escape", "stop_turn", "Stop this turn"),
        # The panel is 34 columns that the transcript does not get. Worth it
        # while you are watching spend, not worth it on an 80-column terminal
        # reading a long answer -- so it is a toggle rather than a decision
        # made once for everybody.
        ("ctrl+t", "toggle_usage", "Sidebar"),
        ("f2", "setup", "Setup"),
    ]
    DEFAULT_CSS = """
    Screen { background: transparent; }
    * { scrollbar-size: 1 1; scrollbar-background: transparent; scrollbar-color: $panel-lighten-2; }
    #topbar { height: 1; padding: 0 1; background: $panel; }
    #body { height: 1fr; }
    #transcript { width: 1fr; height: 1fr; padding: 0 1; }
    #transcript > Static { height: auto; }
    #transcript .user { border-left: thick $accent; padding: 0 1; margin: 1 0 0 0; }
    #transcript .answer { border-left: thick $success; padding: 0 1; margin: 1 0 0 0; }
    #transcript .meta { color: $text-muted; }
    #transcript Collapsible { padding: 0; margin: 1 0 0 0; border: none; background: transparent; }
    #transcript Collapsible > Contents { padding: 0 0 0 2; }
    #transcript CollapsibleTitle { color: $text-muted; padding: 0; }
    .thinking-log { height: auto; max-height: 16; background: transparent; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    #prompt { height: auto; padding: 0 1; border-top: solid $panel-lighten-2; }
    #prompt-glyph { width: 2; height: 1; color: $accent; text-style: bold; }
    #message-input { height: 1; border: none; padding: 0; background: transparent; }
    #message-input:focus { border: none; background: transparent; }
    #hint { height: 1; padding: 0 1; color: $text-muted; }
    #empty-state { height: auto; margin: 1 0; }
    """

    #: What the message box says when it is free. Restored by `_set_busy`,
    #: which swaps in a "still working" placeholder for as long as a turn
    #: owns the session -- the disabled box plus this line are the only
    #: thing that ever tells the user a turn is still going.
    IDLE_PLACEHOLDER = "type a message… (ctrl+p for commands)"
    BUSY_PLACEHOLDER = "working… one turn at a time"

    def __init__(self, ctx: AppContext, workspace: Path | None = None) -> None:
        super().__init__()
        self.ctx = ctx
        self.session = Session(ctx=ctx, workspace=workspace)
        #: The last turn's raw final_output (module docstring, "Copying
        #: cleanly") -- exactly the string save_final() wrote to disk, no
        #: Panel/Markdown wrapper. None before any turn has finished, or
        #: after one that produced nothing.
        self._last_output: str | None = None
        #: Whether a run_turn worker currently owns `self.session` (module
        #: docstring, "One turn at a time"). Touched ONLY on the UI thread --
        #: set in `on_input_submitted` before the worker starts, cleared by
        #: the worker through `call_from_thread` -- so it needs no lock, and
        #: there is no window between the check and the set for a second
        #: Enter to slip through. It is also what the status line reads to
        #: decide whether there is anything to draw.
        self._turn_running = False
        #: One ledger for the SESSION, handed to every turn and every resume
        #: (agent/pipeline/usage.py), so the panel is cumulative by
        #: construction rather than by adding per-turn snapshots up. Cleared
        #: with the session by `action_new_session`.
        self.usage = UsageLedger()
        #: Everything the status line draws, written from the worker thread
        #: and read from the UI thread's tick. Plain attributes rather than a
        #: lock: each is a single assignment of an immutable value, and the
        #: worst a torn read can do is show one stale field for a frame.
        self._phase = ""
        self._model = ""
        self._tool = ""
        self._mode = ""
        self._calls = 0
        self._budget_max: int | None = None
        self._started = 0.0
        self._frame = 0
        #: Set by `escape`, read inside `_call` before it spends again
        #: (agent/pipeline/progress.py). One Event per turn -- reusing one
        #: across turns would carry a stop into the next run.
        self._cancel: threading.Event | None = None
        #: The live answer block for the turn in flight, and when it was last
        #: redrawn. Mounted the first time a reply contains FINAL:.
        self._answer: Static | None = None
        self._drawn_at = 0.0
        #: The last catalogue anything here fetched, so "Pin a model…" does
        #: not pay for four vendor round trips after setup just made them.
        self._model_cache: list | None = None
        self._lessons_busy = False
        #: Last model that answered, for the sidebar after a turn ends.
        self._last_model = ""
        #: Motion state (agent/cli/art.py). The snake walks in the status
        #: line while a turn runs; the caret blinks at the end of a streaming
        #: answer; the odometer rolls the spend figures to their new values.
        self._snake = art.Snake()
        self._snake_frame = art.BRAILLE_BLANK * 2
        self._partial_text: str | None = None
        self._caret_on = True
        self._caret_timer = None
        self._shown_spend: tuple[int, float] = (0, 0.0)
        self._spend_timer = None
        self._turn_tokens: list[int] = []
        self._tokens_before_turn = 0
        saved = _saved_theme()
        if saved and saved in self.available_themes:
            self.theme = saved

    def compose(self) -> ComposeResult:
        yield CopyableStatic(Text(""), id="topbar")
        # The transcript keeps `id="transcript"` and everything that queries
        # for it is unchanged -- this only puts a sibling beside it.
        with Horizontal(id="body"):
            yield VerticalScroll(id="transcript")
            yield Sidebar(self.usage)
        yield Static("", id="status")
        with Horizontal(id="prompt"):
            yield Static(art.PROMPT_GLYPH, id="prompt-glyph")
            yield Input(placeholder=self.IDLE_PLACEHOLDER, id="message-input")
        yield Static(Text(""), id="hint")
        yield Footer()

    def watch_theme(self, theme: str) -> None:
        """Remember a theme picked from the palette's "Change theme". Only a
        real terminal writes it; a headless test app never touches ~/.otto."""
        if self.is_headless or not theme:
            return
        import json
        try:
            UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            UI_STATE_PATH.write_text(json.dumps({"theme": theme}, indent=2) + "\n")
        except OSError:
            pass

    def on_mount(self) -> None:
        # render_update() (shell.py) writes markup keyed to agent.cli.ui.
        # THEME's names (muted, spec, ok, bad, chosen, warn) into whichever
        # RichLog it's handed (module docstring: the per-turn "thinking"
        # log) -- RichLog.write() renders through the App's own plain
        # Console, which knows nothing about those names and raises
        # MissingStyle the first time a board line shows up. Pushing the
        # same theme onto self.console once, here, makes every
        # RichLog.write() for the life of the app resolve identically to
        # the REPL, with zero changes to the shared render_update().
        self.console.push_theme(THEME)
        self.message_box.focus()
        self._post(f"[dim]{art.TAGLINE}[/]", classes="meta")
        # Said once, up front, rather than left to be discovered when a file
        # tool refuses: which directory otto is pointed at decides what every
        # answer this session can possibly be.
        self._post(f"[dim]{describe_workspace(self.session.workspace)}[/]", classes="meta")
        self.transcript.mount(Static(art.art_text(art.EMPTY_STATE, "dim"), id="empty-state"))
        self._refresh_chrome()
        # The clock has to tick on its own: between two model calls nothing
        # reports anything for ten seconds at a stretch, and a status line
        # that only moves when the run moves reads as a frozen app. Five
        # frames a second when animating, so the spinner reads as motion.
        self.set_interval(0.2 if animations_enabled(self) else 1.0, self._tick)
        if not configured_providers():
            # A fresh machine. Nothing can run yet, so say so by opening the
            # one screen that fixes it -- after this mount settles.
            self.call_after_refresh(self.action_setup)

    @property
    def usage_panel(self) -> "UsagePanel":
        return self.query_one("#usage", UsagePanel)

    def _refresh_usage(self) -> None:
        """UI-thread redraw of the token panel and the top bar's spend. The
        worker calls this through `call_from_thread` after each graph update,
        which is often enough to watch a turn spend and rare enough to cost
        nothing."""
        self.usage_panel.refresh_usage()
        self._refresh_topbar()

    # ---- the chrome: top bar, sidebar sections, hint line -------------

    def _refresh_topbar(self) -> None:
        """Redraw the top bar. When the spend changed and motion is on, the
        tokens and dollars roll to their new values over a few frames (the
        odometer); the last frame is always the exact figure."""
        snap = self.usage.snapshot()
        target = (int(snap["total_tokens"]), float(snap["cost"] or 0.0))
        if not animations_enabled(self) or target == self._shown_spend or not snap["calls"]:
            self._shown_spend = target
            self._draw_topbar(snap, *target)
            return
        tokens_path = art.tween(self._shown_spend[0], target[0])
        cost_path = art.tween(self._shown_spend[1], target[1])
        frames = list(zip(tokens_path, cost_path))
        if self._spend_timer is not None:
            self._spend_timer.stop()
        state = {"i": 0}

        def step() -> None:
            i = state["i"]
            tokens, cost = frames[min(i, len(frames) - 1)]
            self._shown_spend = (int(tokens), cost)
            self._draw_topbar(self.usage.snapshot(), int(tokens), cost)
            state["i"] = i + 1
            if i >= len(frames) - 1 and self._spend_timer is not None:
                self._spend_timer.stop()
                self._spend_timer = None

        self._spend_timer = self.set_interval(art.TWEEN_EVERY, step)

    def _draw_topbar(self, snap: dict, tokens: int, cost: float) -> None:
        left = Text()
        left.append("otto", style="bold")
        where = self.session.workspace.name if self.session.workspace else "no workspace"
        left.append(f"  {art.PROMPT_GLYPH}  {where}", style="dim")
        right = Text(style="dim")
        if snap["calls"]:
            right.append(f"{snap['calls']} req · {_thousands(tokens)} tok · ")
            right.append(format_cost(cost) + ("+" if not snap["fully_priced"] else ""))
        else:
            right.append("nothing spent yet")
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")
        grid.add_row(left, right)
        self.query_one("#topbar", Static).update(grid)

    def _refresh_hint(self) -> None:
        """Which seat answers next and on what, from the routing table alone
        -- no network, so it is a promise about the table, not the vendor."""
        head = route_overrides.pinned_spec(Task.REASON) or (TASK_ROUTES[Task.REASON][0].spec or "?")
        why = "pinned" if route_overrides.pinned_spec(Task.REASON) else "default route"
        hint = Text(style="dim")
        hint.append(f"{art.MODE_GLYPHS['solve']} solve → {head}  ({why})")
        hint.append("     esc stop · ctrl+p commands · f2 setup · ctrl+t sidebar")
        self.query_one("#hint", Static).update(hint)

    def _refresh_sidebar(self) -> None:
        try:
            side = self.query_one(Sidebar)
        except Exception:
            return
        model = self._model or self._last_model
        if not model:
            rows = self.usage.snapshot()["models"]
            model = rows[-1]["model"] if rows else ""
        side.refresh_session(workspace=self.session.workspace, turns=self.session.turn,
                             mode=self._mode, model=model, turn_tokens=self._turn_tokens)
        side.refresh_routing(pins=route_overrides.active_pins(), providers=configured_providers())

    def _refresh_chrome(self) -> None:
        """Everything around the transcript, redrawn together. UI thread."""
        self._refresh_topbar()
        self._refresh_hint()
        self._refresh_sidebar()

    # ---- copying (agent/cli/clipboard.py) --------------------------------

    def get_default_screen(self) -> Screen:
        return OttoScreen(id="_default")

    def action_copy_selection(self) -> bool:
        """Drag-select anything, press ctrl+c: the text lands on the
        clipboard with no borders in it. Returns False when nothing is
        selected, so the key can fall through."""
        raw = self.screen.get_selected_text()
        if raw is None:
            return False
        text = clipboard.clean(raw)
        if not text.strip():
            return False
        how = clipboard.copy(self, text)
        chars = len(text)
        self.notify(f"copied {chars} character{'s' if chars != 1 else ''} {how}", timeout=4)
        self.screen.clear_selection()
        return True

    # ---- the status line ----------------------------------------------

    def _tick(self) -> None:
        """On the UI thread. Redraws the status line from whatever the worker
        thread last wrote, so the elapsed clock and the spinner keep moving
        through a long model call."""
        if not self._turn_running:
            return
        self._frame += 1
        if animations_enabled(self):
            self._snake_frame = self._snake.step()
        self._draw_status()

    def _draw_status(self) -> None:
        if not self._turn_running:
            self.query_one("#status", Static).update("")
            return
        line = Text()
        if animations_enabled(self):
            line.append(self._snake_frame, style="bold")
        else:
            line.append(SPINNER[self._frame % len(SPINNER)], style="bold")
        if self._phase:
            line.append(" · ")
            line.append(self._phase, style=art.phase_style(self._phase))
        if self._mode in art.MODE_GLYPHS:
            line.append(" · ")
            line.append(f"{art.MODE_GLYPHS[self._mode]} {self._mode}", style="bold")
        if self._tool:
            line.append(" · ")
            line.append(self._tool, style="bold")
        if self._model:
            line.append(" · ")
            line.append(self._model, style="dim")
        if self._calls:
            line.append(" · ")
            line.append_text(art.meter(self._calls, self._budget_max, warn_at=WRAP_UP_FRACTION))
            line.append(f" {self._calls} calls", style="dim")
        line.append(" · ")
        line.append(_clock(time.monotonic() - self._started), style="dim")
        line.append(" · esc to stop", style="dim")
        self.query_one("#status", Static).update(line)

    def _on_progress(self, update: Progress) -> None:
        """The progress sink, called on the WORKER thread. Does as little as
        possible here and marshals the rest: everything below writes plain
        attributes, and the tick on the UI thread is what turns them into
        pixels. The one exception is a streamed answer, which has to mount
        and update a widget and so goes through call_from_thread --
        throttled, or a fast stream floods the UI thread."""
        self._calls = update.calls or self._calls
        if update.kind == "call_start":
            self._model = update.text
            self._tool = ""
            self._phase = self._phase or "thinking"
        elif update.kind == "phase":
            self._phase = update.text
            self._tool = ""
        elif update.kind == "tool":
            target = (update.detail or {}).get("target", "")
            self._tool = f"{update.text} {target}".strip()
        elif update.kind == "partial":
            self._stream_answer(update.partial)

    def _stream_answer(self, partial: str) -> None:
        """Show the answer as it is written, once there is an answer to show.

        Every frame carries the WHOLE reply, not the next slice -- a
        diffusing route's chunks are successive refinements of one answer --
        so this replaces rather than appends. Before FINAL: appears the reply
        is a tool call, which belongs in the thinking log and not on screen
        as if it were the result.
        """
        marker = partial.find("FINAL:")
        if marker < 0:
            return
        now = time.monotonic()
        if now - self._drawn_at < _STREAM_EVERY:
            return
        self._drawn_at = now
        self.call_from_thread(self._show_partial_answer,
                              partial[marker + len("FINAL:"):].strip())

    def _answer_block(self, body: RenderableType, glyph: str, glyph_style: str, label: str) -> Group:
        """The answer's shape: one header line -- bullet, label, model, clock
        -- above the body. No box (module docstring, "Redesigned")."""
        header = Text()
        header.append(glyph, style=glyph_style)
        header.append(f" {label}", style="bold" if glyph == art.BULLET_ANSWER else "dim")
        model = self._model or self._last_model
        if model:
            header.append(f" · {_short_model(model)}", style="dim")
        if self._started:
            header.append(f" · {_clock(time.monotonic() - self._started)}", style="dim")
        return Group(header, Text(""), body)

    def _show_partial_answer(self, text: str) -> None:
        if self._answer is None:
            self._answer = CopyableStatic("", classes="answer")
            self.transcript.mount(self._answer)
            if animations_enabled(self) and self._caret_timer is None:
                self._caret_timer = self.set_interval(art.CARET_EVERY, self._blink_caret)
        self._partial_text = text
        self._draw_partial()
        self.transcript.scroll_end(animate=False)

    def _draw_partial(self) -> None:
        if self._answer is None or self._partial_text is None:
            return
        body = Text(self._partial_text)
        if animations_enabled(self) and self._caret_on:
            body.append(art.CARET, style="bold")
        self._answer.update(self._answer_block(body, art.BULLET_STREAMING, "dim", "answering…"))

    def _blink_caret(self) -> None:
        self._caret_on = not self._caret_on
        self._draw_partial()

    def _stop_caret(self) -> None:
        if self._caret_timer is not None:
            self._caret_timer.stop()
            self._caret_timer = None
        self._partial_text = None
        self._caret_on = True

    def action_stop_turn(self) -> None:
        """`escape`. Cooperative: agent/pipeline/progress.py checks this
        before every model request, so the stop lands within one call and the
        run never pays for another. Saying so matters -- a key that looks
        like it did nothing for ten seconds is worse than no key."""
        if not self._turn_running or self._cancel is None:
            return
        self._cancel.set()
        self._phase = "stopping after this call"
        self._draw_status()

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    @property
    def message_box(self) -> Input:
        """The one Input that belongs to the app itself. Queried by id, not
        by type: a pushed modal has an Input of its own, and `query_one(Input)`
        would happily hand back whichever came first in the DOM."""
        return self.query_one("#message-input", Input)

    def _set_busy(self, busy: bool) -> None:
        """Take or release the session for a turn (module docstring, "One
        turn at a time"). UI thread only -- a worker calls it through
        `call_from_thread`. Disabling the box is both the lock's visible
        half and the only progress signal the TUI had; re-focusing on
        release means the next message is typed without reaching for the
        mouse.
        """
        self._turn_running = busy
        if not busy:
            self._draw_status()          # or the last phase would linger
        box = self.message_box
        box.disabled = busy
        box.placeholder = self.BUSY_PLACEHOLDER if busy else self.IDLE_PLACEHOLDER
        self.sub_title = "working…" if busy else ""
        if not busy:
            box.focus()

    def _post(self, renderable: RenderableType, classes: str = "") -> None:
        """Mount one *result*-side block: always visible, never collapsed
        (module docstring). `renderable` is anything Static accepts -- a
        markup string using only primitive Rich style words, or a Rich
        renderable like Panel/Table. `classes` picks the look: "user" for
        what the person typed, "answer" for what otto said, "meta" for the
        dim lines in between. Safe to call from the UI thread directly; a
        worker thread must go through `self.call_from_thread` the same way
        it already does for everything else that touches the widget tree.
        """
        self.transcript.mount(CopyableStatic(renderable, classes=classes or None))
        self.transcript.scroll_end(animate=False)

    def _drop_empty_state(self) -> None:
        for widget in self.query("#empty-state"):
            widget.remove()

    def _ask_user_blocking(self, question: str, choices: list[str]) -> str:
        """Called from the run_turn WORKER thread (module docstring). Shows
        AskUserModal on the UI thread via call_from_thread, then blocks
        THIS (worker) thread -- not the UI thread, which keeps redrawing
        normally -- on a threading.Event until the modal dismisses with an
        answer.
        """
        done = threading.Event()
        answer: list[str] = [""]

        def _show() -> None:
            def _dismissed(result: str) -> None:
                answer[0] = result
                done.set()
            self.push_screen(AskUserModal(question, choices), _dismissed)

        self.call_from_thread(_show)
        done.wait()
        return answer[0]

    def _sweep_thinking(self, block: Collapsible) -> None:
        """OpenCode's travelling dot, in the open thinking block's title, for
        as long as the turn runs. One timer per turn; stops itself once the
        block is closed."""
        if not animations_enabled(self):
            return
        state = {"i": 0, "timer": None}

        def step() -> None:
            if block.collapsed or not self._turn_running or not block.is_attached:
                if state["timer"] is not None:
                    state["timer"].stop()
                return
            state["i"] += 1
            sweep = art.dot_sweep(state["i"]).plain
            block.title = f"{self._mode_prefix()}{art.BULLET_THINKING} thinking {sweep}"

        state["timer"] = self.set_interval(0.12, step)

    def _post_thinking(self, thinking_log: RichLog, title: str) -> Collapsible:
        """Mount one *thinking*-side block: a fresh RichLog that
        render_update() writes this turn's board lines and output previews
        into (module docstring), inside a Collapsible that starts OPEN.

        Open, not collapsed -- module docstring, "The thinking log runs
        EXPANDED". A RichLog inside a collapsed Collapsible is never laid
        out, so `RichLog.write()` buffers every call instead of rendering
        it: nothing at all reaches the screen while the turn runs, and the
        whole backlog then renders in one synchronous burst if anyone
        expands it. `_close_thinking` shuts it once the turn is done, which
        is where the collapsed-by-default intent actually belongs.
        """
        thinking_log.add_class("thinking-log")
        block = Collapsible(thinking_log, title=title, collapsed=False)
        self.transcript.mount(block)
        self.transcript.scroll_end(animate=False)
        self._sweep_thinking(block)
        return block

    def _mode_prefix(self) -> str:
        if self._mode in art.MODE_GLYPHS:
            return f"{art.MODE_GLYPHS[self._mode]} {self._mode} · "
        return ""

    def _retitle_thinking(self, block: Collapsible) -> None:
        """The agent changed mode: say which, with its glyph, in the title."""
        block.title = f"{self._mode_prefix()}{art.BULLET_THINKING} thinking…"
        self._refresh_sidebar()

    def _close_thinking(self, block: Collapsible, steps: int,
                        calls: int = 0, elapsed: float = 0.0) -> None:
        """Shut this turn's thinking block now that the answer is on screen,
        labelled with how much is folded away inside it so it is obvious
        there is something to open -- and with what the turn cost, which is
        the one number a person wants after the fact and should not have to
        expand anything to read. Animated, the title sweeps once first;
        headless, it shuts at once."""
        parts = [f"{steps} steps"] if steps else []
        if calls:
            parts.append(f"{calls} model calls")
        if elapsed:
            parts.append(_clock(elapsed))
        prefix = self._mode_prefix()
        # "thought for 0:12 · 5 steps · 3 model calls" -- the clock first,
        # because that is the number a person asks for after the fact.
        parts = ([_clock(elapsed)] if elapsed else []) + [p for p in parts if not p.endswith(_clock(elapsed))]
        final_title = (f"{prefix}{art.BULLET_THOUGHT} thought for {' · '.join(parts)}"
                       if parts else f"{prefix}{art.BULLET_THOUGHT} thought")

        def settle() -> None:
            block.title = final_title
            block.collapsed = True

        if not animations_enabled(self):
            settle()
            return
        frames = list(art.SWEEP_FRAMES)
        state = {"i": 0, "timer": None}

        def step() -> None:
            i = state["i"]
            if i < len(frames):
                block.title = f"{prefix}{frames[i]} thinking…"
                state["i"] = i + 1
                return
            settle()
            if state["timer"] is not None:
                state["timer"].stop()

        state["timer"] = self.set_interval(art.SWEEP_EVERY, step)

    # ---- the command palette (ctrl+p): the arrow-key menu ------------

    def get_system_commands(self, screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Setup…", "Configure providers, keys, models and which model answers each task", self.action_setup)
        yield SystemCommand("Pin a model for a task…", "Choose the model for one task, two picks", self.action_pin_model)
        yield SystemCommand("Workspace…", "Browse to the directory otto may read and write", self.action_workspace)
        yield SystemCommand("Route a task…", "Show how a task resolves, without spending a turn", self.action_pick_route)
        yield SystemCommand("List models", "List every configured model", self.action_models)
        yield SystemCommand("Check providers", "Run otto doctor", self.action_doctor)
        yield SystemCommand("Rate last answer: good", "Score the last answer 1.0", lambda: self.action_score(1.0, ""))
        yield SystemCommand("Rate last answer: bad", "Score the last answer 0.0", lambda: self.action_score(0.0, ""))
        yield SystemCommand("Rate last answer…", "Score the last answer with a value and a comment", self.action_score_dialog)
        yield SystemCommand("Copy last answer", "Copy the raw final answer to your clipboard", self.action_copy_last)
        yield SystemCommand("Copy selection", "Copy the drag-selected text, borders left out (ctrl+c)", self.action_copy_selection)
        yield SystemCommand("Show lessons", "What otto has learned, from ~/.otto/memory/lessons.db", self.action_show_lessons)
        yield SystemCommand("Export lessons…", "Write the lesson bank to a JSON file", self.action_export_lessons)
        yield SystemCommand("Import lessons…", "Merge a JSON file into the lesson bank, dropping duplicates", self.action_import_lessons)
        yield SystemCommand("New session", "Clear history, start fresh", self.action_new_session)
        yield SystemCommand("Toggle sidebar", "Show or hide the session, routing and usage sidebar",
                            self.action_toggle_usage)

    # ---- actions behind those commands --------------------------------

    def action_copy_last(self) -> None:
        """ctrl+y / command palette (module docstring, "Copying cleanly").
        Sends the RAW final_output straight to the system clipboard via
        OSC 52 -- no Panel border, no Markdown rendering, nothing a mouse
        selection would drag in. Doesn't work on macOS Terminal.app
        (Textual's own copy_to_clipboard docstring); says so plainly
        rather than leaving a silent no-op, since there's no way to detect
        that case from here and a copy that quietly did nothing is worse
        than one that explains itself.
        """
        if not self._last_output:
            self._post("[yellow]nothing to copy yet[/]")
            return
        how = clipboard.copy(self, self._last_output)
        self.notify(f"copied the last answer ({len(self._last_output)} characters) {how}", timeout=4)

    def action_pick_route(self) -> None:
        def done(task: Task | None) -> None:
            if task is not None:
                self._route_task(task)
        self.push_screen(TaskPicker(), done)

    @work(thread=True)
    def _route_task(self, task: Task) -> None:
        try:
            d = self.ctx.router.resolve(task)
        except (NoViableRoute, AuthError) as exc:
            self.call_from_thread(self._post, f"[red]no viable model for {task.value}: {exc}[/]")
            return
        why = (" (your pin)" if route_overrides.is_pin(task, TASK_ROUTES[task][d.index]) else
               " (degraded)" if d.fell_back else
               " (on observed results)" if d.chosen_on_evidence else "")
        line = f"{task.value} -> {d.provider}:{d.model.id}{why}"
        self.call_from_thread(self._post, f"[dim]{line}[/]")

    @work(thread=True)
    def action_models(self) -> None:
        from agent.cli.models import models_table
        found = sorted(all_models(None), key=lambda m: (m.provider, m.id))
        self._model_cache = found
        self.call_from_thread(self._post, models_table(found) if found else "[dim]no models matched[/]")

    @work(thread=True)
    def action_doctor(self) -> None:
        from agent.cli.doctor import health_table, router_view
        from agent.router.llm_provider import health_report
        reports = health_report()
        self.call_from_thread(self._post, health_table(reports))
        self.call_from_thread(self._post, router_view(self.ctx.router))

    def action_score(self, value: float, comment: str) -> None:
        if self.session.trace_id is None:
            self._post("[yellow]nothing to rate yet[/]")
            return
        # Reads trace_id HERE, on the UI thread, and hands it to the worker
        # as an argument: by the time the score is posted the next turn may
        # already have replaced it, and a rating must land on the answer the
        # person was actually looking at.
        self._send_score(self.session.trace_id, value, comment)

    @work(thread=True)
    def _send_score(self, trace_id: str, value: float, comment: str) -> None:
        """Off the UI thread, because `create_score`/`flush` are Langfuse
        network calls and `flush` blocks until the queue drains -- on the
        event loop that freezes the whole app, redraws and keystrokes
        included, for as long as the network takes.
        """
        try:
            self.ctx.client.create_score(
                name="user_feedback", value=value, data_type="NUMERIC",
                trace_id=trace_id, comment=comment or None,
            )
            self.ctx.client.flush()
        except Exception as exc:
            self.call_from_thread(
                self._post, f"[red]could not record the score: {type(exc).__name__}: {exc}[/]"
            )
            return
        self.call_from_thread(self._post, f"[dim]scored {value:g}[/]")

    def action_score_dialog(self) -> None:
        def done(result: tuple[float, str] | None) -> None:
            if result is None:
                self._post(r"[yellow]usage: <0-1> \[comment][/]")
                return
            self.action_score(*result)
        self.push_screen(ScoreDialog(), done)

    def action_workspace(self) -> None:
        if self._turn_running:
            # A turn already handed its workspace to run_pipeline_stream, so
            # changing it now would take effect on the NEXT turn while
            # appearing to have taken effect on this one.
            self._post("[yellow]a turn is still running; wait for it to finish[/]")
            return

        def done(answer: str | None) -> None:
            if answer is None:
                return
            if answer.lower() in {"off", "none", ""}:
                self.session.workspace = None
            elif problem := set_workspace(self.session, answer):
                self._post(f"[red]{problem}[/]")
                return
            self._post(f"[dim]{describe_workspace(self.session.workspace)}[/]", classes="meta")
            self._refresh_chrome()

        self.push_screen(WorkspacePrompt(self.session.workspace), done)

    def action_toggle_usage(self) -> None:
        """ctrl+t. The whole sidebar goes, not just the usage section -- 32
        columns the transcript gets back on a narrow terminal."""
        panel = self.usage_panel
        shown = not panel.display
        panel.display = shown
        self.query_one(Sidebar).display = shown

    def action_new_session(self) -> None:
        if self._turn_running:
            # Resetting mid-turn would swap the session (and its memory
            # queue) out from under a worker that is still writing to it.
            self._post("[yellow]a turn is still running; wait for it to finish[/]")
            return
        self._set_busy(True)
        self._reset_session()

    @work(thread=True)
    def _reset_session(self) -> None:
        """Off the UI thread: `Session.reset()` opens a fresh per-session
        SQLite store (agent/memory/wiring.py's new_history_queue), which is
        disk work, not a field assignment."""
        try:
            self.session.reset()
            self._last_output = None
            # The panel says "this session", so a new session starts it at
            # nothing. Cleared in place rather than rebound: the widget holds
            # the same object the turns record into.
            self.usage.by_model.clear()
            self._turn_tokens.clear()
            self._shown_spend = (0, 0.0)
            self.call_from_thread(self._refresh_usage)
            self.call_from_thread(self._refresh_sidebar)
            self.call_from_thread(self._post, "[dim]new session[/]")
        finally:
            self.call_from_thread(self._set_busy, False)

    # ---- setup, pins ----------------------------------------------------

    def _setup_backend(self) -> SetupBackend:
        """Everything the setup screen may call, as callables resolved at
        call time through THIS module's globals -- so a test that patches
        `agent.cli.tui` patches the screen too."""
        return SetupBackend(
            vendor_rows=lambda: provider_setup.vendor_rows(),
            probe=lambda name: provider_setup.probe(name),
            detected_pool=lambda: provider_setup.detected_pool(),
            resolve=lambda task: self.ctx.router.resolve(task),
            routes=lambda: TASK_ROUTES,
            propose=lambda pool: propose(pool),
            pins=lambda: route_overrides.pins(),
            set_pin=lambda task, spec: route_overrides.set_pin(task, spec),
            clear_pin=lambda task: route_overrides.clear_pin(task),
            set_key=lambda name, value: provider_setup.set_key(name, value),
            set_base_url=lambda name, url: provider_setup.set_base_url(name, url),
            add_endpoint=lambda name, label: provider_setup.add_endpoint(name, label),
            reload=lambda: reload_everything(),
        )

    def action_setup(self) -> None:
        """f2 / "Setup…". Refused mid-turn for the same reason the workspace
        is: a change now would land on the next turn while looking like it
        landed on this one."""
        if self._turn_running:
            self._post("[yellow]a turn is still running; wait for it to finish[/]")
            return
        if isinstance(self.screen, SetupScreen):
            return

        def done(summary: dict | None) -> None:
            if summary is None:
                return
            self._model_cache = None
            bits = []
            if summary.get("pinned"):
                bits.append("pinned " + ", ".join(f"{t} -> {s}" for t, s in summary["pinned"].items()))
            if summary.get("cleared"):
                bits.append("cleared " + ", ".join(summary["cleared"]))
            if summary.get("providers_ok"):
                bits.append("providers ok: " + ", ".join(summary["providers_ok"]))
            self._post(f"[dim]setup: {'; '.join(bits) if bits else 'nothing changed'}[/]", classes="meta")
            for problem in summary.get("problems", []):
                self._post(f"[yellow]{problem}[/]")
            self._refresh_chrome()

        self.push_screen(SetupScreen(self._setup_backend()), done)

    def action_pin_model(self) -> None:
        """"Pin a model for a task…": a task, then a model. Two picks."""
        if self._turn_running:
            self._post("[yellow]a turn is still running; wait for it to finish[/]")
            return

        def on_task(task: Task | None) -> None:
            if task is not None:
                self._open_pin_dialog(task)

        self.push_screen(TaskPicker(), on_task)

    @work(thread=True, exit_on_error=False)
    def _open_pin_dialog(self, task: Task) -> None:
        try:
            models = self._model_cache or sorted(all_models(None), key=lambda m: (m.provider, m.id))
            self._model_cache = models
            options = pin_options(task, models, TASK_ROUTES)
            current = route_overrides.pins().get(task, "")
        except Exception as exc:
            self.call_from_thread(self._post, f"[red]could not list models: {type(exc).__name__}: {exc}[/]")
            return

        def show() -> None:
            self.push_screen(ModelPinDialog(task, options, current),
                             lambda spec: self._apply_pin(task, spec))

        self.call_from_thread(show)

    @work(thread=True, exit_on_error=False)
    def _apply_pin(self, task: Task, spec: str | None) -> None:
        if spec is None:
            return
        try:
            if spec == "":
                route_overrides.clear_pin(task)
                line = f"cleared the pin on {task.value}"
            else:
                route_overrides.set_pin(task, spec)
                line = f"pinned {task.value} -> {spec}"
            problems = reload_everything()
        except Exception as exc:
            self.call_from_thread(self._post, f"[red]could not pin: {type(exc).__name__}: {exc}[/]")
            return
        self.call_from_thread(self._post, f"[dim]{line}[/]", "meta")
        for problem in problems:
            self.call_from_thread(self._post, f"[yellow]{problem}[/]")
        self.call_from_thread(self._refresh_chrome)

    # ---- lessons ----------------------------------------------------------

    def action_show_lessons(self) -> None:
        self._show_lessons()

    @work(thread=True, exit_on_error=False)
    def _show_lessons(self) -> None:
        try:
            learned = all_lessons()
        except Exception as exc:
            self.call_from_thread(self._post, f"[red]could not read the lesson bank: {exc}[/]")
            return
        self.call_from_thread(
            self._post, lessons_table(learned, bank_path()) if learned else "[dim]no lessons yet[/]")

    def action_export_lessons(self) -> None:
        if self._lessons_busy:
            self._post("[yellow]a lessons import/export is still running[/]")
            return
        default = Path.home() / f"otto-lessons-{date.today():%Y-%m-%d}.json"

        def done(path: str | None) -> None:
            if path and path != "off":
                self._export_lessons(Path(path).expanduser())

        self.push_screen(PathPicker(Path.home(), mode="save", title="Export lessons to",
                                    initial_text=str(default)), done)

    @work(thread=True, exit_on_error=False)
    def _export_lessons(self, path: Path) -> None:
        self._lessons_busy = True
        try:
            n = export_lessons(path)
            self.call_from_thread(self._post, f"[dim]exported {n} lesson(s) to {path}[/]")
        except Exception as exc:
            self.call_from_thread(self._post, f"[red]export failed: {type(exc).__name__}: {exc}[/]")
        finally:
            self._lessons_busy = False

    def action_import_lessons(self) -> None:
        if self._lessons_busy:
            self._post("[yellow]a lessons import/export is still running[/]")
            return

        def done(path: str | None) -> None:
            if path and path != "off":
                self._import_lessons(Path(path).expanduser())

        self.push_screen(PathPicker(Path.home(), mode="file", title="Import lessons from",
                                    suffixes={".json", ".md", ".txt"}), done)

    @work(thread=True, exit_on_error=False)
    def _import_lessons(self, path: Path) -> None:
        self._lessons_busy = True
        try:
            report = import_lessons(path)
            self.call_from_thread(self._post, f"[dim]imported lessons from {path}: {report.summary()}[/]")
        except Exception as exc:
            self.call_from_thread(self._post, f"[red]import failed: {type(exc).__name__}: {exc}[/]")
        finally:
            self._lessons_busy = False

    # ---- the message box: the one thing that stays typed ---------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # A modal's own Input.Submitted bubbles all the way up to the App
        # (module docstring, "Modal input must not escape into a new turn").
        # Every modal here stops its own event; this is the second lock on
        # the same door, and the one that still holds if a future modal
        # forgets.
        if event.input.id != "message-input":
            return
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._turn_running:
            # Module docstring, "One turn at a time". Nothing in Textual can
            # stop the in-flight worker thread, so the only safe answer is to
            # not start a second one -- and now that escape ends a turn, to
            # say which key does it.
            self._post(
                "[yellow]still working on the previous message -- esc to stop it[/]")
            return
        self._drop_empty_state()
        self._post(f"[bold]you[/] {text}", classes="user")
        # Taken HERE, on the UI thread, before the worker exists -- a flag
        # the worker set for itself would leave a window wide enough for a
        # second Enter.
        self._set_busy(True)
        self.run_turn(text)

    @work(thread=True, group="turn")
    def run_turn(self, text: str) -> None:
        # No exclusive=True: it cancels the asyncio task wrapping this
        # thread, not the thread, so it never stopped anything and only
        # made the double-run look handled (module docstring, "One turn at
        # a time"). `_turn_running` is what actually serialises turns.
        self._cancel = threading.Event()
        self._phase = "reading your message"
        self._model = self._tool = self._mode = ""
        self._calls = 0
        self._budget_max = default_budget().max_model_calls
        self._started = time.monotonic()
        self._answer = None
        self._drawn_at = 0.0
        self._tokens_before_turn = self.usage.total_tokens
        self.call_from_thread(self._draw_status)
        with bind_progress(self._on_progress, cancel=self._cancel):
            self._drive_turn(text)

    def _drive_turn(self, text: str) -> None:
        tally: Counter = Counter()
        steps = 0
        # One fresh RichLog per turn, mounted OPEN so board lines render as
        # they stream in rather than piling up unrendered behind a collapsed
        # container -- render_update() (shell.py) is unchanged, it just
        # writes into this instead of the old shared transcript RichLog
        # (module docstring).
        thinking_log = SelectableRichLog(wrap=True, markup=True, highlight=False)
        block = self.call_from_thread(self._post_thinking, thinking_log, "thinking…")
        human_message = HumanMessage(text)
        # Bounded, not the raw ever-growing list (agent/cli/chat.py's own
        # module docstring has the full Phase 2 reasoning -- shared 1:1
        # with this TUI, both front ends over the same Session).
        history, memory_context = self.session.history_for_graph()
        try:
            stream = run_pipeline_stream(
                text, session_id=self.session.session_id, history=history,
                memory_context=memory_context, workspace=self.session.workspace_arg(),
                usage=self.usage,
            )
            while stream is not None:
                next_stream = None
                for update in stream:
                    if "__ask__" in update:
                        # A specialist or the evaluator got stuck --
                        # module docstring, seventh refinement. Blocks
                        # THIS worker thread (not the UI) until answered,
                        # then resumes the SAME run on the same graph
                        # thread -- not a fresh turn.
                        ask = update["__ask__"]
                        answer = self._ask_user_blocking(ask["question"], ask["choices"])
                        self.call_from_thread(self._post, f"[bold]you[/] {answer}", "user")
                        # Breaking out leaves this generator suspended at its
                        # yield, inside bind_budget/bind_store/bind_workspace,
                        # so their contextvar tokens would be reset from
                        # whatever context the GC runs in rather than this
                        # worker thread. close() unwinds it here instead.
                        stream.close()
                        next_stream = resume_pipeline_stream(
                            answer, thread_id=ask["thread_id"],
                            session_id=self.session.session_id,
                            workspace=self.session.workspace_arg(),
                            usage=self.usage,
                        )
                        break
                    if "__final__" in update:
                        final = update["__final__"]
                        self.session.trace_id = update.get("__trace_id__")
                        raw_output = (final.get("final_output") or "").strip()
                        code = raw_output or "*(no output produced)*"
                        self._last_output = raw_output or None
                        # Replaces the streamed block in place when there is
                        # one. Posting a second block would leave the same
                        # answer on screen twice, once raw and once rendered.
                        self.call_from_thread(self._settle_answer, code)
                        self.session.record_turn(human_message, AIMessage(code) if raw_output else None)
                        if raw_output:
                            # Same reasoning as chat.py: a file survives copying,
                            # a live RichLog selection does not.
                            self.session.turn += 1
                            path = save_final(self.session.session_id, self.session.turn, raw_output, None)
                            self.call_from_thread(self._post, f"[dim]saved to {path}[/]", "meta")
                        continue

                    node, delta = next(iter(update.items()))
                    steps += 1
                    # A mode switch is the one board line worth a glyph.
                    for line in (delta.get("board", []) if isinstance(delta, dict) else []):
                        mode = art.mode_from_board_line(str(line))
                        if mode and mode != self._mode:
                            self._mode = mode
                            self.call_from_thread(self._retitle_thinking, block)
                    self.call_from_thread(render_update, node, delta, tally, thinking_log.write)
                    self.call_from_thread(self._refresh_usage)
                stream = next_stream
        except Cancelled:
            # Asked for, not broken. Its own clause so a person who pressed
            # escape is not shown a traceback for something they chose.
            self.call_from_thread(self._post, "[yellow]stopped[/]")
        except AuthError as exc:
            # No key for the one provider otto cannot run without. Point at
            # the screen that fixes it rather than at a traceback.
            self.call_from_thread(self._post, f"[red]{exc}[/] [dim]-- f2 opens Setup[/]")
        except Exception as exc:  # a provider error mid-turn must not crash the app
            self.call_from_thread(self._post, f"[red]{type(exc).__name__}: {exc}[/]")
        finally:
            # All three in one finally: a turn that died on a provider error
            # still has to hand the session back, or the message box stays
            # disabled and the app looks hung for the rest of its life -- and
            # it still spent tokens getting there. The per-update refresh
            # above is what makes the panel move DURING a turn; this is what
            # makes it right at the end of one, including a turn whose only
            # event was its own answer.
            self.call_from_thread(self._close_thinking, block, steps,
                                  self._calls, time.monotonic() - self._started)
            self._last_model = self._model or self._last_model
            self._turn_tokens.append(max(0, self.usage.total_tokens - self._tokens_before_turn))
            self.call_from_thread(self._stop_caret)
            self.call_from_thread(self._refresh_usage)
            self.call_from_thread(self._refresh_sidebar)
            self.call_from_thread(self._set_busy, False)

    def _settle_answer(self, text: str) -> None:
        """Turn the streamed block into the finished one, or mount it if the
        answer never streamed (a run that died, or one whose reply arrived in
        a single chunk). Animated, the header's bullet sparkles for a few
        frames first and settles on the plain ● every answer wears."""
        widget = self._answer
        if widget is None:
            widget = CopyableStatic("", classes="answer")
            self.transcript.mount(widget)
        self._answer = None
        self._stop_caret()
        try:
            self.query_one(Sidebar).query_one(Banner).shimmer()
        except Exception:
            pass
        body = Markdown(text)
        frames = [self._answer_block(body, glyph, style, "final") for glyph, style in art.SPARKLE_FRAMES]
        if not animations_enabled(self):
            widget.update(frames[-1])
            self.transcript.scroll_end(animate=False)
            return
        state = {"i": 1, "timer": None}

        def step() -> None:
            i = state["i"]
            widget.update(frames[i])
            state["i"] = i + 1
            if state["i"] >= len(frames) and state["timer"] is not None:
                state["timer"].stop()

        widget.update(frames[0])
        self.transcript.scroll_end(animate=False)
        state["timer"] = self.set_interval(art.SPARKLE_EVERY, step)


def tui(
    ctx: typer.Context,
    workspace: Annotated[Optional[Path], typer.Option("--workspace", "-w", help=WORKSPACE_HELP)] = None,
    no_workspace: Annotated[bool, typer.Option("--no-workspace", help=NO_WORKSPACE_HELP)] = False,
) -> None:
    """Launch the full-screen TUI: the arrow-key-menu front end over the same
    pipeline `otto chat` drives."""
    OttoApp(ctx.obj, workspace=resolve_workspace(workspace, no_workspace)).run()
