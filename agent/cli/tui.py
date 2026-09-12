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
RichLog created FOR THAT TURN, wrapped in a collapsed-by-default
`Collapsible`, so it never crowds the screen but is one click away.
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
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Iterable

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import RenderableType
from rich.markdown import Markdown
from rich.panel import Panel
from textual import work
from textual.app import App, ComposeResult, SystemCommand
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Collapsible, Footer, Header, Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from agent.cli.context import AppContext
from agent.cli.output import save_final
from agent.cli.shell import Session, render_update
from agent.cli.ui import THEME
from agent.pipeline.progress import Cancelled, Progress, bind_progress
from agent.pipeline.run import resume_pipeline_stream, run_pipeline_stream
from agent.router.mapping import Task
from agent.router.router import NoViableRoute


# --------------------------------------------------------------------------
# Modals -- every argument that isn't the message itself is a pick, not typed
# --------------------------------------------------------------------------

class TaskPicker(ModalScreen[Task | None]):
    """Arrow-key list for the /route equivalent's task argument."""

    DEFAULT_CSS = """
    TaskPicker { align: center middle; }
    TaskPicker > OptionList { width: 40; height: auto; border: round $accent; }
    """

    def compose(self) -> ComposeResult:
        yield OptionList(*[Option(t.value, id=t.value) for t in Task])

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(Task(event.option.id))


class ScoreDialog(ModalScreen[tuple[float, str] | None]):
    """The one score command that needs a value: everything else here is a
    pick, this one number genuinely isn't."""

    DEFAULT_CSS = """
    ScoreDialog { align: center middle; }
    ScoreDialog > Input { width: 50; }
    """

    def compose(self) -> ComposeResult:
        yield Input(placeholder="0.0-1.0 [comment], Enter to submit, Esc to cancel")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value_text, _, comment = event.value.strip().partition(" ")
        try:
            value = float(value_text)
        except ValueError:
            self.dismiss(None)
            return
        self.dismiss((value, comment.strip()))

    def key_escape(self) -> None:
        self.dismiss(None)


class AskUserModal(ModalScreen[str]):
    """"Widget with multi choice + text bar" (module docstring, seventh
    refinement design call) -- what a paused run's question/choices
    (agent/pipeline/nodes.py's ask_user node, via `{"__ask__": ...}`) is
    shown through. The OptionList is only mounted when there ARE choices
    (an empty list means a genuinely open-ended question); the Input is
    always there, so free text is always an option even alongside choices.
    No Esc-to-cancel here, unlike the other two modals -- the graph really
    is paused waiting on an answer, there's no "nevermind" that doesn't
    leave the run stuck; an empty Enter is treated as a (weak) "no
    preference, continue" rather than dismissed.
    """

    DEFAULT_CSS = """
    AskUserModal { align: center middle; }
    AskUserModal > Vertical { width: 70; height: auto; max-height: 80%; border: round $warning; padding: 1 2; }
    AskUserModal .question { margin-bottom: 1; }
    AskUserModal OptionList { height: auto; max-height: 10; margin-bottom: 1; }
    """

    def __init__(self, question: str, choices: list[str]) -> None:
        super().__init__()
        self._question = question
        self._choices = choices

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(Markdown(f"**otto is asking:**\n\n{self._question}"), classes="question")
            if self._choices:
                yield OptionList(*[Option(c) for c in self._choices])
            yield Input(placeholder="type your answer, Enter to submit")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.prompt))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value.strip())


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

#: Seconds between answer frames that actually cross to the UI thread.
#: Chunks arrive an order of magnitude faster than anyone reads, and
#: marshalling every one of them spends the UI thread drawing frames nobody
#: sees. A tenth of a second still reads as continuous typing.
_STREAM_EVERY = 0.1

#: Frames for the one-character spinner in the status line. Four is enough to
#: read as motion and short enough that a stalled run is obvious -- a frozen
#: spinner says "stuck", a missing one says nothing.
_SPINNER = "|/-\\"


def _clock(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class OttoApp(App):
    TITLE = "otto"
    BINDINGS = [
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+y", "copy_last", "Copy last answer"),
        ("escape", "stop_turn", "Stop this turn"),
    ]
    DEFAULT_CSS = """
    #transcript { height: 1fr; }
    #transcript Collapsible { padding: 0; }
    .thinking-log { height: 12; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    """

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.session = Session(ctx=ctx)
        #: The last turn's raw final_output (module docstring, "Copying
        #: cleanly") -- exactly the string save_final() wrote to disk, no
        #: Panel/Markdown wrapper. None before any turn has finished, or
        #: after one that produced nothing.
        self._last_output: str | None = None
        #: Everything the status line draws, written from the worker thread
        #: and read from the UI thread's one-second tick. Plain attributes
        #: rather than a lock: each is a single assignment of an immutable
        #: value, and the worst a torn read can do is show one stale field
        #: for a tenth of a second.
        self._busy = False
        self._phase = ""
        self._model = ""
        self._tool = ""
        self._calls = 0
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
        #: This turn's collapsed thinking row, so it can be retitled with
        #: what the turn cost once the turn is over.
        self._thinking: Collapsible | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Static("", id="status")
        yield Input(placeholder="type a message… (ctrl+p for commands)", id="message-input")
        yield Footer()

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
        self.query_one(Input).focus()
        self._post("[dim]otto:pipeline[/]")
        # The clock has to tick on its own: between two model calls nothing
        # reports anything for ten seconds at a stretch, and a status line
        # that only moves when the run moves reads as a frozen app.
        self.set_interval(1.0, self._tick)

    # ---- the status line ----------------------------------------------

    def _tick(self) -> None:
        """Once a second, on the UI thread. Redraws the status line from
        whatever the worker thread last wrote, so the elapsed clock and the
        spinner keep moving through a long model call."""
        if not self._busy:
            return
        self._frame += 1
        self._draw_status()

    def _draw_status(self) -> None:
        if not self._busy:
            self.query_one("#status", Static).update("")
            return
        parts = [f"[bold]{_SPINNER[self._frame % len(_SPINNER)]}[/]"]
        if self._phase:
            parts.append(self._phase)
        if self._tool:
            parts.append(f"[bold]{self._tool}[/]")
        if self._model:
            parts.append(f"[dim]{self._model}[/]")
        if self._calls:
            parts.append(f"[dim]{self._calls} calls[/]")
        parts.append(f"[dim]{_clock(time.monotonic() - self._started)}[/]")
        parts.append("[dim]esc to stop[/]")
        self.query_one("#status", Static).update(" · ".join(parts))

    def _on_progress(self, update: Progress) -> None:
        """The progress sink, called on the WORKER thread. Does as little as
        possible here and marshals the rest: everything below writes plain
        attributes, and the once-a-second tick on the UI thread is what turns
        them into pixels. The one exception is a streamed answer, which has
        to mount and update a widget and so goes through call_from_thread --
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

    def _show_partial_answer(self, text: str) -> None:
        if self._answer is None:
            self._answer = Static("")
            self.transcript.mount(self._answer)
        self._answer.update(Panel(text, title="[dim]answering…[/]", border_style="dim"))
        self.transcript.scroll_end(animate=False)

    def action_stop_turn(self) -> None:
        """`escape`. Cooperative: agent/pipeline/progress.py checks this
        before every model request, so the stop lands within one call and the
        run never pays for another. Saying so matters -- a key that looks
        like it did nothing for ten seconds is worse than no key."""
        if not self._busy or self._cancel is None:
            return
        self._cancel.set()
        self._phase = "stopping after this call"
        self._draw_status()

    @property
    def transcript(self) -> VerticalScroll:
        return self.query_one("#transcript", VerticalScroll)

    def _post(self, renderable: RenderableType) -> None:
        """Mount one *result*-side block: always visible, never collapsed
        (module docstring). `renderable` is anything Static accepts -- a
        markup string using only primitive Rich style words, or a Rich
        renderable like Panel/Table. Safe to call from the UI thread
        directly; a worker thread must go through `self.call_from_thread`
        the same way it already does for everything else that touches the
        widget tree.
        """
        self.transcript.mount(Static(renderable))
        self.transcript.scroll_end(animate=False)

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

    def _post_thinking(self, thinking_log: RichLog, title: str) -> Collapsible:
        """Mount one *thinking*-side block: a fresh RichLog, collapsed by
        default, that render_update() writes this turn's board lines and
        output previews into (module docstring). Collapsed rather than
        omitted -- the process is still one click away, just not what the
        user sees by default.

        Returns the Collapsible so the caller can retitle it when the turn
        ends: a row still reading "thinking…" an hour later says nothing
        about which turn it belongs to, and a transcript of several says it
        several times.
        """
        thinking_log.add_class("thinking-log")
        block = Collapsible(thinking_log, title=title, collapsed=True)
        self.transcript.mount(block)
        self.transcript.scroll_end(animate=False)
        return block

    # ---- the command palette (ctrl+p): the arrow-key menu ------------

    def get_system_commands(self, screen) -> Iterable[SystemCommand]:
        yield from super().get_system_commands(screen)
        yield SystemCommand("Route a task…", "Show how a task resolves, without spending a turn", self.action_pick_route)
        yield SystemCommand("List models", "List every configured model", self.action_models)
        yield SystemCommand("Check providers", "Run otto doctor", self.action_doctor)
        yield SystemCommand("Rate last answer: good", "Score the last answer 1.0", lambda: self.action_score(1.0, ""))
        yield SystemCommand("Rate last answer: bad", "Score the last answer 0.0", lambda: self.action_score(0.0, ""))
        yield SystemCommand("Rate last answer…", "Score the last answer with a value and a comment", self.action_score_dialog)
        yield SystemCommand("Copy last answer", "Copy the raw final answer to your clipboard", self.action_copy_last)
        yield SystemCommand("New session", "Clear history, start fresh", self.action_new_session)

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
        self.copy_to_clipboard(self._last_output)
        self._post(
            "[dim]copied the last answer to your clipboard (raw text, not "
            "the panel) -- if nothing pasted, your terminal may not "
            "support this (e.g. macOS Terminal.app doesn't); the saved-to "
            "path above always works[/]"
        )

    def action_pick_route(self) -> None:
        def done(task: Task | None) -> None:
            if task is not None:
                self._route_task(task)
        self.push_screen(TaskPicker(), done)

    @work(thread=True)
    def _route_task(self, task: Task) -> None:
        try:
            d = self.ctx.router.resolve(task)
        except NoViableRoute as exc:
            self.call_from_thread(self._post, f"[red]no viable model for {task.value}: {exc}[/]")
            return
        why = (" (degraded)" if d.fell_back else
               " (on observed results)" if d.chosen_on_evidence else "")
        line = f"{task.value} -> {d.provider}:{d.model.id}{why}"
        self.call_from_thread(self._post, f"[dim]{line}[/]")

    @work(thread=True)
    def action_models(self) -> None:
        from agent.cli.models import models_table
        from agent.router.llm_provider import all_models
        found = sorted(all_models(None), key=lambda m: (m.provider, m.id))
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
        self.ctx.client.create_score(
            name="user_feedback", value=value, data_type="NUMERIC",
            trace_id=self.session.trace_id, comment=comment or None,
        )
        self.ctx.client.flush()
        self._post(f"[dim]scored {value:g}[/]")

    def action_score_dialog(self) -> None:
        def done(result: tuple[float, str] | None) -> None:
            if result is None:
                self._post(r"[yellow]usage: <0-1> \[comment][/]")
                return
            self.action_score(*result)
        self.push_screen(ScoreDialog(), done)

    def action_new_session(self) -> None:
        self.session.reset()
        self._last_output = None
        self._post("[dim]new session[/]")

    # ---- the message box: the one thing that stays typed ---------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if self._busy:
            # `exclusive=True` cancels the previous WORKER, and a worker
            # running a thread cannot be cancelled out from under a blocking
            # network call -- so a second submit used to leave two runs alive
            # writing into the same transcript. Refuse, and say which key
            # ends the one already going.
            self._post("[yellow]a turn is already running -- esc to stop it[/]")
            return
        self._post(f"[bold]you[/] {text}")
        self.run_turn(text)

    @work(thread=True, exclusive=True, group="turn")
    def run_turn(self, text: str) -> None:
        self._cancel = threading.Event()
        self._busy = True
        self._phase = "reading your message"
        self._model = self._tool = ""
        self._calls = 0
        self._started = time.monotonic()
        self._answer = None
        self._drawn_at = 0.0
        self.call_from_thread(self._draw_status)
        try:
            with bind_progress(self._on_progress, cancel=self._cancel):
                self._run_turn(text)
        finally:
            self._busy = False
            elapsed = time.monotonic() - self._started
            self.call_from_thread(self._draw_status)
            self.call_from_thread(self._retitle_thinking, self._calls, elapsed)

    def _run_turn(self, text: str) -> None:
        tally: Counter = Counter()
        # One fresh RichLog per turn, mounted collapsed right away so board
        # lines have somewhere to land as they stream in -- render_update()
        # (shell.py) is unchanged, it just writes into this instead of the
        # old shared transcript RichLog (module docstring).
        thinking_log = RichLog(wrap=True, markup=True, highlight=False)
        self._thinking = self.call_from_thread(
            self._post_thinking, thinking_log, "thinking…")
        human_message = HumanMessage(text)
        # Bounded, not the raw ever-growing list (agent/cli/chat.py's own
        # module docstring has the full Phase 2 reasoning -- shared 1:1
        # with this TUI, both front ends over the same Session).
        history, memory_context = self.session.history_for_graph()
        try:
            stream = run_pipeline_stream(
                text, session_id=self.session.session_id, history=history, memory_context=memory_context,
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
                        self.call_from_thread(self._post, f"[bold]you[/] {answer}")
                        next_stream = resume_pipeline_stream(
                            answer, thread_id=ask["thread_id"], session_id=self.session.session_id,
                        )
                        break
                    if "__final__" in update:
                        final = update["__final__"]
                        self.session.trace_id = update.get("__trace_id__")
                        raw_output = (final.get("final_output") or "").strip()
                        code = raw_output or "*(no output produced)*"
                        self._last_output = raw_output or None
                        # Replaces the streamed block in place when there is
                        # one. Posting a second panel would leave the same
                        # answer on screen twice, once raw and once rendered.
                        self.call_from_thread(
                            self._settle_answer,
                            Panel(Markdown(code), title="[green]final[/]", border_style="green"),
                        )
                        self.session.record_turn(human_message, AIMessage(code) if raw_output else None)
                        if raw_output:
                            # Same reasoning as chat.py: a file survives copying,
                            # a live RichLog selection does not.
                            self.session.turn += 1
                            path = save_final(self.session.session_id, self.session.turn, raw_output, None)
                            self.call_from_thread(self._post, f"[dim]saved to {path}[/]")
                        continue

                    node, delta = next(iter(update.items()))
                    self.call_from_thread(render_update, node, delta, tally, thinking_log.write)
                stream = next_stream
        except Cancelled:
            # Asked for, not broken. Its own clause so a person who pressed
            # escape is not shown a traceback for something they chose.
            self.call_from_thread(self._post, "[yellow]stopped[/]")
        except Exception as exc:  # a provider error mid-turn must not crash the app
            self.call_from_thread(self._post, f"[red]{type(exc).__name__}: {exc}[/]")

    def _retitle_thinking(self, calls: int, elapsed: float) -> None:
        """What that turn cost, on the row that holds how it was spent. The
        one number a person wants after the fact is on the collapsed row, so
        reading it back does not mean expanding anything."""
        if self._thinking is not None:
            self._thinking.title = f"{calls} model calls · {_clock(elapsed)}"
            self._thinking = None

    def _settle_answer(self, panel: Panel) -> None:
        """Turn the streamed block into the finished one, or mount it if the
        answer never streamed (a run that died, or one whose reply arrived in
        a single chunk)."""
        if self._answer is not None:
            self._answer.update(panel)
            self._answer = None
            self.transcript.scroll_end(animate=False)
            return
        self._post(panel)


def tui(ctx: typer.Context) -> None:
    """Launch the full-screen TUI: the arrow-key-menu front end over the same
    pipeline `otto chat` drives."""
    OttoApp(ctx.obj).run()
