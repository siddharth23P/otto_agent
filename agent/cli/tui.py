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

`.thinking-log` is `height: auto` with a `max-height`, not a fixed
`height: 12` (2026-09-12, same screenshot: a two-line log drawn as a
twelve-row box, ten of them blank). RichLog is a ScrollView and this
module's own docstring above says a ScrollView fills rather than sizes to
content -- true for `height: auto` alone, but `auto` capped by
`max-height` measures correctly in Textual 8.2.8 (checked live: 3 lines
gave a height of 3, 43 lines gave 16).
"""

from __future__ import annotations

import threading
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
        event.stop()
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
        # Module docstring, "Modal input must not escape into a new turn":
        # without this the App's own on_input_submitted also fires and runs
        # the pipeline on the literal string "0.8 nice".
        event.stop()
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
        event.stop()
        self.dismiss(str(event.option.prompt))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Module docstring, "Modal input must not escape into a new turn":
        # without this the answer is posted twice and ALSO started as a
        # brand-new turn running alongside the paused one it answers.
        event.stop()
        self.dismiss(event.value.strip())


# --------------------------------------------------------------------------
# The app
# --------------------------------------------------------------------------

class OttoApp(App):
    TITLE = "otto"
    BINDINGS = [
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+y", "copy_last", "Copy last answer"),
    ]
    DEFAULT_CSS = """
    #transcript { height: 1fr; }
    #transcript Collapsible { padding: 0; }
    .thinking-log { height: auto; max-height: 16; }
    """

    #: What the message box says when it is free. Restored by `_set_busy`,
    #: which swaps in a "still working" placeholder for as long as a turn
    #: owns the session -- the disabled box plus this line are the only
    #: thing that ever tells the user a turn is still going.
    IDLE_PLACEHOLDER = "type a message… (ctrl+p for commands)"
    BUSY_PLACEHOLDER = "working… one turn at a time"

    def __init__(self, ctx: AppContext) -> None:
        super().__init__()
        self.ctx = ctx
        self.session = Session(ctx=ctx)
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
        #: Enter to slip through.
        self._turn_running = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
        yield Input(placeholder=self.IDLE_PLACEHOLDER, id="message-input")
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
        self.message_box.focus()
        self._post("[dim]otto:pipeline[/]")

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
        box = self.message_box
        box.disabled = busy
        box.placeholder = self.BUSY_PLACEHOLDER if busy else self.IDLE_PLACEHOLDER
        self.sub_title = "working…" if busy else ""
        if not busy:
            box.focus()

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
        return block

    def _close_thinking(self, block: Collapsible, steps: int) -> None:
        """Shut this turn's thinking block now that the answer is on screen,
        labelled with how much is folded away inside it so it is obvious
        there is something to open."""
        block.title = f"thinking… ({steps} steps)" if steps else "thinking…"
        block.collapsed = True
        self.transcript.scroll_end(animate=False)

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
            self.call_from_thread(self._post, "[dim]new session[/]")
        finally:
            self.call_from_thread(self._set_busy, False)

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
            # stop the in-flight worker thread, so the only safe answer is
            # to not start a second one.
            self._post("[yellow]still working on the previous message[/]")
            return
        self._post(f"[bold]you[/] {text}")
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
        tally: Counter = Counter()
        steps = 0
        # One fresh RichLog per turn, mounted OPEN so board lines render as
        # they stream in rather than piling up unrendered behind a collapsed
        # container -- render_update() (shell.py) is unchanged, it just
        # writes into this instead of the old shared transcript RichLog
        # (module docstring).
        thinking_log = RichLog(wrap=True, markup=True, highlight=False)
        block = self.call_from_thread(self._post_thinking, thinking_log, "thinking…")
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
                        self.call_from_thread(
                            self._post,
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
                    steps += 1
                    self.call_from_thread(render_update, node, delta, tally, thinking_log.write)
                stream = next_stream
        except Exception as exc:  # a provider error mid-turn must not crash the app
            self.call_from_thread(self._post, f"[red]{type(exc).__name__}: {exc}[/]")
        finally:
            # Both in one finally: a turn that died on a provider error still
            # has to hand the session back, or the message box stays disabled
            # and the app looks hung for the rest of its life.
            self.call_from_thread(self._close_thinking, block, steps)
            self.call_from_thread(self._set_busy, False)


def tui(ctx: typer.Context) -> None:
    """Launch the full-screen TUI: the arrow-key-menu front end over the same
    pipeline `otto chat` drives."""
    OttoApp(ctx.obj).run()
