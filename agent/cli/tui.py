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

class OttoApp(App):
    TITLE = "otto"
    BINDINGS = [
        ("ctrl+n", "new_session", "New session"),
        ("ctrl+y", "copy_last", "Copy last answer"),
    ]
    DEFAULT_CSS = """
    #transcript { height: 1fr; }
    #transcript Collapsible { padding: 0; }
    .thinking-log { height: 12; }
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

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="transcript")
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

    def _post_thinking(self, thinking_log: RichLog, title: str) -> None:
        """Mount one *thinking*-side block: a fresh RichLog, collapsed by
        default, that render_update() writes this turn's board lines and
        output previews into (module docstring). Collapsed rather than
        omitted -- the process is still one click away, just not what the
        user sees by default.
        """
        thinking_log.add_class("thinking-log")
        self.transcript.mount(Collapsible(thinking_log, title=title, collapsed=True))
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
        self._post(f"[bold]you[/] {text}")
        self.run_turn(text)

    @work(thread=True, exclusive=True, group="turn")
    def run_turn(self, text: str) -> None:
        tally: Counter = Counter()
        # One fresh RichLog per turn, mounted collapsed right away so board
        # lines have somewhere to land as they stream in -- render_update()
        # (shell.py) is unchanged, it just writes into this instead of the
        # old shared transcript RichLog (module docstring).
        thinking_log = RichLog(wrap=True, markup=True, highlight=False)
        self.call_from_thread(self._post_thinking, thinking_log, "thinking…")
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
                    self.call_from_thread(render_update, node, delta, tally, thinking_log.write)
                stream = next_stream
        except Exception as exc:  # a provider error mid-turn must not crash the app
            self.call_from_thread(self._post, f"[red]{type(exc).__name__}: {exc}[/]")


def tui(ctx: typer.Context) -> None:
    """Launch the full-screen TUI: the arrow-key-menu front end over the same
    pipeline `otto chat` drives."""
    OttoApp(ctx.obj).run()
