"""The TUI's modal screens (split out of agent/cli/tui.py on 2026-09-12).

Every argument that is not the message itself is a pick, not typed -- tui.py's
own docstring has the design call. The one rule every modal here shares, and
the reason they live together: any `on_input_submitted` handler calls
`event.stop()` FIRST. `Input.Submitted` bubbles all the way up to the App,
whose own handler would otherwise start a pipeline turn on the modal's text
(tests/test_tui.py pins this against the installed Textual).

`PathPicker` (2026-09-12, design call: "dir browser for workspace selection")
is the directory browser: a `DirectoryTree` beside a path `Input` that stays
in sync with the highlighted node, so a person can arrow through the tree OR
type a path and have the tree follow. `WorkspacePrompt` is a thin subclass
with the same name and the same Input contract it had as a bare prompt, so
`action_workspace` and its tests did not have to change.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Literal

from rich.markdown import Markdown
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, OptionList, Select, Static, Tree
from textual.widgets._select import InvalidSelectValueError
from textual.widgets.option_list import Option

from agent.router.mapping import Task

__all__ = [
    "TaskPicker", "ScoreDialog", "AskUserModal",
    "PathPicker", "WorkspacePrompt", "ModelPinDialog", "FilteredDirectoryTree",
]


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

    def key_escape(self) -> None:
        self.dismiss(None)


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
        # Module docstring: without this the App's own on_input_submitted
        # also fires and runs the pipeline on the literal string "0.8 nice".
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
    """"Widget with multi choice + text bar" (tui.py's docstring, seventh
    refinement design call) -- what a paused run's question/choices
    (agent/pipeline/nodes.py's ask_user node, via `{"__ask__": ...}`) is
    shown through. The OptionList is only mounted when there ARE choices
    (an empty list means a genuinely open-ended question); the Input is
    always there, so free text is always an option even alongside choices.
    No Esc-to-cancel here, unlike the other modals -- the graph really is
    paused waiting on an answer, there's no "nevermind" that doesn't leave
    the run stuck; an empty Enter is treated as a (weak) "no preference,
    continue" rather than dismissed.
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
        # Module docstring: without this the answer is posted twice and ALSO
        # started as a brand-new turn running alongside the paused one.
        event.stop()
        self.dismiss(event.value.strip())


# --------------------------------------------------------------------------
# The directory browser
# --------------------------------------------------------------------------

PickMode = Literal["dir", "file", "save"]


class FilteredDirectoryTree(DirectoryTree):
    """A DirectoryTree that shows directories (and, in `file` mode, files with
    one of `suffixes`), hiding dot-entries unless asked. `filter_paths` is
    consulted on every directory load, so flipping `show_hidden` needs a
    `reload()` -- checked against Textual 8.2.8."""

    def __init__(self, path: Path, *, mode: PickMode, suffixes: frozenset[str],
                 show_hidden: bool = False, **kwargs) -> None:
        self.mode = mode
        self.suffixes = suffixes
        self.show_hidden = show_hidden
        super().__init__(path, **kwargs)

    def filter_paths(self, paths: Iterable[Path]) -> Iterable[Path]:
        for p in paths:
            if not self.show_hidden and p.name.startswith("."):
                continue
            try:
                is_dir = p.is_dir()
            except OSError:
                continue
            if is_dir:
                yield p
            elif self.mode == "file" and (not self.suffixes or p.suffix.lower() in self.suffixes):
                yield p


class PathPicker(ModalScreen[str | None]):
    """Browse to a directory (or a file) with the arrow keys, or type a path
    and watch the tree follow. Dismisses with the chosen path as a string,
    "off" when `allow_off` and the person asked for no file access, or None
    on cancel. Validation of the result belongs to the caller: `set_workspace`
    is the one place that decides what a usable directory is.

    Modes:
      dir   directories only; Enter on a node, the Use button, or Enter in
            the Input commits.
      file  directories plus files matching `suffixes`; Enter on a file
            commits, Enter on a directory just expands it.
      save  directories only; Enter on a directory rewrites the Input to
            <that dir>/<basename typed so far>, Enter in the Input commits
            whatever is typed, existing or not.
    """

    DEFAULT_CSS = """
    PathPicker { align: center middle; }
    PathPicker > Vertical { width: 90%; max-width: 100; height: 85%; max-height: 40;
                            border: round $accent; padding: 1 2; }
    PathPicker .hint { height: auto; margin-bottom: 1; }
    PathPicker #path { margin-bottom: 1; }
    PathPicker #tree { height: 1fr; border: round $panel-lighten-2; }
    PathPicker .buttons { height: auto; margin-top: 1; }
    PathPicker .buttons Button { margin-right: 1; min-width: 8; }
    PathPicker #problem { height: 1; color: $error; }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("ctrl+r", "go_up", "Up one level"),
        Binding("ctrl+g", "go_home", "Home"),
        Binding("ctrl+o", "toggle_hidden", "Hidden files"),
    ]

    #: Seconds of typing quiet before the tree follows the Input.
    REROOT_AFTER = 0.25

    def __init__(self, start: Path | None, *, mode: PickMode = "dir",
                 title: str = "Choose a directory", allow_off: bool = False,
                 initial_text: str | None = None,
                 suffixes: Iterable[str] = ()) -> None:
        super().__init__()
        self._start = self._existing_dir(start)
        self._mode: PickMode = mode
        self._title = title
        self._allow_off = allow_off
        self._initial = initial_text if initial_text is not None else str(self._start)
        self._suffixes = frozenset(s.lower() for s in suffixes)
        self._syncing = False
        self._reroot_timer = None

    @staticmethod
    def _existing_dir(start: Path | None) -> Path:
        p = Path(start).expanduser() if start is not None else Path.cwd()
        while not p.is_dir() and p != p.parent:
            p = p.parent
        return p

    # ---- widgets ----------------------------------------------------------

    def compose(self) -> ComposeResult:
        what = {"dir": "Use this directory", "file": "Use this file", "save": "Save here"}[self._mode]
        with Vertical():
            yield Static(
                f"[bold]{self._title}[/]\n[dim]Enter chooses · space expands · ctrl+r up · "
                f"ctrl+g home · ctrl+o hidden · Esc cancels"
                + (" · type [bold]off[/] for no file access" if self._allow_off else "") + "[/]",
                classes="hint",
            )
            yield Input(value=self._initial, placeholder="path", id="path")
            yield FilteredDirectoryTree(self._start, mode=self._mode, suffixes=self._suffixes, id="tree")
            with Horizontal(classes="buttons"):
                yield Button(what, id="use", variant="primary")
                yield Button("Up", id="up")
                yield Button("Home", id="home")
                yield Button("Hidden: off", id="hidden")
                if self._allow_off:
                    yield Button("Off", id="off")
                yield Button("Cancel", id="cancel")
            yield Static("", id="problem")

    def on_mount(self) -> None:
        self.query_one("#path", Input).focus()

    @property
    def tree(self) -> FilteredDirectoryTree:
        return self.query_one("#tree", FilteredDirectoryTree)

    @property
    def path_input(self) -> Input:
        return self.query_one("#path", Input)

    # ---- keeping the Input and the tree in step ----------------------------

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        data = getattr(event.node, "data", None)
        path = getattr(data, "path", None)
        if path is None:
            return
        self._syncing = True
        try:
            if self._mode == "save" and Path(path).is_dir():
                self.path_input.value = str(Path(path) / Path(self.path_input.value or "export.json").name)
            else:
                self.path_input.value = str(path)
        finally:
            self._syncing = False

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id != "path" or self._syncing:
            return
        if self._reroot_timer is not None:
            self._reroot_timer.stop()
        self._reroot_timer = self.set_timer(self.REROOT_AFTER, self._maybe_reroot)

    def _maybe_reroot(self) -> None:
        text = self.path_input.value.strip()
        if not text or text.lower() in {"off", "none"}:
            return
        candidate = Path(text).expanduser()
        if self._mode == "save" and not candidate.is_dir():
            candidate = candidate.parent
        try:
            if candidate.is_dir() and candidate.resolve() != Path(self.tree.path).resolve():
                self.tree.path = candidate
        except OSError:
            return

    # ---- committing ------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        # Module docstring: FIRST, or the App starts a turn on the path.
        event.stop()
        if event.input.id != "path":
            return
        text = event.value.strip()
        if self._allow_off and text.lower() in {"off", "none", ""}:
            self.dismiss("off")
            return
        self.dismiss(text)

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        event.stop()
        if self._mode == "dir":
            self.dismiss(str(event.path))
        elif self._mode == "save":
            self._syncing = True
            try:
                self.path_input.value = str(Path(event.path) / Path(self.path_input.value or "export.json").name)
            finally:
                self._syncing = False
        # "file": the tree already expanded it; nothing to commit.

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        event.stop()
        if self._mode == "file":
            self.dismiss(str(event.path))

    def _commit_highlighted(self) -> None:
        node = self.tree.cursor_node
        path = getattr(getattr(node, "data", None), "path", None)
        if self._mode == "save" or path is None:
            self.on_input_submitted(Input.Submitted(self.path_input, self.path_input.value))
            return
        if self._mode == "file" and Path(path).is_dir():
            self.query_one("#problem", Static).update("pick a file, not a directory")
            return
        self.dismiss(str(path))

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        bid = event.button.id
        if bid == "use":
            self._commit_highlighted()
        elif bid == "up":
            await self.action_go_up()
        elif bid == "home":
            await self.action_go_home()
        elif bid == "hidden":
            await self.action_toggle_hidden()
        elif bid == "off":
            self.dismiss("off")
        elif bid == "cancel":
            self.dismiss(None)

    # ---- bindings ----------------------------------------------------------

    def action_cancel(self) -> None:
        self.dismiss(None)

    async def _reroot(self, path: Path) -> None:
        self.tree.path = path
        self._syncing = True
        try:
            if self._mode == "save":
                self.path_input.value = str(path / Path(self.path_input.value or "export.json").name)
            else:
                self.path_input.value = str(path)
        finally:
            self._syncing = False

    async def action_go_up(self) -> None:
        current = Path(self.tree.path)
        if current.parent != current:
            await self._reroot(current.parent)

    async def action_go_home(self) -> None:
        await self._reroot(Path.home())

    async def action_toggle_hidden(self) -> None:
        tree = self.tree
        tree.show_hidden = not tree.show_hidden
        self.query_one("#hidden", Button).label = f"Hidden: {'on' if tree.show_hidden else 'off'}"
        await tree.reload()


class WorkspacePrompt(PathPicker):
    """Where a running session points its file tools (agent/pipeline/
    workspace.py). Same name and same Input contract as the bare prompt it
    replaced -- "off" closes file access, any other text is handed back for
    `set_workspace` to judge -- now with the tree beside it."""

    def __init__(self, current: Path | None) -> None:
        super().__init__(
            current, mode="dir", title="Workspace", allow_off=True,
            initial_text=str(current) if current else "",
        )


# --------------------------------------------------------------------------
# Pinning a model for one task
# --------------------------------------------------------------------------

class ModelPinDialog(ModalScreen[str | None]):
    """One `Select` of the models eligible for `task`. Dismisses with the
    chosen spec, "" to clear the pin, or None on Esc. Enter opens the list,
    Enter chooses, so the whole flow from the palette is three keystrokes."""

    DEFAULT_CSS = """
    ModelPinDialog { align: center middle; }
    ModelPinDialog > Vertical { width: 72; height: auto; border: round $accent; padding: 1 2; }
    ModelPinDialog .hint { margin-bottom: 1; }
    """

    def __init__(self, task: Task, options: list[tuple[str, str]], current: str = "") -> None:
        super().__init__()
        self._seat = task          # `_task` is Textual's own message-pump attribute
        self._options = options
        self._current = current
        self._initial = current

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"[bold]Pin a model for {self._seat.value}[/]\n"
                "[dim]Enter opens the list, Enter chooses, Esc cancels. "
                "The first entry clears the pin.[/]",
                classes="hint",
            )
            select = Select(self._options, allow_blank=False, id="pin")
            yield select

    def on_mount(self) -> None:
        select = self.query_one("#pin", Select)
        try:
            select.value = self._current
        except InvalidSelectValueError:
            self._initial = select.value
        select.focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        event.stop()
        if event.value == self._initial:
            # Textual reports the initial value as a change; that is not a pick.
            self._initial = object()
            return
        self.dismiss("" if event.value is Select.BLANK else str(event.value))

    def key_escape(self) -> None:
        self.dismiss(None)
