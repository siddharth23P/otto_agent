"""The interactive shell: session state, slash commands, completion, and the
per-turn renderer.

Reconciled against the router/planner/solver/summarizer/finder/evaluator
graph (agent/pipeline/, replaced the orchestrator/worker/evaluate/
subtask_consensus/synthesize swarm pipeline on 2026-09-10, which had itself
replaced the Phase 11B code hive on 2026-09-09): the interactive default is
no longer a single resolved (RoutingDecision, chat model) pair, nor a
swarm of N workers -- it is run_pipeline_stream(), a sequence of
AgentState graph updates from a single router-dispatched specialist per
round. Session carries no `d`/`llm`/`config`/`agents` for that reason:
there is nothing left to size or fix, and `/task` from the original 9.5
spec is gone with it -- there is no longer a single task route for a turn
to switch, since each node already routes its own calls (router on
Task.CHAT_FAST, planner on Task.PLAN, solver on Task.REASON, ...)
independently of anything a REPL command could select.

Conversation memory (2026-09-10, same day, Phase 2 of docs/design/tiered-
memory-design.md): `Session` used to carry conversation history as a plain
`list[BaseMessage]`, growing every turn with nothing capping it -- that gap
was flagged, not fixed, when agent/pipeline/run.py's `history` parameter
was first added. It now keeps a `history_queue` (an `agent.memory.queue.
TieredQueue`, "history" kind) instead: `history_for_graph()` is what a
caller hands to `run_pipeline_stream()`/`run_pipeline()`, `record_turn()`
is what a caller writes a finished turn back with -- see agent/memory/
wiring.py for the actual reconstruction/formatting logic, kept out of this
file on purpose (this file's own job stays "session state, slash commands,
completion, the per-turn renderer", not memory-engine internals).

The workspace is session state too (2026-09-12, design call: "we need
filesystem management so we can use it to write code and work on already
implemented codebases"). It is one directory for the whole session rather than
a per-turn argument, because that is what it means to a person: you open otto
on a repository and every turn after that is about that repository. `/workspace`
below is how to look at it or point it somewhere else mid-session, and
`Session.reset()` deliberately KEEPS it -- a new session is a new conversation,
not a different project. See agent/pipeline/workspace.py for what binding one
actually grants and what it does not.
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage, HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.cli.art import decorate_board_line
from agent.cli.context import AppContext
from agent.cli.doctor import health_table, router_view
from agent.cli.models import models_table
from agent.cli.route import _chain, _summary
from agent.cli.sessions import sessions_table
from agent.cli.ui import err, out
from agent.memory import sessions as session_index
from agent.memory import wiring as memory_wiring
from agent.memory.queue import TieredQueue
from agent.router.llm_provider import all_models
from agent.router.mapping import Task
from agent.router.router import NoViableRoute


@dataclass
class Session:
    ctx: AppContext
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str | None = None
    #: The directory every file tool in this session may touch, or None for
    #: no file access at all (module docstring; agent/pipeline/workspace.py
    #: for the boundary). Handed to run_pipeline_stream() on every turn --
    #: NOT bound here, because the contextvar has to be set on whichever
    #: thread actually consumes the stream, which for the TUI is not this one.
    workspace: Path | None = None
    #: Counts turns that actually produced output, for output.py's filenames
    #: -- not every dispatched line (slash commands don't count). Advanced by
    #: `record_turn` (2026-09-13), which is also what writes it to the
    #: session index, so the two cannot disagree about how far a session got.
    turn: int = 0
    #: What the session index calls this session (agent/memory/sessions.py):
    #: the first message, cut short, until `/rename`. Empty until the first
    #: turn is recorded -- an unnamed session is one nothing has happened in.
    title: str = ""
    #: This session's own bounded conversation memory (module docstring) --
    #: `init=False`/set in __post_init__ rather than a `field(default_factory=...)`
    #: because building one needs `session_id`, which isn't available yet
    #: when dataclass field defaults are evaluated (session_id is itself
    #: just becoming set at that point, by ITS OWN default_factory, with no
    #: guaranteed ordering against a sibling field's factory).
    history_queue: TieredQueue = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.history_queue = memory_wiring.new_history_queue(self.session_id)

    def history_for_graph(self) -> tuple[list[BaseMessage], str]:
        """(bounded prior messages, compacted-memory context) for this
        session's NEXT turn -- agent/pipeline/run.py's `history`/
        `memory_context` parameters. See agent/memory/wiring.py's
        history_for_graph() for what each half actually contains."""
        return memory_wiring.history_for_graph(self.history_queue)

    def workspace_arg(self) -> str | None:
        """This session's workspace as run.py wants it: a plain string, or
        None. One accessor rather than `str(s.workspace) if s.workspace else
        None` repeated at four call sites, each of which could get the
        empty-vs-None distinction wrong on its own."""
        return str(self.workspace) if self.workspace is not None else None

    def record_turn(self, human: BaseMessage, ai: BaseMessage | None) -> None:
        """Write one finished turn into this session's memory -- `ai` is
        None for a turn that produced no final output (nothing worth
        remembering as "otto said"), same as chat.py/tui.py already treat
        an empty `raw_output` as nothing to save to disk.

        Also counts the turn (when it produced output) and registers the
        session in the index (agent/memory/sessions.py) -- creating its row
        on the first turn, titled from what the person said, and bumping
        its activity after. Here rather than in the two front ends because
        both must do exactly this and neither may forget: a session the
        index does not know is one `/resume` cannot find.
        """
        memory_wiring.record_turn(self.history_queue, human, ai)
        if ai is not None:
            self.turn += 1
        info = session_index.touch(
            self.session_id,
            title=self.title or session_index.title_from(str(human.content)),
            workspace=self.workspace, turns=self.turn,
        )
        self.title = info.title

    def load(self, ref: str) -> session_index.SessionInfo:
        """Adopt a saved session in place -- the counterpart of `reset()`,
        which mints a new one. `ref` is what agent/memory/sessions.py's
        `resolve` accepts: "last", an id, or a unique prefix; LookupError
        for anything else, with the reason in the message.

        The saved workspace comes back with it when the directory still
        exists: a session about a repository is still about that repository.
        A caller with an explicit `--workspace` sets it after this returns.
        Memory is the session's own file, read back rather than started
        over (agent/memory/wiring.py's `restore=True`).
        """
        info = session_index.resolve(ref)
        self.close()
        self.session_id = info.id
        self.trace_id = None
        self.turn = info.turns
        self.title = info.title
        self.history_queue = memory_wiring.new_history_queue(info.id, restore=True)
        if info.workspace and Path(info.workspace).is_dir():
            self.workspace = Path(info.workspace)
        return info

    def rename(self, title: str) -> None:
        self.title = session_index.rename(self.session_id, title, workspace=self.workspace).title

    def transcript(self) -> tuple[str, list[BaseMessage]]:
        """What a resumed session has to show for itself: everything older
        than the recent turns as the compacted text the model will see, and
        the recent turns as typed messages -- exactly what the next turn's
        prompt is built from, so what is on screen is what otto knows."""
        return self.history_queue.earlier_view(), memory_wiring.recent_messages(self.history_queue)

    def close(self) -> None:
        """Release this session's memory file. `reset()`/`load()` call it on
        the store they replace, and anything about to delete the session
        must call it first: Windows will not delete a file a connection
        still holds open (CI's windows-latest job found this), and POSIX
        would -- silently, leaving the connection writing into an unlinked
        inode. Safe to call twice."""
        self.history_queue.store.close()

    def reset(self) -> None:
        """New session, new memory -- `/new` (chat.py)/"new session"
        (tui.py)'s existing "clear history, start fresh" contract, now
        covering `history_queue` too (a fresh TieredQueue backed by a
        fresh SQLite file under this new session_id, not the old queue
        cleared in place -- the old session's compacted history stays on
        disk, keyed by its own now-abandoned session_id, exactly as
        harmless and exactly as inaccessible as it always was once a
        session ended)."""
        self.close()
        self.session_id = uuid.uuid4().hex
        self.trace_id = None
        self.turn = 0
        self.title = ""
        self.history_queue = memory_wiring.new_history_queue(self.session_id)
        # `workspace` is deliberately NOT reset: "clear history, start fresh"
        # is about the conversation, and a person who opened otto on a repo and
        # then cleared the chat is still working on that repo. Changing it is
        # `/workspace <path>`, which is explicit.


# --------------------------------------------------------------------------
# Slash commands
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Slash:
    name: str
    help: str
    run: Callable[[Session, str], None]
    options: Callable[[], list[str]] = lambda: []


def _route_cmd(session: Session, arg: str) -> None:
    if not arg:
        err.print("[warn]usage: /route <task>[/]")
        return
    try:
        task = Task(arg)
    except ValueError:
        err.print(f"[warn]unknown task {arg!r}; one of {[t.value for t in Task]}[/]")
        return
    try:
        d = session.ctx.router.resolve(task)
    except NoViableRoute as exc:
        out.print(_chain(task, exc.skipped, None))
        err.print("[bad]no viable model[/]")
        return
    out.print(_chain(task, d.skipped, d))
    out.print(_summary(d))


def _models_cmd(session: Session, arg: str) -> None:
    with err.status("fetching catalogues…"):
        found = all_models(None)
    found.sort(key=lambda m: (m.provider, m.id))
    out.print(models_table(found) if found else "[muted]no models matched[/]")


def _doctor_cmd(session: Session, arg: str) -> None:
    from agent.router.llm_provider import health_report

    with err.status("contacting providers"):
        reports = health_report()
    out.print(health_table(reports))
    out.print(router_view(session.ctx.router))


def _score_common(session: Session, value: float, comment: str | None) -> None:
    if session.trace_id is None:
        err.print("[warn]nothing to rate yet[/]")
        return
    session.ctx.client.create_score(
        name="user_feedback",
        value=value,
        data_type="NUMERIC",
        trace_id=session.trace_id,
        comment=comment,
    )
    session.ctx.client.flush()
    err.print(f"[muted]scored {value:g}[/]")


def _good_cmd(session: Session, arg: str) -> None:
    _score_common(session, 1.0, arg.strip() or None)


def _bad_cmd(session: Session, arg: str) -> None:
    _score_common(session, 0.0, arg.strip() or None)


def _score_cmd(session: Session, arg: str) -> None:
    value_text, _, comment_text = arg.partition(" ")
    try:
        value = float(value_text)
    except ValueError:
        err.print(r"[warn]usage: /score <0-1> \[comment][/]")
        return
    _score_common(session, value, comment_text.strip() or None)


def _new_cmd(session: Session, arg: str) -> None:
    session.reset()
    err.print("[muted]new session[/]")


def render_transcript(session: Session, sink: Callable[[object], None] | None = None) -> None:
    """Print a resumed session's history the way it was first shown: the
    compacted part as one dim panel, then each recent turn as the same
    "you" line and final panel chat.py posts live. Shared shape with the
    TUI's own replay, which posts the same pieces into its transcript.
    `sink` defaults to the REPL console at call time, not definition time,
    so a test that swaps the console sees what this prints."""
    sink = sink or out.print
    earlier, messages = session.transcript()
    if earlier:
        sink(Panel(earlier, title="[muted]earlier, compacted[/]", border_style="muted"))
    for message in messages:
        text = str(message.content)
        if isinstance(message, HumanMessage):
            sink(f"[bold]you[/] {text}")
        else:
            sink(Panel(Markdown(text), title="[spec]final[/]", border_style="ok"))


def _sessions_cmd(session: Session, arg: str) -> None:
    rows = session_index.list_sessions(limit=20)
    if not rows:
        out.print("[muted]no saved sessions yet -- a session is saved once a turn finishes[/]")
        return
    out.print(sessions_table(rows, current=session.session_id))
    out.print("[muted]/resume <id> picks one up; /resume last is the newest[/]")


def _resume_cmd(session: Session, arg: str) -> None:
    if not arg:
        err.print("[warn]usage: /resume <id or prefix | last>[/]")
        return
    try:
        info = session.load(arg)
    except LookupError as exc:
        err.print(f"[warn]{exc}[/]")
        return
    err.print(f"[muted]resumed {info.short_id} · {info.label} · {info.turns} turn(s)[/]")
    err.print(f"[muted]{describe_workspace(session.workspace)}[/]")
    render_transcript(session)


def _rename_cmd(session: Session, arg: str) -> None:
    if not arg:
        err.print("[warn]usage: /rename <title>[/]")
        return
    session.rename(arg)
    err.print(f"[muted]session is now called {session.title!r}[/]")


def resolve_workspace(chosen: Path | None, disabled: bool) -> Path | None:
    """What `--workspace PATH` / `--no-workspace` / neither actually means.

    One resolver shared by `otto chat` and `otto tui` so the two cannot drift,
    and the single place the "current directory by default" policy is written
    down (2026-09-12 design call, asked explicitly: a session should work on
    the repo you launched it in, the way every other developer tool does).

    `--no-workspace` wins over `--workspace`, deliberately: the two together
    are a contradiction, and resolving it towards LESS access is the only
    direction that cannot surprise someone.
    """
    if disabled:
        return None
    if chosen is not None:
        return chosen.expanduser().resolve()
    return Path.cwd()


def describe_workspace(workspace: Path | None) -> str:
    """One line saying what file access this session has. Shared with the TUI
    (agent/cli/tui.py) so both front ends say the same thing about the same
    state -- the whole point of Session living in this module."""
    if workspace is None:
        return "no workspace: file tools are off (open one with --workspace, or /workspace <path>)"
    return f"workspace: {workspace}"


def set_workspace(session: Session, path: str) -> str | None:
    """Point `session` at `path`, or return why it cannot be. Returns None on
    success so a caller can treat a string as "show this and stop".

    Rejects a path that is not an existing directory rather than creating it,
    unlike `bind_workspace`, which creates because a harness genuinely wants a
    fresh scratch dir. A person typing a path at a prompt has almost certainly
    typo'd it, and silently creating `~/projcts` is worse than saying so.
    """
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        return f"{path!r} is not a usable path: {exc}"
    if not resolved.exists():
        return f"{resolved} does not exist"
    if not resolved.is_dir():
        return f"{resolved} is not a directory"
    session.workspace = resolved
    return None


def _workspace_cmd(session: Session, arg: str) -> None:
    target = arg.strip()
    if not target:
        out.print(f"[muted]{describe_workspace(session.workspace)}[/]")
        return
    if target in {"off", "none"}:
        session.workspace = None
        err.print("[muted]workspace closed: file tools are off[/]")
        return
    problem = set_workspace(session, target)
    if problem:
        err.print(f"[warn]{problem}[/]")
        return
    err.print(f"[muted]workspace: {session.workspace}[/]")


def _help_cmd(session: Session, arg: str) -> None:
    t = Table(box=None, show_header=False, pad_edge=False)
    t.add_column(style="chosen")
    t.add_column(style="muted")
    for name, slash in sorted(COMMANDS.items()):
        t.add_row(name, slash.help)
    out.print(t)


COMMANDS: dict[str, Slash] = {
    "/route": Slash("/route", "show how a task resolves", _route_cmd, lambda: [t.value for t in Task]),
    "/models": Slash("/models", "list every configured model", _models_cmd),
    "/doctor": Slash("/doctor", "check every provider", _doctor_cmd),
    "/good": Slash("/good", "score the last answer 1.0", _good_cmd),
    "/bad": Slash("/bad", "score the last answer 0.0", _bad_cmd),
    "/score": Slash("/score", r"score the last answer <0-1> \[comment]", _score_cmd),
    "/new": Slash("/new", "clear history, start a fresh session", _new_cmd),
    "/sessions": Slash("/sessions", "list saved sessions, newest first", _sessions_cmd),
    "/resume": Slash(
        "/resume", "pick a saved session back up: <id or prefix>, or last", _resume_cmd,
        lambda: ["last", *(r.short_id for r in session_index.list_sessions(limit=10))],
    ),
    "/rename": Slash("/rename", "give this session a title", _rename_cmd),
    "/workspace": Slash(
        "/workspace", "show the working directory, or <path> to change it (off to close)",
        _workspace_cmd, lambda: ["off"],
    ),
    "/help": Slash("/help", "list these commands", _help_cmd),
}


def dispatch(session: Session, text: str) -> bool:
    """Handle a slash command. Returns True if `text` was one (handled or
    not -- an unknown slash still returns True so it never reaches the
    model)."""
    if not text.startswith("/"):
        return False
    head, _, arg = text.partition(" ")
    slash = COMMANDS.get(head)
    if slash is None:
        err.print(f"[warn]unknown command {head!r}[/]")
        _help_cmd(session, "")
        return True
    slash.run(session, arg.strip())
    return True


# --------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------

class SlashCompleter(Completer):
    def __init__(self, commands: dict[str, Slash]) -> None:
        self._commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        head, sep, arg = text.partition(" ")
        if not sep:
            for name in self._commands:
                if name.startswith(head):
                    yield Completion(name, start_position=-len(head))
            return
        slash = self._commands.get(head)
        if slash is None:
            return
        for option in slash.options():
            if option.startswith(arg):
                yield Completion(option, start_position=-len(arg))


def build_prompt_session() -> PromptSession:
    return PromptSession(
        completer=SlashCompleter(COMMANDS),
        history=FileHistory(str(Path.home() / ".otto_history")),
    )


# --------------------------------------------------------------------------
# Rendering a turn -- plain lines/panels, no animation (there is no fan-out
# left to animate: exactly one specialist runs per round).
# --------------------------------------------------------------------------

#: The one working node. It was four -- planner, solver, summarizer, finder --
#: until they collapsed into a single agent loop that switches mode instead of
#: switching node (agent/pipeline/nodes.py). Still a literal rather than an
#: import, so this display module does not pull in the whole pipeline and its
#: module-level Router() just to know one string.
#:
#: Where its lines come from changed too, and that matters more than the name.
#: LangGraph's "updates" stream emits once per node RETURN, so with one
#: long-running loop this branch used to fire once, at the very end, after
#: minutes of silence. The loop now writes each tool call and mode swap to the
#: "custom" stream as it happens (nodes.py's `_emit`), in this same node-shaped
#: form -- so these lines arrive live and the branch below needs no changes.
_ROLE_NODES = ("agent", "research")


def render_update(node: str, delta: dict, tally: Counter, sink: Callable[[object], None] = out.print) -> None:
    """Render one graph update by calling `sink(renderable_or_string)` once
    (zero times for a node with nothing to show). Handles both shapes that
    can appear on a stream: Phase 8's Hive (spawn/clone/consensus, dead code
    today -- nothing calls agent/graph/run.py's run() from either front end
    -- kept here only because a future caller wiring it back up would want
    the same renderer) and the pipeline's AgentState (router/planner/
    solver/summarizer/finder/evaluator) -- the REPL only ever drives the
    latter.

    `sink` defaults to the REPL's `out.print`; the TUI (tui.py) passes its
    transcript widget's `.write` instead, so the same branch logic drives two
    different front ends over one Rich renderable per call."""

    if node == "spawn":
        for line in delta.get("board", []):
            sink(f"[muted]{line}[/]")
        return
    if node == "clone":
        vote = delta["votes"][0]
        tally[vote["answer"]] += 1
        sink(Panel(
            Markdown(f"**{vote['answer']}**\n\n{vote['rationale']}"),
            title=f"[spec]clone · seed {vote['seed']}[/]", border_style="spec",
        ))
        return
    if node == "consensus":
        for line in delta.get("board", []):
            sink(f"[muted]{line}[/]")
        tally.clear()
        return

    if node in _ROLE_NODES:
        for line in delta.get("board", []):
            sink(f"[muted]{decorate_board_line(line)}[/]")
        text = (delta.get("output") or "").strip()
        if text:
            if len(text) > 400:
                text = text[:400] + "\n…"
            sink(Panel(Markdown(text), title=f"[spec]{node}[/]", border_style="spec"))
        return
    if node == "evaluator":
        for line in delta.get("board", []):
            sink(f"[muted]{decorate_board_line(line)}[/]")
        return
