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
"""

from __future__ import annotations

import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langchain_core.messages import BaseMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from agent.cli.context import AppContext
from agent.cli.doctor import health_table, router_view
from agent.cli.models import models_table
from agent.cli.route import _chain, _summary
from agent.cli.ui import err, out
from agent.router.llm_provider import all_models
from agent.router.mapping import Task
from agent.router.router import NoViableRoute


@dataclass
class Session:
    ctx: AppContext
    history: list[BaseMessage] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str | None = None
    #: Counts turns that actually produced output, for output.py's filenames
    #: -- not every dispatched line (slash commands don't count).
    turn: int = 0


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
    session.history.clear()
    session.session_id = uuid.uuid4().hex
    session.trace_id = None
    session.turn = 0
    err.print("[muted]new session[/]")


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

#: The four specialists (mirrors agent.pipeline.nodes.ROLE_NODES -- not
#: imported directly to avoid this display module pulling in the whole
#: pipeline, including its module-level Router()/provider clients, just to
#: know four literal strings).
_ROLE_NODES = ("planner", "solver", "summarizer", "finder")


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

    if node == "router":
        for line in delta.get("board", []):
            sink(f"[muted]{line}[/]")
        return
    if node in _ROLE_NODES:
        for line in delta.get("board", []):
            sink(f"[muted]{line}[/]")
        text = (delta.get("output") or "").strip()
        if text:
            if len(text) > 400:
                text = text[:400] + "\n…"
            sink(Panel(Markdown(text), title=f"[spec]{node}[/]", border_style="spec"))
        return
    if node == "evaluator":
        for line in delta.get("board", []):
            sink(f"[muted]{line}[/]")
        return
