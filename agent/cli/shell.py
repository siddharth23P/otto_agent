"""The interactive shell: session state, slash commands, completion, and the
per-turn renderers -- the plain status lines and the swarm animation.

Reconciled against Phase 11B (see 9.4's Trap): the interactive default is no
longer a single resolved (RoutingDecision, chat model) pair -- it is
run_smart_stream()/run_code_stream(), a sequence of CodeTask graph updates.
Session carries no `d`/`llm`/`config` for that reason, and `/task` from the
original 9.5 spec is gone with it: there is no longer a single task route for
a turn to switch, since the code-hive already routes each of its own calls
(propose on Task.PLAN, author on Task.REASON, review on Task.CHAT_FAST)
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
from agent.graph.state import ALLOWED_AGENTS
from agent.router.llm_provider import all_models
from agent.router.mapping import Task
from agent.router.router import NoViableRoute


@dataclass
class Session:
    ctx: AppContext
    #: None = size every turn (run_smart_stream); a value from ALLOWED_AGENTS
    #: = skip size() entirely and always run_code_stream with exactly this
    #: many agents (set by --agents at launch, or /agents mid-session).
    agents: int | None = None
    history: list[BaseMessage] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str | None = None


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


def _agents_cmd(session: Session, arg: str) -> None:
    arg = arg.strip().lower()
    if not arg or arg == "auto":
        session.agents = None
        err.print("[muted]agents: sized per turn[/]")
        return
    try:
        n = int(arg)
    except ValueError:
        err.print(f"[warn]usage: /agents <n>|auto, n in {ALLOWED_AGENTS}[/]")
        return
    if n not in ALLOWED_AGENTS:
        err.print(f"[warn]{n} is not allowed; one of {ALLOWED_AGENTS}[/]")
        return
    session.agents = n
    err.print(f"[muted]agents: {n} (fixed)[/]")


def _new_cmd(session: Session, arg: str) -> None:
    session.history.clear()
    session.session_id = uuid.uuid4().hex
    session.trace_id = None
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
    "/agents": Slash("/agents", "fix the agent count, or 'auto' to size per turn", _agents_cmd,
                      lambda: [str(n) for n in ALLOWED_AGENTS] + ["auto"]),
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
# Rendering a turn -- plain lines/panels, no animation
# --------------------------------------------------------------------------

def render_update(node: str, delta: dict, tally: Counter) -> None:
    """Print one graph update. Handles both shapes that can appear on a
    stream: Phase 8's Hive (spawn/clone/consensus) and Phase 11B's CodeTask
    (propose.../stitch) -- the shell only ever drives the latter, but this
    stays honest about both since run()'s generator form (were it wired up
    the same way) would produce the former."""

    if node == "spawn":
        for line in delta.get("board", []):
            out.print(f"[muted]{line}[/]")
        return
    if node == "clone":
        vote = delta["votes"][0]
        tally[vote["answer"]] += 1
        out.print(Panel(
            Markdown(f"**{vote['answer']}**\n\n{vote['rationale']}"),
            title=f"[spec]clone · seed {vote['seed']}[/]", border_style="spec",
        ))
        return
    if node == "consensus":
        for line in delta.get("board", []):
            out.print(f"[muted]{line}[/]")
        tally.clear()
        return

    if node in ("propose", "decomp_consensus", "spawn_parts", "part_consensus", "stitch"):
        for line in delta.get("board", []):
            out.print(f"[muted]{line}[/]")
        return
    if node == "propose_review":
        vote = delta["proposal_votes"][0]
        mark = "[ok]yes[/]" if vote["approve"] else "[bad]no[/]"
        out.print(f"[muted]  split review · seed {vote['voter']}: {mark} — {vote['reason']}[/]")
        return
    if node == "author":
        sub = delta["submissions"][0]
        code = sub["code"].strip()
        if len(code) > 400:
            code = code[:400] + "\n…"
        out.print(Panel(
            Markdown(f"```\n{code}\n```"),
            title=f"[spec]author · part {sub['part']} · round {sub['round']}[/]",
            border_style="spec",
        ))
        return
    if node == "review":
        rv = delta["reviews"][0]
        mark = "[ok]yes[/]" if rv["approve"] else "[bad]no[/]"
        out.print(f"[muted]  part {rv['part']} review · seed {rv['voter']}: {mark} — {rv['reason']}[/]")
        return


# --------------------------------------------------------------------------
# The swarm animation -- "N cute things peeking" while their part is pending,
# settling into a status label as each one starts working and finishes.
# --------------------------------------------------------------------------

_CRITTERS = ("🐹", "🐨", "🦊", "🐰", "🐼", "🐻", "🐸", "🦉", "🐵")


class SwarmAnimator:
    """A small Live animation over `out`, one critter per agent.

    Purely cosmetic: it only reads the same deltas render_update() already
    prints and never touches graph state. `feed()` is meant to be called once
    per streamed update (in addition to render_update(), not instead of it)
    for as long as the turn's part-authoring/review phase is running; each
    call both updates a part's status and advances the animation by one
    frame, so a burst of review votes visibly animates the row rather than
    only redrawing on a timer.
    """

    def __init__(self, console, agents: int) -> None:
        self._console = console
        self._agents = agents
        self._status: dict[int, str] = {i: "assigned" for i in range(agents)}
        self._tick = 0
        self._live = None

    def __enter__(self) -> "SwarmAnimator":
        from rich.live import Live
        self._live = Live(self._frame(), console=self._console, refresh_per_second=8, transient=True)
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        if self._live is not None:
            self._live.__exit__(*exc)

    def _frame(self) -> Panel:
        grid = Table.grid(padding=(0, 2))
        for _ in range(self._agents):
            grid.add_column(justify="center")
        peeking = self._tick % 2 == 0
        row_critter, row_wall, row_label = [], [], []
        for i in range(self._agents):
            status = self._status[i]
            critter = _CRITTERS[i % len(_CRITTERS)]
            if status == "assigned":
                row_critter.append(f"[muted]{critter if peeking else ' '}[/]")
            elif status in ("authoring", "reviewing"):
                row_critter.append(f"[spec]{critter}[/]")
            elif status == "accepted":
                row_critter.append(f"[ok]{critter}✅[/]")
            else:  # exhausted
                row_critter.append(f"[warn]{critter}⚠️[/]")
            row_wall.append("[muted]▔▔▔[/]")
            row_label.append(f"[muted]part {i} · {status}[/]")
        grid.add_row(*row_critter)
        grid.add_row(*row_wall)
        grid.add_row(*row_label)
        return Panel(grid, title="[spec]swarm[/]", border_style="muted", padding=(0, 1))

    def feed(self, node: str, delta: dict) -> None:
        self._tick += 1
        if node == "author":
            part = delta["submissions"][0]["part"]
            self._status[part] = "authoring"
        elif node == "review":
            part = delta["reviews"][0]["part"]
            if self._status.get(part) == "authoring":
                self._status[part] = "reviewing"
        elif node == "part_consensus":
            for i, s in delta.get("part_status", {}).items():
                if s in ("accepted", "exhausted"):
                    self._status[i] = s
        if self._live is not None:
            self._live.update(self._frame())
