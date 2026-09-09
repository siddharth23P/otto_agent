"""The interactive shell: `otto chat`.

Every turn goes through the code hive (Phase 11B) -- sized fresh each turn by
default, or fixed for the whole session with --agents / mid-session with
/agents. There is no single-model turn anymore; see shell.py's module
docstring for what that superseded and why.
"""

from collections import Counter
from typing import Annotated, Optional

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cli.shell import Session, SwarmAnimator, build_prompt_session, dispatch, render_update
from agent.cli.ui import err, out
from agent.graph.code_run import run_code_stream
from agent.graph.code_state import CodeTask
from agent.graph.size import DEFAULT as SIZE_DEFAULT
from agent.graph.smart_run import run_smart_stream
from agent.graph.state import ALLOWED_AGENTS


def _run_turn(s: Session, text: str) -> None:
    tally: Counter = Counter()
    stream = (
        run_code_stream(text, thread_id=s.session_id, agents=s.agents)
        if s.agents is not None
        else run_smart_stream(text, thread_id=s.session_id)
    )

    animator: SwarmAnimator | None = None
    final: CodeTask | None = None
    agents_hint = s.agents

    try:
        for update in stream:
            if "__sizing__" in update:
                agents_hint = update["__sizing__"]
                plural = "" if agents_hint == 1 else "s"
                err.print(f"[muted]sized: {agents_hint} agent{plural}[/]")
                continue
            if "__final__" in update:
                final = update["__final__"]
                s.trace_id = update.get("__trace_id__")
                continue

            node, delta = next(iter(update.items()))
            if node == "spawn_parts" and animator is None:
                animator = SwarmAnimator(out, agents_hint or SIZE_DEFAULT)
                animator.__enter__()
            if animator is not None:
                animator.feed(node, delta)
            render_update(node, delta, tally)
    finally:
        if animator is not None:
            animator.__exit__(None, None, None)

    if final is not None:
        code = (final.get("final_code") or "").strip() or "*(no output produced)*"
        out.print(Panel(Markdown(f"```\n{code}\n```"), title="[spec]final[/]", border_style="ok"))
        s.history.append(AIMessage(code))


def chat(
    ctx: typer.Context,
    agents: Annotated[
        Optional[int],
        typer.Option(help="Fix the agent count for the whole session; omit to size every turn."),
    ] = None,
) -> None:
    """Talk to the swarm."""
    if agents is not None and agents not in ALLOWED_AGENTS:
        err.print(f"[bad]{agents} agents are not allowed; one of {ALLOWED_AGENTS}[/]")
        raise typer.Exit(2)

    s = Session(ctx=ctx.obj, agents=agents)
    banner = f"{s.agents} agents (fixed)" if s.agents is not None else "sized per turn"
    err.print(f"[muted]otto:code · {banner}[/]")
    err.print("[muted]/help for commands[/]")

    prompt_session = build_prompt_session()

    while True:
        try:
            text = prompt_session.prompt("you> ")
        except (EOFError, KeyboardInterrupt):
            out.print("\n[muted]bye[/]")
            break
        if not text.strip():
            continue
        if dispatch(s, text):
            continue

        s.history.append(HumanMessage(text))
        _run_turn(s, text)
