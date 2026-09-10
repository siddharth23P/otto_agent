"""The interactive shell: `otto chat`.

Every turn goes through the pipeline (agent/pipeline/ -- the router/
planner/solver/summarizer/finder/evaluator graph that replaced the
orchestrator/worker/evaluate/subtask_consensus/synthesize swarm on
2026-09-10, which had itself replaced the Phase 11B code hive on
2026-09-09). There is no `agents` concept anymore -- one router dispatch,
one specialist, one evaluator, per round -- so there is nothing to size or
fix for a session; see agent/pipeline/nodes.py's module docstring for the
design discussion behind the swap.
"""

from collections import Counter

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cli.output import save_final
from agent.cli.shell import Session, build_prompt_session, dispatch, render_update
from agent.cli.ui import err, out
from agent.pipeline.run import run_pipeline_stream
from agent.pipeline.state import AgentState


def _run_turn(s: Session, text: str) -> None:
    tally: Counter = Counter()
    final: AgentState | None = None

    for update in run_pipeline_stream(text, session_id=s.session_id):
        if "__final__" in update:
            final = update["__final__"]
            s.trace_id = update.get("__trace_id__")
            continue

        node, delta = next(iter(update.items()))
        render_update(node, delta, tally)

    if final is not None:
        raw_output = (final.get("final_output") or "").strip()
        code = raw_output or "*(no output produced)*"
        out.print(Panel(Markdown(code), title="[spec]final[/]", border_style="ok"))
        s.history.append(AIMessage(code))
        if raw_output:
            # On disk, not just on screen -- selecting a Rich panel's text
            # out of a live terminal mangles box-drawing borders and wrapped
            # lines (13.4's bug hunt). A plain file sidesteps that.
            s.turn += 1
            path = save_final(s.session_id, s.turn, raw_output, None)
            err.print(f"[muted]saved to {path}[/]")


def chat(ctx: typer.Context) -> None:
    """Talk to the pipeline."""
    s = Session(ctx=ctx.obj)
    err.print("[muted]otto:pipeline[/]")
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
