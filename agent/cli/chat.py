"""The interactive shell: `otto chat`.

Every turn goes through the pipeline (agent/pipeline/ -- the router/
planner/solver/summarizer/finder/evaluator graph that replaced the
orchestrator/worker/evaluate/subtask_consensus/synthesize swarm on
2026-09-10, which had itself replaced the Phase 11B code hive on
2026-09-09). There is no `agents` concept anymore -- one router dispatch,
one specialist, one evaluator, per round -- so there is nothing to size or
fix for a session; see agent/pipeline/nodes.py's module docstring for the
design discussion behind the swap.

Mid-run questions (2026-09-10, same day, nodes.py's seventh refinement):
`_run_turn` used to be a single `for update in run_pipeline_stream(...)`
loop. A run can now pause partway through (an `{"__ask__": ...}` event,
not the usual `{"__final__": ...}`) when a specialist or the evaluator
gets stuck on something only the person can answer -- `_run_turn` is a
`while` loop around that same `for` now, so it can render the question,
collect an answer with `_ask_user`, and keep going via
`resume_pipeline_stream()` for as long as the run keeps asking.

Bounded conversation memory (2026-09-10, same day, Phase 2 of claude/
otto-tiered-memory-design.md): `_run_turn` used to read `s.history[:-1]` --
an unbounded, ever-growing raw list `chat()` appended this turn's own
HumanMessage onto just before calling `_run_turn`, then appended the reply
onto after. `Session` now keeps that history in a bounded
`agent.memory.queue.TieredQueue` instead (agent/cli/shell.py); `_run_turn`
builds its own `HumanMessage(text)` locally rather than reading it back off
`s.history`, asks `s.history_for_graph()` for this turn's bounded
`(history, memory_context)`, and records the finished turn with
`s.record_turn()` once it has both halves -- `chat()`'s own loop no longer
touches history at all.
"""

from collections import Counter

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cli.output import save_final
from agent.cli.shell import Session, build_prompt_session, dispatch, render_update
from agent.cli.ui import err, out
from agent.pipeline.run import resume_pipeline_stream, run_pipeline_stream
from agent.pipeline.state import AgentState


def _ask_user(prompt_session, question: str, choices: list[str]) -> str:
    """Block on the person's answer to a mid-run `{"__ask__": ...}` event --
    print the question, list any choices as a picked-by-number menu
    (typing the number OR just typing free text both work; empty/Ctrl-C
    falls back to open text next time), and always accept free text too
    (the "multi choice + text bar" UX design call, nodes.py's module
    docstring, seventh refinement -- the REPL's own version of it: no text
    bar widget here, but the same two ways to answer).
    """
    out.print(Panel(Markdown(question), title="[warn]otto is asking[/]", border_style="warn"))
    if choices:
        for i, choice in enumerate(choices, start=1):
            err.print(f"[muted]{i}.[/] {choice}")
        err.print("[muted]pick a number, or just type your own answer[/]")
    while True:
        try:
            reply = prompt_session.prompt("your answer> ").strip()
        except (EOFError, KeyboardInterrupt):
            return ""
        if not reply:
            continue
        if choices and reply.isdigit() and 1 <= int(reply) <= len(choices):
            return choices[int(reply) - 1]
        return reply


def _run_turn(s: Session, text: str, prompt_session) -> None:
    tally: Counter = Counter()
    final: AgentState | None = None
    human_message = HumanMessage(text)
    # Bounded, not the raw ever-growing list -- module docstring. `history`
    # is however many recent turns still fit verbatim; `memory_context` is
    # whatever's older than that, already compacted (agent/memory/queue.py).
    history, memory_context = s.history_for_graph()

    stream = run_pipeline_stream(
        text, session_id=s.session_id, history=history, memory_context=memory_context,
    )
    while stream is not None:
        next_stream = None
        for update in stream:
            if "__ask__" in update:
                ask = update["__ask__"]
                answer = _ask_user(prompt_session, ask["question"], ask["choices"])
                out.print(f"[bold]you[/] {answer}")
                next_stream = resume_pipeline_stream(
                    answer, thread_id=ask["thread_id"], session_id=s.session_id,
                )
                break
            if "__final__" in update:
                final = update["__final__"]
                s.trace_id = update.get("__trace_id__")
                continue

            node, delta = next(iter(update.items()))
            render_update(node, delta, tally)
        stream = next_stream

    if final is not None:
        raw_output = (final.get("final_output") or "").strip()
        code = raw_output or "*(no output produced)*"
        out.print(Panel(Markdown(code), title="[spec]final[/]", border_style="ok"))
        s.record_turn(human_message, AIMessage(code) if raw_output else None)
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

        _run_turn(s, text, prompt_session)
