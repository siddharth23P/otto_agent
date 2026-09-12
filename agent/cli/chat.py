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

A line that moves (2026-09-12, design call: "optimize latency and steps,
and improve the tui"): the graph streams one update per NODE, and `agent`
is a node that spends every model call and every tool call inside itself.
Measured across the twenty golden items, that is 828 seconds of wall time,
96% of it inside model requests, with nothing printed in the middle of any
of them -- the longest item ran 131 seconds against a still terminal.
`_run_turn` now binds agent/pipeline/progress.py for the length of the
turn and keeps one transient Rich status line alive underneath it, saying
what the run is doing and how long it has been doing it. Ctrl-C sets the
same seam's cancel Event rather than killing the process, so a turn started
by mistake ends within one model call instead of having to be waited out.

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

import threading
import time
from collections import Counter

import typer
from langchain_core.messages import AIMessage, HumanMessage
from rich.markdown import Markdown
from rich.panel import Panel

from agent.cli.output import save_final
from agent.cli.shell import Session, build_prompt_session, dispatch, render_update
from agent.cli.ui import err, out
from agent.pipeline.progress import Cancelled, Progress, bind_progress
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


class _Line:
    """One transient status line under a turn in flight.

    Rich's `Console.status` already draws a spinner that clears itself, so
    this is only the bookkeeping: what the run last reported, and a clock
    that keeps moving through a ten-second model call that reports nothing
    while it runs.

    Every method here runs on the graph's own thread, which for the REPL is
    the main thread -- the same one Rich is drawing from -- so there is
    nothing to marshal and no lock to take.
    """

    def __init__(self, status) -> None:
        self._status = status
        self._started = time.monotonic()
        self.phase = "reading your message"
        self.model = ""
        self.tool = ""
        self.calls = 0

    def __call__(self, update: Progress) -> None:
        self.calls = update.calls or self.calls
        if update.kind == "call_start":
            self.model, self.tool = update.text, ""
            self.phase = self.phase or "thinking"
        elif update.kind == "phase":
            self.phase, self.tool = update.text, ""
        elif update.kind == "tool":
            target = (update.detail or {}).get("target", "")
            self.tool = f"{update.text} {target}".strip()
        else:
            # A streamed partial. Nothing to draw for it here -- the REPL
            # prints the settled answer as one Markdown panel, and redrawing
            # a growing block above a live prompt is what Rich's own docs
            # warn against. Still worth the clock tick.
            pass
        self.draw()

    def clock(self) -> str:
        elapsed = int(time.monotonic() - self._started)
        return f"{elapsed // 60}:{elapsed % 60:02d}"

    def draw(self) -> None:
        bits = [self.phase or "thinking"]
        if self.tool:
            bits.append(f"[chosen]{self.tool}[/]")
        if self.model:
            bits.append(f"[muted]{self.model}[/]")
        if self.calls:
            bits.append(f"[muted]{self.calls} calls[/]")
        bits.append(f"[muted]{self.clock()}[/]")
        bits.append("[muted]ctrl-c to stop[/]")
        self._status.update(" · ".join(bits))


def _run_turn(s: Session, text: str, prompt_session) -> None:
    """One turn, under a live status line and a cancel key.

    Ctrl-C inside a turn ends the turn, not the session. The REPL runs the
    graph on the main thread, so an interrupt lands inside whatever model
    call is in flight and there is nothing to cooperate with -- the cancel
    Event is still set on the way out, because a run can be several frames
    deep and the next `_call` must not spend again while the stack unwinds.
    Ctrl-C at the prompt still exits, which is where a person means it.
    """
    cancel = threading.Event()
    stopped = False
    with err.status("", spinner="dots") as status:
        line = _Line(status)
        line.draw()
        # Nothing reports anything during a ten-second model call, so the
        # clock has to move on its own or the line reads as a hung process.
        ticking = threading.Event()
        clock = threading.Thread(target=_keep_time, args=(line, ticking), daemon=True)
        clock.start()
        try:
            with bind_progress(line, cancel=cancel):
                _drive_turn(s, text, prompt_session)
        except (Cancelled, KeyboardInterrupt):
            cancel.set()
            stopped = True
        finally:
            ticking.set()
            clock.join(timeout=2.0)
    if stopped:
        err.print("[warn]stopped[/]")
    err.print(f"[muted]{line.calls} model calls · {line.clock()}[/]")


def _keep_time(line: "_Line", done: threading.Event) -> None:
    """Redraw the status line once a second until the turn ends.

    A thread rather than a signal or an async task: the REPL has no event
    loop of its own, and Rich's Live is already safe to update from another
    thread.
    """
    while not done.wait(1.0):
        line.draw()


def _drive_turn(s: Session, text: str, prompt_session) -> None:
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
