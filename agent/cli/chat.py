"""Streaming REPL.

Two rendering modes, chosen from the routing decision:

  Inception + diffusing  each chunk is a full snapshot of the answer, getting
                         cleaner as it denoises -> REPLACE the frame (Live)
  everything else        chunks are incremental text, plus reasoning blocks
                         when thinking is enabled -> APPEND
"""

import getpass
import os
import time
import uuid
from typing import Annotated, Any

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langfuse import propagate_attributes
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from agent.cli.ui import err, out
from agent.router.mapping import Task
from agent.router.router import RoutingDecision


def _blocks(chunk: Any) -> tuple[str, str]:
    """Split a chunk into (reasoning, text).

    `chunk.content` is a plain string until thinking is enabled, at which point
    it becomes a list of typed blocks. Handling both is what lets one loop serve
    every provider -- and forgetting it is why `out.print(chunk.content)` starts
    printing Python lists the day you turn thinking on.
    """
    content = chunk.content
    if isinstance(content, str):
        return "", content

    reasoning: list[str] = []
    text: list[str] = []
    for block in content:
        if isinstance(block, str):
            text.append(block)
            continue
        kind = block.get("type")
        if kind in ("thinking", "reasoning"):
            reasoning.append(
                block.get("thinking") or block.get("reasoning") or block.get("text") or ""
            )
        elif kind == "text":
            text.append(block.get("text", ""))
    return "".join(reasoning), "".join(text)


def _stream_diffusing(llm, history: list[BaseMessage], label: str,
                      markdown: bool = True, config: dict = None) -> tuple[str, int]:
    """Inception: replace the frame each snapshot, keep the final state.

    The border and title carry the state that per-character colouring cannot:
    with real snapshots there is no way to know which characters are settled,
    but the step count and the shift from accent to muted on completion say
    the useful part -- how many denoising passes it took, and that it is done.

    The frames arrive through `frame_sink` rather than through the stream,
    because a snapshot is a redraft of the whole answer, not a delta: if the
    model yielded them as chunks, LangChain would concatenate every draft and
    Langfuse would record that pile-up as the answer.
    """
    latest = ""
    steps = 0

    def frame(done: bool) -> Panel:
        suffix = f"{steps} steps" if done else f"denoising · {steps}"
        # Markdown is safe here for the same reason Text is: it renders
        # **bold** and fenced code, but never interprets [...] as Rich style
        # tags, so model output still cannot recolour or crash the UI.
        body = Markdown(latest) if markdown else Text(latest)
        return Panel(
            body,
            title=f"[spec]{label}[/] [muted]· {suffix}[/]",
            title_align="left",
            border_style="muted" if done else "spec",
            padding=(0, 1),
        )

    with Live(frame(False), console=out, refresh_per_second=12) as live:
        def on_frame(text: str) -> None:
            nonlocal latest, steps
            latest = text                        # a snapshot, not a delta
            steps += 1
            live.update(frame(False))

        # A copy, not a mutation: the model is reused across turns, and the
        # sink closes over this turn's Live.
        streaming = llm.model_copy(update={"frame_sink": on_frame})
        for chunk in streaming.stream(history, config=config):
            _, text = _blocks(chunk)
            if text:
                latest = text                    # the settled answer, once
        live.update(frame(True))                 # settle the border
    return latest, steps


def _stream_incremental(llm, history: list[BaseMessage],
                        config: dict = None) -> tuple[str, int]:
    """Everyone else: append text, and surface reasoning as it arrives.

    Returns the same (text, steps) shape as the diffusing helper so the caller
    does not have to care which one ran; steps is 0 because denoising passes
    are a diffusion idea and nothing else has them.
    """
    parts: list[str] = []
    thinking = False
    with err.status("thinking...") as status:
        for chunk in llm.stream(history,config=config):
            reasoning, text = _blocks(chunk)

            if reasoning:
                if not thinking:
                    status.stop()
                    out.print("[muted]· thinking[/]")
                    thinking = True
                out.print(reasoning, end="", markup=False, highlight=False, style="muted")

            if text:
                if not parts:
                    status.stop()
                    if thinking:
                        out.print("\n")
                    out.print("[muted]otto[/] ", end="")
                parts.append(text)
                out.print(text, end="", markup=False, highlight=False)
    out.print()
    return "".join(parts), 0


FEEDBACK = r"""[muted]/good  /bad  /score <0-1> \[comment]  ->  rate the last answer[/]"""


def _auto_scores(turn, d: RoutingDecision, elapsed: float, steps: int) -> None:
    """Scores nobody has to type.

    A score is just a named number attached to a trace, so anything you would
    otherwise read off one trace at a time belongs here -- these are the three
    that only make sense per turn, and that you will want to filter and chart
    across thousands of them.
    """
    turn.score_trace(name="latency_s", value=round(elapsed, 3), data_type="NUMERIC")
    # BOOLEAN scores take 0/1, not True/False.
    turn.score_trace(name="fell_back", value=int(d.fell_back), data_type="BOOLEAN")
    if steps:
        # How many denoising passes the answer took -- the one number that is
        # meaningful for Mercury and meaningless for everyone else.
        turn.score_trace(name="denoise_steps", value=steps, data_type="NUMERIC")


def _feedback(client, trace_id: str | None, text: str) -> bool:
    """Handle /good, /bad and /score. Returns True if `text` was a command.

    Scores arrive after the trace has ended, which is the whole point: a
    judgement about an answer is not available while the answer is streaming.
    That is why this needs the trace id rather than a live span.
    """
    head, _, rest = text.strip().partition(" ")
    if head not in ("/good", "/bad", "/score"):
        return False

    if trace_id is None:
        err.print("[warn]nothing to rate yet[/]")
        return True

    comment = rest.strip() or None
    if head == "/score":
        value, _, comment_text = rest.strip().partition(" ")
        try:
            score = float(value)
        except ValueError:
            err.print(r"[warn]usage: /score <0-1> \[comment][/]")
            return True
        comment = comment_text.strip() or None
    else:
        score = 1.0 if head == "/good" else 0.0

    client.create_score(
        name="user_feedback",
        value=score,
        data_type="NUMERIC",
        trace_id=trace_id,
        comment=comment,
    )
    # Scores queue like everything else; without this they show up whenever
    # the next flush happens, which in a REPL can be minutes later.
    client.flush()
    err.print(f"[muted]scored {score:g}[/]")
    return True


def chat(
    ctx: typer.Context,
    task: Annotated[Task, typer.Option(help="Route to use.")] = Task.CHAT_FAST,
    raw: Annotated[bool, typer.Option("--raw", help="Plain text, no Markdown.")] = False,
) -> None:
    """Talk to whichever model the task resolves to."""
    d = ctx.obj.router.resolve(task)
    handler = ctx.obj.handler
    llm = ctx.obj.router.model_for(d)

    diffusing = d.provider == "inception" and bool(d.params.get("diffusing"))
    mode = "diffusion" if diffusing else "thinking"
    # One session per REPL invocation, so the whole conversation reads as one
    # thing in Langfuse rather than N unrelated traces.
    session_id = uuid.uuid4().hex
    config = trace_config(d, handler=handler)
    err.print(
        f"[muted]{d.provider}:{d.model.id} · {mode}"
        f"{' · degraded' if d.fell_back else ''}[/]"
    )

    err.print(FEEDBACK)

    client = ctx.obj.client
    history: list[BaseMessage] = []
    trace_id: str | None = None

    # Trace-level attributes, set once for the whole REPL. They live here
    # rather than in the callback config because each turn is wrapped in a span
    # of its own below: `propagate_attributes` puts them on every span in the
    # context, so they land on the trace whether or not the handler is the root.
    with propagate_attributes(
        trace_name=f"otto:{d.task.value}",
        session_id=session_id,
        user_id=os.environ.get("OTTO_USER") or getpass.getuser(),
        tags=[d.provider, d.endpoint.value] + (["fell_back"] if d.fell_back else []),
    ):
        while True:
            try:
                text = Prompt.ask("[bold]you[/]", console=out)
            except (EOFError, KeyboardInterrupt):
                out.print("\n[muted]bye[/]")
                break
            if not text.strip():
                continue
            if _feedback(client, trace_id, text):
                continue

            history.append(HumanMessage(text))
            # One span per turn, so there is a trace to hang a score on after
            # the answer is finished -- and so input and output sit on the same
            # observation instead of only inside the LLM call.
            with client.start_as_current_observation(
                name=f"otto:{d.task.value}", as_type="span", input=text
            ) as turn:
                started = time.perf_counter()
                reply, steps = (
                    _stream_diffusing(llm, history, d.model.id, markdown=not raw, config=config)
                    if diffusing
                    else _stream_incremental(llm, history, config=config)
                )
                turn.update(output=reply)
                _auto_scores(turn, d, time.perf_counter() - started, steps)
                # Captured before the span closes; `/good` uses it next turn.
                trace_id = turn.trace_id
            history.append(AIMessage(reply))

def trace_config(d: RoutingDecision, handler) -> dict:
    """Everything Langfuse can group, filter or cost the LLM call by.

    The handler recognises four special metadata keys (langfuse_trace_name,
    langfuse_session_id, langfuse_user_id, langfuse_tags) and would set them on
    the trace -- but only reliably when its own span is the root, and here it
    is not: `chat` wraps each turn in a span so there is something to score.
    So the trace-level four are set with `propagate_attributes` at the top of
    the REPL, and what is left here is observation metadata: the routing
    decision, which is what makes a fallback searchable months later.
    """
    return {
        "callbacks": [handler],
        "metadata": {
            "otto_model": d.model.id,
            "otto_provider": d.provider,
            "otto_endpoint": d.endpoint.value,
            "otto_candidate": d.index,
            "otto_fell_back": d.fell_back,
            "otto_context_window": d.model.context_window,
            "otto_max_output_tokens": d.model.max_output_tokens,
            "otto_capabilities": sorted(c.value for c in d.model.capabilities),
            "otto_params": {k: str(v) for k, v in d.params.items()},
            "otto_skipped": [str(s) for s in d.skipped],
        },
    }