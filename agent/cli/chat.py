"""Streaming REPL.

Two rendering modes, chosen from the routing decision:

  Inception + diffusing  each chunk is a full snapshot of the answer, getting
                         cleaner as it denoises -> REPLACE the frame (Live)
  everything else        chunks are incremental text, plus reasoning blocks
                         when thinking is enabled -> APPEND
"""

from typing import Annotated, Any

import typer
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from agent.cli.ui import err, out
from agent.router.mapping import Task


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
                      markdown: bool = True) -> str:
    """Inception: replace the frame each chunk, keep the final state.

    The border and title carry the state that per-character colouring cannot:
    with real snapshots there is no way to know which characters are settled,
    but the step count and the shift from accent to muted on completion say
    the useful part -- how many denoising passes it took, and that it is done.
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
        for chunk in llm.stream(history):
            _, text = _blocks(chunk)
            if text:
                latest = text                    # a snapshot, not a delta
                steps += 1
                live.update(frame(False))
        live.update(frame(True))                 # settle the border
    return latest


def _stream_incremental(llm, history: list[BaseMessage]) -> str:
    """Everyone else: append text, and surface reasoning as it arrives."""
    parts: list[str] = []
    thinking = False
    with err.status("thinking...") as status:
        for chunk in llm.stream(history):
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
    return "".join(parts)


def chat(
    ctx: typer.Context,
    task: Annotated[Task, typer.Option(help="Route to use.")] = Task.CHAT_FAST,
    raw: Annotated[bool, typer.Option("--raw", help="Plain text, no Markdown.")] = False,
) -> None:
    """Talk to whichever model the task resolves to."""
    d = ctx.obj.router.resolve(task)
    llm = ctx.obj.router.model_for(d)

    diffusing = d.provider == "inception" and bool(d.params.get("diffusing"))
    mode = "diffusion" if diffusing else "thinking"
    err.print(
        f"[muted]{d.provider}:{d.model.id} · {mode}"
        f"{' · degraded' if d.fell_back else ''}[/]"
    )

    history: list[BaseMessage] = []
    while True:
        try:
            text = Prompt.ask("[bold]you[/]", console=out)
        except (EOFError, KeyboardInterrupt):
            out.print("\n[muted]bye[/]")
            break
        if not text.strip():
            continue

        history.append(HumanMessage(text))
        reply = (
            _stream_diffusing(llm, history, d.model.id, markdown=not raw) if diffusing
            else _stream_incremental(llm, history)
        )
        history.append(AIMessage(reply))
