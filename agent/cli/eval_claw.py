"""`otto eval-claw` -- Claw-Eval's 300 tasks against Otto's own agent.

Their `claw-eval run` can only point at a model endpoint, so the number it
gives is about a model inside their agent loop. This command runs OTTO on the
task -- its graph, its memory, its tools, its evaluator -- and hands the
resulting trace to their own graders, so the score is about the agent.
See agent/eval/claw_bench.py for the seam that makes that possible.

Needs a Claw-Eval checkout (--claw-root or $CLAW_EVAL_ROOT) with its
requirements installed, and Docker for the 169 tasks whose files live in a
container. Start with --limit: every task is a full agent run.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import typer
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.eval.claw_bench import (
    ClawEvalUnavailable,
    claw_root,
    load_claw,
    needs_container,
    run_task_file,
    select_tasks,
)


def _summary(outcomes: list) -> dict:
    scored = [o for o in outcomes if o is not None]
    if not scored:
        return {"tasks": 0}
    n = len(scored)
    return {
        "tasks": n,
        "passed": sum(1 for o in scored if o.passed),
        "pass_rate": round(sum(1 for o in scored if o.passed) / n, 4),
        "mean_task_score": round(sum(o.task_score for o in scored) / n, 4),
        "mean_completion": round(sum(o.completion for o in scored) / n, 4),
        "mean_robustness": round(sum(o.robustness for o in scored) / n, 4),
        "mean_communication": round(sum(o.communication for o in scored) / n, 4),
        "errors": sum(1 for o in scored if o.error),
        "mean_wall_time_s": round(sum(o.wall_time_s for o in scored) / n, 2),
    }


def eval_claw_cmd(
    claw_root_opt: Annotated[
        Optional[Path],
        typer.Option("--claw-root", help="Claw-Eval checkout; defaults to $CLAW_EVAL_ROOT."),
    ] = None,
    tag: Annotated[
        Optional[str],
        typer.Option(help="Capability to run: general, multimodal, user_agent, multi_service."),
    ] = None,
    task: Annotated[
        Optional[str], typer.Option(help="Substring of a task directory name, e.g. 'T01' or 'M001'."),
    ] = None,
    limit: Annotated[Optional[int], typer.Option(help="Run at most this many tasks.")] = 5,
    architecture: Annotated[
        str, typer.Option(help="'graph' (the real pipeline) or 'single' (one-conversation control)."),
    ] = "graph",
    config: Annotated[
        Optional[Path], typer.Option(help="Claw-Eval config.yaml (judge, sandbox, user-agent models)."),
    ] = None,
    trace_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write traces (default: <checkout>/traces/otto_<time>).")
    ] = None,
    no_judge: Annotated[bool, typer.Option("--no-judge", help="Skip the LLM judge.")] = False,
    port_offset: Annotated[int, typer.Option(help="Shift every mock service port, for parallel runs.")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
) -> None:
    if architecture not in {"graph", "single"}:
        err.print(f"architecture must be 'graph' or 'single', not {architecture!r}")
        raise typer.Exit(2)

    try:
        root = claw_root(claw_root_opt)
        claw = load_claw(root)
    except ClawEvalUnavailable as exc:
        err.print(str(exc))
        raise typer.Exit(2) from None

    cfg = claw.load_config(str(config) if config else None)
    # Their own factory, so the judge model, key and base URL come from the
    # same config a `claw-eval run` would use rather than from a second copy.
    judge = claw.cli._make_judge(cfg, SimpleNamespace(no_judge=no_judge, judge_model=None))

    tasks = select_tasks(root, tag=tag, pattern=task, limit=limit)
    if not tasks:
        err.print("no tasks matched")
        raise typer.Exit(1)

    out_dir = Path(trace_dir) if trace_dir else root / "traces" / f"otto-{architecture}-{time.strftime('%y%m%d-%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    err.print(f"{len(tasks)} task(s) -> {out_dir}")

    outcomes = []
    for i, task_yaml in enumerate(tasks, 1):
        name = task_yaml.parent.name
        err.print(f"[{i}/{len(tasks)}] {name}")
        try:
            outcome = run_task_file(
                claw, task_yaml,
                trace_dir=out_dir, cfg=cfg, judge=judge,
                architecture=architecture, port_offset=port_offset,
            )
        except Exception as exc:
            # One task's container or service failing is not a reason to lose
            # the other 299 -- report it and carry on, the way their batch does.
            err.print(f"  [bad]harness error[/] {type(exc).__name__}: {exc}")
            continue
        outcomes.append(outcome)
        flag = "[ok]pass[/]" if outcome.passed else "[warn]fail[/]"
        detail = f" ({outcome.error})" if outcome.error else ""
        out.print(
            f"  {flag} score={outcome.task_score:.2f} "
            f"completion={outcome.completion:.2f} service-tools={outcome.tool_calls} "
            f"actions={outcome.agent_actions} "
            f"{outcome.wall_time_s:.0f}s{detail}"
        )

    report = {
        "architecture": architecture,
        "tag": tag,
        "trace_dir": str(out_dir),
        "summary": _summary(outcomes),
        "tasks": [
            {
                "task_id": o.task_id, "task_score": o.task_score, "passed": o.passed,
                "completion": o.completion, "robustness": o.robustness,
                "communication": o.communication, "safety": o.safety,
                "tool_calls": o.tool_calls, "agent_actions": o.agent_actions,
                "wall_time_s": o.wall_time_s, "error": o.error,
            }
            for o in outcomes
        ],
    }
    (out_dir / "otto_summary.json").write_text(json.dumps(report, indent=2))

    if as_json:
        out.print_json(data=report)
        return
    s = report["summary"]
    out.print(
        f"\n{s.get('passed', 0)}/{s.get('tasks', 0)} passed "
        f"({s.get('pass_rate', 0):.1%})  mean score {s.get('mean_task_score', 0):.3f}  "
        f"completion {s.get('mean_completion', 0):.3f}  "
        f"{s.get('errors', 0)} harness/agent error(s)"
    )
    out.print(f"report: {out_dir / 'otto_summary.json'}")
