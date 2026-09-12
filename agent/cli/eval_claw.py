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
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import typer
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.memory.lessons import bind_bank, read_only
from agent.memory.store import MemoryStore
from agent.router.outcomes import bind_log
from agent.router.outcomes import read_only as routing_read_only
from agent.eval.claw_bench import (
    ClawEvalUnavailable,
    claw_root,
    grading_fingerprint,
    load_claw,
    missing_service_keys,
    run_task_file,
    select_tasks,
    split_tasks,
)


def _summary(outcomes: list, claw=None) -> dict:
    scored = [o for o in outcomes if o is not None]
    if not scored:
        return {"tasks": 0}
    n = len(scored)
    summary = {
        "tasks": n,
        "passed": sum(1 for o in scored if o.passed),
        "pass_rate": round(sum(1 for o in scored if o.passed) / n, 4),
        "mean_task_score": round(sum(o.task_score for o in scored) / n, 4),
        "mean_completion": round(sum(o.completion for o in scored) / n, 4),
        "mean_robustness": round(sum(o.robustness for o in scored) / n, 4),
        "mean_communication": round(sum(o.communication for o in scored) / n, 4),
        "errors": sum(1 for o in scored if o.error),
        "mean_wall_time_s": round(sum(o.wall_time_s for o in scored) / n, 2),
        # The cost axis. None of the 300 graders read it, so without this a
        # change that doubles spend for a tenth of a point reads as a win.
        "mean_model_calls": round(sum(o.model_calls for o in scored) / n, 2),
        "total_model_calls": sum(o.model_calls for o in scored),
    }

    repeated = [o for o in scored if len(o.trials) > 1]
    if repeated and claw is not None:
        k = min(len(o.trials) for o in repeated)
        summary["trials"] = k
        # Per task first, then averaged: pass^k over a task's own repeats says
        # "does it do this reliably", which is the property a self-improving
        # loop is most able to fake by finding one lucky trial.
        summary["mean_pass_hat_k"] = round(
            sum(claw.compute_pass_hat_k(o.trials, k=k) for o in repeated) / len(repeated), 4
        )
        summary["mean_pass_at_k"] = round(
            sum(claw.compute_pass_at_k(o.trials, k=k) for o in repeated) / len(repeated), 4
        )
        summary["score_spread"] = round(
            sum(max(o.trials) - min(o.trials) for o in repeated) / len(repeated), 4
        )
    return summary


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
    max_seconds: Annotated[
        Optional[float],
        typer.Option(help="Cap each task's budget below its own (tasks allow 120-900s). "
                          "Cheaper samples, and lower scores -- say so when reporting."),
    ] = None,
    trials: Annotated[
        int,
        typer.Option(help="Run each task this many times and report pass^k and the "
                          "spread. One run cannot tell a real change from judge "
                          "variance -- 260 of the 300 tasks are LLM-judged."),
    ] = 1,
    split: Annotated[
        str,
        typer.Option(help="'all', 'dev' (tune against these), or 'holdout' (never "
                          "tuned against; the honest number). The split is a hash "
                          "of the task id, so it does not move between runs."),
    ] = "all",
    lesson_bank: Annotated[
        Optional[Path],
        typer.Option(help="Lesson bank to read and write (default: ~/.otto/memory/"
                          "lessons.db). Point a measurement at its own file so it "
                          "does not learn from -- or teach -- your working one."),
    ] = None,
    no_learning: Annotated[
        bool,
        typer.Option("--no-learning", help="Read no lessons and write none. This is "
                                           "the compute-matched baseline every "
                                           "self-improvement claim has to be shown "
                                           "beside."),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full report as JSON.")] = False,
) -> None:
    if architecture not in {"graph", "single"}:
        err.print(f"architecture must be 'graph' or 'single', not {architecture!r}")
        raise typer.Exit(2)
    if split not in {"all", "dev", "holdout"}:
        err.print(f"split must be 'all', 'dev' or 'holdout', not {split!r}")
        raise typer.Exit(2)
    if trials < 1:
        err.print("trials must be at least 1")
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

    # Split BEFORE the limit, so --limit takes the first N of the chosen side
    # rather than trimming the pool and then splitting a different set each
    # time the pool changes.
    tasks = select_tasks(root, tag=tag, pattern=task)
    if split != "all":
        development, reserved = split_tasks(tasks)
        tasks = reserved if split == "holdout" else development
    if limit:
        tasks = tasks[:limit]
    if not tasks:
        err.print("no tasks matched")
        raise typer.Exit(1)

    out_dir = Path(trace_dir) if trace_dir else root / "traces" / f"otto-{architecture}-{time.strftime('%y%m%d-%H%M')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    err.print(f"{len(tasks)} task(s) -> {out_dir}")

    # What the loop is allowed to learn, decided once, here, rather than per
    # task -- the discipline is a property of the whole measurement.
    #
    #   --no-learning   nothing read, nothing written. The baseline.
    #   --split holdout lessons read, none written back. Tests TRANSFER: the
    #                   loop never tunes against these tasks, which is the
    #                   difference one evolved system got 31.7 points wrong.
    #   otherwise       read and write. This is where lessons come from.
    learning = ExitStack()
    if no_learning:
        learning.enter_context(bind_bank(None))
        # Routing adapts from observed outcomes too, so the baseline has to
        # hold that still as well. A "no learning" arm that quietly reordered
        # the model chain partway through would be measuring two things.
        learning.enter_context(bind_log(None))
    else:
        if lesson_bank:
            learning.enter_context(bind_bank(MemoryStore(lesson_bank)))
            # Beside the bank, so a measurement's routing evidence travels
            # with its lessons instead of leaking into the working install.
            learning.enter_context(bind_log(Path(lesson_bank).with_suffix(".seats.db")))
        if split == "holdout":
            # Read what the development runs learned, write nothing back --
            # lessons and routing evidence alike. That is what makes the
            # held-out number answer "does this TRANSFER" rather than "did the
            # loop find something that works on what it was tuned on".
            learning.enter_context(read_only())
            learning.enter_context(routing_read_only())
    err.print(
        "learning: " + ("off (baseline)" if no_learning else
                        "read-only (held out)" if split == "holdout" else "on")
    )

    outcomes = []
    try:
        _run_tasks(claw, tasks, outcomes, out_dir=out_dir, cfg=cfg, judge=judge,
                   architecture=architecture, port_offset=port_offset,
                   max_seconds=max_seconds, trials=trials)
    finally:
        learning.close()

    report = _build_report(
        outcomes, claw, architecture=architecture, tag=tag, split=split,
        trials=trials, max_seconds=max_seconds, out_dir=out_dir, cfg=cfg,
        judge=judge, no_learning=no_learning,
    )
    (out_dir / "otto_summary.json").write_text(json.dumps(report, indent=2))

    if as_json:
        out.print_json(data=report)
        return
    _print_summary(report, out_dir, split)


def _run_tasks(claw, tasks, outcomes, *, out_dir, cfg, judge, architecture,
               port_offset, max_seconds, trials) -> None:
    for i, task_yaml in enumerate(tasks, 1):
        name = task_yaml.parent.name
        err.print(f"[{i}/{len(tasks)}] {name}")
        missing = missing_service_keys(claw.TaskDefinition.from_yaml(task_yaml))
        if missing:
            err.print(
                f"  [warn]{', '.join(missing)} not set[/] -- this task's service "
                "reaches the real internet and will return nothing, so the score "
                "below measures the environment, not the agent"
            )
        try:
            outcome = run_task_file(
                claw, task_yaml,
                trace_dir=out_dir, cfg=cfg, judge=judge,
                architecture=architecture, port_offset=port_offset,
                max_seconds=max_seconds, trials=trials,
            )
        except Exception as exc:
            # One task's container or service failing is not a reason to lose
            # the other 299 -- report it and carry on, the way their batch does.
            err.print(f"  [bad]harness error[/] {type(exc).__name__}: {exc}")
            continue
        outcomes.append(outcome)
        flag = "[ok]pass[/]" if outcome.passed else "[warn]fail[/]"
        for item in outcome.checklist:
            err.print(f"    [{item.get('status', '?')}] {item.get('text', '')[:90]}")
        detail = f" ({outcome.error})" if outcome.error else ""
        spread = (
            " trials=" + "/".join(f"{t:.2f}" for t in outcome.trials)
            if len(outcome.trials) > 1 else ""
        )
        out.print(
            f"  {flag} score={outcome.task_score:.2f} "
            f"completion={outcome.completion:.2f} service-tools={outcome.tool_calls} "
            f"actions={outcome.agent_actions} calls={outcome.model_calls} "
            f"{outcome.wall_time_s:.0f}s{spread}{detail}"
        )


def _build_report(outcomes, claw, *, architecture, tag, split, trials,
                  max_seconds, out_dir, cfg, judge, no_learning) -> dict:
    return {
        "architecture": architecture,
        "learning": "off" if no_learning else "read-only" if split == "holdout" else "on",
        "tag": tag,
        "max_seconds": max_seconds,
        "split": split,
        "trials": trials,
        "trace_dir": str(out_dir),
        # What the scores mean. Two reports whose fingerprints differ are not
        # comparable, however similar the numbers look.
        "grading": grading_fingerprint(claw, cfg, judge),
        "summary": _summary(outcomes, claw),
        "tasks": [
            {
                "task_id": o.task_id, "task_score": o.task_score, "passed": o.passed,
                "completion": o.completion, "robustness": o.robustness,
                "communication": o.communication, "safety": o.safety,
                "tool_calls": o.tool_calls, "agent_actions": o.agent_actions,
                "checklist": o.checklist, "model_calls": o.model_calls,
                "trials": o.trials,
                "wall_time_s": o.wall_time_s, "error": o.error,
            }
            for o in outcomes
        ],
    }


def _print_summary(report: dict, out_dir: Path, split: str) -> None:
    s = report["summary"]
    out.print(
        f"\n{s.get('passed', 0)}/{s.get('tasks', 0)} passed "
        f"({s.get('pass_rate', 0):.1%})  mean score {s.get('mean_task_score', 0):.3f}  "
        f"completion {s.get('mean_completion', 0):.3f}  "
        f"{s.get('errors', 0)} harness/agent error(s)"
    )
    out.print(
        f"cost: {s.get('mean_model_calls', 0):.1f} model calls and "
        f"{s.get('mean_wall_time_s', 0):.0f}s per task"
    )
    if "mean_pass_hat_k" in s:
        k = s["trials"]
        out.print(
            f"reliability over {k} trials: pass^{k} {s['mean_pass_hat_k']:.3f}  "
            f"pass@{k} {s['mean_pass_at_k']:.3f}  "
            f"mean spread {s['score_spread']:.3f}"
        )
    grading = report["grading"]
    out.print(
        f"grading: {grading['otto_grading_path']} / claw {grading['claw_eval_revision'] or '?'} "
        f"/ judge {grading['judge']} / split {split}"
    )
    out.print(f"report: {out_dir / 'otto_summary.json'}")
