"""`otto eval-swe` -- SWE-bench Verified, graded by each repository's own tests.

The only benchmark here whose verdict nobody involved can argue with: 500 real
GitHub issues, and the maintainers' own test suite decides. See
agent/eval/swe_bench.py for what Otto is and is not told.

Needs Docker. The official per-instance images are about 3.5GB each and are
pulled on first use, so start with --limit 1 and expect the first run of a
repository to spend several minutes downloading before the agent starts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from agent.cli.ui import err, out
from agent.eval.swe_bench import (
    SweBenchUnavailable, load_dataset, resolve_image, run_instance, select,
)


def eval_swe_cmd(
    repo: Annotated[
        Optional[str], typer.Option(help="Only instances from repositories matching this."),
    ] = None,
    instance: Annotated[
        Optional[str], typer.Option(help="Substring of an instance id, e.g. 'astropy-12907'."),
    ] = None,
    difficulty: Annotated[
        Optional[str],
        typer.Option(help="Filter on the dataset's own estimate: '<15', '15 min', '1-4', '>4'."),
    ] = None,
    limit: Annotated[int, typer.Option(help="Run at most this many instances.")] = 1,
    max_seconds: Annotated[
        float, typer.Option(help="Wall-clock budget per instance, before grading."),
    ] = 1800.0,
    max_model_calls: Annotated[
        Optional[int], typer.Option(help="Override the default spend ceiling per instance."),
    ] = None,
    trace_dir: Annotated[
        Optional[Path], typer.Option(help="Where to write each instance's diff."),
    ] = None,
    dataset: Annotated[
        Optional[Path], typer.Option(help="Cached parquet (default: agent/eval/data/)."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    try:
        instances = select(
            load_dataset(dataset), repo=repo, pattern=instance,
            difficulty=difficulty, limit=limit,
        )
    except SweBenchUnavailable as exc:
        err.print(str(exc))
        raise typer.Exit(2) from None

    if not instances:
        err.print("no instances matched")
        raise typer.Exit(1)

    out_dir = Path(trace_dir) if trace_dir else Path("/tmp") / f"otto-swe-{time.strftime('%y%m%d-%H%M')}"
    err.print(f"{len(instances)} instance(s) -> {out_dir}")

    outcomes = []
    for i, item in enumerate(instances, 1):
        err.print(f"[{i}/{len(instances)}] {item.instance_id}  ({item.difficulty or 'unrated'})")
        image, emulated = resolve_image(item.instance_id)
        err.print(f"  image {image}"
                  + ("  [warn](x86_64 under emulation -- no arm64 build)[/]"
                     if emulated else ""))
        try:
            outcome = run_instance(
                item, max_seconds=max_seconds,
                max_model_calls=max_model_calls, trace_dir=out_dir,
            )
        except SweBenchUnavailable as exc:
            err.print(f"  [bad]harness error[/] {exc}")
            continue
        outcomes.append(outcome)
        flag = "[ok]resolved[/]" if outcome.resolved else "[warn]not resolved[/]"
        detail = f" ({outcome.error})" if outcome.error else ""
        out.print(
            f"  {flag}  {outcome.grade.summary()}  "
            f"{outcome.diff_lines} changed line(s)  calls={outcome.model_calls}  "
            f"{outcome.wall_time_s:.0f}s{detail}"
        )
        if outcome.grade.error:
            err.print(f"    [bad]{outcome.grade.error}[/]")
        for failure in outcome.grade.failures[:3]:
            err.print(f"    still failing: {failure[:100]}")

    if not outcomes:
        raise typer.Exit(1)

    resolved = sum(1 for o in outcomes if o.resolved)
    # How much of this number came from emulated instances, and how it splits.
    #
    # Upstream publishes arm64 for only part of the set, and django -- 231 of
    # the 500 instances -- had no arm64 build in every instance tried. So on
    # Apple silicon roughly half the benchmark runs under emulation, several
    # times slower, and a resolve rate that mixes the two is not a sample of
    # SWE-bench. swe_bench.EMULATION_TIME_FACTOR stops the budget being the
    # thing that decides those instances; it does NOT make them comparable,
    # which is why the split is reported rather than smoothed away.
    emulated = [o for o in outcomes if o.emulated]
    native = [o for o in outcomes if not o.emulated]

    def _rate(group):
        if not group:
            return None
        return round(sum(1 for o in group if o.resolved) / len(group), 4)

    report = {
        "instances": len(outcomes),
        "resolved": resolved,
        "resolve_rate": round(resolved / len(outcomes), 4),
        "emulated_instances": len(emulated),
        "native_instances": len(native),
        "emulated_resolve_rate": _rate(emulated),
        "native_resolve_rate": _rate(native),
        #: The repositories whose instances ran emulated, which is the thing
        #: a reader needs to judge whether the sample is representative --
        #: "half of it was django, emulated" is a different number from the
        #: same rate measured natively.
        "emulated_repos": sorted({o.instance_id.split("__")[0] for o in emulated}),
        "mean_model_calls": round(sum(o.model_calls for o in outcomes) / len(outcomes), 1),
        "mean_wall_time_s": round(sum(o.wall_time_s for o in outcomes) / len(outcomes), 1),
        "trace_dir": str(out_dir),
        "results": [
            {
                "instance_id": o.instance_id, "resolved": o.resolved,
                "fail_to_pass": [o.grade.fail_to_pass_passed, o.grade.fail_to_pass_total],
                "pass_to_pass": [o.grade.pass_to_pass_passed, o.grade.pass_to_pass_total],
                "diff_lines": o.diff_lines, "model_calls": o.model_calls,
                "emulated": o.emulated,
                "wall_time_s": round(o.wall_time_s, 1),
                "error": o.error or o.grade.error,
            }
            for o in outcomes
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    if as_json:
        out.print_json(data=report)
        return
    out.print(
        f"\n{resolved}/{len(outcomes)} resolved ({report['resolve_rate']:.1%})  "
        f"{report['mean_model_calls']:.0f} model calls and "
        f"{report['mean_wall_time_s']:.0f}s per instance"
    )
    if emulated:
        # A SWE-bench number from a machine that emulated part of its sample
        # has to say so, in the same breath as the number.
        native_text = (
            f"{report['native_resolve_rate']:.1%}" if native else "no native instances"
        )
        err.print(
            f"[warn]{len(emulated)}/{len(outcomes)} instances ran under x86_64 "
            f"emulation (no arm64 build): "
            f"{', '.join(report['emulated_repos'])}.[/]\n"
            f"[muted]emulated {report['emulated_resolve_rate']:.1%} vs native "
            f"{native_text} -- quote the split, not the combined rate, and say "
            f"this machine emulated part of the sample.[/]"
        )
    out.print(f"report: {out_dir / 'report.json'}")
