"""Wires the golden dataset into Langfuse as a first-class Dataset, so
`otto eval` runs show up as comparable "Dataset Runs" in the Langfuse UI
instead of only a local pass/fail table printed to the terminal. This is
the mechanism Langfuse's SDK actually expects an eval harness to use --
Dataset.run_experiment() (v4) -- rather than something bolted on top of it.

Design:
  - The local `agent/eval/golden/*.json` files stay the single source of
    truth for golden items (git-diffable, readable/reviewable with no
    network access, and what runner.py's offline-friendly run_golden()
    still uses directly).
  - sync_golden_to_langfuse() upserts every local item into a remote
    Langfuse Dataset (DATASET_NAME), keyed by the item's own id, so it is
    idempotent: edit a local JSON file and the next eval run updates the
    same remote item rather than creating a duplicate. Dataset-item
    metadata only carries `domain` and `golden_id` -- deliberately NOT the
    checker source. Langfuse auto-propagates item metadata onto the trace
    as `experiment_item_metadata.<key>` span attributes for observability,
    and those are capped at 200 chars; a checker script is routinely
    longer, so it got silently dropped there (harmless, but noisy -- one
    warning per item per run). The evaluator below instead looks the
    checker up locally by `golden_id`, which is both quieter and more
    robust: correctness no longer depends on a value round-tripping
    through Langfuse at all.
  - run_golden_experiment() runs the real pipeline against that dataset via
    Langfuse's run_experiment(), which traces every item, records a
    pass/fail score per item, and groups the whole run as one named
    "Dataset Run" comparable against every earlier run of the same
    dataset in the Langfuse UI. Plain per-call tracing (already always on,
    see agent/pipeline/run.py) does not give you that grouping or that
    score -- this is what adds it.

Correctness itself is still decided by runner.check_output() -- the
checker executes for real (see that module's docstring on what "multiple
correct answers" means here). Langfuse becomes the place results are
recorded and compared over time; it does not change how "correct" is
decided.

No `agents` anywhere in this module (2026-09-10): see agent/pipeline/
run.py's module docstring for why the graph that replaced the swarm
pipeline has nothing left to size.
"""
from __future__ import annotations

import uuid
from typing import Any

from agent.eval.runner import GoldenItem, check_output, load_golden
from agent.pipeline.run import run_pipeline

DATASET_NAME = "otto-golden"


def _client():
    from langfuse import get_client

    return get_client()


def _ensure_dataset(client: Any) -> None:
    try:
        client.get_dataset(DATASET_NAME)
    except Exception:
        client.create_dataset(
            name=DATASET_NAME,
            description=(
                "Otto's code/math (+ NP-hard) golden set -- execution-checked "
                "against a known-correct answer. Source of truth is "
                "agent/eval/golden/*.json; synced here by "
                "agent.eval.langfuse_sync.sync_golden_to_langfuse()."
            ),
        )


def sync_golden_to_langfuse(*, client: Any = None) -> int:
    """Upsert every local golden item into the Langfuse dataset. Idempotent:
    the golden item's own id is reused as the Langfuse dataset-item id, so
    re-running this after editing a local JSON file updates the same
    remote item instead of creating a duplicate. Returns the item count.
    """
    client = client or _client()
    _ensure_dataset(client)
    items = load_golden()
    for item in items:
        client.create_dataset_item(
            dataset_name=DATASET_NAME,
            id=item.id,
            input=item.prompt,
            metadata={"domain": item.domain, "golden_id": item.id},
        )
    return len(items)


def _by_id() -> dict[str, GoldenItem]:
    """Fresh local lookup by id -- cheap (small JSON files, no network),
    and the reason the evaluator never needs the checker to round-trip
    through Langfuse at all. Called once per evaluator invocation rather
    than cached at import time so an edited local golden file is picked up
    immediately, same as run_golden()'s load_golden() calls elsewhere.
    """
    return {item.id: item for item in load_golden()}


def _task(*, item, **kwargs) -> str:
    final = run_pipeline(
        item.input,
        session_id=f"golden-{item.id}-{uuid.uuid4().hex[:6]}",
    )
    return (final.get("final_output") or "").strip()


def _correctness_evaluator(*, input, output, expected_output=None, metadata=None, **kwargs):
    metadata = metadata or {}
    golden_id = metadata.get("golden_id")
    item = _by_id().get(golden_id)
    if item is None:
        return {
            "name": "golden_pass", "value": False, "data_type": "BOOLEAN",
            "comment": f"no local golden item for id={golden_id!r} (was it renamed or deleted?)",
        }
    passed, evidence = check_output(item, output or "")
    return {
        "name": "golden_pass",
        "value": bool(passed),
        "comment": evidence[:500],
        "data_type": "BOOLEAN",
    }


def run_golden_experiment(
    *,
    only_domain: str | None = None,
    run_name: str | None = None,
    client: Any = None,
):
    """Sync the golden set, then run the real pipeline against it as a
    tracked Langfuse Dataset Run: one experiment per call, comparable
    against every earlier run of the same dataset in the Langfuse UI.

    Returns a langfuse.experiment.ExperimentResult -- `.format()` renders
    it for the terminal, `.dataset_run_url` links straight to the run.
    """
    client = client or _client()
    sync_golden_to_langfuse(client=client)
    dataset = client.get_dataset(DATASET_NAME)

    items = dataset.items
    if only_domain:
        items = [i for i in items if (i.metadata or {}).get("domain") == only_domain]
    if not items:
        raise ValueError(f"no golden items for domain={only_domain!r}")

    return client.run_experiment(
        name=run_name or "otto eval",
        data=items,
        task=lambda *, item, **kwargs: _task(item=item, **kwargs),
        evaluators=[_correctness_evaluator],
        # Deliberately sequential. run_experiment() defaults to
        # max_concurrency=50, which will happily fire that many task()
        # calls at once -- and every one of them goes through the same
        # module-level ROUTER singleton (agent/pipeline/nodes.py). A live
        # run with the default concurrency corrupted several items' output
        # (a few characters sheared off the front of otherwise-correct
        # code -- e.g. "return" arriving as "urn") in a way two isolated,
        # sequential replays of the exact same call never reproduced --
        # consistent with a shared client/stream buffer racing across
        # concurrent calls, not a model or prompt problem. A golden-set
        # eval has no throughput requirement that justifies the risk;
        # revisit only if ROUTER's provider clients are made verifiably
        # concurrency-safe first.
        max_concurrency=1,
    )
