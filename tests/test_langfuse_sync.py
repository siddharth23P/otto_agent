"""Coverage for agent/eval/langfuse_sync.py against a fake Langfuse client --
no network, no real Langfuse project needed. Checks the four things that
would silently break the wiring without raising anywhere obvious: dataset
items get the golden item's own id (so re-syncing is idempotent) and carry
only small (domain, golden_id) metadata -- NOT the checker source, which
is what triggered Langfuse's 200-char span-attribute warning in real use
-- run_golden_experiment() filters by domain against that metadata, and
the evaluator it hands to run_experiment() looks the checker up locally by
golden_id and actually runs it, rather than depending on anything that
round-tripped through Langfuse.

No `agents` anywhere in this module or these tests (2026-09-10): the
router/planner/solver/summarizer/finder/evaluator graph that replaced the
swarm pipeline has nothing to size -- see agent/pipeline/run.py's module
docstring.
"""
from dataclasses import dataclass, field
from typing import Any

from agent.eval import langfuse_sync as ls


@dataclass
class _FakeDatasetItem:
    id: str
    input: str
    metadata: dict


@dataclass
class _FakeDataset:
    items: list


@dataclass
class _FakeClient:
    created_items: list = field(default_factory=list)
    dataset_exists: bool = True
    run_experiment_calls: list = field(default_factory=list)

    def get_dataset(self, name):
        if not self.dataset_exists:
            raise RuntimeError("not found")
        return _FakeDataset(items=[
            _FakeDatasetItem(id=c["id"], input=c["input"], metadata=c["metadata"])
            for c in self.created_items
        ])

    def create_dataset(self, *, name, description=None):
        self.dataset_exists = True

    def create_dataset_item(self, *, dataset_name, id, input, metadata):
        self.created_items = [c for c in self.created_items if c["id"] != id]
        self.created_items.append({"id": id, "input": input, "metadata": metadata})

    def run_experiment(self, *, name, data, task, evaluators, metadata=None, run_name=None, max_concurrency=50):
        self.run_experiment_calls.append({
            "name": name, "data": data, "metadata": metadata, "max_concurrency": max_concurrency,
        })
        # Mimic just enough of the real framework to exercise task+evaluator.
        results = []
        for item in data:
            output = task(item=item)
            evals = [ev(input=item.input, output=output, metadata=item.metadata) for ev in evaluators]
            results.append((item, output, evals))
        return results


def test_sync_upserts_every_local_golden_item_with_small_metadata_only():
    client = _FakeClient(dataset_exists=False)
    n = ls.sync_golden_to_langfuse(client=client)

    assert n == len(client.created_items)
    ids = {c["id"] for c in client.created_items}
    assert "code_01" in ids and "nphard_tsp_01" in ids  # both an original and a new NP-hard item
    nphard = next(c for c in client.created_items if c["id"] == "nphard_ksp_01")
    assert nphard["metadata"] == {"domain": "code", "golden_id": "nphard_ksp_01"}
    # The checker source (over 200 chars, every real item) must NOT be in
    # the metadata Langfuse propagates onto trace attributes -- that's
    # exactly what triggered the "dropping value" warning in real use.
    for c in client.created_items:
        assert "checker" not in c["metadata"]


def test_sync_is_idempotent_by_id():
    client = _FakeClient(dataset_exists=False)
    ls.sync_golden_to_langfuse(client=client)
    first_count = len(client.created_items)
    ls.sync_golden_to_langfuse(client=client)
    assert len(client.created_items) == first_count  # re-sync updates, doesn't duplicate


def test_run_golden_experiment_filters_by_domain_from_dataset_item_metadata():
    client = _FakeClient(dataset_exists=False)

    def fake_run_pipeline(text, *, session_id):
        return {"final_output": "def is_prime(n):\n    if n < 2: return False\n    return all(n % d for d in range(2, int(n**0.5)+1))\n"}

    orig = ls.run_pipeline
    ls.run_pipeline = fake_run_pipeline
    try:
        result = ls.run_golden_experiment(only_domain="math", client=client)
    finally:
        ls.run_pipeline = orig

    call = client.run_experiment_calls[0]
    assert all(item.metadata["domain"] == "math" for item in call["data"])
    assert len(call["data"]) > 0


def test_run_golden_experiment_forces_sequential_execution():
    # A live run at the default max_concurrency=50 corrupted several items'
    # output (characters sheared off the front of otherwise-correct code)
    # in a way sequential replays of the exact same call never reproduced
    # -- every module-level ROUTER call in a concurrent run shares state
    # that was never verified concurrency-safe. This must stay pinned to 1
    # until that's actually fixed, not silently drift back to a default.
    client = _FakeClient(dataset_exists=False)

    def fake_run_pipeline(text, *, session_id):
        return {"final_output": "OK"}

    orig = ls.run_pipeline
    ls.run_pipeline = fake_run_pipeline
    try:
        ls.run_golden_experiment(client=client)
    finally:
        ls.run_pipeline = orig

    assert client.run_experiment_calls[0]["max_concurrency"] == 1


def test_evaluator_looks_up_the_checker_locally_by_golden_id_and_actually_runs_it():
    # A correct is_prime should pass code_01's real checker (loaded from
    # agent/eval/golden/code_01.json, not from anything Langfuse carried);
    # a broken one should fail it.
    good = "def is_prime(n):\n    if n < 2: return False\n    return all(n % d for d in range(2, int(n**0.5)+1))\n"
    bad = "def is_prime(n):\n    return True\n"
    metadata = {"domain": "code", "golden_id": "code_01"}

    ev_good = ls._correctness_evaluator(input="prompt", output=good, metadata=metadata)
    ev_bad = ls._correctness_evaluator(input="prompt", output=bad, metadata=metadata)

    assert ev_good["value"] is True
    assert ev_bad["value"] is False


def test_evaluator_reports_failure_rather_than_crashing_on_an_unknown_id():
    result = ls._correctness_evaluator(
        input="prompt", output="anything", metadata={"domain": "code", "golden_id": "does-not-exist"}
    )
    assert result["value"] is False
    assert "does-not-exist" in result["comment"]
