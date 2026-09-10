"""Coverage for agent/eval/terminal_bench.py -- specifically the deadline that
keeps a task from spending past its own budget.

Terminal-Bench enforces its agent timeout with asyncio.wait_for over a
run_in_executor thread, and a thread in the default executor cannot be
cancelled. The harness stops awaiting, scores the task `agent_timeout` and
tears the container down; our call keeps running and keeps spending. On the
first real batch that overran every capped task by 1.7-1.8x, and killed one
outright when its next `docker exec` hit a container that no longer existed.
Both of those are what this file pins down.

Skipped wholesale when terminal-bench isn't installed: it's an optional
benchmark dependency, not something the engine or the CLI needs.
"""
import time

import pytest

pytest.importorskip("terminal_bench")

from agent.eval import terminal_bench as tbench  # noqa: E402
from pathlib import Path  # noqa: E402

def test_task_budget_is_discovered_from_the_datasets_own_task_yaml(tmp_path, monkeypatch):
    """Terminal-Bench never tells the agent what timeout it is enforcing, and
    the caps genuinely differ per task, so a hardcoded number would be wrong
    for most of them."""

    cache = tmp_path / ".cache" / "terminal-bench" / "core" / "0.1.1" / "some-task"
    cache.mkdir(parents=True)
    (cache / "task.yaml").write_text("max_agent_timeout_sec: 360.0\nmax_test_timeout_sec: 60.0\n")
    monkeypatch.setattr(tbench.Path, "home", staticmethod(lambda: tmp_path))

    logging_dir = tmp_path / "run" / "some-task" / "trial" / "agent-logs"

    assert tbench._task_budget_sec(logging_dir) == 360.0
    assert tbench._task_budget_sec(tmp_path / "run" / "absent" / "t" / "agent-logs") is None
    assert tbench._task_budget_sec(None) is None


def test_the_runner_warns_then_stops_once_the_budget_is_spent():
    """Two stages on purpose: the warning is recoverable and gives the agent a
    chance to write its work out, the stop is not, because by then the harness
    has already scored the task and further calls buy nothing."""

    class _FakeContainer:
        def exec_run(self, *a, **k):
            raise AssertionError("should not reach the container past the deadline")

    agent = tbench.OttoTerminalAgent()
    run = agent._runner(type("S", (), {"container": _FakeContainer()})())

    agent._wrap_up_at = time.monotonic() - 1
    agent._deadline = time.monotonic() + 60
    agent._max_seconds = 360
    stdout, stderr, code = run("ls", 10.0)
    assert code == 1 and "time is nearly up" in stderr

    agent._deadline = time.monotonic() - 1
    with pytest.raises(tbench.DeadlineExceeded):
        run("ls", 10.0)


def test_a_removed_container_is_reported_not_raised_as_a_docker_error():
    """The harness tears the container down when its own timeout fires; the
    next exec used to surface as an opaque 404 and killed the whole run."""

    class _GoneContainer:
        def exec_run(self, *a, **k):
            raise RuntimeError('404 Client Error ... No such container: abc123')

    agent = tbench.OttoTerminalAgent()
    run = agent._runner(type("S", (), {"container": _GoneContainer()})())

    with pytest.raises(tbench.ContainerGone):
        run("ls", 10.0)
