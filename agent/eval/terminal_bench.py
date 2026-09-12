"""Terminal-Bench adapter: Otto as a `tb run --agent-import-path` agent.

Terminal-Bench (Stanford/Laude Institute) gives an agent a task description
and a Docker container, and grades it by running the task's own tests inside
that container afterwards. Everything the task is about -- the repo, the
broken service, the files the grader will look at -- lives in the container
and does not exist on the host, which is the whole reason
agent/pipeline/execution.py exists: this file binds a CommandRunner that
routes every shell-shaped Otto tool through `docker exec` into the task's
container, and then just runs the ordinary pipeline.

Nothing about Otto is special-cased for the benchmark. The graph, the prompts,
the router and the memory engine are exactly what `otto chat` runs; the only
difference is where the commands land.

Usage (needs INCEPTION_API_KEY, a running Docker daemon, and a tool budget
that suits an agentic task rather than a node checking its own arithmetic --
see agent/pipeline/nodes.py's MAX_TOOL_ITERATIONS on why the default of 5 is
right for chat and wrong here):

    OTTO_MAX_TOOL_ITERATIONS=40 uv run tb run \\
        --dataset terminal-bench-core==0.1.1 \\
        --agent-import-path agent.eval.terminal_bench:OttoTerminalAgent \\
        --n-tasks 5

WHAT THIS ADAPTER DOES NOT DO, stated plainly because both would flatter the
results if left implicit:

  * It runs commands with `docker exec`, not by typing into the task's tmux
    session. exec gives real exit codes and cleanly separated stdout/stderr,
    which the tool loop needs and pane-scraping cannot reliably provide. The
    cost is that Terminal-Bench's asciinema recording will not show the
    agent's own commands, only the session it never touched. Tasks that grade
    the *terminal session's* state rather than the container's will therefore
    score as failures for this agent even where the work was done.
  * It reports zero tokens. Otto's pipeline does not thread usage back out of
    the router today, and inventing a number would be worse than a zero that
    is obviously a placeholder.
"""
from __future__ import annotations

import re
import shlex
import time
import uuid
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

from agent.pipeline.budget import Budget, bind_budget
from agent.pipeline.execution import bind_command_runner
from agent.pipeline.run import run_pipeline

#: Appended to the task's own instruction. Terminal-Bench phrases tasks for an
#: agent that is already sitting at a shell in the container; Otto's prompts
#: never assume that, so this says where it is and what "done" means. It is
#: deliberately about the environment, not about how to solve anything -- a
#: benchmark adapter that coaches strategy stops measuring the agent.
_ENVIRONMENT_NOTE = """

You are working inside a Linux container. execute_bash, execute_python,
read_file, write_file, edit_file and list_files all act INSIDE that container,
on the real files the task is about -- not on any local copy. Each shell
command runs on its own, though the working directory carries over from one to
the next.

Finish by actually making the change in the container. An explanation of what
should be done, with the container left untouched, does not complete the task.

Before you give your final answer, prove the task is done by running a command
whose output shows it -- the thing the task asked for, actually working. If
that check does not show what you expected, the task is not finished: keep
going. Do not report success you have not just watched happen.
"""

#: Fraction of the task's own agent budget after which the runner starts
#: telling the agent to wrap up. Terminal-Bench enforces its timeout with
#: asyncio.wait_for over a run_in_executor thread, and a thread in the default
#: executor cannot be cancelled: the harness stops awaiting, scores the task
#: `agent_timeout`, and tears the container down, while our call keeps running
#: and keeps spending. Measured on the first real batch, tasks overran their
#: cap by 1.7-1.8x that way, and one died outright when its next `docker exec`
#: hit a container that no longer existed.
#:
#: So the deadline is enforced on this side, in two stages. At this fraction
#: the agent is told, through an ordinary failed ToolResult, that time is
#: nearly up -- which is recoverable: it can still write its work out and
#: answer. At the full budget it is stopped outright, since the harness has
#: already scored the task by then and everything after that point buys
#: nothing.
WRAP_UP_FRACTION = 0.8

#: Left off the task's own budget, so the stop lands before the harness's
#: teardown rather than racing it.
DEADLINE_MARGIN_SEC = 15.0

#: How long any single command inside the container may run. Generous compared
#: with the tool default because the commands that matter here are real ones --
#: a build, a package install, a test suite -- not the seconds-long snippets
#: execute_bash was originally sized for.
COMMAND_TIMEOUT_SEC = 300.0

#: Printed by every command so the adapter can carry the working directory
#: across `docker exec` calls, which are otherwise each their own process with
#: no memory of a previous `cd`. Unlikely enough to collide with real output,
#: and stripped before the agent ever sees it.
_PWD_MARKER = "__OTTO_PWD__"


class DeadlineExceeded(Exception):
    """Raised out of the command runner once the task's own time is up, to
    unwind the pipeline immediately rather than let it keep calling a model
    for a task the harness has already scored."""


class ContainerGone(Exception):
    """The task container was torn down underneath us -- the harness moved on.
    Same meaning as DeadlineExceeded in practice; kept separate so the log
    says which of the two actually happened."""


def _task_budget_sec(logging_dir: Path | None) -> float | None:
    """The task's own `max_agent_timeout_sec`, discovered from its task.yaml.

    Terminal-Bench does not pass the timeout it is enforcing to the agent, and
    the caps genuinely differ per task (360s, 600s and 2400s all appear in
    terminal-bench-core), so a single hardcoded number would be wrong for most
    of them. The logging dir is `<run>/<task-id>/<trial>/agent-logs`, which
    gives the task id, and the dataset cache holds that task's yaml.

    Returns None when it cannot be found, which means "no self-imposed
    deadline" -- a caller that wants one regardless can pass max_seconds.
    """
    if logging_dir is None:
        return None
    try:
        task_id = logging_dir.parent.parent.name
    except (AttributeError, IndexError):
        return None
    cache = Path.home() / ".cache" / "terminal-bench"
    for yaml_path in cache.glob(f"*/*/{task_id}/task.yaml"):
        match = re.search(r"max_agent_timeout_sec:\s*([\d.]+)", yaml_path.read_text())
        if match:
            return float(match.group(1))
    return None


class OttoTerminalAgent(BaseAgent):
    """Otto's own graph, with its shell tools pointed into the task container."""

    @staticmethod
    def name() -> str:
        return "otto"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cwd = kwargs.get("cwd", "/app")
        self._max_seconds = float(kwargs["max_seconds"]) if "max_seconds" in kwargs else None
        self._deadline: float | None = None
        self._wrap_up_at: float | None = None

    def _runner(self, session: TmuxSession, transcript: Path | None = None):
        """A CommandRunner (agent/pipeline/execution.py) backed by this task's
        container, tracking the working directory across calls.

        `docker exec` starts a fresh process every time, so a `cd` in one
        command would be invisible to the next -- which is not how any shell
        behaves, and not what a model driving one expects. Each command is
        therefore run from the last known directory and asked to report where
        it ended up.

        Every command and its result is appended to `transcript`. Without it a
        failed task shows only the final answer, and the final answer is
        exactly the thing not to trust -- two of the first three failures were
        Otto reporting a fix it had not actually made, which is invisible
        unless you can see what it really ran.
        """
        container = session.container

        def run(command: str, timeout: float) -> tuple[str, str, int]:
            now = time.monotonic()
            if self._deadline is not None and now >= self._deadline:
                raise DeadlineExceeded(f"task budget of {self._max_seconds:.0f}s is spent")
            if self._wrap_up_at is not None and now >= self._wrap_up_at:
                remaining = self._deadline - now if self._deadline else 0
                return (
                    "", f"time is nearly up ({remaining:.0f}s left). Stop exploring, make "
                    "sure the change is actually written into the container, and give "
                    "your final answer now.", 1,
                )

            wrapped = (
                f"cd {shlex.quote(self._cwd)} 2>/dev/null || cd /; "
                f"{command}\n"
                f"__otto_rc=$?; printf '%s%s\\n' '{_PWD_MARKER}' \"$(pwd)\"; "
                f"exit $__otto_rc"
            )
            try:
                result = container.exec_run(
                    ["timeout", str(int(min(timeout, COMMAND_TIMEOUT_SEC))),
                     "bash", "-lc", wrapped],
                    demux=True,
                )
            except Exception as exc:  # docker.errors.NotFound and friends
                if "No such container" in str(exc) or "404" in str(exc):
                    raise ContainerGone("the task container was removed") from exc
                raise
            raw_out, raw_err = result.output if isinstance(result.output, tuple) else (result.output, b"")
            stdout = (raw_out or b"").decode(errors="replace")
            stderr = (raw_err or b"").decode(errors="replace")

            stdout, self._cwd = _split_pwd_marker(stdout, self._cwd)
            # `timeout` reports 124 when it kills the command; the tool loop
            # already understands -1 as "timed out" (agent/pipeline/tools.py).
            code = -1 if result.exit_code == 124 else result.exit_code
            _append_transcript(transcript, command, stdout, stderr, code)
            return stdout, stderr, code

        return run

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        prompt = self._render_instruction(instruction) + _ENVIRONMENT_NOTE
        session_id = f"tbench-{uuid.uuid4().hex[:12]}"

        budget = self._max_seconds or _task_budget_sec(logging_dir)
        if budget is not None:
            self._max_seconds = budget
            usable = max(budget - DEADLINE_MARGIN_SEC, 1.0)
            self._deadline = time.monotonic() + usable
            self._wrap_up_at = time.monotonic() + usable * WRAP_UP_FRACTION

        state = None
        try:
            transcript = (logging_dir / "otto-transcript.txt") if logging_dir else None
            # See agent/pipeline/budget.py: the runner's own checks below
            # cover a single long command, but only a budget checked before
            # every model call can see where a run's time actually goes.
            budget = Budget.until(self._deadline) if self._deadline else None
            with bind_command_runner(self._runner(session, transcript)), bind_budget(budget):
                state = run_pipeline(prompt, session_id=session_id)
        except Exception as exc:  # a harness run must report, never crash out
            # LangGraph wraps whatever a node raised, so match on the text
            # rather than the type -- the two stop conditions are ours and
            # neither is an agent error worth flagging as one.
            stopped = any(
                marker in f"{type(exc).__name__}: {exc}"
                for marker in ("DeadlineExceeded", "ContainerGone")
            )
            if logging_dir is not None:
                (logging_dir / "otto-error.txt").write_text(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                failure_mode=FailureMode.AGENT_TIMEOUT if stopped
                else FailureMode.UNKNOWN_AGENT_ERROR,
            )

        if logging_dir is not None:
            (logging_dir / "otto-prompt.txt").write_text(prompt)
            (logging_dir / "otto-final.txt").write_text(str((state or {}).get("output", "")))

        return AgentResult(
            total_input_tokens=0,   # see the module docstring
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
        )


def _append_transcript(path: Path | None, command: str, out: str, err: str, code: int) -> None:
    """One command and its result, appended. Truncated per field: a transcript
    is for seeing what the agent did, and a `find /` that returned 40,000 lines
    tells you nothing more than its first twenty do."""
    if path is None:
        return
    block = [f"$ {command}", f"[exit {code}]"]
    if out.strip():
        block.append("stdout:\n" + out[:2000])
    if err.strip():
        block.append("stderr:\n" + err[:1000])
    with path.open("a") as handle:
        handle.write("\n".join(block) + "\n" + "-" * 60 + "\n")


def _split_pwd_marker(stdout: str, fallback: str) -> tuple[str, str]:
    """Strip the trailing working-directory marker off a command's output and
    return (clean output, new working directory). A command killed by the
    timeout never prints it, so the previous directory stands."""
    lines = stdout.splitlines()
    for index in range(len(lines) - 1, -1, -1):
        if lines[index].startswith(_PWD_MARKER):
            directory = lines[index][len(_PWD_MARKER):].strip() or fallback
            del lines[index]
            return "\n".join(lines), directory
    return stdout, fallback


class OttoSingleAgent(OttoTerminalAgent):
    """The single-agent control (agent/eval/single_agent.py) behind the same
    container plumbing, so the only difference from OttoTerminalAgent is the
    architecture: one continuous conversation instead of a graph that restarts
    every node from a two-message summary.

        tb run --agent-import-path agent.eval.terminal_bench:OttoSingleAgent
    """

    @staticmethod
    def name() -> str:
        return "otto-single"

    def perform_task(self, instruction, session, logging_dir=None) -> AgentResult:
        from agent.eval.single_agent import run_single_agent

        prompt = self._render_instruction(instruction) + _ENVIRONMENT_NOTE
        budget = self._max_seconds or _task_budget_sec(logging_dir)
        if budget is not None:
            self._max_seconds = budget
            usable = max(budget - DEADLINE_MARGIN_SEC, 1.0)
            self._deadline = time.monotonic() + usable
            self._wrap_up_at = time.monotonic() + usable * WRAP_UP_FRACTION

        transcript = (logging_dir / "otto-transcript.txt") if logging_dir else None
        try:
            with bind_command_runner(self._runner(session, transcript)):
                answer, taken = run_single_agent(prompt)
        except Exception as exc:
            stopped = any(
                marker in f"{type(exc).__name__}: {exc}"
                for marker in ("DeadlineExceeded", "ContainerGone")
            )
            if logging_dir is not None:
                (logging_dir / "otto-error.txt").write_text(f"{type(exc).__name__}: {exc}")
            return AgentResult(
                failure_mode=FailureMode.AGENT_TIMEOUT if stopped
                else FailureMode.UNKNOWN_AGENT_ERROR,
            )

        if logging_dir is not None:
            (logging_dir / "otto-final.txt").write_text(answer)
            (logging_dir / "otto-actions.txt").write_text("\n".join(taken))
        return AgentResult(failure_mode=FailureMode.NONE)
