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

import shlex
import uuid
from pathlib import Path

from terminal_bench.agents.base_agent import AgentResult, BaseAgent
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.terminal.tmux_session import TmuxSession

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
"""

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


class OttoTerminalAgent(BaseAgent):
    """Otto's own graph, with its shell tools pointed into the task container."""

    @staticmethod
    def name() -> str:
        return "otto"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cwd = kwargs.get("cwd", "/app")

    def _runner(self, session: TmuxSession):
        """A CommandRunner (agent/pipeline/execution.py) backed by this task's
        container, tracking the working directory across calls.

        `docker exec` starts a fresh process every time, so a `cd` in one
        command would be invisible to the next -- which is not how any shell
        behaves, and not what a model driving one expects. Each command is
        therefore run from the last known directory and asked to report where
        it ended up.
        """
        container = session.container

        def run(command: str, timeout: float) -> tuple[str, str, int]:
            wrapped = (
                f"cd {shlex.quote(self._cwd)} 2>/dev/null || cd /; "
                f"{command}\n"
                f"__otto_rc=$?; printf '%s%s\\n' '{_PWD_MARKER}' \"$(pwd)\"; "
                f"exit $__otto_rc"
            )
            result = container.exec_run(
                ["timeout", str(int(min(timeout, COMMAND_TIMEOUT_SEC))),
                 "bash", "-lc", wrapped],
                demux=True,
            )
            raw_out, raw_err = result.output if isinstance(result.output, tuple) else (result.output, b"")
            stdout = (raw_out or b"").decode(errors="replace")
            stderr = (raw_err or b"").decode(errors="replace")

            stdout, self._cwd = _split_pwd_marker(stdout, self._cwd)
            # `timeout` reports 124 when it kills the command; the tool loop
            # already understands -1 as "timed out" (agent/pipeline/tools.py).
            code = -1 if result.exit_code == 124 else result.exit_code
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

        try:
            with bind_command_runner(self._runner(session)):
                state = run_pipeline(prompt, session_id=session_id)
        except Exception as exc:  # a harness run must report, never crash out
            if logging_dir is not None:
                (logging_dir / "otto-error.txt").write_text(f"{type(exc).__name__}: {exc}")
            return AgentResult(failure_mode=FailureMode.UNKNOWN_AGENT_ERROR)

        if logging_dir is not None:
            (logging_dir / "otto-prompt.txt").write_text(prompt)
            (logging_dir / "otto-final.txt").write_text(str(state.get("output", "")))

        return AgentResult(
            total_input_tokens=0,   # see the module docstring
            total_output_tokens=0,
            failure_mode=FailureMode.NONE,
        )


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
