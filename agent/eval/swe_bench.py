"""SWE-bench Verified: 500 real GitHub issues, graded by the repository's own
test suite.

The benchmark Otto most needed and did not have. Every other harness here
grades an ANSWER -- Claw-Eval with a model judge, the golden set with a
checker somebody wrote, HLE against a reference string. This one grades a
DIFF by running the tests the maintainers wrote for the bug, which is the only
signal in the whole eval surface that nobody involved can argue with.

HOW AN INSTANCE WORKS. Each row names a repository, a commit, the issue text
as it was actually filed, and two lists of test node ids: FAIL_TO_PASS, which
the real fix made pass, and PASS_TO_PASS, which it had to leave alone. The
official images -- one per instance -- carry the repo checked out at that
commit with its environment already built, which is the only practical way to
run twelve codebases' worth of dependencies. Otto gets the issue text and the
container; it does not get the tests, the patch, or the hints.

  resolved = every FAIL_TO_PASS passes AND every PASS_TO_PASS still passes

Both halves matter and the second is the interesting one: an agent that
deletes the failing assertion passes the first and fails the second, which is
why "did the tests go green" is not the same question as "was the bug fixed".

WHAT OTTO IS AND IS NOT TOLD. The issue text and the repository, which is what
a maintainer would have. Not the test names -- an agent told which test to
make pass writes to the test rather than to the bug, and the resulting number
measures nothing. The test patch is applied AFTER Otto stops, for the same
reason Claw-Eval injects grader files after the agent finishes.

ARM64. The official images publish an arm64 variant alongside x86_64, so this
runs natively on Apple silicon rather than under emulation. `image_for`
prefers the host's architecture and says so when it falls back, because a
benchmark silently running under qemu would report timeouts as failures.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
import platform
import re
import shlex
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: The cached dataset. Fetched once from Hugging Face's parquet endpoint; no
#: `datasets` dependency, and no token -- SWE-bench is not gated.
DEFAULT_DATASET = Path(__file__).resolve().parent / "data" / "swe-bench-verified.parquet"

PARQUET_URL = (
    "https://huggingface.co/api/datasets/princeton-nlp/SWE-bench_Verified"
    "/parquet/default/test/0.parquet"
)

#: Where the official images put the checked-out repository.
WORKDIR = "/testbed"

#: How long one command may run inside the container. Test suites in these
#: repositories are slow; a cap that suits a shell command would kill the
#: grading run itself.
COMMAND_TIMEOUT_S = 900

#: Grading's own budget, separate from the agent's. Some of these suites take
#: several minutes and one of them has 644 tests in its PASS_TO_PASS set; a
#: grader sharing the agent's clock scores "out of time" instead of scoring
#: the work.
GRADING_BUDGET_S = 2400.0

#: Default wall-clock budget for one instance's AGENT phase, before grading.
#: Generous because these are real bugs in unfamiliar codebases, and bounded
#: because a stuck run must not hold a container open forever.
DEFAULT_MAX_SECONDS = 1800.0

_PWD_MARKER = "__OTTO_PWD__"


class SweBenchUnavailable(Exception):
    """The dataset or Docker is not usable, said as a sentence."""


# --------------------------------------------------------------------------
# The dataset
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Instance:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    difficulty: str = ""

    @property
    def image(self) -> str:
        return image_for(self.instance_id)

    @property
    def runnable_image(self) -> tuple[str, bool]:
        return resolve_image(self.instance_id)


def _as_list(value) -> list[str]:
    """FAIL_TO_PASS arrives as a JSON string in the parquet, and as a list
    from some mirrors. Accept both rather than depending on which."""
    if isinstance(value, str):
        try:
            return list(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return [value]
    return list(value or [])


def load_dataset(path: Path | None = None, *, download: bool = True) -> list[Instance]:
    path = Path(path or DEFAULT_DATASET)
    if not path.exists():
        if not download:
            raise SweBenchUnavailable(
                f"no dataset at {path}. Re-run without --no-download, or fetch "
                f"{PARQUET_URL} into that path."
            )
        _download(path)
    try:
        import pandas
    except ImportError as exc:  # pragma: no cover - environment problem
        raise SweBenchUnavailable(
            "reading the dataset needs pandas and pyarrow: uv pip install pandas pyarrow"
        ) from exc

    rows = pandas.read_parquet(path).to_dict("records")
    return [
        Instance(
            instance_id=str(r["instance_id"]),
            repo=str(r["repo"]),
            base_commit=str(r["base_commit"]),
            problem_statement=str(r["problem_statement"]),
            test_patch=str(r["test_patch"]),
            fail_to_pass=_as_list(r.get("FAIL_TO_PASS")),
            pass_to_pass=_as_list(r.get("PASS_TO_PASS")),
            difficulty=str(r.get("difficulty") or ""),
        )
        for r in rows
    ]


def _download(path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("fetching SWE-bench Verified to %s", path)
    try:
        with urllib.request.urlopen(PARQUET_URL, timeout=120) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        raise SweBenchUnavailable(f"could not fetch the dataset: {exc}") from exc


def image_name(instance_id: str, arch: str) -> str:
    """The official image name for one architecture.

    The id is lowercased and its `__` separator becomes `_1776_`, which is the
    upstream convention -- Docker tags cannot carry a double underscore in the
    position SWE-bench's ids use it.
    """
    slug = instance_id.lower().replace("__", "_1776_")
    return f"swebench/sweb.eval.{arch}.{slug}:latest"


def host_arch() -> str:
    return "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x86_64"


def image_for(instance_id: str) -> str:
    """This machine's architecture. What to PULL is `resolve_image` below."""
    return image_name(instance_id, host_arch())


#: How much longer an emulated instance gets than a native one.
#:
#: Emulation is "several times slower" (resolve_image below), so a budget
#: tuned for native turns a machine problem into a recorded agent failure.
#: Three is deliberately a round guess rather than a measurement: the honest
#: fix for the emulated half is to run it on x86_64, and this only stops the
#: budget being the thing that decides the score. It does NOT make an
#: emulated number comparable to a native one, which is why every report
#: says how many of its instances were emulated.
EMULATION_TIME_FACTOR = 3.0


def resolve_image(instance_id: str) -> tuple[str, bool]:
    """(image, emulated) -- the arm64 build when one exists, else x86_64.

    Upstream publishes arm64 for only part of the set: of six instances tried
    at random, two had one and four did not. Without this the harness reports
    "pull access denied" for two thirds of the benchmark and calls it a
    harness error, which reads as a broken integration rather than as a
    missing build.

    An x86_64 image on Apple silicon runs under emulation, several times
    slower, so a time budget tuned for native will kill work that was going
    fine. The flag exists so the caller can say so and widen the budget
    rather than reporting a timeout as a failure.
    """
    native = image_name(instance_id, host_arch())
    if host_arch() == "x86_64" or _manifest_exists(native):
        return native, False
    return image_name(instance_id, "x86_64"), True


@lru_cache(maxsize=None)
def _manifest_exists(image: str) -> bool:
    """Whether the registry has this image, asked without downloading 3.5GB
    to find out.

    CACHED on the IMAGE NAME, which is the expensive part -- one `docker
    manifest inspect` round trip per miss, and `resolve_image` is now asked
    twice per instance (once to size the deadline, once by start_container).
    Deliberately not cached on `resolve_image` itself: that would capture
    `host_arch()` in the key, and a test monkeypatching the architecture
    then reads a previous test's answer. The registry's answer for a given
    image name does not change inside one run.
    """
    try:
        return _docker("manifest", "inspect", image, timeout=90).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def select(instances: list[Instance], *, repo: str | None = None,
           pattern: str | None = None, difficulty: str | None = None,
           limit: int | None = None) -> list[Instance]:
    chosen = instances
    if repo:
        chosen = [i for i in chosen if repo in i.repo]
    if pattern:
        chosen = [i for i in chosen if pattern in i.instance_id]
    if difficulty:
        chosen = [i for i in chosen if i.difficulty.startswith(difficulty)]
    return chosen[:limit] if limit else chosen


# --------------------------------------------------------------------------
# The container, as a CommandRunner
# --------------------------------------------------------------------------

def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout)


def start_container(instance: Instance, *, name: str | None = None) -> tuple[str, bool]:
    """Start the instance's image. Returns (container name, emulated)."""
    name = name or f"otto-swe-{uuid.uuid4().hex[:10]}"
    image, emulated = instance.runnable_image
    args = ["run", "-d", "--name", name, "-w", WORKDIR]
    if emulated:
        args += ["--platform", "linux/amd64"]
    result = _docker(*args, image, "sleep", "infinity", timeout=600)
    if result.returncode != 0:
        raise SweBenchUnavailable(
            f"could not start {image}: {result.stderr.strip()[:300]}"
        )
    return name, emulated


def stop_container(name: str) -> None:
    _docker("rm", "-f", name, timeout=120)


def container_runner(name: str, deadline: "Deadline"):
    """A CommandRunner (agent/pipeline/execution.py) backed by this instance's
    container.

    The working directory is tracked across calls for the same reason
    terminal_bench.py tracks it: every `docker exec` is a fresh process, so a
    `cd` in one command would be invisible to the next, which is not how any
    shell behaves.
    """
    state = {"cwd": WORKDIR}

    def run(command: str, timeout: float) -> tuple[str, str, int]:
        refusal = deadline.refuse()
        if refusal is not None:
            return refusal
        wrapped = (
            f"cd {shlex.quote(state['cwd'])} 2>/dev/null || cd {WORKDIR}; "
            f"{command}\n"
            f"__otto_rc=$?; printf '%s%s\\n' '{_PWD_MARKER}' \"$(pwd)\"; "
            f"exit $__otto_rc"
        )
        cap = int(min(timeout or COMMAND_TIMEOUT_S, COMMAND_TIMEOUT_S))
        result = _docker("exec", name, "bash", "-lc",
                         f"timeout {cap} bash -lc {shlex.quote(wrapped)}",
                         timeout=cap + 30)
        stdout, state["cwd"] = _split_marker(result.stdout, state["cwd"])
        code = -1 if result.returncode == 124 else result.returncode
        return stdout, result.stderr, code

    return run


def _split_marker(stdout: str, fallback: str) -> tuple[str, str]:
    if _PWD_MARKER not in stdout:
        return stdout, fallback
    head, _, tail = stdout.rpartition(_PWD_MARKER)
    cwd = tail.strip().splitlines()[0].strip() if tail.strip() else fallback
    return head.rstrip("\n"), cwd or fallback


@dataclass
class Deadline:
    """Wall-clock budget for one instance, with a wrap-up warning before the
    hard stop -- the same shape claw_bench.py uses, and for the same reason: a
    run killed mid-command reports nothing, one that is told to finish hands
    back something gradable."""

    hard_at: float
    wrap_up_at: float
    warned: bool = False

    @classmethod
    def of(cls, seconds: float, *, wrap_up: float = 0.85) -> "Deadline":
        now = time.monotonic()
        return cls(hard_at=now + seconds, wrap_up_at=now + seconds * wrap_up)

    def refuse(self) -> tuple[str, str, int] | None:
        now = time.monotonic()
        if now >= self.hard_at:
            return ("", "the time budget for this instance is spent -- stop and "
                        "report what you changed", 1)
        if now >= self.wrap_up_at and not self.warned:
            self.warned = True
            return ("", f"about {self.hard_at - now:.0f}s left. Make sure the edit is "
                        "actually written to disk, then finish.", 1)
        return None


# --------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------

PROMPT = (
    "You are working in a checkout of {repo} at {WORKDIR}. The environment is "
    "already installed; do not reinstall it.\n\n"
    "Fix the issue below by editing the source. Do not edit or add tests -- "
    "the project's own test suite will be run against your change, and a "
    "change to a test is not a fix.\n\n"
    "ISSUE:\n{problem}\n\n"
    "When you are done, say what you changed and why in one short paragraph. "
    "The files on disk are what is graded, not your description of them."
)


def build_prompt(instance: Instance) -> str:
    return PROMPT.format(repo=instance.repo, WORKDIR=WORKDIR,
                         problem=instance.problem_statement.strip())


# --------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------

#: How a repository's tests are invoked. Every SWE-bench repo is pytest or
#: unittest under the hood, and the official harness uses each project's own
#: entry point; pytest reads unittest node ids too, which is what lets one
#: command cover both.
#: `--color=no` asks politely; `parse_report` strips the escapes anyway,
#: because a project can force colour back on from its own config.
#:
#: Tests are named by FILE, not by node id, and that is the difference
#: between a working grader and one that reports nonsense. Given explicit
#: node ids pytest is all-or-nothing: if ONE of them does not resolve -- a
#: parametrisation that moved, an id the dataset recorded from a slightly
#: different tree -- it prints "no tests ran" and every id in the batch
#: scores as not-passed. Measured on astropy-13236: 644 PASS_TO_PASS tests,
#: one bad id, 0/644 on an UNCHANGED repository.
#:
#: Running the files and looking the ids up afterwards costs some extra
#: tests and cannot be sabotaged by one stale name.
#: `--continue-on-collection-errors` is the same argument one level up: a
#: file that will not import should cost its own tests, not the whole run.
TEST_COMMAND = ("python -m pytest -rA --tb=no --color=no -p no:cacheprovider "
                "--continue-on-collection-errors {tests}")

#: How many test FILES go in one pytest invocation. Bounded so a suite that
#: hangs costs one batch rather than the whole grading run.
FILES_PER_BATCH = 20


def files_of(node_ids) -> list[str]:
    """The distinct files named by these node ids, in first-seen order."""
    seen: dict[str, None] = {}
    for node in node_ids:
        seen.setdefault(node.split("::", 1)[0], None)
    return list(seen)

#: Terminal colour and cursor escapes. pytest colourises its summary whenever
#: a project's own config asks it to, regardless of whether anything is
#: attached to the output.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_STATUS = re.compile(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", re.M)


def parse_report(output: str) -> dict[str, str]:
    """Test node id -> status, from pytest's `-rA` short summary.

    Parsed rather than trusting the exit code: a run where one unrelated test
    errors out has a non-zero exit and can still have every graded test
    passing, and the opposite is just as possible.

    THE ESCAPES COME OFF FIRST, and that is not defensive tidying. astropy's
    own setup.cfg turns colour on, so the summary arrives as
    `\x1b[32mPASSED\x1b[0m astropy/...::\x1b[1mTestSingleTable::test_simple`
    -- the status no longer starts the line and the node id has an escape in
    the middle of it. Both halves of the pattern miss, every test reads as
    not-passed, and the harness reports 0 of 141 passing on a repository
    where nothing is wrong. Measured: that is exactly what two instances
    scored before this line existed.
    """
    return {node: status for status, node in _STATUS.findall(_ANSI.sub("", output))}


@dataclass
class Grade:
    resolved: bool
    fail_to_pass_passed: int
    fail_to_pass_total: int
    pass_to_pass_passed: int
    pass_to_pass_total: int
    error: str = ""
    #: Node ids that should pass and did not. The first thing to read.
    failures: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"F2P {self.fail_to_pass_passed}/{self.fail_to_pass_total} "
                f"P2P {self.pass_to_pass_passed}/{self.pass_to_pass_total}")


def grade(name: str, instance: Instance, *, run) -> Grade:
    """Apply the test patch, run both test sets, and decide.

    The test patch goes in HERE, after the agent has stopped, so the tests are
    never visible while the work is being done.
    """
    if instance.test_patch.strip():
        applied = run(
            "cat > /tmp/otto_test.patch <<'OTTO_PATCH_EOF'\n"
            + instance.test_patch
            + "\nOTTO_PATCH_EOF\n"
            f"cd {WORKDIR} && git apply -v /tmp/otto_test.patch",
            300,
        )
        if applied[2] != 0:
            return Grade(False, 0, len(instance.fail_to_pass), 0,
                         len(instance.pass_to_pass),
                         error=f"test patch would not apply: {applied[1][:200]}")

    wanted = instance.fail_to_pass + instance.pass_to_pass
    if not wanted:
        return Grade(False, 0, 0, 0, 0, error="the instance names no tests")

    files = files_of(wanted)
    report: dict[str, str] = {}
    for start in range(0, len(files), FILES_PER_BATCH):
        batch = files[start:start + FILES_PER_BATCH]
        stdout, stderr, _ = run(
            TEST_COMMAND.format(tests=" ".join(shlex.quote(f) for f in batch)),
            COMMAND_TIMEOUT_S,
        )
        report.update(parse_report(stdout + "\n" + stderr))

    f2p = [t for t in instance.fail_to_pass if report.get(t) == "PASSED"]
    p2p = [t for t in instance.pass_to_pass if report.get(t) == "PASSED"]
    missed = ([t for t in instance.fail_to_pass if report.get(t) != "PASSED"]
              + [t for t in instance.pass_to_pass if report.get(t) != "PASSED"])
    return Grade(
        resolved=len(f2p) == len(instance.fail_to_pass)
                 and len(p2p) == len(instance.pass_to_pass),
        fail_to_pass_passed=len(f2p), fail_to_pass_total=len(instance.fail_to_pass),
        pass_to_pass_passed=len(p2p), pass_to_pass_total=len(instance.pass_to_pass),
        failures=missed[:10],
    )


def diff_of(run) -> str:
    """What the agent actually changed, for the record. The one artefact worth
    keeping from a failed instance."""
    stdout, _, _ = run(f"cd {WORKDIR} && git diff", 120)
    return stdout


# --------------------------------------------------------------------------
# One instance, end to end
# --------------------------------------------------------------------------

@dataclass
class InstanceOutcome:
    instance_id: str
    resolved: bool
    grade: Grade
    model_calls: int
    wall_time_s: float
    actions: int
    diff_lines: int
    #: Ran under x86_64 emulation because no arm64 build exists. Several times
    #: slower, so a timeout here is about the machine, not the agent.
    emulated: bool = False
    answer: str = ""
    error: str = ""
    diff: str = ""


def run_instance(
    instance: Instance,
    *,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    max_model_calls: int | None = None,
    trace_dir: Path | None = None,
) -> InstanceOutcome:
    """Start the container, run Otto on the issue, then grade.

    Otto's own graph, prompts, router and memory -- the only difference from
    `otto chat` is that every shell-shaped tool lands in the container, which
    is the whole point of the command-runner seam.
    """
    from agent.pipeline.budget import Budget, bind_budget, default_budget
    from agent.pipeline.execution import bind_command_runner
    from agent.pipeline.run import run_pipeline
    from agent.pipeline.workspace import bind_workspace

    started = time.monotonic()
    # Which image this instance will run on has to be settled BEFORE the
    # deadline, because it decides how long the deadline should be. An
    # x86_64 image under emulation on Apple silicon is several times slower,
    # and django -- 231 of the 500 instances -- had no arm64 build in every
    # instance tried. At a budget tuned for native, roughly half the
    # benchmark was being killed by the machine and recorded as the agent
    # failing. See EMULATION_TIME_FACTOR.
    _image, emulated = instance.runnable_image
    deadline = Deadline.of(max_seconds * (EMULATION_TIME_FACTOR if emulated else 1.0))
    name, _ = start_container(instance)
    answer = error = ""
    state: dict = {}
    try:
        run = container_runner(name, deadline)
        budget = (Budget(max_model_calls=max_model_calls) if max_model_calls
                  else default_budget())
        # The workspace is bound to a HOST temp dir that nothing uses: the
        # file tools route through the command runner, and leaving the
        # workspace unbound would let a stray path resolve against the repo
        # Otto itself is running from.
        import tempfile

        with tempfile.TemporaryDirectory(prefix="otto-swe-") as scratch:
            with bind_workspace(scratch), bind_command_runner(run), bind_budget(budget):
                try:
                    state = run_pipeline(
                        build_prompt(instance),
                        session_id=f"swe-{instance.instance_id}-{uuid.uuid4().hex[:6]}",
                    ) or {}
                    answer = str(state.get("final_output") or state.get("output") or "")
                except Exception as exc:  # noqa: BLE001 -- one instance must not end the run
                    error = f"{type(exc).__name__}: {exc}"
                    logger.warning("%s raised: %s", instance.instance_id, error)

        # Grading gets its OWN runner with no deadline. `run` carries the
        # AGENT's clock, and once that is spent it refuses every command --
        # including `git apply` and the test suite. An instance that used its
        # full budget was therefore scored "test patch would not apply: the
        # time budget for this instance is spent", which is not a verdict
        # about the work at all. Measured: astropy-13398 spent 1649s, changed
        # nothing, and was graded on a refusal rather than on its own tests.
        ungated = container_runner(name, Deadline.of(GRADING_BUDGET_S))
        # Read the diff BEFORE the test patch goes in, so it is the agent's
        # work and not the agent's work plus the grader's.
        diff = diff_of(ungated)
        verdict = grade(name, instance, run=ungated)
    finally:
        stop_container(name)

    if trace_dir:
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f"{instance.instance_id}.diff").write_text(diff)

    return InstanceOutcome(
        instance_id=instance.instance_id,
        emulated=emulated,
        resolved=verdict.resolved,
        grade=verdict,
        model_calls=int(state.get("model_calls") or 0),
        wall_time_s=time.monotonic() - started,
        actions=len(state.get("actions") or ()),
        diff_lines=sum(1 for line in diff.splitlines()
                       if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))),
        answer=answer, error=error, diff=diff,
    )
