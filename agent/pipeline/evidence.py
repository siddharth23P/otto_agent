"""What a run actually PROVED, as opposed to what it said.

A ledger and a policy, and deliberately neither a runner nor a judge. It never
executes anything, and it never decides whether an answer is right. All it
knows is a structural fact: code was changed, and nothing has been run since
that would have failed if the change were wrong.

WHY THIS EXISTS BESIDE THE EVALUATOR. Otto already spends two model calls at
the end of a run asking a model whether the answer holds -- criteria written
from the task, then scored. That is the right instrument for "is this answer
correct", and it is the wrong instrument for "was anything checked at all",
because the model is being asked to notice an absence in its own work. The
absence is visible in the action record for free.

So this costs no model call in the ordinary case. It costs one extra exchange
in exactly the case that deserves it: the run edited code and is trying to
finish without having run anything.

PASSIVE, AND BOUNDED. It cannot block a run. `needs_check` fires at most once
(the caller holds the flag), and the note it produces explicitly accepts "there
is nothing to run here" as an answer -- an agent that cannot route around a
guard lies to it instead, which is the measured failure of enforcement that
has no way to say no: one system blocked 94% of non-compliant actions while
its safe-success rate stayed under 5%, because the agent hallucinated its way
past the block.

PROSE IS EXEMPT. A README or a design note has no runtime behaviour, and
demanding a verification command for one is how a guard teaches an agent to
run something meaningless to satisfy it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Extensions with no runtime behaviour to check. A turn that touched only
#: these never triggers anything.
PROSE_SUFFIXES = frozenset({
    ".md", ".markdown", ".mdx", ".rst", ".txt", ".text", ".adoc", ".org",
    ".log", ".csv", ".tsv", ".json5",
})

#: Extension-less files that are prose by convention.
PROSE_NAMES = frozenset({
    "license", "licence", "notice", "authors", "contributors", "changelog",
    "codeowners", "readme", "todo",
})

#: Tools that change a file's contents. Not the same list as TOOL_TIERS'
#: mutating set, which is about irreversibility -- `execute_bash` mutates and
#: is not an edit, because the ledger cannot tell what a shell command touched.
EDIT_TOOLS = frozenset({"write_file", "edit_file", "predict_edit"})

#: Tools that can constitute a check. Both run something; whether a particular
#: invocation counts is decided by `is_check` below.
RUN_TOOLS = frozenset({"execute_bash", "execute_python"})

#: Command words that mean "this would have failed if the code were wrong".
#: Not an attempt at completeness -- a list that tried to name every build
#: system would be wrong more often than this one, and being wrong here costs
#: a wasted exchange. Everything absent falls through to the general rule
#: below, which is what actually carries most languages.
CHECK_WORDS = (
    "pytest", "unittest", "nosetests", "tox", "nox",
    "jest", "vitest", "mocha", "ava", "playwright", "cypress",
    "cargo", "gradle", "mvn", "dotnet", "rspec", "phpunit",
    "ctest", "bazel", "make", "ninja", "cmake",
    "tsc", "mypy", "pyright", "ruff", "flake8", "pylint", "eslint",
    "clippy", "shellcheck", "golangci-lint", "vet",
)

#: Sub-commands that make a general-purpose runner a check.
CHECK_PHRASES = (
    "npm test", "npm run test", "pnpm test", "yarn test", "bun test",
    "go test", "go build", "go vet", "cargo test", "cargo check",
    "python -m", "uv run", "npm run build", "pnpm build", "make test",
    "gh pr checks", "pre-commit run",
)

_WORD = re.compile(r"[A-Za-z0-9_.\-]+")


def is_prose(path: str) -> bool:
    """Whether editing this path could not possibly need a check."""
    name = path.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    if not name:
        return True
    if "." in name:
        return name[name.rindex("."):] in PROSE_SUFFIXES
    return name in PROSE_NAMES


def is_check(tool: str, body: str, returncode: int) -> bool:
    """Whether this call is evidence that the code still works.

    A FAILING check is not evidence, and deliberately does not count: a run
    that ends on a red test suite has not proved anything, and treating the
    attempt as the proof is exactly the confusion this file exists to remove.

    `execute_python` always counts when it exits cleanly. Running code IS the
    check for a script, and the whole reason the general rule is here: a list
    of build-system names cannot keep up, but "you ran it and it did not
    explode" holds in every language.
    """
    if tool not in RUN_TOOLS or returncode != 0:
        return False
    if tool == "execute_python":
        return True
    lowered = body.lower()
    if any(phrase in lowered for phrase in CHECK_PHRASES):
        return True
    return bool(set(_WORD.findall(lowered)) & set(CHECK_WORDS))


@dataclass
class Ledger:
    """One run's record of edits and proofs, in the order they happened.

    Per run, not persisted. What a previous run proved says nothing about the
    file this one just changed.
    """

    #: Code paths edited since the last passing check, in first-touched order.
    unproven: list[str] = field(default_factory=list)
    #: Whether anything at all has been checked, for a clearer note.
    ever_checked: bool = False

    def record(self, tool: str, body: str, returncode: int) -> None:
        if is_check(tool, body, returncode):
            self.ever_checked = True
            self.unproven.clear()
            return
        if tool in EDIT_TOOLS and returncode == 0:
            path = body.strip().split("\n", 1)[0].strip()
            if path and not is_prose(path) and path not in self.unproven:
                self.unproven.append(path)

    @property
    def needs_check(self) -> bool:
        return bool(self.unproven)


#: Shown once, when a run tries to finish with code changed and nothing run.
#:
#: It names the files rather than asking a general question, because "did you
#: verify?" is answerable with "yes" by a model that did not, and "app.py and
#: two others have changed since anything last ran" is not.
UNPROVEN_NOTE = (
    "Before that answer stands: {files} changed and nothing has run since. "
    "Run the thing that would FAIL if the change were wrong -- the test suite, "
    "the build, or the code itself -- and read what it prints.\n"
    "If there is genuinely nothing to run here, say so in one line and finish. "
    "Do not invent a command to satisfy this."
)

#: How many paths the note lists before it says "and N others". A note that
#: reprints a forty-file refactor is a note nobody reads.
MAX_LISTED = 3


def render_note(ledger: Ledger) -> str:
    paths = ledger.unproven
    shown = ", ".join(paths[:MAX_LISTED])
    if len(paths) > MAX_LISTED:
        shown += f" and {len(paths) - MAX_LISTED} other file(s)"
    verb = "has" if len(paths) == 1 else "have"
    return UNPROVEN_NOTE.format(files=f"{shown} {verb}")
