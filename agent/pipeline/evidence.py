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

#: The tool that LOADS a page. Every one of its operations is a fresh load
#: in a real browser (agent/pipeline/browsing.py), and a load of a page in
#: the workspace fails the call when the page throws -- so a call that came
#: back clean is "you ran it and it did not explode" for a page, exactly as
#: `execute_python` exiting 0 is for a script.
LOAD_TOOLS = frozenset({"browse", "exercise"})

#: Files whose runtime is a browser. A test suite can exercise the logic in
#: one of these and prove nothing about whether it draws: the chess game that
#: motivated this passed its own move-generation tests and `node --check` with
#: a stray token in the script that threw before the board was rendered. For
#: these, a python or shell check is not the check that would have failed.
WEB_SUFFIXES = frozenset({
    ".html", ".htm", ".css", ".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx",
    ".vue", ".svelte",
})

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
    if returncode != 0:
        return False
    if tool in LOAD_TOOLS:
        return True
    if tool not in RUN_TOOLS:
        return False
    if tool == "execute_python":
        return True
    lowered = body.lower()
    if any(phrase in lowered for phrase in CHECK_PHRASES):
        return True
    return bool(set(_WORD.findall(lowered)) & set(CHECK_WORDS))


def loads_a_page(tool: str, body: str) -> bool:
    """Whether this call put a page in a browser. `browse` always does; an
    `exercise` does when its first step is `open` or `serve` -- a shell or
    terminal walkthrough proves a command, not a page."""
    if tool == "browse":
        return True
    if tool != "exercise":
        return False
    first = next((l.strip() for l in body.splitlines() if l.strip()), "")
    return first.split(" ", 1)[0].lower() in ("open", "serve")


#: Paths that are tests rather than the thing under test. Asking a run to
#: USE a test file the way a person would is a hold that teaches nothing.
_TEST_HINTS = ("test_", "_test.", "/tests/", "/test/", "spec.", ".spec.", "conftest")


def is_test(path: str) -> bool:
    lowered = "/" + path.strip().lower()
    return any(hint in lowered for hint in _TEST_HINTS)


def is_web(path: str) -> bool:
    """Whether this path runs in a browser, so only a browser can check it."""
    name = path.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    return "." in name and name[name.rindex("."):] in WEB_SUFFIXES


@dataclass
class Ledger:
    """One run's record of edits and proofs, in the order they happened.

    Per run, not persisted. What a previous run proved says nothing about the
    file this one just changed.
    """

    #: Code paths edited since the last passing check, in first-touched order.
    unproven: list[str] = field(default_factory=list)
    #: Web paths edited since the last page LOAD, in first-touched order.
    #: Kept apart from `unproven` because a passing test suite clears that
    #: one and says nothing about this one -- see WEB_SUFFIXES.
    unrendered: list[str] = field(default_factory=list)
    #: Code paths edited since the last WALKTHROUGH -- an `exercise` of any
    #: kind that came back clean. A test the agent wrote and ran clears
    #: `unproven`; it does not clear this. Measured live, five runs in a row
    #: (a CLI, an API, a curses app among them) each wrote its own harness,
    #: passed it, and finished without once using the thing the way a
    #: person would, and the judge approved every one. Tests are not this.
    unused: list[str] = field(default_factory=list)
    #: The first line of the last `exercise` that passed with ONE step -- a
    #: launch and nothing after it. Proves the thing starts and nothing
    #: more, so it does not clear `unused`: live, a model launched its
    #: AppKit app with `mac ./counter` alone and the judge took "1/1 steps
    #: passed" for the buttons working.
    launch_only: str = ""
    #: Whether anything at all has been checked, for a clearer note.
    ever_checked: bool = False

    def record(self, tool: str, body: str, returncode: int) -> None:
        if is_check(tool, body, returncode):
            self.ever_checked = True
            self.unproven.clear()
            if loads_a_page(tool, body):
                self.unrendered.clear()
            if tool == "exercise":
                lines = [l.strip() for l in body.splitlines() if l.strip()]
                if len(lines) <= 1:
                    self.launch_only = lines[0] if lines else "exercise"
                else:
                    self.unused.clear()
                    self.launch_only = ""
            return
        if tool in EDIT_TOOLS and returncode == 0:
            path = body.strip().split("\n", 1)[0].strip()
            if path and not is_prose(path) and path not in self.unproven:
                self.unproven.append(path)
            if path and is_web(path) and path not in self.unrendered:
                self.unrendered.append(path)
            if path and not is_prose(path) and not is_test(path) and path not in self.unused:
                self.unused.append(path)

    @property
    def needs_check(self) -> bool:
        return bool(self.unproven)

    @property
    def needs_use(self) -> bool:
        """Whether code changed and no walkthrough has used it since. Only
        worth asking when `exercise` is reachable -- the caller knows."""
        return bool(self.unused)

    @property
    def needs_render(self) -> bool:
        """Whether a page changed and no browser has loaded it since. Only
        worth asking about when a browser is reachable, which is the
        caller's to know -- see `render_note`."""
        return bool(self.unrendered)


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

#: Shown once, when a run tries to finish with a PAGE changed that no browser
#: has loaded since -- and only when a browser is reachable, because a note
#: that names a tool the run cannot call is a note that teaches it to lie.
#:
#: Names the command rather than the principle, for the same reason the note
#: above names the files: "did you check it in a browser?" is answerable with
#: "yes" by a model that ran `node --check`.
UNRENDERED_NOTE = (
    "Before that answer stands: {files} changed and no browser has loaded "
    "it since. A test of the logic is not a load of the page. Run `exercise` "
    "with the steps a person would take -- it fails at the first one that "
    "does not hold, or if the page throws -- and `look <question>` to see "
    "what it actually draws, and read what comes back.\n"
    "If it is genuinely not something a browser shows, say so in one line "
    "and finish."
)

#: Shown once, when a run tries to finish with code changed that nothing has
#: USED -- and only when `exercise` is reachable. Names the kinds, because
#: "did you try it?" is answerable with "yes" by a model that ran its own
#: harness, and "run `exercise` with `tty counter.py`" is not.
UNUSED_NOTE = (
    "Before that answer stands: {files} changed and nothing has used it the "
    "way a person would. A test you wrote is not that. Run `exercise` with "
    "the steps a person takes -- `run <cmd>` for a command line, `tty <cmd>` "
    "for a program in a terminal, `serve <cmd>` then `open http://127.0.0.1:"
    "PORT/` or `request GET ...` for an app or API, `open <path>` for a page, "
    "`mac|linux|windows <app or command>` for a desktop app, `android|ios "
    "<app>` for a mobile app -- with steps that would FAIL if it were broken, "
    "and read what comes back.\n"
    "If it is genuinely nothing a person runs -- a library, a config -- say "
    "so in one line and finish. Do not invent a walkthrough to satisfy this."
)

#: Shown once, when a run tries to finish on a walkthrough that only
#: launched the thing. Distinct from UNUSED_NOTE because the answer to that
#: one was "I ran exercise" -- and it was, with nothing after the first
#: line. Naming what is missing is what makes the next attempt differ.
LAUNCH_ONLY_NOTE = (
    "Before that answer stands: `{first}` only launched it -- nothing was "
    "clicked, typed or checked, so it proves the thing starts and nothing "
    "more. Run `exercise` again with the steps a person takes after the "
    "launch: click what they would click, `expect` what they would then "
    "see, and at least one thing that would FAIL if the feature were broken. "
    "Do not add a test mode to the thing itself; use it as it is."
)

#: How many paths the note lists before it says "and N others". A note that
#: reprints a forty-file refactor is a note nobody reads.
MAX_LISTED = 3


def _listed(paths: list[str]) -> str:
    shown = ", ".join(paths[:MAX_LISTED])
    if len(paths) > MAX_LISTED:
        shown += f" and {len(paths) - MAX_LISTED} other file(s)"
    verb = "has" if len(paths) == 1 else "have"
    return f"{shown} {verb}"


def render_note(ledger: Ledger, *, browser: bool = False, exercise: bool = False) -> str:
    """The note for a run that is trying to finish early.

    `browser` is whether this run can load a page and `exercise` whether it
    can walk anything at all (agent/pipeline/tools.py's reachable_tools).
    The most specific ask the run can act on wins: a changed page nobody
    loaded, then changed code nothing used, then changed code nothing ran --
    a walkthrough IS a run, so asking for it covers the weaker ask too.
    Without the tools, the page is just a file and the file just needs
    running.
    """
    if browser and ledger.needs_render:
        return UNRENDERED_NOTE.format(files=_listed(ledger.unrendered))
    if exercise and ledger.launch_only:
        return LAUNCH_ONLY_NOTE.format(first=ledger.launch_only)
    if exercise and ledger.needs_use:
        return UNUSED_NOTE.format(files=_listed(ledger.unused))
    return UNPROVEN_NOTE.format(files=_listed(ledger.unproven))
