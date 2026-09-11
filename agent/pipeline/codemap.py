"""Where a name is defined and what touches it, answered from the syntax tree
rather than from a search.

The gap this fills is structure. Otto can already grep (`execute_bash`) and
rank files by meaning (`rag`), and neither answers the two questions that
actually start a code change: where is this defined, and what breaks if I
change it. Grep answers them with every string that happens to match --
comments, docstrings, an unrelated method of the same name -- and ranking
answers a different question entirely.

DETERMINISTIC, LOCAL, NO MODEL CALLS. Parsing is the whole mechanism, which is
the property worth copying from the graph-building tools: a definition is
either at that line or it is not, and no budget is spent deciding. An index of
a few thousand files costs a second and is then reused until the files change.

PYTHON ONLY, DELIBERATELY AND VISIBLY. The `ast` module is in the standard
library; forty languages would mean a parser dependency, a build step, and a
wheel for every platform Otto runs on. So this covers Python exactly and says
so when asked about anything else, rather than answering partially and letting
the caller assume coverage. For other languages the shell is still there, and
`rag` still ranks whole files -- a worse answer, honestly labelled, beats a
confident one drawn from a language this cannot read.

WHAT IT DOES NOT DO. It does not resolve calls to their definitions across
modules -- `save()` in two files is two names here, not one edge. Doing that
properly means import resolution and type inference, which is where an
honest-looking graph starts inventing edges. Naming every place a name is used
is less than a call graph and is not wrong.
"""
from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Directories never walked. Vendored code and build output are not the
#: codebase, and indexing them buries the answer under dependencies.
SKIP_DIRS = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".tox", "site-packages",
    ".idea", ".vscode", "target", ".next", ".claude",
})

#: Ceiling on files indexed in one pass. A repository past this is one where
#: the agent should be narrowing with a path first, and an index that takes a
#: minute to build is an index nobody waits for.
MAX_FILES = 4000

#: Ceiling on a single file. Past this it is generated, and a generated file's
#: definitions are not what anyone is looking for.
MAX_FILE_BYTES = 1_000_000

#: How many results one answer carries. The tool returns text into a prompt,
#: so an unbounded list is a compaction problem later.
MAX_HITS = 40


@dataclass(frozen=True)
class Definition:
    name: str
    kind: str          # "class" | "function" | "method"
    path: str
    line: int
    parent: str = ""   # the class, for a method

    def rendered(self) -> str:
        where = f"{self.parent}." if self.parent else ""
        return f"{self.path}:{self.line}  {self.kind} {where}{self.name}"


@dataclass
class FileIndex:
    path: str
    definitions: list[Definition] = field(default_factory=list)
    #: Modules this file imports, as written.
    imports: list[str] = field(default_factory=list)
    #: Every name USED here, with the line -- attribute access and plain calls
    #: alike. Not resolved to a definition; see the module docstring.
    uses: dict[str, list[int]] = field(default_factory=dict)
    error: str = ""


class _Walker(ast.NodeVisitor):
    def __init__(self, index: FileIndex) -> None:
        self.index = index
        self._class: list[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.index.definitions.append(
            Definition(node.name, "class", self.index.path, node.lineno)
        )
        self._class.append(node.name)
        self.generic_visit(node)
        self._class.pop()

    def _function(self, node) -> None:
        kind = "method" if self._class else "function"
        self.index.definitions.append(
            Definition(node.name, kind, self.index.path, node.lineno,
                       parent=self._class[-1] if self._class else "")
        )
        self.generic_visit(node)

    visit_FunctionDef = _function
    visit_AsyncFunctionDef = _function

    def visit_Import(self, node: ast.Import) -> None:
        self.index.imports.extend(alias.name for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self.index.imports.append(node.module)

    def visit_Name(self, node: ast.Name) -> None:
        self.index.uses.setdefault(node.id, []).append(node.lineno)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.index.uses.setdefault(node.attr, []).append(node.lineno)
        self.generic_visit(node)


def index_file(path: Path, root: Path) -> FileIndex:
    relative = str(path.relative_to(root))
    index = FileIndex(path=relative)
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        index.error = f"unreadable: {exc}"
        return index
    try:
        tree = ast.parse(source, filename=relative)
    except SyntaxError as exc:
        # A file that does not parse is reported, not skipped silently: "no
        # definitions found" and "this file is broken" are different answers
        # and only one of them is about the search.
        index.error = f"syntax error at line {exc.lineno}"
        return index
    _Walker(index).visit(tree)
    return index


def index_tree(root: Path, *, max_files: int = MAX_FILES) -> dict[str, FileIndex]:
    """Every Python file under `root`, parsed. Skips what SKIP_DIRS names."""
    found: dict[str, FileIndex] = {}
    for path in sorted(root.rglob("*.py")):
        if len(found) >= max_files:
            logger.info("code map stopped at %d files", max_files)
            break
        if SKIP_DIRS.intersection(path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        index = index_file(path, root)
        found[index.path] = index
    return found


# --------------------------------------------------------------------------
# The three questions
# --------------------------------------------------------------------------

def defines(files: dict[str, FileIndex], name: str) -> list[Definition]:
    """Every definition of `name`, exactly. Not a substring match: `save` and
    `save_all` are different functions and conflating them is what makes grep
    a poor answer to this question."""
    return [d for index in files.values() for d in index.definitions if d.name == name]


def uses(files: dict[str, FileIndex], name: str) -> list[tuple[str, int]]:
    """Every place `name` is referenced, as (path, line).

    A reference, not a call edge -- see the module docstring on why this
    deliberately stops short of resolving them.
    """
    return sorted(
        (index.path, line)
        for index in files.values()
        for line in index.uses.get(name, ())
    )


def importers(files: dict[str, FileIndex], module: str) -> list[str]:
    """Files importing `module`, or anything under it."""
    return sorted(
        index.path for index in files.values()
        if any(i == module or i.startswith(module + ".") for i in index.imports)
    )


def outline(files: dict[str, FileIndex], path: str) -> list[Definition]:
    index = files.get(path)
    return list(index.definitions) if index else []


def render(files: dict[str, FileIndex], query: str) -> str:
    """One question, answered as text. `query` is `<verb> <argument>`.

    Four verbs rather than a query language: the failure being avoided is a
    tool whose syntax the model has to get right before it learns anything,
    which costs a round trip every time it guesses.
    """
    verb, _, argument = query.strip().partition(" ")
    verb, argument = verb.lower(), argument.strip()
    if not argument:
        return ("say `define <name>`, `uses <name>`, `imports <module>` or "
                "`outline <path>`")

    if verb in ("define", "defines", "definition"):
        hits = defines(files, argument)
        if not hits:
            return f"nothing defines {argument!r} in the Python files here"
        return _cap(d.rendered() for d in hits)

    if verb in ("uses", "used", "callers", "refs"):
        hits = uses(files, argument)
        if not hits:
            return f"nothing references {argument!r} in the Python files here"
        return _cap(f"{path}:{line}" for path, line in hits)

    if verb in ("imports", "importers"):
        hits = importers(files, argument)
        if not hits:
            return f"nothing imports {argument!r}"
        return _cap(hits)

    if verb == "outline":
        hits = outline(files, argument)
        if not hits:
            known = files.get(argument)
            if known and known.error:
                return f"{argument}: {known.error}"
            return f"no Python file indexed at {argument!r}"
        return _cap(d.rendered() for d in hits)

    return f"{verb!r} is not one of define|uses|imports|outline"


def _cap(lines) -> str:
    listed = list(lines)
    shown = listed[:MAX_HITS]
    if len(listed) > MAX_HITS:
        shown.append(f"... and {len(listed) - MAX_HITS} more")
    return "\n".join(shown)
