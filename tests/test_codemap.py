"""Structure from the syntax tree, which grep cannot give and ranking does
not answer.

Two questions start most code changes: where is this defined, and what breaks
if I change it. Grep answers them with every string that happens to match --
comments, docstrings, an unrelated method of the same name -- and `rag`
answers a different question entirely. This reads the tree.

No model call, nothing leaves the machine, Python only and openly so.
"""
import pytest

from agent.pipeline import codemap as cm
from agent.pipeline.tools import code_map
from agent.pipeline.workspace import bind_workspace

SOURCE = '''
import os
from collections import Counter

CONSTANT = 1


def save(path):
    return os.path.join(path, "x")


def save_all(paths):
    return [save(p) for p in paths]


class Store:
    def save(self):
        return Counter()

    async def flush(self):
        self.save()
'''


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "store.py").write_text(SOURCE)
    (tmp_path / "pkg" / "other.py").write_text(
        "from pkg.store import Store\n\ndef go():\n    Store().save()\n"
    )
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "junk.py").write_text("def save(): pass\n")
    return cm.index_tree(tmp_path)


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

def test_vendored_code_is_not_the_codebase(tree):
    """Indexing dependencies buries the answer under other people's code."""
    assert "node_modules/junk.py" not in tree
    assert set(tree) == {"pkg/store.py", "pkg/other.py"}


def test_classes_functions_and_methods_are_all_found(tree):
    kinds = {(d.name, d.kind) for d in tree["pkg/store.py"].definitions}

    assert ("Store", "class") in kinds
    assert ("save_all", "function") in kinds
    assert ("save", "method") in kinds
    assert ("flush", "method") in kinds, "an async method is still a method"


def test_a_method_remembers_its_class(tree):
    method = next(d for d in tree["pkg/store.py"].definitions
                  if d.kind == "method" and d.name == "save")

    assert method.parent == "Store"
    assert "Store.save" in method.rendered()


def test_a_file_that_does_not_parse_says_so(tmp_path):
    """"No definitions found" and "this file is broken" are different
    answers, and only one of them is about the search."""
    (tmp_path / "broken.py").write_text("def oops(:\n")
    index = cm.index_tree(tmp_path)["broken.py"]

    assert "syntax error" in index.error
    assert index.definitions == []


# --------------------------------------------------------------------------
# The questions
# --------------------------------------------------------------------------

def test_define_finds_every_definition_of_an_exact_name(tree):
    hits = cm.defines(tree, "save")

    assert {(d.path, d.kind) for d in hits} == {
        ("pkg/store.py", "function"), ("pkg/store.py", "method"),
    }


def test_a_longer_name_is_a_different_name(tree):
    """`save` and `save_all` are different functions. Conflating them is what
    makes grep a poor answer to this question."""
    assert all(d.name == "save" for d in cm.defines(tree, "save"))
    assert cm.defines(tree, "sav") == []


def test_uses_spans_files(tree):
    paths = {path for path, _ in cm.uses(tree, "save")}

    assert paths == {"pkg/store.py", "pkg/other.py"}


def test_importers_include_submodules(tree):
    assert cm.importers(tree, "pkg") == ["pkg/other.py"]
    assert cm.importers(tree, "pkg.store") == ["pkg/other.py"]
    assert cm.importers(tree, "os") == ["pkg/store.py"]


def test_outline_is_in_file_order(tree):
    lines = [d.line for d in cm.outline(tree, "pkg/store.py")]

    assert lines == sorted(lines)


def test_a_question_with_no_answer_says_so_rather_than_returning_nothing(tree):
    assert "nothing defines" in cm.render(tree, "define nonexistent")
    assert "nothing references" in cm.render(tree, "uses nonexistent")
    assert "nothing imports" in cm.render(tree, "imports nonexistent")


def test_an_unknown_verb_names_the_ones_that_work(tree):
    answer = cm.render(tree, "frobnicate save")

    for verb in ("define", "uses", "imports", "outline"):
        assert verb in answer


def test_a_bare_verb_asks_for_an_argument(tree):
    assert "define <name>" in cm.render(tree, "define")


def test_a_long_answer_is_capped(tmp_path):
    """The answer goes into a prompt. An unbounded list is a compaction
    problem one turn later."""
    body = "\n".join(f"def f{i}():\n    return widget\n" for i in range(200))
    (tmp_path / "many.py").write_text(body)
    answer = cm.render(cm.index_tree(tmp_path), "uses widget")

    assert "and 160 more" in answer
    assert answer.count("\n") <= cm.MAX_HITS


# --------------------------------------------------------------------------
# As a tool
# --------------------------------------------------------------------------

def test_the_tool_answers_from_the_workspace(tmp_path):
    (tmp_path / "app.py").write_text("class Widget:\n    def spin(self):\n        pass\n")

    with bind_workspace(str(tmp_path)):
        result = code_map("define Widget")

    assert result.returncode == 0
    assert "app.py:1" in result.stdout


def test_the_index_notices_an_edit_the_agent_just_made(tmp_path):
    """A cached index that went stale mid-run would answer confidently about
    code that no longer exists, which is worse than being slow."""
    path = tmp_path / "app.py"
    path.write_text("def before():\n    pass\n")

    with bind_workspace(str(tmp_path)):
        assert "app.py" in code_map("define before").stdout
        path.write_text("def after():\n    pass\n")
        assert "nothing defines" in code_map("define before").stdout
        assert "app.py" in code_map("define after").stdout


def test_a_tree_with_no_python_says_what_it_reads(tmp_path):
    """It covers Python exactly and says so, rather than answering partially
    and letting the caller assume coverage."""
    (tmp_path / "main.go").write_text("package main\n")

    with bind_workspace(str(tmp_path)):
        result = code_map("define main")

    assert result.returncode == 1
    assert "Python only" in result.stderr
