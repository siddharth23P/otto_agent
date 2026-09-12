"""Coverage for the workspace tier: agent/pipeline/workspace.py's per-run
binding and path confinement, and the four file tools plus the workspace-aware
execute_bash/execute_python in agent/pipeline/tools.py.

The point of the whole tier is that a task can read a file, change it, run
something, and read the result back -- which nothing in this repo could do
before, since execute_bash/execute_python each ran in a temp dir destroyed
inside the single call. So the tests that matter most here are the ones about
state SURVIVING between calls, and about it never surviving outside the
directory the caller chose.
"""
import os

import pytest

from conftest import posix_filesystem, posix_only

import agent.pipeline.tools as pt
from agent.pipeline.workspace import (
    OutsideWorkspace,
    bind_workspace,
    current_workspace,
    resolve_in_workspace,
)


@pytest.fixture
def workspace(tmp_path):
    with bind_workspace(tmp_path) as ws:
        yield ws


# ---- confinement ---------------------------------------------------------


def test_nothing_is_bound_by_default():
    assert current_workspace() is None


def test_resolve_rejects_paths_that_escape_the_root(workspace):
    for escape in ["../outside.txt", "/etc/passwd", "a/../../outside.txt",
                   "a/b/../../../outside.txt"]:
        with pytest.raises(OutsideWorkspace):
            resolve_in_workspace(escape)


@posix_filesystem
def test_resolve_rejects_a_symlink_pointing_out_of_the_workspace(workspace):
    os.symlink("/etc", workspace / "link")

    with pytest.raises(OutsideWorkspace):
        resolve_in_workspace("link/passwd")


def test_resolve_allows_a_file_that_does_not_exist_yet(workspace):
    """A write has to pass the check before the file is there -- the case a
    plain Path.resolve() on a missing leaf would not cover."""
    assert resolve_in_workspace("new/dir/file.py") == workspace / "new/dir/file.py"


def test_binding_restores_the_previous_workspace_on_exit(tmp_path):
    outer, inner = tmp_path / "outer", tmp_path / "inner"
    with bind_workspace(outer):
        with bind_workspace(inner):
            assert current_workspace() == inner.resolve()
        assert current_workspace() == outer.resolve()
    assert current_workspace() is None


# ---- write_file / read_file ----------------------------------------------


def test_write_then_read_round_trips_and_creates_parent_dirs(workspace):
    written = pt.write_file("pkg/sub/mod.py\ndef f():\n    return 1\n")

    assert written.ok
    assert (workspace / "pkg/sub/mod.py").read_text() == "def f():\n    return 1\n"
    assert "def f():" in pt.read_file("pkg/sub/mod.py").stdout


def test_write_file_keeps_separator_looking_content_verbatim(workspace):
    """Why there is no delimiter between the path line and the body: any
    delimiter that can be typed can appear in a real file, and a write that
    truncates at a `---` inside a document is worse than a plainer format."""
    body = "doc.md\n# Title\n\n---\n\nsection after a horizontal rule\n"

    pt.write_file(body)

    assert (workspace / "doc.md").read_text() == "# Title\n\n---\n\nsection after a horizontal rule\n"


def test_read_file_can_return_just_a_line_range(workspace):
    pt.write_file("f.txt\n" + "\n".join(f"line {i}" for i in range(1, 21)))

    result = pt.read_file("f.txt:5-7")

    assert "line 5" in result.stdout and "line 7" in result.stdout
    assert "line 4" not in result.stdout and "line 8" not in result.stdout


def test_read_file_reports_a_missing_file_rather_than_raising(workspace):
    result = pt.read_file("nope.txt")

    assert not result.ok
    assert "not a file" in result.stderr


# ---- edit_file -----------------------------------------------------------


def test_edit_file_replaces_an_exact_unique_snippet(workspace):
    pt.write_file("m.py\ndef f():\n    return 1\n")

    result = pt.edit_file("m.py\n---OLD---\n    return 1\n---NEW---\n    return 2")

    assert result.ok
    assert (workspace / "m.py").read_text() == "def f():\n    return 2\n"


def test_edit_file_refuses_when_the_old_text_is_not_there(workspace):
    pt.write_file("m.py\ndef f():\n    return 1\n")

    result = pt.edit_file("m.py\n---OLD---\n    return 99\n---NEW---\n    return 2")

    assert not result.ok
    assert "never appears" in result.stderr
    assert (workspace / "m.py").read_text() == "def f():\n    return 1\n"


def test_edit_file_refuses_an_ambiguous_match_rather_than_guessing(workspace):
    pt.write_file("m.py\nx = 1\ny = 1\n")

    result = pt.edit_file("m.py\n---OLD---\n= 1\n---NEW---\n= 2")

    assert not result.ok
    assert "appears 2 times" in result.stderr
    assert (workspace / "m.py").read_text() == "x = 1\ny = 1\n"


# ---- list_files ----------------------------------------------------------


def test_list_files_skips_vcs_and_build_noise(workspace):
    pt.write_file("src/app.py\n#")
    pt.write_file(".git/objects/abcdef\nbinary")
    pt.write_file("node_modules/lib/index.js\n//")

    listing = pt.list_files("").stdout

    assert "src/app.py" in listing
    assert ".git" not in listing
    assert "node_modules" not in listing


# ---- state surviving between calls ---------------------------------------


def test_bash_and_python_share_the_workspace_across_calls(workspace):
    pt.write_file("m.py\ndef f():\n    return 41 + 1\n")

    assert pt.execute_bash("cat m.py").stdout.strip().endswith("return 41 + 1")
    assert pt.execute_python("import m; print(m.f())").stdout.strip() == "42"

    pt.execute_bash("echo made-by-bash > from_bash.txt")

    assert (workspace / "from_bash.txt").exists()
    assert "made-by-bash" in pt.read_file("from_bash.txt").stdout


def test_execute_python_never_leaves_its_snippet_in_the_workspace(workspace):
    """A snippet.py appearing in the repo under edit would show up in the
    agent's own next listing and in `git status`, indistinguishable from a
    file the task asked for."""
    pt.execute_python("print('hello')")

    assert not (workspace / "snippet.py").exists()


def test_with_no_workspace_bound_execution_stays_throwaway():
    """The old behaviour, unchanged, and what every ordinary chat turn gets."""
    pt.execute_bash("echo leaked > should_not_persist.txt")

    second = pt.execute_bash("ls -1")

    assert "should_not_persist.txt" not in second.stdout


# ---- remote mode: the same tools, acting inside a container ---------------


@pytest.fixture
def container(tmp_path):
    """A stand-in for a Docker container: a real shell, run in tmp_path, bound
    as the command runner. Exercises the remote branch of every tool -- the
    base64 shipping, the find pruning, the exact-once edit script -- without
    needing Docker in the test suite.
    """
    if os.name == "nt":
        pytest.skip("exercises the container path by running POSIX commands "
                    "on the host shell; cmd.exe is not one")

    import subprocess

    def runner(command, timeout):
        proc = subprocess.run(
            command, shell=True, cwd=tmp_path, capture_output=True,
            text=True, timeout=timeout,
        )
        return proc.stdout, proc.stderr, proc.returncode

    from agent.pipeline.execution import bind_command_runner

    with bind_command_runner(runner):
        yield tmp_path


def test_remote_write_and_read_round_trip(container):
    pt.write_file("pkg/mod.py\ndef f():\n    return 1\n")

    assert (container / "pkg/mod.py").read_text() == "def f():\n    return 1\n"
    assert "def f():" in pt.read_file("pkg/mod.py").stdout


def test_remote_write_survives_content_that_would_break_quoting(container):
    """Why content is shipped base64-encoded rather than in a heredoc: a file
    can contain quotes, backslashes, dollar signs, or the delimiter itself."""
    nasty = "s.py\nprint('$HOME `whoami` \\\\n \"quoted\"')\nOTTO_EOF\n"

    pt.write_file(nasty)

    assert (container / "s.py").read_text() == nasty.partition("\n")[2]


def test_remote_read_honours_a_line_range(container):
    pt.write_file("f.txt\n" + "\n".join(f"line {i}" for i in range(1, 21)))

    result = pt.read_file("f.txt:5-7")

    assert "line 5" in result.stdout and "line 7" in result.stdout
    assert "line 4" not in result.stdout and "line 8" not in result.stdout


def test_remote_edit_replaces_exactly_once(container):
    pt.write_file("m.py\ndef f():\n    return 1\n")

    result = pt.edit_file("m.py\n---OLD---\n    return 1\n---NEW---\n    return 2")

    assert result.ok
    assert (container / "m.py").read_text() == "def f():\n    return 2\n"


def test_remote_edit_refuses_an_ambiguous_match_like_the_local_one(container):
    pt.write_file("m.py\nx = 1\ny = 1\n")

    result = pt.edit_file("m.py\n---OLD---\n= 1\n---NEW---\n= 2")

    assert not result.ok
    assert "appears 2 times" in result.stderr
    assert (container / "m.py").read_text() == "x = 1\ny = 1\n"


def test_remote_bash_and_python_run_where_the_runner_says(container):
    pt.write_file("m.py\ndef f():\n    return 41 + 1\n")

    assert pt.execute_bash("cat m.py").stdout.strip().endswith("return 41 + 1")
    assert pt.execute_python("import m; print(m.f())").stdout.strip() == "42"


def test_remote_list_files_prunes_the_same_noise(container):
    pt.write_file("src/app.py\n#")
    pt.write_file(".git/objects/abcdef\nbinary")

    listing = pt.list_files(".").stdout

    assert "src/app.py" in listing
    assert "objects" not in listing


def test_remote_python_survives_a_snippet_containing_the_heredoc_delimiter(container):
    result = pt.execute_python("print('OTTO_EOF')\nprint('still running')")

    assert result.ok
    assert "still running" in result.stdout



def test_write_file_accepts_the_json_shape_a_model_reaches_for(workspace):
    """Observed in a Terminal-Bench transcript: handed two values to put in one
    string, the model produced JSON, and the literal body created a file named
    `{`. The prompts now spell the real format out; this is the safety net."""
    pt.write_file('{"path": "pkg/m.py", "content": "def f():\\n    return 1\\n"}')

    assert (workspace / "pkg/m.py").read_text() == "def f():\n    return 1\n"


def test_write_file_still_treats_a_json_looking_first_line_as_a_path(workspace):
    """Only a real object with both keys is redirected -- anything else is
    still a path, so a file genuinely named after a brace still works."""
    result = pt.write_file('{"not": "a write body"}\nsome content\n')

    assert not result.ok or (workspace / '{"not": "a write body"}').exists()


# ---- everything runs to completion, in view ------------------------------


def test_a_backgrounded_command_is_refused_not_quietly_run():
    """Every tool call in this graph is synchronous, and the loop depends on
    it: the model decides what to do next from what the last command actually
    printed. A backgrounded command returns instantly with empty stdout and
    exit 0, which reads as "it worked" for work that has not started."""
    for command in ["echo hi &", "nohup server &", "setsid worker", "sleep 1 &\necho done"]:
        result = pt.execute_bash(command)
        assert not result.ok, command
        assert "backgrounds or detaches" in result.stderr


def test_ordinary_commands_are_not_mistaken_for_backgrounding():
    """`&&`, and an `&` inside a quoted argument, are not detaching."""
    for command in ["a && b", "grep x y | wc -l", "awk '{print $1 & 2}' f", "ls -la"]:
        assert pt._detaching_reason(command) is None, command


def test_long_output_keeps_both_ends(workspace):
    """A compiler prints its first and most informative error at the top and
    then cascades. Keeping only the tail tells the model what happened last
    instead of what went wrong first."""
    clipped = pt._clip("START" + ("x" * (pt._TAIL * 2)) + "END")

    assert clipped.startswith("START")
    assert clipped.endswith("END")
    assert "characters omitted" in clipped
    assert len(clipped) < pt._TAIL + 200


def test_short_output_is_untouched():
    assert pt._clip("just a line") == "just a line"


# --------------------------------------------------------------------------
# A path field the model filled with something that is not a path
# --------------------------------------------------------------------------
#
# Claw-Eval task C01. The model wrote a sentence of prose where a path goes;
# `Path.exists()` raised OSError(ENAMETOOLONG), because pathlib swallows only
# ENOENT, ENOTDIR, EBADF and ELOOP; every file tool catches OutsideWorkspace and
# nothing else, so it unwound the whole graph. A mortgage comparison the agent
# had already computed and verified was discarded after 1096 seconds, and the
# grader saw a conversation with no assistant messages at all.

@posix_filesystem
def test_a_path_too_long_for_the_filesystem_fails_cleanly(tmp_path):
    with bind_workspace(tmp_path):
        with pytest.raises(OutsideWorkspace):
            resolve_in_workspace("x" * 5000)


def test_the_exact_shape_that_cost_a_run_is_a_failed_result_not_a_crash(tmp_path):
    """Reproduces C01's own body: a path, then the model's prose after it.

    The POINT is a failed ToolResult rather than an exception out of the tool
    loop -- that crash cost 1096 seconds of verified work. WHICH refusal it
    is depends on the platform: POSIX reaches the length limit and calls it
    unusable, Windows resolves the leading slash somewhere else entirely and
    calls it outside the workspace. Both are the behaviour being asserted, so
    the assertion accepts either rather than pinning the wording of an error
    that is allowed to differ.
    """
    body = "/dev/stdin\n\nLet me verify the calculations manually" * 100
    with bind_workspace(tmp_path):
        result = pt.read_file(body)
    assert result.returncode == 1
    assert ("not a usable path" in result.stderr
            or "outside the workspace" in result.stderr), result.stderr


def test_a_nul_byte_in_a_path_fails_cleanly_too(tmp_path):
    with bind_workspace(tmp_path):
        assert pt.read_file("we\x00ird.txt").returncode == 1


def test_an_ordinary_missing_file_still_says_so_plainly(tmp_path):
    """The new handler must not swallow the ordinary case into a confusing
    message."""
    with bind_workspace(tmp_path):
        result = pt.read_file("nope.txt")
    assert result.returncode == 1
    assert "not a file in the workspace" in result.stderr


def test_an_escape_is_still_reported_as_an_escape(tmp_path):
    with bind_workspace(tmp_path):
        with pytest.raises(OutsideWorkspace, match="outside"):
            resolve_in_workspace("../../etc/passwd")


# --------------------------------------------------------------------------
# edit_file's cascade
# --------------------------------------------------------------------------
#
# Exact match alone was the whole implementation. Models quote `old_text` with
# whitespace drift -- a tab that became spaces, a trailing space lost in the
# read, an indentation level changed when the snippet was echoed back -- and
# the field's answer is a cascade of looser matches, not a smarter model.
#
# Every pass still requires a UNIQUE hit. Looseness buys tolerance of how the
# text was quoted, never tolerance of ambiguity about where it goes.

def _edit(tmp_path, original, old, new):
    (tmp_path / "f.py").write_text(original)
    with bind_workspace(tmp_path):
        result = pt.edit_file(f"f.py\n---OLD---\n{old}\n---NEW---\n{new}")
    return result, (tmp_path / "f.py").read_text()


def test_an_exact_quote_still_matches(tmp_path):
    result, after = _edit(tmp_path, "a = 1\nb = 2\n", "b = 2", "b = 3")
    assert result.returncode == 0
    assert "b = 3" in after


def test_trailing_whitespace_drift_is_tolerated(tmp_path):
    """The most common drift, and invisible in a diff."""
    result, after = _edit(tmp_path, "def f():   \n    return 1\n",
                          "def f():\n    return 1", "def f():\n    return 2")
    assert result.returncode == 0, result.stderr
    assert "return 2" in after
    assert "matched ignoring trailing whitespace" in result.stdout


def test_a_snippet_requoted_at_a_different_indent_is_tolerated(tmp_path):
    result, after = _edit(tmp_path, "class C:\n        x = 1\n        y = 2\n",
                          "x = 1\ny = 2", "x = 9\ny = 9")
    assert result.returncode == 0, result.stderr
    assert "x = 9" in after


def test_a_snippet_with_a_misremembered_middle_is_anchored(tmp_path):
    """Last resort, and narrow: three lines minimum, both anchors unique."""
    original = "start_marker\nline one\nline two\nline three\nend_marker\n"
    result, after = _edit(tmp_path, original,
                          "start_marker\n...\nend_marker", "replaced")
    assert result.returncode == 0, result.stderr
    assert "replaced" in after
    assert "line two" not in after


def test_an_exact_match_is_reported_without_a_note(tmp_path):
    """A clean quote should not be told it was loose. Note that a trailing
    space only needs the looser pass when it is INTERIOR to the span -- a
    single line's trailing space is still an exact substring match."""
    result, _ = _edit(tmp_path, "x = 1   \n", "x = 1", "x = 2")
    assert result.returncode == 0
    assert "matched" not in result.stdout


def test_a_looser_match_still_refuses_when_it_is_ambiguous(tmp_path):
    """Looseness is tolerance of how the text was quoted, never of which one
    was meant. Guessing writes the edit into the wrong place."""
    result, after = _edit(tmp_path, "v = 1   \nother\nv = 1\n", "v = 1", "v = 2")
    assert result.returncode == 1
    assert "exactly once" in result.stderr
    assert "v = 2" not in after


def test_text_that_is_simply_absent_says_so(tmp_path):
    result, _ = _edit(tmp_path, "a = 1\n", "completely different", "x")
    assert result.returncode == 1
    assert "never appears" in result.stderr


def test_an_exact_match_wins_over_a_looser_one(tmp_path):
    """The cascade is ordered: a snippet that matches exactly somewhere must
    not be relocated by a looser pass finding somewhere else first."""
    original = "  target\ntarget\n"
    result, after = _edit(tmp_path, original, "target", "hit")
    assert result.returncode == 1, "an exact-but-ambiguous match should refuse"


# --------------------------------------------------------------------------
# The container path uses the same matcher
# --------------------------------------------------------------------------
#
# It used to carry its own exact-match copy, which would now need its own
# cascade and would have to stay in step with the local one forever. That is
# exactly how two copies of the tool loop came to differ on the single line
# that mattered. One implementation, reached twice.

def _fake_container(files: dict):
    """A command runner backed by a dict, honouring the read/write scripts."""
    import base64
    import shlex as _shlex

    def run(command: str, timeout: float):
        parts = _shlex.split(command)
        path = parts[-2] if len(parts) >= 2 and parts[-1] != parts[-2] else parts[-1]
        if "b64encode" in command:
            target = parts[-1]
            if target not in files:
                return "", f"cannot read {target}", 2
            return base64.b64encode(files[target].encode()).decode(), "", 0
        target, payload = parts[-2], parts[-1]
        files[target] = base64.b64decode(payload).decode()
        return f"edited {target}", "", 0

    return run


def test_a_container_edit_tolerates_the_same_drift(tmp_path):
    from agent.pipeline.execution import bind_command_runner

    files = {"/workspace/f.py": "def f():   \n    return 1\n"}
    with bind_command_runner(_fake_container(files)):
        result = pt.edit_file(
            "/workspace/f.py\n---OLD---\ndef f():\n    return 1\n---NEW---\ndef f():\n    return 2"
        )

    assert result.returncode == 0, result.stderr
    assert "return 2" in files["/workspace/f.py"]
    assert "matched ignoring trailing whitespace" in result.stdout


def test_a_container_edit_refuses_an_ambiguous_match(tmp_path):
    from agent.pipeline.execution import bind_command_runner

    files = {"/workspace/f.py": "v = 1\nother\nv = 1\n"}
    with bind_command_runner(_fake_container(files)):
        result = pt.edit_file("/workspace/f.py\n---OLD---\nv = 1\n---NEW---\nv = 2")

    assert result.returncode == 1
    assert "exactly once" in result.stderr
    assert "v = 2" not in files["/workspace/f.py"]


def test_a_container_edit_reports_a_missing_file_rather_than_writing_one(tmp_path):
    from agent.pipeline.execution import bind_command_runner

    files = {}
    with bind_command_runner(_fake_container(files)):
        result = pt.edit_file("/workspace/gone.py\n---OLD---\na\n---NEW---\nb")

    assert result.returncode == 1
    assert files == {}
