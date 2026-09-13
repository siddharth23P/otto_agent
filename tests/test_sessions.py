"""Saved sessions: the index (agent/memory/sessions.py), the queue's ability
to come back from disk (agent/memory/queue.py, restore=True), and the REPL
commands over both (agent/cli/shell.py).

Before 2026-09-13 none of this existed: a session was a uuid and a SQLite
file only compaction ever wrote to, so a conversation that ended before its
first compaction -- nearly all of them -- was gone with the process. These
pin the whole chain: a turn recorded in one Session, found in the index,
loaded into another Session, and handed to the graph as the same history.
"""
from __future__ import annotations

import io
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console

import agent.memory.queue as q
from agent.cli import shell
from agent.cli.sessions import sessions_table
from agent.cli.shell import Session, dispatch
from agent.cli.ui import THEME
from agent.memory import sessions as index
from agent.memory import store as store_module
from agent.memory import wiring
from agent.memory.embeddings import EmbeddingUnavailable
from agent.memory.store import MemoryStore, session_db_path


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(q, "count_tokens", len)

    def _raise(texts):
        raise EmbeddingUnavailable("no embeddings in tests")

    monkeypatch.setattr(q, "embed", _raise)
    # The queue's compaction callback reaches for the router; nothing here
    # should ever compact through a model.
    monkeypatch.setattr(wiring, "summarize_for_memory", lambda prompt: "1. a summary [1]")


class FakeCtx:
    pass


def _render(renderable) -> str:
    """Through the REPL's own theme: the panels name styles (muted, spec,
    ok) that only agent/cli/ui.py's console knows."""
    buf = io.StringIO()
    Console(file=buf, width=120, force_terminal=False, theme=THEME).print(renderable)
    return buf.getvalue()


# --------------------------------------------------------------------------
# The queue comes back from disk
# --------------------------------------------------------------------------

def test_a_restored_queue_has_what_the_old_one_had(tmp_path):
    """X verbatim, Y's raw overflow verbatim, Y's bullets, and the generation
    counter -- everything `history_for_graph` reads."""
    store = MemoryStore(tmp_path / "s.db")
    first = q.TieredQueue("history", store, summarize=lambda p: "1. older stuff [1,2]",
                          x_budget=10, y_budget=12)
    first.append("aaaa")       # x
    first.append("bbbb")       # x
    first.append("cccc")       # 12 > 10: all three to y (12 tokens, at the budget, no compaction)
    first.append("dddddd")     # x
    first.append("eeeeee")     # 12 > 10: x to y -> y has 24 > 12 -> compaction
    first.append("ffff")       # x again
    assert first.recent_items == ["ffff"]
    assert first._y_raw == []
    assert [b.text for b in first._y_bullets]
    generation = first._generation

    second = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=10, y_budget=12,
                           restore=True)

    assert second.recent_items == ["ffff"]
    assert second._y_raw == []
    assert [b.text for b in second._y_bullets] == [b.text for b in first._y_bullets]
    assert [b.hash_refs for b in second._y_bullets] == [b.hash_refs for b in first._y_bullets]
    assert second._generation == generation
    assert second.current_view() == first.current_view()
    store.close()


def test_y_raw_survives_a_restart_before_it_is_compacted(tmp_path):
    store = MemoryStore(tmp_path / "s.db")
    first = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=6, y_budget=1000)
    first.append("abcd")
    first.append("efgh")   # x overflows into y, y far under budget
    first.append("ijkl")
    assert first._y_raw == ["abcd", "efgh"] and first.recent_items == ["ijkl"]

    second = q.TieredQueue("history", store, summarize=lambda p: "", x_budget=6, y_budget=1000,
                           restore=True)

    assert second._y_raw == ["abcd", "efgh"]
    assert second.recent_items == ["ijkl"]
    store.close()


def test_without_restore_a_queue_starts_empty_over_a_written_store(tmp_path):
    """The benchmarks build queues over stores they control and must start
    from exactly what they constructed."""
    store = MemoryStore(tmp_path / "s.db")
    q.TieredQueue("history", store, summarize=lambda p: "", x_budget=100, y_budget=100).append("hello")

    assert not q.TieredQueue("history", store, summarize=lambda p: "", x_budget=100, y_budget=100).has_content
    store.close()


def test_a_restored_queue_keeps_going_from_where_it_was(tmp_path):
    """The next compaction after a restart supersedes the generation on disk
    rather than writing a second generation 1 beside it."""
    store = MemoryStore(tmp_path / "s.db")
    first = q.TieredQueue("history", store, summarize=lambda p: "1. first [1,2]", x_budget=4, y_budget=6)
    for item in ("aaaa", "bbbb", "cccc", "dddd"):
        first.append(item)
    assert first._generation >= 1

    second = q.TieredQueue("history", store, summarize=lambda p: "1. later [1,2,3]", x_budget=4,
                           y_budget=6, restore=True)
    for item in ("eeee", "ffff", "gggg", "hhhh"):
        second.append(item)

    live = store.current_bullets("history")
    assert live, "a compaction happened after the restart"
    assert {b.generation for b in live} == {second._generation}
    assert second._generation > first._generation
    assert store.pending("history") == [("x", item) for item in second.recent_items]
    store.close()


def test_the_store_knows_whether_anything_was_ever_written(tmp_path):
    store = MemoryStore(tmp_path / "s.db")
    assert store.is_empty()
    store.add_pending("history", "hi")
    assert not store.is_empty()
    store.close()


# --------------------------------------------------------------------------
# The index
# --------------------------------------------------------------------------

def test_the_first_turn_names_the_session_and_later_ones_do_not(tmp_path):
    info = index.touch("abc", title="fix the flaky test", workspace=tmp_path, turns=1)
    assert info.title == "fix the flaky test" and info.turns == 1 and info.workspace == str(tmp_path)

    later = index.touch("abc", title="something else", workspace=None, turns=2)
    assert later.title == "fix the flaky test"
    assert later.turns == 2 and later.workspace is None
    assert later.created_at == info.created_at
    assert later.last_active_at >= info.last_active_at


def test_a_title_is_the_first_line_cut_short():
    assert index.title_from("  fix   the\nflaky test  ") == "fix the flaky test"
    long = "x" * 100
    assert len(index.title_from(long)) == index.TITLE_LENGTH
    assert index.title_from(long).endswith("…")


def test_listing_is_newest_activity_first():
    index.touch("old", title="old", workspace=None, turns=1)
    index.touch("new", title="new", workspace=None, turns=1)
    index.touch("old", workspace=None, turns=2)   # active again
    assert [s.id for s in index.list_sessions()] == ["old", "new"]
    assert [s.id for s in index.list_sessions(limit=1)] == ["old"]


def test_resolve_takes_last_an_id_or_a_unique_prefix():
    index.touch("abcdef01", title="one", workspace=None, turns=1)
    index.touch("abcxyz02", title="two", workspace=None, turns=1)
    assert index.resolve("last").id == "abcxyz02"
    assert index.resolve("abcdef01").id == "abcdef01"
    assert index.resolve("abcd").id == "abcdef01"
    with pytest.raises(LookupError, match="ambiguous"):
        index.resolve("abc")
    with pytest.raises(LookupError, match="no session matches"):
        index.resolve("zzz")
    with pytest.raises(LookupError, match="no session given"):
        index.resolve("   ")


def test_resolve_last_with_nothing_saved_says_so():
    with pytest.raises(LookupError, match="no saved sessions"):
        index.resolve("last")


def test_rename_sets_the_title_and_can_create_the_row():
    index.touch("abc", title="first words", workspace=None, turns=1)
    assert index.rename("abc", "  a real   name ").title == "a real name"
    assert index.touch("abc", title="ignored", workspace=None, turns=2).title == "a real name"
    # Naming a session nothing has happened in keeps it, deliberately.
    assert index.rename("fresh", "kept").turns == 0
    assert index.get("fresh").title == "kept"


def test_delete_removes_the_row_and_the_memory_file():
    MemoryStore.for_session("abc").close()
    index.touch("abc", title="t", workspace=None, turns=1)
    assert session_db_path("abc").exists()

    assert index.delete("abc")
    assert index.get("abc") is None
    assert not session_db_path("abc").exists()
    assert not index.delete("abc")


def test_prune_removes_only_empty_orphans_and_stale_rows(tmp_path):
    memory = store_module.DB_DIR
    # An indexed session with a file: untouched.
    MemoryStore.for_session("kept").close()
    index.touch("kept", title="kept", workspace=None, turns=1)
    # An orphan nothing was written to: the test-suite litter this exists for.
    MemoryStore.for_session("litter").close()
    # An orphan that holds history: not ours to delete.
    s = MemoryStore.for_session("history")
    s.add_pending("history", "something said")
    s.close()
    # The lesson bank lives in the same directory and is never a session.
    MemoryStore(memory / "lessons.db").close()
    # A row whose file is gone.
    index.touch("gone", title="gone", workspace=None, turns=3)

    report = index.prune()

    assert report.removed_files == 1 and report.dropped_rows == 1 and report.kept_orphans == 1
    assert sorted(p.name for p in memory.glob("*.db")) == ["history.db", "kept.db", "lessons.db"]
    assert [r.id for r in index.list_sessions()] == ["kept"]
    assert "removed 1 empty file(s)" in report.summary() and "kept 1" in report.summary()


def test_describe_age_is_coarse():
    now = datetime(2026, 9, 13, 12, 0, tzinfo=timezone.utc)
    at = lambda **kw: (now - timedelta(**kw)).isoformat()  # noqa: E731
    assert index.describe_age(at(seconds=5), now) == "just now"
    assert index.describe_age(at(minutes=7), now) == "7m ago"
    assert index.describe_age(at(hours=3), now) == "3h ago"
    assert index.describe_age(at(days=2, hours=5), now) == "2d ago"


def test_the_table_marks_the_current_session(tmp_path):
    index.touch("abcdef0123", title="the one I am in", workspace=tmp_path / "repo", turns=4)
    index.touch("fedcba9876", title="another", workspace=None, turns=1)
    text = _render(sessions_table(index.list_sessions(), current="abcdef0123"))
    assert "abcdef01 ◂" in text and "the one I am in" in text and "repo" in text
    assert "fedcba98" in text and "another" in text and "off" in text


# --------------------------------------------------------------------------
# Session: record, load, resume
# --------------------------------------------------------------------------

def test_recording_a_turn_registers_the_session_and_counts_it(tmp_path):
    s = Session(ctx=FakeCtx(), workspace=tmp_path)
    assert index.list_sessions() == [] and s.title == ""

    s.record_turn(HumanMessage("please fix the build"), AIMessage("done"))
    assert s.turn == 1 and s.title == "please fix the build"
    info = index.get(s.session_id)
    assert info.turns == 1 and info.workspace == str(tmp_path)

    # A turn with no output is recorded (the question is worth coming back
    # to) but not counted (output.py's filenames number turns with output).
    s.record_turn(HumanMessage("and the tests?"), None)
    assert s.turn == 1 and index.get(s.session_id).turns == 1
    assert s.title == "please fix the build"


def test_a_loaded_session_hands_the_graph_the_same_history(tmp_path):
    first = Session(ctx=FakeCtx(), workspace=tmp_path)
    first.record_turn(HumanMessage("what is 2+2"), AIMessage("4"))
    first.record_turn(HumanMessage("and 3+3"), AIMessage("6"))
    before = first.history_for_graph()
    sid = first.session_id

    second = Session(ctx=FakeCtx(), workspace=None)
    info = second.load(sid[:8])

    assert info.id == sid and second.session_id == sid
    assert second.turn == 2 and second.title == "what is 2+2"
    assert second.workspace == tmp_path, "the saved workspace comes back"
    assert second.history_for_graph() == before
    earlier, messages = second.transcript()
    assert earlier == "" and [m.content for m in messages] == ["what is 2+2", "4", "and 3+3", "6"]


def test_a_saved_workspace_that_no_longer_exists_is_not_restored(tmp_path):
    gone = tmp_path / "gone"
    gone.mkdir()
    first = Session(ctx=FakeCtx(), workspace=gone)
    first.record_turn(HumanMessage("hi"), AIMessage("hello"))
    gone.rmdir()

    second = Session(ctx=FakeCtx(), workspace=tmp_path)
    second.load(first.session_id)
    assert second.workspace == tmp_path


def test_reset_forgets_the_title_and_the_old_session_stays_listed(tmp_path):
    s = Session(ctx=FakeCtx(), workspace=None)
    s.record_turn(HumanMessage("first"), AIMessage("ok"))
    old = s.session_id
    s.reset()
    assert s.title == "" and s.turn == 0 and s.session_id != old
    assert [r.id for r in index.list_sessions()] == [old]


def test_reset_and_load_close_the_store_they_replace(tmp_path):
    """Windows will not delete an open file, so a session's file must be
    released the moment the session stops using it -- CI's windows-latest
    job caught `delete` failing on exactly this. Asserted through the
    connection rather than a delete, so it holds on every platform."""
    import sqlite3

    s = Session(ctx=FakeCtx(), workspace=None)
    s.record_turn(HumanMessage("hi"), AIMessage("hello"))
    saved = s.session_id
    first_store = s.history_queue.store
    s.reset()
    with pytest.raises(sqlite3.ProgrammingError):
        first_store.pending("history")
    second_store = s.history_queue.store
    s.load(saved)
    with pytest.raises(sqlite3.ProgrammingError):
        second_store.pending("history")
    assert [m.content for m in s.transcript()[1]] == ["hi", "hello"]
    s.close()
    s.close()   # idempotent
    assert index.delete(saved)


@pytest.mark.skipif(os.name != "nt", reason="only Windows refuses to delete an open file")
def test_deleting_an_open_session_file_is_a_clear_error(tmp_path):
    s = Session(ctx=FakeCtx(), workspace=None)
    s.record_turn(HumanMessage("hi"), AIMessage("hello"))
    with pytest.raises(OSError, match="still open"):
        index.delete(s.session_id)
    s.close()
    assert index.delete(s.session_id)


def test_load_of_an_unknown_ref_changes_nothing(tmp_path):
    s = Session(ctx=FakeCtx(), workspace=tmp_path)
    sid = s.session_id
    with pytest.raises(LookupError):
        s.load("nope")
    assert s.session_id == sid and s.workspace == tmp_path


def test_open_session_applies_an_explicit_workspace_over_the_saved_one(tmp_path, monkeypatch):
    from agent.cli.chat import open_session

    saved = tmp_path / "saved"
    saved.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    first = Session(ctx=FakeCtx(), workspace=saved)
    first.record_turn(HumanMessage("hi"), AIMessage("hello"))
    monkeypatch.chdir(tmp_path)

    assert open_session(FakeCtx(), None, False, "last").workspace == saved
    assert open_session(FakeCtx(), other, False, "last").workspace == other.resolve()
    assert open_session(FakeCtx(), None, True, "last").workspace is None
    assert open_session(FakeCtx(), None, False, None).workspace == tmp_path.resolve()
    with pytest.raises(LookupError):
        open_session(FakeCtx(), None, False, "nope")


# --------------------------------------------------------------------------
# The slash commands
# --------------------------------------------------------------------------

def _capture(monkeypatch) -> list[str]:
    seen: list[str] = []
    monkeypatch.setattr(shell.out, "print", lambda r, *a, **k: seen.append(_render(r) if not isinstance(r, str) else r))
    monkeypatch.setattr(shell.err, "print", lambda r, *a, **k: seen.append(_render(r) if not isinstance(r, str) else r))
    return seen


def test_sessions_lists_and_resume_replays(tmp_path, monkeypatch):
    first = Session(ctx=FakeCtx(), workspace=tmp_path)
    first.record_turn(HumanMessage("what is 2+2"), AIMessage("**4**"))
    seen = _capture(monkeypatch)

    s = Session(ctx=FakeCtx(), workspace=None)
    assert dispatch(s, "/sessions")
    assert any(first.session_id[:8] in line and "what is 2+2" in line for line in seen)

    seen.clear()
    assert dispatch(s, "/resume last")
    assert s.session_id == first.session_id and s.workspace == tmp_path
    joined = "\n".join(seen)
    assert "resumed " + first.session_id[:8] in joined
    assert "you" in joined and "what is 2+2" in joined and "4" in joined


def test_resume_and_rename_report_their_mistakes(monkeypatch):
    seen = _capture(monkeypatch)
    s = Session(ctx=FakeCtx(), workspace=None)
    dispatch(s, "/resume")
    dispatch(s, "/resume nothing-like-this")
    dispatch(s, "/rename")
    dispatch(s, "/sessions")
    joined = "\n".join(seen)
    assert "usage: /resume" in joined and "no session matches" in joined
    assert "usage: /rename" in joined and "no saved sessions yet" in joined


def test_rename_names_the_session_for_the_list(monkeypatch):
    seen = _capture(monkeypatch)
    s = Session(ctx=FakeCtx(), workspace=None)
    s.record_turn(HumanMessage("some long first message"), AIMessage("ok"))
    dispatch(s, "/rename build fix")
    assert s.title == "build fix" and index.get(s.session_id).title == "build fix"
    s.record_turn(HumanMessage("more"), AIMessage("ok"))
    assert index.get(s.session_id).title == "build fix"
    assert any("build fix" in line for line in seen)


def test_the_completer_offers_saved_ids_after_resume():
    index.touch("abcdef0123", title="t", workspace=None, turns=1)
    options = shell.COMMANDS["/resume"].options()
    assert options[0] == "last" and "abcdef01" in options


def test_reading_never_creates_the_index():
    path = index.index_path()
    assert not path.exists()
    assert index.list_sessions() == [] and index.get("x") is None
    with pytest.raises(LookupError):
        index.resolve("x")
    assert not index.delete("x")
    assert not path.exists()
    index.touch("x", title="t", workspace=None, turns=1)
    assert path.exists()


# --------------------------------------------------------------------------
# `otto sessions`
# --------------------------------------------------------------------------

def _cli():
    """The command on its own Typer app -- agent/cli/main.py imports every
    command module and builds the router at import, none of which this
    needs."""
    import typer
    from typer.testing import CliRunner

    from agent.cli.sessions import sessions_cmd

    app = typer.Typer()
    app.command()(sessions_cmd)
    runner = CliRunner()
    return lambda *args: runner.invoke(app, list(args))


def test_the_command_lists_renames_deletes_and_prunes(tmp_path):
    run = _cli()
    assert "no saved sessions yet" in run().output

    MemoryStore.for_session("abcdef0123").close()
    index.touch("abcdef0123", title="first words", workspace=tmp_path, turns=2)
    MemoryStore.for_session("litter").close()

    listed = run()
    assert listed.exit_code == 0 and "abcdef01" in listed.output and "first words" in listed.output

    renamed = run("--rename", "abcd", "--title", "the fix")
    assert renamed.exit_code == 0 and index.get("abcdef0123").title == "the fix"
    assert run("--rename", "abcd").exit_code == 2, "--rename without --title"
    assert run("--delete", "zzz").exit_code == 1

    pruned = run("--prune")
    assert "removed 1 empty file(s)" in pruned.output
    assert not session_db_path("litter").exists() and session_db_path("abcdef0123").exists()

    deleted = run("--delete", "last")
    assert deleted.exit_code == 0 and "the fix" in deleted.output
    assert index.list_sessions() == [] and not session_db_path("abcdef0123").exists()


# --------------------------------------------------------------------------
# Export / import
# --------------------------------------------------------------------------

def _compacted_session(tmp_path) -> Session:
    """A session with every kind of memory: retired chunks, live bullets,
    Y's raw overflow and X -- so an export has to carry all four."""
    s = Session(ctx=FakeCtx(), workspace=tmp_path)
    s.history_queue = q.TieredQueue("history", s.history_queue.store, summarize=lambda p: "1. early [1,2]",
                                    x_budget=30, y_budget=40)
    s.record_turn(HumanMessage("first question, long enough"), AIMessage("first answer, also long"))
    s.record_turn(HumanMessage("second question here"), AIMessage("second answer here"))
    s.record_turn(HumanMessage("third"), AIMessage("3"))
    s.record_turn(HumanMessage("fourth"), AIMessage("4"))
    return s


def test_an_export_round_trips_into_the_same_history(tmp_path):
    first = _compacted_session(tmp_path)
    hashes = first.history_queue.store.chunk_hashes("history")
    assert hashes, "compaction happened"
    refs = [b.hash_refs for b in first.history_queue._y_bullets]
    assert refs
    before = first.history_for_graph()
    first.rename("moved")

    path = index.export_session(first.session_id, tmp_path / "out" / "s.json")
    assert path.exists()
    first.close()                    # the exporting otto has quit...
    index.delete(first.session_id)   # ...and this is another machine

    info = index.import_session(path)
    assert info.id == first.session_id, "the id is kept when nothing here has it"
    assert info.title == "moved" and info.turns == 4 and info.workspace == str(tmp_path)

    second = Session(ctx=FakeCtx(), workspace=None)
    second.load(info.id)
    assert second.history_for_graph() == before
    assert second.history_queue.store.chunk_hashes("history") == hashes
    assert [b.hash_refs for b in second.history_queue._y_bullets] == refs


def test_importing_beside_the_original_makes_a_second_session(tmp_path):
    first = Session(ctx=FakeCtx(), workspace=None)
    first.record_turn(HumanMessage("hi"), AIMessage("hello"))
    path = index.export_session(first.session_id, tmp_path / "s.json")

    info = index.import_session(path)

    assert info.id != first.session_id
    assert {r.id for r in index.list_sessions()} == {first.session_id, info.id}
    copy = Session(ctx=FakeCtx(), workspace=None)
    copy.load(info.id)
    assert [m.content for m in copy.transcript()[1]] == ["hi", "hello"]


def test_import_rejects_what_is_not_an_export(tmp_path):
    (tmp_path / "bad.json").write_text("{}")
    (tmp_path / "text.json").write_text("not json")
    with pytest.raises(ValueError, match="not an otto session export"):
        index.import_session(tmp_path / "bad.json")
    with pytest.raises(ValueError, match="not JSON"):
        index.import_session(tmp_path / "text.json")
    with pytest.raises(ValueError, match="cannot read"):
        index.import_session(tmp_path / "missing.json")
    (tmp_path / "future.json").write_text('{"version": 99, "session": {}, "pending": []}')
    with pytest.raises(ValueError, match="newer otto"):
        index.import_session(tmp_path / "future.json")
    assert index.list_sessions() == []


def test_export_of_an_unsaved_session_is_refused(tmp_path):
    with pytest.raises(LookupError):
        index.export_session("nothing", tmp_path / "s.json")


def test_the_command_exports_and_imports(tmp_path, monkeypatch):
    from agent.cli.sessions import default_export_path

    run = _cli()
    first = Session(ctx=FakeCtx(), workspace=None)
    first.record_turn(HumanMessage("hi"), AIMessage("hello"))
    monkeypatch.chdir(tmp_path)

    exported = run("--export", "last")
    default = default_export_path(index.get(first.session_id))
    # The path is printed unwrapped (soft_wrap) so it can be copied; the
    # check strips newlines anyway so a narrower CI console cannot fail it.
    assert exported.exit_code == 0 and default.exists()
    assert str(default) in exported.output.replace("\n", "")

    assert run("--export", "last", "--to", str(tmp_path / "named.json")).exit_code == 0
    imported = run("--import", str(tmp_path / "named.json"))
    assert imported.exit_code == 0 and "imported" in imported.output and "--resume" in imported.output
    assert len(index.list_sessions()) == 2
    assert run("--import", str(tmp_path / "missing.json")).exit_code == 1
