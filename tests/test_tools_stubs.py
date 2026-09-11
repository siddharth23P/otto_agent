"""Coverage for the tool box (agent/pipeline/tools.py): execute_bash (real,
2026-09-10), the two stubs (web_search, rag), and complete_code/
predict_edit (real, added the same day once the graph skeleton was proven
-- Mercury's FIM/edit endpoints, routed through the same `Router` class
every chat call already goes through).

Both stubs must fail cleanly with a legible reason rather than crash or
silently return an empty success -- that's what lets a caller's _tool_loop
(agent/pipeline/nodes.py) fall back to its own knowledge and say so, per
each role prompt's own instructions, instead of mistaking "not implemented"
for "found nothing." complete_code/predict_edit must fail the same clean
way on a ProviderError (no key, no viable route, a live API error) --
never let a routing/provider failure crash the whole tool loop.
"""
import pytest

from agent.pipeline import tools as pt
from agent.pipeline.tools import TOOL_DISPATCH, TOOL_TIERS, execute_bash, rag, web_search
from agent.router.llm_provider.base import ProviderError


def test_execute_bash_runs_a_real_command_and_captures_stdout():
    result = execute_bash("echo hello")

    assert result.ok
    assert result.stdout.strip() == "hello"
    assert result.returncode == 0


def test_execute_bash_captures_a_nonzero_exit_and_stderr():
    result = execute_bash("echo oops 1>&2; exit 3")

    assert not result.ok
    assert result.returncode == 3
    assert "oops" in result.stderr


def test_execute_bash_times_out_on_a_hanging_command():
    result = execute_bash("sleep 5", timeout=0.2)

    assert result.timed_out
    assert not result.ok


def test_web_search_without_a_key_degrades_instead_of_crashing():
    """No ANTHROPIC_API_KEY means the WEB route has no viable candidate -- and
    it deliberately has no fallback, because a model with no web access would
    answer from memory while looking like a search. That must reach the tool
    loop as an ordinary failed call, not as an exception."""
    result = web_search("what shipped in Python 3.14")

    assert not result.ok
    assert "web_search failed" in result.stderr


def test_web_search_refuses_an_empty_query():
    assert not web_search("   ").ok


def test_rag_without_a_workspace_says_there_are_no_files():
    """rag searches the FILES in the bound workspace; recall_memory searches
    what this conversation said. Keeping those distinct in the failure text
    matters as much as in the docstrings -- two tools that sound alike cost
    tool-loop turns."""
    result = rag("where is the retry budget configured")

    assert not result.ok
    assert "no workspace is bound" in result.stderr


def test_rag_refuses_an_empty_query():
    assert not rag("  ").ok


def test_every_tool_is_registered_with_a_tier_and_dispatchable():
    assert set(TOOL_TIERS) == {
        "execute_python", "execute_bash", "web_search", "rag",
        "complete_code", "predict_edit", "recall_memory",
        "read_file", "list_files", "write_file", "edit_file", "view_image",
        "browse", "browse_act", "look", "look_act",
    }
    assert set(TOOL_DISPATCH) == set(TOOL_TIERS)


def test_every_mutating_tool_is_held_before_it_runs():
    """The invariant the tier system exists for, restated now that it has teeth.

    It used to be "nothing reachable may be irreversible", enforced by asserting
    the MUTATING tier was EMPTY -- which made the tier a naming convention with
    a tripwire rather than a mechanism. Then `browse_act` arrived: clicking a
    button on a live site genuinely is irreversible, and refusing to have such
    a tool would have meant refusing to drive a browser at all.

    So the invariant is now the stronger one it was always standing in for:
    anything irreversible is HELD for a check before it runs
    (nodes.py's `_mutates` and MUTATION_GATE_NOTE). A tier that no longer
    matches the gate is the bug this catches.
    """
    from agent.pipeline import nodes as pn

    for name, tier in TOOL_TIERS.items():
        assert pn._mutates(name) == (tier == pt.MUTATING), (
            f"{name} is tiered {tier} but the gate disagrees"
        )
    assert any(tier == pt.MUTATING for tier in TOOL_TIERS.values()), (
        "no tool is MUTATING, so the gate is untested by this invariant"
    )


def test_a_workspace_write_is_not_treated_as_irreversible():
    """Writing into a directory the caller opened and can throw away is
    recoverable -- write it again. Gating it measurably cost a model call per
    file and bought nothing."""
    from agent.pipeline import nodes as pn

    assert not pn._mutates("write_file")
    assert not pn._mutates("edit_file")


def test_every_workspace_tool_refuses_when_no_workspace_is_bound():
    """What keeps this from widening an ordinary chat turn: a run that never
    opened a workspace cannot write anywhere at all."""
    workspace_tools = [name for name, tier in TOOL_TIERS.items() if tier == pt.WORKSPACE]
    assert workspace_tools
    for name in workspace_tools:
        result = TOOL_DISPATCH[name]("some/path.txt\nbody")
        assert not result.ok, f"{name} should refuse with no workspace bound"
        assert "no workspace is bound" in result.stderr


def test_recall_memory_fails_cleanly_with_no_store_bound_for_this_run():
    # No agent.memory.session.bind_store() context active -- the state
    # every offline test (and agent/eval/'s golden runner) starts from.
    result = pt.recall_memory("what did we discuss earlier")

    assert not result.ok
    assert result.returncode != 0
    assert "no session memory is bound" in result.stderr


def test_recall_memory_searches_the_bound_store(tmp_path, monkeypatch):
    from agent.memory.session import bind_store
    from agent.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "session.db")
    try:
        store.add_bullet("history", 1, "talked about N Queens", ["h1"], None)
        store.add_chunk("history", "h1", "the user asked to solve N Queens with brute force")

        with bind_store(store):
            result = pt.recall_memory("N Queens")
    finally:
        store.close()

    assert result.ok
    assert "N Queens" in result.stdout


def test_recall_memory_wraps_an_unexpected_failure_cleanly(tmp_path, monkeypatch):
    from agent.memory.session import bind_store
    from agent.memory.store import MemoryStore

    store = MemoryStore(tmp_path / "session.db")

    def _boom(*args, **kwargs):
        raise RuntimeError("store is on fire")

    monkeypatch.setattr(pt, "recall", _boom)
    try:
        with bind_store(store):
            result = pt.recall_memory("anything")
    finally:
        store.close()

    assert not result.ok
    assert "recall_memory failed" in result.stderr


# --------------------------------------------------------------------------
# complete_code / predict_edit -- against a fake Router, no real network.
# _get_router() is lazy (module-level `_router` starts None) specifically so
# importing this module never requires INCEPTION_API_KEY; these tests patch
# the getter rather than constructing a real Router.
# --------------------------------------------------------------------------

class _FakeRouter:
    def __init__(self, fim_result=None, edit_result=None, fim_exc=None, edit_exc=None):
        self._fim_result = fim_result
        self._edit_result = edit_result
        self._fim_exc = fim_exc
        self._edit_exc = edit_exc
        self.fim_calls: list[tuple] = []
        self.edit_calls: list[str] = []

    def fim(self, prefix, suffix="", **kwargs):
        self.fim_calls.append((prefix, suffix))
        if self._fim_exc is not None:
            raise self._fim_exc
        return self._fim_result

    def code_edit(self, code_to_edit, **kwargs):
        self.edit_calls.append(code_to_edit)
        if self._edit_exc is not None:
            raise self._edit_exc
        return self._edit_result


def _install(monkeypatch, fake):
    monkeypatch.setattr(pt, "_get_router", lambda: fake)


def test_complete_code_passes_the_whole_body_as_prefix_with_no_suffix_marker(monkeypatch):
    fake = _FakeRouter(fim_result="def f():\n    return 1")
    _install(monkeypatch, fake)

    result = pt.complete_code("def f():\n    ")

    assert result.ok
    assert result.stdout == "def f():\n    return 1"
    assert fake.fim_calls == [("def f():\n    ", "")]


def test_complete_code_splits_prefix_and_suffix_on_the_marker_line(monkeypatch):
    fake = _FakeRouter(fim_result="return 1")
    _install(monkeypatch, fake)

    result = pt.complete_code("def f():\n    \n---SUFFIX---\n    # end of file")

    assert result.ok
    assert fake.fim_calls == [("def f():\n    ", "    # end of file")]


def test_complete_code_turns_a_provider_error_into_a_failing_result_not_a_crash(monkeypatch):
    fake = _FakeRouter(fim_exc=ProviderError("no viable model for code_complete"))
    _install(monkeypatch, fake)

    result = pt.complete_code("def f():")

    assert not result.ok
    assert result.returncode == 1
    assert "complete_code failed" in result.stderr
    assert "no viable model" in result.stderr


def test_predict_edit_passes_the_body_through_unchanged(monkeypatch):
    fake = _FakeRouter(edit_result="def f():\n    return 2  # fixed")
    _install(monkeypatch, fake)

    result = pt.predict_edit("def f():\n    return 2  # <|cursor|>bug")

    assert result.ok
    assert result.stdout == "def f():\n    return 2  # fixed"
    assert fake.edit_calls == ["def f():\n    return 2  # <|cursor|>bug"]


def test_predict_edit_turns_a_provider_error_into_a_failing_result_not_a_crash(monkeypatch):
    fake = _FakeRouter(edit_exc=ProviderError("INCEPTION_API_KEY not set"))
    _install(monkeypatch, fake)

    result = pt.predict_edit("def f(): pass")

    assert not result.ok
    assert result.returncode == 1
    assert "predict_edit failed" in result.stderr


def test_get_router_lazily_constructs_and_caches_a_real_router(monkeypatch):
    # Confirms importing/using this module never eagerly builds a Router
    # (that would require INCEPTION_API_KEY just to import tools.py, which
    # would break agent/eval/runner.py's zero-network checker path) -- the
    # singleton is only built on first actual use, and reused after that.
    monkeypatch.setattr(pt, "_router", None)
    built = []

    class _Sentinel:
        pass

    def fake_router_cls(*a, **kw):
        instance = _Sentinel()
        built.append(instance)
        return instance

    monkeypatch.setattr(pt, "Router", fake_router_cls)

    first = pt._get_router()
    second = pt._get_router()

    assert first is second
    assert len(built) == 1
