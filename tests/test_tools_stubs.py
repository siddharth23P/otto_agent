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


def test_web_search_stub_fails_cleanly_with_a_legible_reason():
    result = web_search("who won the game last night")

    assert not result.ok
    assert result.returncode != 0
    assert "not implemented" in result.stderr
    assert "who won the game last night" in result.stderr


def test_rag_stub_fails_cleanly_with_a_legible_reason():
    result = rag("what does our onboarding doc say")

    assert not result.ok
    assert "not implemented" in result.stderr


def test_all_six_tools_are_registered_read_only():
    assert set(TOOL_TIERS) == {
        "execute_python", "execute_bash", "web_search", "rag",
        "complete_code", "predict_edit",
    }
    assert all(tier == "read_only" for tier in TOOL_TIERS.values())
    assert set(TOOL_DISPATCH) == set(TOOL_TIERS)


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
