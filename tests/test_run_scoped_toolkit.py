"""Coverage for agent/pipeline/toolkit.py -- tools that exist for one run only.

The seam matters because it is the ONLY way a benchmark can hand Otto a tool
it was not built with (agent/eval/claw_bench.py binds Claw-Eval's per-task
JSON-schema tools through it), and because getting the unbinding wrong would
leak one task's tools into the next without failing anything loudly.

Offline throughout: the fake LLM below is tests/test_tool_loop.py's, so a
tool loop can be driven without a provider.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from agent.pipeline import nodes as pn
from agent.pipeline.tools import ToolResult
from agent.pipeline.toolkit import (
    ExtraTool,
    bind_extra_tools,
    current_extra_tools,
    dispatch_table,
    json_body,
    render_note,
)


class _FakeMultiStreamModel:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._replies.pop(0))


def _echo(name="gmail_search", **kw):
    seen = []

    def call(body: str) -> ToolResult:
        seen.append(body)
        return ToolResult(stdout=f"{name} saw {body.strip()}", stderr="", returncode=0)

    tool = ExtraTool(
        name=name,
        description=kw.get("description", "Search the mailbox."),
        call=call,
        schema=kw.get("schema", {
            "type": "object",
            "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"],
        }),
    )
    return tool, seen


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------

def test_nothing_is_bound_by_default():
    assert current_extra_tools() == {}
    assert render_note() == ""


def test_binding_adds_a_tool_and_unbinding_removes_it():
    tool, _ = _echo()
    assert "gmail_search" not in dispatch_table()
    with bind_extra_tools([tool]):
        assert "gmail_search" in dispatch_table()
    assert "gmail_search" not in dispatch_table()


def test_a_binding_replaces_rather_than_merges_with_an_outer_one():
    """One task's tools must never survive into the next -- the reason this
    replaces instead of merging."""
    first, _ = _echo("task_a_tool")
    second, _ = _echo("task_b_tool")
    with bind_extra_tools([first]):
        with bind_extra_tools([second]):
            table = dispatch_table()
            assert "task_b_tool" in table
            assert "task_a_tool" not in table
        assert "task_a_tool" in dispatch_table()


def test_binding_none_unbinds():
    tool, _ = _echo()
    with bind_extra_tools([tool]):
        with bind_extra_tools(None):
            assert current_extra_tools() == {}


def test_the_standing_tools_are_still_there():
    tool, _ = _echo()
    with bind_extra_tools([tool]):
        table = dispatch_table()
        assert "execute_bash" in table and "read_file" in table


def test_an_extra_tool_wins_a_name_collision():
    """A benchmark that declares its own web_search means ITS service, and
    silently serving Otto's would answer from the wrong source."""
    tool, seen = _echo("web_search")
    with bind_extra_tools([tool]):
        dispatch_table()["web_search"]('{"query": "x"}')
    assert seen == ['{"query": "x"}']


# --------------------------------------------------------------------------
# The note that tells a prompt these exist
# --------------------------------------------------------------------------

def test_the_note_names_every_tool_with_its_shape():
    tool, _ = _echo()
    with bind_extra_tools([tool]):
        note = render_note()
    assert "gmail_search" in note
    assert "query: string" in note
    assert "limit: integer (optional)" in note
    assert "Search the mailbox." in note


def test_the_note_says_these_are_additional():
    """A note read as a REPLACEMENT for the standing menu would stop the agent
    reaching for execute_bash on a task that needs both."""
    tool, _ = _echo()
    with bind_extra_tools([tool]):
        assert "in addition" in render_note()


def test_a_free_text_tool_says_so_instead_of_showing_an_empty_object():
    tool, _ = _echo(schema={})
    with bind_extra_tools([tool]):
        assert "(free text)" in render_note()


# --------------------------------------------------------------------------
# JSON bodies
# --------------------------------------------------------------------------

def test_json_body_parses_a_plain_object():
    assert json_body("t", '{"query": "hello"}') == {"query": "hello"}


def test_json_body_tolerates_a_fence_and_surrounding_prose():
    """Both are things models reliably do; refusing them costs a whole tool
    iteration to teach something the parse can just handle."""
    assert json_body("t", 'Here you go:\n```json\n{"query": "hi"}\n```') == {"query": "hi"}


def test_json_body_returns_a_failed_result_for_prose():
    result = json_body("gmail_search", "search for invoices please")
    assert isinstance(result, ToolResult)
    assert result.returncode == 1
    assert "JSON object" in result.stderr


def test_json_body_returns_a_failed_result_for_broken_json():
    result = json_body("gmail_search", '{"query": }')
    assert isinstance(result, ToolResult)
    assert result.returncode == 1
    assert "not valid JSON" in result.stderr


def test_json_body_rejects_a_json_array():
    result = json_body("gmail_search", '["a", "b"]')
    assert isinstance(result, ToolResult)
    assert result.returncode == 1


# --------------------------------------------------------------------------
# Through the real tool loop
# --------------------------------------------------------------------------

def test_the_tool_loop_calls_a_bound_tool_and_feeds_back_its_result():
    tool, seen = _echo()
    llm = _FakeMultiStreamModel([
        'ACTION: gmail_search\nCODE:\n{"query": "invoices"}',
        "FINAL:\nfound them",
    ])
    with bind_extra_tools([tool]):
        output = pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("go")])
    assert output == "found them"
    assert seen == ['{"query": "invoices"}']
    assert "gmail_search saw" in llm.calls[1][-1]


def test_the_tool_loop_shows_the_note_to_the_model():
    tool, _ = _echo()
    llm = _FakeMultiStreamModel(["FINAL:\ndone"])
    with bind_extra_tools([tool]):
        pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("go")])
    assert any("gmail_search" in m for m in llm.calls[0])


def test_the_tool_loop_adds_no_note_when_nothing_is_bound():
    llm = _FakeMultiStreamModel(["FINAL:\ndone"])
    pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("go")])
    assert len(llm.calls[0]) == 2


def test_an_unbound_tool_is_still_refused_by_name():
    llm = _FakeMultiStreamModel([
        'ACTION: gmail_search\nCODE:\n{"query": "x"}',
        "FINAL:\ndone",
    ])
    pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("go")])
    assert "is not available" in llm.calls[1][-1]
