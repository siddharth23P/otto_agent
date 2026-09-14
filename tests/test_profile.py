"""agent/pipeline/profile.py: a host takes standing tools off the menu for
one run, and both the dispatch table and the advertised menu follow."""
from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from agent.pipeline import nodes as pn
from agent.pipeline.profile import bind_tool_profile, disabled_tools
from agent.pipeline.toolkit import ExtraTool, bind_extra_tools, dispatch_table
from agent.pipeline.tools import ToolResult, reachable_tools


class _FakeModel:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = []

    def stream(self, messages):
        self.calls.append([m.content for m in messages])
        yield AIMessageChunk(content=self._replies.pop(0))


def test_nothing_is_disabled_by_default():
    assert disabled_tools() == frozenset()
    assert "execute_bash" in dispatch_table()
    assert "execute_bash" in reachable_tools()


def test_a_disabled_tool_leaves_the_dispatch_table_and_the_menu():
    with bind_tool_profile({"execute_bash", "execute_python"}):
        table = dispatch_table()
        assert "execute_bash" not in table and "execute_python" not in table
        assert "read_file" in table
        live = reachable_tools()
        assert "execute_bash" not in live and "execute_python" not in live
        assert "web_search" in live
    assert "execute_bash" in dispatch_table()
    assert "execute_bash" in reachable_tools()


def test_an_inner_binding_replaces_the_outer_one():
    with bind_tool_profile({"execute_bash"}):
        with bind_tool_profile({"web_search"}):
            assert disabled_tools() == frozenset({"web_search"})
            assert "execute_bash" in dispatch_table()
        assert disabled_tools() == frozenset({"execute_bash"})


def test_binding_none_restores_the_whole_toolbox():
    with bind_tool_profile({"execute_bash"}):
        with bind_tool_profile(None):
            assert disabled_tools() == frozenset()
            assert "execute_bash" in dispatch_table()


def test_a_run_scoped_tool_under_a_disabled_name_is_still_served():
    """The disabled set is about the tools that cannot run HERE; a host that
    binds its own execute_bash to a sandbox means that one."""
    seen = []
    mine = ExtraTool(name="execute_bash", description="Run in my sandbox.",
                     call=lambda body: (seen.append(body), ToolResult("ran", "", 0))[1],
                     mutates=False)
    with bind_tool_profile({"execute_bash"}), bind_extra_tools([mine]):
        dispatch_table()["execute_bash"]("ls")
    assert seen == ["ls"]


def test_the_agent_prompt_does_not_advertise_a_disabled_tool():
    """The menu is composed from reachable_tools() at run time, so a
    disabled tool must be absent from the ACTION: enumeration the model
    sees, not only from the table."""
    with bind_tool_profile({"execute_bash", "execute_python", "browse", "browse_act",
                            "exercise", "look", "look_act"}):
        prompt = pn.compose_agent_prompt(reachable_tools())
    assert "execute_bash" not in prompt
    assert "web_search" in prompt


def test_the_tool_loop_never_runs_a_disabled_tool():
    llm = _FakeModel(["ACTION: execute_bash\nCODE:\necho leaked", "FINAL:\ndone"])
    with bind_tool_profile({"execute_bash"}):
        output = pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("go")])
    assert output == "done"
    assert not any("leaked" in m and "stdout" in m for m in llm.calls[1])
