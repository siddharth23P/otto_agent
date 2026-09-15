"""Deciding once per turn whether the phone is needed (agent/pipeline/nodes.py
`needs_phone`, agent/embed.py `SessionHandle.run`).

Asked from the app to write a research paper, Otto drove the phone: `otto
serve` bound the phone tools, prompt and seats for every turn before any model
read the request, and gave the turn no workspace, so the research route could
not run. These pin the decision (one cheap call, failing toward yes), what a
turn binds on and off the phone, and that the choice holds for the whole turn.
Offline: the decision call and the pipeline are fakes."""
from __future__ import annotations

import os
import threading

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent import embed
from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, PHONE_SEATS
from agent.pipeline import nodes as pn
from agent.pipeline import run as pipeline
from agent.pipeline.profile import disabled_tools
from agent.pipeline.progress import Cancelled, report
from agent.pipeline.toolkit import ExtraTool, dispatch_table, render_note
from agent.pipeline.tools import ToolResult, reachable_tools
from agent.pipeline.usage import current_usage
from agent.pipeline.workspace import bind_workspace
from agent.router import overrides as ov
from agent.router.llm_provider.base import ProviderError
from agent.router.mapping import Task
from tests.test_embed import _own_environment, configured  # noqa: F401 -- fixtures


def _tool(name):
    return ExtraTool(name=name, description="x", call=lambda body: ToolResult("", "", 0), mutates=False)


PHONE_TOOLS = [_tool("phone_screen"), _tool("phone_act")]


@pytest.fixture
def serve_style(tmp_path, monkeypatch):
    """Keys from a file, as `otto serve` configures them: no subprocess
    tools are taken away by default."""
    monkeypatch.setattr(embed, "_configured", {})
    keys = tmp_path / "keys.env"
    keys.write_text("".join(f"{n}=test-placeholder-not-a-real-key\n" for n in embed.KEY_VARS))
    return embed.configure(tmp_path / "home", env_file=keys)


def _phone_kwargs():
    return dict(tools=PHONE_TOOLS, guidance=PHONE_GUIDANCE, disabled_tools=PHONE_DISABLED_STANDING_TOOLS,
                seats=PHONE_SEATS)


def _observe(monkeypatch, seen):
    def fake_run(text, **kwargs):
        chain = ov.bound_chain(Task.EVALUATE)
        seen.update(
            phone_tool="phone_screen" in dispatch_table(),
            guidance=PHONE_GUIDANCE[:40] in render_note(),
            disabled=disabled_tools(),
            evaluate=chain[0].spec if chain else None,
            workspace=kwargs.get("workspace"),
        )
        if kwargs.get("workspace"):
            with bind_workspace(kwargs["workspace"]):
                seen["bash"] = "execute_bash" in reachable_tools()
                seen["research"] = pn.research_available()
        yield {"__final__": {"final_output": "done"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)


def _phases(events):
    return [e for e in events if e["type"] == "progress" and e["kind"] == "phase"]


# --------------------------------------------------------------------------
# the decision call
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply, expected", [
    ("PHONE: no", False), ("phone: YES", True), ("- PHONE: no\n", False), ("**PHONE:** no", False),
    ("", True), ("I think not", True), ("PHONE: maybe", True), ("PHONE: no\nPHONE: yes", False),
])
def test_the_reply_is_read_and_anything_unclear_is_yes(reply, expected):
    assert pn.parse_phone_decision(reply) is expected


class _Scripted:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self.reply)


def test_one_call_on_the_chat_seat_sees_the_request_and_the_conversation(monkeypatch):
    fake = _Scripted("PHONE: no")
    tasks = []
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda task, *a, **kw: (tasks.append(task), fake)[1])
    history = [HumanMessage("add milk to my cart"), HumanMessage("done")]
    assert pn.needs_phone("write a research note on tides", history) is False
    assert tasks == [Task.CHAT_FAST] and len(fake.seen) == 1
    body = fake.seen[0][-1].content
    assert "write a research note on tides" in body and "add milk to my cart" in body
    assert fake.seen[0][0].content == pn.PHONE_DECIDE_PROMPT


@pytest.mark.parametrize("error", [ProviderError("vendor down"), RuntimeError("no route")])
def test_a_failed_decision_keeps_the_phone(monkeypatch, error):
    def boom(llm, messages):
        raise error

    monkeypatch.setattr(pn, "_call", boom)
    assert pn.needs_phone("turn on dark mode") is True


def test_a_stop_during_the_decision_is_not_a_failure(monkeypatch):
    def stopped(llm, messages):
        raise Cancelled("stopped")

    monkeypatch.setattr(pn, "_call", stopped)
    with pytest.raises(Cancelled):
        pn.needs_phone("turn on dark mode")


# --------------------------------------------------------------------------
# what a turn binds
# --------------------------------------------------------------------------

def test_on_the_phone_a_turn_binds_exactly_what_it_did_before(serve_style, monkeypatch, decide_phone):
    monkeypatch.setattr(pn, "needs_phone", lambda text, history=(): True)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("turn on dark mode", events=events.append, **_phone_kwargs())
    assert seen == {"phone_tool": True, "guidance": True, "disabled": PHONE_DISABLED_STANDING_TOOLS,
                    "evaluate": PHONE_SEATS["evaluate"], "workspace": None}
    phase = _phases(events)
    assert [p["text"] for p in phase] == ["working on your phone"] and phase[0]["detail"] == {"phone": True}
    assert handle.phone is True
    handle.close()


def test_off_the_phone_a_turn_runs_like_the_tui(serve_style, monkeypatch, decide_phone):
    monkeypatch.setattr(pn, "needs_phone", lambda text, history=(): False)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("write a research paper on tides", events=events.append, **_phone_kwargs())
    expected_ws = str(serve_style / "workspaces" / handle.id)
    assert seen == {"phone_tool": False, "guidance": False, "disabled": frozenset(), "evaluate": None,
                    "workspace": expected_ws, "bash": True, "research": True}
    assert os.path.isdir(expected_ws)
    phase = _phases(events)
    assert [p["text"] for p in phase] == ["answering here"] and phase[0]["detail"] == {"phone": False}
    assert events[0] is phase[0], "the phase is the turn's first event"
    assert handle.phone is False
    handle.close()


def test_keys_from_the_host_keep_the_subprocess_tools_off_the_phone_too(configured, monkeypatch, decide_phone):
    monkeypatch.setattr(pn, "needs_phone", lambda text, history=(): False)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    handle.run("write a note", events=lambda e: None, **_phone_kwargs())
    assert seen["disabled"] == frozenset(embed.SUBPROCESS_TOOLS) and seen["bash"] is False
    handle.run("write a note", events=lambda e: None, off_phone_disabled_tools=("browse",), **_phone_kwargs())
    assert seen["disabled"] == frozenset({"browse"})
    handle.close()


def test_the_host_can_say_and_nothing_is_asked(serve_style, monkeypatch, decide_phone):
    def never(text, history=()):
        raise AssertionError("the host said")

    monkeypatch.setattr(pn, "needs_phone", never)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("x", events=events.append, phone=False, **_phone_kwargs())
    assert seen["phone_tool"] is False
    handle.run("x", events=events.append, phone=True, **_phone_kwargs())
    assert seen["phone_tool"] is True and seen["workspace"] is None
    assert [p["detail"]["phone"] for p in _phases(events)] == [False, True]
    handle.run("x", events=events.append, phone=False, workspace=None, **_phone_kwargs())
    assert seen["workspace"] is None
    handle.close()


def test_with_the_decision_off_a_phone_turn_is_unchanged(serve_style, monkeypatch):
    """conftest's default: no call, no phase event, the phone's bindings."""
    def never(text, history=()):
        raise AssertionError("DECIDE_PHONE is off")

    monkeypatch.setattr(pn, "needs_phone", never)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("x", events=events.append, **_phone_kwargs())
    assert seen["phone_tool"] is True and not _phases(events)
    handle.close()


def test_a_turn_without_phone_tools_asks_nothing_and_gets_its_workspace(serve_style, monkeypatch, decide_phone):
    def never(text, history=()):
        raise AssertionError("no phone to decide about")

    monkeypatch.setattr(pn, "needs_phone", never)
    seen: dict = {}
    _observe(monkeypatch, seen)
    handle = embed.Runtime().open_session()
    events = []
    handle.run("x", events=events.append)
    assert not _phases(events) and seen["workspace"].endswith(handle.id) and handle.phone is None
    handle.close()


def test_the_decision_holds_across_a_question_and_its_answer(serve_style, monkeypatch, decide_phone):
    calls = []
    monkeypatch.setattr(pn, "needs_phone", lambda text, history=(): calls.append(text) or False)
    seen = []

    def fake_run(text, **kwargs):
        seen.append(("run", "phone_screen" in dispatch_table(), kwargs["workspace"]))
        yield {"__ask__": {"question": "Which?", "choices": ["a", "b"], "thread_id": "t1"}}

    def fake_resume(answer, **kwargs):
        seen.append(("resume", "phone_screen" in dispatch_table(), kwargs["workspace"]))
        yield {"__final__": {"final_output": "picked " + answer}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    monkeypatch.setattr(pipeline, "resume_pipeline_stream", fake_resume)
    handle = embed.Runtime().open_session()

    def on_event(event):
        if event["type"] == "ask":
            threading.Thread(target=handle.answer, args=(event["thread_id"], "b")).start()

    handle.run("write it up", events=on_event, **_phone_kwargs())
    assert calls == ["write it up"]
    assert seen[0][1] is False and seen[1][1] is False and seen[0][2] == seen[1][2] is not None
    handle.close()


def test_the_decision_is_counted_stoppable_and_never_streams_as_an_answer(serve_style, monkeypatch,
                                                                         decide_phone):
    handle = embed.Runtime().open_session()
    ledgers = []

    def deciding(text, history=()):
        ledgers.append(current_usage())
        report("partial", partial="PHONE: no")
        return False

    monkeypatch.setattr(pn, "needs_phone", deciding)
    seen: dict = {}
    _observe(monkeypatch, seen)
    events = []
    handle.run("x", events=events.append, **_phone_kwargs())
    assert ledgers == [handle.usage]
    assert not [e for e in events if e["type"] == "progress" and e["kind"] == "partial"]

    def stopped(text, history=()):
        raise Cancelled("stopped")

    monkeypatch.setattr(pn, "needs_phone", stopped)
    events.clear()
    handle.run("x", events=events.append, **_phone_kwargs())
    assert events == [{"type": "error", "code": "cancelled", "message": "stopped"}]
    handle.close()
