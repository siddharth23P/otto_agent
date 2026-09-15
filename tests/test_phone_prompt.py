"""A phone run's agent prompt (agent/pipeline/nodes.py compose_phone_prompt):
the phone's own, short, and the judge's left exactly as it was. Offline --
a scripted model and a fake phone."""
from __future__ import annotations

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from agent.memory import lessons as L
from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, JsonBackend, phone_tools
from agent.pipeline import nodes as pn
from agent.pipeline.profile import bind_tool_profile
from agent.pipeline.toolkit import bind_extra_tools
from agent.pipeline.tools import reachable_tools
from tests.phone_fakes import FakePhone


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self._replies.pop(0) if self._replies else "FINAL:\ndone")


def _state(**overrides) -> dict:
    base = {
        "messages": [HumanMessage("add milk to the cart")],
        "board": [], "node": None, "feedback": "", "output": None, "context": "", "node_error": None,
        "pending_question": None, "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [], "model_calls": 0, "rejections": 0,
    }
    base.update(overrides)
    return base


def _install(monkeypatch, fake):
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["milk is in the cart"]))


class _phone:
    """The bindings agent/server/app.py makes for a phone turn."""

    def __enter__(self):
        self._stack = [L.bind_bank(None),
                       bind_extra_tools(phone_tools(JsonBackend(FakePhone([]))), guidance=PHONE_GUIDANCE),
                       bind_tool_profile(PHONE_DISABLED_STANDING_TOOLS)]
        for cm in self._stack:
            cm.__enter__()
        return self

    def __exit__(self, *exc):
        for cm in reversed(self._stack):
            cm.__exit__(*exc)


def _seed(monkeypatch, state=None):
    fake = _Scripted(["FINAL:\nmilk is in the cart"])
    _install(monkeypatch, fake)
    result = pn.agent(state or _state())
    return fake.seen[0], result


def test_a_phone_run_is_seeded_with_the_phone_prompt(monkeypatch):
    with _phone():
        seed, _ = _seed(monkeypatch)
        agent_prompt = pn.compose_agent_prompt(reachable_tools())
    prompt = seed[0].content
    assert isinstance(seed[0], SystemMessage)
    assert "phone_act" in prompt and "phone_screen" in prompt and "FINAL:" in prompt
    assert "ACTION: <" in prompt and "One tool call per reply." in prompt
    assert "Before FINAL, the last screen you read must show the result." in prompt
    for absent in ("exercise", "switch_mode", "delegate", "MODE", "Four habits", "execute_bash"):
        assert absent not in prompt, absent
    assert len(prompt) <= 0.4 * len(agent_prompt), (len(prompt), len(agent_prompt))


def test_a_phone_run_is_not_told_a_mode(monkeypatch):
    with _phone():
        seed, _ = _seed(monkeypatch)
    assert not any(str(m.content).startswith("MODE:") for m in seed)
    assert isinstance(seed[-1], HumanMessage) and "TASK:\nadd milk" in seed[-1].content


def test_the_resume_rebuilds_the_same_bytes(monkeypatch):
    with _phone():
        seed, first = _seed(monkeypatch)
        fake = _Scripted(["FINAL:\nmilk is in the cart, checked"])
        _install(monkeypatch, fake)
        pn.agent(_state(transcript=first.update["transcript"], feedback="look again",
                        checklist=first.update.get("checklist")))
    resumed = fake.seen[0]
    assert resumed[0].content == seed[0].content
    assert [m.content for m in resumed[1:len(seed)]] == [m.content for m in seed[1:]]


def test_what_the_phone_prompt_describes_is_what_the_parser_accepts():
    with _phone():
        from agent.pipeline.toolkit import dispatch_table

        allowed = dispatch_table()
    assert pn._parse_worker_reply('ACTION: phone_act\nCODE:\n{"op": "tap", "target": "[3]"}',
                                  allowed=allowed) == ("action", "phone_act", '{"op": "tap", "target": "[3]"}')
    assert pn._parse_worker_reply("FINAL:\nmilk is in the cart", allowed=allowed) == (
        "final", "", "milk is in the cart")


def test_a_phone_run_is_not_reminded_to_write_a_script():
    with _phone():
        assert pn.TOOL_BUILDING_NOTE not in pn._reminders(pn.REMINDER_EVERY, None)
    assert pn.TOOL_BUILDING_NOTE in pn._reminders(pn.REMINDER_EVERY, None)


def test_the_judge_reads_the_same_first_message_on_a_phone(monkeypatch):
    def judge_first(phone: bool):
        fake = _Scripted(["FINAL:\nAPPROVE: yes\nWHY: the cart shows it"])
        monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
        state = _state(output="milk is in the cart",
                       checklist=[{"text": "milk is in the cart", "status": "pending"}])
        monkeypatch.setattr(pn, "_distil", lambda state, succeeded: [])
        if phone:
            with _phone():
                pn.evaluator(state)
        else:
            with L.bind_bank(None):
                pn.evaluator(state)
        return fake.seen[0][0].content

    on_phone = judge_first(True)
    assert on_phone == judge_first(False)
    assert "Android phone" not in on_phone


def test_a_coding_run_is_seeded_exactly_as_before(monkeypatch):
    with L.bind_bank(None):
        seed, _ = _seed(monkeypatch, _state(messages=[HumanMessage("fix the failing test")]))
        expected = pn.compose_agent_prompt(reachable_tools())
    assert seed[0].content == expected
    assert str(seed[-1].content).startswith("MODE: ")
