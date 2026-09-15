"""What one phone run costs the agent loop, as a number CI can hold.

A scripted model and a fake phone walk a shopping search: read the results,
then five scrolls and five taps, then answer. Nothing here calls a vendor.
The loop is measured, not the criteria call before it or the judge and the
lesson after it (those have their own tests). Two numbers matter on a phone:
how many model calls the run makes, and how much text the largest call
carries -- every call re-sends what came before it.

BASELINE (2026-09-15, before the phone speed-ups): 12 loop calls, largest
call 75518 characters, last call 75518 characters. The ceilings below are
the baseline; each speed-up lowers them.
"""
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.memory import lessons as L
from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, JsonBackend, phone_tools
from agent.pipeline import nodes as pn
from agent.pipeline.profile import bind_tool_profile
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import FakePhone, node, snapshot

MAX_LOOP_CALLS = 12
MAX_LARGEST_CALL_CHARS = 75518

SCRIPT = [
    "ACTION: phone_screen\nCODE:\n{}",
    *[step for k in range(5) for step in (
        'ACTION: phone_act\nCODE:\n{"op": "scroll", "direction": "down"}',
        f'ACTION: phone_act\nCODE:\n{{"op": "tap", "target": "[{10 + k}]"}}',
    )],
    "FINAL:\nthe cheapest is Example phone model 10 at Rs 9,999",
]


def results_screen(sid: str):
    """A results page the size a real one digests to: about 6k characters."""
    nodes = [
        node(1, "", d="All Filters Icon", r="button", b=(135, 560, 228, 650), c=True, v="s-all-filters-announce"),
        node(2, "", r="web", b=(0, 350, 1440, 2698), s=True),
    ]
    for k in range(3, 63):
        nodes.append(node(k, f"Example phone model {k} 128 GB, 6.1 inch display, fast charging, Rs {9000 + k}",
                          r="view", b=(0, 300 + k * 40, 1440, 330 + k * 40), c=True))
    return snapshot(sid, "in.amazon.mShop.android.shopping", "Amazon", nodes)


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self._replies.pop(0) if self._replies else "FINAL:\ndone")


def _state():
    return {
        "messages": [HumanMessage("find the cheapest phone in these results")],
        "board": [], "node": None, "feedback": "", "output": None, "context": "", "node_error": None,
        "pending_question": None, "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [], "model_calls": 0, "rejections": 0,
    }


def run_script(monkeypatch, script=SCRIPT):
    phone = FakePhone([results_screen(f"s{i}") for i in range(1, 40)])
    fake = _Scripted(script)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["the cheapest phone is named with its price"]))
    with L.bind_bank(None), bind_extra_tools(phone_tools(JsonBackend(phone)), guidance=PHONE_GUIDANCE), \
            bind_tool_profile(PHONE_DISABLED_STANDING_TOOLS):
        result = pn.agent(_state())
    sizes = [sum(len(str(m.content)) for m in call) for call in fake.seen]
    return result, phone, sizes


def test_a_ten_action_search_stays_within_its_loop_budget(monkeypatch):
    result, phone, sizes = run_script(monkeypatch)
    assert result.goto == "evaluator"
    assert result.update["output"].startswith("the cheapest is")
    assert sum(1 for call in phone.calls if call[0] in ("tap_node", "scroll")) == 10
    assert len(sizes) <= MAX_LOOP_CALLS
    assert max(sizes) <= MAX_LARGEST_CALL_CHARS
