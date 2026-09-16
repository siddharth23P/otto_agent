"""Old phone screens folded out of the transcript (agent/phone/digest.py
fold_result, agent/pipeline/nodes.py _fold_old_results). Offline."""
from __future__ import annotations

import math

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage

from agent.memory import lessons as L
from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, JsonBackend, phone_tools
from agent.phone.digest import MAX_FOLDED_CHARS, SCREEN_FOLDED, fold_result, render_digest
from agent.pipeline import nodes as pn
from agent.pipeline.profile import bind_tool_profile
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import AMAZON_PRODUCT, BLINKIT_SEARCH, FakePhone, node, snapshot
from tests.test_phone_call_budget import results_screen


class _phone:
    def __init__(self, screens=()):
        self.phone = FakePhone(list(screens))

    def __enter__(self):
        self._stack = [L.bind_bank(None),
                       bind_extra_tools(phone_tools(JsonBackend(self.phone)), guidance=PHONE_GUIDANCE),
                       bind_tool_profile(PHONE_DISABLED_STANDING_TOOLS)]
        for cm in self._stack:
            cm.__enter__()
        return self.phone

    def __exit__(self, *exc):
        for cm in reversed(self._stack):
            cm.__exit__(*exc)


# ---- fold_result -----------------------------------------------------------

def test_a_fold_keeps_the_app_the_capture_and_the_prices():
    folded = fold_result("tapped 'Submit'\n" + render_digest(AMAZON_PRODUCT))
    assert folded.startswith("tapped 'Submit'\n" + SCREEN_FOLDED)
    assert "app: Amazon (in.amazon.mShop.android.shopping) snapshot s6" in folded
    assert '"PHILIPS 100W Magnetic Type-C to Type-C Fast Charging Cable" ₹559' in folded
    assert "[5]" not in folded and "add-to-cart-button" not in folded


def test_a_price_in_its_own_label_is_its_own_title():
    screen = snapshot("s3", "com.grofers.customerapp", "Blinkit", [
        node(1, "Amul Taaza Toned Milk 500 ml", b=(0, 0, 500, 40)),
        node(2, "Nandini Toned Milk 1 L, Rs 54", b=(0, 100, 500, 140)),
    ])
    folded = fold_result(render_digest(screen))
    assert "Blinkit (com.grofers.customerapp) snapshot s3" in folded
    assert '"Nandini Toned Milk 1 L" Rs 54' in folded
    assert "Amul" not in folded, "a title with no price is not an item"


def test_a_screen_with_no_prices_says_so():
    folded = fold_result(render_digest(BLINKIT_SEARCH))
    assert folded == f"{SCREEN_FOLDED} app: Blinkit (com.grofers.customerapp) snapshot s1; seen: no prices"


def test_a_long_results_page_folds_to_at_most_ten_items_and_the_ceiling():
    digest = render_digest(results_screen("s9"))
    assert len(digest) > 5000
    folded = fold_result(digest)
    assert len(folded) <= MAX_FOLDED_CHARS
    assert "Rs 9003" in folded and folded.count("; ") <= 10


def test_an_ad_keeps_its_flag():
    screen = snapshot("s2", "in.amazon.mShop.android.shopping", "Amazon", [
        node(1, "Sponsored Ad - Example Kettle 1.5 L steel"),
        node(2, "₹1,299"),
        node(3, "Sponsored Ad - Kettle deal ₹999"),
    ])
    folded = fold_result(render_digest(screen))
    assert '"Example Kettle 1.5 L steel" ₹1,299' in folded
    assert "₹999 ad" in folded


def test_what_follows_the_digest_is_kept():
    text = "stdout:\ndone\n" + render_digest(BLINKIT_SEARCH) + "\nstderr:\n\nreturncode: 0\n\nStill open:\n- milk"
    folded = fold_result(text)
    assert folded.startswith("stdout:\ndone\n" + SCREEN_FOLDED)
    assert folded.endswith("\nstderr:\n\nreturncode: 0\n\nStill open:\n- milk")


def test_what_is_not_a_screen_is_not_folded():
    assert fold_result("stdout:\ntotal 12\n-rw-r--r-- 1 me staff 10 a.txt\nreturncode: 0") is None
    assert fold_result("the app: x (y) is open") is None
    assert fold_result("") is None
    once = fold_result(render_digest(AMAZON_PRODUCT))
    assert fold_result(once) is None


def test_the_seven_screen_tools_fold_and_the_rest_do_not():
    tools = {t.name: t for t in phone_tools(JsonBackend(FakePhone([])))}
    assert {n for n, t in tools.items() if t.fold is not None} == {
        "phone_screen", "phone_act", "phone_do", "phone_commit", "phone_open", "phone_settings", "phone_install"}


# ---- _fold_old_results -------------------------------------------------------

def _transcript(n_screens: int, *, extra_web: bool = False) -> list:
    messages: list = [SystemMessage("prompt"), HumanMessage("TASK:\nfind the cheapest phone")]
    if extra_web:
        messages += [AIMessage("ACTION: web_search\nCODE:\ncheap phones"),
                     HumanMessage(pn.THIRD_PARTY_RESULT + "\nstdout:\n" + render_digest(results_screen("w1"))
                                  + "\nstderr:\n\nreturncode: 0")]
    for k in range(n_screens):
        messages += [AIMessage('ACTION: phone_act\nCODE:\n{"op": "scroll", "direction": "down"}'),
                     HumanMessage(f"{pn.THIRD_PARTY_RESULT}\nstdout:\nscrolled down\n"
                                  f"{render_digest(results_screen(f's{k}'))}\nstderr:\n\nreturncode: 0")]
    return messages


def test_nothing_is_folded_until_enough_screens_pile_up():
    messages = _transcript(5)
    before = [m.content for m in messages]
    with _phone():
        assert pn._fold_old_results(messages) == 0
    assert [m.content for m in messages] == before


def test_at_six_screens_the_first_four_fold_and_the_last_two_stay_whole():
    messages = _transcript(6)
    before = [m.content for m in messages]
    with _phone():
        assert pn._fold_old_results(messages) == 4
        assert pn._fold_old_results(messages) == 0, "idempotent"
    results = [i for i, m in enumerate(messages) if str(m.content).startswith(pn.THIRD_PARTY_RESULT)]
    for i in results[:4]:
        assert SCREEN_FOLDED in messages[i].content and "scrolled down" in messages[i].content
        assert messages[i].content.endswith("returncode: 0")
    for i in results[4:]:
        assert messages[i].content == before[i]


def test_a_result_that_is_not_a_phone_call_is_left_alone():
    messages = _transcript(6, extra_web=True)
    web = messages[3].content
    with _phone():
        pn._fold_old_results(messages)
    assert messages[3].content == web


def test_a_run_without_the_phone_tools_folds_nothing():
    messages = _transcript(8)
    before = [m.content for m in messages]
    assert pn._fold_old_results(messages) == 0
    assert [m.content for m in messages] == before


def test_compaction_never_sends_a_phone_screen_to_the_evicted_store(monkeypatch):
    kept: list[str] = []
    monkeypatch.setattr(pn, "_keep_evicted", lambda text: kept.append(text) or True)
    messages = _transcript(3)
    messages += [AIMessage("ACTION: recall_memory\nCODE:\nphones"),
                 HumanMessage("TOOL RESULT:\nstdout:\n" + "x" * 5000 + "\nreturncode: 0")]
    messages += [HumanMessage("filler")] * (pn.KEEP_VERBATIM + 1)
    with _phone():
        assert pn._compact(messages) >= 4
    assert kept and all("app: Amazon" not in text for text in kept)
    assert all(SCREEN_FOLDED not in text for text in kept)


# ---- in the loop ------------------------------------------------------------

class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self._replies.pop(0) if self._replies else "FINAL:\ndone")


def test_a_long_phone_run_breaks_its_shared_prefix_once_per_fold(monkeypatch):
    steps = 17
    script = ["ACTION: phone_screen\nCODE:\n{}",
              *['ACTION: phone_act\nCODE:\n{"op": "scroll", "direction": "down"}'] * (steps - 1),
              "FINAL:\nthe cheapest is Example phone model 3 at Rs 9003"]
    fake = _Scripted(script)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    monkeypatch.setattr(pn, "_rubric", lambda llm, task: pn.Rubric(["the cheapest phone is named"]))
    state = {
        "messages": [HumanMessage("find the cheapest phone")],
        "board": [], "node": None, "feedback": "", "output": None, "context": "", "node_error": None,
        "pending_question": None, "pending_choices": None, "asking_role": None, "final_output": None,
        "actions": [], "transcript": None, "mode": None, "mode_log": [], "model_calls": 0, "rejections": 0,
    }
    with _phone([results_screen(f"s{i}") for i in range(60)]):
        result = pn.agent(state)
    assert result.goto == "evaluator"

    breaks = 0
    for prev, cur in zip(fake.seen, fake.seen[1:]):
        if [m.content for m in cur[:len(prev)]] != [m.content for m in prev]:
            breaks += 1
    assert 0 < breaks <= math.ceil(steps / pn.SCREEN_FOLD_EVERY)
    last = fake.seen[-1]
    whole = [m for m in last if "keyboard: hidden" in str(m.content)]
    assert pn.SCREENS_KEPT <= len(whole) < pn.SCREENS_KEPT + pn.SCREEN_FOLD_EVERY
    assert sum(SCREEN_FOLDED in str(m.content) for m in last) == steps - len(whole)
