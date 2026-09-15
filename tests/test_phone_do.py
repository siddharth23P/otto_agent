"""agent/phone/tools.py phone_do: several phone_act steps in one call, each
judged exactly as phone_act would judge it, against the screen the step
before left. Offline: a fake phone and a scripted model."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage

from agent.phone import PHONE_DISABLED_STANDING_TOOLS, PHONE_GUIDANCE, JsonBackend, phone_tools
from agent.pipeline import nodes as pn
from agent.pipeline.profile import bind_tool_profile
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import AMAZON_PRODUCT, BLINKIT_CART, BLINKIT_SEARCH, CHAT_WITH_OTP, PHONEPE, FakePhone, node, snapshot

ACTIONS = ("tap", "tap_node", "type_text", "press", "swipe", "scroll", "launch")

TYPED = snapshot("s7", "com.grofers.customerapp", "Blinkit", [
    node(1, "milk", r="edit-field", b=(60, 180, 1020, 260), e=True, c=True, f=True),
], keyboard=True)

RESULTS = snapshot("s8", "com.grofers.customerapp", "Blinkit", [
    node(1, "milk", r="edit-field", b=(60, 180, 1020, 260), e=True, c=True),
    node(2, "Amul Gold Full Cream Milk 1 L", r="text", b=(60, 600, 900, 660), c=True),
    node(3, "ADD", r="button", b=(900, 600, 1040, 660), c=True),
    node(4, "ADD", r="button", b=(900, 800, 1040, 860), c=True),
])

CHECKOUT = snapshot("s9", "com.grofers.customerapp", "Blinkit", [
    node(1, "Order summary", r="text"),
    node(2, "Delivery instructions", r="edit-field", b=(60, 900, 1020, 980), e=True, c=True),
    node(3, "Grand total ₹128", r="text"),
])

ZOMATO = snapshot("s10", "com.application.zomato", "Zomato", [node(1, "Order food", r="text", c=True)])

AMAZON_RESULTS = snapshot("s11", "in.amazon.mShop.android.shopping", "Amazon", [
    node(1, "PHILIPS 100W Magnetic Type-C Cable ₹559", r="view", b=(0, 600, 1440, 800), c=True),
])


def _do(phone, *steps, screen=True):
    by_name = {t.name: t.call for t in phone_tools(JsonBackend(phone))}
    if screen:
        assert by_name["phone_screen"]("{}").ok
    return by_name["phone_do"](json.dumps({"steps": list(steps)}))


def _acted(phone):
    return [call for call in phone.calls if call[0] in ACTIONS]


def test_three_steps_come_back_as_three_lines_and_one_screen():
    phone = FakePhone([BLINKIT_SEARCH, BLINKIT_SEARCH, TYPED, RESULTS])
    result = _do(phone, {"op": "tap", "target": "Search for products"},
                 {"op": "type", "text": "milk"}, {"op": "press", "key": "enter"})
    assert result.ok, result.stderr
    lines = result.stdout.split("\n")
    assert lines[:3] == ["1. tapped [1] 'Search for products'", "2. typed 'milk'", "3. pressed enter"]
    assert result.stdout.count("app: ") == 1
    assert "snapshot: s8" in result.stdout
    assert _acted(phone) == [("tap_node", "s1", 1, False, False), ("type_text", "milk", -1), ("press", "enter")]


def test_a_later_step_is_resolved_on_the_screen_the_step_before_left():
    phone = FakePhone([BLINKIT_SEARCH, RESULTS, BLINKIT_CART])
    result = _do(phone, {"op": "tap", "target": "Milk"}, {"op": "tap", "target": "Amul Gold"})
    assert result.ok, result.stderr
    assert _acted(phone) == [("tap_node", "s1", 2, False, False), ("tap_node", "s8", 2, False, False)]


def test_an_ambiguous_step_stops_and_the_rest_never_reach_the_phone():
    phone = FakePhone([BLINKIT_SEARCH, RESULTS, BLINKIT_CART])
    result = _do(phone, {"op": "tap", "target": "Milk"}, {"op": "tap", "target": "ADD"},
                 {"op": "tap", "target": "View cart"})
    assert not result.ok
    assert result.stderr.startswith("phone_do: stopped at step 2 of 3: 'ADD' matches more than one element")
    assert result.stderr.endswith("; step 3 not run")
    assert result.stdout.startswith("1. tapped [2] 'Milk'\napp: Blinkit")
    assert _acted(phone) == [("tap_node", "s1", 2, False, False)]


def test_a_pay_button_mid_sequence_is_refused_by_the_guard():
    phone = FakePhone([BLINKIT_SEARCH, BLINKIT_CART])
    result = _do(phone, {"op": "tap", "target": "View cart"}, {"op": "tap", "target": "Proceed to Pay"},
                 {"op": "press", "key": "back"})
    assert result.stderr.startswith("GUARD: phone_do: stopped at step 2 of 3: 'Proceed to Pay ₹28' is a payment step")
    assert _acted(phone) == [("tap_node", "s1", 5, False, False)]
    # The cart it stopped on is shown: what is in it is the answer.
    assert "Amul Taaza Toned Milk 500 ml x1" in result.stdout


def test_buy_now_by_its_id_mid_sequence_is_refused_by_the_guard():
    phone = FakePhone([AMAZON_RESULTS, AMAZON_PRODUCT])
    result = _do(phone, {"op": "tap", "target": "PHILIPS"}, {"op": "tap", "target": "#buy-now-button"})
    assert result.stderr.startswith("GUARD: phone_do: stopped at step 2 of 2:")
    assert "buy-now-button" in result.stderr
    assert not any(call[0] == "tap_node" and call[1] == "s6" for call in phone.calls)


def test_enter_after_typing_on_a_checkout_is_refused_mid_sequence():
    phone = FakePhone([CHECKOUT, CHECKOUT])
    result = _do(phone, {"op": "type", "text": "ring the bell", "target": "Delivery instructions"},
                 {"op": "press", "key": "enter"})
    assert result.stderr.startswith("GUARD: phone_do: stopped at step 2 of 2: this screen is a checkout")
    assert _acted(phone) == [("type_text", "ring the bell", 2)]


def test_a_change_of_app_stops_the_steps_written_for_the_one_before():
    phone = FakePhone([BLINKIT_SEARCH, ZOMATO, ZOMATO])
    result = _do(phone, {"op": "tap", "target": "Milk"}, {"op": "tap", "target": "Order food"},
                 {"op": "press", "key": "back"})
    assert result.stderr.startswith("phone_do: stopped at step 1 of 3: the app in front is now Zomato")
    assert result.stderr.endswith("; steps 2-3 not run")
    assert "app: Zomato" in result.stdout
    assert _acted(phone) == [("tap_node", "s1", 2, False, False)]


def test_the_last_step_may_change_the_app():
    phone = FakePhone([BLINKIT_SEARCH, BLINKIT_SEARCH, ZOMATO])
    result = _do(phone, {"op": "scroll", "direction": "down"}, {"op": "press", "key": "home"})
    assert result.ok, result.stderr


def test_a_send_stops_with_the_phone_commit_hint():
    phone = FakePhone([CHAT_WITH_OTP, CHAT_WITH_OTP])
    result = _do(phone, {"op": "type", "text": "on my way", "target": "Type a message"},
                 {"op": "tap", "target": "Send"})
    assert result.stderr.startswith("phone_do: stopped at step 2 of 2: 'Send' cannot be taken back; use phone_commit")
    assert _acted(phone) == [("type_text", "on my way", 2)]


def test_a_guarded_screen_after_a_step_stops_and_is_not_described():
    phone = FakePhone([BLINKIT_SEARCH, PHONEPE])
    result = _do(phone, {"op": "tap", "target": "View cart"}, {"op": "press", "key": "back"})
    assert result.stderr.startswith("GUARD: phone_do: stopped at step 1 of 2: com.phonepe.app is a payment")
    assert result.stdout == "1. tapped [5] 'View cart'"
    assert _acted(phone) == [("tap_node", "s1", 5, False, False)]


def test_an_unreadable_screen_after_a_step_stops():
    class NoScreenBack(FakePhone):
        def scroll(self, direction, i):
            self.calls.append(("scroll", direction, i))
            return json.dumps({"ok": True, "data": {"done": direction}})

    phone = NoScreenBack([BLINKIT_SEARCH])
    by_name = {t.name: t.call for t in phone_tools(JsonBackend(phone))}
    assert by_name["phone_screen"]("{}").ok
    phone.fail = {"tree": {"code": "timeout", "message": "no tree"}}
    result = by_name["phone_do"](json.dumps({"steps": [{"op": "scroll", "direction": "down"},
                                                       {"op": "tap", "target": "ADD"}]}))
    assert result.stderr.startswith("phone_do: stopped at step 1 of 2: the screen after it could not be read")
    assert "(could not read the screen after that: no tree)" in result.stdout
    assert not any(call[0] == "tap_node" for call in phone.calls)


def test_a_bad_call_is_rejected_whole_before_the_phone_is_touched():
    step = {"op": "scroll", "direction": "down"}
    for steps, said in (([step] * 6, "got 6"), ([], "got 0"),
                        ([step, {"steps": [step]}], "step 2: missing required field(s): op"),
                        ([step, {"op": "scroll", "direction": "down", "steps": [step]}], "step 2: unknown field 'steps'"),
                        ([step, step, {"op": "fly"}], "step 3: field 'op' must be one of"),
                        ([step, "tap"], "step 2 is not a phone_act body")):
        phone = FakePhone([BLINKIT_SEARCH])
        result = _do(phone, *steps, screen=False)
        assert not result.ok and said in result.stderr, (said, result.stderr)
        assert phone.calls == [], said
    phone = FakePhone([BLINKIT_SEARCH])
    by_name = {t.name: t.call for t in phone_tools(JsonBackend(phone))}
    assert "missing required field(s): steps" in by_name["phone_do"]('{"op": "tap"}').stderr
    assert phone.calls == []


def test_a_stale_screen_stops_without_a_guard():
    phone = FakePhone([BLINKIT_SEARCH], fail={"tap_node": {"code": "stale", "message": "the screen changed"}})
    result = _do(phone, {"op": "scroll", "direction": "down"}, {"op": "tap", "target": "ADD"},
                 {"op": "press", "key": "back"})
    assert result.stderr == "phone_do: stopped at step 2 of 3: the screen changed; step 3 not run"
    assert result.stdout.startswith("1. scrolled down\napp: Blinkit")
    assert not any(call[0] == "press" for call in phone.calls)


def test_a_hand_over_stops_with_a_guard_and_says_the_person_has_it():
    phone = FakePhone([BLINKIT_SEARCH],
                      fail={"tap_node": {"code": "guard", "message": "secure window", "handover": True}})
    result = _do(phone, {"op": "scroll", "direction": "down"}, {"op": "tap", "target": "ADD"},
                 {"op": "press", "key": "back"})
    assert result.stderr.startswith("GUARD: phone_do: stopped at step 2 of 3: secure window; step 3 not run -- "
                                    "the phone has handed control to the person")
    assert result.stdout == "1. scrolled down"
    assert not any(call[0] == "press" for call in phone.calls)


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen = []

    def stream(self, messages):
        self.seen.append(list(messages))
        yield AIMessageChunk(content=self._replies.pop(0))


def test_one_reply_doing_three_actions_is_one_model_call():
    steps = [{"op": "tap", "target": "Search for products"}, {"op": "type", "text": "milk"},
             {"op": "press", "key": "enter"}]
    phone = FakePhone([BLINKIT_SEARCH, BLINKIT_SEARCH, TYPED, RESULTS])
    llm = _Scripted([
        "ACTION: phone_screen\nCODE:\n{}",
        "ACTION: phone_do\nCODE:\n" + json.dumps({"steps": steps}),
        "FINAL:\nsearched for milk",
    ])
    with bind_extra_tools(phone_tools(JsonBackend(phone)), guidance=PHONE_GUIDANCE), \
            bind_tool_profile(PHONE_DISABLED_STANDING_TOOLS):
        output = pn._tool_loop(llm, [SystemMessage("role"), HumanMessage("search for milk")])
    assert output.startswith("searched for milk")
    assert len(_acted(phone)) == 3
    assert len(llm.seen) == 3


def test_the_phone_prompt_says_phone_do_shares_the_one_call():
    prompt = pn.compose_phone_prompt(["phone_screen", "phone_act", "phone_do"])
    assert "One tool call per reply. phone_do runs several steps in that one call." in prompt
    assert "phone_do runs" not in pn.compose_phone_prompt(["phone_screen", "phone_act"])
    assert "send them as one phone_do" in PHONE_GUIDANCE
