"""agent/phone/backend.py: the JSON envelope becomes Python, and every
failure is one PhoneError shape."""
import json

import pytest

from agent.phone.backend import JsonBackend, PhoneBackend, PhoneError
from tests.phone_fakes import PNG, BLINKIT_SEARCH, FakePhone


def test_a_fake_phone_satisfies_the_protocol():
    assert isinstance(JsonBackend(FakePhone()), PhoneBackend)


def test_ok_envelopes_unwrap_to_their_data():
    backend = JsonBackend(FakePhone([BLINKIT_SEARCH]))
    assert backend.tree()["snapshot_id"] == "s1"
    assert backend.foreground() == {"package": "com.grofers.customerapp", "label": "Blinkit"}


def test_error_envelopes_become_phone_errors_with_code_and_handover():
    phone = FakePhone([BLINKIT_SEARCH], fail={"tap_node": {"code": "guard", "message": "payment screen", "handover": True}})
    with pytest.raises(PhoneError) as caught:
        JsonBackend(phone).tap_node("s1", 4)
    assert caught.value.code == "guard" and caught.value.handover
    assert "payment screen" in str(caught.value)


def test_an_unknown_code_is_normalised():
    phone = FakePhone(fail={"tree": {"code": "weird", "message": "x"}})
    with pytest.raises(PhoneError) as caught:
        JsonBackend(phone).tree()
    assert caught.value.code == "failed" and not caught.value.handover


def test_positional_argument_order_is_the_contract():
    phone = FakePhone([BLINKIT_SEARCH, BLINKIT_SEARCH, BLINKIT_SEARCH, BLINKIT_SEARCH])
    backend = JsonBackend(phone)
    backend.tap_node("s1", 4, long=True)
    backend.type_text("milk")
    backend.scroll("down", 5)
    backend.open_settings("display")
    backend.install(query="Wikipedia")
    assert phone.calls == [("tap_node", "s1", 4, True, False), ("type_text", "milk", -1),
                           ("scroll", "down", 5), ("open_settings", "display", ""),
                           ("install", "", "Wikipedia")]


def test_screenshot_accepts_base64_bytes_and_refuses_nothing():
    assert JsonBackend(FakePhone()).screenshot() == PNG

    class Raw:
        def screenshot(self):
            return PNG

    assert JsonBackend(Raw()).screenshot() == PNG

    class Empty:
        def screenshot(self):
            return json.dumps({"ok": True, "data": None})

    with pytest.raises(PhoneError):
        JsonBackend(Empty()).screenshot()


def test_a_bridge_without_a_method_is_unsupported():
    with pytest.raises(PhoneError) as caught:
        JsonBackend(object()).tree()
    assert caught.value.code == "unsupported"


def test_a_bridge_exception_is_a_failed_step_not_a_crash():
    class Broken:
        def tree(self):
            raise RuntimeError("binder died")

    with pytest.raises(PhoneError) as caught:
        JsonBackend(Broken()).tree()
    assert "binder died" in str(caught.value)


def test_non_json_and_non_object_replies_are_named():
    class Odd:
        def tree(self):
            return "not json"

        def apps(self):
            return "[1, 2]"

    with pytest.raises(PhoneError):
        JsonBackend(Odd()).tree()
    with pytest.raises(PhoneError):
        JsonBackend(Odd()).apps()


def test_a_python_fake_may_hand_back_plain_dicts():
    class Plain:
        def tree(self):
            return BLINKIT_SEARCH

    assert JsonBackend(Plain()).tree() is BLINKIT_SEARCH
