"""A fake phone for the agent/phone tests: canned snapshots, recorded calls,
and the same JSON envelope the Kotlin bridge speaks."""
from __future__ import annotations

import base64
import json

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def node(i, t="", *, d="", r="text", b=(0, 0, 100, 40), c=False, e=False, s=False, p=False, f=False, k=None, v=""):
    return {"i": i, "t": t, "d": d, "r": r, "b": list(b), "c": c, "e": e, "s": s, "p": p, "f": f, "k": k, "v": v}


def snapshot(sid, package, label, nodes, *, keyboard=False, secure=False):
    return {"snapshot_id": sid, "app": {"package": package, "label": label},
            "screen": {"w": 1080, "h": 2400}, "keyboard": keyboard, "secure": secure, "nodes": nodes}


BLINKIT_SEARCH = snapshot("s1", "com.grofers.customerapp", "Blinkit", [
    node(1, "Search for products", r="edit-field", b=(60, 180, 1020, 260), e=True, c=True),
    node(2, "Milk", r="text", b=(60, 400, 400, 460), c=True),
    node(3, "Amul Taaza Toned Milk 500 ml", r="text", b=(60, 600, 900, 660)),
    node(4, "ADD", r="button", b=(900, 600, 1040, 660), c=True),
    node(5, "View cart", r="button", b=(60, 2200, 1020, 2300), c=True),
])

BLINKIT_CART = snapshot("s2", "com.grofers.customerapp", "Blinkit", [
    node(1, "Cart", r="text"),
    node(2, "Amul Taaza Toned Milk 500 ml x1", r="text"),
    node(3, "₹28", r="text"),
    node(4, "Proceed to Pay ₹28", r="button", b=(60, 2200, 1020, 2300), c=True),
])

SETTINGS_DISPLAY = snapshot("s3", "com.android.settings", "Settings", [
    node(1, "Display", r="text"),
    node(2, "Font size", r="text", b=(60, 900, 900, 960), c=True),
    node(3, "Brightness level", r="text", b=(60, 700, 900, 760), c=True),
])

PHONEPE = snapshot("s4", "com.phonepe.app", "PhonePe", [
    node(1, "Enter UPI PIN", r="text"),
    node(2, "", r="edit-field", e=True, p=True),
])

#: Amazon's product page as the phone reads it (2026-09-15, a Galaxy S23): a
#: WebView whose form buttons all read "Submit", with what they do only in
#: their HTML id.
AMAZON_PRODUCT = snapshot("s6", "in.amazon.mShop.android.shopping", "Amazon", [
    node(1, "PHILIPS 100W Magnetic Type-C to Type-C Fast Charging Cable", r="text"),
    node(2, "Submit", r="radio", b=(56, 1200, 517, 1500), c=True, k=True),
    node(3, "Submit", r="radio", b=(577, 1200, 1031, 1500), c=True, k=False),
    node(4, "₹559", r="text"),
    node(5, "Submit", r="button", b=(52, 2000, 1387, 2140), c=True, v="add-to-cart-button"),
    node(6, "Submit", r="button", b=(52, 2180, 1387, 2320), c=True, v="buy-now-button"),
    node(7, "", r="web", b=(0, 350, 1440, 2698), s=True),
])

CHAT_WITH_OTP = snapshot("s5", "com.whatsapp", "WhatsApp", [
    node(1, "Mom: the OTP for the parcel is 4471", r="text"),
    node(2, "Type a message", r="edit-field", e=True, c=True),
    node(3, "Send", r="button", c=True, d="Send"),
])


class FakePhone:
    """Speaks the JSON envelope. `screens` is the queue of snapshots that
    `tree()` and every action's `after` hand back, in order."""

    def __init__(self, screens=(), apps=(), fail=None):
        self.screens = list(screens)
        self.app_list = list(apps)
        self.calls: list[tuple] = []
        self.fail = fail or {}  # method -> PhoneError-like dict

    def _next(self):
        if len(self.screens) > 1:
            return self.screens.pop(0)
        return self.screens[0] if self.screens else snapshot("s0", "com.android.launcher", "Home", [])

    def _ok(self, data):
        return json.dumps({"ok": True, "data": data})

    def _reply(self, method, data):
        if method in self.fail:
            return json.dumps({"ok": False, "error": self.fail[method]})
        return self._ok(data)

    def tree(self):
        self.calls.append(("tree",))
        return self._reply("tree", self._next())

    def foreground(self):
        self.calls.append(("foreground",))
        current = self.screens[0] if self.screens else self._next()
        return self._reply("foreground", current["app"])

    def tap(self, x, y):
        self.calls.append(("tap", x, y))
        return self._reply("tap", {"done": f"tapped {x},{y}", "after": self._next()})

    def tap_node(self, sid, i, long, commit):
        self.calls.append(("tap_node", sid, i, long, commit))
        return self._reply("tap_node", {"done": "tapped", "after": self._next()})

    def type_text(self, text, i):
        self.calls.append(("type_text", text, i))
        return self._reply("type_text", {"done": "typed", "after": self._next()})

    def press(self, key):
        self.calls.append(("press", key))
        return self._reply("press", {"done": key, "after": self._next()})

    def swipe(self, direction, x=None, y=None):
        self.calls.append(("swipe", direction) if x is None else ("swipe", direction, x, y))
        return self._reply("swipe", {"done": direction, "after": self._next()})

    def scroll(self, direction, i):
        self.calls.append(("scroll", direction, i))
        return self._reply("scroll", {"done": direction, "after": self._next()})

    def screenshot(self):
        self.calls.append(("screenshot",))
        if "screenshot" in self.fail:
            return json.dumps({"ok": False, "error": self.fail["screenshot"]})
        return self._ok({"png_b64": base64.b64encode(PNG).decode()})

    def apps(self):
        self.calls.append(("apps",))
        return self._reply("apps", {"apps": self.app_list})

    def launch(self, package):
        self.calls.append(("launch", package))
        label = next((a["label"] for a in self.app_list if a["package"] == package), package)
        return self._reply("launch", {"package": package, "label": label, "after": self._next()})

    def open_settings(self, page, package):
        self.calls.append(("open_settings", page, package))
        return self._reply("open_settings", {"page": page, "after": self._next()})

    def install(self, package, query):
        self.calls.append(("install", package, query))
        return self._reply("install", {"state": "installing", "after": self._next()})
