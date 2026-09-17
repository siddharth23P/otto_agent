"""agent/phone/tools.py phone_action: the phone's own action registry, offered as one tool."""
import json
import logging

from agent.phone import JsonBackend, phone_tools
from agent.phone.tools import _action_description, _action_groups
from agent.pipeline.toolkit import MAX_TOOL_DESCRIPTION_CHARS
from tests.phone_fakes import FakePhone


def _a(group, name, summary, *params, effect="change"):
    fields = []
    for p in params:
        key, _, choices = p.partition(":")
        fields.append({"name": key.rstrip("?"), "type": "string", "required": not key.endswith("?"),
                       **({"choices": choices.split("|")} if choices else {})})
    return {"name": name, "summary": summary, "effect": effect, "group": group, "params": fields}


#: The Android app's catalog (otto_android actions/ActionCatalog.kt) as its bridge sends it, 2026-09-17.
APP_MENU = [
    _a("clock", "alarm.set", "set an alarm in the clock app", "hour", "minute", "label?",
       "days?:mon|tue|wed|thu|fri|sat|sun"),
    _a("clock", "timer.set", "start a countdown timer", "seconds", "label?"),
    _a("clock", "alarm.show", "open the clock app's alarms", effect="read"),
    _a("clock", "calendar.add", "open a new calendar event for the person to save", "title", "start", "end?",
       "location?", "notes?", effect="confirm"),
    _a("device", "flashlight", "turn the torch on or off", "on"),
    _a("device", "volume", "set a volume (0-100) or step it", "stream:media|ring|alarm|notification|call",
       "level?", "step?:up|down|mute|unmute"),
    _a("device", "media", "control what is playing", "command:play|pause|toggle|next|previous|stop"),
    _a("device", "dnd", "set Do Not Disturb", "mode:off|on|priority|alarms"),
    _a("device", "panel", "open a quick panel for the person to switch (apps cannot switch these)",
       "name:internet|wifi|bluetooth|nfc|volume", effect="confirm"),
    _a("message", "sms.compose", "write an SMS for the person to send", "to", "body", effect="confirm"),
    _a("message", "email.compose", "write an email for the person to send", "to", "subject?", "body?", "cc?",
       effect="confirm"),
    _a("message", "whatsapp.compose", "write a WhatsApp message for the person to send", "phone", "text",
       effect="confirm"),
    _a("message", "dial", "put a number in the dialer for the person to call", "number", effect="confirm"),
    _a("message", "contacts.find", "look up a contact's numbers and emails", "name", effect="read"),
    _a("files", "share", "offer a file otto made (documents/...) or text to another app", "path?", "text?", "app?",
       effect="confirm"),
    _a("files", "open_url", "open a web page", "url"),
    _a("files", "maps", "show a place, or start navigation", "query", "navigate?"),
    _a("files", "clipboard.copy", "copy text to the clipboard", "text"),
    _a("files", "note.create", "write a note (Keep) for the person to save", "text", "title?", effect="confirm"),
]


def test_the_apps_whole_menu_fits_in_four_tools():
    groups = _action_groups(APP_MENU)
    assert list(groups) == ["phone_clock", "phone_device", "phone_message", "phone_files"]
    for offered in groups.values():
        text = _action_description(offered)
        assert len(text) <= MAX_TOOL_DESCRIPTION_CHARS
        for action in offered:  # every action and every field is named
            assert action["name"] + "{" in text
            for p in action["params"]:
                assert p["name"] in text
    assert "volume{stream:media|ring|alarm|notification|call,level?,step?:up|down|mute|unmute}" in \
        _action_description(groups["phone_device"])

MENU = [
    {"name": "alarm.set", "summary": "set an alarm", "effect": "change",
     "params": [{"name": "hour", "type": "integer", "required": True},
                {"name": "minute", "type": "integer", "required": True},
                {"name": "label", "type": "string", "required": False}]},
    {"name": "sms.compose", "summary": "write an SMS for the person to send", "effect": "confirm",
     "params": [{"name": "to", "type": "string", "required": True}, {"name": "body", "type": "string", "required": True}]},
    {"name": "contacts.find", "summary": "look up a contact", "effect": "read",
     "params": [{"name": "name", "type": "string", "required": True}]},
    {"name": "share", "summary": "offer a file", "effect": "confirm",
     "params": [{"name": "path", "type": "string", "required": False}, {"name": "app", "type": "string", "required": False}]},
]
for _entry, _group in zip(MENU, ("clock", "message", "message", "files")):
    _entry["group"] = _group


class ActionPhone(FakePhone):
    def __init__(self, replies=None, fail=None):
        super().__init__(fail=fail)
        self.replies = replies or {}

    def actions(self):
        self.calls.append(("actions",))
        return self._reply("actions", {"actions": MENU})

    def run_action(self, name, args):
        self.calls.append(("run_action", name, json.loads(args)))
        return self._reply("run_action", self.replies.get(name, {"done": f"did {name}"}))


def _tools(phone):
    return {t.name: t for t in phone_tools(JsonBackend(phone))}


class _AnyGroup:
    """Calls the tool that offers the body's action, as the model would."""

    def __init__(self, tools):
        self.tools = tools

    def call(self, body):
        action = json.loads(body).get("action")
        for tool in self.tools.values():
            if action in tool.schema.get("properties", {}).get("action", {}).get("enum", ()):
                return tool.call(body)
        return self.tools["phone_clock"].call(body)


def _tool(phone):
    tools = _tools(phone)
    return tools, (_AnyGroup(tools) if "phone_clock" in tools else None)


def test_no_registry_no_tool():
    tools, action = _tool(FakePhone())
    assert action is None and "phone_screen" in tools


def test_a_failing_registry_is_no_tool():
    _, action = _tool(ActionPhone(fail={"actions": {"message": "boom", "code": "failed"}}))
    assert action is None


def test_the_menu_is_in_the_descriptions():
    tools = _tools(ActionPhone())
    assert list(tools)[:4] == ["phone_clock", "phone_message", "phone_files", "phone_screen"]
    assert not any(tools[n].mutates for n in ("phone_clock", "phone_message", "phone_files"))
    assert "alarm.set{hour,minute,label?} set an alarm." in tools["phone_clock"].description
    message = tools["phone_message"]
    assert "sms.compose{to,body} write an SMS for the person to send (you); contacts.find{name}" in message.description
    assert message.schema["properties"]["action"]["enum"] == ["sms.compose", "contacts.find"]


def test_an_action_from_another_group_is_refused():
    phone = ActionPhone()
    result = _tools(phone)["phone_clock"].call('{"action": "sms.compose", "to": "1", "body": "x"}')
    assert not result.ok and "one of alarm.set" in result.stderr
    assert not any(c[0] == "run_action" for c in phone.calls)


def test_a_change_runs_with_its_arguments():
    phone = ActionPhone()
    _, action = _tool(phone)
    result = action.call('{"action": "alarm.set", "hour": 7, "minute": 30}')
    assert result.ok and result.stdout == "did alarm.set"
    assert phone.calls[-1] == ("run_action", "alarm.set", {"hour": 7, "minute": 30})


def test_a_confirm_action_hands_over():
    phone = ActionPhone({"sms.compose": {"done": "SMS ready for the person to send", "handed_over": True}})
    _, action = _tool(phone)
    result = action.call('{"action": "sms.compose", "to": "+911234567890", "body": "on my way"}')
    assert result.ok and result.stdout.startswith("GUARD: phone_message: SMS ready")
    assert "handed control to the person" in result.stdout


def test_data_comes_back_as_json():
    data = {"contacts": [{"name": "Asha", "phones": ["+911234567890"]}]}
    _, action = _tool(ActionPhone({"contacts.find": {"done": "1 contact", "data": data}}))
    result = action.call('{"action": "contacts.find", "name": "Asha"}')
    assert result.stdout.split("\n", 1) == ["1 contact", json.dumps(data)]


def test_unknown_actions_and_phone_refusals():
    phone = ActionPhone(fail={"run_action": {"message": "hour is at most 23", "code": "invalid"}})
    _, action = _tool(phone)
    unknown = action.call('{"action": "teleport"}')
    assert not unknown.ok and "one of alarm.set" in unknown.stderr
    assert ("run_action", "teleport", {}) not in phone.calls
    refused = action.call('{"action": "alarm.set", "hour": 30, "minute": 0}')
    assert not refused.ok and "at most 23" in refused.stderr and not refused.stderr.startswith("GUARD")


def test_sharing_into_a_payment_app_is_refused_before_the_phone():
    phone = ActionPhone()
    _, action = _tool(phone)
    result = action.call('{"action": "share", "path": "documents/a.pdf", "app": "com.phonepe.app"}')
    assert "GUARD" in (result.stdout + result.stderr)
    assert not any(c[0] == "run_action" for c in phone.calls)


def test_what_is_written_and_to_whom_stays_out_of_the_log(caplog):
    _, action = _tool(ActionPhone({"sms.compose": {"done": "SMS ready", "handed_over": True},
                                   "contacts.find": {"done": "1 contact", "data": {"phones": ["+91999"]}}}))
    with caplog.at_level(logging.INFO, logger="agent.phone.tools"):
        action.call('{"action": "sms.compose", "to": "+915550001111", "body": "secret plans"}')
        action.call('{"action": "contacts.find", "name": "Asha"}')
    logged = caplog.text
    assert "secret plans" not in logged and "+915550001111" not in logged and "+91999" not in logged
    assert "body=<12 chars>" in logged
