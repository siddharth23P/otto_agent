"""agent/phone/actions.py and the phone_find / phone_action tools: the phone's premapped actions as a
dictionary the agent searches, never listed in the prompt."""
import json
import logging

from agent.phone import JsonBackend, phone_tools
from agent.phone import actions
from agent.pipeline.toolkit import MAX_TOOL_DESCRIPTION_CHARS, bind_extra_tools, render_note
from tests.phone_fakes import FakePhone


def _a(group, name, summary, *params, effect="change", keywords=()):
    # "name", "name?" optional, "name#" integer, "name[]" list, "name:a|b" choices.
    fields = []
    for p in params:
        key, _, choices = p.partition(":")
        required = not key.endswith("?")
        key = key.rstrip("?")
        kind = "integer" if key.endswith("#") else "string_list" if key.endswith("[]") else "string"
        fields.append({"name": key.rstrip("#[]"), "type": kind, "required": required,
                       **({"choices": choices.split("|")} if choices else {})})
    return {"name": name, "summary": summary, "effect": effect, "group": group, "keywords": list(keywords),
            "params": fields}


#: The Android app's catalog (otto_android actions/ActionCatalog.kt) as its bridge sends it, 2026-09-17.
APP_MENU = [
    _a("clock", "alarm.set", "set an alarm in the clock app", "hour#", "minute#", "label?",
       "days[]?:mon|tue|wed|thu|fri|sat|sun", keywords=["wake", "morning"]),
    _a("clock", "timer.set", "start a countdown timer", "seconds", "label?"),
    _a("clock", "alarm.show", "open the clock app's alarms", effect="read"),
    _a("clock", "calendar.add", "open a new calendar event for the person to save", "title", "start", "end?",
       "location?", "notes?", effect="confirm"),
    _a("device", "flashlight", "turn the torch on or off", "on", keywords=["light"]),
    _a("device", "volume", "set a volume (0-100) or step it", "stream:media|ring|alarm|notification|call",
       "level?", "step?:up|down|mute|unmute"),
    _a("device", "media", "control what is playing", "command:play|pause|toggle|next|previous|stop"),
    _a("device", "dnd", "set Do Not Disturb", "mode:off|on|priority|alarms"),
    _a("device", "panel", "open a quick panel for the person to switch (apps cannot switch these)",
       "name:internet|wifi|bluetooth|nfc|volume", effect="confirm"),
    _a("message", "sms.compose", "write an SMS for the person to send", "to", "body", effect="confirm",
       keywords=["text", "message"]),
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


def _tool(phone):
    tools = {t.name: t for t in phone_tools(JsonBackend(phone))}
    return tools, tools.get("phone_action")


def test_search_finds_by_name_keyword_summary_and_field():
    def names(query):
        return [e["name"] for e in actions.search(APP_MENU, query)]
    assert names("alarm.set") == ["alarm.set"]
    assert names("set an alarm for 7am")[0] == "alarm.set"
    assert names("wake me up tomorrow")[0] == "alarm.set"
    assert names("text Asha that I'm late")[0] == "sms.compose"
    assert names("turn on the flashlight")[0] == "flashlight"
    assert names("light")[0] == "flashlight"
    assert names("share the pdf")[0] == "share"
    assert names("navigate to the airport")[0] == "maps"
    assert names("mute the ringer")[0] == "volume"
    assert names("zzz qqq") == [] and names("   ") == [] and names("please the") == []
    assert len(actions.search(APP_MENU, "the person to send")) <= actions.MAX_FOUND


def test_an_entry_is_described_in_full():
    alarm = actions.describe(actions.search(APP_MENU, "alarm.set")[0])
    assert alarm.splitlines()[0] == "alarm.set -- set an alarm in the clock app"
    assert "  days: list of mon|tue|wed|thu|fri|sat|sun, optional" in alarm
    assert "  label: text, optional" in alarm
    sms = actions.describe(actions.search(APP_MENU, "sms.compose")[0])
    assert "the person finishes it" in sms.splitlines()[0]
    ranged = actions.describe({"name": "x", "summary": "y", "params": [
        {"name": "hour", "type": "integer", "min": 0.0, "max": 23.0, "doc": "0-23"}]})
    assert "  hour: integer 0..23, required -- 0-23" in ranged
    assert actions.describe({"name": "alarm.show", "summary": "open alarms"}).endswith("(no fields)")


def test_a_bad_menu_is_no_menu():
    assert actions.clean_menu(None) == [] and actions.clean_menu([1, {"name": ""}, {"summary": "x"}]) == []
    assert actions.clean_menu([{"name": "a"}]) == [{"name": "a"}]


def test_no_registry_no_tools():
    tools, _ = _tool(FakePhone())
    assert "phone_find" not in tools and "phone_action" not in tools and "phone_screen" in tools
    tools, _ = _tool(ActionPhone(fail={"actions": {"message": "boom", "code": "failed"}}))
    assert "phone_find" not in tools


def test_the_prompt_does_not_grow_with_the_menu():
    class BigPhone(ActionPhone):
        def actions(self):
            return self._reply("actions", {"actions": APP_MENU * 10})
    small, _ = _tool(ActionPhone())
    big, _ = _tool(BigPhone())
    assert list(small)[:2] == ["phone_find", "phone_action"]
    assert not small["phone_find"].mutates and not small["phone_action"].mutates
    with bind_extra_tools(list(small.values())):
        small_note = render_note()
    with bind_extra_tools(list(big.values())):
        big_note = render_note()
    assert small_note == big_note and "alarm.set" not in small_note
    for t in small.values():
        assert len(t.description) <= MAX_TOOL_DESCRIPTION_CHARS


def test_find_returns_entries_and_the_whole_list():
    tools, _ = _tool(ActionPhone())
    found = tools["phone_find"].call('{"query": "set an alarm"}')
    assert found.ok and found.stdout.startswith("alarm.set -- set an alarm")
    assert "hour: integer, required" in found.stdout and found.stdout.endswith("...fields}.")
    listing = tools["phone_find"].call("{}")
    assert listing.stdout.splitlines() == ["4 actions; search one to see its fields.", "clock: alarm.set",
                                           "message: sms.compose, contacts.find", "files: share"]
    nothing = tools["phone_find"].call('{"query": "order a pizza"}')
    assert nothing.ok and "use the screen tools" in nothing.stdout


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
    assert result.ok and result.stdout.startswith("GUARD: phone_action: SMS ready")
    assert "handed control to the person" in result.stdout


def test_data_comes_back_as_json():
    data = {"contacts": [{"name": "Asha", "phones": ["+911234567890"]}]}
    _, action = _tool(ActionPhone({"contacts.find": {"done": "1 contact", "data": data}}))
    result = action.call('{"action": "contacts.find", "name": "Asha"}')
    assert result.stdout.split("\n", 1) == ["1 contact", json.dumps(data)]


def test_unknown_actions_and_bad_fields():
    phone = ActionPhone(fail={"run_action": {"message": "hour is at most 23", "code": "invalid"}})
    _, action = _tool(phone)
    unknown = action.call('{"action": "alarm.create"}')
    assert not unknown.ok and "no action 'alarm.create'; nearest: alarm.set" in unknown.stderr
    assert not any(c[0] == "run_action" for c in phone.calls)
    refused = action.call('{"action": "alarm.set", "hour": 30, "minute": 0}')
    assert not refused.ok and "at most 23" in refused.stderr and not refused.stderr.startswith("GUARD")
    assert "alarm.set -- set an alarm" in refused.stderr and "minute: integer, required" in refused.stderr


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
