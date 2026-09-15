"""agent/server/: the protocol end to end, offline, with a fake pipeline
and a client that plays the phone."""
from __future__ import annotations

import asyncio
import json
import threading

import pytest
from websockets.sync.client import connect

from agent import embed
from agent.pipeline import run as pipeline
from agent.pipeline.toolkit import dispatch_table
from agent.server import protocol
from agent.server.app import OttoServer, bound_port
from tests.phone_fakes import BLINKIT_SEARCH

TOKEN = "s3cret-token"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setattr(embed, "_configured", {})
    for name in embed.KEY_VARS:
        monkeypatch.setenv(name, "test-placeholder-not-a-real-key")
    embed.configure(tmp_path / "home")


@pytest.fixture
def server(configured):
    """An OttoServer on a free loopback port, on its own loop and thread."""
    yield from _serving(OttoServer(TOKEN))


def _serving(srv):
    loop = asyncio.new_event_loop()
    ready = asyncio.Event()
    stop = loop.create_future()

    async def main():
        task = asyncio.ensure_future(srv.run("127.0.0.1", 0, ready=ready))
        await ready.wait()
        await stop
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

    thread = threading.Thread(target=lambda: loop.run_until_complete(main()), daemon=True)
    thread.start()
    for _ in range(200):
        if bound_port(srv):
            break
        threading.Event().wait(0.02)
    yield f"ws://127.0.0.1:{bound_port(srv)}/"
    loop.call_soon_threadsafe(stop.set_result, None)
    thread.join(10)
    loop.close()


def _hello(url, token=TOKEN, version=protocol.PROTOCOL_VERSION, capabilities=("phone",)):
    ws = connect(url, open_timeout=5)
    ws.send(protocol.encode("hello", protocol_version=version, token=token, capabilities=list(capabilities)))
    return ws, json.loads(ws.recv(timeout=5))


def _recv_until(ws, kind, *, answer_device=None, timeout=10):
    """Frames until one of `kind`, answering device_calls on the way."""
    seen = []
    while True:
        frame = json.loads(ws.recv(timeout=timeout))
        seen.append(frame)
        if frame["type"] == "device_call" and answer_device is not None:
            ws.send(json.dumps({"type": "device_result", "id": frame["id"], **answer_device(frame)}))
            continue
        if frame["type"] == kind or (kind == "event" and frame["type"] == "event"):
            if kind != "event" or frame["event"]["type"] in ("final", "error", "ask"):
                return frame, seen


def test_hello_negotiates_and_a_bad_token_is_refused(server):
    ws, reply = _hello(server)
    assert reply["type"] == "hello_ok"
    assert reply["protocol_version"] == protocol.PROTOCOL_VERSION and reply["api_version"] == embed.API_VERSION
    ws.send(protocol.encode("ping"))
    assert json.loads(ws.recv(timeout=5))["type"] == "pong"
    ws.close()
    ws, reply = _hello(server, token="wrong")
    assert reply["type"] == "error" and reply["code"] == "hello" and "token" in reply["message"]
    ws.close()


def test_a_newer_client_is_told_to_update_otto(server):
    ws, reply = _hello(server, version=protocol.PROTOCOL_VERSION + 1)
    assert reply["type"] == "error" and "update otto" in reply["message"]
    ws.close()


def test_a_turn_reaches_the_phone_through_device_calls_and_pauses_for_an_answer(server, monkeypatch):
    def fake_run(text, **kwargs):
        # The phone tools are bound: read the screen through the socket.
        result = dispatch_table()["phone_screen"]("{}")
        assert result.ok, result.stderr
        assert '[4] "ADD" button' in result.stdout
        yield {"agent": {"board": ["read the screen"]}}
        yield {"__ask__": {"question": "Add which?", "choices": ["Amul", "Nandini"], "thread_id": "t1"}}

    def fake_resume(answer, **kwargs):
        yield {"__final__": {"final_output": f"added {answer}"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    monkeypatch.setattr(pipeline, "resume_pipeline_stream", fake_resume)

    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="add milk"))
    calls = []

    def phone(frame):
        calls.append((frame["method"], frame["args"]))
        return {"ok": True, "data": BLINKIT_SEARCH}

    ask, seen = _recv_until(ws, "event", answer_device=phone)
    assert calls == [("tree", [])]
    assert ask["event"]["type"] == "ask" and ask["event"]["choices"] == ["Amul", "Nandini"]
    session_id = ask["session_id"]
    assert any(f["type"] == "event" and f["event"]["type"] == "board" for f in seen)
    ws.send(protocol.encode("answer", session_id=session_id, thread_id="t1", text="Amul"))
    final, _ = _recv_until(ws, "event")
    assert final["event"]["type"] == "final" and final["event"]["text"] == "added Amul"

    ws.send(protocol.encode("sessions", op="list"))
    listed = json.loads(ws.recv(timeout=5))
    assert listed["type"] == "sessions_result" and listed["sessions"][0]["id"] == session_id
    ws.close()


def test_a_phone_that_refuses_hands_over(server, monkeypatch):
    def fake_run(text, **kwargs):
        result = dispatch_table()["phone_screen"]("{}")
        yield {"__final__": {"final_output": result.stderr}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="what is on screen"))
    final, _ = _recv_until(ws, "event", answer_device=lambda f: {
        "ok": False, "error": {"code": "guard", "message": "payment app in front", "handover": True}})
    assert final["event"]["text"].startswith("GUARD: phone_screen: payment app in front")
    ws.close()


def test_a_client_without_phone_capability_gets_no_phone_tools(server, monkeypatch):
    def fake_run(text, **kwargs):
        yield {"__final__": {"final_output": "phone_screen" in dispatch_table() and "yes" or "no"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server, capabilities=())
    ws.send(protocol.encode("turn", text="hi"))
    final, _ = _recv_until(ws, "event")
    assert final["event"]["text"] == "no"
    ws.close()


def test_a_second_turn_in_a_running_session_is_refused(server, monkeypatch):
    gate = threading.Event()

    def fake_run(text, **kwargs):
        gate.wait(5)
        yield {"__final__": {"final_output": "ok"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="one"))
    started = json.loads(ws.recv(timeout=5))
    sid = started["session_id"]
    ws.send(protocol.encode("turn", session_id=sid, text="two"))
    busy = json.loads(ws.recv(timeout=5))
    assert busy["type"] == "error" and busy["code"] == "busy"
    gate.set()
    final, _ = _recv_until(ws, "event")
    assert final["event"]["type"] == "final"
    ws.close()


def test_the_cli_command_is_registered_and_the_token_persists(monkeypatch, tmp_path):
    from agent.cli import serve as serve_cmd
    from agent.cli.main import app
    from agent.config import envfile

    assert any(c.name == "serve" for c in app.registered_commands)
    monkeypatch.setattr(envfile, "ENV_PATH", tmp_path / ".env")
    monkeypatch.delenv(serve_cmd.TOKEN_ENV, raising=False)
    first = serve_cmd.ensure_token(None)
    assert first and serve_cmd.TOKEN_ENV + "=" in (tmp_path / ".env").read_text()
    assert serve_cmd.ensure_token(None) == first
    assert serve_cmd.ensure_token("given") == "given"
    assert serve_cmd.pairing_url("0.0.0.0", 8765) == "ws://127.0.0.1:8765/"


def test_a_connection_may_not_hold_more_than_the_session_cap(server, monkeypatch):
    from agent.server import app as server_app

    monkeypatch.setattr(server_app, "MAX_SESSIONS_PER_CONNECTION", 2)
    ws, _ = _hello(server)
    for _ in range(2):
        ws.send(protocol.encode("sessions", op="open"))
        assert json.loads(ws.recv(timeout=5))["type"] == "sessions_result"
    ws.send(protocol.encode("sessions", op="open"))
    reply = json.loads(ws.recv(timeout=5))
    assert reply["type"] == "error" and reply["code"] == "no_session" and "sessions" in reply["message"]
    ws.close()


def test_a_browser_page_is_refused_before_the_token_is_tried(server):
    """A native client sends no Origin header; a page always does, and a
    page has no business on this port unless the server was told its
    origin (2026-09-14 review)."""
    from websockets.exceptions import InvalidStatus

    with pytest.raises(InvalidStatus) as caught:
        connect(server, open_timeout=5, additional_headers={"Origin": "http://evil.example"})
    assert caught.value.response.status_code == 403
    ws, reply = _hello(server)  # no Origin: as before
    assert reply["type"] == "hello_ok"
    ws.close()


def test_an_allowed_origin_connects(configured):
    srv = OttoServer(TOKEN, allowed_origins=("http://localhost:3000/",))
    assert srv.check_origin(None, _Request({"Origin": "http://localhost:3000"})) is None
    assert srv.check_origin(None, _Request({})) is None
    assert srv.check_origin(_Conn(), _Request({"Origin": "http://evil.example"})) == (403, "origin not allowed\n")


class _Request:
    def __init__(self, headers):
        self.headers = headers


class _Conn:
    def respond(self, status, text):
        return (int(status), text)


def test_turns_run_on_the_servers_own_bounded_pool(server, monkeypatch):
    import threading

    names = []

    def fake_run(text, **kwargs):
        names.append(threading.current_thread().name)
        yield {"__final__": {"text": "done"}}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server, capabilities=())
    ws.send(protocol.encode("turn", text="hi"))
    _recv_until(ws, "event")
    ws.close()
    assert names and names[0].startswith("otto-turn")


def test_a_session_id_that_is_not_one_is_refused_and_touches_nothing(server):
    """`sessions{op:"delete", session_id:"../lessons"}` used to reach
    `sessions.delete` as a path (2026-09-15)."""
    from agent.memory import store as store_module

    store_module.DB_DIR.mkdir(parents=True, exist_ok=True)
    planted = store_module.DB_DIR / "lessons.db"
    planted.write_text("the bank")
    ws, _ = _hello(server)
    for frame in (protocol.encode("sessions", op="delete", session_id="../lessons"),
                  protocol.encode("sessions", op="delete", session_id="lessons"),
                  protocol.encode("sessions", op="open", ref="../x"),
                  protocol.encode("sessions", op="transcript", ref="a/b"),
                  protocol.encode("turn", session_id="../x", text="hi"),
                  protocol.encode("answer", session_id="../x", thread_id="t", text="a"),
                  protocol.encode("cancel", session_id="nope")):
        ws.send(frame)
        reply = json.loads(ws.recv(timeout=5))
        assert reply["type"] == "error" and reply["code"] == "invalid_session", frame
    assert planted.read_text() == "the bank"
    ws.send(protocol.encode("ping"))
    assert json.loads(ws.recv(timeout=5))["type"] == "pong"
    ws.close()



# --------------------------------------------------------------------------
# where a turn runs, and what the token grants (2026-09-15)
# --------------------------------------------------------------------------

def _spy_run(monkeypatch):
    got: list[dict] = []
    real = embed.SessionHandle.run

    def spy(self, text, **kwargs):
        got.append(kwargs)
        return real(self, text, **kwargs)

    monkeypatch.setattr(embed.SessionHandle, "run", spy)
    monkeypatch.setattr(pipeline, "run_pipeline_stream",
                        lambda text, **kw: iter([{"__final__": {"final_output": "ok"}, "__trace_id__": None}]))
    return got


def test_a_turn_can_say_whether_it_runs_on_the_phone_and_started_carries_the_budget(server, monkeypatch):
    from agent.embed import SUBPROCESS_TOOLS
    from agent.pipeline.budget import default_budget

    got = _spy_run(monkeypatch)
    ws, _ = _hello(server)
    for mode, expected in (("off", False), ("on", True), ("auto", None), (None, None)):
        fields = {"text": "hi"} if mode is None else {"text": "hi", "phone": mode}
        ws.send(protocol.encode("turn", **fields))
        started = json.loads(ws.recv(timeout=5))
        assert started["event"] == {"type": "started", "session_id": started["session_id"],
                                    "budget_max": default_budget().max_model_calls}
        final, _ = _recv_until(ws, "event")
        assert final["event"]["type"] == "final"
        assert got[-1]["phone"] is expected and tuple(got[-1]["off_phone_disabled_tools"]) == ()
    assert not set(SUBPROCESS_TOOLS) & set(got[0]["off_phone_disabled_tools"])
    ws.send(protocol.encode("turn", text="hi", phone="maybe"))
    reply = json.loads(ws.recv(timeout=5))
    assert reply["type"] == "error" and reply["code"] == "invalid"
    ws.close()
    ws, _ = _hello(server, capabilities=())
    ws.send(protocol.encode("turn", text="hi", phone="on"))
    reply = json.loads(ws.recv(timeout=5))
    assert reply["type"] == "error" and reply["code"] == "no_phone"
    ws.close()


@pytest.fixture
def no_exec_server(configured):
    yield from _serving(OttoServer(TOKEN, no_exec=True))


def test_no_exec_takes_the_subprocess_tools_from_every_turn(no_exec_server, monkeypatch):
    from agent.embed import SUBPROCESS_TOOLS

    got = _spy_run(monkeypatch)
    for capabilities in (("phone",), ()):
        ws, _ = _hello(no_exec_server, capabilities=capabilities)
        ws.send(protocol.encode("turn", text="hi", phone="off" if capabilities else "auto"))
        _recv_until(ws, "event")
        ws.close()
        assert set(SUBPROCESS_TOOLS) <= set(got[-1]["off_phone_disabled_tools"])
        assert set(SUBPROCESS_TOOLS) <= set(got[-1]["disabled_tools"])


def test_listening_beyond_loopback_says_what_the_token_grants():
    from agent.cli import serve as serve_cmd

    for host in ("127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"):
        assert serve_cmd.exposure_warning(host) is None, host
    for host in ("0.0.0.0", "::", "", "192.168.1.5", "my-laptop.local"):
        warning = serve_cmd.exposure_warning(host)
        assert warning and "pairing token" in warning and "run commands" in warning, host
        assert "adb reverse tcp:8765 tcp:8765" in warning and "--no-exec" in warning
    quieter = serve_cmd.exposure_warning("0.0.0.0", no_exec=True)
    assert "run commands" not in quieter and "pairing token" in quieter
    import inspect

    assert "no_exec" in inspect.signature(serve_cmd.serve).parameters


# --------------------------------------------------------------------------
# protocol 2 framing: features, request ids, lanes (2026-09-15)
# --------------------------------------------------------------------------

def _frame(ws, timeout=5):
    return json.loads(ws.recv(timeout=timeout))


def test_every_client_is_told_the_features_and_a_v1_client_is_still_served(server):
    for version in (1, protocol.PROTOCOL_VERSION):
        ws, reply = _hello(server, version=version)
        assert reply["type"] == "hello_ok" and reply["protocol_version"] == 2 and reply["min_protocol"] == 1
        assert "ids" in reply["features"] and "turn.phone" in reply["features"]
        ws.close()


def test_request_ids_are_echoed_and_a_message_without_one_is_answered_as_in_v1(server):
    ws, _ = _hello(server)
    ws.send(json.dumps({"type": "sessions", "op": "list", "id": "r1"}))
    assert _frame(ws) == {"type": "sessions_result", "id": "r1", "op": "list", "sessions": []}
    ws.send(json.dumps({"type": "sessions", "op": "list", "id": 7}))
    assert _frame(ws)["id"] == 7
    ws.send(protocol.encode("sessions", op="list"))
    assert ws.recv(timeout=5) == protocol.encode("sessions_result", op="list", sessions=[])
    ws.send(json.dumps({"type": "sessions", "op": "nope", "id": "r2"}))
    assert _frame(ws) == {"type": "error", "id": "r2", "code": "unknown", "message": "unknown sessions op 'nope'"}
    ws.send(json.dumps({"type": "whatever", "id": "r3"}))
    assert _frame(ws)["id"] == "r3"
    ws.send(json.dumps({"type": "ping", "id": "p"}))
    assert _frame(ws) == {"type": "pong", "id": "p"}
    ws.send(protocol.encode("ping"))
    assert ws.recv(timeout=5) == protocol.encode("pong")
    ws.send(json.dumps({"type": "turn", "text": "", "id": "t"}))
    assert _frame(ws) == {"type": "error", "id": "t", "code": "empty", "message": "turn needs text"}
    for bad in (True, "x" * 65, [1], "", 2 ** 60):
        ws.send(json.dumps({"type": "sessions", "op": "list", "id": bad}))
        reply = _frame(ws)
        assert reply["type"] == "error" and reply["code"] == "invalid" and "id" not in reply, bad
    ws.close()


def test_a_failing_request_answers_with_its_id_and_the_connection_lives(server, monkeypatch):
    def broken(self, limit=20):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(embed.Runtime, "list_sessions", broken)
    ws, _ = _hello(server)
    ws.send(json.dumps({"type": "sessions", "op": "list", "id": "boom"}))
    assert _frame(ws) == {"type": "error", "id": "boom", "code": "failed",
                          "message": "RuntimeError: disk on fire"}
    ws.send(protocol.encode("ping"))
    assert _frame(ws)["type"] == "pong"
    ws.close()


def test_a_slow_request_does_not_hold_up_a_ping(server, monkeypatch):
    release = threading.Event()

    def slow(self, limit=20):
        release.wait(5)
        return []

    monkeypatch.setattr(embed.Runtime, "list_sessions", slow)
    ws, _ = _hello(server)
    ws.send(json.dumps({"type": "sessions", "op": "list", "id": "slow"}))
    ws.send(json.dumps({"type": "ping", "id": "quick"}))
    assert _frame(ws) == {"type": "pong", "id": "quick"}
    release.set()
    assert _frame(ws)["id"] == "slow"
    ws.close()


def test_a_running_session_is_not_deleted_and_a_turn_id_rides_on_started(server, monkeypatch):
    gate = threading.Event()

    def fake_run(text, **kwargs):
        gate.wait(5)
        yield {"__final__": {"final_output": "ok"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(json.dumps({"type": "turn", "text": "one", "id": "turn-1"}))
    started = _frame(ws)
    assert started["id"] == "turn-1" and started["event"]["type"] == "started"
    sid = started["session_id"]
    ws.send(json.dumps({"type": "sessions", "op": "delete", "session_id": sid, "id": "d1"}))
    assert _frame(ws) == {"type": "error", "id": "d1", "code": "busy",
                          "message": "that session is running a turn; stop it first"}
    gate.set()
    final, _ = _recv_until(ws, "event")
    assert final["event"]["type"] == "final" and "id" not in final
    ws.send(json.dumps({"type": "sessions", "op": "transcript", "ref": sid, "id": "tr"}))
    transcript = _frame(ws)
    assert transcript["id"] == "tr" and transcript["session_id"] == sid
    ws.send(protocol.encode("sessions", op="transcript", ref=sid))
    assert _frame(ws)["id"] == sid, "without a request id, v1's shape"
    ws.send(json.dumps({"type": "sessions", "op": "delete", "session_id": sid, "id": "d2"}))
    assert _frame(ws) == {"type": "sessions_result", "id": "d2", "op": "delete", "session_id": sid,
                          "deleted": True}
    ws.close()


def test_a_connection_knows_whether_its_client_is_on_this_computer():
    from agent.server.app import _peer_is_loopback

    class _Ws:
        def __init__(self, address):
            self.remote_address = address

    for address in (("127.0.0.1", 5), ("::1", 5, 0, 0), ("::ffff:127.0.0.1", 5, 0, 0)):
        assert _peer_is_loopback(_Ws(address)), address
    for address in (("10.0.0.2", 5), ("192.168.1.9", 5), None, ("not an address", 1)):
        assert not _peer_is_loopback(_Ws(address)), address


# --------------------------------------------------------------------------
# sessions: close, rename, export, import, usage (protocol 2)
# --------------------------------------------------------------------------

def _request(ws, **message):
    ws.send(json.dumps(message))
    return _frame(ws)


def _one_turn(ws, monkeypatch, text="hello there", tokens=(30, 12)):
    def fake_run(t, **kwargs):
        kwargs["usage"].record("nobody:unpriced", {"input_tokens": tokens[0], "output_tokens": tokens[1]})
        yield {"__final__": {"final_output": "hi back"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws.send(protocol.encode("turn", text=text))
    final, _ = _recv_until(ws, "event")
    assert final["event"]["type"] == "final"
    return final["session_id"]


def test_closing_a_session_frees_its_place_under_the_cap(server, monkeypatch):
    from agent.server import app as server_app

    monkeypatch.setattr(server_app, "MAX_SESSIONS_PER_CONNECTION", 2)
    ws, _ = _hello(server)
    first = _request(ws, type="sessions", op="open", id=1)["session_id"]
    _request(ws, type="sessions", op="open", id=2)
    assert _request(ws, type="sessions", op="open", id=3)["code"] == "no_session"
    assert _request(ws, type="sessions", op="close", session_id=first, id=4) == {
        "type": "sessions_result", "id": 4, "op": "close", "session_id": first, "closed": True}
    assert _request(ws, type="sessions", op="open", id=5)["type"] == "sessions_result"
    assert _request(ws, type="sessions", op="close", session_id=first, id=6)["closed"] is False
    ws.close()


def test_a_session_is_renamed_and_the_list_shows_it(server, monkeypatch):
    ws, _ = _hello(server)
    sid = _one_turn(ws, monkeypatch)
    assert _request(ws, type="sessions", op="rename", session_id=sid, title="  tide   notes ", id="r") == {
        "type": "sessions_result", "id": "r", "op": "rename", "session_id": sid, "title": "tide notes"}
    rows = _request(ws, type="sessions", op="list", id="l")["sessions"]
    assert rows[0]["id"] == sid and rows[0]["title"] == "tide notes"
    assert _request(ws, type="sessions", op="close", session_id=sid, id="c")["closed"] is True
    assert _request(ws, type="sessions", op="rename", session_id=sid, title="closed", id="r2")["title"] == "closed"
    assert _request(ws, type="sessions", op="rename", session_id="f" * 32, title="x", id="r3")["code"] == "no_session"
    assert _request(ws, type="sessions", op="rename", session_id=sid, title="   ", id="r4")["code"] == "invalid"
    assert _request(ws, type="sessions", op="rename", session_id=sid, title="x" * 201, id="r5")["code"] == "invalid"
    ws.close()


def test_an_export_imports_back_as_a_session_and_carries_no_paths(server, monkeypatch):
    import re

    from agent.memory import sessions as index

    ws, _ = _hello(server)
    sid = _one_turn(ws, monkeypatch, text="what tides are")
    index.touch(sid, title="", workspace="/Users/someone/private", turns=1)
    exported = _request(ws, type="sessions", op="export", session_id=sid, id="e")
    assert exported["op"] == "export" and exported["session_id"] == sid
    assert re.fullmatch(r"otto-session-[0-9a-f]{8}-\d{4}-\d{2}-\d{2}\.json", exported["filename"])
    assert exported["data"]["session"]["workspace"] is None and "/Users/" not in json.dumps(exported["data"])
    imported = _request(ws, type="sessions", op="import", data=exported["data"], id="i")
    assert imported["type"] == "sessions_result" and imported["op"] == "import"
    assert imported["session_id"] != sid and imported["title"] == "what tides are" and imported["turns"] == 1
    listed = {row["id"] for row in _request(ws, type="sessions", op="list", id="l")["sessions"]}
    assert listed == {sid, imported["session_id"]}
    transcript = _request(ws, type="sessions", op="transcript", ref=imported["session_id"], id="t")
    assert [m["text"] for m in transcript["messages"]] == ["what tides are", "hi back"]
    # A client-chosen workspace never comes in with an import.
    planted = dict(exported["data"], session={**exported["data"]["session"], "id": "a" * 32, "workspace": "/"})
    again = _request(ws, type="sessions", op="import", data=planted, id="i2")
    assert again["session_id"] == "a" * 32 and index.get("a" * 32).workspace is None
    for bad in ({}, {"session": {}, "pending": [{"tier": "z", "text": "x"}]}, "not an object"):
        reply = _request(ws, type="sessions", op="import", data=bad, id="bad")
        assert reply["type"] == "error" and reply["code"] == "invalid", bad
    assert _request(ws, type="sessions", op="export", session_id="b" * 32, id="e2")["code"] == "no_session"
    ws.close()


def test_usage_reports_the_sessions_spend_turn_by_turn(server, monkeypatch):
    ws, _ = _hello(server)
    sid = _one_turn(ws, monkeypatch, tokens=(30, 12))
    ws.send(protocol.encode("turn", session_id=sid, text="again"))
    _recv_until(ws, "event")
    usage = _request(ws, type="sessions", op="usage", session_id=sid, id="u")
    assert usage["op"] == "usage" and usage["session_id"] == sid
    assert usage["usage"]["total_tokens"] == 84 and usage["turn_tokens"] == [42, 42]
    assert usage["turn"] == {"tokens": 42, "calls": 1, "cost": None}
    assert usage["title"] == "hello there" and usage["turns"] == 2
    assert _request(ws, type="sessions", op="usage", session_id="c" * 32, id="u2")["code"] == "no_session"
    ws.close()


# --------------------------------------------------------------------------
# setup, doctor, models (protocol 2)
# --------------------------------------------------------------------------

def test_setup_status_is_masks_and_names_never_a_key(server):
    ws, _ = _hello(server)
    raw = None
    ws.send(json.dumps({"type": "setup", "op": "status", "id": "s"}))
    raw = ws.recv(timeout=10)
    status = json.loads(raw)
    assert status["type"] == "setup_result" and status["id"] == "s" and status["op"] == "status"
    assert status["ready"] is True and status["setup_write"] is True
    assert set(status["keys"]) == set(embed.KEY_VARS)
    assert {row["name"] for row in status["vendors"]} >= {"inception", "openai", "anthropic", "gemini"}
    assert set(status["vendors"][0]) == {"name", "label", "key_var", "url_var", "key_present", "url_present",
                                         "custom", "masked_key"}
    assert status["version"]["api"] == embed.API_VERSION
    assert "test-placeholder-not-a-real-key" not in raw
    ws.close()


def test_a_key_is_set_only_by_name_from_this_computer_and_never_echoed(server, monkeypatch, caplog):
    import logging
    import os

    caplog.set_level(logging.DEBUG)
    secret = "sk-live-do-not-leak-4321"
    original = os.environ.get("OPENAI_API_KEY", "")
    ws, _ = _hello(server)
    frames = []
    try:
        for bad in ({"name": "OTTO_SERVE_TOKEN", "value": secret}, {"name": "PATH", "value": secret},
                    {"name": "OPENAI_API_KEY", "value": secret + "\nINCEPTION_API_KEY=x"},
                    {"name": "OPENAI_API_KEY", "value": "x" * 513}):
            ws.send(json.dumps({"type": "setup", "op": "set_key", "id": "bad", **bad}))
            frames.append(ws.recv(timeout=10))
            assert json.loads(frames[-1])["code"] == "invalid"
        ws.send(json.dumps({"type": "setup", "op": "set_key", "name": "OPENAI_API_KEY", "value": secret, "id": "k"}))
        frames.append(ws.recv(timeout=10))
        assert json.loads(frames[-1]) == {"type": "setup_result", "id": "k", "op": "set_key",
                                          "name": "OPENAI_API_KEY", "masked": "********4321", "ready": True}
        assert os.environ["OPENAI_API_KEY"] == secret
    finally:
        ws.close()
        embed.set_key("OPENAI_API_KEY", original)
    assert not any(secret in frame for frame in frames)
    assert secret not in caplog.text


def test_a_client_elsewhere_may_read_setup_but_not_change_it(server, monkeypatch):
    from agent.server import app as server_app

    monkeypatch.setattr(server_app, "_peer_is_loopback", lambda websocket: False)
    ws, _ = _hello(server)
    status = _request(ws, type="setup", op="status", id=1)
    assert status["setup_write"] is False
    reply = _request(ws, type="setup", op="set_key", name="OPENAI_API_KEY", value="sk-x", id=2)
    assert reply["type"] == "error" and reply["code"] == "forbidden" and "sk-x" not in json.dumps(reply)
    ws.close()


@pytest.fixture
def remote_setup_server(configured):
    yield from _serving(OttoServer(TOKEN, allow_remote_setup=True))


def test_allow_remote_setup_lets_a_client_elsewhere_set_a_key(remote_setup_server, monkeypatch):
    import os

    from agent.server import app as server_app

    monkeypatch.setattr(server_app, "_peer_is_loopback", lambda websocket: False)
    original = os.environ.get("GEMINI_API_KEY", "")
    ws, _ = _hello(remote_setup_server)
    try:
        assert _request(ws, type="setup", op="status", id=1)["setup_write"] is True
        assert _request(ws, type="setup", op="set_key", name="GEMINI_API_KEY", value="g-1234", id=2)["masked"] \
            == "********1234"
    finally:
        ws.close()
        embed.set_key("GEMINI_API_KEY", original)


def test_keys_do_not_change_under_a_running_turn(server, monkeypatch):
    gate = threading.Event()

    def fake_run(text, **kwargs):
        gate.wait(5)
        yield {"__final__": {"final_output": "ok"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="one"))
    assert _frame(ws)["event"]["type"] == "started"
    other, _ = _hello(server)
    reply = _request(other, type="setup", op="set_key", name="OPENAI_API_KEY", value="sk-9999", id="k")
    assert reply["type"] == "error" and reply["code"] == "busy", "a turn on another connection counts"
    gate.set()
    _recv_until(ws, "event")
    ws.close()
    other.close()


def test_a_probe_names_a_vendor_and_reports_its_models(server, monkeypatch):
    from agent.router import setup as router_setup
    from agent.router.llm_provider.base import Capability, HealthReport, ModelInfo, ProviderStatus

    seen = []

    def fake_probe(name):
        seen.append(name)
        return router_setup.ProbeResult(HealthReport(name, ProviderStatus.OK, model_count=1),
                                        [ModelInfo("gpt-5-mini", "openai", capabilities=frozenset({Capability.CHAT}))])

    monkeypatch.setattr(router_setup, "probe", fake_probe)
    ws, _ = _hello(server)
    assert _request(ws, type="setup", op="probe", name="openai", id="p") == {
        "type": "setup_result", "id": "p", "op": "probe", "name": "openai", "ok": True, "status": "ok",
        "detail": "", "model_count": 1,
        "models": [{"spec": "openai:gpt-5-mini", "provider": "openai", "id": "gpt-5-mini", "display_name": None,
                    "capabilities": ["chat"], "context_window": None, "max_output_tokens": None}]}
    for bad in ("../x", "nope", None):
        assert _request(ws, type="setup", op="probe", name=bad, id="q")["code"] == "invalid"
    assert seen == ["openai"]
    ws.close()


def test_doctor_reports_each_provider_and_the_conclusion(server, monkeypatch):
    from agent.router import llm_provider
    from agent.router.llm_provider.base import HealthReport, ProviderStatus

    monkeypatch.setattr(llm_provider, "health_report", lambda: [
        HealthReport("inception", ProviderStatus.OK, 3, ""),
        HealthReport("openai", ProviderStatus.AUTH_FAILED, 0, "401")])
    ws, _ = _hello(server)
    report = _request(ws, type="doctor", id="d")
    assert report["type"] == "doctor_result" and report["id"] == "d"
    assert report["providers"] == [
        {"provider": "inception", "status": "ok", "models": 3, "detail": ""},
        {"provider": "openai", "status": "auth failed", "models": 0, "detail": "401"}]
    assert report["ready"] is True and report["required"] == "inception"
    assert "inception" not in report["also_configured"] and isinstance(report["also_configured"], list)
    ws.close()


def test_a_slow_model_catalogue_does_not_block_a_pong(server, monkeypatch):
    from agent.router import llm_provider
    from agent.router.llm_provider.base import Capability, ModelInfo

    release = threading.Event()

    def slow(capability=None):
        release.wait(5)
        return [ModelInfo("mercury-2.5", "inception", display_name="Mercury 2.5",
                          capabilities=frozenset({Capability.CHAT, Capability.TOOLS}), context_window=128000)]

    monkeypatch.setattr(llm_provider, "all_models", slow)
    ws, _ = _hello(server)
    ws.send(json.dumps({"type": "models", "id": "m"}))
    ws.send(json.dumps({"type": "ping", "id": "p"}))
    assert _frame(ws) == {"type": "pong", "id": "p"}
    ws.send(json.dumps({"type": "sessions", "op": "list", "id": "l"}))
    assert _frame(ws)["id"] == "l", "a slow catalogue does not hold up the session lane either"
    release.set()
    assert _frame(ws) == {"type": "models_result", "id": "m", "models": [
        {"spec": "inception:mercury-2.5", "provider": "inception", "id": "mercury-2.5",
         "display_name": "Mercury 2.5", "capabilities": ["chat", "tools"], "context_window": 128000,
         "max_output_tokens": None}]}
    ws.close()


def test_pin_options_live_with_the_setup_data_layer():
    from agent.cli import setup_screen
    from agent.router import setup as router_setup

    assert setup_screen.pin_options is router_setup.pin_options and setup_screen.NO_PIN == router_setup.NO_PIN


# --------------------------------------------------------------------------
# routing (protocol 2)
# --------------------------------------------------------------------------

@pytest.fixture
def routes_file(tmp_path):
    """routes.json in tmp_path for every thread: bind_routes is a contextvar,
    and the server applies pins on its own threads. The live table is
    re-applied from the usual file afterwards."""
    import os

    from agent.router.reload import reload_everything

    path = tmp_path / "routes.json"
    before = os.environ.get("OTTO_ROUTES")
    os.environ["OTTO_ROUTES"] = str(path)
    try:
        yield path
    finally:
        if before is None:
            os.environ.pop("OTTO_ROUTES", None)
        else:
            os.environ["OTTO_ROUTES"] = before
        reload_everything()


def test_routing_lists_every_task_with_its_pin_default_binding_and_phone_seat(server, routes_file):
    from agent.phone import PHONE_SEATS
    from agent.router import overrides
    from agent.router.mapping import Task

    ws, _ = _hello(server)
    reply = _request(ws, type="routing", op="list", id="r")
    assert reply["type"] == "routing_result" and reply["op"] == "list"
    by = {row["task"]: row for row in reply["routes"]}
    assert set(by) == {t.value for t in Task}
    assert set(by["reason"]) == {"task", "pin", "default", "provider_only", "phone_seat"}
    assert by["web"]["provider_only"]["provider"] == "anthropic" and by["reason"]["provider_only"] is None
    assert by["evaluate"]["phone_seat"] == PHONE_SEATS["evaluate"] and by["reason"]["phone_seat"] is None
    assert by["reason"]["default"] == (overrides.shipped(Task.REASON)[0].spec
                                       or overrides.shipped(Task.REASON)[0].provider_name)
    assert all(row["pin"] is None for row in reply["routes"])
    ws.close()


def test_routing_options_are_the_tuis_pin_choices(server, monkeypatch):
    from agent.router import llm_provider
    from agent.router.llm_provider.base import Capability, ModelInfo

    monkeypatch.setattr(llm_provider, "all_models", lambda capability=None: [
        ModelInfo("claude-x", "anthropic", capabilities=frozenset({Capability.CHAT, Capability.TOOLS})),
        ModelInfo("gpt-5-mini", "openai", capabilities=frozenset({Capability.CHAT, Capability.TOOLS}))])
    ws, _ = _hello(server)
    reply = _request(ws, type="routing", op="options", task="web", id="o")
    assert reply["type"] == "routing_result" and reply["task"] == "web"
    assert reply["options"][0] == {"label": "(no pin — default route)", "spec": ""}
    assert all(o["spec"].startswith("anthropic:") for o in reply["options"][1:]), "web is anthropic-only"
    assert _request(ws, type="routing", op="options", task="../x", id="bad")["code"] == "invalid"
    ws.close()


def test_a_pin_is_set_and_cleared_from_this_computer_and_a_bad_one_is_named(server, routes_file):
    ws, _ = _hello(server)
    pinned = _request(ws, type="routing", op="pin", task="reason", spec="openai:gpt-5-mini", id="p")
    assert pinned["type"] == "routing_result", pinned
    assert {k: pinned[k] for k in ("id", "op", "task", "pin")} == {
        "id": "p", "op": "pin", "task": "reason", "pin": "openai:gpt-5-mini"}
    assert isinstance(pinned["problems"], list)
    assert json.loads(routes_file.read_text())["pins"]["reason"] == "openai:gpt-5-mini"
    rows = {r["task"]: r for r in _request(ws, type="routing", op="list", id="l")["routes"]}
    assert rows["reason"]["pin"] == "openai:gpt-5-mini"
    wrong = _request(ws, type="routing", op="pin", task="web", spec="openai:gpt-5-mini", id="w")
    assert wrong["type"] == "error" and wrong["code"] == "invalid_pin" and "anthropic" in wrong["message"]
    assert _request(ws, type="routing", op="pin", task="reason", spec="nocolon", id="n")["code"] == "invalid_pin"
    assert _request(ws, type="routing", op="pin", task="nope", spec="openai:x", id="t")["code"] == "invalid"
    assert _request(ws, type="routing", op="pin", task="reason", id="s")["code"] == "invalid"
    cleared = _request(ws, type="routing", op="clear", task="reason", id="c")
    assert {k: cleared[k] for k in ("op", "task", "pin")} == {"op": "clear", "task": "reason", "pin": None}
    assert "reason" not in json.loads(routes_file.read_text()).get("pins", {})
    ws.close()


def test_routing_is_not_changed_from_elsewhere_or_under_a_turn(server, routes_file, monkeypatch):
    from agent.server import app as server_app

    gate = threading.Event()

    def fake_run(text, **kwargs):
        gate.wait(5)
        yield {"__final__": {"final_output": "ok"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="one"))
    assert _frame(ws)["event"]["type"] == "started"
    assert _request(ws, type="routing", op="clear", task="reason", id="b")["code"] == "busy"
    gate.set()
    _recv_until(ws, "event")
    monkeypatch.setattr(server_app, "_peer_is_loopback", lambda websocket: False)
    remote, _ = _hello(server)
    assert _request(remote, type="routing", op="pin", task="reason", spec="openai:gpt-5-mini", id="f")["code"] \
        == "forbidden"
    assert _request(remote, type="routing", op="list", id="l")["type"] == "routing_result"
    assert not routes_file.exists()
    ws.close()
    remote.close()


# --------------------------------------------------------------------------
# lessons and app notes (protocol 2)
# --------------------------------------------------------------------------

AMAZON = "in.amazon.mShop.android.shopping"


@pytest.fixture
def lesson_bank(tmp_path, monkeypatch):
    """A bank in tmp_path for every thread (bind_bank is a contextvar the
    server's threads cannot see), seeded with rows written straight to the
    store: two workspace lessons, one phone lesson, one Amazon note."""
    from agent.memory import lessons as L
    from agent.memory.store import MemoryStore

    monkeypatch.setattr(L, "DB_DIR", tmp_path / "bank")
    store = MemoryStore(L.bank_path())
    seeded = {}
    for kind, lesson in ((L.KIND, L.Lesson("a test fails on import", "check the path first")),
                         (L.KIND, L.Lesson("a build is slow", "cache the wheel", "failed")),
                         (L.PHONE_KIND, L.Lesson("a list will not scroll", "name it by its number")),
                         (L.APP_NOTE_PREFIX + AMAZON, L.Lesson("the results page", "sponsored items are marked ad"))):
        with L.bind_kind(kind):
            text = lesson.rendered()
            store.add_chunk(kind, L._hash(text), text)
            seeded.setdefault(kind, []).append(L._hash(text))
    store.close()
    return seeded


def test_lessons_are_listed_deleted_and_cleared_by_kind(server, lesson_bank):
    ws, _ = _hello(server)
    listed = _request(ws, type="lessons", op="list", kind="lesson", id="l")
    assert listed["type"] == "lessons_result" and listed["kind"] == "lesson"
    assert [row["lesson_id"] for row in listed["lessons"]] == lesson_bank["lesson"]
    assert listed["lessons"][1] == {"lesson_id": lesson_bank["lesson"][1], "cue": "a build is slow",
                                    "action": "cache the wheel", "outcome": "failed",
                                    "text": "When a build is slow: cache the wheel [failed]"}
    gone = lesson_bank["lesson"][0]
    assert _request(ws, type="lessons", op="delete", kind="lesson", lesson_id=gone, id="d") == {
        "type": "lessons_result", "id": "d", "op": "delete", "kind": "lesson", "lesson_id": gone, "deleted": True}
    assert _request(ws, type="lessons", op="delete", kind="lesson", lesson_id=gone, id="d2")["deleted"] is False
    assert len(_request(ws, type="lessons", op="list", kind="lesson", id="l2")["lessons"]) == 1
    assert _request(ws, type="lessons", op="clear", kind="phone_lesson", id="c") == {
        "type": "lessons_result", "id": "c", "op": "clear", "kind": "phone_lesson", "removed": 1}
    assert _request(ws, type="lessons", op="list", kind="phone_lesson", id="l3")["lessons"] == []
    for bad in ({"kind": "app_note:../x"}, {"kind": "nope"}, {"kind": "app_note:"}, {"kind": None},
                {"kind": "lesson", "op": "delete", "lesson_id": "abc"},
                {"kind": "lesson", "op": "delete", "lesson_id": gone.upper()}):
        reply = _request(ws, type="lessons", **{"op": "list", **bad, "id": "bad"})
        assert reply["type"] == "error" and reply["code"] == "invalid", bad
    assert _request(ws, type="lessons", op="list", kind="lesson", id="l4")["lessons"][0]["cue"] == "a build is slow"
    ws.close()


def test_app_notes_list_shipped_and_learned_and_only_learned_ones_go(server, lesson_bank):
    ws, _ = _hello(server)
    listed = {row["package"]: row for row in _request(ws, type="notes", op="list", id="n")["notes"]}
    assert listed[AMAZON] == {"package": AMAZON, "seeded": True, "learned": 1}
    assert listed["com.android.settings"]["seeded"] is True
    got = _request(ws, type="notes", op="get", package=AMAZON, id="g")
    assert got["type"] == "notes_result" and got["package"] == AMAZON and "Sort" in got["seeded"]
    note_id = lesson_bank["app_note:" + AMAZON][0]
    assert [row["lesson_id"] for row in got["learned"]] == [note_id]
    assert any("sponsored items are marked ad" in line for line in got["shown"])
    assert _request(ws, type="notes", op="delete", package=AMAZON, lesson_id=note_id, id="d") == {
        "type": "notes_result", "id": "d", "op": "delete", "package": AMAZON, "lesson_id": note_id, "deleted": True}
    assert _request(ws, type="notes", op="get", package=AMAZON, id="g2")["learned"] == []
    for bad in ({"op": "get", "package": "../x"}, {"op": "delete", "package": "a/b", "lesson_id": note_id},
                {"op": "delete", "package": AMAZON, "lesson_id": "x"}):
        assert _request(ws, type="notes", **bad, id="bad")["code"] == "invalid", bad
    ws.close()


def test_learned_things_are_not_deleted_under_a_running_turn(server, lesson_bank, monkeypatch):
    gate = threading.Event()

    def fake_run(text, **kwargs):
        gate.wait(5)
        yield {"__final__": {"final_output": "ok"}, "__trace_id__": None}

    monkeypatch.setattr(pipeline, "run_pipeline_stream", fake_run)
    ws, _ = _hello(server)
    ws.send(protocol.encode("turn", text="one"))
    assert _frame(ws)["event"]["type"] == "started"
    note_id = lesson_bank["app_note:" + AMAZON][0]
    assert _request(ws, type="lessons", op="clear", kind="lesson", id="c")["code"] == "busy"
    assert _request(ws, type="notes", op="delete", package=AMAZON, lesson_id=note_id, id="d")["code"] == "busy"
    assert _request(ws, type="lessons", op="list", kind="lesson", id="l")["type"] == "lessons_result"
    gate.set()
    _recv_until(ws, "event")
    ws.close()


def test_the_lesson_bank_and_the_app_notes_agree_on_what_a_package_is():
    from agent.memory import lessons as L
    from agent.phone import notes as N

    for package in (AMAZON, "com.android.settings", "a.b", "../x", "a/b", "x", "", "a..b", "1a.b", "a.b_c.D9"):
        assert L.valid_kind(L.APP_NOTE_PREFIX + package) == N.valid_package(package), package
    assert "in.amazon.mShop.android.shopping" in N.seeded_packages()


# --------------------------------------------------------------------------
# files: a research document from the session's workspace (protocol 2)
# --------------------------------------------------------------------------

def test_a_research_document_is_fetched_from_the_sessions_workspace_and_nowhere_else(server, monkeypatch, tmp_path):
    import base64
    import os
    import time

    sid = "d" * 32
    root = embed.session_workspace(sid)
    older = root / "otto_research" / "first-draft"
    newer = root / "otto_research" / "tides"
    for folder in (older, newer):
        folder.mkdir(parents=True)
        (folder / "document.md").write_text(f"# {folder.name}\n")
    (newer / "document.docx").write_bytes(b"PK\x03\x04docx")
    past = time.time() - 60
    os.utime(older / "document.md", (past, past))

    ws, _ = _hello(server)
    docx = _request(ws, type="files", session_id=sid, name="document.docx", id="f")
    assert {k: docx[k] for k in ("type", "id", "op", "session_id", "name", "path", "format", "mime", "size")} == {
        "type": "files_result", "id": "f", "op": "get", "session_id": sid, "name": "document.docx",
        "path": "otto_research/tides/document.docx", "format": "docx",
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document", "size": 8}
    assert base64.b64decode(docx["data"]) == b"PK\x03\x04docx"
    newest = _request(ws, type="files", session_id=sid, name="document.md", id="m")
    assert newest["path"] == "otto_research/tides/document.md"
    named = _request(ws, type="files", op="get", session_id=sid, name="otto_research/first-draft/document.md", id="n")
    assert base64.b64decode(named["data"]) == b"# first-draft\n"

    for bad in ("../document.md", "otto_research/../../document.md", "notes.txt", "otto_research/.x/document.md",
                "/etc/document.md", None):
        assert _request(ws, type="files", session_id=sid, name=bad, id="bad")["code"] == "invalid", bad
    assert _request(ws, type="files", session_id=sid, name="document.pdf", id="p")["code"] == "not_found"
    assert _request(ws, type="files", session_id="e" * 32, name="document.md", id="e")["code"] == "not_found"
    assert _request(ws, type="files", session_id="../x", name="document.md", id="s")["code"] == "invalid_session"
    assert _request(ws, type="files", op="put", session_id=sid, name="document.md", id="u")["code"] == "unknown"

    if os.name != "nt":
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "document.pdf").write_bytes(b"%PDF secret")
        (root / "otto_research" / "escape").symlink_to(outside, target_is_directory=True)
        for name in ("document.pdf", "otto_research/escape/document.pdf"):
            assert _request(ws, type="files", session_id=sid, name=name, id="l")["code"] == "not_found", name

    monkeypatch.setattr(embed, "FILE_MAX_BYTES", 4)
    too_big = _request(ws, type="files", session_id=sid, name="document.docx", id="t")
    assert too_big["type"] == "error" and too_big["code"] == "too_large"
    ws.close()
