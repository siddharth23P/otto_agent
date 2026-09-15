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
    srv = OttoServer(TOKEN)
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
