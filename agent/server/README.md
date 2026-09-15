# agent/server/

`otto serve`: the agent behind a WebSocket, for a client that has the hands.

The phone helper's product runtime is Otto embedded in the app. This is the
same agent reached over a socket, for two cases: a developer's laptop while
the app is being worked on, and a Linux userland on the phone when the
embedded runtime is unavailable. Optional: `pip install "otto-cli-agent[serve]"`.

| module | what it is |
| --- | --- |
| `protocol.py` | the message types in both directions, `PROTOCOL_VERSION` and the oldest client still served |
| `proxy.py` | `SocketPhone`: a `PhoneBackend` whose phone is the connected client, one `device_call` per method, answered by a `device_result` |
| `app.py` | the server: a browser origin refused before the handshake unless `--allow-origin` names it, `hello` with a token compared in constant time, turns on the server's own bounded pool (`MAX_TURN_WORKERS`) through `agent/embed.py`, events forwarded as they happen, sessions listed and resumed |

```bash
uv run otto serve                 # loopback, port 8765, a token written to the env file
adb reverse tcp:8765 tcp:8765     # the phone reaches that loopback port over USB
uv run otto serve --no-exec       # no shell or Python for any turn
uv run otto serve --qr            # a pairing code, with `pip install qrcode`
uv run otto serve --host 0.0.0.0  # reachable on the network you are on -- read below first
```

**What the token grants.** A turn decides first whether it needs the phone
(agent/embed.py, one cheap call; `turn.phone` can say `on` or `off` instead).
A turn that does not runs the way `otto tui` does: in a workspace of its own
under `<OTTO_HOME>/workspaces/<session_id>`, with shell and Python on the
computer running `otto serve`. So whoever holds the pairing token can run
commands on that computer. Keep the server on `127.0.0.1` and connect the
phone with `adb reverse tcp:8765 tcp:8765`; `otto serve` warns when bound
anywhere wider. `--no-exec` takes the shell and Python away from every turn
(file tools stay inside the workspace).

A connection may hold at most eight sessions and runs one turn per session
at a time, which bounds what a client holding the token can spend. The
pairing string is `ws://host:port/#token`. A client says
`hello{protocol_version, token, capabilities: ["phone"]}`; with the `phone`
capability its turns get the phone tools (agent/phone/), and every tool call
becomes a `device_call` the client answers with the same JSON envelope the
in-app bridge uses.
