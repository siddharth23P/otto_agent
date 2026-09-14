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
uv run otto serve --host 0.0.0.0  # reachable on the network you are on
uv run otto serve --qr            # a pairing code, with `pip install qrcode`
```

A connection may hold at most eight sessions and runs one turn per session
at a time, which bounds what a client holding the token can spend. The
pairing string is `ws://host:port/#token`. A client says
`hello{protocol_version, token, capabilities: ["phone"]}`; with the `phone`
capability its turns get the phone tools (agent/phone/), and every tool call
becomes a `device_call` the client answers with the same JSON envelope the
in-app bridge uses.
