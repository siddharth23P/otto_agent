"""`otto serve`: Otto behind a WebSocket, for a client that has the hands.

The phone helper's primary runtime is Otto inside the app. This is the
other two ways the same app reaches the same agent: a developer's laptop
during app work (the app pairs to `otto serve` over Wi-Fi), and a Linux
userland on the phone itself when the embedded runtime is unavailable. One
server, one protocol (agent/server/protocol.py), one proxy that turns the
client's phone into a PhoneBackend (agent/server/proxy.py).
"""
