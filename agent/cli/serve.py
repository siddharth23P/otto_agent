"""`otto serve`: the agent behind a WebSocket for the phone app (agent/server/)."""
from __future__ import annotations

import asyncio
import os
import secrets
from typing import Annotated, Optional

import typer

from agent.cli.ui import err, out

TOKEN_ENV = "OTTO_SERVE_TOKEN"


def pairing_url(host: str, port: int) -> str:
    shown = "127.0.0.1" if host in ("", "0.0.0.0", "::") else host
    return f"ws://{shown}:{port}/"


def ensure_token(explicit: str | None) -> str:
    """The token from --token, else OTTO_SERVE_TOKEN, else a new one written
    to the env file so the app stays paired across restarts."""
    if explicit:
        return explicit
    existing = os.environ.get(TOKEN_ENV, "").strip()
    if existing:
        return existing
    from agent.config import envfile

    token = secrets.token_urlsafe(24)
    envfile.set_value(TOKEN_ENV, token)
    return token


def serve(
    host: Annotated[str, typer.Option(help="Interface to listen on; loopback unless you mean otherwise.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8765,
    token: Annotated[Optional[str], typer.Option(help=f"Shared secret the app presents; default {TOKEN_ENV} or a generated one.")] = None,
    qr: Annotated[bool, typer.Option("--qr", help="Print a pairing QR code (needs the qrcode package).")] = False,
    allow_origin: Annotated[Optional[list[str]], typer.Option(
        "--allow-origin", help="A browser origin allowed to connect (repeatable). Pages are refused otherwise.")] = None,
) -> None:
    """Serve the agent over a WebSocket for the phone app."""
    try:
        import websockets  # noqa: F401
    except ImportError:
        err.print("[bad]otto serve needs the websockets package: pip install 'otto-cli-agent[serve]'[/]")
        raise typer.Exit(2)
    from agent import embed
    from agent.config.envfile import ENV_PATH
    from agent.config.home import otto_home
    from agent.server.app import OttoServer

    embed.configure(otto_home(), env_file=ENV_PATH)
    secret = ensure_token(token)
    url = pairing_url(host, port)
    payload = f"{url}#{secret}"
    err.print(f"[muted]otto serve[/] listening on [bold]{url}[/]")
    err.print(f"[muted]pair the app with[/] {payload}")
    if qr:
        try:
            import qrcode

            code = qrcode.QRCode(border=1)
            code.add_data(payload)
            code.print_ascii(invert=True)
        except ImportError:
            err.print("[warn]--qr needs `pip install qrcode`; the URL above pairs the same way[/]")
    if not embed.ready():
        err.print("[warn]INCEPTION_API_KEY is not set; turns will fail until it is (otto tui -> Setup)[/]")
    try:
        asyncio.run(OttoServer(secret, allowed_origins=tuple(allow_origin or ())).run(host, port))
    except KeyboardInterrupt:
        out.print("[muted]bye[/]")
