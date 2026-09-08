"""Turn provider-layer exceptions into CLI errors.

The router raises `AuthError` from `Router.__init__` (3.2) and
`NoViableRoute` / `RoutingDegraded` from `resolve()`. Those are the right
exceptions for a library, but a CLI must never answer a user with a traceback:
it buries the diagnosis and trains people to ignore stack traces.

`friendly` is applied once per command at registration, so a new command can
never forget it.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import typer

from agent.cli.ui import err
from agent.router.llm_provider.base import AuthError, ProviderError
from agent.router.router import NoViableRoute, RoutingDegraded

F = TypeVar("F", bound=Callable[..., Any])

#: Exit codes, so a shell script can tell these apart.
EXIT_NO_KEY = 2
EXIT_NO_ROUTE = 3
EXIT_DEGRADED = 4
EXIT_PROVIDER = 5


def friendly(command: F) -> F:
    """Render expected failures as CLI errors instead of tracebacks."""

    @functools.wraps(command)          # keeps the signature Typer introspects
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return command(*args, **kwargs)
        except AuthError as exc:
            err.print(f"[bad]{exc}[/]")
            err.print("[muted]run `otto doctor` to see which providers are configured[/]")
            raise typer.Exit(EXIT_NO_KEY) from None
        except NoViableRoute as exc:
            err.print(f"[bad]{exc}[/]")
            raise typer.Exit(EXIT_NO_ROUTE) from None
        except RoutingDegraded as exc:
            err.print(f"[warn]{exc}[/]")
            err.print("[muted]drop --strict to allow the fallback[/]")
            raise typer.Exit(EXIT_DEGRADED) from None
        except ProviderError as exc:
            # Anything else the provider layer defines: still ours, still not
            # a traceback the user should have to read.
            err.print(f"[bad]{type(exc).__name__}: {exc}[/]")
            raise typer.Exit(EXIT_PROVIDER) from None

    return wrapper  # type: ignore[return-value]
