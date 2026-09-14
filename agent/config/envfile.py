"""Read and write the one `.env` otto loads (agent/cli/main.py).

Until the setup screen existed, keys got into otto by somebody opening this
file in an editor. This module is that edit, done by code: add or replace one
`KEY=value` line, leave every other line -- comments included -- exactly as it
was, and make the new value visible to the running process at the same time.

python-dotenv's `set_key`/`unset_key` do the file half (they rewrite through a
temporary file and `os.replace`, preserve mode, and replace in place rather
than appending a duplicate). What this module adds is the environment half
and the rule that no function here ever returns, logs or formats a secret --
`set_value` hands back the MASKED form, and that is the only thing a caller
needs to show.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import set_key, unset_key

from agent.config.home import env_file

#: The file agent/cli/main.py loads at import: the repository root's `.env`
#: from a checkout, `$OTTO_ENV_FILE` when set, `<OTTO_HOME>/.env` for an
#: installed wheel (agent/config/home.py). One definition, imported by
#: main.py, so the writer and the loader cannot drift apart.
ENV_PATH: Path = env_file()

KEY_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def masked(value: str | None) -> str:
    """The display form of a secret: eight stars and the last four characters,
    the same rule as `BaseProvider.masked_key`. "not set" for nothing."""
    if not value:
        return "not set"
    tail = value[-4:] if len(value) >= 4 else ""
    return f"{'*' * 8}{tail}"


def present(key: str) -> bool:
    """Whether the running process has a non-empty value for `key`."""
    return bool(os.environ.get(key))


def set_value(key: str, value: str, *, path: Path | None = None) -> str:
    """Write `KEY=value` to `path` and into `os.environ`. Returns the masked
    value, which is all a caller should ever display.

    An empty value means "remove it" -- a blank `KEY=` line would read as set
    to every `is_configured()` check that only asks for presence.
    """
    if not KEY_RE.match(key or ""):
        raise ValueError(f"{key!r} is not a valid environment variable name")
    value = (value or "").strip()
    if not value:
        unset_value(key, path=path)
        return masked(None)
    # Resolved at call time, not bound as a default: a default would freeze
    # the path at import, and both the tests and an embedding host that
    # reassigns ENV_PATH would silently write somewhere else.
    path = Path(ENV_PATH if path is None else path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
    set_key(str(path), key, value, quote_mode="auto")
    _private(path)
    os.environ[key] = value
    return masked(value)


def _private(path: Path) -> None:
    """Owner-only, on every write and not only on creation: the file may
    have been made by a host app or by hand with a wider mode, and since
    0.2.0 it also carries the `otto serve` pairing token. Windows has no
    such mode; a failure to set it is not a failure to write."""
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def unset_value(key: str, *, path: Path | None = None) -> None:
    """Remove `key` from `path` (if present) and from `os.environ`."""
    if not KEY_RE.match(key or ""):
        raise ValueError(f"{key!r} is not a valid environment variable name")
    path = Path(ENV_PATH if path is None else path)
    if path.exists():
        # unset_key logs a warning and returns (None, key) when the key is
        # absent; neither is an error for a caller that wants it gone.
        unset_key(str(path), key)
    os.environ.pop(key, None)
