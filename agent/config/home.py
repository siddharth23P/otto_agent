"""Where Otto keeps its per-installation state, and where its `.env` is.

Two facts that used to be constants: `~/.otto` for the memory stores, the
session index, the outcome log, routes and lessons (agent/memory/store.py,
agent/memory/sessions.py, agent/router/outcomes.py), and the repository
root's `.env` for keys (agent/config/envfile.py). Both are wrong the moment
Otto runs anywhere but a developer's shell. Embedded in an Android app there
is no `HOME` at all -- `Path.home()` raises -- and no repository, so an
installed wheel has nowhere to write a key to.

So they are functions of two environment variables, read at import time by
the modules that own the constants:

    OTTO_HOME       the directory for every per-installation file
    OTTO_ENV_FILE   the .env Otto loads at start and writes keys into

Read at import rather than at every call, deliberately: the constants they
feed (`store.DB_DIR`, `sessions.DEFAULT_INDEX_PATH`, `outcomes.DB_DIR`,
`envfile.ENV_PATH`) are what tests monkeypatch and what `bind_index`,
`bind_log` and `bind_store` override per test, and a value that moved
underneath those seams would make every one of them lie. The contract is
therefore: set the variables BEFORE `import agent.<anything>`, which is what
agent/embed.py's `configure()` does for an embedding host.
"""
from __future__ import annotations

import os
from pathlib import Path

HOME_ENV = "OTTO_HOME"
ENV_FILE_ENV = "OTTO_ENV_FILE"

#: The repository root when Otto runs from a checkout, else None. An
#: installed wheel's `agent/` sits in site-packages, which has no
#: pyproject.toml two levels up.
_CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def otto_home() -> Path:
    """`$OTTO_HOME`, else `~/.otto`.

    Raises RuntimeError naming the variable when neither can be resolved,
    which is the honest answer on a host with no home directory rather than a
    `Path.home()` traceback from inside an unrelated import.
    """
    raw = os.environ.get(HOME_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    try:
        return Path.home() / ".otto"
    except (RuntimeError, KeyError) as exc:
        raise RuntimeError(
            f"no home directory to put Otto's state in -- set {HOME_ENV} to a "
            "writable directory before importing agent"
        ) from exc


def running_from_checkout() -> bool:
    return (_CHECKOUT_ROOT / "pyproject.toml").is_file()


def env_file() -> Path:
    """`$OTTO_ENV_FILE`, else the checkout's `.env`, else `<otto_home>/.env`.

    The checkout case keeps `otto tui`'s setup screen writing the file the
    README tells a developer to create; an installed wheel has no such file
    and keeps keys beside its other state instead.
    """
    raw = os.environ.get(ENV_FILE_ENV, "").strip()
    if raw:
        return Path(raw).expanduser()
    if running_from_checkout():
        return _CHECKOUT_ROOT / ".env"
    return otto_home() / ".env"
