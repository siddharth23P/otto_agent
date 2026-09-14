"""agent/config/home.py: every per-installation path follows OTTO_HOME, and
the .env follows OTTO_ENV_FILE.

The constants are read at import, so the real test is a subprocess that sets
the variables and THEN imports -- an in-process monkeypatch of the variables
would prove nothing about modules already imported by the suite."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent.config import home


def test_otto_home_follows_the_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(home.HOME_ENV, str(tmp_path / "state"))
    assert home.otto_home() == tmp_path / "state"


def test_otto_home_defaults_under_the_home_directory(monkeypatch):
    monkeypatch.delenv(home.HOME_ENV, raising=False)
    assert home.otto_home() == Path.home() / ".otto"


def test_env_file_follows_its_own_variable(monkeypatch, tmp_path):
    monkeypatch.setenv(home.ENV_FILE_ENV, str(tmp_path / "keys.env"))
    assert home.env_file() == tmp_path / "keys.env"


def test_env_file_is_the_checkout_dotenv_when_running_from_one(monkeypatch):
    monkeypatch.delenv(home.ENV_FILE_ENV, raising=False)
    assert home.running_from_checkout()
    assert home.env_file() == Path(home.__file__).resolve().parents[2] / ".env"


def test_env_file_falls_back_beside_the_state_outside_a_checkout(monkeypatch, tmp_path):
    monkeypatch.delenv(home.ENV_FILE_ENV, raising=False)
    monkeypatch.setenv(home.HOME_ENV, str(tmp_path))
    monkeypatch.setattr(home, "running_from_checkout", lambda: False)
    assert home.env_file() == tmp_path / ".env"


def test_no_home_directory_is_a_named_error(monkeypatch):
    monkeypatch.delenv(home.HOME_ENV, raising=False)

    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(home.Path, "home", staticmethod(no_home))
    try:
        home.otto_home()
    except RuntimeError as exc:
        assert home.HOME_ENV in str(exc)
    else:
        raise AssertionError("expected a RuntimeError naming OTTO_HOME")


_PROBE = r"""
import json
from agent.config import envfile
from agent.memory import store, sessions, lessons
from agent.router import outcomes, overrides
from agent.router.llm_provider import temperature
from agent.cli import output
print(json.dumps({
    "store": str(store.DB_DIR),
    "index": str(sessions.DEFAULT_INDEX_PATH),
    "outcomes": str(outcomes.DB_DIR),
    "routes": str(overrides.routes_path()),
    "temperature": str(temperature.STORE_PATH),
    "lessons": str(lessons.bank_path()),
    "env": str(envfile.ENV_PATH),
    "output": str(output.OUTPUT_DIR),
}))
"""


def test_every_constant_resolves_under_otto_home_when_set_before_import(tmp_path):
    state = tmp_path / "state"
    env = {**os.environ, home.HOME_ENV: str(state), home.ENV_FILE_ENV: str(tmp_path / "k.env"),
           "OTTO_OUTPUT_DIR": str(tmp_path / "out")}
    env.pop("OTTO_ROUTES", None)
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True, text=True, env=env,
                          cwd=str(Path(home.__file__).resolve().parents[2]), timeout=120)
    assert proc.returncode == 0, proc.stderr
    paths = json.loads(proc.stdout.strip().splitlines()[-1])
    for key in ("store", "index", "outcomes", "routes", "temperature", "lessons"):
        assert Path(paths[key]).is_relative_to(state), (key, paths[key])
    assert paths["env"] == str(tmp_path / "k.env")
    assert paths["output"] == str(tmp_path / "out")


def test_set_value_writes_the_current_env_path_not_the_import_time_one(monkeypatch, tmp_path):
    """`otto tui`'s setup screen calls set_value with no path; a default bound
    at import would send its writes to wherever ENV_PATH pointed when the
    suite started."""
    from agent.config import envfile

    target = tmp_path / ".env"
    monkeypatch.setattr(envfile, "ENV_PATH", target)
    monkeypatch.delenv("OTTO_TEST_KEY", raising=False)
    envfile.set_value("OTTO_TEST_KEY", "abcd1234")
    assert "OTTO_TEST_KEY=" in target.read_text()
    envfile.unset_value("OTTO_TEST_KEY")
    assert "OTTO_TEST_KEY=" not in target.read_text()


def test_set_value_makes_the_file_private_even_when_it_already_was_not(monkeypatch, tmp_path):
    """The file now carries the serve token too; a copy made by a host or
    by hand with a wide mode is tightened on every write, not only when
    otto creates it (2026-09-14 review)."""
    import os
    import stat

    from agent.config import envfile

    if os.name == "nt":
        import pytest

        pytest.skip("POSIX file modes")
    target = tmp_path / ".env"
    target.write_text("OTHER=1\n")
    target.chmod(0o644)
    monkeypatch.delenv("OTTO_TEST_KEY", raising=False)
    envfile.set_value("OTTO_TEST_KEY", "abcd1234", path=target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    envfile.unset_value("OTTO_TEST_KEY", path=target)
