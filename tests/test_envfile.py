"""The .env writer: a code path that edits a file a person also edits by
hand, so what it must not do -- clobber comments, duplicate lines, leak the
value -- matters as much as what it does."""
from __future__ import annotations

import os

import pytest
from dotenv import dotenv_values

from agent.config import envfile
from tests.conftest import posix_filesystem


@pytest.fixture
def env_path(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# keys for otto\n"
        "INCEPTION_API_KEY=inc-1234\n"
        "\n"
        "# tracing\n"
        "LANGFUSE_BASE_URL=https://cloud.langfuse.com\n"
    )
    return p


def test_adding_a_key_keeps_every_other_line(env_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    shown = envfile.set_value("OPENAI_API_KEY", "sk-verysecret9876", path=env_path)

    text = env_path.read_text()
    assert "# keys for otto" in text and "# tracing" in text
    assert text.count("INCEPTION_API_KEY=") == 1
    assert dotenv_values(env_path)["OPENAI_API_KEY"] == "sk-verysecret9876"
    assert os.environ["OPENAI_API_KEY"] == "sk-verysecret9876"
    assert shown == "********9876"
    assert "verysecret" not in shown


def test_replacing_a_key_edits_it_in_place(env_path, monkeypatch):
    monkeypatch.setenv("INCEPTION_API_KEY", "inc-1234")
    envfile.set_value("INCEPTION_API_KEY", "inc-5678", path=env_path)

    lines = env_path.read_text().splitlines()
    assert lines.count("INCEPTION_API_KEY='inc-5678'") + lines.count("INCEPTION_API_KEY=inc-5678") == 1
    assert dotenv_values(env_path)["INCEPTION_API_KEY"] == "inc-5678"
    assert lines[0] == "# keys for otto", "order preserved"
    assert os.environ["INCEPTION_API_KEY"] == "inc-5678"


def test_a_value_with_spaces_survives_the_round_trip(env_path, monkeypatch):
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    envfile.set_value("LOCAL_BASE_URL", "http://host:1234/v1 # not a comment", path=env_path)
    assert dotenv_values(env_path)["LOCAL_BASE_URL"] == "http://host:1234/v1 # not a comment"


def test_unsetting_removes_the_row_and_the_variable(env_path, monkeypatch):
    monkeypatch.setenv("INCEPTION_API_KEY", "inc-1234")
    envfile.unset_value("INCEPTION_API_KEY", path=env_path)
    assert "INCEPTION_API_KEY" not in dotenv_values(env_path)
    assert "INCEPTION_API_KEY" not in os.environ
    assert "# tracing" in env_path.read_text()


def test_an_empty_value_means_unset(env_path, monkeypatch):
    monkeypatch.setenv("INCEPTION_API_KEY", "inc-1234")
    assert envfile.set_value("INCEPTION_API_KEY", "   ", path=env_path) == "not set"
    assert "INCEPTION_API_KEY" not in os.environ


def test_unsetting_a_missing_key_is_not_an_error(env_path, monkeypatch):
    monkeypatch.delenv("NOPE_API_KEY", raising=False)
    envfile.unset_value("NOPE_API_KEY", path=env_path)
    envfile.unset_value("NOPE_API_KEY", path=env_path.parent / "absent.env")


@posix_filesystem
def test_a_missing_file_is_created_private(tmp_path, monkeypatch):
    monkeypatch.delenv("X_API_KEY", raising=False)
    target = tmp_path / "sub" / ".env"
    envfile.set_value("X_API_KEY", "abcd", path=target)
    assert target.exists()
    assert (target.stat().st_mode & 0o777) == 0o600
    assert dotenv_values(target)["X_API_KEY"] == "abcd"


@pytest.mark.parametrize("bad", ["lower_case", "1STARTS", "HAS SPACE", "", "DASH-KEY"])
def test_a_bad_key_name_is_rejected_before_touching_anything(env_path, bad):
    before = env_path.read_text()
    with pytest.raises(ValueError):
        envfile.set_value(bad, "x", path=env_path)
    assert env_path.read_text() == before


def test_masked_never_shows_more_than_the_tail():
    assert envfile.masked("sk-verysecret9876") == "********9876"
    assert envfile.masked("abc") == "********"
    assert envfile.masked(None) == "not set"
    assert envfile.masked("") == "not set"
