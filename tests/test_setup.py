"""agent/router/setup.py: the data layer under the TUI's setup screen."""
from __future__ import annotations

import pytest

from agent.router import overrides, setup
from agent.router.llm_provider.base import HealthReport, ProviderStatus


@pytest.fixture
def routes_file(tmp_path):
    with overrides.bind_routes(tmp_path / "routes.json"):
        yield tmp_path / "routes.json"


def test_vendor_rows_mask_keys_and_list_custom_endpoints(routes_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-verysecret1234")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_API_KEY", "x")
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    overrides.save(overrides.Overrides(endpoints={"local": overrides.EndpointSpec("Local vLLM")}))

    rows = {r.name: r for r in setup.vendor_rows()}

    assert rows["openai"].masked_key == "********1234" and rows["openai"].key_present
    assert "verysecret" not in repr(rows["openai"])
    assert rows["gemini"].key_present is False and rows["gemini"].masked_key == "not set"
    assert rows["local"].custom and rows["local"].label == "Local vLLM"
    assert rows["local"].url_var == "LOCAL_BASE_URL" and not rows["local"].url_present
    assert rows["inception"].label.endswith("(required)")


def test_probe_never_raises(monkeypatch):
    class NoKey:
        env_var = "X_API_KEY"

        @classmethod
        def check(cls):
            return HealthReport("x", ProviderStatus.NO_KEY, detail="X_API_KEY is not set")

    monkeypatch.setattr(setup, "provider_class", lambda name: NoKey)
    result = setup.probe("x")
    assert result.report.status is ProviderStatus.NO_KEY and result.models == [] and not result.ok

    monkeypatch.setattr(setup, "provider_class", lambda name: (_ for _ in ()).throw(RuntimeError("boom")))
    assert setup.probe("y").report.status is ProviderStatus.ERROR


def test_set_base_url_validates_and_strips(routes_file, monkeypatch, tmp_path):
    from agent.config import envfile
    monkeypatch.setattr(envfile, "ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(setup, "reload_everything", lambda: [])
    with pytest.raises(ValueError):
        setup.set_base_url("local", "localhost:1234")
    monkeypatch.delenv("LOCAL_BASE_URL", raising=False)
    # envfile.set_value's default path is bound at def time; pass through the module's current value.
    monkeypatch.setattr(envfile, "set_value", lambda k, v, path=None: monkeypatch.setenv(k, v) or "***")
    setup.set_base_url("local", "http://localhost:1234/v1/")
    import os
    assert os.environ["LOCAL_BASE_URL"] == "http://localhost:1234/v1"
