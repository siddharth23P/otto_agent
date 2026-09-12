"""agent/router/reload.py: the one call after keys or pins change."""
from __future__ import annotations

from agent.router import llm_provider, overrides, reload as reload_mod
from agent.router.router import Router


def test_reload_runs_the_three_steps_in_order(monkeypatch):
    seen: list[str] = []
    monkeypatch.setattr(overrides, "apply", lambda *a, **k: seen.append("apply") or ["p"])
    monkeypatch.setattr(llm_provider, "reset", lambda: seen.append("reset"))
    monkeypatch.setattr(Router, "reset_all", classmethod(lambda cls: seen.append("routers")))

    assert reload_mod.reload_everything() == ["p"]
    assert seen == ["apply", "reset", "routers"]


def test_a_key_set_at_runtime_reaches_a_live_router(monkeypatch, tmp_path):
    from agent.config import envfile
    from agent.router.router import FakeCatalogue

    class EnvCatalogue(FakeCatalogue):
        def is_configured(self, provider):
            import os
            return provider == "inception" or bool(os.environ.get(f"{provider.upper()}_API_KEY"))

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    r = Router(catalogue=EnvCatalogue({"inception": [], "gemini": []}))
    assert "gemini" not in r.usable()
    with overrides.bind_routes(tmp_path / "routes.json"):
        envfile.set_value("GEMINI_API_KEY", "g-1234", path=tmp_path / ".env")
        reload_mod.reload_everything()
    assert "gemini" in r.usable()
