"""What the setup screen reads and writes -- the data layer under the wizard.

Every function here is something the TUI's SetupScreen calls; nothing in the
screen touches `os.environ`, the `.env` file or the registry directly. That is
what makes the screen testable against fakes and this module testable without
a screen.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from agent.config import envfile
from agent.router import overrides
from agent.router.llm_provider import (
    all_models,
    builtin_provider_names,
    get_provider,
    provider_class,
)
from agent.router.llm_provider import custom
from agent.router.llm_provider.base import HealthReport, ModelInfo, ProviderError, ProviderStatus
from agent.router.mapping import Candidate, Task
from agent.router.overrides import PROVIDER_ONLY
from agent.router.reload import reload_everything

LABELS = {
    "inception": "Inception (required)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Google Gemini",
}


@dataclass(frozen=True)
class VendorRow:
    name: str
    label: str
    key_var: str
    url_var: str | None
    key_present: bool
    url_present: bool
    custom: bool
    masked_key: str


@dataclass(frozen=True)
class ProbeResult:
    report: HealthReport
    models: list[ModelInfo]

    @property
    def ok(self) -> bool:
        return self.report.ok


def vendor_rows() -> list[VendorRow]:
    rows: list[VendorRow] = []
    for name in builtin_provider_names():
        var = provider_class(name).env_var
        rows.append(VendorRow(
            name=name, label=LABELS.get(name, name), key_var=var, url_var=None,
            key_present=envfile.present(var), url_present=True, custom=False,
            masked_key=envfile.masked(os.environ.get(var)),
        ))
    for name, spec in sorted(overrides.endpoints().items()):
        key, url = custom.key_var(name), custom.url_var(name)
        rows.append(VendorRow(
            name=name, label=spec.label or name, key_var=key, url_var=url,
            key_present=envfile.present(key), url_present=envfile.present(url), custom=True,
            masked_key=envfile.masked(os.environ.get(key)),
        ))
    return rows


def probe(name: str) -> ProbeResult:
    """A real call. Never raises -- an unreachable vendor is a row, not a crash."""
    try:
        report = provider_class(name).check()
    except Exception as exc:  # unknown name, import failure
        return ProbeResult(HealthReport(name, ProviderStatus.ERROR, detail=str(exc)), [])
    if not report.ok:
        return ProbeResult(report, [])
    try:
        return ProbeResult(report, get_provider(name).list_models())
    except ProviderError as exc:
        return ProbeResult(HealthReport(name, ProviderStatus.ERROR, detail=str(exc)), [])


def probe_all() -> dict[str, ProbeResult]:
    return {row.name: probe(row.name) for row in vendor_rows()}


def set_key(name: str, value: str) -> str:
    """Write `<VAR>_API_KEY` for `name` (built-in or custom) and reload.
    Returns the masked value, which is all the caller should show."""
    var = provider_class(name).env_var if name in builtin_provider_names() else custom.key_var(name)
    shown = envfile.set_value(var, value)
    reload_everything()
    return shown


def set_base_url(name: str, url: str) -> None:
    custom.validate_name(name)
    url = (url or "").strip().rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("base URL must start with http:// or https://")
    envfile.set_value(custom.url_var(name), url)
    reload_everything()


def add_endpoint(name: str, label: str = "") -> None:
    overrides.add_endpoint(name, label)
    reload_everything()


def remove_endpoint(name: str) -> None:
    overrides.remove_endpoint(name)
    envfile.unset_value(custom.key_var(name))
    envfile.unset_value(custom.url_var(name))
    reload_everything()


#: The "no pin" choice, first in every task's options.
NO_PIN = ("(no pin — default route)", "")


def pin_options(task: Task, models: list[ModelInfo],
                routes: Mapping[Task, tuple[Candidate, ...]]) -> list[tuple[str, str]]:
    """The choices for one task's pin: no pin, then every model the seat
    could use -- the head candidate's `requires`, narrowed to the provider a
    task is bound to (agent/router/overrides.py PROVIDER_ONLY). Here rather
    than in the TUI's screen so `otto serve` offers the phone app the same
    list (agent/cli/setup_screen.py re-exports it)."""
    head = routes[task][0]
    bound = PROVIDER_ONLY.get(task)
    eligible = [
        m for m in models
        if head.requires <= m.capabilities and (bound is None or m.provider == bound[0])
    ]
    eligible.sort(key=lambda m: (m.provider, m.id))
    return [NO_PIN] + [(m.spec, m.spec) for m in eligible]


def detected_pool() -> list[ModelInfo]:
    return sorted(all_models(None), key=lambda m: (m.provider, m.id))
