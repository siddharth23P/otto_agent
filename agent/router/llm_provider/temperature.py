"""What temperature each model will actually honour.

One number cannot be right for every vendor, and a wrong one fails in three
different ways -- which is why this is a table rather than a constant:

  * Inception SILENTLY RESETS. Its documented range is 0.5-1.0, and anything
    outside it is not clamped and not rejected but reset to the model default
    of 1.0. So asking for 0.0 -- which this repo did for the evaluator, the
    router, the summarizer, the finder and the planner, from the day they were
    written -- got the most random setting available for exactly the nodes
    written to be the most careful. That bug was invisible because nothing
    errored.
  * OpenAI's reasoning models REJECT. `o4-mini` answers temperature=0.0 with
    `400 Unsupported value: 'temperature' does not support 0.0 with this
    model`, while `gpt-5-mini` and `gpt-4o-mini` accept it. Verified against
    the live API, not inferred from the family name.
  * Anthropic honours 0-1 on Opus 4.6, Sonnet 4.6, Haiku 4.5 and earlier,
    but from the Opus 4.7 generation on (Opus 4.7/4.8/5, Sonnet 5, Fable)
    any value except the default 1.0 is a 400: "`temperature` is deprecated
    for this model". Verified live. That flipped in the same generation
    that dropped `thinking: {type: "enabled"}`, and the Models API publishes
    THAT flag per model, so the published flag decides and the model name
    is only a fallback for a catalogue entry without a capabilities block.
  * Gemini simply honours what it is given, over 0-2.

So a route asks for the temperature it WANTS, and this decides what the model
can be given. Clamping beats dropping where a range exists, because the
closest honoured value preserves the caller's intent; dropping is only right
where the model admits no choice at all.

And for a model none of the tables know -- a new vendor, a custom endpoint, a
release newer than this file -- the answer is learned rather than guessed.
The first call is made with the temperature the route asked for. If the model
refuses it (`looks_like_temperature_rejection`), the call wrapper in
agent/pipeline/nodes.py retries without one and records the refusal here
(`note_rejects_temperature`), in ~/.otto/temperature.json, so every later
call in this run and every future run leaves the parameter out. Persisted,
unlike retired.py's learned set, because the two failures differ in kind: a
vendor outage misread as a retirement would hide a live model forever, but a
refused temperature is deterministic -- the same request gets the same 400
tomorrow -- and the worst a wrong entry can do is fall back to the model's
default temperature.
"""
from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

from agent.router.outcomes import DB_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemperaturePolicy:
    """How one model treats `temperature`.

    `low`/`high` bound what it honours. `fixed` means the model accepts only
    its own default, so the parameter must be dropped entirely rather than
    clamped -- sending any value is a 400.
    """
    low: float = 0.0
    high: float = 2.0
    fixed: bool = False

    def apply(self, temperature: float) -> float | None:
        """The value to send, or None to send nothing at all."""
        if self.fixed:
            return None
        return min(max(temperature, self.low), self.high)


#: Models that accept no temperature but their own. Matched by pattern because
#: the family grows faster than this table can: OpenAI's reasoning line (o1,
#: o3, o4, ...) all behave this way, and so does every Claude model from the
#: Opus 4.7 generation on. For Anthropic this is only the fallback -- the
#: published capability flag (`_PUBLISHED_FIXED_SIGNALS`) is consulted first.
_FIXED_TEMPERATURE_PATTERNS = {
    "openai": (re.compile(r"^o\d"),),
    "anthropic": (
        re.compile(r"^claude-(?:opus-4-(?:[7-9]|\d{2,})|opus-[5-9]|sonnet-[5-9]|fable|mythos)"),
    ),
}

#: Per-provider defaults. A provider absent here is left alone, which is the
#: honest default for a vendor whose behaviour nobody has measured.
_BY_PROVIDER: dict[str, TemperaturePolicy] = {
    # Documented 0.5-1.0, and out-of-range is reset to 1.0 rather than clamped.
    "inception": TemperaturePolicy(low=0.5, high=1.0),
    "anthropic": TemperaturePolicy(low=0.0, high=1.0),
    "openai": TemperaturePolicy(low=0.0, high=2.0),
    "gemini": TemperaturePolicy(low=0.0, high=2.0),
}

#: Overrides for a specific id, when it differs from its provider's norm.
_BY_MODEL: dict[tuple[str, str], TemperaturePolicy] = {}


def register_provider(
    name: str,
    policy: TemperaturePolicy = TemperaturePolicy(low=0.0, high=2.0),
    fixed_patterns: tuple[re.Pattern, ...] = (re.compile(r"^(?:.*/)?o\d"),),
) -> None:
    """Give a custom OpenAI-compatible endpoint a temperature policy.

    OpenAI's honoured range by default, plus OpenAI's own "the o-series takes
    no temperature" rule -- prefix-aware, because an aggregator such as
    OpenRouter serves those models as `openai/o3`. Nothing here has been
    measured against a specific server; a person who knows better edits the
    tables the same way they would for a built-in.
    """
    _BY_PROVIDER[name] = policy
    _FIXED_TEMPERATURE_PATTERNS[name] = tuple(fixed_patterns)


#: Field names a vendor may use to publish a model's own ceiling. Gemini
#: reports `max_temperature` per model and it is NOT uniform -- most cap at 2,
#: but several cap at 1, which a per-provider guess of 0-2 would overshoot.
_MAX_TEMPERATURE_FIELDS = ("max_temperature", "maxTemperature")


def published_maximum(model) -> float | None:
    """The model's own temperature ceiling, if its vendor publishes one.

    Preferred over anything in this file's tables: it comes from the vendor,
    per model, and updates itself when they change it. `ModelInfo.raw` is
    whatever the SDK returned, so it may be a dict or an object.
    """
    raw = getattr(model, "raw", None)
    if raw is None:
        return None
    for field in _MAX_TEMPERATURE_FIELDS:
        value = raw.get(field) if isinstance(raw, dict) else getattr(raw, field, None)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return None


def _lookup(raw, *path):
    """Walk `path` through nested dicts or objects; None if any step is missing."""
    node = raw
    for key in path:
        if node is None:
            return None
        node = node.get(key) if isinstance(node, dict) else getattr(node, key, None)
    return node


def _anthropic_fixed(raw) -> bool | None:
    """Whether a Claude model refuses `temperature`, read off its published flags.

    The Models API publishes no sampling flag. It does publish whether the
    model still takes `thinking: {type: "enabled"}`, and Anthropic removed
    that and `temperature` in the same generation (Opus 4.7): every model
    that reports `enabled: {supported: false}` rejects temperature, every
    model that reports `supported: true` honours it. A proxy, but a
    per-model one that the vendor maintains -- a new Claude model needs no
    edit here. None when the payload has no such flag.
    """
    supported = _lookup(raw, "capabilities", "thinking", "types", "enabled", "supported")
    if isinstance(supported, bool):
        return not supported
    return None


#: Per-provider readers of the vendor's own "takes no temperature" signal.
#: Preferred over `_FIXED_TEMPERATURE_PATTERNS`, for the same reason
#: `published_maximum` beats `_BY_PROVIDER`: per model, and self-maintaining.
_PUBLISHED_FIXED_SIGNALS = {
    "anthropic": _anthropic_fixed,
}


def published_fixed(provider: str, model) -> bool | None:
    """Whether the vendor says this model admits no temperature but its own.

    True or False when the vendor publishes a signal for it; None when it
    publishes nothing, so the name patterns get to decide.
    """
    reader = _PUBLISHED_FIXED_SIGNALS.get(provider)
    raw = getattr(model, "raw", None)
    if reader is None or raw is None:
        return None
    return reader(raw)


# ---- learned at call time ---------------------------------------------------

#: Where a refusal is remembered between runs. Beside the other
#: per-installation state (agent/router/outcomes.py's DB_DIR), and readable:
#: each entry carries when and what the vendor said, so a person can tell a
#: real refusal from a one-off and delete the line if they disagree.
STORE_PATH: Path = DB_DIR / "temperature.json"

_DEFAULT_STORE = object()
#: `bind_store` target: the sentinel means STORE_PATH, None means memory only.
_STORE: ContextVar[object] = ContextVar("otto_temperature_store", default=_DEFAULT_STORE)

#: provider -> {model id: "<date>: <what the vendor said>"}, one per store
#: file, loaded on first use. The memory-only store (bind_store(None)) is the
#: entry under None.
_learned_by_store: dict[Path | None, dict[str, dict[str, str]]] = {}


@contextmanager
def bind_store(path: Path | str | None) -> Iterator[None]:
    """Remember refusals somewhere else, or (None) for this process only.

    Tests bind a fresh temporary file so a suite can neither read what the
    developer's machine learned nor teach it something -- the same rule
    tests/conftest.py applies to the lesson bank and the outcome log.
    """
    token = _STORE.set(None if path is None else Path(path))
    try:
        yield
    finally:
        _STORE.reset(token)


def _store_path() -> Path | None:
    bound = _STORE.get()
    return STORE_PATH if bound is _DEFAULT_STORE else bound  # type: ignore[return-value]


def _load(path: Path) -> dict[str, dict[str, str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("%s is unreadable (%s) -- starting with nothing learned", path, exc)
        return {}
    entries = data.get("rejects_temperature") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return {}
    return {
        provider: {str(m): str(why) for m, why in models.items()}
        for provider, models in entries.items()
        if isinstance(models, dict)
    }


def _save(path: Path, learned: dict[str, dict[str, str]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(
            json.dumps({"version": 1, "rejects_temperature": learned}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        tmp.replace(path)
    except OSError as exc:
        logger.warning("could not write %s (%s) -- learned for this process only", path, exc)


def _learned() -> dict[str, dict[str, str]]:
    path = _store_path()
    if path not in _learned_by_store:
        _learned_by_store[path] = _load(path) if path is not None else {}
    return _learned_by_store[path]


def rejects_temperature(provider: str, model_id: str) -> bool:
    """True if a call to this model has already been refused for its temperature."""
    return model_id in _learned().get(provider, {})


def note_rejects_temperature(provider: str, model_id: str, reason: str) -> None:
    """Record that `provider:model_id` refuses `temperature`.

    Called from the failure path once the vendor has said so. Takes effect
    for every later `policy_for` in this process and, through the store, in
    every future one.
    """
    learned = _learned()
    if model_id in learned.get(provider, {}):
        return
    learned.setdefault(provider, {})[model_id] = f"{date.today().isoformat()}: {reason[:300]}"
    path = _store_path()
    if path is not None:
        _save(path, learned)
    logger.warning(
        "%s:%s refuses temperature (%s) -- it will be called without one from now on. "
        "Remembered in %s; delete the entry there to re-test it.",
        provider, model_id, reason, path or "this process only",
    )


#: HTTP statuses a vendor uses for "your request is malformed". A refusal of
#: `temperature` is one of these; anything else that happens to mention the
#: word (a 429 quoting the request back, say) is not.
_REJECTION_STATUSES = (400, 422)


def looks_like_temperature_rejection(exc: Exception) -> bool:
    """True if `exc` is a model refusing the `temperature` it was sent.

    Anthropic: `400 temperature is deprecated for this model`. OpenAI's
    o-series: `400 Unsupported value: 'temperature' does not support 0.0`.
    langchain_anthropic raises a bare ValueError for a model it knows takes
    none, before any request goes out. Duck-typed on the status the same way
    agent/router/llm_provider/base.py's translate_unknown reads it, so this
    stays free of vendor SDK imports.

    A range complaint ("must be between 0 and 1") is caught too. That is a
    slightly blunt reading -- clamping would preserve more intent -- but for
    a model no table knows, calling it at its default temperature beats
    failing every call, and the recorded reason tells a person which range
    to add.
    """
    if "temperature" not in str(exc).lower():
        return False
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if isinstance(status, int):
        return status in _REJECTION_STATUSES
    return True


def policy_for(provider: str, model_id: str, model=None) -> TemperaturePolicy | None:
    """The policy for one model, or None if nothing here knows this provider.

    Order of authority: an explicit per-model entry, then a refusal learned
    from the model itself, then a model that admits no choice at all -- by
    the vendor's published flag where there is one, by name otherwise --
    then the vendor's OWN published ceiling for this model, and only then a
    per-provider default. The published figures beat the tables because they
    are per model and maintain themselves; a learned refusal beats both
    because it is what this exact model actually said.
    """
    if (provider, model_id) in _BY_MODEL:
        return _BY_MODEL[(provider, model_id)]
    if rejects_temperature(provider, model_id):
        return TemperaturePolicy(fixed=True)
    fixed = published_fixed(provider, model)
    if fixed is None:
        fixed = any(p.match(model_id) for p in _FIXED_TEMPERATURE_PATTERNS.get(provider, ()))
    if fixed:
        return TemperaturePolicy(fixed=True)
    default = _BY_PROVIDER.get(provider)
    ceiling = published_maximum(model)
    if ceiling is not None:
        return TemperaturePolicy(low=default.low if default else 0.0, high=ceiling)
    return default


def apply_to_params(provider: str, model_id: str, params: dict, model=None) -> dict:
    """`params` with `temperature` adjusted to what this model honours.

    Returns a new dict; drops the key entirely for a fixed-temperature model.
    Pass `model` (a ModelInfo) so the vendor's own published ceiling can be
    used in preference to this file's per-provider defaults.
    """
    if "temperature" not in params:
        return params
    policy = policy_for(provider, model_id, model)
    if policy is None:
        return params
    adjusted = policy.apply(params["temperature"])
    updated = dict(params)
    if adjusted is None:
        del updated["temperature"]
    else:
        updated["temperature"] = adjusted
    return updated
