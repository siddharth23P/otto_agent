"""What a person has pinned on top of the shipped routing table, and where.

`agent/router/mapping.py`'s TASK_ROUTES is a measured, hand-tuned table in
source. It cannot know which keys THIS machine has or which model its owner
wants judging their work, and editing Python to say so is not a setting. So
this module keeps a small file beside the other per-installation state:

    ~/.otto/routes.json
    {
      "version": 1,
      "pins":      {"reason": "openai:gpt-5-mini", "chat_fast": "ollama:llama3.2:latest"},
      "endpoints": {"ollama": {"label": "Ollama",
                               "capabilities": {"llama3.2:latest": ["chat", "tools"]}}}
    }

`apply()` turns that into the live table, IN PLACE: each custom endpoint is
registered as a provider (agent/router/llm_provider/custom.py) and each pin
becomes a Candidate at the HEAD of that task's chain, with the shipped chain
kept behind it as the fallback path. In place, because two Routers read
TASK_ROUTES at resolve time -- the pipeline's module-level one and the CLI's
-- and a new dict would reach neither.

What a pin does not do: it does not bypass the endpoint. A pin on VISION still
requires the model to see, a pin on CODE_COMPLETE still has to be an Inception
FIM model, and WEB stays Anthropic because the web_search tool binds
Anthropic's own server-side search (agent/pipeline/tools.py). `validate_pin`
says so up front rather than letting a pin resolve cleanly and fail mid-run.

Keys are NOT here. They stay in the repo `.env` (agent/config/envfile.py),
which the four built-in vendors already read; a custom endpoint's
`<NAME>_API_KEY` and `<NAME>_BASE_URL` rows go beside them. This file holds
names and choices, nothing secret, and can be copied between machines as is.
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from agent.router import mapping
from agent.router.llm_provider import custom, is_custom, register_custom, unregister_custom
from agent.router.llm_provider import temperature
from agent.router.llm_provider.base import Capability
from agent.router.mapping import (
    ENDPOINT_CAPABILITY,
    INCEPTION_ONLY_ENDPOINTS,
    PARAMS_BY_PROVIDER,
    TASK_ROUTES,
    Candidate,
    MappingError,
    Task,
)
from agent.router.outcomes import DB_DIR

log = logging.getLogger(__name__)

VERSION = 1
#: Set to skip the file entirely -- an eval harness that must measure the
#: shipped table, not this machine's preferences.
IGNORE_ENV = "OTTO_IGNORE_ROUTES"
#: Point at a different file (tests; a second profile).
PATH_ENV = "OTTO_ROUTES"

#: Which provider a task is bound to regardless of pins, and why.
PROVIDER_ONLY: dict[Task, tuple[str, str]] = {
    Task.WEB: ("anthropic", "web_search binds Anthropic's server-side search tool"),
    Task.CODE_COMPLETE: ("inception", "only Inception serves the FIM endpoint"),
    Task.CODE_EDIT: ("inception", "only Inception serves the edit endpoint"),
}


class PinError(ValueError):
    """A pin that can be seen to be wrong without asking a provider."""


@dataclass(frozen=True)
class EndpointSpec:
    label: str = ""
    #: model id -> capability names, overriding the heuristic classifier.
    capabilities: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class Overrides:
    pins: dict[Task, str] = field(default_factory=dict)
    endpoints: dict[str, EndpointSpec] = field(default_factory=dict)
    version: int = VERSION


# --------------------------------------------------------------------------
# Where the file is
# --------------------------------------------------------------------------

_UNSET = object()
_PATH: ContextVar = ContextVar("otto_routes_path", default=_UNSET)


def routes_path() -> Path:
    bound = _PATH.get()
    if bound is not _UNSET:
        return Path(bound)
    env = os.environ.get(PATH_ENV, "").strip()
    return Path(env).expanduser() if env else DB_DIR / "routes.json"


@contextmanager
def bind_routes(path: Path | str) -> Iterator[Path]:
    """Use `path` for the duration -- a test's tmp_path. Same seam shape as
    agent/router/outcomes.py's bind_log."""
    token = _PATH.set(Path(path))
    try:
        yield Path(path)
    finally:
        _PATH.reset(token)


# --------------------------------------------------------------------------
# Load / save
# --------------------------------------------------------------------------

def load(path: Path | None = None) -> Overrides:
    """The file's content, or an empty Overrides. Never raises: a corrupt
    preferences file must not stop otto from starting -- it is logged and
    the shipped table is used."""
    path = path or routes_path()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return Overrides()
    except (OSError, ValueError) as exc:
        log.warning("ignoring %s: %s", path, exc)
        return Overrides()
    if not isinstance(raw, dict) or raw.get("version") != VERSION:
        log.warning("ignoring %s: unsupported format", path)
        return Overrides()

    pins: dict[Task, str] = {}
    for key, spec in (raw.get("pins") or {}).items():
        try:
            task = Task(key)
        except ValueError:
            log.warning("%s: unknown task %r, skipped", path, key)
            continue
        if isinstance(spec, str) and spec.strip():
            pins[task] = spec.strip()

    endpoints: dict[str, EndpointSpec] = {}
    for name, body in (raw.get("endpoints") or {}).items():
        try:
            custom.validate_name(name)
        except ValueError as exc:
            log.warning("%s: %s, skipped", path, exc)
            continue
        body = body if isinstance(body, dict) else {}
        caps = {
            str(model_id): tuple(str(c) for c in values)
            for model_id, values in (body.get("capabilities") or {}).items()
            if isinstance(values, (list, tuple))
        }
        endpoints[name] = EndpointSpec(label=str(body.get("label") or ""), capabilities=caps)
    return Overrides(pins=pins, endpoints=endpoints)


def save(o: Overrides, path: Path | None = None) -> None:
    path = path or routes_path()
    payload = {
        "version": VERSION,
        "pins": {task.value: spec for task, spec in sorted(o.pins.items(), key=lambda kv: kv[0].value)},
        "endpoints": {
            name: {"label": spec.label,
                   **({"capabilities": {m: list(c) for m, c in spec.capabilities.items()}}
                      if spec.capabilities else {})}
            for name, spec in sorted(o.endpoints.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Applying to the live table
# --------------------------------------------------------------------------

#: The table as shipped -- snapshotted once, before any pin touches it, so a
#: cleared pin restores exactly what mapping.py declared.
_SHIPPED: dict[Task, tuple[Candidate, ...]] = dict(TASK_ROUTES)
#: The pin Candidate at the head of each pinned task, by identity. What
#: `is_pin()` answers from -- the router asks it to skip evidence reordering.
_ACTIVE: dict[Task, Candidate] = {}
_REGISTERED: set[str] = set()
_PROBLEMS: list[str] = []


def shipped(task: Task) -> tuple[Candidate, ...]:
    return _SHIPPED[task]


def is_pin(task: Task, c: Candidate) -> bool:
    return _ACTIVE.get(task) is c


def pinned_spec(task: Task) -> str | None:
    pin = _ACTIVE.get(task)
    return pin.spec if pin is not None else None


def active_pins() -> dict[Task, str]:
    return {task: c.spec for task, c in _ACTIVE.items() if c.spec}


def last_problems() -> list[str]:
    return list(_PROBLEMS)


def validate_pin(task: Task, spec: str) -> None:
    """Shape and policy only -- whether the model exists is a question for a
    live catalogue, which the setup screen asks before offering it."""
    provider, sep, model = (spec or "").partition(":")
    if not sep or not provider or not model:
        raise PinError(f"{spec!r} is not 'provider:model'")
    if provider not in mapping.known_providers():
        raise PinError(
            f"unknown provider {provider!r}; known: {sorted(mapping.known_providers())}")
    bound = PROVIDER_ONLY.get(task)
    if bound and provider != bound[0]:
        raise PinError(f"{task.value} must stay on {bound[0]}: {bound[1]}")


def _pin_requires(head: Candidate) -> frozenset[Capability]:
    # The endpoint's own capability always, VISION always. Nothing else: a
    # person's explicit choice is not gated by a heuristic REASONING tag.
    return frozenset({ENDPOINT_CAPABILITY[head.endpoint]}) | (head.requires & {Capability.VISION})


def _pin_params(provider: str, chain: tuple[Candidate, ...]) -> dict:
    head = chain[0]
    for c in chain:
        if c.provider_name == provider:
            return dict(c.params)
    allowed = PARAMS_BY_PROVIDER.get(provider, {}).get(head.endpoint)
    if allowed is not None:
        return {k: v for k, v in head.params.items() if k in allowed}
    return {k: v for k, v in head.params.items() if k in ("temperature", "max_tokens")}


def _register_endpoints(endpoints: Mapping[str, EndpointSpec]) -> list[str]:
    problems: list[str] = []
    for name in list(_REGISTERED - set(endpoints)):
        unregister_custom(name)
        mapping.unregister_provider_name(name)
        _REGISTERED.discard(name)
    for name, spec in endpoints.items():
        try:
            cls = custom.openai_compatible(name, capability_overrides=spec.capabilities)
            register_custom(name, cls)
        except Exception as exc:  # bad name, SDK missing
            problems.append(f"endpoint {name!r} not registered: {exc}")
            continue
        mapping.register_provider_name(name, params_like="openai")
        temperature.register_provider(name)
        _REGISTERED.add(name)
    return problems


def apply(routes: dict[Task, tuple[Candidate, ...]] = TASK_ROUTES,
          o: Overrides | None = None, *,
          shipped_routes: Mapping[Task, tuple[Candidate, ...]] | None = None) -> list[str]:
    """Rebuild `routes` from the shipped table plus `o`. Idempotent. Returns
    the problems it found; every task with a problem keeps its shipped chain.

    `shipped_routes` is for a caller working on a copy of the table (tests);
    the live table always restores from the snapshot taken at import.
    """
    o = load() if o is None else o
    base = dict(shipped_routes) if shipped_routes is not None else (
        _SHIPPED if routes is TASK_ROUTES else dict(routes))
    problems = _register_endpoints(o.endpoints)

    for task, chain in base.items():
        routes[task] = chain
    _ACTIVE.clear()

    for task, spec in o.pins.items():
        chain = base[task]
        try:
            validate_pin(task, spec)
            provider, _, _ = spec.partition(":")
            pin = Candidate(
                spec=spec,
                requires=_pin_requires(chain[0]),
                endpoint=chain[0].endpoint,
                params=_pin_params(provider, chain),
            )
            new_chain = (pin, *[c for c in chain if c.spec != spec])
            mapping.validate({**routes, task: new_chain})
        except (PinError, MappingError) as exc:
            problems.append(f"pin {task.value} -> {spec}: {exc}")
            continue
        routes[task] = new_chain
        _ACTIVE[task] = pin

    _PROBLEMS[:] = problems
    for problem in problems:
        log.warning("routes: %s", problem)
    return problems


def apply_at_startup() -> None:
    """Called by agent/cli/main.py before the pipeline imports. Swallows
    everything: a preferences file can degrade routing, never prevent it."""
    if os.environ.get(IGNORE_ENV, "").strip():
        return
    try:
        apply()
    except Exception as exc:  # pragma: no cover -- last line of defence
        log.warning("could not apply %s: %s", routes_path(), exc)


# --------------------------------------------------------------------------
# Edits -- each saves the file and re-applies
# --------------------------------------------------------------------------

def pins() -> dict[Task, str]:
    return dict(load().pins)


def endpoints() -> dict[str, EndpointSpec]:
    return dict(load().endpoints)


def set_pin(task: Task, spec: str) -> None:
    o = load()
    if o.endpoints:
        _register_endpoints(o.endpoints)   # so validate_pin knows custom names
    validate_pin(task, spec)
    new = Overrides(pins={**o.pins, task: spec}, endpoints=o.endpoints)
    problems = [p for p in apply(o=new) if p.startswith(f"pin {task.value} ")]
    if problems:
        apply(o=o)
        raise PinError(problems[0])
    save(new)


def clear_pin(task: Task) -> None:
    o = load()
    if task not in o.pins:
        return
    new = Overrides(pins={t: s for t, s in o.pins.items() if t != task}, endpoints=o.endpoints)
    save(new)
    apply(o=new)


def add_endpoint(name: str, label: str = "") -> None:
    custom.validate_name(name)
    o = load()
    spec = o.endpoints.get(name, EndpointSpec())
    new = Overrides(pins=o.pins, endpoints={**o.endpoints, name: EndpointSpec(label or spec.label, spec.capabilities)})
    save(new)
    apply(o=new)


def remove_endpoint(name: str) -> None:
    """Forget an endpoint and every pin on it. The caller unsets the .env rows."""
    o = load()
    new = Overrides(
        pins={t: s for t, s in o.pins.items() if s.partition(":")[0] != name},
        endpoints={n: e for n, e in o.endpoints.items() if n != name},
    )
    save(new)
    apply(o=new)


def set_capabilities(endpoint: str, model_id: str, caps: Iterable[Capability | str]) -> None:
    o = load()
    if endpoint not in o.endpoints:
        raise PinError(f"no endpoint called {endpoint!r}")
    spec = o.endpoints[endpoint]
    names = tuple(Capability(c).value for c in caps)
    new_spec = EndpointSpec(spec.label, {**spec.capabilities, model_id: names})
    new = Overrides(pins=o.pins, endpoints={**o.endpoints, endpoint: new_spec})
    save(new)
    apply(o=new)
