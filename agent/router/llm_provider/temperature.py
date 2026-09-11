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
  * Anthropic and Gemini simply honour what they are given, over different
    ranges (0-1 and 0-2).

So a route asks for the temperature it WANTS, and this decides what the model
can be given. Clamping beats dropping where a range exists, because the
closest honoured value preserves the caller's intent; dropping is only right
where the model admits no choice at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


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
#: o3, o4, ...) all behave this way.
_FIXED_TEMPERATURE_PATTERNS = {
    "openai": (re.compile(r"^o\d"),),
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


def policy_for(provider: str, model_id: str) -> TemperaturePolicy | None:
    """The policy for one model, or None if this provider is unmeasured."""
    if (provider, model_id) in _BY_MODEL:
        return _BY_MODEL[(provider, model_id)]
    for pattern in _FIXED_TEMPERATURE_PATTERNS.get(provider, ()):
        if pattern.match(model_id):
            return TemperaturePolicy(fixed=True)
    return _BY_PROVIDER.get(provider)


def apply_to_params(provider: str, model_id: str, params: dict) -> dict:
    """`params` with `temperature` adjusted to what this model honours.

    Returns a new dict; drops the key entirely for a fixed-temperature model.
    """
    if "temperature" not in params:
        return params
    policy = policy_for(provider, model_id)
    if policy is None:
        return params
    adjusted = policy.apply(params["temperature"])
    updated = dict(params)
    if adjusted is None:
        del updated["temperature"]
    else:
        updated["temperature"] = adjusted
    return updated
