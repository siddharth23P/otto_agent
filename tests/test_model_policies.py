"""Coverage for the two per-model policies the router applies: which models a
vendor will still serve (retired.py) and what temperature each one honours
(temperature.py).

Both exist because a catalogue is not a promise and a vendor is not uniform.
Entirely offline -- no keys, no network.
"""
import pytest

from agent.router.llm_provider import retired
from agent.router.llm_provider.temperature import (
    TemperaturePolicy,
    apply_to_params,
    policy_for,
)


# ---- retired models -------------------------------------------------------


def test_a_curated_retirement_is_recognised():
    """gemini-2.5-flash is still returned by the models endpoint and answers
    generateContent with 404 'no longer available' -- the exact combination
    this guards against, since Router._select can only check a pin exists."""
    assert retired.is_retired("gemini", "gemini-2.5-flash")
    assert not retired.is_retired("gemini", "gemini-3-flash-preview")


def test_only_a_permanent_refusal_reads_as_retirement():
    """Kept narrow deliberately: treating a transient failure as a retirement
    would quietly shrink the catalogue for the rest of the run."""
    assert retired.looks_retired(Exception("404 NOT_FOUND: model is no longer available"))
    assert retired.looks_retired(Exception("This model has been deprecated"))
    assert not retired.looks_retired(Exception("429 rate limit exceeded"))
    assert not retired.looks_retired(Exception("503 service unavailable, try again"))


def test_a_retirement_learned_at_call_time_holds_for_the_process():
    """Without this the next node in the same run resolves the same dead id
    and fails the same way."""
    assert not retired.is_retired("openai", "gpt-imaginary")

    retired.note_retired("openai", "gpt-imaginary", "404 does not exist")

    assert retired.is_retired("openai", "gpt-imaginary")
    retired._LEARNED.get("openai", set()).discard("gpt-imaginary")


def test_the_catalogue_filter_hides_retired_models(monkeypatch):
    """Filtered at the catalogue so routing, `otto models`, the doctor and the
    health report all see the same thing."""
    from agent.router.llm_provider.base import BaseProvider, Capability, ModelInfo

    class _Fake(BaseProvider):
        name = "gemini"
        env_var = "GEMINI_API_KEY"

        def _build_client(self):
            return None

        def _fetch_models(self):
            return [
                ModelInfo(id=i, provider="gemini", display_name=i,
                          capabilities=frozenset({Capability.CHAT}),
                          context_window=None, max_output_tokens=None, raw={})
                for i in ("gemini-2.5-flash", "gemini-3-flash-preview")
            ]

        def chat_model(self, model_id, **kwargs):
            raise NotImplementedError

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    ids = [m.id for m in _Fake().list_models()]

    assert ids == ["gemini-3-flash-preview"]


# ---- temperature ----------------------------------------------------------


def test_inception_is_clamped_up_rather_than_silently_reset():
    """Its documented range is 0.5-1.0 and out-of-range is reset to the model
    default of 1.0 -- so asking for 0.0 got the MOST random setting available,
    for exactly the nodes written to be the most careful."""
    assert apply_to_params("inception", "mercury-2.5", {"temperature": 0.0}) == {"temperature": 0.5}


def test_a_reasoning_model_that_rejects_temperature_has_it_dropped():
    """o4-mini answers temperature=0.0 with a 400; gpt-5-mini accepts it.
    Verified against the live API rather than inferred from the family name."""
    assert apply_to_params("openai", "o4-mini", {"temperature": 0.0, "max_tokens": 8}) == {"max_tokens": 8}
    assert apply_to_params("openai", "gpt-5-mini", {"temperature": 0.0})["temperature"] == 0.0


def test_vendors_that_honour_zero_keep_it():
    assert apply_to_params("anthropic", "claude-haiku-4-5-20251001", {"temperature": 0.0})["temperature"] == 0.0
    assert apply_to_params("gemini", "gemini-3-flash-preview", {"temperature": 0.0})["temperature"] == 0.0


def test_an_out_of_range_value_is_clamped_to_the_nearest_honoured_one():
    """Clamping beats dropping where a range exists: the closest honoured
    value preserves what the caller was asking for."""
    assert apply_to_params("gemini", "gemini-3-flash-preview", {"temperature": 2.5})["temperature"] == 2.0


def test_params_without_a_temperature_are_untouched():
    params = {"max_tokens": 64}

    assert apply_to_params("inception", "mercury-2.5", params) is params


def test_an_unmeasured_provider_is_left_alone():
    """The honest default for a vendor whose behaviour nobody has measured."""
    assert policy_for("mistral", "large") is None
    assert apply_to_params("mistral", "large", {"temperature": 0.0})["temperature"] == 0.0


def test_a_fixed_policy_drops_rather_than_clamps():
    assert TemperaturePolicy(fixed=True).apply(0.7) is None
    assert TemperaturePolicy(low=0.5, high=1.0).apply(0.0) == 0.5


# ---- models that are alive but cannot do the job -------------------------


def test_non_chat_model_families_are_filtered_out():
    """Capability detection is generous -- Gemini's provider tags every
    gemini-* model VISION-capable. A pin hides that; a capability fallback does
    not, and picked a text-to-speech model the first time one was tried."""
    from agent.router.llm_provider.retired import is_serviceable, is_unusable

    assert is_unusable("gemini", "gemini-2.5-flash-preview-tts")
    assert is_unusable("gemini", "gemini-3.5-transcribe")
    assert is_unusable("gemini", "imagen-4.0-generate-001")
    assert is_unusable("openai", "text-embedding-3-small")
    assert is_unusable("openai", "whisper-1")

    assert not is_unusable("gemini", "gemini-3-flash-preview")
    assert not is_unusable("openai", "gpt-5-mini")
    assert is_serviceable("gemini", "gemini-3-flash-preview")


def test_a_model_that_serves_a_different_api_reads_as_permanently_unusable():
    """Gemini answers a generateContent call to some ids with 'This model only
    supports Interactions API'. Not a retirement, but picking it twice in one
    run is pure waste."""
    from agent.router.llm_provider.retired import looks_retired

    assert looks_retired(Exception("400 INVALID_ARGUMENT: This model only supports Interactions API"))


def test_vision_is_named_gemini_flash_versions_then_other_vendors_never_a_text_model():
    """The person's order (2026-09-17): Gemini 3.8, 3.7, 3.6 Flash -- released names, no preview,
    no -latest alias -- and then Claude and GPT-5-mini, which also see. Never Inception."""
    from agent.router.llm_provider.base import Capability
    from agent.router.mapping import TASK_ROUTES, Task

    chain = TASK_ROUTES[Task.VISION]

    assert [c.spec for c in chain] == [
        "gemini:gemini-3.8-flash", "gemini:gemini-3.7-flash", "gemini:gemini-3.6-flash",
        "anthropic:claude-haiku-4-5-20251001", "openai:gpt-5-mini",
    ]
    assert all(Capability.VISION in c.requires for c in chain)
    assert not any("preview" in c.spec or "latest" in c.spec for c in chain)
    assert "temperature" not in chain[-1].params, "a reasoning model takes no temperature"


def test_a_vendors_own_published_ceiling_beats_the_provider_default():
    """Gemini publishes max_temperature per model and it is NOT uniform: most
    cap at 2, several at 1. A per-provider guess of 0-2 overshoots those."""
    from agent.router.llm_provider.temperature import apply_to_params, published_maximum

    class _Model:
        id = "some-capped-model"
        raw = {"max_temperature": 1.0}

    assert published_maximum(_Model()) == 1.0
    sent = apply_to_params("gemini", "some-capped-model", {"temperature": 1.8}, _Model())
    assert sent["temperature"] == 1.0


def test_the_published_ceiling_is_read_from_a_dict_or_an_object():
    """ModelInfo.raw is whatever the SDK returned."""
    from agent.router.llm_provider.temperature import published_maximum

    class _Obj:
        max_temperature = 2.0

    class _WithObj:
        raw = _Obj()

    class _WithDict:
        raw = {"maxTemperature": 2.0}

    assert published_maximum(_WithObj()) == 2.0
    assert published_maximum(_WithDict()) == 2.0
    assert published_maximum(object()) is None


def test_a_model_that_admits_no_choice_still_wins_over_a_published_ceiling():
    """o4-mini rejects every value but its default, so a range would be wrong
    even if OpenAI published one."""
    from agent.router.llm_provider.temperature import policy_for

    class _Model:
        raw = {"max_temperature": 2.0}

    assert policy_for("openai", "o4-mini", _Model()).fixed


# ---- anthropic: temperature removed from the Opus 4.7 generation on ---------


def _claude(model_id: str, *, enabled_thinking: bool | None):
    """A ModelInfo shaped like the anthropic provider builds it: `raw` is the
    Models API payload. None means a payload with no capabilities block."""
    from agent.router.llm_provider.base import ModelInfo

    raw = {"id": model_id}
    if enabled_thinking is not None:
        raw["capabilities"] = {"thinking": {"supported": True, "types": {
            "adaptive": {"supported": True},
            "enabled": {"supported": enabled_thinking},
        }}}
    return ModelInfo(id=model_id, provider="anthropic", raw=raw)


def test_a_claude_model_that_dropped_enabled_thinking_has_temperature_dropped():
    """Opus 4.7 answers temperature=0.0 with `400 temperature is deprecated
    for this model`; it went in the same generation as `thinking: enabled`,
    which the Models API does publish. Verified live."""
    model = _claude("claude-opus-4-7", enabled_thinking=False)
    assert apply_to_params("anthropic", model.id, {"temperature": 0.0, "max_tokens": 8}, model) == {"max_tokens": 8}


def test_a_claude_model_that_still_takes_enabled_thinking_keeps_temperature():
    model = _claude("claude-haiku-4-5-20251001", enabled_thinking=True)
    assert apply_to_params("anthropic", model.id, {"temperature": 0.0}, model)["temperature"] == 0.0


def test_the_published_flag_beats_the_model_name():
    """A future Claude name this file has never seen still gets it right,
    in both directions, because the vendor's flag decides."""
    unseen_new = _claude("claude-haiku-9", enabled_thinking=False)
    assert "temperature" not in apply_to_params("anthropic", unseen_new.id, {"temperature": 0.0}, unseen_new)
    looks_new_but_is_not = _claude("claude-opus-5", enabled_thinking=True)
    assert apply_to_params("anthropic", looks_new_but_is_not.id, {"temperature": 0.0}, looks_new_but_is_not)["temperature"] == 0.0


@pytest.mark.parametrize("model_id", [
    "claude-opus-4-7", "claude-opus-4-8", "claude-opus-5", "claude-sonnet-5", "claude-fable-5-1",
])
def test_without_a_capabilities_block_the_known_family_is_recognised_by_name(model_id):
    """The fallback for a catalogue entry fetched before the flag existed, or
    a bare id with no catalogue entry at all."""
    assert apply_to_params("anthropic", model_id, {"temperature": 0.0}, _claude(model_id, enabled_thinking=None)) == {}
    assert apply_to_params("anthropic", model_id, {"temperature": 0.0}) == {}


@pytest.mark.parametrize("model_id", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"])
def test_without_a_capabilities_block_the_older_family_keeps_temperature(model_id):
    assert apply_to_params("anthropic", model_id, {"temperature": 0.0})["temperature"] == 0.0
