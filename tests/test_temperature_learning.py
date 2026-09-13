"""A model no table knows gets its temperature policy by asking it.

The first call carries the temperature the route wanted. If the model refuses
it, `_call` retries without one and the refusal is remembered -- for the rest
of the process through `policy_for`, and for future ones through
~/.otto/temperature.json. Entirely offline: the "vendor" here is a fake that
refuses like Anthropic does.
"""
import json
from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent.pipeline.nodes import _call
from agent.router.llm_provider import retired
from agent.router.llm_provider.base import ModelNotFound, ProviderError, translate_unknown
from agent.router.llm_provider.temperature import (
    apply_to_params,
    bind_store,
    looks_like_temperature_rejection,
    note_rejects_temperature,
    policy_for,
    rejects_temperature,
)


class _Refusal(Exception):
    """Shaped like a vendor SDK's 400: a message and a status_code."""

    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.status_code = status_code


ANTHROPIC_MESSAGE = (
    "Error code: 400 - {'type': 'error', 'error': {'type': 'invalid_request_error', "
    "'message': '`temperature` is deprecated for this model.'}}"
)


# ---- recognising a refusal ----------------------------------------------------


def test_a_400_naming_temperature_is_a_refusal():
    assert looks_like_temperature_rejection(_Refusal(ANTHROPIC_MESSAGE))
    assert looks_like_temperature_rejection(
        _Refusal("Unsupported value: 'temperature' does not support 0.0 with this model."))


def test_a_client_side_refusal_without_a_status_counts_too():
    """langchain_anthropic raises a bare ValueError for a model it knows takes
    none, before any request is made."""
    assert looks_like_temperature_rejection(
        ValueError("`temperature` is not supported for claude-fable-5-1 at non-default values."))


def test_other_failures_that_mention_the_word_are_not_refusals():
    assert not looks_like_temperature_rejection(_Refusal("rate limited; request had temperature=0", 429))
    assert not looks_like_temperature_rejection(_Refusal("model overloaded", 529))
    assert not looks_like_temperature_rejection(_Refusal("max_tokens must be positive"))


# ---- remembering it ------------------------------------------------------------


def test_a_learned_refusal_drops_temperature_for_the_rest_of_the_process():
    assert apply_to_params("newvendor", "shiny-1", {"temperature": 0.0}) == {"temperature": 0.0}

    note_rejects_temperature("newvendor", "shiny-1", "400 temperature is deprecated")

    assert policy_for("newvendor", "shiny-1").fixed
    assert apply_to_params("newvendor", "shiny-1", {"temperature": 0.0, "max_tokens": 8}) == {"max_tokens": 8}
    assert apply_to_params("newvendor", "shiny-2", {"temperature": 0.0}) == {"temperature": 0.0}


def test_a_learned_refusal_beats_a_table_that_would_have_sent_one():
    """The table says Haiku honours 0-1; the model itself outranks the table."""
    note_rejects_temperature("anthropic", "claude-haiku-4-5-20251001", "400 whatever it said")

    assert "temperature" not in apply_to_params("anthropic", "claude-haiku-4-5-20251001", {"temperature": 0.0})


def test_a_refusal_survives_into_the_next_process(tmp_path):
    """Written to the store as it is learned and read back by a fresh process
    -- simulated by dropping the in-memory copy and binding the same file."""
    from agent.router.llm_provider import temperature as module

    store = tmp_path / "temperature.json"
    with bind_store(store):
        note_rejects_temperature("newvendor", "shiny-1", "AnthropicInvalidRequestError: 400 ...")
    on_disk = json.loads(store.read_text())
    assert "AnthropicInvalidRequestError" in on_disk["rejects_temperature"]["newvendor"]["shiny-1"]

    module._learned_by_store.pop(store, None)  # forget, as a new process would have
    with bind_store(store):
        assert rejects_temperature("newvendor", "shiny-1")
        assert not rejects_temperature("newvendor", "shiny-2")


def test_an_unreadable_store_starts_empty_rather_than_crashing(tmp_path):
    store = tmp_path / "temperature.json"
    store.write_text("{not json")
    with bind_store(store):
        assert not rejects_temperature("anyone", "anything")


# ---- not a retirement ---------------------------------------------------------


def test_a_refused_temperature_is_not_read_as_a_retired_model():
    """Anthropic says "`temperature` is deprecated", which matches retired.py's
    "is deprecated" marker. The model is alive; hiding it for the rest of the
    run would be the wrong lesson."""
    exc = _Refusal(ANTHROPIC_MESSAGE)

    translated = translate_unknown(exc, provider="anthropic", model_id="claude-opus-4-7")

    assert isinstance(translated, ProviderError)
    assert not isinstance(translated, ModelNotFound)
    assert not retired.is_retired("anthropic", "claude-opus-4-7")


# ---- the retry itself ---------------------------------------------------------


class _RefusingLLM:
    """A chat model that refuses any temperature and answers without one.
    `model_copy` is what _call uses to rebuild it, as pydantic models offer."""

    model = "shiny-1"
    _otto_provider = "newvendor"

    def __init__(self, temperature, log):
        self.temperature = temperature
        self.log = log

    def stream(self, messages):
        self.log.append(self.temperature)
        if self.temperature is not None:
            raise _Refusal(ANTHROPIC_MESSAGE)
        yield SimpleNamespace(content="the answer", response_metadata={"finish_reason": "stop"},
                              usage_metadata=None)

    def model_copy(self, update):
        return _RefusingLLM(update.get("temperature", self.temperature), self.log)


def test_a_refused_call_is_retried_without_temperature_and_remembered():
    log = []
    llm = _RefusingLLM(temperature=0.0, log=log)

    assert _call(llm, [HumanMessage("hi")]) == "the answer"

    assert log == [0.0, None], "one refused request, one retried without"
    assert rejects_temperature("newvendor", "shiny-1")
    # And the router will not send one to this model again.
    assert apply_to_params("newvendor", "shiny-1", {"temperature": 0.3}) == {}


def test_a_call_that_sent_no_temperature_is_not_retried_for_it():
    """The message alone is not enough: if no temperature went out, the
    refusal is about something else and must surface as the error it is."""
    log = []
    llm = _RefusingLLM(temperature=None, log=log)

    def refuse(messages):
        log.append("called")
        raise _Refusal(ANTHROPIC_MESSAGE)
        yield  # noqa: unreachable -- keeps this a generator like stream()

    llm.stream = refuse
    try:
        _call(llm, [HumanMessage("hi")])
    except ProviderError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected the refusal to surface")

    assert log == ["called"]
