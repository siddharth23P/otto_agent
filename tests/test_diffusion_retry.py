"""Regression test for the diffusion-truncation bug surfaced by a real VRP-task
transcript: one part's "author" output was a single bare `return
total_distance, penalty` line, no function around it -- a Mercury diffusing
call cut off by max_tokens before the denoising process converged.

Unlike autoregressive truncation, a diffusion answer cut short is not a clean
prefix: any position in the text can still hold unconverged noise, not just
the tail. So `_call()` treats a diffusing call's finish_reason=='length' as a
retryable failure (doubling max_tokens each time), not as ordinary "ran out of
budget" text -- which is exactly how a non-diffusing truncation is still
handled: accepted as-is.

This also covers the smaller gap the fix depends on: `ChatInception._stream()`
previously discarded `finish_reason` entirely on the diffusion path, so there
was no way for any caller to tell a converged snapshot from an interrupted
one in the first place.

`_call()` was ported verbatim from the retired agent/graph/code_nodes.py into
agent/pipeline/nodes.py when the code hive was replaced by the pipeline
(2026-09-09) -- this test moved with it, import only.
"""
from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent.pipeline.nodes import MAX_DIFFUSION_RETRIES, _call
from agent.router.llm_provider.base import ProviderError
from agent.router.llm_provider.inception_provider import ChatInception


def _chunk(content="", finish_reason=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=content), finish_reason=finish_reason)],
    )


class _FakeCompletions:
    def __init__(self, scripted_streams):
        self.scripted_streams = list(scripted_streams)
        self.calls: list[dict] = []

    def create(self, **payload):
        self.calls.append(payload)
        return iter(self.scripted_streams.pop(0))


class _FakeClient:
    def __init__(self, scripted_streams):
        self.chat = SimpleNamespace(completions=_FakeCompletions(scripted_streams))


def test_clean_diffusion_response_is_not_retried():
    stream = [_chunk("partial noisy"), _chunk("def f():\n    return 1", finish_reason="stop")]
    client = _FakeClient([stream])
    llm = ChatInception(client=client, model="mercury-2.5", diffusing=True, max_tokens=1024)

    text = _call(llm, [HumanMessage("hi")])

    assert text == "def f():\n    return 1"
    assert len(client.chat.completions.calls) == 1


def test_one_truncation_is_retried_at_doubled_max_tokens_then_accepted():
    streams = [
        [_chunk("return total_distance, penalty", finish_reason="length")],
        [_chunk("def evaluate():\n    ...\n    return total_distance, penalty", finish_reason="stop")],
    ]
    client = _FakeClient(streams)
    llm = ChatInception(client=client, model="mercury-2.5", diffusing=True, max_tokens=1024)

    text = _call(llm, [HumanMessage("hi")])

    assert text == "def evaluate():\n    ...\n    return total_distance, penalty"
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 1024
    assert calls[1]["max_tokens"] == 2048


def test_truncation_on_every_attempt_raises_instead_of_returning_garbage():
    streams = [[_chunk("garbled", finish_reason="length")] for _ in range(MAX_DIFFUSION_RETRIES)]
    client = _FakeClient(streams)
    llm = ChatInception(client=client, model="mercury-2.5", diffusing=True, max_tokens=512)

    with pytest.raises(ProviderError, match="refusing to hand an unconverged diffusion snapshot"):
        _call(llm, [HumanMessage("hi")])

    assert len(client.chat.completions.calls) == MAX_DIFFUSION_RETRIES


def test_non_diffusing_truncation_is_accepted_as_is():
    client = _FakeClient([[_chunk("normal streamed text", finish_reason="length")]])
    llm = ChatInception(client=client, model="mercury-edit-2", diffusing=False, max_tokens=64)

    text = _call(llm, [HumanMessage("hi")])

    assert text == "normal streamed text"
    assert len(client.chat.completions.calls) == 1


# ---- an empty stream from the provider ------------------------------------


class _EmptyStreamLLM:
    """Stands in for a provider that yields nothing: langchain_core raises
    ValueError("No generation chunks were returned") from inside stream()
    itself, so _call()'s `reply is None` branch never sees it."""

    model = "mercury-2.5"
    max_tokens = 1024
    diffusing = True

    def __init__(self, failures: int, then: str = ""):
        self.remaining = failures
        self.then = then
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        if self.remaining > 0:
            self.remaining -= 1
            raise ValueError("No generation chunks were returned")
        yield SimpleNamespace(
            content=self.then, response_metadata={"finish_reason": "stop"},
        )

    def model_copy(self, update):
        return self


def test_an_empty_stream_is_retried_and_then_succeeds():
    llm = _EmptyStreamLLM(failures=1, then="the real answer")

    assert _call(llm, [HumanMessage("hi")]) == "the real answer"
    assert llm.calls == 2


def test_a_persistently_empty_stream_returns_empty_rather_than_raising():
    """It must not unwind the graph: an empty answer is a case every caller
    here already handles (the unparseable-reply retry), an exception is not."""
    llm = _EmptyStreamLLM(failures=MAX_DIFFUSION_RETRIES)

    assert _call(llm, [HumanMessage("hi")]) == ""
    assert llm.calls == MAX_DIFFUSION_RETRIES


def test_an_unrelated_value_error_still_propagates():
    class _Boom(_EmptyStreamLLM):
        def stream(self, messages):
            raise ValueError("something else entirely")
            yield  # pragma: no cover

    with pytest.raises(ValueError, match="something else entirely"):
        _call(_Boom(failures=0), [HumanMessage("hi")])
