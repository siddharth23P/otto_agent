"""Streaming contract for ChatInception.

Both of these were live bugs, and both were invisible in the terminal: the
answer looked right on screen while what LangChain accumulated -- and therefore
what Langfuse recorded -- was wrong. That is the whole reason these assertions
are about the *accumulated message*, not about what the panel showed.

No network: the SDK client is a stand-in that replays a scripted stream.
"""

from types import SimpleNamespace

import pytest

from agent.router.llm_provider.inception_provider import ChatInception


# --------------------------------------------------------------------------
# A stand-in for `client.chat.completions`
# --------------------------------------------------------------------------


def content_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def usage_chunk(usage) -> SimpleNamespace:
    """The final chunk: `choices` is empty and the usage payload rides along."""
    return SimpleNamespace(choices=[], usage=usage)


class FakeCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.payload = None

    def create(self, **payload):
        self.payload = payload
        return iter(self.chunks)


def model(chunks, **kwargs) -> ChatInception:
    completions = FakeCompletions(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    llm = ChatInception(client=client, model="mercury-2", **kwargs)
    llm._fake = completions  # for payload assertions
    return llm


def accumulate(llm, prompt="hi"):
    """What LangChain hands to `on_llm_end` -- i.e. what Langfuse stores."""
    total = None
    for chunk in llm.stream(prompt):
        total = chunk if total is None else total + chunk
    return total


# --------------------------------------------------------------------------
# Diffusion: snapshots, not deltas
# --------------------------------------------------------------------------

FRAMES = ["he__o w__ld", "hel_o wor_d", "hello world"]


def test_diffusion_accumulates_only_the_final_snapshot():
    """The regression: every draft used to be concatenated into the answer."""
    llm = model([content_chunk(f) for f in FRAMES], diffusing=True)

    assert accumulate(llm).content == "hello world"


def test_diffusion_frames_reach_the_sink_in_order():
    seen: list[str] = []
    llm = model([content_chunk(f) for f in FRAMES], diffusing=True, frame_sink=seen.append)

    accumulate(llm)

    assert seen == FRAMES


def test_diffusion_yields_exactly_one_content_chunk():
    """Snapshots must not be yielded, or accumulation breaks again."""
    llm = model([content_chunk(f) for f in FRAMES], diffusing=True)

    texts = [c.content for c in llm.stream("hi") if c.content]

    assert texts == ["hello world"]


def test_diffusion_without_a_sink_still_returns_the_answer():
    """A sink is a display concern; correctness must not depend on one."""
    llm = model([content_chunk(f) for f in FRAMES], diffusing=True)

    assert accumulate(llm).content == "hello world"


def test_frame_sink_never_reaches_the_api_or_the_callbacks():
    """Excluded from serialisation: a callable in invocation_params would be
    handed to every callback, and Langfuse would store its repr."""
    llm = model([content_chunk(f) for f in FRAMES], diffusing=True, frame_sink=lambda _: None)

    accumulate(llm)

    assert "frame_sink" not in llm._fake.payload
    assert "frame_sink" not in llm._get_invocation_params()


def test_an_empty_diffusion_stream_raises_rather_than_answering_nothing():
    """LangChain's own guard, documented here because the diffusion path is
    the one that could plausibly yield no chunks at all: a stream that never
    produced a snapshot must not look like a successful empty answer."""
    llm = model([], diffusing=True)

    with pytest.raises(ValueError, match="No generation chunks"):
        accumulate(llm)


# --------------------------------------------------------------------------
# Everyone else: real deltas, appended
# --------------------------------------------------------------------------


def test_non_diffusing_stream_appends_deltas():
    llm = model([content_chunk("hel"), content_chunk("lo")])

    assert accumulate(llm).content == "hello"


def test_non_diffusing_stream_yields_every_delta():
    llm = model([content_chunk("hel"), content_chunk("lo")])

    assert [c.content for c in llm.stream("hi") if c.content] == ["hel", "lo"]


# --------------------------------------------------------------------------
# Usage: the SDK does not type it on the streaming path
# --------------------------------------------------------------------------

EXPECTED = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}


def test_usage_arriving_as_a_dict():
    """The regression: ChatCompletionChunk declares no `usage` field, so
    pydantic keeps the payload in extras as a plain dict."""
    llm = model(
        [content_chunk("hi"),
         usage_chunk({"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})]
    )

    assert accumulate(llm).usage_metadata == EXPECTED


def test_usage_arriving_as_an_object():
    llm = model(
        [content_chunk("hi"),
         usage_chunk(SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15))]
    )

    assert accumulate(llm).usage_metadata == EXPECTED


def test_usage_total_is_derived_when_the_vendor_omits_it():
    llm = model(
        [content_chunk("hi"), usage_chunk({"prompt_tokens": 12, "completion_tokens": 3})]
    )

    assert accumulate(llm).usage_metadata == EXPECTED


def test_usage_survives_the_diffusion_path():
    llm = model(
        [content_chunk(f) for f in FRAMES]
        + [usage_chunk({"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15})],
        diffusing=True,
    )

    total = accumulate(llm)

    assert total.content == "hello world"
    assert total.usage_metadata == EXPECTED


def test_a_usage_chunk_with_no_counts_is_ignored():
    """A bare terminal chunk must not report a phantom 0/0 generation."""
    llm = model([content_chunk("hi"), usage_chunk({})])

    assert accumulate(llm).usage_metadata is None


# --------------------------------------------------------------------------
# The request itself
# --------------------------------------------------------------------------


def test_streaming_asks_for_usage():
    """Without stream_options the API never sends a usage chunk at all, and
    every streamed call silently costs nothing."""
    llm = model([content_chunk("hi")])

    accumulate(llm)

    assert llm._fake.payload["stream"] is True
    assert llm._fake.payload["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("flag", [True, None])
def test_diffusing_is_only_sent_when_set(flag):
    llm = model([content_chunk("hi")], diffusing=flag)

    accumulate(llm)

    assert llm._fake.payload.get("diffusing") == flag if flag else "diffusing" not in llm._fake.payload
