"""Coverage for the per-model token ledger (agent/pipeline/usage.py) and the
panel that reads it (agent/cli/tui.py's UsagePanel).

`model_calls` was the only number a turn reported about what it spent, and it
is the same number whether a request went to a small diffusion model or carried
a 60k-token transcript to a frontier one. The tokens were already arriving --
every provider attaches langchain's `usage_metadata`, and the Inception adapter
goes as far as asking the API to send the usage chunk -- and `_call` read the
text off the reply and dropped the rest.

Three pieces, tested separately:

  * UsageLedger itself -- accumulating per model, tolerating a provider that
    reports nothing or reports rubbish, and the `reported` flag that keeps
    "said zero" apart from "said nothing".
  * `_call` recording into whatever ledger is bound, once per HTTP REQUEST
    (retries included) rather than once per logical call.
  * the panel's own formatting, and the wiring that gets a ledger from the app
    to the pipeline and back onto the screen.
"""
import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.pipeline import nodes as pn
from agent.pipeline.usage import (
    UsageLedger, bind_usage, current_usage, record_usage,
)


# --------------------------------------------------------------------------
# UsageLedger
# --------------------------------------------------------------------------

def test_a_ledger_accumulates_calls_and_tokens_per_model():
    led = UsageLedger()
    led.record("sonnet", {"input_tokens": 100, "output_tokens": 10})
    led.record("sonnet", {"input_tokens": 200, "output_tokens": 20})
    led.record("mercury", {"input_tokens": 5, "output_tokens": 1})

    by_name = {m.model: m for m in led.models()}
    assert by_name["sonnet"].calls == 2
    assert by_name["sonnet"].input_tokens == 300
    assert by_name["sonnet"].output_tokens == 30
    assert by_name["mercury"].calls == 1


def test_totals_are_the_sum_across_models():
    led = UsageLedger()
    led.record("a", {"input_tokens": 10, "output_tokens": 1})
    led.record("b", {"input_tokens": 20, "output_tokens": 2})

    assert led.calls == 2
    assert led.input_tokens == 30
    assert led.output_tokens == 3
    assert led.total_tokens == 33


def test_models_come_back_in_the_order_they_first_answered():
    # The order models first appear is the order the turn escalated through
    # them, and sorting by name or by size would throw that away.
    led = UsageLedger()
    for name in ("mercury", "sonnet", "mercury", "opus"):
        led.record(name, {"input_tokens": 1, "output_tokens": 1})

    assert [m.model for m in led.models()] == ["mercury", "sonnet", "opus"]


def test_a_model_that_reports_nothing_still_has_its_calls_counted():
    led = UsageLedger()
    led.record("quiet", None)
    led.record("quiet", {})

    entry = led.models()[0]
    assert entry.calls == 2
    assert entry.total_tokens == 0
    # The whole point: "said nothing" must not read as "cost nothing".
    assert entry.reported is False


def test_reported_is_true_as_soon_as_any_call_reports():
    led = UsageLedger()
    led.record("m", None)
    led.record("m", {"input_tokens": 7, "output_tokens": 0})

    assert led.models()[0].reported is True


@pytest.mark.parametrize("usage", [
    {"input_tokens": None, "output_tokens": None},
    {"input_tokens": "lots", "output_tokens": "some"},
    {"nothing": "useful"},
])
def test_rubbish_from_a_provider_is_survived_not_raised(usage):
    # Four vendors' adapters feed this. A field that is missing or the wrong
    # type must never be the reason a turn fails.
    led = UsageLedger()
    led.record("m", usage)

    assert led.models()[0].calls == 1
    assert led.total_tokens == 0


def test_a_model_with_no_name_is_recorded_rather_than_dropped():
    led = UsageLedger()
    led.record("", {"input_tokens": 1, "output_tokens": 1})

    assert led.models()[0].model == "unknown"


def test_a_snapshot_is_plain_data_and_does_not_change_under_a_reader():
    led = UsageLedger()
    led.record("m", {"input_tokens": 10, "output_tokens": 1})
    snap = led.snapshot()
    led.record("m", {"input_tokens": 10, "output_tokens": 1})

    assert snap["total_tokens"] == 11  # the later write did not reach it
    assert snap["models"][0]["model"] == "m"


# --------------------------------------------------------------------------
# Binding
# --------------------------------------------------------------------------

def test_nothing_bound_makes_recording_a_no_op():
    # A direct _call in a unit test must not need ceremony.
    assert current_usage() is None
    record_usage("m", {"input_tokens": 1})  # must not raise


def test_binding_a_ledger_makes_it_the_one_recorded_into():
    led = UsageLedger()
    with bind_usage(led):
        record_usage("m", {"input_tokens": 4, "output_tokens": 1})

    assert led.total_tokens == 5
    assert current_usage() is None  # unbound again on the way out


# --------------------------------------------------------------------------
# _call -- the one place every model REQUEST passes through
# --------------------------------------------------------------------------

class _FakeModel:
    def __init__(self, *chunks):
        self._chunks = chunks
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        yield from self._chunks


def _chunk(text: str, usage: dict | None = None) -> AIMessageChunk:
    if usage is not None:
        # langchain validates the shape; a real provider always fills this in.
        usage = dict(usage)
        usage.setdefault(
            "total_tokens",
            (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0),
        )
    return AIMessageChunk(content=text, usage_metadata=usage)


def test_call_records_the_usage_the_provider_streamed_back():
    led = UsageLedger()
    fake = _FakeModel(_chunk("hello", {
        "input_tokens": 120, "output_tokens": 8, "total_tokens": 128,
    }))

    with bind_usage(led):
        pn._call(fake, [HumanMessage("hi")])

    assert led.total_tokens == 128
    assert led.models()[0].calls == 1


def test_call_records_a_model_that_streams_back_no_usage_at_all():
    led = UsageLedger()
    fake = _FakeModel(_chunk("hello"))

    with bind_usage(led):
        pn._call(fake, [HumanMessage("hi")])

    entry = led.models()[0]
    assert entry.calls == 1
    assert entry.reported is False


def test_call_records_against_the_model_that_actually_answered(monkeypatch):
    led = UsageLedger()
    fake = _FakeModel(_chunk("hi", {"input_tokens": 1, "output_tokens": 1}))
    monkeypatch.setattr(pn, "_model_label", lambda llm: "the-real-model-id")

    with bind_usage(led):
        pn._call(fake, [HumanMessage("hi")])

    assert led.models()[0].model == "the-real-model-id"


def test_call_with_no_ledger_bound_still_works():
    fake = _FakeModel(_chunk("hello", {"input_tokens": 1, "output_tokens": 1}))

    assert pn._call(fake, [HumanMessage("hi")]) == "hello"


# --------------------------------------------------------------------------
# The panel's formatting
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, "0"), (999, "999"), (1000, "1.0k"), (55_700, "55.7k"),
    (999_999, "1000.0k"), (1_000_000, "1.00M"), (2_500_000, "2.50M"),
])
def test_token_counts_are_shown_at_the_order_a_person_reads(n, expected):
    from agent.cli.tui import _thousands
    assert _thousands(n) == expected


@pytest.mark.parametrize("full,short", [
    ("us.anthropic.claude-sonnet-4-20250514-v1:0", "claude-sonnet-4"),
    ("openai/gpt-4o-mini", "gpt-4o-mini"),
    ("mercury-coder-small", "mercury-coder-small"),
    ("gemini-2.5-flash", "gemini-2.5-flash"),
    ("", "unknown"),
    ("us.anthropic.claude-3-5-haiku-20241022-v1:0", "claude-3-5-haiku"),
])
def test_a_model_id_is_trimmed_to_the_part_that_distinguishes_it(full, short):
    # The vendor prefix and date stamp are identical on every row, so they
    # cost width and distinguish nothing.
    from agent.cli.tui import _short_model
    assert _short_model(full) == short
