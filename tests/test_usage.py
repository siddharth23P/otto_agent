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


# --------------------------------------------------------------------------
# Pricing (agent/pipeline/pricing.py). Rates are DATA with an expiry date, so
# what is tested here is the machinery around them -- lookup, the override
# file, cache accounting, and the refusal to price what it does not know --
# never the numbers themselves, which are expected to change.
# --------------------------------------------------------------------------

import json

from agent.pipeline import pricing


@pytest.fixture(autouse=True)
def _no_price_file(monkeypatch):
    """Never let a developer's own OTTO_MODEL_PRICES change these results."""
    monkeypatch.delenv(pricing.PRICES_ENV, raising=False)
    pricing.reset_overrides()
    yield
    pricing.reset_overrides()


@pytest.mark.parametrize("model", [
    "claude-haiku-4-5-20251001",
    "us.anthropic.claude-haiku-4-5-20251001",
    "anthropic:claude-haiku-4-5-20251001",
    "claude-haiku-4-5",
])
def test_one_entry_covers_every_spelling_of_the_same_model(model):
    # A dated revision, a region-prefixed id and a vendor-prefixed spec are
    # the same model, and the table should not have to list all of them.
    assert pricing.rate_for(model) is pricing.PRICES["claude-haiku-4-5"]


def test_a_model_with_no_rate_is_not_priced_at_zero():
    # The whole discipline: "no rate" and "free" must not look alike.
    assert pricing.rate_for("mercury-2.5") is None
    assert pricing.cost_of("mercury-2.5", input_tokens=10_000, output_tokens=500) is None


def test_input_and_output_are_charged_at_their_own_rates():
    rate = pricing.Rate(input=2.0, output=10.0)

    assert rate.cost(1_000_000, 0) == pytest.approx(2.0)
    assert rate.cost(0, 1_000_000) == pytest.approx(10.0)


def test_cached_tokens_are_taken_OUT_of_input_before_it_is_charged():
    # langchain's `input_tokens` INCLUDES the cached ones. Charging both the
    # full input and the cache on top double-counts the largest number here.
    rate = pricing.Rate(input=10.0, output=0.0, cached_input=1.0)

    # 1M input of which 900k was a cache read: 100k at 10, 900k at 1.
    assert rate.cost(1_000_000, 0, cached_input_tokens=900_000) == pytest.approx(
        0.1 * 10.0 + 0.9 * 1.0
    )


def test_cache_writes_are_charged_at_their_own_premium():
    rate = pricing.Rate(input=10.0, output=0.0, cached_input=1.0, cache_write=12.5)

    assert rate.cost(1_000_000, 0, cache_write_tokens=1_000_000) == pytest.approx(12.5)


def test_a_model_with_no_cache_rate_charges_cache_reads_as_plain_input():
    # Understating by guessing a discount would be worse than charging full.
    rate = pricing.Rate(input=10.0, output=0.0)

    assert rate.cost(1_000_000, 0, cached_input_tokens=1_000_000) == pytest.approx(10.0)


def test_a_price_file_overrides_the_built_in_rate(tmp_path, monkeypatch):
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"claude-haiku-4-5": {"input": 99.0, "output": 99.0}}))
    monkeypatch.setenv(pricing.PRICES_ENV, str(path))
    pricing.reset_overrides()

    assert pricing.rate_for("claude-haiku-4-5-20251001").input == 99.0


def test_a_price_file_can_price_a_model_the_table_has_never_heard_of(tmp_path, monkeypatch):
    # The point of the override: correcting or ADDING a rate is config, not a
    # code change -- see the module docstring on data with an expiry date.
    path = tmp_path / "prices.json"
    path.write_text(json.dumps({"mercury-2.5": {"input": 1.0, "output": 2.0}}))
    monkeypatch.setenv(pricing.PRICES_ENV, str(path))
    pricing.reset_overrides()

    assert pricing.cost_of("mercury-2.5", input_tokens=1_000_000,
                           output_tokens=0) == pytest.approx(1.0)


def test_a_broken_price_file_is_ignored_rather_than_fatal(tmp_path, monkeypatch):
    # A typo in a config file must not be the reason a turn cannot run.
    path = tmp_path / "prices.json"
    path.write_text("{not json at all")
    monkeypatch.setenv(pricing.PRICES_ENV, str(path))
    pricing.reset_overrides()

    assert pricing.rate_for("claude-haiku-4-5") is pricing.PRICES["claude-haiku-4-5"]


def test_a_missing_price_file_is_ignored_rather_than_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv(pricing.PRICES_ENV, str(tmp_path / "nope.json"))
    pricing.reset_overrides()

    assert pricing.rate_for("gpt-5-mini") is not None


@pytest.mark.parametrize("amount,shown", [
    (None, "--"),
    (0.0, "$0.000"),
    (0.0004, "$0.0004"),
    (0.0912, "$0.091"),
    (1.5, "$1.500"),
    (42.128, "$42.13"),
])
def test_a_cost_is_shown_at_a_precision_that_says_something(amount, shown):
    # Sub-cent turns are normal for the cheap tiers here, so rounding to cents
    # would show "$0.00" for most of a session and then jump.
    assert pricing.format_cost(amount) == shown


# --------------------------------------------------------------------------
# Cost through the ledger
# --------------------------------------------------------------------------

def test_the_ledger_prices_a_model_it_knows():
    led = UsageLedger()
    led.record("gpt-5-mini", {"input_tokens": 1_000_000, "output_tokens": 0})

    assert led.models()[0].cost == pytest.approx(pricing.PRICES["gpt-5-mini"].input)
    assert led.fully_priced is True


def test_the_ledger_records_cache_reads_and_writes_separately():
    led = UsageLedger()
    led.record("claude-haiku-4-5", {
        "input_tokens": 100_000, "output_tokens": 1_000,
        "input_token_details": {"cache_read": 80_000, "cache_creation": 5_000},
    })

    entry = led.models()[0]
    assert entry.input_tokens == 100_000   # the total, cache included
    assert entry.cached_input_tokens == 80_000
    assert entry.cache_write_tokens == 5_000


def test_caching_makes_a_turn_cheaper_rather_than_being_ignored():
    cached, plain = UsageLedger(), UsageLedger()
    usage = {"input_tokens": 500_000, "output_tokens": 1_000}
    plain.record("claude-haiku-4-5", dict(usage))
    cached.record("claude-haiku-4-5",
                  {**usage, "input_token_details": {"cache_read": 450_000}})

    assert cached.cost < plain.cost


def test_a_ledger_with_an_unpriced_model_says_its_total_is_short():
    led = UsageLedger()
    led.record("gpt-5-mini", {"input_tokens": 1000, "output_tokens": 10})
    led.record("mercury-2.5", {"input_tokens": 1000, "output_tokens": 10})

    assert led.fully_priced is False
    assert led.cost > 0  # what IS known is still worth showing
    assert led.snapshot()["fully_priced"] is False


def test_a_model_that_reported_no_tokens_has_no_cost_rather_than_zero():
    led = UsageLedger()
    led.record("gpt-5-mini", None)

    assert led.models()[0].cost is None


def test_an_unpriced_model_that_reported_nothing_does_not_make_a_total_short():
    # `fully_priced` asks about models that REPORTED usage. One that said
    # nothing contributes no tokens, so it cannot be missing from a total.
    led = UsageLedger()
    led.record("gpt-5-mini", {"input_tokens": 100, "output_tokens": 1})
    led.record("quiet-unpriced-model", None)

    assert led.fully_priced is True
