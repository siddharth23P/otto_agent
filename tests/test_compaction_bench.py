"""The compaction bench itself: does it measure what it claims to.

An instrument that scores every policy the same is worse than none, because
it licenses whatever policy is currently shipped. These tests hold the two
properties that make the numbers mean something -- it can tell policies apart,
and a constraint it says survived really is in the text.
"""
import tempfile
from pathlib import Path

import pytest

from agent.eval import compaction_bench as cb


@pytest.fixture
def root(tmp_path):
    return tmp_path


def test_the_constraints_are_actually_planted():
    conversation = cb.build_conversation(turns=120)
    text = "\n".join(conversation)

    for constraint in cb.CONSTRAINTS:
        assert constraint in text


def test_they_are_spread_out_rather_than_clustered():
    """Position is the thing being tested. A policy that keeps a verbatim tail
    scores perfectly on the last few and badly on the first, and clustered
    constraints would hide that in the mean."""
    conversation = cb.build_conversation(turns=120)
    positions = [i for i, item in enumerate(conversation)
                 if any(c in item for c in cb.CONSTRAINTS)]

    assert len(positions) == len(cb.CONSTRAINTS)
    assert max(positions) - min(positions) > len(conversation) // 2


def test_the_stand_in_summariser_is_lossy_and_cites():
    """It has to be both. A summariser that lost nothing would score every
    policy at 100%; one that cited nothing would break the queue's own
    carry-forward and score every policy at zero."""
    summarize = cb.keep_fraction_summarizer(0.35)
    reply = summarize("1. the deployment key is AK-4417-QX and must never be logged\n"
                      "2. ran the tests, all green and nothing surprising in the output")

    assert "[1]" in reply and "[2]" in reply
    assert "must never be logged" not in reply, "nothing was lost, so nothing is measured"


def test_the_bench_can_tell_two_policies_apart(root):
    """The instrument's whole job. If everything scores the same it is
    measuring nothing and every later comparison is worthless.

    Compared type-BLIND on both sides, because that is where the budget still
    decides the answer -- with protection on, constraint survival stops
    depending on the budget at all, which is the finding rather than a flaw in
    the bench."""
    conversation = cb.build_conversation(turns=120)
    roomy = cb.run_policy("roomy", {"x_budget": 1200, "y_budget": 4000, "protect": ""},
                          root=root, conversation=conversation)
    cramped = cb.run_policy("cramped", {"x_budget": 150, "y_budget": 400, "protect": ""},
                            root=root, conversation=conversation)

    assert roomy.survives > cramped.survives
    assert cramped.compactions > roomy.compactions


def test_protecting_what_the_person_said_beats_not_protecting_it(root):
    """The measurement the bench was built to make. Same budgets, one knob."""
    conversation = cb.build_conversation(turns=120)
    protected = cb.run_policy("on", {"x_budget": 400, "y_budget": 1200},
                              root=root, conversation=conversation)
    blind = cb.run_policy("off", {"x_budget": 400, "y_budget": 1200, "protect": ""},
                          root=root, conversation=conversation)

    assert protected.survives > blind.survives, (
        f"type-aware {protected.survives}/{protected.total} did not beat "
        f"type-blind {blind.survives}/{blind.total}"
    )


def test_a_constraint_counted_as_surviving_is_really_there(root):
    """Guards against the signature match drifting loose enough to score
    noise as a pass."""
    conversation = cb.build_conversation(turns=120)
    result = cb.run_policy("roomy", cb.POLICIES["roomy"], root=root,
                           conversation=conversation)

    assert result.survives == result.total
    assert result.missing == []


def test_everything_stays_recoverable_even_when_the_view_loses_it(root):
    """The chunk store is the safety net, and the gap between the two columns
    is the honest cost of compaction: text the model will not see unless it
    thinks to go looking. Measured on a type-blind policy, which is the only
    way to produce that gap now."""
    conversation = cb.build_conversation(turns=120)
    result = cb.run_policy("blind", {"x_budget": 150, "y_budget": 400, "protect": ""},
                           root=root, conversation=conversation)

    assert result.recoverable > result.survives, (
        "nothing the view lost could be found again, so the chunk store is "
        "not acting as the safety net the design claims"
    )
    # NOT `== total`. Under the test suite the embedding backend is
    # unavailable (conftest strips the hosted spec and there are no real
    # keys), so recall degrades to unranked most-recent exactly as
    # agent/memory/retrieval.py documents -- 2 of 8 rather than 8 of 8 with a
    # working embedder. Asserting the full count here would be asserting that
    # a developer machine has a model downloaded.
