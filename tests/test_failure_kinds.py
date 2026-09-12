"""Reading the SHAPE of a failure out of what a run already wrote.

A score says a run got worse. It does not say what broke, and a regression
nobody can attribute is one nobody can scope a repair to: an injected fault was
localised 64.8% of the time from a structured trace and 13.0% of the time from
the task outcome alone.

Free by construction -- every signal is already in the action record, the
checklist and the answer. No model call, nothing new recorded.
"""
from agent.eval.failure_kinds import CASCADE_RUN, DESCRIPTIONS, classify, summarise


def test_a_clean_run_carries_no_kinds():
    """The taxonomy has to be silent on a good run, or it is noise on every
    line of every report."""
    assert classify(
        actions=["solve: write_file a.py -> ok", "solve: execute_python -> ok: 4"],
        answer="the answer is 4", resolved=True,
    ) == []


def test_a_run_that_changed_nothing_says_so():
    """Recuris' zero-write episode. The base agent it measured ended 40%+ of
    write-requiring episodes having executed none."""
    kinds = classify(actions=["solve: read_file a.py -> ok"], answer="looks fine",
                     resolved=False)

    assert "no_writes" in kinds
    assert "read_only" in kinds


def test_taking_no_action_at_all_is_its_own_shape():
    """Distinct from `no_writes`: one looked and did not act, the other never
    started. They point at different bugs."""
    kinds = classify(actions=[], answer="here you go", resolved=False)

    assert "no_actions" in kinds
    assert "no_writes" not in kinds


def test_a_run_of_failing_calls_is_a_cascade():
    kinds = classify(
        actions=["s: execute_bash x -> FAILED (exit 1): nope"] * CASCADE_RUN,
        answer="", resolved=False,
    )

    assert "error_cascade" in kinds


def test_two_failures_are_a_coincidence_not_a_cascade():
    """Same reasoning as the provider breaker: one is noise, two is a
    coincidence, three is a pattern."""
    kinds = classify(
        actions=["s: execute_bash x -> FAILED (exit 1): nope"] * (CASCADE_RUN - 1),
        answer="gave up", resolved=False,
    )

    assert "error_cascade" not in kinds


def test_failures_have_to_be_consecutive_to_cascade():
    """A run that recovers between failures is not cascading, it is working."""
    kinds = classify(
        actions=["s: execute_bash a -> FAILED (exit 1): x",
                 "s: execute_bash b -> ok",
                 "s: execute_bash c -> FAILED (exit 1): x",
                 "s: execute_bash d -> ok",
                 "s: execute_bash e -> FAILED (exit 1): x"],
        answer="done", resolved=True,
    )

    assert "error_cascade" not in kinds


def test_a_failing_first_call_is_recorded_separately():
    """The first call is where a run commits to an approach, which is why it
    is its own mode rather than part of whatever follows."""
    kinds = classify(actions=["s: execute_bash x -> FAILED (exit 1): no",
                              "s: write_file a.py -> ok"],
                     answer="done", resolved=True)

    assert "wrong_first_call" in kinds


def test_answering_with_criteria_still_open_is_flagged():
    """Hallucinated completion -- the most expensive shape, because it reads
    as success everywhere except the grader."""
    kinds = classify(
        actions=["s: write_file report.md -> ok"],
        checklist=[{"status": "pending", "text": "the totals are checked"}],
        answer="All done, totals verified.", resolved=False,
    )

    assert "answered_with_criteria_open" in kinds


def test_a_resolved_run_is_not_accused_of_hallucinating():
    """The harness's own verdict wins. A passing run with a stale checklist
    entry is a checklist problem, not a lie."""
    kinds = classify(
        actions=["s: write_file report.md -> ok"],
        checklist=[{"status": "pending", "text": "the totals are checked"}],
        answer="All done.", resolved=True,
    )

    assert "answered_with_criteria_open" not in kinds


def test_a_blocked_criterion_is_not_the_agents_fault():
    kinds = classify(
        actions=["s: execute_bash curl -> FAILED (exit 7): refused"],
        checklist=[{"status": "blocked", "text": "the API responds"}],
        answer="the endpoint is unreachable", resolved=False,
    )

    assert "blocked_by_environment" in kinds


def test_producing_no_answer_is_recorded():
    assert "no_answer" in classify(actions=["s: read_file a.py -> ok"],
                                   answer="   ", resolved=False)


def test_a_run_can_carry_several_kinds():
    """Forcing one label would throw away the combination, which is usually
    the informative part."""
    kinds = classify(
        actions=["s: execute_bash x -> FAILED (exit 1): no"] * CASCADE_RUN,
        answer="", resolved=False,
    )

    assert len(kinds) > 1


def test_a_batch_is_summarised_most_common_first():
    """The number a comparison is actually read from."""
    counts = summarise([
        ["no_writes", "read_only"],
        ["no_writes"],
        ["error_cascade"],
    ])

    assert list(counts) == ["no_writes", "error_cascade", "read_only"]
    assert counts["no_writes"] == 2


def test_every_kind_has_a_sentence_a_person_can_read():
    """A report of bare slugs is a report nobody reads."""
    produced = set()
    for kw in (
        dict(actions=[], answer="", resolved=False),
        dict(actions=["s: read_file a -> ok"], answer="x", resolved=False),
        dict(actions=["s: execute_bash x -> FAILED (exit 1): n"] * CASCADE_RUN,
             answer="", resolved=False),
        dict(actions=["s: write_file a -> ok"], answer="done",
             checklist=[{"status": "pending", "text": "t"}], resolved=False),
        dict(actions=["s: write_file a -> ok"], answer="done",
             checklist=[{"status": "blocked", "text": "t"}], resolved=False),
    ):
        produced.update(classify(**kw))

    assert produced, "the fixtures produced no kinds at all"
    assert produced <= set(DESCRIPTIONS), produced - set(DESCRIPTIONS)


def test_an_unrecorded_action_list_claims_nothing():
    """None means nobody recorded the actions; [] means the run took none.
    Collapsing the two tagged every report written before this existed as
    having done nothing -- caught by reading it back against reports already
    on disk, not by review."""
    assert classify(actions=None, answer="the answer", resolved=True) == []
    assert classify(actions=None, answer="the answer", resolved=False) == []


def test_a_recorded_empty_list_still_says_no_actions():
    assert "no_actions" in classify(actions=[], answer="x", resolved=False)


# --------------------------------------------------------------------------
# What the tool menu costs
# --------------------------------------------------------------------------
#
# Otto had six tools and has seventeen, and the menu is restated on every call
# -- 27% of the system prompt before a word of the task. Tool OVERUSE is a
# measured cost, and whether an agent manages its tool context at all depends
# on model strength. Nothing here argues for removing a tool; it argues for
# having the number before the menu grows again.

def test_tool_calls_are_counted_most_used_first():
    from agent.eval.failure_kinds import tool_usage

    counts = tool_usage(["solve: read_file a.py -> ok",
                         "solve: read_file b.py -> ok",
                         "solve: write_file a.py -> ok"])

    assert list(counts) == ["read_file", "write_file"]
    assert counts["read_file"] == 2


def test_calls_are_also_broken_down_by_seat():
    """Per mode is per MODEL, which is the breakdown that says whether a
    weaker seat is flailing with a menu it cannot hold."""
    from agent.eval.failure_kinds import usage_by_seat

    counts = usage_by_seat(["solve: read_file a -> ok", "solve: write_file a -> ok",
                            "plan: execute_bash ls -> ok"])

    assert counts == {"solve": 2, "plan": 1}


def test_a_batch_totals_across_runs():
    from agent.eval.failure_kinds import merge_usage, tool_usage

    one = tool_usage(["solve: read_file a -> ok"])
    two = tool_usage(["solve: read_file b -> ok", "solve: rag x -> ok"])

    assert merge_usage([one, two]) == {"read_file": 2, "rag": 1}


def test_an_unrecorded_run_counts_nothing():
    from agent.eval.failure_kinds import tool_usage, usage_by_seat

    assert tool_usage(None) == {}
    assert usage_by_seat(None) == {}
