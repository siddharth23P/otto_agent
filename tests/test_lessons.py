"""The lesson bank -- Otto's self-evolution, and the restraints on it.

Every number asserted here is a measured one rather than a taste, and the
tests are written so that loosening one has to be deliberate. The evidence
those numbers come from is in agent/memory/lessons.py's docstring; the short
version is that automatic self-improvement benchmarked against plain repeated
sampling does not consistently win, so the version worth having is small,
capped, and adjudicated at write time.
"""
import numpy as np
import pytest

from agent.memory import lessons as L
from agent.memory.store import MemoryStore


@pytest.fixture
def bank(tmp_path):
    store = MemoryStore(tmp_path / "lessons.db")
    with L.bind_bank(store):
        yield store
    store.close()


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic vectors keyed on the words in a text, so similarity is
    something these tests can control rather than something a model decides."""
    vocabulary: dict[str, int] = {}

    def vector(text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for word in text.lower().split():
            v[vocabulary.setdefault(word, len(vocabulary) % 64)] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    monkeypatch.setattr(L, "embed", lambda texts: [vector(t) for t in texts])
    monkeypatch.setattr(L, "embed_query", vector)
    monkeypatch.setattr(L, "current_model_name", lambda: "fake")
    return vector


def _lesson(cue, action="do the thing", outcome="worked"):
    return L.Lesson(cue=cue, action=action, outcome=outcome)


# --------------------------------------------------------------------------
# The caps
# --------------------------------------------------------------------------

def test_a_run_may_contribute_at_most_three_lessons(bank, fake_embeddings):
    """A run that writes ten has not learned ten things, it has summarised
    itself -- and ten more items in the bank crowd out everything at k=1."""
    kept = L.record_lessons([_lesson(f"situation number {i}") for i in range(10)])

    assert len(kept) == L.MAX_PER_RUN == 3
    assert len(L.all_lessons()) == 3


def test_only_one_lesson_comes_back(bank, fake_embeddings):
    """Task-time procedural recall peaks at k=1 and loses about 7 points by
    k=5: retrieved context competes with the task for attention."""
    L.record_lessons([_lesson("editing a file"), _lesson("editing a config")])

    assert len(L.recall_lessons("editing a file")) == 1


def test_an_off_topic_lesson_is_not_offered_at_all(bank, fake_embeddings):
    """The failure being avoided is always-on injection, which measured 16.4
    points BELOW injecting nothing. Silence is a correct answer."""
    L.record_lessons([_lesson("parsing a calendar invite")])

    assert L.recall_lessons("compile a rust binary and measure its size") == []


def test_an_empty_bank_returns_nothing_rather_than_guessing(bank, fake_embeddings):
    assert L.recall_lessons("anything at all") == []


# --------------------------------------------------------------------------
# Write-time adjudication
# --------------------------------------------------------------------------

def test_the_same_lesson_said_twice_is_stored_once(bank, fake_embeddings):
    """Deciding at READ time whether a memory still applies is a decision
    that, measured, gets made 3.3% of the time. Moving it to the write took
    memory validity from 8.7% to 68.0%."""
    L.record_lessons([_lesson("a path does not exist", "check the working directory")])
    again = L.record_lessons([_lesson("a path does not exist", "check the working directory")])

    assert again == []
    assert len(L.all_lessons()) == 1


def test_a_genuinely_different_lesson_still_gets_in(bank, fake_embeddings):
    L.record_lessons([_lesson("a path does not exist", "check the working directory")])
    kept = L.record_lessons([_lesson("a test hangs", "run it with a timeout")])

    assert len(kept) == 1
    assert len(L.all_lessons()) == 2


def test_a_failed_run_is_learned_from_too(bank, fake_embeddings):
    """The half of the signal that says what NOT to do, and usually the
    sharper half."""
    L.record_lessons([_lesson("the judge rejects an answer", "quote the evidence", "failed")])

    recalled = L.recall_lessons("the judge rejects an answer")
    assert recalled and recalled[0].outcome == "failed"
    assert "[failed]" in recalled[0].rendered()


# --------------------------------------------------------------------------
# The off switch, which the held-out measurement depends on
# --------------------------------------------------------------------------

def test_binding_none_turns_learning_off(tmp_path):
    """Not "use the default" -- OFF. A held-out task set stops being held out
    the moment a run against it writes into the bank."""
    with L.bind_bank(None):
        assert not L.learning_enabled()
        assert L.record_lessons([_lesson("anything")]) == []
        assert L.recall_lessons("anything") == []


def test_a_bound_bank_is_reported_as_available(bank):
    assert L.learning_enabled()


# --------------------------------------------------------------------------
# Degradation
# --------------------------------------------------------------------------

def test_no_embeddings_means_no_recall_rather_than_a_wrong_one(bank, monkeypatch):
    """Most-recent is the wrong fallback here. Unranked means irrelevant, and
    an irrelevant lesson is worse than none -- which is the opposite of the
    call agent/memory/retrieval.py makes for question answering, because that
    read is a different job."""
    monkeypatch.setattr(L, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(L, "embed_query", lambda q: (_ for _ in ()).throw(RuntimeError("down")))
    L.record_lessons([_lesson("something")])

    assert L.recall_lessons("something") == []


def test_learning_continues_while_the_embedder_is_down(bank, monkeypatch):
    """It stores unranked rather than refusing. A bank that silently stops
    growing whenever the embedding service blips would be worse than one with
    a few unrankable rows in it, which become rankable on the next reembed."""
    monkeypatch.setattr(L, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError("down")))

    assert len(L.record_lessons([_lesson("something")])) == 1


# --------------------------------------------------------------------------
# Parsing a distiller's reply
# --------------------------------------------------------------------------

def test_a_fenced_json_reply_still_yields_its_lessons():
    reply = (
        "Here is what I learned:\n```json\n"
        '[{"cue": "a command hangs", "action": "add a timeout", "outcome": "failed"}]'
        "\n```\nHope that helps."
    )
    parsed = L.parse_distilled(reply)

    assert len(parsed) == 1
    assert parsed[0].cue == "a command hangs"
    assert parsed[0].outcome == "failed"


def test_an_unparseable_reply_costs_the_lesson_not_the_run():
    assert L.parse_distilled("I could not think of anything useful.") == []


def test_a_reply_with_more_than_three_is_trimmed():
    items = ",".join(
        f'{{"cue": "c{i}", "action": "a{i}"}}' for i in range(9)
    )
    assert len(L.parse_distilled(f"[{items}]")) == L.MAX_PER_RUN


def test_an_entry_missing_its_cue_or_action_is_dropped():
    reply = '[{"cue": "", "action": "do it"}, {"cue": "when x", "action": ""}, ' \
            '{"cue": "when y", "action": "do y"}]'
    parsed = L.parse_distilled(reply)

    assert [p.cue for p in parsed] == ["when y"]


def test_the_outcome_defaults_to_how_the_run_went():
    """A distiller that forgets the field should not turn a failed run's
    lesson into a recommendation."""
    parsed = L.parse_distilled('[{"cue": "when x", "action": "do y"}]',
                               outcome_default="failed")

    assert parsed[0].outcome == "failed"


def test_a_lesson_survives_a_round_trip_through_storage(bank, fake_embeddings):
    original = _lesson("a container refuses to start", "read the daemon log", "failed")
    L.record_lessons([original])

    assert L.all_lessons() == [original]


def test_read_only_reads_but_does_not_write(bank, fake_embeddings):
    """The held-out combination. Collapsing read and write into one switch
    would mean the honest number could only come from a run that learns
    nothing -- which measures something else, because the question is whether
    the lessons TRANSFER."""
    L.record_lessons([_lesson("a path does not exist", "check the working directory")])

    with L.read_only():
        assert L.recall_lessons("a path does not exist"), "read path was closed too"
        assert not L.learning_enabled()
        assert L.record_lessons([_lesson("something new entirely")]) == []

    assert len(L.all_lessons()) == 1
