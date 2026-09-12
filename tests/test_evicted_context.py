"""The second compaction tier: tool output evicted from a run's transcript,
kept where `recall_memory` can find it again.

`_compact` was one-way until this. A tool result older than the recent tail
was replaced by its one-line summary and the bytes were gone, which is why the
prompt could honestly say "you can run that again" and could not say "you can
search for it". Now it can.

Deliberately no summariser and no bullets on this tier. The result was already
reduced to a line in `actions` when the call ran; summarising it again would
pay a model call to abstract an abstraction, which is the operation that made
one model fail 54% of problems it had previously solved.
"""
import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.memory import retrieval as mr
from agent.memory.session import bind_store
from agent.memory.store import MemoryStore
from agent.pipeline import nodes as pn
from agent.pipeline import tools as pt


@pytest.fixture
def store(tmp_path):
    s = MemoryStore(tmp_path / "session.db")
    with bind_store(s):
        yield s
    s.close()


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Deterministic vectors keyed on words, so ranking is something these
    tests control rather than something a model decides."""
    vocabulary: dict[str, int] = {}

    def vector(text: str) -> np.ndarray:
        v = np.zeros(64, dtype=np.float32)
        for word in text.lower().replace("/", " ").split():
            v[vocabulary.setdefault(word, len(vocabulary) % 64)] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    for module in (pn, mr):
        monkeypatch.setattr(module, "embed", lambda texts: [vector(t) for t in texts],
                            raising=False)
    monkeypatch.setattr(mr, "embed_query", vector)
    monkeypatch.setattr(pn, "embedding_model_name", lambda: "fake")
    monkeypatch.setattr(mr, "current_model_name", lambda: "fake")
    return vector


def _transcript(*results: str) -> list:
    """A seed plus one exchange per result, old enough to be compactable."""
    messages: list = [SystemMessage("prompt"), HumanMessage("TASK: do it")]
    for text in results:
        messages.append(AIMessage("ACTION: execute_bash\nCODE:\nrun it"))
        messages.append(HumanMessage(f"TOOL RESULT:\n{text}"))
    # The tail is protected from compaction, so pad past it.
    for _ in range(pn.KEEP_VERBATIM):
        messages.append(AIMessage("thinking"))
    return messages


LONG = "the deployment key is AK-4417-QX and the region is eu-west-2. " * 20


def test_an_evicted_result_is_still_searchable(store, fake_embeddings):
    messages = _transcript(LONG)
    assert pn._compact(messages) == 1, "nothing was compacted, so nothing was evicted"

    found = mr.recall_chunks(store, mr.EVICTED_KIND, "deployment key region")
    assert "AK-4417-QX" in found


def test_the_transcript_itself_still_shrinks(store, fake_embeddings):
    """Keeping a copy must not stop the eviction doing its job."""
    messages = _transcript(LONG)
    before = pn._transcript_size(messages)
    pn._compact(messages)

    assert pn._transcript_size(messages) < before


def test_recall_memory_reports_both_tiers_separately(store, fake_embeddings):
    """One is what was said, the other is what a command printed. Concatenated
    unlabelled, a tool result reads as something the person told it."""
    pn._compact(_transcript(LONG))

    result = pt.recall_memory("deployment key region")
    assert result.returncode == 0
    assert "FROM EARLIER TOOL OUTPUT IN THIS RUN" in result.stdout
    assert "AK-4417-QX" in result.stdout


def test_nothing_stored_still_reads_as_nothing(store, fake_embeddings):
    result = pt.recall_memory("anything at all")
    assert result.returncode == 0
    assert result.stdout == mr.NOTHING_COMPACTED


def test_a_run_with_no_store_bound_does_not_fail(fake_embeddings):
    """Every run outside a chat session, the whole benchmark harness included.
    Compaction must work there exactly as before."""
    messages = _transcript(LONG)
    assert pn._compact(messages) == 1


def test_a_broken_embedder_does_not_fail_the_run(store, monkeypatch):
    """An unembedded chunk is still recoverable once the store is re-embedded.
    A run must not die over a memory it was only trying to keep."""
    monkeypatch.setattr(pn, "embed",
                        lambda texts: (_ for _ in ()).throw(RuntimeError("down")))

    assert pn._compact(_transcript(LONG)) == 1
    assert store.chunk_hashes(mr.EVICTED_KIND), "the text was dropped, not just unranked"


def test_the_evicted_tier_is_not_summarised_into_bullets(store, fake_embeddings):
    """No model call, and no bullet layer. `recall_chunks` exists precisely so
    this tier needs neither."""
    pn._compact(_transcript(LONG))

    assert store.chunk_hashes(mr.EVICTED_KIND)
    assert store.current_bullets(mr.EVICTED_KIND) == []


def test_the_conversation_tier_is_untouched_by_this(store, fake_embeddings):
    """Two kinds, one store. Evicted tool output must not turn up as something
    the person said."""
    pn._compact(_transcript(LONG))

    assert store.chunk_hashes(mr.HISTORY_KIND) == []


def test_an_identical_result_evicted_twice_is_stored_once(store, fake_embeddings):
    """Content-addressed, like everything else in the store. An agent that
    runs the same command in two runs should not pay for two copies."""
    pn._compact(_transcript(LONG))
    pn._compact(_transcript(LONG))

    assert len(store.chunk_hashes(mr.EVICTED_KIND)) == 1


def test_the_stub_says_the_output_is_searchable_when_it_is(store, fake_embeddings):
    messages = _transcript(LONG)
    pn._compact(messages)

    assert any(pn.EVICTED_HINT.strip() in pn._content_text(m.content) for m in messages)


def test_the_stub_promises_nothing_when_nothing_was_stored(fake_embeddings):
    """No store bound. Telling the model to search for something nothing kept
    would spend a call being told no memory is bound -- and before the second
    tier existed, "run it again" was the correct advice, so it stands here."""
    messages = _transcript(LONG)
    pn._compact(messages)

    assert not any(pn.EVICTED_HINT.strip() in pn._content_text(m.content) for m in messages)


# --------------------------------------------------------------------------
# Type-aware compaction, in the conversation queue
# --------------------------------------------------------------------------

def test_what_the_person_said_is_never_handed_to_the_summariser(tmp_path):
    """Type-blind compaction is destructive in a specific way: it is the
    CONSTRAINTS that go. Constraint recall falls to 53% at 50% compression and
    24% at 10%, where a type-aware policy holds 100/95/80.

    In a conversation the type is visible from the speaker. What the person
    said is the requirement; what Otto said is a report of work, and a report
    can be summarised without losing anything the next turn has to honour.
    """
    from agent.memory.queue import TieredQueue
    from agent.memory.store import MemoryStore

    seen: list[str] = []

    def summarize(prompt: str) -> str:
        seen.append(prompt)
        return "- something happened [1]"

    store = MemoryStore(tmp_path / "q.db")
    queue = TieredQueue(kind="history", store=store, summarize=summarize,
                        x_budget=40, y_budget=120)
    for i in range(60):
        queue.append(f"otto: did some work, step {i}, nothing notable about it")
        if i == 5:
            queue.append("you: never write the deployment key to a log")

    assert seen, "nothing was compacted, so the test proved nothing"
    assert not any("deployment key" in prompt for prompt in seen), (
        "the constraint was handed to the summariser"
    )
    assert "deployment key" in queue.current_view(), (
        "the constraint is not in what a prompt would be built from"
    )
    store.close()


def test_turning_the_protection_off_puts_it_back_in_the_blender(tmp_path):
    """The knob exists so the bench can measure the difference. If this stops
    working, `otto eval-compaction`'s type-blind arms are measuring nothing."""
    from agent.memory.queue import TieredQueue
    from agent.memory.store import MemoryStore

    seen: list[str] = []
    store = MemoryStore(tmp_path / "q.db")
    queue = TieredQueue(kind="history", store=store,
                        summarize=lambda p: seen.append(p) or "- x [1]",
                        x_budget=40, y_budget=120, protect="")
    for i in range(60):
        queue.append(f"otto: did some work, step {i}, nothing notable about it")
        if i == 5:
            queue.append("you: never write the deployment key to a log")

    assert any("deployment key" in prompt for prompt in seen)
    store.close()
