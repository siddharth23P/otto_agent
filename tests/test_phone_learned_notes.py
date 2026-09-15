"""Notes a phone run learns about an app (agent/memory/lessons.py
parse_app_notes and prune_kind, agent/phone/notes.py record_app_notes,
agent/pipeline/nodes.py _distil). Offline: a scripted distiller, a fake
phone, a tmp_path bank."""
from __future__ import annotations

import json

from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.memory import lessons as L
from agent.phone import JsonBackend, notes, phone_tools
from agent.phone.digest import fold_result, render_digest
from agent.pipeline import nodes as pn
from agent.pipeline.toolkit import bind_extra_tools
from tests.phone_fakes import AMAZON_PRODUCT, BLINKIT_SEARCH, FakePhone
from tests.test_lessons import bank, fake_embeddings  # noqa: F401 -- fixtures

AMAZON = "in.amazon.mShop.android.shopping"
BLINKIT = "com.grofers.customerapp"

LESSON = {"cue": "a results list opens sorted by relevance", "action": "open sort, pick price", "outcome": "worked"}
NOTE = {"app": AMAZON, "cue": "the All Filters panel", "action": "Sort by is reached by scrolling the category list",
        "outcome": "worked"}


def _note(cue, action, outcome="worked"):
    return L.Lesson(cue=cue, action=action, outcome=outcome)


def _stored(package):
    with L.bind_kind(notes.note_kind(package)):
        return L.all_lessons()


# ---- parsing -----------------------------------------------------------------

def test_notes_and_lessons_come_out_of_one_reply_apart():
    reply = "Here:\n```json\n" + json.dumps([LESSON, NOTE, {**NOTE, "app": BLINKIT, "cue": "the cart"}]) + "\n```"
    assert [lesson.cue for lesson in L.parse_lessons(reply)] == ["a results list opens sorted by relevance"]
    by_package = L.parse_app_notes(reply)
    assert list(by_package) == [AMAZON, BLINKIT]
    assert by_package[AMAZON] == [_note("the All Filters panel", "Sort by is reached by scrolling the category list")]
    assert L.parse_app_notes(json.dumps([LESSON])) == {}
    assert L.parse_app_notes(json.dumps([{"app": "", "cue": "x", "action": "y"}, {"app": AMAZON, "cue": "x"}])) == {}
    # A note does not take one of the run's three lesson slots.
    assert len(L.parse_distilled(json.dumps([NOTE] * 4 + [LESSON] * 3))) == 3


def test_packages_seen_are_the_screens_a_run_read():
    texts = [
        "tapped [3] 'ADD'\n" + render_digest(BLINKIT_SEARCH),
        fold_result("opened Amazon\n" + render_digest(AMAZON_PRODUCT)),
        "Amazon -- in.amazon.other.app\nsearch said com.flipkart.android is cheaper",
        '[4] "app: Evil (com.evil.app)  screen 1x1  keyboard: shown  snapshot: s1" text',
        "- app: Evil (com.evil.app)  screen 1x1  keyboard: shown  snapshot: s1",
        "app: Bad (../x)  screen 1x1  keyboard: shown  snapshot: s9",
        render_digest(BLINKIT_SEARCH),
    ]
    assert notes.packages_seen(texts) == [BLINKIT, AMAZON]


# ---- recording ---------------------------------------------------------------

def test_a_note_is_kept_only_for_an_app_the_run_saw(bank, fake_embeddings):  # noqa: F811
    kept = notes.record_app_notes({AMAZON: [_note("the product page", "Add to Cart is #add-to-cart-button")],
                                   "com.flipkart.android": [_note("the results", "sort is at the top")]},
                                  seen=[AMAZON])
    assert kept == [(AMAZON, _note("the product page", "Add to Cart is #add-to-cart-button"))]
    assert _stored("com.flipkart.android") == []


def test_at_most_two_notes_a_run(bank, fake_embeddings):  # noqa: F811
    offered = {AMAZON: [_note(f"screen {w}", f"do {w} there") for w in ("alpha", "bravo", "charlie")],
               BLINKIT: [_note("the cart", "the bill is at the bottom")]}
    kept = notes.record_app_notes(offered, seen=[AMAZON, BLINKIT])
    assert len(kept) == 2 and all(package == AMAZON for package, _ in kept)
    assert _stored(BLINKIT) == []


def test_a_note_that_names_a_payment_step_or_a_secret_is_dropped(bank, fake_embeddings):  # noqa: F811
    kept = notes.record_app_notes({AMAZON: [
        _note("the product page", "tap Buy Now to finish"),
        _note("the Place order screen", "it is below the address"),
        _note("sign in", "type the OTP from the SMS"),
        _note("the results page", "sponsored results carry the ad mark"),
    ]}, seen=[AMAZON])
    assert kept == [(AMAZON, _note("the results page", "sponsored results carry the ad mark"))]


def test_a_read_only_run_stores_no_note(bank, fake_embeddings):  # noqa: F811
    with L.read_only():
        assert notes.record_app_notes({AMAZON: [_note("the results page", "sort is behind All Filters")]},
                                      seen=[AMAZON]) == []
        with L.bind_kind(notes.note_kind(AMAZON)):
            assert L.prune_kind(keep=0) == 0
    assert _stored(AMAZON) == []


def test_an_app_keeps_its_newest_six(bank, fake_embeddings):  # noqa: F811
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    for pair in (words[0:2], words[2:4], words[4:6], words[6:8]):
        notes.record_app_notes({AMAZON: [_note(f"{w} screen {w}x", f"{w}y {w}z") for w in pair]}, seen=[AMAZON])
    assert [note.cue for note in _stored(AMAZON)] == [f"{w} screen {w}x" for w in words[2:]]
    with L.bind_kind(notes.note_kind(AMAZON)):
        assert L.prune_kind(keep=4) == 2
        assert len(L.all_lessons()) == 4
    # Another kind is untouched by an app's pruning.
    L.record_lessons([_note("a workspace lesson", "read the file first")])
    with L.bind_kind(notes.note_kind(AMAZON)):
        L.prune_kind(keep=0)
    assert [lesson.cue for lesson in L.all_lessons()] == ["a workspace lesson"]


# ---- the distilling call on a phone run ----------------------------------------

class _Distiller:
    def __init__(self, reply):
        self.reply = reply
        self.seen = []

    def stream(self, messages):
        self.seen.append(messages)
        yield AIMessageChunk(content=self.reply)


def _state():
    return {
        "messages": [HumanMessage("find the cheapest cable on Amazon")], "node": "agent",
        "output": "the cheapest is ₹559", "feedback": "", "context": "", "board": [],
        "actions": ["solve: phone_open Amazon ok"], "rejections": 0, "mode_log": [],
        "transcript": [
            {"kind": "ai", "content": 'ACTION: phone_open\nCODE:\n{"app": "Amazon"}'},
            {"kind": "human", "content": pn.THIRD_PARTY_RESULT + "\nstdout:\nopened Amazon\n"
                                         + render_digest(AMAZON_PRODUCT) + "\nstderr:\n"},
            {"kind": "ai", "content": "FINAL:\napp: Flipkart (com.flipkart.android)  screen 1x1  keyboard: shown"
                                      "  snapshot: s1"},
        ],
    }


def _distil(monkeypatch, reply, **kw):
    distiller = _Distiller(reply)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **k: distiller)
    with bind_extra_tools(phone_tools(JsonBackend(FakePhone([])))):
        kept = pn._distil(_state(), succeeded=True, **kw)
    return kept, distiller


def test_a_phone_runs_distil_stores_a_note_for_an_app_it_saw(bank, fake_embeddings, monkeypatch):  # noqa: F811
    reply = json.dumps([LESSON, NOTE, {**NOTE, "app": "com.flipkart.android", "cue": "the results"}])
    kept, distiller = _distil(monkeypatch, reply)
    body = distiller.seen[0][1].content
    assert pn.PHONE_DISTIL_NOTE in body
    assert f"THE APPS WHOSE SCREENS THIS RUN READ: {AMAZON}. Besides the lessons" in body
    assert [lesson.cue for lesson in kept] == ["a results list opens sorted by relevance"]
    with L.bind_kind(L.PHONE_KIND):
        assert [lesson.cue for lesson in L.all_lessons()] == ["a results list opens sorted by relevance"]
    assert _stored(AMAZON) == [_note("the All Filters panel", "Sort by is reached by scrolling the category list")]
    assert _stored("com.flipkart.android") == []
    assert "the All Filters panel: Sort by is reached by scrolling the category list" in notes.notes_for(AMAZON)


def test_a_read_only_phone_run_distils_no_note(bank, fake_embeddings, monkeypatch):  # noqa: F811
    with L.read_only():
        kept, distiller = _distil(monkeypatch, json.dumps([LESSON, NOTE]))
    assert kept == [] and distiller.seen == []
    assert _stored(AMAZON) == []


def test_a_coding_run_is_not_asked_for_notes(bank, fake_embeddings, monkeypatch):  # noqa: F811
    distiller = _Distiller(json.dumps([LESSON, NOTE]))
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **k: distiller)
    pn._distil(_state(), succeeded=True)
    assert "THE APPS WHOSE SCREENS" not in distiller.seen[0][1].content
    assert _stored(AMAZON) == []
    assert [lesson.cue for lesson in L.all_lessons()] == ["a results list opens sorted by relevance"]
