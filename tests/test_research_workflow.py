"""The research workflow: outline, sections with a ledger, checks, assembly.

The model is scripted; the files are real (a throwaway workspace). Counts of
model calls are exact on purpose -- the workflow's whole claim is that it
spends a known number of calls for a known amount of document.
"""

import json
import uuid

import pytest
from langchain_core.messages import AIMessageChunk, HumanMessage

from agent.memory.lessons import bind_bank
from agent.pipeline import nodes as pn
from agent.pipeline import research as rs
from agent.pipeline.budget import RECON_NOTE, Budget, bind_budget
from agent.pipeline.run import _initial
from agent.pipeline.workspace import bind_workspace


class _Scripted:
    def __init__(self, replies):
        self._replies = list(replies)
        self.seen: list[list] = []
        self.calls = 0

    def stream(self, messages):
        self.calls += 1
        self.seen.append(list(messages))
        reply = self._replies.pop(0) if self._replies else "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: ok"
        yield AIMessageChunk(content=reply)


@pytest.fixture
def small(monkeypatch):
    """Sections short enough to script by hand."""
    monkeypatch.setattr(rs, "MIN_SECTION_WORDS", 5)


def _install(monkeypatch, replies):
    fake = _Scripted(replies)
    monkeypatch.setattr(pn.ROUTER, "chat_model", lambda *a, **kw: fake)
    return fake


def _state(task="write about the salt empire", **overrides) -> dict:
    base = {**_initial(task), "checklist": [
        {"text": "three sections", "status": "pending", "evidence": ""},
    ], "route": "research"}
    base.update(overrides)
    return base


def _outline(n=3, *, min_words=20, parts=("Setting", "Event"), checks=(), questions=(),
             fmt="md", ledger=None):
    return json.dumps({
        "title": "The Salt Empire", "abstract": "A history in parts.", "format": fmt,
        "ledger": ledger or {"entities": {"Ilyra": "queen"}, "facts": [], "open_threads": []},
        "sections": [{
            "title": f"Generation {i}", "brief": f"what happened in generation {i}",
            "min_words": min_words, "required_parts": list(parts),
            "checks": list(checks), "research_questions": list(questions),
        } for i in range(1, n + 1)],
    })


def _prose(n):
    return " ".join(f"w{i}" for i in range(n)) + "."


def _section(i, *, words=30, parts=("Setting", "Event"), fact=None, ledger=True,
             heading=True, extra=""):
    body = f"## Generation {i}\n\n" if heading else ""
    for part in parts:
        body += f"### {part}\n\n{_prose(words // max(len(parts), 1))}\n\n"
    body += extra
    if ledger:
        state = {"entities": {"Ilyra": "queen"}, "facts": [fact] if fact else [],
                 "open_threads": []}
        body += "\n```ledger\n" + json.dumps(state) + "\n```\n"
    return body


def _run_graph(monkeypatch, replies, tmp_path, task="write about the salt empire",
               budget=None):
    fake = _install(monkeypatch, replies)
    config = {"configurable": {"thread_id": f"test:{uuid.uuid4().hex}"},
              "recursion_limit": pn._RECURSION_SAFETY_NET}
    with bind_workspace(str(tmp_path)), bind_budget(budget or Budget(max_model_calls=60)), \
            bind_bank(None):
        final = pn.app.invoke(_initial(task), config)
    return fake, final


def _human(messages) -> str:
    return "\n".join(m.content for m in messages if isinstance(m, HumanMessage))


# --------------------------------------------------------------------------
# outline
# --------------------------------------------------------------------------

def test_outline_json_parses_and_clamps():
    data = json.loads(_outline(3, min_words=9000))
    data["sections"][1]["title"] = data["sections"][0]["title"]  # a duplicate
    outline = rs._parse_outline("```json\n" + json.dumps(data) + "\n```")

    assert outline.title == "The Salt Empire"
    assert [s.min_words for s in outline.sections] == [rs.MAX_SECTION_WORDS] * 3
    assert len({s.slug for s in outline.sections}) == 3
    assert outline.sections[0].path == "sections/01-generation-1.md"
    assert outline.ledger["entities"] == {"Ilyra": "queen"}


def test_an_outline_with_no_sections_is_refused():
    with pytest.raises(ValueError):
        rs._parse_outline('{"title": "x", "sections": []}')


def test_a_bad_outline_is_retried_once_then_falls_back_to_agent(monkeypatch, tmp_path):
    fake = _install(monkeypatch, ["not json", "still not json"])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state())

    assert result.goto == "agent"
    assert result.update["route"] == "agent"
    assert fake.calls == 2
    assert "not one usable JSON object" in _human(fake.seen[1])


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------

def test_a_section_is_written_from_the_plain_reply_and_the_ledger_is_split(
        monkeypatch, tmp_path, small):
    """No ACTION protocol for writers: a line that starts with ACTION: is
    prose here, and the ledger block never reaches the file."""
    fake = _install(monkeypatch, [
        _outline(1),
        _section(1, words=40, fact="salt is taxed", extra="ACTION: execute_bash is a phrase.\n\n"),
    ])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state())

    written = (tmp_path / "otto_research/write-about-the-salt-empire/sections/01-generation-1.md").read_text()
    assert "ACTION: execute_bash is a phrase." in written
    assert "```ledger" not in written
    assert result.goto == "evaluator"
    ledger = json.loads((tmp_path / "otto_research/write-about-the-salt-empire/ledger.json").read_text())
    assert ledger["facts"] == ["salt is taxed"]
    assert fake.calls == 2  # outline, one write -- no checks asked, none paid for


def test_a_short_section_gets_one_revision_and_the_better_draft_wins(monkeypatch, tmp_path, small):
    fake = _install(monkeypatch, [
        _outline(1, min_words=50),
        _section(1, words=20),
        _section(1, words=60),
    ])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state())

    assert fake.calls == 3
    assert "IT FAILS:" in _human(fake.seen[2])
    assert "the minimum is 50" in _human(fake.seen[2])
    written = (tmp_path / "otto_research/write-about-the-salt-empire/sections/01-generation-1.md").read_text()
    assert rs._words(written) >= 50
    assert "revised once" in result.update["output"]


def test_a_missing_required_part_fails_the_code_check():
    section = rs.Section(1, "g", "G", "", 5, ("Setting", "Event"), (), ())
    check = rs._check_section(section, "## G\n\n### Setting\n\nsome words here now.\n", had_ledger=True)

    assert check.missing_parts == ["Event"]
    assert any("### Event" in f for f in check.failures())


def test_a_tool_call_reply_is_a_failure_not_a_section():
    section = rs.Section(1, "g", "G", "", 5, (), (), ())
    check = rs._check_section(section, "ACTION: write_file\nCODE:\nx.md\nhello there.", had_ledger=False)

    assert check.leaked_protocol


def test_a_cut_off_section_is_noticed():
    section = rs.Section(1, "g", "G", "", 5, (), (), ())
    assert rs._check_section(section, "## G\n\nthe king rode out and", had_ledger=False).truncated
    assert not rs._check_section(section, "## G\n\nthe king rode out.", had_ledger=False).truncated
    assert not rs._check_section(section, "## G\n\nthe king rode out and", had_ledger=True).truncated


def test_the_ledger_carries_from_section_one_to_section_three(monkeypatch, tmp_path, small):
    fake = _install(monkeypatch, [
        _outline(3),
        _section(1, fact="the Edict of Salt was passed"),
        _section(2, fact="the Edict of Salt was passed"),
        _section(3),
    ])

    with bind_workspace(str(tmp_path)):
        rs.run_research(_state())

    third = _human(fake.seen[3])
    assert "the Edict of Salt was passed" in third
    assert "THE PREVIOUS SECTION ENDS" in third
    assert "## Generation 2" in third or "Generation 2" in third
    assert "<- this one" in third


def test_a_missing_ledger_block_keeps_the_previous_ledger(monkeypatch, tmp_path, small):
    fake = _install(monkeypatch, [
        _outline(3),
        _section(1, fact="the Edict of Salt was passed"),
        _section(2, ledger=False),
        _section(3),
    ])

    with bind_workspace(str(tmp_path)):
        rs.run_research(_state())

    assert "the Edict of Salt was passed" in _human(fake.seen[3])


def test_the_ledger_is_a_union_and_the_founding_facts_survive_trimming(monkeypatch):
    """A writer that rewrites the ledger drops what its section did not
    touch. The founding charter must not disappear by section ten."""
    old = {"entities": {"Charter": "passed in year 3", "Ilyra": "queen"},
           "facts": ["founding fact"], "open_threads": ["succession"]}
    new = {"entities": {"Ilyra": "dead", "Marek": "king"},
           "facts": ["middle fact " * 8, "latest fact"], "open_threads": []}

    merged = rs._merge_ledger(old, new)

    assert merged["entities"] == {"Charter": "passed in year 3", "Ilyra": "dead", "Marek": "king"}
    assert merged["facts"][0] == "founding fact" and merged["facts"][-1] == "latest fact"
    assert merged["open_threads"] == [], "closing a thread is the writer's call"

    monkeypatch.setattr(rs, "LEDGER_MAX_CHARS", 160)
    trimmed = rs._merge_ledger(old, new)
    assert trimmed["facts"] == ["founding fact", "latest fact"]


# --------------------------------------------------------------------------
# assembly and conversion
# --------------------------------------------------------------------------

def test_assembly_has_title_contents_and_every_section_in_order():
    outline = rs._parse_outline(_outline(3))
    written = [(s, f"## {s.title}\n\n{_prose(10)}") for s in outline.sections]

    doc = rs._assemble(outline, written, incomplete=[])

    assert doc.startswith("# The Salt Empire\n")
    assert "## Contents" in doc
    assert "1. [Generation 1](#generation-1)" in doc
    assert doc.index("## Generation 1") < doc.index("## Generation 2") < doc.index("## Generation 3")
    assert "3 sections, 30 words." in doc


def test_conversion_runs_when_docx_is_asked_for(monkeypatch, tmp_path, small):
    _install(monkeypatch, [_outline(1, fmt="docx"), _section(1)])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state())

    target = tmp_path / "otto_research/write-about-the-salt-empire/document.docx"
    assert target.exists()
    assert "Also `document.docx`" in result.update["output"]
    from docx import Document
    assert any("Generation 1" in p.text for p in Document(str(target)).paragraphs)


def test_a_failed_conversion_keeps_the_markdown(monkeypatch, tmp_path, small):
    monkeypatch.setitem(rs.CONVERT_SCRIPTS, "docx", "import sys\nprint('nope')\nsys.exit(1)\n")
    _install(monkeypatch, [_outline(1), _section(1)])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state(task="write about the salt empire as a word doc"))

    assert (tmp_path / "otto_research/write-about-the-salt-empire-as/document.md").exists()
    assert "Conversion to docx failed" in result.update["output"]
    assert result.goto == "evaluator"


def test_the_format_is_read_from_the_request_when_the_outline_says_md():
    outline = rs._parse_outline(_outline(1))
    assert rs._wanted_format("send it as a PDF please", outline) == "pdf"
    assert rs._wanted_format("an excel sheet of it", outline) == "xlsx"
    assert rs._wanted_format("just write it", outline) is None


# --------------------------------------------------------------------------
# budget
# --------------------------------------------------------------------------

def test_running_out_of_budget_assembles_what_exists_and_ends(monkeypatch, tmp_path, small):
    """rubric (1), outline (2), section 1 (3), section 2 (4): spent. Section
    3 is named as missing, and no judgment is attempted."""
    fake, final = _run_graph(monkeypatch, [
        "KIND: research\n- three sections",
        _outline(3),
        _section(1), _section(2), _section(3),
    ], tmp_path, budget=Budget(max_model_calls=4))

    assert fake.calls == 4
    assert final["route"] == "research"
    assert "INCOMPLETE" in final["final_output"]
    assert "3. Generation 3" in final["final_output"]
    doc = (tmp_path / "otto_research/write-about-the-salt-empire/document.md").read_text()
    assert "## Generation 2" in doc and "## Generation 3" not in doc
    assert final["model_calls"] == 4


def test_the_wrap_up_stretch_skips_gathering_and_checks(monkeypatch, tmp_path, small):
    budget = Budget(max_model_calls=100)
    for _ in range(85):
        budget.spend()
    fake = _install(monkeypatch, [
        _outline(2, checks=("names the queen",), questions=("who was queen?",)),
        _section(1), _section(2),
    ])

    with bind_workspace(str(tmp_path)), bind_budget(budget):
        result = rs.run_research(_state())

    assert fake.calls == 3, "outline plus one write per section, nothing else"
    assert result.goto == "evaluator"


def test_the_lean_tier_still_revises_a_section_the_code_checks_fail(monkeypatch, tmp_path, small):
    """Six calls left for three sections after the outline: no gathering, no
    model checks, but a section that misses its minimum is still given its
    one revision. (The tier is re-read before every section, so a budget
    that frees up climbs back to full -- which is why the number is nine
    and not twelve.)"""
    budget = Budget(max_model_calls=9)
    fake = _install(monkeypatch, [
        _outline(3, min_words=50, checks=("names the queen",)),
        _section(1, words=20), _section(1, words=60),
        _section(2, words=60), _section(3, words=60),
    ])

    with bind_workspace(str(tmp_path)), bind_budget(budget):
        rs.run_research(_state())

    assert fake.calls == 5


# --------------------------------------------------------------------------
# gathering workers
# --------------------------------------------------------------------------

def test_a_gathering_worker_is_bounded_and_its_notes_reach_the_writer(monkeypatch, tmp_path, small):
    notes = "otto_research/write-about-the-salt-empire/notes/01-generation-1.md"
    fake = _install(monkeypatch, [
        _outline(1, questions=("what did the salt tax fund?",)),
        f"ACTION: write_file\nCODE:\n{notes}\nfound: the tax funded the roads",
        "FINAL:\nthe tax funded the roads",
        _section(1),
    ])

    with bind_workspace(str(tmp_path)):
        result = rs.run_research(_state())

    assert fake.calls == 4
    worker_first = fake.seen[1]
    assert "delegate" not in worker_first[0].content
    assert worker_first[-1].content.startswith("MODE: find")
    assert "what did the salt tax fund?" in _human(worker_first)
    writer = _human(fake.seen[3])
    assert "found: the tax funded the roads" in writer
    assert "NOTES GATHERED" in writer
    assert any("find(worker)" in line for line in result.update["actions"])


def test_the_recon_note_never_reaches_a_worker(monkeypatch, tmp_path, small):
    budget = Budget(max_model_calls=20)
    for _ in range(5):
        budget.spend()  # past the reconnaissance stretch, note not yet said
    fake = _install(monkeypatch, [
        _outline(1, questions=("who was queen?",)),
        "FINAL:\nIlyra",
        _section(1),
    ])

    with bind_workspace(str(tmp_path)), bind_budget(budget):
        rs.run_research(_state())

    for call in fake.seen:
        assert RECON_NOTE not in _human(call)


# --------------------------------------------------------------------------
# end to end through the graph
# --------------------------------------------------------------------------

def test_the_motivating_shape_end_to_end(monkeypatch, tmp_path, small):
    """Six calls: rubric, outline, three sections, one judgment. No checks
    were asked for so none were paid for, and a document run distils no
    lesson."""
    fake, final = _run_graph(monkeypatch, [
        "KIND: research\n- three generations, each written out\n- each part at least 20 words",
        _outline(3),
        _section(1, fact="the Edict of Salt was passed"),
        _section(2, fact="the Edict of Salt was passed"),
        _section(3),
        "FINAL:\nMET: 2/2\nBLOCKED: no\nAPPROVE: yes\nWHY: every section is there",
    ], tmp_path)

    assert fake.calls == 6
    assert final["route"] == "research"
    assert final["final_output"].startswith("Document written to `otto_research/")
    assert (tmp_path / final["document_path"]).exists()
    assert "the Edict of Salt was passed" in _human(fake.seen[4])
    assert "| 3 | Generation 3 |" in final["final_output"]
    assert any("running the research workflow" in line for line in final["board"])


def test_the_judge_sees_the_report_and_the_weakest_section(monkeypatch, tmp_path, small):
    fake, _ = _run_graph(monkeypatch, [
        "KIND: research\n- three sections",
        _outline(3, min_words=5),
        _section(1, words=40), _section(2, words=12), _section(3, words=40),
    ], tmp_path)

    judgment = _human(fake.seen[-1])
    assert "REPORT ON THE DOCUMENT" in judgment
    assert "| # | section | words |" in judgment
    assert "WEAKEST SECTION IN FULL (Generation 2" in judgment
    assert "EXACT" in judgment and "nothing to recount" in judgment


def test_a_rejection_rewrites_the_weakest_sections_and_rejudges(monkeypatch, tmp_path, small):
    fake, final = _run_graph(monkeypatch, [
        "KIND: research\n- three sections",
        _outline(3),
        _section(1), _section(2), _section(3),
        "FINAL:\nMET: 0/1\nBLOCKED: no\nAPPROVE: no\nWHY: too thin",
        _section(1, words=50), _section(2, words=50), _section(3, words=50),
        "FINAL:\nMET: 1/1\nBLOCKED: no\nAPPROVE: yes\nWHY: fuller now",
    ], tmp_path)

    assert fake.calls == 10
    assert final["route"] == "research"
    assert final["final_output"].startswith("Document written")
    rewrite = _human(fake.seen[6])
    assert "the evaluator rejected the document: " in rewrite and "too thin" in rewrite
    assert any("revising after rejection" in line for line in final["board"])
    doc = (tmp_path / final["document_path"]).read_text()
    assert rs._words(doc) >= 150


def test_a_no_verdict_rejection_resubmits_the_document_unchanged(monkeypatch, tmp_path, small):
    """A judge that ran out of its own replies, or answered in tool-call
    XML, has said nothing about the document. Rewriting sections over it
    cost three revision passes on the first live run."""
    fake = _install(monkeypatch, [_outline(3), _section(1), _section(2), _section(3)])
    with bind_workspace(str(tmp_path)):
        first = rs.run_research(_state())
    assert fake.calls == 4

    for feedback in (pn.NO_VERDICT_NOTE, "<function_calls>\n<invoke name=\"execute_bash\">"):
        with bind_workspace(str(tmp_path)):
            again = rs.run_research(_state(feedback=feedback,
                                           document_path=first.update["document_path"]))
        assert fake.calls == 4, "no writer call for a rejection that judged nothing"
        assert again.goto == "evaluator"
        assert any("no verdict" in line for line in again.update["board"])
        assert again.update["output"].startswith("Document written")
