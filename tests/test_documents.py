"""Documents otto makes in this process (agent/pipeline/documents.py, the make_document tool)."""
from __future__ import annotations

import zipfile

import pytest

from agent.pipeline import documents, tools as pt
from agent.pipeline.workspace import bind_workspace

MARKDOWN = """# Tutoring marketplace in India

A **go-to-market** plan for [our app](https://example.com).

## Market size

| Segment | Learners | Share |
| --- | ---: | ---: |
| K-12 | 1,200,000 | 40% |
| Test prep | 450,000 | 15.5% |

## Channels

- Schools and coaching centres
- Parent communities
  on WhatsApp
1. Launch in Pune
2. Expand to Bengaluru

```
budget = 12_00_000
```
"""


def test_the_markdown_reader_finds_every_block():
    blocks = documents.parse(MARKDOWN)
    assert [b.kind for b in blocks] == ["heading", "paragraph", "heading", "table", "heading",
                                        "bullets", "numbered", "code"]
    assert blocks[1].text == "A go-to-market plan for our app."
    assert blocks[3].rows == [["Segment", "Learners", "Share"], ["K-12", "1,200,000", "40%"],
                              ["Test prep", "450,000", "15.5%"]]
    assert blocks[5].items == ["Schools and coaching centres", "Parent communities on WhatsApp"]
    assert blocks[6].items == ["Launch in Pune", "Expand to Bengaluru"]
    assert documents.title_of(blocks, "x") == "Tutoring marketplace in India"


def test_a_word_document_has_the_headings_lists_and_table(tmp_path):
    from docx import Document

    path = documents.write(MARKDOWN, tmp_path / "plan.docx", "docx")
    doc = Document(str(path))
    texts = [p.text for p in doc.paragraphs]
    assert "Market size" in texts and "Launch in Pune" in texts
    assert doc.tables[0].cell(1, 1).text == "1,200,000"
    assert doc.core_properties.title == "Tutoring marketplace in India"


def test_a_pdf_is_a_real_pdf_with_the_text(tmp_path):
    path = documents.write(MARKDOWN, tmp_path / "plan.pdf", "pdf")
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and data.rstrip().endswith(b"%%EOF") and len(data) > 1500
    assert b"/Type /Page" in data and b"Tutoring marketplace in India" in data  # the title, in its info


def test_a_workbook_puts_each_table_on_its_own_sheet_with_numbers(tmp_path):
    from openpyxl import load_workbook

    path = documents.write(MARKDOWN, tmp_path / "plan.xlsx", "xlsx")
    wb = load_workbook(str(path))
    assert wb.sheetnames == ["Market size", "Notes"]
    sheet = wb["Market size"]
    assert [c.value for c in sheet[1]] == ["Segment", "Learners", "Share"]
    assert sheet["B2"].value == 1200000 and sheet["C3"].value == pytest.approx(0.155)
    assert sheet["A1"].font.bold


def test_a_table_only_request_makes_no_notes_sheet(tmp_path):
    from openpyxl import load_workbook

    path = documents.write("| a | b |\n|---|---|\n| 1 | 2 |\n", tmp_path / "t.xlsx", "xlsx")
    assert load_workbook(str(path)).sheetnames == ["t"]


def test_a_presentation_has_a_title_slide_a_slide_per_heading_and_a_table(tmp_path):
    from pptx import Presentation

    path = documents.write(MARKDOWN, tmp_path / "plan.pptx", "pptx")
    slides = Presentation(str(path)).slides
    titles = [s.shapes.title.text for s in slides]
    assert titles[0] == "Tutoring marketplace in India"
    assert "Channels" in titles and titles.count("Market size") >= 1
    assert any(shape.has_table for s in slides for shape in s.shapes)


def test_an_unknown_format_is_refused(tmp_path):
    with pytest.raises(ValueError):
        documents.write("x", tmp_path / "x.odt", "odt")


# ---- the tool ---------------------------------------------------------------------------------

def test_make_document_writes_into_documents(tmp_path):
    with bind_workspace(str(tmp_path)):
        result = pt.make_document("xlsx Market Numbers!\n" + MARKDOWN)
    assert result.ok and result.stdout.startswith("made documents/Market_Numbers.xlsx (")
    assert zipfile.is_zipfile(tmp_path / "documents" / "Market_Numbers.xlsx")


def test_make_document_converts_a_file_already_in_the_workspace(tmp_path):
    (tmp_path / "attachments").mkdir()
    (tmp_path / "attachments" / "f1-GTM.docx.txt").write_text(MARKDOWN)
    with bind_workspace(str(tmp_path)):
        result = pt.make_document("pdf GTM\nfrom attachments/f1-GTM.docx.txt")
    assert result.ok, result.stderr
    assert (tmp_path / "documents" / "GTM.pdf").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("body, why", [
    ("odt report\n# x", "first line must be"),
    ("pdf report\n", "nothing to put in the document"),
    ("pdf report\nfrom ../../etc/passwd", "outside"),
    ("pdf report\nfrom missing.md", "no such file"),
])
def test_make_document_refuses_clearly(tmp_path, body, why):
    with bind_workspace(str(tmp_path)):
        result = pt.make_document(body)
    assert not result.ok and why in result.stderr


def test_make_document_needs_a_workspace():
    assert pt.TOOL_NEEDS["make_document"] == pt.NEEDS_WORKSPACE
    assert "make_document" in pt.TOOL_DISPATCH
