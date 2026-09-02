"""Phase 1 extraction tests.

Covers the specific hazards the spec calls out: mid-word run splits, theme
vs literal fonts, paragraph/table interleaving, nested tables, multi-section
different-first-page footers, header/footer absence, and both field kinds.
"""
from __future__ import annotations

from app.extractor import build_doc
from app.extractor import Paragraph, TableRef
from tests import make_fixtures as mf


def _doc(document, name="fixture.docx"):
    return build_doc(mf.to_bytes(document), name)


# --------------------------------------------------------------------------
def test_flow_order_interleaves_paragraphs_and_tables():
    doc = _doc(mf.doc_interleave())
    top_kinds = [("T" if isinstance(b, TableRef) else "P") for b in doc.blocks]
    assert top_kinds == ["P", "T", "P", "T", "P"]

    flow = [p.text for p in doc.flow_ordered()]
    assert flow == ["Alpha", "Table-one-cell", "Beta",
                    "Table-two-cell", "Gamma"]

    # block indices are strictly increasing in flow order
    idx = [p.block_index for p in doc.flow_ordered()]
    assert idx == sorted(idx)
    assert len(set(idx)) == len(idx)


def test_nested_table_does_not_break_walk():
    doc = _doc(mf.doc_nested_table())
    texts = [p.text for p in doc.flow_ordered()]
    assert texts[0] == "Before"
    assert texts[-1] == "After"
    assert "OuterCell" in texts
    # nested cells present
    assert {"n00", "n01", "n10", "n11"}.issubset(set(texts))
    # outer + nested table both registered
    assert len(doc.tables) == 2


def test_split_word_runs_merge():
    doc = _doc(mf.doc_split_word())
    para = doc.flow_ordered()[0]
    assert para.text == "proofreading"
    assert len(para.runs) == 1
    assert para.runs[0].font.name == "Arial"


def test_theme_and_literal_same_face_are_not_a_difference():
    doc = _doc(mf.doc_theme_vs_literal())
    # guard: the default template's minor font is Cambria
    assert doc.styles.theme_minor == "Cambria"
    para = doc.flow_ordered()[0]
    # both runs resolve to Cambria -> they merge into a single run
    assert para.text == "themed literal"
    assert len(para.runs) == 1
    assert para.runs[0].font.name == "Cambria"


def test_textbox_content_is_not_dropped():
    doc = _doc(mf.doc_textbox())
    texts = [p.text for p in doc.flow_ordered()]
    assert "Anchor paragraph" in texts
    assert "Following paragraph" in texts
    assert "Text box line one" in texts
    assert "Text box line two" in texts
    # text-box paragraphs are flagged
    assert any(p.from_textbox and "Text box" in p.text
               for p in doc.flow_ordered())


def test_three_sections_with_different_first_page_footers():
    doc = _doc(mf.doc_three_sections_diff_first())
    assert len(doc.sections) == 3
    for sec in doc.sections:
        assert sec.different_first_page is True
        # effective default + first-page footers both present
        assert "default" in sec.footers
        assert "first" in sec.footers
        default_kinds = [f.kind for f in sec.footers["default"].fields]
        first_kinds = [f.kind for f in sec.footers["first"].fields]
        assert "PAGE" in default_kinds
        assert "PAGE" in first_kinds


def test_document_with_no_headers_or_footers():
    doc = _doc(mf.doc_no_headers_footers())
    assert doc.all_headers() == []
    assert doc.all_footers() == []


def test_both_field_kinds_detected():
    doc = _doc(mf.doc_both_field_kinds())
    fields = [f for p in doc.body_paragraphs() for f in p.fields]
    simple_page = [f for f in fields if f.kind == "PAGE" and f.simple]
    complex_page = [f for f in fields if f.kind == "PAGE" and not f.simple]
    numpages = [f for f in fields if f.kind == "NUMPAGES"]
    assert simple_page, "simple PAGE field not detected"
    assert complex_page, "complex PAGE field not detected"
    assert numpages, "NUMPAGES field not detected"


def test_core_properties_extracted():
    d = mf.Document()
    mf.set_core(d, title="My Title", author="Jane Doe",
                last_modified_by="Jane Doe", subject="subj", keywords="k1, k2")
    d.add_paragraph("body")
    doc = _doc(d)
    assert doc.core.title == "My Title"
    assert doc.core.author == "Jane Doe"
    assert doc.core.last_modified_by == "Jane Doe"
    assert doc.core.subject == "subj"
    assert doc.core.keywords == "k1, k2"


def test_table_cell_positions_and_locations():
    doc = _doc(mf.doc_interleave())
    cell_paras = [p for p in doc.flow_ordered() if p.in_table]
    assert cell_paras
    for p in cell_paras:
        assert p.table_pos is not None
        assert p.location.startswith("Table ")


def test_literal_font_resolution():
    d = mf.Document()
    mf.body_para(d, "Arial body text", font="Arial", size=12)
    doc = _doc(d)
    para = doc.flow_ordered()[0]
    assert para.runs[0].font.name == "Arial"
    assert para.runs[0].font.size == 12.0
    assert para.runs[0].font.name_is_theme is False
