"""Rules against a document that never used Word's heading styles.

Mirrors a real uploaded SOP: sections marked only by a bigger bold line, and
the title, version and author stated in a document-information table instead
of in metadata. That shape used to leave rules 1, 4 and 12 unable to evaluate
and made rule 10 report that the document had no headings at all.
"""
from __future__ import annotations

import pytest

from app.engine import evaluate_all, get_rule, run_rule
from app.extractor import build_doc
from app.rules.base import (
    COMMON_REQUIRED_SECTIONS, RuleConfig, iter_headings,
)
from tests import make_fixtures as mf
from tests.conftest import StubChecker

FILENAME = "Employee_Onboarding_SOP_v2.1.docx"


def _doc(document=None, filename=FILENAME):
    return build_doc(mf.to_bytes(document if document is not None
                                 else mf.unstyled_sop()), filename)


def _cfg(**kw):
    kw.setdefault("filename", FILENAME)
    kw.setdefault("language_checker", StubChecker())
    kw.setdefault("required_sections", list(COMMON_REQUIRED_SECTIONS))
    return RuleConfig(**kw)


def _run(rule_id, doc=None, config=None):
    return run_rule(get_rule(rule_id), doc or _doc(), config or _cfg())


# ---- heading inference --------------------------------------------------
def test_visual_headings_are_inferred():
    headings = [(h.level, h.text) for h in iter_headings(_doc())]
    assert headings[0] == (1, "Document Reviewer - Test Document")
    assert (2, "Objective") in headings
    assert (2, "Procedure") in headings


def test_body_text_is_never_inferred_as_a_heading():
    inferred = {p.text.strip() for p in _doc().body_paragraphs()
                if p.props.inferred_heading_level is not None}
    assert not any(t.startswith("The objective") for t in inferred)
    assert "Signature: ______________________________" not in inferred
    # a small centred subtitle is not a heading either
    assert "Sample document for testing selected compliance rules" \
        not in inferred


def test_a_styled_document_is_taken_at_its_word():
    """Inference must never second-guess a file that declared headings."""
    doc = build_doc(mf.to_bytes(mf.golden_sop()), "Quality Control Procedure.docx")
    assert all(p.props.inferred_heading_level is None
               for p in doc.body_paragraphs())
    assert [h.text for h in iter_headings(doc)]      # still has real headings


# ---- rules that could not evaluate before -------------------------------
def test_rule_01_matches_the_information_table_title():
    f = _run(1)
    assert f.passed is True, f.message
    assert any("Employee Onboarding SOP" in e for e in f.evidence)


def test_rule_04_reads_versions_out_of_table_cells():
    f = _run(4)
    assert f.passed is True, f.message
    assert "2.1" in f.message


def test_rule_04_still_flags_a_real_disagreement():
    document = mf.unstyled_sop()
    document.tables[0].rows[3].cells[1].text = "3.0"   # states 3.0, history 2.1
    f = _run(4, doc=_doc(document))
    assert f.passed is False
    assert "3.0" in f.message and "2.1" in f.message


def test_rule_12_scores_a_short_body():
    f = _run(12)
    assert f.passed is not None, f.message
    assert "Flesch" in " ".join(f.evidence)


def test_rule_12_fails_when_there_is_too_little_text():
    """Too little text to score is a failure, not an abstention -- rule 10 is
    the only rule that may answer 'not evaluated'."""
    f = _run(12, config=_cfg(readability_min_words=5000))
    assert f.passed is False
    assert "5000-word floor" in f.message


# ---- rules that were answering with the wrong evidence ------------------
def test_rule_02_ignores_a_tool_generated_author():
    f = _run(2)
    assert f.passed is True
    assert any("python-docx" in e and "ignored" in e for e in f.evidence)
    assert any("John Smith" in e for e in f.evidence)


def test_rule_02_fails_when_only_the_tool_name_is_there():
    f = _run(2, doc=_doc(mf.unstyled_sop(doc_info=False)))
    assert f.passed is False, f.message


def test_rule_07_prefers_the_signature_block_own_date():
    """The revision table ends as close to the approval block as its own
    date line, and its dates are history, not signature dates."""
    f = _run(7)
    assert f.passed is True
    assert all("25/08/2026" in e for e in f.evidence)
    assert not any("15/08/2026" in e for e in f.evidence)


def test_rule_10_sees_the_inferred_headings():
    f = _run(10)
    present = " ".join(f.evidence)
    for section in ("Scope", "Responsibilities", "Procedure"):
        assert section in present
    # the document really is missing these two, so it should still fail
    assert f.passed is False
    assert "Purpose" in f.message and "References" in f.message


def test_rule_08_does_not_flag_the_title_as_body_text():
    f = _run(8)
    assert not any("Paragraph 0" in loc for loc in f.locations), f.evidence


def test_rule_10_asks_for_a_section_list_when_none_is_given():
    f = _run(10, config=_cfg(required_sections=[]))
    assert f.passed is None
    assert "Please enter the sections" in f.message


# ---- the headline: every rule reaches a verdict -------------------------
def test_no_rule_is_left_unevaluated():
    findings = evaluate_all(_doc(), _cfg())
    unevaluated = {f.rule_id: f.message for f in findings if f.passed is None}
    assert not unevaluated, unevaluated


@pytest.mark.parametrize("rule_id,expected", [
    (1, True), (2, True), (3, True), (4, True), (5, True), (6, True),
    (7, True), (9, True),
    (8, False), (10, False), (11, False), (12, False), (13, False),
])
def test_verdicts_are_stable(rule_id, expected):
    doc, config = _doc(), _cfg()
    assert run_rule(get_rule(rule_id), doc, config).passed is expected
