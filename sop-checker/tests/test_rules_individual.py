"""Targeted per-rule assertions: na-states, evidence, and edge behaviour."""
from __future__ import annotations

import pytest

from app.engine import get_rule
from app.rules import RuleConfig
from tests import make_fixtures as mf
from tests.conftest import StubChecker


def _run(rule_id, doc, config):
    return get_rule(rule_id).evaluate(doc, config)


# ---- rule 1 -------------------------------------------------------------
def test_rule1_na_without_title_or_heading(extract_doc, config):
    d = mf.Document()
    d.add_paragraph("just some plain text with no heading")
    f = _run(1, extract_doc(d, filename="anything.docx"), config)
    assert f.passed is False
    assert "states no title" in f.message


def test_rule1_pass_on_match(extract_doc, config):
    d = mf.Document()
    mf.set_core(d, title="Onboarding Guide")
    d.add_heading("Onboarding Guide", level=1)
    f = _run(1, extract_doc(d, filename="Onboarding-Guide-v2.docx"), config)
    assert f.passed is True


def _rule1(extract_doc, config, filename, title="Onboarding Guide"):
    d = mf.Document()
    mf.set_core(d, title=title)
    d.add_heading(title, level=1)
    return _run(1, extract_doc(d, filename=filename), config)


@pytest.mark.parametrize("filename", [
    "Onboarding Guide final.docx",      # bare keyword, no version digits
    "Onboarding Guide copy.docx",
    "Onboarding Guide - Copy.docx",     # Windows duplicate
    "Onboarding Guide draft.docx",
    "Onboarding Guide (1).docx",        # browser duplicate download
    "Onboarding Guied.docx",            # transposition typo
    "Guide for Onboarding.docx",        # reordered tokens
])
def test_rule1_fuzzy_accepts_near_misses(extract_doc, config, filename):
    f = _rule1(extract_doc, config, filename)
    assert f.passed is True, f.message
    assert f.confidence == "heuristic"


@pytest.mark.parametrize("filename", [
    "Completely Other Name.docx",
    "Invoice Template.docx",
    "Safety Manual.docx",
])
def test_rule1_fuzzy_still_rejects_unrelated_names(extract_doc, config,
                                                   filename):
    assert _rule1(extract_doc, config, filename).passed is False


def test_rule1_fuzzy_rejects_document_number_mismatch(extract_doc, config):
    """Near-identical strings that disagree on a number are different docs."""
    f = _rule1(extract_doc, config, "SOP-002 Cleaning Procedure.docx",
               title="SOP-001 Cleaning Procedure")
    assert f.passed is False, f.message


def test_rule1_threshold_of_one_restores_exact_matching(extract_doc):
    # a typo is the case only the fuzzy pass can rescue -- suffix noise is
    # already handled deterministically in normalisation
    cfg = RuleConfig(title_match_threshold=1.0)
    assert _rule1(extract_doc, cfg, "Onboarding Guied.docx").passed is False
    assert _rule1(extract_doc, cfg, "Onboarding Guide (1).docx").passed is True


def test_rule1_suffix_noise_is_stripped_not_merely_tolerated(extract_doc,
                                                             config):
    """final / copy / (n) should match exactly, independent of title length,
    so a short title is not penalised by a fixed-length suffix."""
    for name in ["Guide final.docx", "Guide copy.docx", "Guide (1).docx"]:
        f = _rule1(extract_doc, config, name, title="Guide")
        assert f.passed is True, f.message
        assert not any("similarity" in e for e in f.evidence), f.evidence


def test_rule1_reports_the_score_as_evidence(extract_doc, config):
    f = _rule1(extract_doc, config, "Invoice Template.docx")
    assert any("similarity" in e for e in f.evidence), f.evidence


# ---- rule 3 -------------------------------------------------------------
def test_rule3_fails_without_revision_table(extract_doc, config):
    f = _run(3, extract_doc(mf.golden_sop(break_revision_section=True)), config)
    assert f.passed is False
    assert "no revision dates" in f.message


def test_rule3_reports_future_and_unordered(extract_doc, config):
    f = _run(3, extract_doc(mf.golden_sop(break_rev_dates=True)), config)
    assert f.passed is False
    joined = f.message.lower()
    assert "future" in joined or "order" in joined


# ---- rule 5 -------------------------------------------------------------
def test_rule5_fail_without_section(extract_doc, config):
    f = _run(5, extract_doc(mf.golden_sop(break_revision_section=True)), config)
    assert f.passed is False


# ---- rule 7 -------------------------------------------------------------
def test_rule7_fails_without_signatures(extract_doc, config):
    f = _run(7, extract_doc(mf.golden_sop(break_signature=True)), config)
    assert f.passed is False


# ---- rule 8 -------------------------------------------------------------
def test_rule8_locations_point_at_offending_paragraph(extract_doc, config):
    f = _run(8, extract_doc(mf.golden_sop(break_fonts=True)), config)
    assert f.passed is False
    assert f.locations
    assert any("Times New Roman" in e for e in f.evidence)


# ---- rule 10 ------------------------------------------------------------
def test_rule10_na_when_no_required_list(extract_doc):
    cfg = RuleConfig(required_sections=[], language_checker=StubChecker())
    f = _run(10, extract_doc(mf.golden_sop()), cfg)
    assert f.passed is None
    assert "Please enter the sections" in f.message


def test_rule10_na_is_the_default_state(extract_doc):
    """No section list is configured until somebody supplies one."""
    cfg = RuleConfig(language_checker=StubChecker())
    assert cfg.required_sections == []
    assert _run(10, extract_doc(mf.golden_sop()), cfg).passed is None


def test_rule10_lists_missing(extract_doc, config):
    f = _run(10, extract_doc(mf.golden_sop(break_required=True)), config)
    assert f.passed is False
    assert "References" in f.message


# ---- rule 11 ------------------------------------------------------------
def test_rule11_fails_without_footers(extract_doc, config):
    f = _run(11, extract_doc(mf.doc_no_headers_footers()), config)
    assert f.passed is False
    assert "limitation" in f.message.lower() or "does not verify" in f.message


def test_rule11_flags_hardcoded_number(extract_doc, config):
    f = _run(11, extract_doc(mf.golden_sop(break_page_field=True)), config)
    assert f.passed is False
    assert "hardcoded" in f.message.lower()


def test_rule11_passes_with_field_and_notes_numpages(extract_doc, config):
    f = _run(11, extract_doc(mf.golden_sop()), config)
    assert f.passed is True
    assert "NUMPAGES" in f.message


# ---- rule 12 ------------------------------------------------------------
def test_rule12_na_on_thin_document(extract_doc, config):
    d = mf.Document()
    mf.body_para(d, "Short and sweet body text here today.")
    f = _run(12, extract_doc(d), config)
    assert f.passed is False
    assert "word floor" in f.message


# ---- rule 13 ------------------------------------------------------------
def test_rule13_reports_missing_details(extract_doc, config):
    f = _run(13, extract_doc(mf.golden_sop(break_footer_details=True)), config)
    assert f.passed is False
    assert "document ID" in f.message or "confidentiality" in f.message
