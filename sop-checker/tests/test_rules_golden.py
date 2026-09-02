"""Golden-document and per-rule fail-fixture tests + engine invariants."""
from __future__ import annotations

import pytest

from app.engine import evaluate_all, rules_metadata
from tests import make_fixtures as mf


def _by_id(findings):
    return {f.rule_id: f for f in findings}


def test_registry_has_13_rules():
    meta = rules_metadata()
    assert [m["id"] for m in meta] == list(range(1, 14))
    assert all(m["name"] and m["description"] for m in meta)


def test_golden_passes_every_rule(extract_doc, config):
    doc = extract_doc(mf.golden_sop())
    findings = evaluate_all(doc, config)
    failed = [(f.rule_id, f.message) for f in findings if f.passed is not True]
    assert not failed, f"golden should pass all rules, got: {failed}"


# rule_id -> golden_sop kwarg that should make exactly that rule fail
FAIL_FLAGS = {
    2: dict(break_author=True),
    3: dict(break_rev_dates=True),
    4: dict(break_version=True),
    5: dict(break_revision_section=True),
    6: dict(break_signature=True),
    7: dict(break_sig_dates=True),
    8: dict(break_fonts=True),
    10: dict(break_required=True),
    11: dict(break_page_field=True),
    12: dict(break_readability=True),
    13: dict(break_footer_details=True),
}


# Defects that legitimately take a second rule down with them. Since every
# rule now fails rather than abstains when its subject is missing, deleting
# the revision table really does mean there are no revision dates to check,
# and deleting the signature blocks really does mean no signature is dated.
CASCADES = {
    5: {3},     # no revision section -> no revision dates
    6: {7},     # no signature blocks -> no signature dates
}


@pytest.mark.parametrize("rule_id,flags", list(FAIL_FLAGS.items()))
def test_single_defect_fails_only_its_rule(rule_id, flags, extract_doc, config):
    doc = extract_doc(mf.golden_sop(**flags))
    findings = _by_id(evaluate_all(doc, config))
    assert findings[rule_id].passed is False, findings[rule_id].message
    # no rule outside the known cascade should regress to a hard failure
    allowed = {rule_id} | CASCADES.get(rule_id, set())
    collateral = [(rid, f.message) for rid, f in findings.items()
                  if rid not in allowed and f.passed is False]
    assert not collateral, f"unexpected collateral failures: {collateral}"


@pytest.mark.parametrize("rule_id,expected", sorted(CASCADES.items()))
def test_cascades_are_intentional(rule_id, expected, extract_doc, config):
    """The cascade is the point: a missing subject is a failure now, so the
    dependent rule must report it too rather than staying silent."""
    doc = extract_doc(mf.golden_sop(**FAIL_FLAGS[rule_id]))
    findings = _by_id(evaluate_all(doc, config))
    for dependent in expected:
        assert findings[dependent].passed is False, findings[dependent].message


def test_rule1_fails_on_filename_mismatch(extract_doc, config):
    doc = extract_doc(mf.golden_sop(), filename="Completely Other Name.docx")
    findings = _by_id(evaluate_all(doc, config))
    assert findings[1].passed is False


def test_missing_content_is_a_failure_not_an_abstention(extract_doc, config):
    """An empty document must be reported as failing, never as passing and
    never as 'we could not tell'. Rule 10 is the one exception: its missing
    input is the caller's section list, not anything about the document."""
    empty = mf.Document()
    empty.add_paragraph("x")
    doc = extract_doc(empty, filename="x.docx")
    findings = evaluate_all(doc, config)
    for f in findings:
        assert f.passed is not None, (f.rule_id, f.message)
        assert f.message


def test_rule_10_is_the_only_rule_that_can_abstain(extract_doc):
    from app.rules.base import RuleConfig
    empty = mf.Document()
    empty.add_paragraph("x")
    doc = extract_doc(empty, filename="x.docx")
    # no section list configured -> only rule 10 abstains
    findings = evaluate_all(doc, RuleConfig(filename="x.docx"))
    abstained = [f.rule_id for f in findings if f.passed is None]
    assert abstained == [10]
