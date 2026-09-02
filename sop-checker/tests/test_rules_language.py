"""Rule 9 (language) tests with a stubbed checker -- no JVM required."""
from __future__ import annotations

from app.engine import get_rule
from app.rules import RuleConfig
from app.rules.base import LanguageIssue
from tests import make_fixtures as mf
from tests.conftest import StubChecker


def _run9(doc, checker=None, ignore=None):
    cfg = RuleConfig(language_checker=checker, ignore_words=ignore or [])
    return get_rule(9).evaluate(doc, cfg)


def _issue(rule_id="MORFOLOGIK_RULE_EN_US", context="teh team here"):
    return LanguageIssue(message="Possible spelling mistake", context=context,
                         offset=0, length=3, rule_id=rule_id,
                         replacements=["the"])


def test_rule9_fails_without_checker(extract_doc):
    f = _run9(extract_doc(mf.golden_sop()), checker=None)
    assert f.passed is False
    assert "unavailable" in f.message


def test_rule9_pass_when_no_issues(extract_doc):
    f = _run9(extract_doc(mf.golden_sop()), checker=StubChecker([]))
    assert f.passed is True


def test_rule9_fail_reports_issues(extract_doc):
    f = _run9(extract_doc(mf.golden_sop()), checker=StubChecker([_issue()]))
    assert f.passed is False
    assert f.evidence and f.locations


def test_rule9_disabled_rule_ids_are_filtered(extract_doc):
    checker = StubChecker([_issue(rule_id="WHITESPACE_RULE")])
    f = _run9(extract_doc(mf.golden_sop()), checker=checker)
    assert f.passed is True


def test_rule9_ignore_list_suppresses(extract_doc):
    checker = StubChecker([_issue(context="Acme Widget is fine")])
    f = _run9(extract_doc(mf.golden_sop()), checker=checker, ignore=["acme"])
    assert f.passed is True


def test_rule9_fails_when_only_short_paragraphs(extract_doc):
    d = mf.Document()
    d.add_paragraph("one two")   # under the word floor
    d.add_paragraph("three four")
    f = _run9(extract_doc(d), checker=StubChecker([_issue()]))
    assert f.passed is False
