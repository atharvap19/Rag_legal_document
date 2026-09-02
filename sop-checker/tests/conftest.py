"""Shared fixtures for rule and API tests."""
from __future__ import annotations

import pytest

from app.extractor import build_doc
from app.rules import RuleConfig
from app.rules.base import COMMON_REQUIRED_SECTIONS
from app.rules.base import LanguageIssue
from tests import make_fixtures as mf


class StubChecker:
    """Stand-in for the LanguageTool wrapper so the suite needs no JVM."""

    def __init__(self, issues=None):
        self._issues = list(issues or [])

    def check(self, text: str):
        return list(self._issues)


@pytest.fixture
def stub_checker():
    return StubChecker()


@pytest.fixture
def language_issue():
    return LanguageIssue(
        message="Possible spelling mistake found",
        context="teh team",
        offset=0, length=3,
        rule_id="MORFOLOGIK_RULE_EN_US",
        replacements=["the"],
    )


@pytest.fixture
def config(stub_checker):
    # rule 10 has no built-in section list, so the suite states one
    return RuleConfig(language_checker=stub_checker,
                      required_sections=list(COMMON_REQUIRED_SECTIONS))


@pytest.fixture
def extract_doc():
    def _extract(document, filename="Quality Control Procedure.docx"):
        return build_doc(mf.to_bytes(document), filename)
    return _extract
