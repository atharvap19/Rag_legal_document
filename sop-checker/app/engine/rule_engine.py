"""Rule registry and evaluation.

One place that knows which rules exist: id -> instance, the metadata the API
and UI build themselves from, and the guard that turns an unexpected rule
crash into an un-evaluable finding instead of a failed analysis.

To add a rule: drop a module in app/rules/ exposing a module-level ``RULE``
and add it to _MODULES below.
"""
from __future__ import annotations

from typing import Optional

from app.extractor import Doc
from app.rules.base import Finding, Rule, RuleConfig, rule_metadata
from app.rules import rule_01_title
from app.rules import rule_02_author
from app.rules import rule_03_revision_dates
from app.rules import rule_04_version
from app.rules import rule_05_revision_section
from app.rules import rule_06_signature
from app.rules import rule_07_signature_dates
from app.rules import rule_08_formatting
from app.rules import rule_09_language
from app.rules import rule_10_sections
from app.rules import rule_11_page_numbers
from app.rules import rule_12_readability
from app.rules import rule_13_footer

_MODULES = [
    rule_01_title,
    rule_02_author,
    rule_03_revision_dates,
    rule_04_version,
    rule_05_revision_section,
    rule_06_signature,
    rule_07_signature_dates,
    rule_08_formatting,
    rule_09_language,
    rule_10_sections,
    rule_11_page_numbers,
    rule_12_readability,
    rule_13_footer,
]

REGISTRY: dict[int, Rule] = {}
for _m in _MODULES:
    _rule = _m.RULE
    REGISTRY[_rule.id] = _rule


def all_rules() -> list[Rule]:
    return [REGISTRY[i] for i in sorted(REGISTRY)]


def get_rule(rule_id: int) -> Optional[Rule]:
    return REGISTRY.get(rule_id)


def rules_metadata() -> list[dict]:
    return [rule_metadata(r) for r in all_rules()]


def evaluate_all(doc: Doc, config: RuleConfig) -> list[Finding]:
    return [run_rule(r, doc, config) for r in all_rules()]


def run_rule(rule: Rule, doc: Doc, config: RuleConfig) -> Finding:
    """Evaluate a rule, converting an unexpected crash into an un-evaluable
    finding rather than failing the whole analysis."""
    try:
        return rule.evaluate(doc, config)
    except Exception as exc:  # pragma: no cover - defensive
        return Finding(
            rule_id=rule.id,
            rule_name=rule.name,
            passed=None,
            severity="info",
            message=f"Rule could not be evaluated: {exc!r}",
            evidence=[],
            locations=[],
            confidence="certain",
        )
