"""The 13 SOP rules.

Each module exposes a single module-level ``RULE`` instance. Rules consume
only the extractor's Doc model and a RuleConfig -- no python-docx, lxml or
FastAPI below this package. The registry that collects them lives in
app/engine/rule_engine.py.
"""
from .base import (
    Finding,
    LanguageChecker,
    LanguageIssue,
    Rule,
    RuleConfig,
)

__all__ = [
    "Finding",
    "LanguageChecker",
    "LanguageIssue",
    "Rule",
    "RuleConfig",
]
