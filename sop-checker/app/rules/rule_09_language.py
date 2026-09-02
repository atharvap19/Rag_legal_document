"""Rule 9 - Language errors via an injected LanguageTool checker.

The checker is fed one paragraph at a time -- never concatenated document
text, which would invent sentence boundaries and flood the report with false
capitalisation errors. Headings, table cells and short paragraphs are skipped.
Whitespace/quote/sentence-start rules are disabled, and a caller-supplied
ignore list suppresses acronyms and product names.
"""
from __future__ import annotations

from app.extractor import Doc
from .base import Rule, RuleConfig, Finding, heading_level


# The spec asks to disable whitespace, smart-quote and sentence-start-capital
# rules. Those intents map to several concrete LanguageTool ids depending on
# version, so we disable the whole family. This is the single source of truth,
# reused by the LanguageTool wrapper in app/deps.py.
DISABLED_RULE_IDS = {
    # whitespace noise
    "WHITESPACE_RULE",
    "CONSECUTIVE_SPACES",
    "SENTENCE_WHITESPACE",
    # smart quotes
    "EN_QUOTES",
    # capitalisation at (invented) sentence starts
    "UPPERCASE_SENTENCE_START",
}

MAX_REPORTED = 50


class Rule09(Rule):
    id = 9
    name = "Language errors"
    severity = "warning"
    description = ("Grammar and spelling issues found by LanguageTool, "
                   "checked one paragraph at a time.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        checker = config.language_checker
        if checker is None:
            return self.fail(
                "Language checking is unavailable (no LanguageTool "
                "instance), so the text could not be checked.")

        ignore = {w.lower() for w in (config.ignore_words or [])}
        min_words = config.min_words_for_language

        issues: list[str] = []
        locations: list[str] = []
        checked = 0

        for p in doc.body_paragraphs():
            if p.in_table:
                continue
            if heading_level(p) is not None:
                continue
            text = p.text.strip()
            if len(text.split()) < min_words:
                continue
            checked += 1
            for m in checker.check(text):
                if m.rule_id in DISABLED_RULE_IDS:
                    continue
                snippet = m.context.strip() or text
                if self._ignored(m, snippet, ignore):
                    continue
                issues.append(f"{m.message} — …{snippet}…")
                locations.append(p.location)
                if len(issues) >= MAX_REPORTED:
                    break
            if len(issues) >= MAX_REPORTED:
                break

        if checked == 0:
            return self.fail(
                "No body paragraphs long enough to language-check "
                f"(the floor is {min_words} words).")

        if issues:
            return self.fail(
                f"{len(issues)} language issue(s) found"
                + (" (capped)" if len(issues) >= MAX_REPORTED else "") + ".",
                evidence=issues, locations=locations, confidence="heuristic")
        return self.ok(
            f"No language issues found across {checked} paragraph(s).",
            confidence="heuristic")

    def _ignored(self, match, snippet: str, ignore: set) -> bool:
        if not ignore:
            return False
        # prefer the precisely flagged span; fall back to the context snippet
        span = (getattr(match, "matched_text", "") or snippet).lower()
        return any(term in span for term in ignore)


RULE = Rule09()
