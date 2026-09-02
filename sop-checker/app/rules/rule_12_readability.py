"""Rule 12 - Readability (Flesch Reading Ease + Gunning Fog)."""
from __future__ import annotations

import textstat

from app.extractor import Doc
from .base import Rule, RuleConfig, Finding, heading_level


class Rule12(Rule):
    id = 12
    name = "Readability"
    severity = "warning"
    description = ("Body text should meet the configured Flesch Reading Ease "
                   "floor and Gunning Fog ceiling.")

    def _body_text(self, doc: Doc) -> tuple[str, int]:
        parts: list[str] = []
        for p in doc.body_paragraphs():
            if p.in_table:
                continue
            if heading_level(p) is not None:
                continue
            if p.word_count() < 3:
                continue
            parts.append(p.text.strip())
        text = " ".join(parts)
        return text, len(text.split())

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        text, words = self._body_text(doc)
        if words < config.readability_min_words:
            return self.fail(
                f"Only {words} words of body text, below the "
                f"{config.readability_min_words}-word floor for a reliable "
                "readability score.")

        flesch = round(textstat.flesch_reading_ease(text), 1)
        fog = round(textstat.gunning_fog(text), 1)
        evidence = [
            f"Flesch Reading Ease: {flesch} (min {config.readability_flesch_min})",
            f"Gunning Fog: {fog} (max {config.readability_fog_max})",
            f"body words: {words}",
        ]

        problems = []
        if flesch < config.readability_flesch_min:
            problems.append(
                f"Flesch {flesch} below floor {config.readability_flesch_min}")
        if fog > config.readability_fog_max:
            problems.append(
                f"Gunning Fog {fog} above ceiling {config.readability_fog_max}")

        if problems:
            return self.fail(
                "Readability outside thresholds: " + "; ".join(problems) + ".",
                evidence=evidence, confidence="heuristic")
        return self.ok(
            f"Readability within thresholds (Flesch {flesch}, Fog {fog}).",
            evidence=evidence, confidence="heuristic")


RULE = Rule12()
