"""Rule 1 - Title / filename match."""
from __future__ import annotations

import re

from app.extractor import Doc
from .base import (
    Rule,
    RuleConfig,
    Finding,
    first_heading1,
    digits_conflict,
    normalize_filename,
    normalize_title,
    similarity,
    table_labeled_values,
)


# a 'Document Title | ...' row in a document-information table states the
# title as plainly as the core property does, and survives Save As
_TITLE_LABEL = re.compile(r"^(?:document\s+)?(?:title|name)$", re.IGNORECASE)


class Rule01(Rule):
    id = 1
    name = "Title / filename match"
    severity = "warning"
    description = ("The document title (core property, first heading, or a "
                   "'Document Title' row in an information table) should "
                   "match the uploaded file name, allowing for near-misses "
                   "like a 'final' or '(1)' suffix.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        filename = config.filename or doc.filename
        file_key = normalize_filename(filename)
        if not file_key:
            return self.fail(
                "No file name available to compare the document title "
                "against.")

        title = doc.core.title
        h1 = first_heading1(doc)
        h1_text = h1.text if h1 else None

        candidates = []
        if title:
            candidates.append(("title", title, normalize_title(title)))
        if h1_text:
            candidates.append(("Heading 1", h1_text, normalize_title(h1_text)))
        table_titles = table_labeled_values(doc, _TITLE_LABEL)
        if table_titles:
            value, table_loc = table_titles[0]
            candidates.append(
                ("information-table title", value, normalize_title(value)))

        if not candidates:
            return self.fail(
                "The document states no title to match against the file name "
                f"'{filename}': no title in the metadata, no heading, and no "
                "title row in an information table.",
                evidence=[f"file name: {filename}"],
                confidence="certain",
            )

        evidence = [f"file name: {filename}"] + [
            f"{label}: {value!r}" for label, value, _ in candidates]
        locations = [h1.location] if h1 else []
        if table_titles:
            locations.append(table_loc)

        exact = [c for c in candidates if c[2] == file_key]
        if exact:
            how = ", ".join(c[0] for c in exact)
            return self.ok(
                f"File name matches the document {how}.",
                evidence=evidence, locations=locations, confidence="heuristic")

        # No exact hit -- fall back to a fuzzy comparison. A numbering
        # disagreement (SOP-001 vs SOP-002) is disqualifying no matter how
        # alike the rest of the string is, so those never reach the threshold.
        threshold = config.title_match_threshold
        scored = [
            (label, key, 0.0 if digits_conflict(key, file_key)
             else similarity(key, file_key))
            for label, _value, key in candidates
        ]
        best_label, _best_key, best_score = max(scored, key=lambda c: c[2])
        evidence.append(
            f"best similarity: {best_score:.2f} ({best_label}) "
            f"vs threshold {threshold:.2f}")

        if best_score >= threshold:
            return self.ok(
                f"File name is a close match for the document {best_label} "
                f"(similarity {best_score:.2f}).",
                evidence=evidence, locations=locations, confidence="heuristic")
        return self.fail(
            "Document title does not match the uploaded file name "
            f"(best similarity {best_score:.2f}, need {threshold:.2f}).",
            evidence=evidence, locations=locations, confidence="heuristic")


RULE = Rule01()
