"""Rule 13 - Footer details: doc ID, confidentiality text, page number."""
from __future__ import annotations

import re

from app.extractor import Doc, HeaderFooter
from .base import Rule, RuleConfig, Finding
from .rule_11_page_numbers import _has_field, _looks_hardcoded


class Rule13(Rule):
    id = 13
    name = "Footer details"
    severity = "warning"
    description = ("Footers should carry a document ID, confidentiality "
                   "marking, and a page number, across all sections and "
                   "different-first-page footers.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        footers = doc.all_footers()
        if not footers:
            return self.fail(
                "No footers present, so the document carries no document ID, "
                "confidentiality marking or page number.")

        doc_id_re = re.compile(config.doc_id_pattern)
        conf_terms = [t.lower() for t in config.confidentiality_terms]

        found_docid = None
        found_conf = None
        found_page = None
        evidence: list[str] = []

        for hf in footers:
            text = hf.text()
            low = text.lower()
            if found_docid is None:
                m = doc_id_re.search(text)
                if m:
                    found_docid = (m.group(0), hf.location)
            if found_conf is None:
                for term in conf_terms:
                    if term in low:
                        found_conf = (term, hf.location)
                        break
            if found_page is None:
                if _has_field(hf, "PAGE") or _looks_hardcoded(hf):
                    kind = "PAGE field" if _has_field(hf, "PAGE") else "typed number"
                    found_page = (kind, hf.location)
            evidence.append(f"{hf.location}: {text.strip()!r}")

        missing = []
        locations = []
        if found_docid:
            locations.append(found_docid[1])
        else:
            missing.append("document ID")
        if found_conf:
            locations.append(found_conf[1])
        else:
            missing.append("confidentiality marking")
        if found_page:
            locations.append(found_page[1])
        else:
            missing.append("page number")

        detail = []
        if found_docid:
            detail.append(f"doc ID {found_docid[0]!r}")
        if found_conf:
            detail.append(f"confidentiality {found_conf[0]!r}")
        if found_page:
            detail.append(f"page number ({found_page[0]})")

        if missing:
            return self.fail(
                "Footer is missing: " + ", ".join(missing) + ".",
                evidence=evidence + ["found: " + ("; ".join(detail) or "none")],
                locations=sorted(set(locations)) or [hf.location for hf in footers],
                confidence="heuristic")
        return self.ok(
            "Footer contains " + "; ".join(detail) + ".",
            evidence=evidence, locations=sorted(set(locations)),
            confidence="heuristic")


RULE = Rule13()
