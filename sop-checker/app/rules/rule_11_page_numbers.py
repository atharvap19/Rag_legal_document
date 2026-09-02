"""Rule 11 - Page numbers in footers.

A .docx does not record pagination; nothing in the XML says where a page
begins. So we verify that a PAGE *field* exists in the footer and flag a
hardcoded typed number with no field code -- that is the only way numbering
actually goes wrong, since a field-based footer is correct by construction.
We cannot verify that page 7 renders '7', and the finding says so.
"""
from __future__ import annotations

import re

from app.extractor import Doc, HeaderFooter
from .base import Rule, RuleConfig, Finding


PAGE_LITERAL_RE = re.compile(
    r"\bpage\s+\d+\b|\bpage\s+\d+\s*(?:of|/)\s*\d+\b", re.IGNORECASE)
BARE_NUMBER_RE = re.compile(r"^\s*[-–]?\s*\d{1,4}\s*[-–]?\s*$")

_LIMITATION = (" (A .docx cannot be paginated from XML, so this checks for a "
               "PAGE field and flags hardcoded numbers; it does not verify "
               "rendered page numbers.)")


def _has_field(hf: HeaderFooter, kind: str) -> bool:
    return any(f.kind == kind for f in hf.fields)


def _looks_hardcoded(hf: HeaderFooter) -> bool:
    if _has_field(hf, "PAGE"):
        return False
    text = hf.text().strip()
    if not text:
        return False
    if PAGE_LITERAL_RE.search(text):
        return True
    for line in text.splitlines():
        if BARE_NUMBER_RE.match(line):
            return True
    return False


class Rule11(Rule):
    id = 11
    name = "Page numbers"
    severity = "warning"
    description = ("Every section footer should carry a PAGE field; a "
                   "hardcoded typed page number is flagged as the real "
                   "failure mode.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        footers = doc.all_footers()
        if not footers:
            return self.fail(
                "No footers present, so there is no page number anywhere."
                + _LIMITATION)

        hardcoded: list[HeaderFooter] = []
        with_page: list[HeaderFooter] = []
        with_numpages = False
        for hf in footers:
            if _has_field(hf, "PAGE"):
                with_page.append(hf)
            if _has_field(hf, "NUMPAGES"):
                with_numpages = True
            if _looks_hardcoded(hf):
                hardcoded.append(hf)

        numpages_note = (" NUMPAGES present for 'Page X of Y'."
                         if with_numpages else
                         " No NUMPAGES field (no 'Page X of Y').")

        if hardcoded:
            return self.fail(
                f"{len(hardcoded)} footer(s) contain a hardcoded page number "
                "with no PAGE field." + numpages_note + _LIMITATION,
                evidence=[f"{hf.text().strip()!r}" for hf in hardcoded],
                locations=[hf.location for hf in hardcoded],
                confidence="certain")

        if not with_page:
            return self.fail(
                "Footers are present but none contains a PAGE field."
                + numpages_note + _LIMITATION,
                evidence=[f"{hf.location}: {hf.text().strip()!r}"
                          for hf in footers],
                locations=[hf.location for hf in footers],
                confidence="certain")

        return self.ok(
            f"{len(with_page)} footer(s) use a PAGE field." + numpages_note
            + _LIMITATION,
            evidence=[f"{hf.location}: PAGE field" for hf in with_page],
            locations=[hf.location for hf in with_page],
            confidence="certain")


RULE = Rule11()
