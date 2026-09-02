"""Rule 6 - Signature blocks present."""
from __future__ import annotations

import re

from app.extractor import Doc, Table
from .base import (
    Rule,
    RuleConfig,
    Finding,
    signature_paragraphs,
    table_header_cells,
)


_SIG_TABLE_NAME = re.compile(r"\bname\b", re.IGNORECASE)
_SIG_TABLE_OTHER = re.compile(
    r"\b(?:signature|designation|title|role|date)\b", re.IGNORECASE)


def is_signature_table(table: Table) -> bool:
    headers = table_header_cells(table)
    if not headers:
        return False
    joined = " ".join(headers)
    return bool(_SIG_TABLE_NAME.search(joined)
               and _SIG_TABLE_OTHER.search(joined))


class Rule06(Rule):
    id = 6
    name = "Signature blocks"
    severity = "error"
    description = ("A 'Prepared/Reviewed/Approved by' block or a "
                   "name/designation/signature/date table must be present.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        sig_paras = signature_paragraphs(doc)
        evidence = [p.text.strip() for p in sig_paras]
        locations = [p.location for p in sig_paras]

        sig_tables = [t for t in doc.tables if is_signature_table(t)]
        for t in sig_tables:
            evidence.append("signature table: "
                            + ", ".join(table_header_cells(t)))
            locations.append(f"Table {t.table_index + 1}")

        if sig_paras or sig_tables:
            return self.ok(
                f"Found {len(sig_paras)} signature label(s) and "
                f"{len(sig_tables)} signature table(s).",
                evidence=evidence, locations=locations, confidence="heuristic")
        return self.fail(
            "No signature block (Prepared/Reviewed/Approved by) or "
            "signature table found.",
            confidence="heuristic")


RULE = Rule06()
