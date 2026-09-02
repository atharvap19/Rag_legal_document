"""Rule 4 - Version consistency.

Collect version strings from the body, every header and footer, the revision
table and core subject/keywords. The document's *current* version (stated
outside the revision history) must be consistent, and should match the latest
revision-history entry. The revision table naturally lists many versions, so
its entries are treated as history -- only its latest entry is compared.
"""
from __future__ import annotations

import re

from app.extractor import Doc, Table
from .base import (
    Rule, RuleConfig, Finding, find_revision_table, find_versions,
    header_column_index, table_labeled_values,
)


# A version in a table is written as a bare number in its own cell -- the
# word 'Version' is the neighbouring label or the column header, so the
# prose regex in find_versions never sees the two together.
_VERSION_LABEL = re.compile(
    r"^(?:version|revision|rev)\.?(?:\s*(?:no|number|#))?\.?$", re.IGNORECASE)
_VERSION_VALUE = re.compile(r"^v?\.?\s*(\d+(?:\.\d+){0,3})$", re.IGNORECASE)


def _version_key(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def _bare_version(text: str):
    """'2.1' or 'v2.1' -> '2.1'; anything else -> None."""
    m = _VERSION_VALUE.match(text.strip())
    return m.group(1) if m else None


def _column_versions(table: Table) -> list[str]:
    """Version numbers from the version column of a revision table."""
    col = header_column_index(table, _VERSION_LABEL)
    if col is None:
        return []
    out = []
    for row in table.rows[1:]:
        if col < len(row.cells):
            v = _bare_version(row.cells[col].text())
            if v:
                out.append(v)
    return out


class Rule04(Rule):
    id = 4
    name = "Version consistency"
    severity = "warning"
    description = ("All version numbers stated across the document (headers, "
                   "footers, body, metadata) must agree and match the latest "
                   "revision-history entry.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        table = find_revision_table(doc)
        table_para_ids = set()
        table_versions: list[str] = []
        if table is not None:
            for p in table.iter_paragraphs():
                table_para_ids.add(p.block_index)
                for v in find_versions(p.text):
                    table_versions.append(v)
            table_versions.extend(_column_versions(table))

        current: list[tuple[str, str]] = []   # (version, location)

        for p in doc.flow_ordered():
            if p.block_index in table_para_ids:
                continue
            for v in find_versions(p.text):
                current.append((v, p.location))

        for hf in doc.all_headers() + doc.all_footers():
            for v in find_versions(hf.text()):
                current.append((v, hf.location))

        # 'Version | 2.1' in a document-information table states the current
        # version; the revision table's own header row is skipped because its
        # neighbouring cell ('Date') is not shaped like a version
        for value, loc in table_labeled_values(doc, _VERSION_LABEL):
            bare = _bare_version(value)
            if bare and (table is None or loc != f"Table {table.table_index + 1}"):
                current.append((bare, loc))

        for label, value in (("core subject", doc.core.subject),
                             ("core keywords", doc.core.keywords)):
            if value:
                for v in find_versions(value):
                    current.append((v, label))

        if not current and not table_versions:
            return self.fail(
                "No version number stated anywhere in the document -- not in "
                "the body, headers, footers, metadata or any table.")

        evidence = [f"{v} @ {loc}" for v, loc in current]
        distinct = sorted({v for v, _ in current}, key=_version_key)

        if len(distinct) > 1:
            return self.fail(
                "Inconsistent version numbers: " + ", ".join(distinct) + ".",
                evidence=evidence,
                locations=sorted({loc for _, loc in current}),
                confidence="certain")

        if not distinct:
            latest = max(table_versions, key=_version_key)
            return self._make(
                True,
                "Version numbers appear only in the revision history; "
                f"latest is {latest}.",
                evidence=[f"revision history: {sorted(set(table_versions), key=_version_key)}"],
                confidence="heuristic")

        stated = distinct[0]
        if table_versions:
            latest = max(table_versions, key=_version_key)
            if _version_key(stated) != _version_key(latest):
                return self.fail(
                    f"Stated version {stated} does not match the latest "
                    f"revision-history entry {latest}.",
                    evidence=evidence + [f"revision history latest: {latest}"],
                    locations=sorted({loc for _, loc in current}),
                    confidence="heuristic")

        return self._make(
            True, f"All version references agree on {stated}.",
            evidence=evidence,
            locations=sorted({loc for _, loc in current}),
            confidence="certain")


RULE = Rule04()
