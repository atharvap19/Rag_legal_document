"""Rule 5 - Revision section present (heading + populated table)."""
from __future__ import annotations

from app.extractor import Doc
from .base import (
    Rule,
    RuleConfig,
    Finding,
    find_revision_heading,
    find_revision_table,
    table_is_populated,
    tables_after_heading,
)


class Rule05(Rule):
    id = 5
    name = "Revision section present"
    severity = "error"
    description = ("A revision/version-history heading with at least one "
                   "populated table beneath it.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        heading = find_revision_heading(doc)
        if heading is None:
            # a revision table without a heading is still a partial signal
            table = find_revision_table(doc)
            if table is not None and table_is_populated(table):
                return self._make(
                    True,
                    "A populated revision table is present (no explicit "
                    "heading matched).",
                    evidence=[f"Table {table.table_index + 1}"],
                    locations=[f"Table {table.table_index + 1}"],
                    confidence="heuristic")
            return self.fail(
                "No revision/version-history section heading found.")

        beneath = tables_after_heading(doc, heading)
        populated = [t for t in beneath if table_is_populated(t)]
        if populated:
            t = populated[0]
            return self.ok(
                "Revision-history section present with a populated table.",
                evidence=[heading.text, f"Table {t.table_index + 1}"],
                locations=[heading.location, f"Table {t.table_index + 1}"])
        return self.fail(
            "Revision-history heading found but no populated table beneath it.",
            evidence=[heading.text],
            locations=[heading.location])


RULE = Rule05()
