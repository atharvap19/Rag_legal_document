"""Rule 3 - Revision-history dates parse, are ordered, and not in the future."""
from __future__ import annotations

import re
from datetime import datetime, timedelta

from app.extractor import Doc
from .base import (
    Rule,
    RuleConfig,
    Finding,
    find_revision_table,
    header_column_index,
    find_dates,
    parse_date,
)


_DATE_HDR = re.compile(r"\bdate\b", re.IGNORECASE)


class Rule03(Rule):
    id = 3
    name = "Revision-history dates"
    severity = "error"
    description = ("Dates in the revision history must parse, be "
                   "chronologically non-decreasing, and none in the future.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        table = find_revision_table(doc)
        if table is None:
            return self.fail(
                "No revision-history table found, so the document records no "
                "revision dates.")

        col = header_column_index(table, _DATE_HDR)
        rows = table.rows[1:] if len(table.rows) > 1 else []
        cells: list[str] = []
        for row in rows:
            if col is not None and col < len(row.cells):
                cells.append(row.cells[col].text().strip())
            else:
                # no explicit Date column: scan the whole row for a date
                joined = " ".join(c.text() for c in row.cells)
                cells.append(joined)
        cells = [c for c in cells if c.strip()]

        if not cells:
            return self.fail(
                "The revision table has a header but no data rows, so it "
                "records no revision dates.")

        now = datetime.now()
        horizon = now + timedelta(days=config.future_grace_days)
        parsed: list[tuple[str, datetime]] = []
        evidence: list[str] = []
        problems: list[str] = []

        for raw in cells:
            if col is not None:
                dt = parse_date(raw)
                pretty = raw
            else:
                hits = find_dates(raw)
                dt = hits[0][1] if hits else None
                pretty = hits[0][0] if hits else raw
            if dt is None:
                problems.append(f"unparseable date: {raw!r}")
                evidence.append(raw)
                continue
            evidence.append(f"{pretty} -> {dt.date().isoformat()}")
            if dt > horizon:
                problems.append(f"future date: {pretty!r}")
            parsed.append((pretty, dt))

        # chronological (non-decreasing) order
        for (a_txt, a), (b_txt, b) in zip(parsed, parsed[1:]):
            if b < a:
                problems.append(
                    f"out of order: {b_txt!r} precedes {a_txt!r}")

        if problems:
            return self.fail(
                "Revision-history dates have problems: "
                + "; ".join(problems) + ".",
                evidence=evidence, locations=[f"Table {table.table_index + 1}"])
        return self.ok(
            f"All {len(parsed)} revision date(s) parse, are ordered, and are "
            "not in the future.",
            evidence=evidence, locations=[f"Table {table.table_index + 1}"])


RULE = Rule03()
