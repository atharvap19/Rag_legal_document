"""Rule 7 - Dates near signature blocks.

For each signature block (rule 6), a date must appear within a bounded window
of flow-order blocks or in the same table row -- never merely 'somewhere in
the document'.
"""
from __future__ import annotations

from app.extractor import Doc, Paragraph
from .base import (
    Rule,
    RuleConfig,
    Finding,
    signature_paragraphs,
    find_dates,
    find_revision_table,
)


class Rule07(Rule):
    id = 7
    name = "Dates near signatures"
    severity = "warning"
    description = ("Each signature block must have a date within a bounded "
                   "window of flow-order blocks or in the same table row.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        sig_paras = signature_paragraphs(doc)
        if not sig_paras:
            return self.fail(
                "No signature blocks found, so no signature carries a date.")

        flow = doc.flow_ordered()
        pos_by_index = {p.block_index: i for i, p in enumerate(flow)}
        window = max(1, config.date_window)

        # revision-history rows are the document's history, not the date
        # somebody signed on, and a long revision table can end within the
        # window of the approval block directly beneath it
        revision = find_revision_table(doc)
        skip = ({p.block_index for p in revision.iter_paragraphs()}
                if revision is not None else set())

        dated: list[str] = []
        undated: list[str] = []
        evidence: list[str] = []

        for sig in sig_paras:
            found = self._nearby_date(doc, sig, flow, pos_by_index, window,
                                      skip)
            if found:
                dated.append(sig.location)
                evidence.append(f"{sig.text.strip()!r} -> date {found}")
            else:
                undated.append(sig.location)
                evidence.append(f"{sig.text.strip()!r} -> no nearby date")

        if undated:
            return self.fail(
                f"{len(undated)} signature block(s) have no nearby date.",
                evidence=evidence, locations=undated, confidence="heuristic")
        return self.ok(
            f"All {len(dated)} signature block(s) have a nearby date.",
            evidence=evidence, locations=dated, confidence="heuristic")

    def _nearby_date(self, doc: Doc, sig: Paragraph, flow: list[Paragraph],
                     pos_by_index: dict, window: int, skip: set):
        # own text
        hits = find_dates(sig.text)
        if hits:
            return hits[0][0]
        # same table row
        if sig.in_table and sig.table_pos is not None:
            t, r, _ = sig.table_pos
            if t < len(doc.tables):
                row = doc.tables[t].rows[r]
                for cell in row.cells:
                    ch = find_dates(cell.text())
                    if ch:
                        return ch[0][0]
        # bounded flow-order window
        pos = pos_by_index.get(sig.block_index)
        if pos is not None:
            lo = max(0, pos - window)
            hi = min(len(flow), pos + window + 1)
            # nearest first: the date that belongs to this signature is the
            # closest one, not merely the earliest in the window
            for i in sorted(range(lo, hi), key=lambda i: (abs(i - pos), i)):
                p = flow[i]
                if p is sig or p.block_index in skip:
                    continue
                ch = find_dates(p.text)
                if ch:
                    return ch[0][0]
        return None


RULE = Rule07()
