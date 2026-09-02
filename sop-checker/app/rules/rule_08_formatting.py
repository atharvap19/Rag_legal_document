"""Rule 8 - Font and spacing consistency.

Establish the dominant body font, size and spacing from body-level paragraphs,
weighted by character count, then flag body paragraphs that deviate. Headings,
captions, table cells and list paragraphs are excluded from both the baseline
and the check. Bold emphasis inside a paragraph is fine because we compare each
paragraph's *dominant* (char-weighted) face, so a few bold words do not shift
it; a whole paragraph in a different face does.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

from app.extractor import Doc, Paragraph
from .base import Rule, RuleConfig, Finding, heading_level


MIN_BASELINE_PARAS = 3


def _is_caption(p: Paragraph) -> bool:
    sid = (p.props.style_id or "").lower()
    name = (p.props.style_name or "").lower()
    return "caption" in sid or "caption" in name


def _selected(p: Paragraph) -> bool:
    if p.in_table or p.from_textbox:
        return False
    if heading_level(p) is not None:
        return False
    if p.props.is_list or _is_caption(p):
        return False
    return p.word_count() >= 3


def _weighted_mode(counter: Counter):
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _para_family(p: Paragraph) -> Optional[str]:
    c: Counter = Counter()
    for r in p.runs:
        if r.text.strip() and r.font.name:
            c[r.font.name] += len(r.text)
    return _weighted_mode(c)


def _para_size(p: Paragraph) -> Optional[float]:
    c: Counter = Counter()
    for r in p.runs:
        if r.text.strip() and r.font.size is not None:
            c[r.font.size] += len(r.text)
    return _weighted_mode(c)


class Rule08(Rule):
    id = 8
    name = "Font and spacing consistency"
    severity = "warning"
    description = ("Body paragraphs should share one font family, size, line "
                   "spacing, paragraph spacing and alignment.")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        paras = [p for p in doc.body_paragraphs() if _selected(p)]
        if len(paras) < MIN_BASELINE_PARAS:
            return self.fail(
                f"Only {len(paras)} body paragraph(s) -- too few to establish "
                "a formatting baseline, so the document has almost no body "
                "text to be consistent about.")

        fam_c: Counter = Counter()
        size_c: Counter = Counter()
        ls_c: Counter = Counter()
        sb_c: Counter = Counter()
        sa_c: Counter = Counter()
        align_c: Counter = Counter()

        for p in paras:
            chars = len(p.text)
            for r in p.runs:
                if r.text.strip() and r.font.name:
                    fam_c[r.font.name] += len(r.text)
                if r.text.strip() and r.font.size is not None:
                    size_c[r.font.size] += len(r.text)
            ls_c[_r(p.props.line_spacing)] += chars
            sb_c[_r(p.props.space_before)] += chars
            sa_c[_r(p.props.space_after)] += chars
            align_c[p.props.alignment or "LEFT"] += chars

        base_family = _weighted_mode(fam_c)
        base_size = _weighted_mode(size_c)
        base_ls = _weighted_mode(ls_c)
        base_sb = _weighted_mode(sb_c)
        base_sa = _weighted_mode(sa_c)
        base_align = _weighted_mode(align_c)

        evidence: list[str] = []
        locations: list[str] = []

        for p in paras:
            deviations = []
            fam = _para_family(p)
            if base_family and fam and fam != base_family:
                deviations.append(f"font {fam!r} vs {base_family!r}")
            size = _para_size(p)
            if base_size and size and size != base_size:
                deviations.append(f"size {size} vs {base_size}")
            if base_ls is not None and _r(p.props.line_spacing) != base_ls:
                deviations.append(
                    f"line spacing {_r(p.props.line_spacing)} vs {base_ls}")
            if _r(p.props.space_before) != base_sb:
                deviations.append(
                    f"space before {_r(p.props.space_before)} vs {base_sb}")
            if _r(p.props.space_after) != base_sa:
                deviations.append(
                    f"space after {_r(p.props.space_after)} vs {base_sa}")
            if (p.props.alignment or "LEFT") != base_align:
                deviations.append(
                    f"alignment {(p.props.alignment or 'LEFT')} vs {base_align}")
            if deviations:
                evidence.append(f"{p.location}: " + "; ".join(deviations))
                locations.append(p.location)

        baseline = (f"font {base_family!r}, size {base_size}, line spacing "
                    f"{base_ls}, space {base_sb}/{base_sa}, align {base_align}")
        if locations:
            return self.fail(
                f"{len(locations)} body paragraph(s) deviate from the dominant "
                f"formatting ({baseline}).",
                evidence=evidence, locations=locations, confidence="heuristic")
        return self.ok(
            f"All {len(paras)} body paragraphs share the dominant formatting "
            f"({baseline}).",
            confidence="heuristic")


def _r(value):
    """Round floats for stable comparison; pass through None."""
    if value is None:
        return None
    return round(float(value), 2)


RULE = Rule08()
