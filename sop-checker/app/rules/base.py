"""What every rule shares: the Finding/Rule contract, RuleConfig, and the
text and document helpers the rules build on.

Rules see only the extractor's :class:`Doc` model and a :class:`RuleConfig`.
They never import python-docx, lxml or FastAPI -- that boundary is what keeps
them unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable
import re
from datetime import datetime
from typing import Optional
from dateutil import parser as date_parser
from rapidfuzz import fuzz
from typing import Iterable, Optional
from app.extractor import Doc, HeadingEntry, Paragraph, Table


# ========================================================================
# Finding, Rule protocol, and RuleConfig
# ========================================================================
Severity = str          # "error" | "warning" | "info"
Confidence = str        # "certain" | "heuristic"


# --------------------------------------------------------------------------
# Finding
# --------------------------------------------------------------------------
@dataclass
class Finding:
    rule_id: int
    rule_name: str
    passed: Optional[bool]          # None => could not evaluate
    severity: Severity
    message: str
    evidence: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    confidence: Confidence = "certain"


# --------------------------------------------------------------------------
# Language checker interface (rule 9)
# --------------------------------------------------------------------------
@dataclass
class LanguageIssue:
    message: str
    context: str
    offset: int
    length: int
    rule_id: str
    matched_text: str = ""
    replacements: list[str] = field(default_factory=list)


@runtime_checkable
class LanguageChecker(Protocol):
    """What rule 9 needs from a LanguageTool wrapper. Injected via config so
    the rule stays pure and tests can stub it without a JVM."""

    def check(self, text: str) -> list[LanguageIssue]: ...


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_AUTHOR_NAMES = [
    "user", "windows user", "administrator", "admin", "owner",
    "author", "microsoft office user", "your name", "",
    # names left behind by whatever generated the file -- these identify a
    # tool, not a person, and are exactly what this rule exists to catch
    "python-docx", "python docx", "docx", "microsoft word", "word",
    "libreoffice", "openoffice", "apache poi", "docx4j", "unknown", "n/a",
]

# A suggestion, not a default: which sections an SOP must carry is a house
# rule, so rule 10 asks for the list rather than assuming one. This is what
# the page offers as placeholder text.
COMMON_REQUIRED_SECTIONS = [
    "Purpose", "Scope", "Responsibilities", "Procedure", "References",
]

DEFAULT_CONFIDENTIALITY_TERMS = [
    "confidential", "internal use only", "proprietary", "restricted",
]


@dataclass
class RuleConfig:
    filename: Optional[str] = None
    default_authors: list[str] = field(
        default_factory=lambda: list(DEFAULT_AUTHOR_NAMES))
    # empty on purpose -- rule 10 reports "not evaluated" until told what
    # sections to look for
    required_sections: list[str] = field(default_factory=list)
    ignore_words: list[str] = field(default_factory=list)
    language_checker: Optional[LanguageChecker] = None
    min_words_for_language: int = 4
    # below this many words of body text the scores are too noisy to report
    readability_min_words: int = 50
    readability_flesch_min: float = 30.0
    readability_fog_max: float = 18.0
    doc_id_pattern: str = r"\b[A-Z]{2,}[-_/][A-Z0-9]+(?:[-_/][A-Z0-9]+)*\b"
    # rule 1: how close a filename must be to the title to count as a
    # match. 1.0 restores the old exact-equality behaviour.
    title_match_threshold: float = 0.85
    confidentiality_terms: list[str] = field(
        default_factory=lambda: list(DEFAULT_CONFIDENTIALITY_TERMS))
    date_window: int = 4          # rule 7 proximity, in flow-order blocks
    future_grace_days: int = 1    # rule 3 tolerance for clock skew

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "RuleConfig":
        cfg = cls()
        if not data:
            return cfg
        for key, value in data.items():
            if hasattr(cfg, key) and key != "language_checker":
                setattr(cfg, key, value)
        return cfg


# --------------------------------------------------------------------------
# Rule base
# --------------------------------------------------------------------------
class Rule:
    id: int = 0
    name: str = ""
    severity: Severity = "error"
    description: str = ""

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        raise NotImplementedError

    # -- finding constructors -----------------------------------------
    def _make(self, passed: Optional[bool], message: str,
              evidence: Optional[list[str]] = None,
              locations: Optional[list[str]] = None,
              severity: Optional[Severity] = None,
              confidence: Confidence = "certain") -> Finding:
        return Finding(
            rule_id=self.id,
            rule_name=self.name,
            passed=passed,
            severity=severity or self.severity,
            message=message,
            evidence=list(evidence or []),
            locations=list(locations or []),
            confidence=confidence,
        )

    def ok(self, message: str, **kw) -> Finding:
        return self._make(True, message, **kw)

    def fail(self, message: str, **kw) -> Finding:
        return self._make(False, message, **kw)

    def na(self, message: str, **kw) -> Finding:
        # a rule that cannot evaluate is never a pass
        kw.setdefault("severity", "info")
        return self._make(None, message, **kw)


def rule_metadata(rule: Rule) -> dict:
    return {
        "id": rule.id,
        "name": rule.name,
        "severity": rule.severity,
        "description": rule.description,
    }


# ========================================================================
# Pure text helpers: normalisation, versions, dates, similarity
# ========================================================================
# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------
_VERSION_SUFFIX = re.compile(
    r"[\s._-]*(?:v|ver|version|rev|revision|draft|final|copy)\.?\s*"
    r"\d+(?:\.\d+)*\s*$",
    re.IGNORECASE,
)
_SEP = re.compile(r"[\s._\-/\\]+")


def strip_extension(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]{1,5}$", "", name or "")


def normalize_key(text: str) -> str:
    """Casefold, collapse separators/whitespace to single spaces, trim."""
    if not text:
        return ""
    return _SEP.sub(" ", text).strip().casefold()


# A bare trailing keyword needs a separator in front of it, or "improv"
# would lose its "v". The (n) marker is capped at three digits so a year
# like "(2024)" survives.
_COPY_SUFFIX = re.compile(
    r"(?:[\s._-]+(?:draft|final|copy|revised|updated)|\s*\(\d{1,3}\))\s*$",
    re.IGNORECASE,
)


def strip_noise(text: str) -> str:
    """Peel trailing version and duplicate markers until none are left, so
    'Guide final (1)' and 'Guide v2' both reduce to 'Guide'."""
    prev = None
    while prev != text:
        prev = text
        text = _VERSION_SUFFIX.sub("", text)
        text = _COPY_SUFFIX.sub("", text)
    return text


def normalize_filename(name: str) -> str:
    return normalize_key(strip_noise(strip_extension(name or "")))


def normalize_title(text: str) -> str:
    return normalize_key(strip_noise(text or ""))


def strip_leading_number(text: str) -> str:
    """Drop a leading section number like '3.', '3.1', 'A.' from a heading."""
    return re.sub(r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z])[.)]?\s+", "", text or "")


# --------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------
_DIGIT_RUN = re.compile(r"\d+")


def _digit_sets(text: str) -> set[str]:
    """Digit runs with leading zeros dropped, so 'sop-001' -> {'1'}."""
    return {run.lstrip("0") or "0" for run in _DIGIT_RUN.findall(text)}


def digits_conflict(a: str, b: str) -> bool:
    """True when both strings carry numbers and those numbers disagree.

    Guards the fuzzy comparison: 'sop-002 cleaning' and 'sop-001 cleaning'
    are 96% alike as characters but name different documents, and for a
    compliance check that difference is the whole point.
    """
    da, db = _digit_sets(a), _digit_sets(b)
    return bool(da and db and da != db)


def similarity(a: str, b: str) -> float:
    """Fuzzy closeness of two normalised keys, 0.0-1.0.

    The better of a character-level ratio and a token-sorted ratio, so that
    reordered words ('procedure for cleaning' vs 'cleaning procedure') score
    on their content rather than their arrangement. rapidfuzz reports 0-100;
    the rules work in 0-1, so scale here and nowhere else.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return max(fuzz.ratio(a, b), fuzz.token_sort_ratio(a, b)) / 100.0


# --------------------------------------------------------------------------
# versions
# --------------------------------------------------------------------------
_VERSION_RE = re.compile(
    r"\b(?:v|ver|version|rev|revision|draft)\.?\s*[:#]?\s*"
    r"(\d+(?:\.\d+){0,3})\b",
    re.IGNORECASE,
)


def find_versions(text: str) -> list[str]:
    """Return normalised version numbers found in text (e.g. '1.2')."""
    if not text:
        return []
    return [m.group(1) for m in _VERSION_RE.finditer(text)]


# --------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------
_MONTHS = (
    r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|"
    r"Nov(?:ember)?|Dec(?:ember)?"
)
_DATE_PATTERNS = [
    re.compile(r"\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b"),            # 2024-01-31
    re.compile(r"\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b"),          # 31/01/2024
    re.compile(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\.?,?\s+\d{{2,4}}\b", re.I),  # 31 Jan 2024
    re.compile(rf"\b(?:{_MONTHS})\.?\s+\d{{1,2}},?\s+\d{{2,4}}\b", re.I),  # Jan 31, 2024
]


def find_date_strings(text: str) -> list[str]:
    if not text:
        return []
    found: list[tuple[int, str]] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            found.append((m.start(), m.group(0)))
    found.sort()
    # de-duplicate overlapping matches by span start
    seen_spans: list[str] = []
    for _, s in found:
        if s not in seen_spans:
            seen_spans.append(s)
    return seen_spans


def parse_date(text: str, dayfirst: bool = True) -> Optional[datetime]:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return date_parser.parse(text, dayfirst=dayfirst, fuzzy=False)
    except (ValueError, OverflowError, TypeError):
        return None


def find_dates(text: str, dayfirst: bool = True) -> list[tuple[str, datetime]]:
    out: list[tuple[str, datetime]] = []
    for s in find_date_strings(text):
        dt = parse_date(s, dayfirst=dayfirst)
        if dt is not None:
            out.append((s, dt))
    return out


def has_date(text: str) -> bool:
    return bool(find_dates(text))


# ========================================================================
# Doc-structure helpers: headings, tables, signatures
# ========================================================================
REVISION_HEADING_RE = re.compile(
    r"\b(?:revision|version|change|amendment|document)\s+"
    r"(?:history|control|log|record)\b"
    r"|\brecord of (?:changes|amendments)\b"
    r"|\bchange log\b",
    re.IGNORECASE,
)

_DATE_HDR = re.compile(r"\bdate\b", re.IGNORECASE)
_REVISION_HDR = re.compile(
    r"\b(?:rev(?:ision)?|version|ver|no\.?|change|description|author|"
    r"amendment|issue)\b",
    re.IGNORECASE,
)

SIGNATURE_LABEL_RE = re.compile(
    r"\b(?:prepared|reviewed|approved|authori[sz]ed|checked|issued|verified|"
    r"released)\s+by\b|\bsignature\b|\bsign(?:ed)?\s*&?\s*date\b",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------
# headings
# --------------------------------------------------------------------------
def heading_level(p: Paragraph) -> Optional[int]:
    """Heading level from Word's own declaration, falling back to the
    extractor's visual inference for documents that used direct formatting
    instead of heading styles."""
    if p.props.heading_level is not None:
        return p.props.heading_level
    if p.props.outline_level is not None:
        return p.props.outline_level + 1
    return p.props.inferred_heading_level


def is_heading(p: Paragraph) -> bool:
    return heading_level(p) is not None and not p.in_table


def iter_headings(doc: Doc) -> Iterable[HeadingEntry]:
    for p in doc.flow_ordered():
        lvl = heading_level(p)
        if lvl is not None and not p.in_table and p.text.strip():
            yield HeadingEntry(level=lvl, text=p.text.strip(),
                               block_index=p.block_index, location=p.location)


def first_heading1(doc: Doc) -> Optional[Paragraph]:
    for p in doc.flow_ordered():
        if not p.in_table and heading_level(p) == 1 and p.text.strip():
            return p
    return None


def next_heading_block_index(doc: Doc, after: int,
                             max_level: Optional[int] = None) -> int:
    best = None
    for h in iter_headings(doc):
        if h.block_index > after and (max_level is None or h.level <= max_level):
            if best is None or h.block_index < best:
                best = h.block_index
    return best if best is not None else 10 ** 9


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------
_LABEL_MAX_WORDS = 4


def table_labeled_values(doc: Doc, pattern: re.Pattern) -> list[tuple[str, str]]:
    """Values from ``label | value`` table rows whose label matches.

    Document-information tables ('Document Title | Employee Onboarding SOP',
    'Version | 2.1') are a staple of real SOPs, and because the label and the
    value sit in different cells no regex over paragraph text ever sees them
    as one string. Returns every (value, location) match in document order so
    the caller can reject ones that do not look like what it wants -- a
    revision table's 'Version' header column would otherwise hand back the
    next header cell.
    """
    out: list[tuple[str, str]] = []
    for table in doc.tables:
        for row in table.rows:
            if len(row.cells) < 2:
                continue
            label = row.cells[0].text().strip()
            if not label or len(label.split()) > _LABEL_MAX_WORDS:
                continue
            if not pattern.search(label):
                continue
            value = row.cells[1].text().strip()
            if value:
                out.append((value, f"Table {table.table_index + 1}"))
    return out


def table_header_cells(table: Table) -> list[str]:
    if not table.rows:
        return []
    return [normalize_key(c.text()) for c in table.rows[0].cells]


def table_is_populated(table: Table) -> bool:
    """At least one data row (beyond the header) with text."""
    if len(table.rows) < 2:
        return False
    for row in table.rows[1:]:
        if any(c.text().strip() for c in row.cells):
            return True
    return False


def find_revision_heading(doc: Doc) -> Optional[HeadingEntry]:
    for h in iter_headings(doc):
        if REVISION_HEADING_RE.search(h.text):
            return h
    # also allow a bold-ish plain paragraph acting as a heading
    for p in doc.flow_ordered():
        if not p.in_table and REVISION_HEADING_RE.search(p.text) \
                and len(p.text.split()) <= 6:
            return HeadingEntry(level=heading_level(p) or 99, text=p.text.strip(),
                                block_index=p.block_index, location=p.location)
    return None


def looks_like_revision_table(table: Table) -> bool:
    headers = table_header_cells(table)
    if not headers:
        return False
    joined = " ".join(headers)
    return bool(_DATE_HDR.search(joined) and _REVISION_HDR.search(joined))


def find_revision_table(doc: Doc) -> Optional[Table]:
    """A table whose header row looks like a revision log (date + a revision
    keyword). Prefer one that follows a revision-history heading."""
    heading = find_revision_heading(doc)
    candidates = [t for t in doc.tables if looks_like_revision_table(t)]
    if heading is not None:
        after = [t for t in candidates if t.block_index > heading.block_index]
        if after:
            return min(after, key=lambda t: t.block_index)
    if candidates:
        return min(candidates, key=lambda t: t.block_index)
    return None


def header_column_index(table: Table, pattern: re.Pattern) -> Optional[int]:
    for i, cell in enumerate(table_header_cells(table)):
        if pattern.search(cell):
            return i
    return None


def tables_after_heading(doc: Doc, heading: HeadingEntry) -> list[Table]:
    end = next_heading_block_index(doc, heading.block_index,
                                   max_level=heading.level)
    return [t for t in doc.tables
            if heading.block_index < t.block_index < end]


# --------------------------------------------------------------------------
# signatures
# --------------------------------------------------------------------------
def signature_paragraphs(doc: Doc) -> list[Paragraph]:
    """Paragraphs (body or table cell) that carry a signature-block label."""
    out = []
    for p in doc.flow_ordered():
        if SIGNATURE_LABEL_RE.search(p.text):
            out.append(p)
    return out


def normalized_heading_text(text: str) -> str:
    return normalize_key(strip_leading_number(text))
