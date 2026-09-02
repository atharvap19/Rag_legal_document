"""The analysis pipeline: bytes in, findings out.

One place that owns the whole path an upload takes -- validate, extract,
evaluate, summarise -- so the route handlers stay thin and the pipeline can
be driven directly from a test or a script without FastAPI.

The LanguageTool adapter lives here too: it is the one rule dependency that
needs a process-wide resource, and rule 9 must never import
language_tool_python itself.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from app.extractor import Doc, build_doc
from app.engine import evaluate_all, get_rule, run_rule
from app.rules.base import Finding, LanguageIssue, RuleConfig
from app.rules.rule_09_language import DISABLED_RULE_IDS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUTPUT_DIR = os.path.join(ROOT, "output")

log = logging.getLogger(__name__)

MAX_MB = 20
MAX_BYTES = MAX_MB * 1024 * 1024
_ZIP_MAGIC = b"PK\x03\x04"

# disabled at the engine for speed; rule 9 also filters them defensively
_DISABLED = sorted(DISABLED_RULE_IDS)


class UploadRejected(Exception):
    """A bad upload, with the HTTP status the API should report."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


# --------------------------------------------------------------------------
# LanguageTool
# --------------------------------------------------------------------------
class LanguageToolChecker:
    """Adapts a language_tool_python instance to the LanguageChecker protocol,
    serialising all access through a lock."""

    def __init__(self, tool, lock: threading.Lock):
        self._tool = tool
        self._lock = lock

    def check(self, text: str) -> list[LanguageIssue]:
        if not text.strip():
            return []
        with self._lock:
            matches = self._tool.check(text)
        issues = []
        for m in matches:
            issues.append(LanguageIssue(
                message=getattr(m, "message", ""),
                context=getattr(m, "context", "") or "",
                offset=getattr(m, "offset", 0),
                # snake_case (language_tool_python >= 2.6) with camelCase fallback
                length=getattr(m, "error_length",
                               getattr(m, "errorLength", 0)),
                rule_id=getattr(m, "rule_id", getattr(m, "ruleId", "")),
                matched_text=getattr(m, "matched_text",
                                     getattr(m, "matchedText", "")) or "",
                replacements=list(getattr(m, "replacements", []) or [])[:5],
            ))
        return issues


def create_language_tool():
    """Create the LanguageTool instance (downloads its jar on first ever use
    and starts a JVM). Kept import-local so importing the app package does not
    require language_tool_python to be importable in test environments that
    inject a stub."""
    import language_tool_python

    tool = language_tool_python.LanguageTool("en-US")
    try:
        tool.disabled_rules.update(_DISABLED)
    except Exception:
        pass
    return tool


def build_checker(existing_tool=None,
                  lock: Optional[threading.Lock] = None) -> LanguageToolChecker:
    lock = lock or threading.Lock()
    tool = existing_tool or create_language_tool()
    return LanguageToolChecker(tool, lock)


# --------------------------------------------------------------------------
# Upload -> Doc
# --------------------------------------------------------------------------
def validate_upload(filename: Optional[str], stream) -> tuple[str, bytes]:
    """Check extension, size and zip magic before python-docx ever sees it."""
    name = filename or "document.docx"
    if not name.lower().endswith(".docx"):
        raise UploadRejected(
            415, "Only .docx files are accepted (got a different type).")
    data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise UploadRejected(
            413, f"File exceeds the {MAX_MB} MB upload limit.")
    if len(data) < 4 or data[:4] != _ZIP_MAGIC:
        raise UploadRejected(
            422, "File is not a valid .docx (OOXML zip) package.")
    return name, data


def extract(filename: Optional[str], stream) -> tuple[str, Doc]:
    """Validate an upload and turn it into the Doc model the rules consume."""
    name, data = validate_upload(filename, stream)
    try:
        return name, build_doc(io.BytesIO(data), name)
    except Exception as exc:
        raise UploadRejected(422, f"Could not parse the document: {exc}")


# --------------------------------------------------------------------------
# Doc -> findings
# --------------------------------------------------------------------------
def analyze(doc: Doc, config: RuleConfig) -> list[Finding]:
    return evaluate_all(doc, config)


def analyze_one(rule_id: int, doc: Doc, config: RuleConfig) -> Finding:
    rule = get_rule(rule_id)
    if rule is None:
        raise UploadRejected(404, f"No rule with id {rule_id}.")
    return run_rule(rule, doc, config)


def doc_to_dict(doc: Doc) -> dict:
    """The Doc model as plain JSON-able dicts, for /api/extract."""
    data = asdict(doc)
    core = data.get("core") or {}
    for key in ("created", "modified"):
        val = core.get(key)
        if isinstance(val, datetime):
            core[key] = val.isoformat()
    return data


# --------------------------------------------------------------------------
# Extraction output -- the Doc model, written to output/ so what the rules
# actually see can be read back after a run
# --------------------------------------------------------------------------
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def output_stem(filename: Optional[str]) -> str:
    """A safe file stem from a client-supplied upload name: only the basename
    survives and anything outside [A-Za-z0-9._-] collapses, so a crafted
    filename cannot write outside OUTPUT_DIR."""
    base = os.path.basename(filename or "document.docx")
    base = re.sub(r"\.docx$", "", base, flags=re.IGNORECASE)
    stem = _UNSAFE_NAME.sub("_", base).strip("._")
    return (stem or "document")[:100]


def extraction_path(filename: Optional[str]) -> str:
    return os.path.join(OUTPUT_DIR,
                        f"{output_stem(filename)}.extracted.json")


def _json_default(value):
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def save_extraction(data: dict, filename: Optional[str]) -> str:
    """Write an extraction dict (from :func:`doc_to_dict`) to
    ``output/<stem>.extracted.json`` and return the path. One file per
    document name, overwritten on each run, so the directory always holds the
    latest extraction for each document."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = extraction_path(filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False,
                  default=_json_default)
    return path


def save_extraction_quietly(data: dict, filename: Optional[str]):
    """save_extraction, but a disk problem must never fail an analysis.
    Returns the path, or None if the write failed."""
    try:
        return save_extraction(data, filename)
    except OSError as exc:
        log.warning("Could not write extraction output for %r: %s",
                    filename, exc)
        return None
