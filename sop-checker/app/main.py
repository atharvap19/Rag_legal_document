"""FastAPI app: schemas, routes, lifespan-managed LanguageTool, static mount.

The page and API share one origin, so there is deliberately no CORS
middleware. Static files are mounted after the API router so they never
shadow the /api routes.

Handlers are plain ``def`` (not ``async def``) on purpose: python-docx parsing
and LanguageTool calls are blocking and CPU-bound, so FastAPI runs them in its
threadpool and the event loop stays free. Making these async would block every
other request for the duration.
"""
from __future__ import annotations

import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import (
    APIRouter, FastAPI, File, Form, HTTPException, Request, UploadFile,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import pipeline
from app.engine import rules_metadata
from app.pipeline import (
    LanguageToolChecker, UploadRejected, create_language_tool,
)
from app.rules.base import LanguageChecker, RuleConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEB_DIR = os.path.join(ROOT, "web")


# --------------------------------------------------------------------------
# Schemas -- FindingModel mirrors the Finding dataclass exactly
# --------------------------------------------------------------------------
class FindingModel(BaseModel):
    rule_id: int
    rule_name: str
    passed: Optional[bool] = Field(
        None, description="None means the rule could not be evaluated.")
    severity: str
    message: str
    evidence: list[str] = []
    locations: list[str] = []
    confidence: str


class RuleMeta(BaseModel):
    id: int
    name: str
    severity: str
    description: str


class Summary(BaseModel):
    total: int
    passed: int
    failed: int
    not_evaluated: int
    errors: int
    warnings: int


class AnalyzeResponse(BaseModel):
    filename: str
    summary: Summary
    findings: list[FindingModel]
    extraction_file: Optional[str] = Field(
        None, description="Path of the saved extraction JSON, relative to the "
                          "project root; null if it could not be written.")
    extraction: Optional[dict] = Field(
        None, description="The extracted Doc model, present only when the "
                          "request asked for it.")


class AnalyzeConfig(BaseModel):
    """Optional client-supplied configuration for an analysis run."""
    required_sections: Optional[list[str]] = None
    ignore_words: Optional[list[str]] = None
    default_authors: Optional[list[str]] = None
    confidentiality_terms: Optional[list[str]] = None
    doc_id_pattern: Optional[str] = None
    readability_min_words: Optional[int] = None
    readability_flesch_min: Optional[float] = None
    readability_fog_max: Optional[float] = None
    title_match_threshold: Optional[float] = None
    date_window: Optional[int] = None

    def to_rule_config(self, filename: str, language_checker) -> RuleConfig:
        cfg = RuleConfig(filename=filename, language_checker=language_checker)
        for field_name, value in self.model_dump(exclude_none=True).items():
            if hasattr(cfg, field_name):
                setattr(cfg, field_name, value)
        return cfg


def finding_to_model(f) -> FindingModel:
    return FindingModel(
        rule_id=f.rule_id, rule_name=f.rule_name, passed=f.passed,
        severity=f.severity, message=f.message, evidence=f.evidence,
        locations=f.locations, confidence=f.confidence,
    )


def summarize(findings) -> Summary:
    passed = sum(1 for f in findings if f.passed is True)
    failed = sum(1 for f in findings if f.passed is False)
    na = sum(1 for f in findings if f.passed is None)
    errors = sum(1 for f in findings
                 if f.passed is False and f.severity == "error")
    warnings = sum(1 for f in findings
                   if f.passed is False and f.severity == "warning")
    return Summary(total=len(findings), passed=passed, failed=failed,
                   not_evaluated=na, errors=errors, warnings=warnings)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
router = APIRouter(prefix="/api")


def _extract(file: UploadFile):
    """Run the upload through the pipeline, mapping its rejections to HTTP."""
    try:
        return pipeline.extract(file.filename, file.file)
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, exc.detail)


def _save_extraction(doc, filename: str) -> tuple[Optional[str], dict]:
    """Serialise the Doc, write it to output/, and hand back both the saved
    path (project-relative, for display) and the dict."""
    data = pipeline.doc_to_dict(doc)
    path = pipeline.save_extraction_quietly(data, filename)
    rel = os.path.relpath(path, ROOT).replace(os.sep, "/") if path else None
    return rel, data


def _config(config: Optional[str], filename: str, checker) -> RuleConfig:
    if not config:
        return AnalyzeConfig().to_rule_config(filename, checker)
    try:
        data = json.loads(config)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, f"Invalid config JSON: {exc}")
    try:
        model = AnalyzeConfig(**data)
    except Exception as exc:
        raise HTTPException(422, f"Invalid config: {exc}")
    return model.to_rule_config(filename, checker)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/rules", response_model=list[RuleMeta])
def get_rules():
    return rules_metadata()


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: Request,
            file: UploadFile = File(...),
            config: Optional[str] = Form(None),
            include_extraction: bool = Form(False)):
    """Run every rule. The extraction is always written to output/; it is only
    echoed in the response when ``include_extraction`` is set, since a long
    document's Doc model dwarfs its findings."""
    name, doc = _extract(file)
    checker = getattr(request.app.state, "language_checker", None)
    cfg = _config(config, name, checker)
    findings = pipeline.analyze(doc, cfg)
    saved, data = _save_extraction(doc, name)
    return AnalyzeResponse(
        filename=name,
        summary=summarize(findings),
        findings=[finding_to_model(f) for f in findings],
        extraction_file=saved,
        extraction=data if include_extraction else None,
    )


@router.post("/analyze/{rule_id}", response_model=FindingModel)
def analyze_one(rule_id: int, request: Request,
                file: UploadFile = File(...),
                config: Optional[str] = Form(None)):
    name, doc = _extract(file)
    checker = getattr(request.app.state, "language_checker", None)
    cfg = _config(config, name, checker)
    try:
        return finding_to_model(pipeline.analyze_one(rule_id, doc, cfg))
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, exc.detail)


@router.post("/extract")
def extract_endpoint(file: UploadFile = File(...), save: bool = True):
    """The extracted Doc model on its own, with ``saved_to`` naming the copy
    left in output/ (pass ``?save=false`` to skip writing that copy)."""
    name, doc = _extract(file)
    if not save:
        return pipeline.doc_to_dict(doc)
    saved, data = _save_extraction(doc, name)
    return {**data, "saved_to": saved}


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
def create_app(language_checker: Optional[LanguageChecker] = None) -> FastAPI:
    """Build the app. If ``language_checker`` is provided (tests), it is used
    as-is; otherwise a real LanguageTool JVM is created in the lifespan."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        lock = threading.Lock()
        app.state.lt_lock = lock
        app.state._owned_tool = None
        if language_checker is not None:
            app.state.language_checker = language_checker
        else:
            tool = create_language_tool()
            app.state._owned_tool = tool
            app.state.language_checker = LanguageToolChecker(tool, lock)
        try:
            yield
        finally:
            owned = getattr(app.state, "_owned_tool", None)
            if owned is not None:
                try:
                    owned.close()
                except Exception:
                    pass

    app = FastAPI(
        title="SOP Compliance Checker",
        version="1.0.0",
        description="Library/algorithm-based .docx SOP compliance checker.",
        lifespan=lifespan,
    )

    # API first, so static serving cannot shadow it
    app.include_router(router)

    os.makedirs(WEB_DIR, exist_ok=True)
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app


app = create_app()
