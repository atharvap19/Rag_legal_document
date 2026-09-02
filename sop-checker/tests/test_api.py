"""API tests via TestClient. LanguageTool is stubbed so no JVM is needed."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app import pipeline
from tests import make_fixtures as mf
from tests.conftest import StubChecker

DOCX_MIME = ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document")

# rule 10 has no built-in section list, so a request that wants a verdict
# from it has to say what the sections are
SECTIONS = {"required_sections": ["Purpose", "Scope", "Responsibilities",
                                  "Procedure", "References"]}


@pytest.fixture(autouse=True)
def output_dir(tmp_path, monkeypatch):
    """Keep extraction output out of the real output/ directory."""
    d = tmp_path / "output"
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", str(d))
    return d


@pytest.fixture
def client():
    app = create_app(language_checker=StubChecker())
    with TestClient(app) as c:
        yield c


def _golden_bytes():
    return mf.to_bytes(mf.golden_sop()).getvalue()


def _files(name="Quality Control Procedure.docx", data=None):
    return {"file": (name, data if data is not None else _golden_bytes(),
                     DOCX_MIME)}


# ---- basic endpoints ----------------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_rules_metadata(client):
    r = client.get("/api/rules")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 13
    assert body[0]["id"] == 1 and body[0]["name"]


# ---- analyze happy path -------------------------------------------------
def test_analyze_happy_path(client):
    r = client.post("/api/analyze", files=_files(),
                    data={"config": json.dumps(SECTIONS)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filename"].endswith(".docx")
    assert body["summary"]["total"] == 13
    assert body["summary"]["passed"] == 13, body["findings"]
    assert len(body["findings"]) == 13


def test_analyze_without_sections_leaves_rule_10_unevaluated(client):
    """No section list configured -> rule 10 asks for one instead of
    guessing a standard the document was never written to."""
    body = client.post("/api/analyze", files=_files()).json()
    rule10 = {f["rule_id"]: f for f in body["findings"]}[10]
    assert rule10["passed"] is None
    assert "Please enter the sections" in rule10["message"]
    assert body["summary"]["not_evaluated"] == 1


def test_analyze_with_config_changes_result(client):
    cfg = json.dumps({"required_sections": ["Nonexistent Section"]})
    r = client.post("/api/analyze", files=_files(),
                    data={"config": cfg})
    assert r.status_code == 200
    findings = {f["rule_id"]: f for f in r.json()["findings"]}
    assert findings[10]["passed"] is False


# ---- validation errors --------------------------------------------------
def test_analyze_rejects_non_docx(client):
    r = client.post("/api/analyze",
                    files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415
    assert "docx" in r.json()["detail"].lower()


def test_analyze_rejects_corrupt_zip(client):
    r = client.post("/api/analyze",
                    files=_files(name="broken.docx", data=b"not a zip at all"))
    assert r.status_code == 422
    assert "valid" in r.json()["detail"].lower()


def test_analyze_rejects_oversize(client, monkeypatch):
    monkeypatch.setattr(pipeline, "MAX_BYTES", 64)
    big = b"PK\x03\x04" + b"x" * 200
    r = client.post("/api/analyze", files=_files(name="big.docx", data=big))
    assert r.status_code == 413


# ---- single-rule execution ----------------------------------------------
def test_analyze_single_rule(client):
    r = client.post("/api/analyze/11", files=_files())
    assert r.status_code == 200
    body = r.json()
    assert body["rule_id"] == 11
    assert body["passed"] is True


def test_analyze_single_rule_unknown_id(client):
    r = client.post("/api/analyze/999", files=_files())
    assert r.status_code == 404


# ---- extract ------------------------------------------------------------
def test_extract_returns_doc_model(client):
    r = client.post("/api/extract", files=_files())
    assert r.status_code == 200
    body = r.json()
    for key in ("filename", "core", "blocks", "tables", "sections", "styles"):
        assert key in body
    assert any(b.get("kind") == "paragraph" for b in body["blocks"])


# ---- extraction output --------------------------------------------------
def test_extract_writes_output_file(client, output_dir):
    r = client.post("/api/extract", files=_files())
    assert r.status_code == 200
    saved = output_dir / "Quality_Control_Procedure.extracted.json"
    assert saved.exists()
    on_disk = json.loads(saved.read_text(encoding="utf-8"))
    assert on_disk["blocks"] == r.json()["blocks"]
    # the saved path is reported, but never written into the file itself
    assert r.json()["saved_to"].endswith(
        "Quality_Control_Procedure.extracted.json")
    assert "saved_to" not in on_disk


def test_extract_can_skip_saving(client, output_dir):
    r = client.post("/api/extract?save=false", files=_files())
    assert r.status_code == 200
    assert "saved_to" not in r.json()
    assert not output_dir.exists()


def test_analyze_saves_extraction_and_can_echo_it(client, output_dir):
    plain = client.post("/api/analyze", files=_files()).json()
    assert plain["extraction"] is None          # not echoed unless asked
    assert plain["extraction_file"].startswith("output/") or \
        plain["extraction_file"].endswith(".extracted.json")
    assert (output_dir / "Quality_Control_Procedure.extracted.json").exists()

    full = client.post("/api/analyze", files=_files(),
                       data={"include_extraction": "true"}).json()
    assert full["extraction"]["filename"].endswith(".docx")
    assert full["extraction"]["blocks"]


def test_analyze_survives_unwritable_output(client, monkeypatch):
    def boom(*_a, **_kw):
        raise OSError("disk is read-only")
    monkeypatch.setattr(pipeline, "save_extraction", boom)
    body = client.post("/api/analyze", files=_files()).json()
    assert body["extraction_file"] is None
    assert body["summary"]["total"] == 13


def test_output_stem_cannot_escape_the_output_dir():
    assert pipeline.output_stem("../../etc/passwd.docx") == "passwd"
    assert pipeline.output_stem("C:\\Windows\\evil.docx") == "evil"
    assert pipeline.output_stem("Quality Control Procedure.docx") == \
        "Quality_Control_Procedure"
    assert pipeline.output_stem("...docx") == "document"
    assert pipeline.output_stem(None) == "document"
