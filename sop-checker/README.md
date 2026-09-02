# SOP Compliance Checker

Upload a `.docx` through a web page; the API reports which of 13 SOP rules the
document fails, with the offending text and its location.

Every check is **library- or algorithm-based**. There is no LLM anywhere, no
document renderer (no Word/LibreOffice/Aspose), and no PDF conversion.
Extraction reads the OOXML directly with `python-docx` + `lxml`. The front end
is plain ES-module JavaScript served by FastAPI itself, so page and API share
one origin and no CORS is needed.

## Architecture

```
extract/   .docx (OOXML) -> pure Doc model (flow order, styles, runs, fields, sections)
rules/     13 rules that consume ONLY the Doc model and return Findings
app/       FastAPI: routes, models, LanguageTool singleton, static mount
web/       index.html + app.js (ES module) + style.css  (no build step)
tests/     pytest suite + make_fixtures.py (fixtures rebuild on demand)
output/    extraction JSON written per run: what the rules actually saw
```

The boundary is strict: **rules never import `python-docx`, `lxml` or FastAPI.**
That is what makes them unit-testable and what let the extraction layer be
rewritten without touching a rule.

### Extraction (Phase 1)

- **Flow order** — walks `body` in document order, dispatching on `w:p`,
  `w:tbl`, `w:sdt` (content controls unwrapped) and pulling `w:txbxContent`
  (text boxes). Table cells recurse. Every paragraph gets a stable
  `block_index`; rule 7 depends on this ordering.
- **Run merging** — adjacent runs with identical resolved formatting are
  merged, so a word Word split across three runs (spell-check / rev-ids)
  becomes one again.
- **Style resolution** — a run's font is resolved up the chain (direct `w:rPr`
  → character style → paragraph style → `w:docDefaults`), following
  `w:basedOn` to the root. Theme fonts (`w:asciiTheme="minorHAnsi"`) resolve
  against `theme1.xml`. A theme-resolved face and a literally-named identical
  face are treated as the same.
- **Fields** — both complex (`w:fldChar`/`w:instrText`) and simple
  (`w:fldSimple`) fields are detected, so a live `PAGE` field is distinguished
  from a typed number.
- **Sections** — headers/footers (default / first / even), different-first-page
  and linked-to-previous state come from the relationship parts.

### The 13 rules (Phase 3)

| # | Rule | Severity |
|---|------|----------|
| 1 | Title / filename match | warning |
| 2 | Author name and role | error |
| 3 | Revision-history dates (parse / ordered / not future) | error |
| 4 | Version consistency | warning |
| 5 | Revision section present | error |
| 6 | Signature blocks | error |
| 7 | Dates near signatures | warning |
| 8 | Font and spacing consistency | warning |
| 9 | Language errors (LanguageTool) | warning |
| 10 | Required sections | error |
| 11 | Page numbers | warning |
| 12 | Readability (Flesch + Gunning Fog) | warning |
| 13 | Footer details (doc ID / confidentiality / page number) | warning |

Every rule returns exactly one `Finding`. A rule that *cannot* be evaluated
(e.g. the revision table is absent) returns `passed=None` ("not evaluated"),
never a pass.

### Known limitation — rule 11

A `.docx` does not record pagination; nothing in the XML says where page 4
begins. So rule 11 verifies a `PAGE` **field** exists in the footer and flags a
hardcoded typed number (the only way numbering actually goes wrong, since a
field-based footer is correct by construction). It cannot verify that page 7
renders "7", and the finding says so. No renderer is added to close this gap.

## Setup

Requires Python 3.11+ and a JRE (LanguageTool runs on the JVM).

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows;  source .venv/bin/activate on *nix
pip install -r requirements.txt
```

`language_tool_python` downloads its engine (~200 MB) on first use.

## Run

```bash
uvicorn app.main:app --reload
```

Then open <http://127.0.0.1:8000/>. The page is served by FastAPI; upload a
`.docx` and the results build themselves from `GET /api/rules`.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/analyze` | multipart `.docx` (+ optional JSON `config`) → all findings |
| POST | `/api/analyze/{id}` | run a single rule (handy for iterating on rule 8) |
| GET  | `/api/rules` | registry metadata; the client builds its UI from this |
| POST | `/api/extract` | the `Doc` model as JSON, for debugging extraction |
| GET  | `/api/health` | liveness |

### Extraction output

Every `/api/analyze` and `/api/extract` call also writes what the extractor
produced to `output/<document-name>.extracted.json` — the whole `Doc` model the
13 rules were handed, before any rule ran. One file per document name,
overwritten each run, so `output/` always holds the latest extraction; the
upload name is sanitised down to `[A-Za-z0-9._-]` so it cannot write outside
that directory. `POST /api/extract?save=false` skips the write, and a failed
write is logged rather than failing the analysis (`extraction_file` comes back
`null`).

The same JSON is shown in the page's **Extraction output** section, either
whole or through a *Summary* view — counts, core properties, headings, field
codes, header/footer text, table shapes, and the first paragraphs with their
resolved fonts. `/api/analyze` only echoes it in the response body when the
request sends `include_extraction=true` (the page does), since a long
document's `Doc` model dwarfs its findings.

Handlers are plain `def` (not `async def`) on purpose: python-docx parsing and
LanguageTool calls are blocking/CPU-bound, so FastAPI runs them in its
threadpool and the event loop stays free. The LanguageTool instance is created
once in the lifespan, stored on `app.state`, and guarded by a `threading.Lock`.

Uploads are validated (extension, zip magic bytes, size cap) with `415` for a
non-docx, `413` for oversize, and `422` for a corrupt file.

### Documents that never used heading styles

Many real SOPs mark a section by bolding a line and bumping its size rather
than applying Heading 1/2. Word records nothing for those, so `outline_level`
and `heading_level` are both `None` and the file looks to the rules like one
unstructured run of text -- which used to leave rules 1 and 4 unevaluated and
made rule 10 report that the document had no headings at all.

The extractor therefore infers structure for such files, filling
`props.inferred_heading_level`: a paragraph reads as a heading when it is
short, is not a sentence, and is either larger than the dominant body text or
bold where bold is otherwise rare. Levels come from ranking the sizes found.
The pass runs **only when the document declared no heading anywhere**, so a
properly styled file is always taken at its word, and the field stays visible
and separate in the extraction JSON -- it is never confused with what Word
actually said.

The same documents state their title, version and author in a
document-information table (`Version | 2.1`), where the label and the value
are separate cells that no regex over paragraph text can see as one string.
`table_labeled_values` in [`app/rules/base.py`](app/rules/base.py) reads those
rows for rules 1, 2 and 4.

### Optional config (JSON, sent as the `config` form field)

```json
{
  "required_sections": ["Purpose", "Scope", "Procedure"],
  "_comment": "required_sections has no default: rule 10 reports 'not evaluated' until you supply one",
  "ignore_words": ["Acme", "QMS"],
  "readability_min_words": 50,
  "readability_flesch_min": 30,
  "readability_fog_max": 18,
  "doc_id_pattern": "\\b[A-Z]{2,}[-_/][A-Z0-9]+\\b"
}
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite stubs LanguageTool, so **no JVM is needed to run the tests**. It
covers the extraction hazards (mid-word run splits, theme vs literal fonts,
paragraph/table interleaving, nested tables, multi-section different-first-page
footers, header/footer absence, both field kinds), every rule (a golden
document that passes all 13, plus a single-defect fixture per rule), and the
API (happy path, non-docx, corrupt zip, oversize, single-rule).

Regenerate the on-disk fixtures at any time:

```bash
python -m tests.make_fixtures
```
