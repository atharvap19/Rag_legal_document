// SOP Compliance Checker front end. Plain ES module, no build step.
// The API and this page share an origin, so all fetches are relative.

const TIMEOUT_MS = 120000; // LanguageTool over a long doc is genuinely slow.

const state = {
  file: null,
  rules: new Map(),   // id -> { id, name, severity, description }
  result: null,       // last /api/analyze response
  extraction: null,   // the Doc model from that same response
};

// A whole Doc model can run to megabytes of pretty-printed JSON; painting all
// of it into a <pre> locks the tab up. Downloads and copies still get it all.
const MAX_RENDERED_CHARS = 400000;

const els = {};

function cache() {
  els.dropzone = document.getElementById("dropzone");
  els.fileInput = document.getElementById("file-input");
  els.fileName = document.getElementById("file-name");
  els.analyzeBtn = document.getElementById("analyze-btn");
  els.downloadBtn = document.getElementById("download-btn");
  els.status = document.getElementById("status");
  els.resultsSection = document.getElementById("results-section");
  els.summary = document.getElementById("summary");
  els.findings = document.getElementById("findings");
  els.filterStatus = document.getElementById("filter-status");
  els.cfgRequired = document.getElementById("cfg-required");
  els.cfgIgnore = document.getElementById("cfg-ignore");
  els.cfgFlesch = document.getElementById("cfg-flesch");
  els.extractSection = document.getElementById("extract-section");
  els.extractJson = document.getElementById("extract-json");
  els.extractView = document.getElementById("extract-view");
  els.extractSize = document.getElementById("extract-size");
  els.extractPath = document.getElementById("extract-path");
  els.copyExtractBtn = document.getElementById("copy-extract-btn");
  els.downloadExtractBtn = document.getElementById("download-extract-btn");
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------
async function init() {
  cache();
  wireEvents();
  await loadRules();
}

async function loadRules() {
  try {
    const res = await fetch("/api/rules");
    if (!res.ok) throw new Error(`status ${res.status}`);
    const rules = await res.json();
    rules.forEach((r) => state.rules.set(r.id, r));
  } catch (err) {
    setStatus(`Could not load rule metadata: ${err.message}`, "error");
  }
}

function wireEvents() {
  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      els.fileInput.click();
    }
  });
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length) selectFile(e.target.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("dragover");
    })
  );
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) selectFile(file);
  });

  els.analyzeBtn.addEventListener("click", analyze);
  els.downloadBtn.addEventListener("click", downloadJson);
  els.filterStatus.addEventListener("change", renderFindings);
  els.extractView.addEventListener("change", renderExtraction);
  els.copyExtractBtn.addEventListener("click", copyExtraction);
  els.downloadExtractBtn.addEventListener("click", downloadExtraction);
}

// ---------------------------------------------------------------------------
// file selection (client-side validation is fast feedback; server is truth)
// ---------------------------------------------------------------------------
function selectFile(file) {
  if (!file.name.toLowerCase().endsWith(".docx")) {
    state.file = null;
    els.analyzeBtn.disabled = true;
    els.fileName.textContent = "";
    setStatus(`"${file.name}" is not a .docx file.`, "error");
    return;
  }
  state.file = file;
  els.fileName.textContent = file.name;
  els.analyzeBtn.disabled = false;
  setStatus("");
}

function buildConfig() {
  const cfg = {};
  const list = (v) =>
    v.split(",").map((s) => s.trim()).filter(Boolean);
  if (els.cfgRequired.value.trim())
    cfg.required_sections = list(els.cfgRequired.value);
  if (els.cfgIgnore.value.trim())
    cfg.ignore_words = list(els.cfgIgnore.value);
  if (els.cfgFlesch.value !== "")
    cfg.readability_flesch_min = Number(els.cfgFlesch.value);
  // readability_fog_max is not exposed here; rule 12 uses its own default
  return Object.keys(cfg).length ? cfg : null;
}

// ---------------------------------------------------------------------------
// analyze
// ---------------------------------------------------------------------------
async function analyze() {
  if (!state.file) return;

  const form = new FormData();
  // Do NOT set Content-Type: the browser must write the multipart boundary.
  form.append("file", state.file, state.file.name);
  const config = buildConfig();
  if (config) form.append("config", JSON.stringify(config));
  form.append("include_extraction", "true");

  setBusy(true);
  setStatus("Analyzing… this can take a while for long documents.", "busy");

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (!res.ok) {
      const detail = await readError(res);
      setStatus(`Server rejected the file (${res.status}): ${detail}`, "error");
      return;
    }
    state.result = await res.json();
    state.extraction = state.result.extraction || null;
    setStatus(
      `Analyzed "${state.result.filename}".`,
      "ok"
    );
    els.downloadBtn.disabled = false;
    renderSummary();
    renderFindings();
    els.resultsSection.hidden = false;
    renderExtraction();
  } catch (err) {
    clearTimeout(timer);
    if (err.name === "AbortError") {
      setStatus(
        `Timed out after ${TIMEOUT_MS / 1000}s. The document may be very long; ` +
          `try again or reduce its size.`,
        "error"
      );
    } else {
      setStatus(`Request failed: ${err.message}`, "error");
    }
  } finally {
    setBusy(false);
  }
}

async function readError(res) {
  try {
    const body = await res.json();
    return typeof body.detail === "string"
      ? body.detail
      : JSON.stringify(body.detail);
  } catch {
    return res.statusText || "unknown error";
  }
}

// ---------------------------------------------------------------------------
// rendering
// ---------------------------------------------------------------------------
function renderSummary() {
  const s = state.result.summary;
  els.summary.replaceChildren();
  const tiles = [
    ["Passed", s.passed, "pass"],
    ["Failed", s.failed, "fail"],
    ["Not evaluated", s.not_evaluated, "na"],
    ["Total", s.total, "total"],
  ];
  for (const [label, value, cls] of tiles) {
    const tile = document.createElement("div");
    tile.className = `tile tile-${cls}`;
    const num = document.createElement("span");
    num.className = "tile-num";
    num.textContent = String(value);
    const lab = document.createElement("span");
    lab.className = "tile-label";
    lab.textContent = label;
    tile.append(num, lab);
    els.summary.appendChild(tile);
  }
}

function stateOf(f) {
  if (f.passed === true) return "pass";
  if (f.passed === false) return "fail";
  return "na";
}

function passesFilter(f) {
  const sf = els.filterStatus.value;
  const st = stateOf(f);
  if (sf === "failed" && st !== "fail") return false;
  if (sf === "passed" && st !== "pass") return false;
  if (sf === "na" && st !== "na") return false;
  return true;
}

function renderFindings() {
  els.findings.replaceChildren();
  if (!state.result) return;
  const visible = state.result.findings.filter(passesFilter);
  if (!visible.length) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "No findings match the current filters.";
    els.findings.appendChild(li);
    return;
  }
  for (const f of visible) els.findings.appendChild(row(f));
}

const STATE_LABEL = { pass: "Passed", fail: "Failed", na: "Not evaluated" };

function row(f) {
  const st = stateOf(f);
  const li = document.createElement("li");
  li.className = `finding finding-${st}`;

  const header = document.createElement("button");
  header.className = "finding-header";
  header.type = "button";
  header.setAttribute("aria-expanded", "false");

  const badge = document.createElement("span");
  badge.className = `badge badge-${st}`;
  badge.textContent = STATE_LABEL[st];

  const title = document.createElement("span");
  title.className = "finding-title";
  title.textContent = `${f.rule_id}. ${f.rule_name}`;

  const msg = document.createElement("span");
  msg.className = "finding-msg";
  msg.textContent = f.message;

  const chevron = document.createElement("span");
  chevron.className = "chevron";
  chevron.setAttribute("aria-hidden", "true");
  chevron.textContent = "›";

  header.append(badge, title, msg, chevron);

  const detail = document.createElement("div");
  detail.className = "finding-detail";
  detail.hidden = true;
  detail.appendChild(detailBody(f));

  header.addEventListener("click", () => {
    const open = detail.hidden;
    detail.hidden = !open;
    header.setAttribute("aria-expanded", String(open));
    li.classList.toggle("open", open);
  });

  li.append(header, detail);
  return li;
}

function detailBody(f) {
  const frag = document.createDocumentFragment();

  const meta = state.rules.get(f.rule_id);
  if (meta && meta.description) {
    const d = document.createElement("p");
    d.className = "rule-desc";
    d.textContent = meta.description;
    frag.appendChild(d);
  }

  frag.appendChild(labelledList("Evidence", f.evidence, "evidence"));
  frag.appendChild(labelledList("Locations", f.locations, "locations"));

  const conf = document.createElement("p");
  conf.className = "confidence";
  conf.textContent = `Confidence: ${f.confidence}`;
  frag.appendChild(conf);
  return frag;
}

function labelledList(label, items, cls) {
  const wrap = document.createElement("div");
  wrap.className = `detail-block ${cls}`;
  const h = document.createElement("h4");
  h.textContent = label;
  wrap.appendChild(h);
  if (!items || !items.length) {
    const p = document.createElement("p");
    p.className = "muted";
    p.textContent = "—";
    wrap.appendChild(p);
    return wrap;
  }
  const ul = document.createElement("ul");
  for (const item of items) {
    const li = document.createElement("li");
    // document-derived text: assign via textContent, never innerHTML
    li.textContent = item;
    ul.appendChild(li);
  }
  wrap.appendChild(ul);
  return wrap;
}

// ---------------------------------------------------------------------------
// extraction output -- the Doc model the rules were handed
// ---------------------------------------------------------------------------
function renderExtraction() {
  const doc = state.extraction;
  if (!doc) {
    els.extractSection.hidden = true;
    return;
  }
  els.extractSection.hidden = false;

  const saved = state.result && state.result.extraction_file;
  els.extractPath.textContent = saved || "(not written to disk)";

  const view = viewData(doc, els.extractView.value);
  const text = JSON.stringify(view, null, 2);
  const truncated = text.length > MAX_RENDERED_CHARS;

  // document-derived text: textContent, never innerHTML
  els.extractJson.textContent = truncated
    ? text.slice(0, MAX_RENDERED_CHARS) +
      `\n\n… truncated for display (${fmtSize(text.length)} total). ` +
      `Use Download or Copy for the complete JSON.`
    : text;

  const paras = allParagraphs(doc);
  els.extractSize.textContent =
    `${fmtSize(text.length)} · ${doc.blocks.length} blocks · ` +
    `${paras.length} paragraphs · ${doc.tables.length} tables`;
}

function viewData(doc, view) {
  if (view === "full") return doc;
  if (view === "summary") return extractionSummary(doc);
  return doc[view] ?? null;   // blocks | core | sections | tables | styles
}

/** Every paragraph in flow order, table cells included -- the JS twin of the
 *  extractor's _iter_para, so these counts match what the rules iterate. */
function allParagraphs(doc) {
  const byIndex = new Map((doc.tables || []).map((t) => [t.table_index, t]));
  const out = [];
  const walk = (blocks) => {
    for (const b of blocks || []) {
      if (b.kind === "paragraph") {
        out.push(b);
      } else if (b.kind === "table") {
        const t = byIndex.get(b.table_index);
        if (!t) continue;
        for (const row of t.rows || [])
          for (const cell of row.cells || []) walk(cell.blocks);
      }
    }
  };
  walk(doc.blocks);
  return out;
}

/** A readable digest of the extraction: what was found, and where. */
function extractionSummary(doc) {
  const paras = allParagraphs(doc);
  const headers = [], footers = [];
  for (const sec of doc.sections || []) {
    for (const hf of Object.values(sec.headers || {})) headers.push(hf);
    for (const hf of Object.values(sec.footers || {})) footers.push(hf);
  }

  const fields = [];
  for (const p of paras)
    for (const f of p.fields || [])
      fields.push({ kind: f.kind, instruction: f.instruction,
                    result: f.result, location: p.location });
  for (const hf of headers.concat(footers))
    for (const f of hf.fields || [])
      fields.push({
        kind: f.kind, instruction: f.instruction, result: f.result,
        location: `${hf.kind} (${hf.which}), section ${hf.section_index}`,
      });

  const styles = doc.styles || {};

  return {
    filename: doc.filename,
    core: doc.core,
    counts: {
      blocks: doc.blocks.length,
      paragraphs: paras.length,
      non_empty_paragraphs: paras.filter((p) => p.text.trim()).length,
      runs: paras.reduce((n, p) => n + (p.runs || []).length, 0),
      tables: doc.tables.length,
      sections: (doc.sections || []).length,
      headers: headers.length,
      footers: footers.length,
      fields: fields.length,
      styles: Object.keys(styles.styles || {}).length,
    },
    theme_fonts: {
      major: styles.theme_major ?? null,
      minor: styles.theme_minor ?? null,
      default_paragraph_style: styles.default_para_style ?? null,
    },
    headings: paras
      .filter((p) => p.props.heading_level)
      .map((p) => ({ level: p.props.heading_level, text: p.text,
                     style: p.props.style_name, location: p.location })),
    fields,
    headers_footers: headers.concat(footers).map((hf) => ({
      kind: hf.kind, which: hf.which, section: hf.section_index,
      linked_to_previous: hf.is_linked_to_previous,
      text: (hf.paragraphs || []).map((p) => p.text).filter(Boolean),
    })),
    first_paragraphs: paras.slice(0, 20).map(paraDigest),
    table_shapes: (doc.tables || []).map((t) => ({
      table_index: t.table_index,
      rows: (t.rows || []).length,
      columns: t.rows && t.rows.length ? t.rows[0].cells.length : 0,
      first_row: t.rows && t.rows.length
        ? t.rows[0].cells.map(cellText)
        : [],
    })),
  };
}

function cellText(cell) {
  return (cell.blocks || [])
    .filter((b) => b.kind === "paragraph")
    .map((b) => b.text)
    .join(" ")
    .trim();
}

function paraDigest(p) {
  const font = (p.runs || [])[0] ? p.runs[0].font : null;
  return {
    block_index: p.block_index,
    location: p.location,
    style: p.props.style_name || p.props.style_id,
    heading_level: p.props.heading_level,
    alignment: p.props.alignment,
    font: font
      ? `${font.name ?? "?"} ${font.size ?? "?"}pt${font.bold ? " bold" : ""}`
      : null,
    in_table: p.in_table,
    text: p.text.length > 160 ? p.text.slice(0, 160) + "…" : p.text,
  };
}

async function copyExtraction() {
  if (!state.extraction) return;
  const text = JSON.stringify(
    viewData(state.extraction, els.extractView.value), null, 2);
  try {
    await navigator.clipboard.writeText(text);
    setStatus(`Copied ${fmtSize(text.length)} of JSON to the clipboard.`, "ok");
  } catch (err) {
    // clipboard access needs a secure context (https or localhost)
    setStatus(`Could not copy: ${err.message}. Use Download instead.`, "error");
  }
}

function downloadExtraction() {
  if (!state.extraction) return;
  const base = (state.result ? state.result.filename : "document")
    .replace(/\.docx$/i, "");
  saveBlob(JSON.stringify(state.extraction, null, 2), `${base}.extracted.json`);
}

function fmtSize(chars) {
  if (chars < 1024) return `${chars} B`;
  if (chars < 1024 * 1024) return `${(chars / 1024).toFixed(1)} KB`;
  return `${(chars / 1024 / 1024).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function downloadJson() {
  if (!state.result) return;
  // the Doc model has its own download button; keep this file about findings
  const { extraction, ...findings } = state.result;
  const base = state.result.filename.replace(/\.docx$/i, "");
  saveBlob(JSON.stringify(findings, null, 2), `${base}.findings.json`);
}

function saveBlob(text, filename) {
  const url = URL.createObjectURL(
    new Blob([text], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function setStatus(text, kind = "") {
  els.status.textContent = text;
  els.status.className = `status${kind ? " status-" + kind : ""}`;
}

function setBusy(busy) {
  els.analyzeBtn.disabled = busy || !state.file;
  els.dropzone.classList.toggle("busy", busy);
}

init();
