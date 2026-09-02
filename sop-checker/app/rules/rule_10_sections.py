"""Rule 10 - Required sections present."""
from __future__ import annotations

from app.extractor import Doc
from .base import (
    COMMON_REQUIRED_SECTIONS,
    Rule,
    RuleConfig,
    Finding,
    iter_headings,
    normalized_heading_text,
    normalize_key,
)


class Rule10(Rule):
    id = 10
    name = "Required sections"
    severity = "error"
    description = ("The configured required sections must all appear as "
                   "headings (a leading section number is allowed).")

    def evaluate(self, doc: Doc, config: RuleConfig) -> Finding:
        required = [r for r in (config.required_sections or []) if r.strip()]
        if not required:
            return self.na(
                "Please enter the sections this document must contain -- "
                "there is no default, because which sections an SOP needs is "
                "a house rule. For example: "
                + ", ".join(COMMON_REQUIRED_SECTIONS) + ".")

        headings = list(iter_headings(doc))
        heading_keys = {normalized_heading_text(h.text): h for h in headings}
        heading_norm = [(normalized_heading_text(h.text), h) for h in headings]

        present: list[str] = []
        missing: list[str] = []
        locations: list[str] = []

        for req in required:
            rkey = normalize_key(req)
            match = None
            if rkey in heading_keys:
                match = heading_keys[rkey]
            else:
                for hkey, h in heading_norm:
                    if hkey.startswith(rkey):
                        match = h
                        break
            if match is not None:
                present.append(req)
                locations.append(match.location)
            else:
                missing.append(req)

        evidence = [f"headings: {[h.text for h in headings]}"]
        if missing:
            return self.fail(
                "Missing required section(s): " + ", ".join(missing) + ".",
                evidence=evidence + [f"present: {present}"],
                locations=locations, confidence="heuristic")
        return self.ok(
            "All required sections are present: " + ", ".join(present) + ".",
            locations=locations, confidence="heuristic")


RULE = Rule10()
