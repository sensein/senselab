# Evidence-first triage summary redesign — 2026-08-26

## Purpose

Make the per-recording report scannable for a speech scientist or otolaryngologist without changing
triage decisions. The report is a view over stored evidence, not a diagnostic or a second decision
engine.

## Observed baseline

- The current PDF puts the time lanes first and the decision, routing and provenance in a dense,
  monospaced second page. A reviewer has to find the result after the visual detail.
- The existing word lane avoids tick-label collision, but one row does not make every dense ASR or
  consensus word inspectable. Reusing a small fixed number of cycling lanes keeps time alignment
  while making neighbouring words separable.
- Existing rerendered campaign PDFs are the visual baseline; no clinical data is fetched or copied
  into the repository.

## Design and checklist

- [x] Put a short header and decision summary at the start: task/hint, file triage, release,
  discard/flag evidence, screened kinds, routing and branch outcomes.
- [x] Keep the full evidence in a clearly separated detail page: airway, speech, voice and redaction
  each state their outcome, reason, timing/count evidence and element ids.
- [x] Render words as time-aligned tokens distributed cyclically across a small number of lanes;
  preserve PII masking and fitted-label behaviour.
- [x] Create one versioned structured report object. Render both PDF and JSON from that object so
  file/release decisions, flags, routing and evidence cannot drift.
- [x] Add first-class JSON fields for recording/task context, file and release decisions,
  flags/discard grounds, screened kinds, routing/branch decisions, evidence item ids/timing/node
  provenance, and sibling artifact paths; retain existing fields for compatibility.
- [x] Test the new summary fields, word-lane layout and page hierarchy; run focused report tests.
- [x] Produce two local demonstration reports, render their PDFs to PNG, visually inspect them, and
  commit only a stable manifest rather than generated report content.

## Non-goals

No airway, routing, taxonomy, speech, voice, phonation or redaction decision logic changes. No
clinical interpretation or diagnostic language.
