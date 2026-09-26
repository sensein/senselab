"""The ``audio_analysis`` workflow's per-pass PII adapter over the standalone text task.

``senselab.text.tasks.pii_detection`` is deliberately standalone: it scans a ``str`` or
``ScriptLine`` and knows nothing about "pass", "perturbation", or an ASR-model ensemble --
those are ``audio_analysis`` concepts, not text-detection ones. This module is where that
workflow vocabulary lives instead. It exists to do two things the task API cannot do for
itself:

1. **Re-attach ``perturbation``.** The workflow keys every artifact on it (``pii.json``
   lives at ``L1/perturbation/<name>/pii.json``); the task API's :class:`PiiReport` carries
   no such field because a standalone caller has no pass to tag.
2. **Own the multi-ASR ensemble.** A pass typically runs several ASR backends over the same
   audio; :func:`detect_pii_in_pass` scans each backend's transcript separately via
   :func:`~senselab.text.tasks.pii_detection.api.detect_pii`, then corroborates findings
   *across* those transcripts -- a span only one ASR backend transcribed (and no sibling ASR
   confirms) is the prototypical hallucination. The standalone path sees one transcript per
   call and structurally cannot do this; the asymmetry belongs here.

No category-severity weighting is applied anywhere in this pipeline (no SSN > date scaling):
in pediatric and clinical voice data, the nominally most severe Presidio categories
(``US_SSN``, ``CREDIT_CARD``) have near-zero true-positive rate and are dominated by ASR
digit hallucinations, so weighting them up would inflate exactly the hits a reviewer should
de-prioritise. See :func:`senselab.text.tasks.pii_detection.api._compute_detection_confidence`
for where that scoring actually happens; this module only supplies the cross-ASR pooling it
runs over.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from senselab.text.tasks.pii_detection.api import (
    PiiReport,
    PiiSpan,
    _compute_detection_confidence,
    detect_pii,
)

__all__ = ["PiiPassReport", "detect_pii_in_pass", "report_to_dict"]


@dataclass
class PiiPassReport(PiiReport):
    """A :class:`PiiReport` tagged with the workflow pass it was scanned for.

    ``perturbation`` is keyword-only and has no default: a pass report divorced from the
    pass it describes is a bug at the call site, not a value worth defaulting silently.
    """

    perturbation: str = field(kw_only=True)


def _build_full_text(resolved: Any) -> str:  # noqa: ANN401 -- accepts list / dict / ScriptLine
    """Concatenate the ``text`` fields of an ASR resolution into one string.

    Accepts either a list of ``ScriptLine``-shaped dicts / objects, or a single dict /
    object. Whitespace-only entries are dropped so they don't waste the detectors' compute
    budget. Moved here from the task layer alongside ``detect_pii_in_pass``: ``asr_resolved``
    (``{asr_model_id -> resolved_asr_result}``) is workflow shape, not a standalone-detector
    concept.
    """
    items = resolved if isinstance(resolved, list) else [resolved]
    parts: list[str] = []
    for line in items:
        if isinstance(line, dict):
            t = line.get("text") or ""
        else:
            t = getattr(line, "text", "") or ""
        if t and t.strip():
            parts.append(t)
    return " ".join(parts)


def detect_pii_in_pass(
    *,
    perturbation: str,
    asr_resolved: dict[str, Any],
    detectors: list[str] | None = None,
    presidio_score_threshold: float = 0.4,
    gliner_model: str | None = None,
    gliner_labels: list[str] | None = None,
    gliner_threshold: float = 0.5,
    require_cross_model_corroboration: bool = True,
) -> PiiPassReport:
    """Scan all ASR transcripts for one pass and return a unified, cross-ASR-corroborated report.

    Each non-empty ASR transcript is scanned independently via
    :func:`~senselab.text.tasks.pii_detection.api.detect_pii` (one subprocess dispatch
    covering every transcript in the pass, since ``detect_pii`` batches a sequence of
    inputs into a single call), then the resulting spans are pooled and corroborated across
    ASR backends -- the thing a single ``detect_pii`` call cannot do because it sees each
    input independently.

    Args:
        perturbation: e.g. ``"raw"``. Stamped onto the returned :class:`PiiPassReport` and,
            per span, into :func:`report_to_dict`'s output -- the workflow keys artifacts on
            it, so it has to survive the round trip even though :class:`PiiSpan` itself
            carries no such field.
        asr_resolved: ``{asr_model_id -> resolved_asr_result}``.
        detectors: Subset of detector names to run inside the subprocess venv. ``None``
            (default) runs both ``"presidio"`` and ``"gliner"``. Pass ``["presidio"]`` to
            skip the GLiNER model load entirely; ``["gliner"]`` to skip Presidio. An empty
            list short-circuits -- no subprocess spawned, report has ``detector_used=None``
            and ``contains_pii=False`` with an explicit ``"pii_disabled"`` failure note so
            the workflow can tell "we deliberately didn't check" apart from "the check
            failed".
        presidio_score_threshold: Presidio entities below this score are dropped at
            extraction time. 0.4 is permissive enough to catch standard phone-number
            formats; cross-model corroboration still gates the boolean flag for borderline
            scores.
        gliner_model: HuggingFace model id for GLiNER. ``None`` uses the subprocess module's
            default (``nvidia/gliner-pii``). Ignored when ``"gliner"`` is excluded from
            ``detectors``.
        gliner_labels: Labels passed to GLiNER's ``predict_entities``. ``None`` uses the
            subprocess module's curated default set. Ignored when ``"gliner"`` is excluded
            from ``detectors``.
        gliner_threshold: Drop GLiNER predictions below this score.
        require_cross_model_corroboration: When ``True`` (default), only flip
            ``contains_pii`` to ``True`` when a ``(category, normalized_text)`` pair is
            detected by >= 2 ASR models. Filters out hallucinated entities present in only
            one ASR's transcript. When fewer than 2 ASR models are available, any single
            detection counts.

    Returns:
        ``PiiPassReport`` with per-span detail, the detector(s) used, and any failure
        reasons.
    """
    failures: dict[str, str] = {}

    # Build per-ASR concatenated transcripts up front, in a stable order, so the pooled
    # spans below can be mapped back to the ASR model that produced each one.
    transcripts_by_asr: dict[str, str] = {}
    for asr_model, resolved in asr_resolved.items():
        full_text = _build_full_text(resolved)
        if full_text.strip():
            transcripts_by_asr[asr_model] = full_text

    if not transcripts_by_asr:
        # Nothing to scan -- every ASR result was empty / whitespace.
        return PiiPassReport(
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
            perturbation=perturbation,
        )

    # Explicit ``detectors=[]`` means the caller has chosen to disable PII detection for
    # this pass. Surface that as a distinct failure reason so a downstream auditor can tell
    # "didn't run on purpose" apart from "ran but found nothing" and "subprocess crashed".
    if detectors is not None and len(detectors) == 0:
        failures["pii_disabled"] = "PII detection disabled by caller (detectors=[])."
        return PiiPassReport(
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
            perturbation=perturbation,
        )

    asr_models = list(transcripts_by_asr.keys())
    detect_kwargs: dict[str, Any] = {
        "detectors": detectors,
        "presidio_score_threshold": presidio_score_threshold,
        "gliner_threshold": gliner_threshold,
    }
    if gliner_model is not None:
        detect_kwargs["gliner_model"] = gliner_model
    if gliner_labels is not None:
        detect_kwargs["gliner_labels"] = gliner_labels

    # A list input keeps this to one subprocess dispatch for the whole pass (``detect_pii``
    # batches a sequence into a single call) while still giving each ASR transcript its own
    # report to pool below. ``detect_pii`` never raises -- a subprocess crash or "no detector
    # loaded" comes back as a failure note on every report in the batch -- so there's no
    # separate try/except here.
    reports = detect_pii([transcripts_by_asr[m] for m in asr_models], **detect_kwargs)
    if not isinstance(reports, list):
        reports = [reports]  # pragma: no cover -- defensive; a list input always returns a list

    # Bound to a local so the ``is None`` check below narrows a plain ``str`` for
    # ``.split`` -- mypy won't carry that narrowing through a fresh ``reports[0]`` lookup.
    batch_detector_used = reports[0].detector_used if reports else None
    if batch_detector_used is None:
        # Every report in the batch shares one subprocess dispatch, so "did detection
        # actually run" is uniform across the batch -- checking the first report speaks for
        # all of them.
        for r in reports:
            failures.update(r.failures)
        print(f"warn: PII detection did not run for pass {perturbation!r}: {failures}", file=sys.stderr)
        return PiiPassReport(
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
            perturbation=perturbation,
        )

    detectors_used = batch_detector_used.split(",")

    # Re-attach the ASR model identity ``detect_pii`` cannot know (it materializes spans
    # with a batch-index placeholder), and pool every model's spans together for cross-ASR
    # corroboration below.
    spans_by_asr: dict[str, list[PiiSpan]] = {}
    pooled_spans: list[PiiSpan] = []
    for asr_model, report in zip(asr_models, reports):
        for s in report.spans:
            s.asr_model = asr_model
        spans_by_asr[asr_model] = report.spans
        pooled_spans.extend(report.spans)
        failures.update(report.failures)

    # Cross-ASR-model corroboration. A ``(category, normalized_text)`` pair detected by at
    # least two ASR backends is treated as real PII; everything else is a candidate that
    # might be an ASR hallucination. When the workflow only invokes one ASR backend, any
    # single hit counts (the corroboration check is informative, not load-bearing, in that
    # case).
    if not pooled_spans:
        contains_pii = False
    elif not require_cross_model_corroboration:
        contains_pii = True
    else:
        norm_keys: Counter[tuple[str, str]] = Counter()
        for per_model in spans_by_asr.values():
            seen_in_model: set[tuple[str, str]] = set()
            for s in per_model:
                key = (s.category, (s.text or "").strip().lower())
                if key in seen_in_model:
                    continue
                seen_in_model.add(key)
                norm_keys[key] += 1
        contains_pii = any(count >= 2 for count in norm_keys.values())
        if not contains_pii and len(spans_by_asr) < 2:
            contains_pii = True

    categories = sorted({s.category for s in pooled_spans})
    detection_confidence = _compute_detection_confidence(
        pooled_spans, n_asr_models=len(spans_by_asr), n_detectors_run=len(detectors_used)
    )
    return PiiPassReport(
        contains_pii=contains_pii,
        n_spans=len(pooled_spans),
        categories=categories,
        spans=pooled_spans,
        failures=failures,
        detector_used=",".join(detectors_used),
        detection_confidence=detection_confidence,
        perturbation=perturbation,
    )


def report_to_dict(report: PiiPassReport) -> dict[str, Any]:
    """Convert a ``PiiPassReport`` into a JSON-serializable dict.

    Every span in a ``PiiPassReport`` was scanned for the same pass, so rather than
    carrying a redundant per-span ``perturbation`` field on :class:`PiiSpan` itself (which
    would put workflow vocabulary back onto a task-layer type), it's stamped onto each
    serialized span here, uniformly, from ``report.perturbation``.
    """
    return {
        "contains_pii": report.contains_pii,
        "n_spans": report.n_spans,
        "categories": report.categories,
        "detector_used": report.detector_used,
        "detection_confidence": report.detection_confidence,
        "spans": [{**s.model_dump(exclude_none=True), "perturbation": report.perturbation} for s in report.spans],
        "failures": report.failures,
    }
