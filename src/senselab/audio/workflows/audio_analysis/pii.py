"""PII detection over ASR transcripts.

Detection runs in an isolated subprocess venv (Presidio + GLiNER on
Python 3.13) so the host process doesn't need ``presidio-analyzer`` /
``spacy`` / ``gliner`` installed — see ``pii_subprocess.py`` for the
venv contents and worker. Two detectors run in parallel inside the venv:

1. **Microsoft Presidio Analyzer** — regex + spaCy-NER orchestrator with
   purpose-built recognizers for emails, phone numbers, SSNs, credit
   cards, IP addresses, dates, and locations.
2. **GLiNER PII** (``nvidia/gliner-pii`` by default) — a transformer-
   based zero-shot NER model fine-tuned on ~100k synthetic PII / PHI
   records. Catches the long tail Presidio misses (especially medical /
   health entities).

GLiNER's lowercase labels are normalized to Presidio's uppercase scheme
inside the worker so the cross-model corroboration logic below — which
keys on ``(category, text.lower())`` — sees the two detectors' hits on
the same entity as the same finding.

When the subprocess fails to start or both detectors fail to load, the
report records an explicit failure reason
(``failures["pii_subprocess"]``) and ``contains_pii`` defaults to
``False`` — the caller learns the check didn't actually run rather
than getting a silent all-clear.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

from senselab.audio.workflows.audio_analysis.pii_subprocess import (
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
    detect_pii_via_subprocess,
)

# Re-export the canonical detector names so ``analyze_audio.py`` (and
# any other caller wiring up a ``--pii-detectors`` flag) can reference
# them as ``pii.DETECTOR_PRESIDIO`` / ``pii.DETECTOR_GLINER`` rather
# than reaching into the subprocess-specific module.
__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_PRESIDIO",
    "PiiPassReport",
    "PiiSpan",
    "detect_pii_in_pass",
    "report_to_dict",
]


@dataclass
class PiiSpan:
    """One PII detection in a transcript."""

    text: str
    category: str  # presidio entity_type, e.g. "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"
    source: str  # "presidio" or "gliner/<original_label>"
    asr_model: str
    pass_label: str | None = None
    score: float | None = None  # detector confidence in [0, 1]


@dataclass
class PiiPassReport:
    """Aggregated PII findings for one pass."""

    pass_label: str
    contains_pii: bool
    n_spans: int
    categories: list[str]
    spans: list[PiiSpan] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    # Comma-joined list of detectors that successfully ran inside the
    # subprocess venv for this report — e.g. ``"presidio,gliner"`` when
    # both loaded cleanly, ``"presidio"`` when GLiNER failed but
    # Presidio worked, or ``None`` when neither detector ran.
    detector_used: str | None = None
    # Continuous detection confidence in ``[0, 1]`` computed from per-
    # detector raw scores plus cross-detector and cross-ASR-model
    # agreement. ``None`` ⇔ detectors did not actually run (subprocess
    # failure, ``detectors=[]`` short-circuit, all detectors failed to
    # load) — distinct from ``0.0`` which means "ran, found nothing".
    # No category-severity weighting: in pediatric / clinical voice
    # data the most-"severe" Presidio categories (US_SSN, CREDIT_CARD)
    # have near-zero true-positive rate and are dominated by ASR
    # digit-hallucinations, so weighting them up would inflate exactly
    # the hits a reviewer should de-prioritize.
    detection_confidence: float | None = None


def _compute_detection_confidence(spans: list[PiiSpan], n_asr_models: int) -> float:
    """Aggregate per-span detector scores into a single ``[0, 1]`` confidence.

    Combines three signals per unique ``(category, normalized_text)`` finding:

    - **max raw detector confidence** on that finding (Presidio's analyzer
      score or GLiNER's prediction probability)
    - **cross-detector agreement** — both Presidio and GLiNER independently
      flagged the same (category, normalized_text) (factor of 1.0) vs only
      one detector (0.5). Two-detector agreement is the strongest "is this
      a real entity or hallucinated?" signal we have at this layer.
    - **cross-ASR-model agreement** — fraction of available ASR transcripts
      that contain the finding. A span only one ASR transcribed (and that
      neither sibling ASR confirms) is the prototypical hallucination case.

    Then ``max()`` across findings — any single high-confidence corroborated
    finding raises the alarm, matching how the transcript / single-speaker
    axes combine their internal signals.

    Deliberately NO category-severity weighting (no SSN > date scaling) —
    in pediatric voice data the categories nominally most "severe" have
    near-zero true-positive rate and are dominated by ASR digit
    hallucinations; weighting them up would inflate the wrong cases.

    Args:
        spans: All PII spans collected for this pass.
        n_asr_models: Total number of ASR backends whose transcripts were
            scanned. Used as the denominator for cross-ASR agreement so
            single-ASR setups don't get penalised relative to multi-ASR.

    Returns:
        Confidence in ``[0, 1]``. ``0.0`` when ``spans`` is empty (the
        "detectors ran, nothing found" case). The "detectors did not run
        at all" case is communicated separately via ``detector_used=None``
        on the enclosing report; callers should branch on that, not on
        this number.
    """
    if not spans:
        return 0.0
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for s in spans:
        normalized = s.text.strip().lower()
        if not normalized:
            continue
        key = (s.category, normalized)
        g = groups.setdefault(
            key,
            {"detectors": set(), "asrs": set(), "max_score": 0.0},
        )
        # ``source`` shape is "presidio" or "gliner/<original_label>" —
        # take the part before the first ``/`` so both produce one bucket.
        detector_root = s.source.split("/", 1)[0] if s.source else "unknown"
        g["detectors"].add(detector_root)
        g["asrs"].add(s.asr_model)
        if s.score is not None:
            g["max_score"] = max(g["max_score"], float(s.score))
    if not groups:
        return 0.0
    denom_asrs = max(1, n_asr_models)
    risks: list[float] = []
    for g in groups.values():
        detector_agreement = len(g["detectors"]) / 2  # 0.5 single, 1.0 both
        asr_agreement = min(1.0, len(g["asrs"]) / denom_asrs)
        risks.append(g["max_score"] * detector_agreement * asr_agreement)
    return max(risks) if risks else 0.0


def _build_full_text(resolved: Any) -> str:  # noqa: ANN401 — accepts list / dict / ScriptLine
    """Concatenate the ``text`` fields of an ASR resolution into one string.

    Accepts either a list of ``ScriptLine``-shaped dicts / objects, or a
    single dict / object. Whitespace-only entries are dropped so they
    don't waste the detectors' compute budget.
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
    pass_label: str,
    asr_resolved: dict[str, Any],
    detectors: list[str] | None = None,
    presidio_score_threshold: float = 0.4,
    gliner_model: str | None = None,
    gliner_labels: list[str] | None = None,
    gliner_threshold: float = 0.5,
    require_cross_model_corroboration: bool = True,
) -> PiiPassReport:
    """Scan all ASR transcripts for one pass and return a unified PII report.

    Detection runs in the ``pii-detection`` subprocess venv via
    ``detect_pii_via_subprocess``. By default both Presidio and GLiNER
    run on every transcript and their findings are merged before this
    function applies cross-ASR-model corroboration. Callers that only
    want one detector — e.g. Presidio's deterministic regex without
    paying GLiNER's model-load cost, or GLiNER's model-based extraction
    without Presidio's structured-data recognizers — can narrow the set
    via ``detectors``.

    Args:
        pass_label: e.g. ``"raw_16k"``.
        asr_resolved: ``{asr_model_id → resolved_asr_result}``.
        detectors: Subset of detector names to run inside the subprocess
            venv. ``None`` (default) runs both ``"presidio"`` and
            ``"gliner"``. Pass ``["presidio"]`` to skip the GLiNER model
            load entirely; ``["gliner"]`` to skip Presidio. An empty list
            short-circuits — no subprocess spawned, report has
            ``detector_used=None`` and ``contains_pii=False`` with an
            explicit ``"pii_disabled"`` failure note so the workflow
            can tell "we deliberately didn't check" apart from
            "the check failed".
        presidio_score_threshold: Presidio entities below this score are
            dropped at extraction time. 0.4 is permissive enough to catch
            standard phone-number formats; cross-model corroboration
            still gates the boolean flag for borderline scores.
        gliner_model: HuggingFace model id for GLiNER. ``None`` uses the
            subprocess module's default (``nvidia/gliner-pii``). Ignored
            when ``"gliner"`` is excluded from ``detectors``.
        gliner_labels: Labels passed to GLiNER's ``predict_entities``.
            ``None`` uses the subprocess module's curated default set.
            Ignored when ``"gliner"`` is excluded from ``detectors``.
        gliner_threshold: Drop GLiNER predictions below this score.
        require_cross_model_corroboration: When ``True`` (default), only
            flip ``contains_pii`` to ``True`` when a ``(category,
            normalized_text)`` pair is detected by ≥2 ASR models. Filters
            out hallucinated entities present in only one ASR's
            transcript. When fewer than 2 ASR models are available, any
            single detection counts.

    Returns:
        ``PiiPassReport`` with per-span detail, the detector(s) used, and
        any failure reasons.
    """
    failures: dict[str, str] = {}

    # Build per-ASR concatenated transcripts up front so we send the
    # subprocess one request covering every ASR backend for this pass.
    transcripts_by_asr: dict[str, str] = {}
    for asr_model, resolved in asr_resolved.items():
        full_text = _build_full_text(resolved)
        if not full_text.strip():
            continue
        transcripts_by_asr[asr_model] = full_text

    if not transcripts_by_asr:
        # Nothing to scan — every ASR result was empty / whitespace.
        return PiiPassReport(
            pass_label=pass_label,
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
        )

    # Explicit ``detectors=[]`` means the caller has chosen to disable
    # PII detection for this pass. Surface that as a distinct failure
    # reason so a downstream auditor can tell "didn't run on purpose"
    # apart from "ran but found nothing" and "subprocess crashed".
    if detectors is not None and len(detectors) == 0:
        failures["pii_disabled"] = "PII detection disabled by caller (detectors=[])."
        return PiiPassReport(
            pass_label=pass_label,
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
        )

    try:
        subprocess_kwargs: dict[str, Any] = {
            "presidio_score_threshold": presidio_score_threshold,
            "gliner_threshold": gliner_threshold,
        }
        if detectors is not None:
            subprocess_kwargs["detectors"] = detectors
        if gliner_model is not None:
            subprocess_kwargs["gliner_model"] = gliner_model
        if gliner_labels is not None:
            subprocess_kwargs["gliner_labels"] = gliner_labels
        result = detect_pii_via_subprocess(transcripts_by_asr, **subprocess_kwargs)
    except Exception as exc:  # noqa: BLE001 — caller needs the report to continue
        msg = f"PII subprocess failed: {type(exc).__name__}: {exc}"
        failures["pii_subprocess"] = msg
        print(f"warn: {msg}", file=sys.stderr)
        return PiiPassReport(
            pass_label=pass_label,
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
        )

    spans_by_asr_raw = result.get("spans_by_asr", {})
    failures.update(result.get("failures", {}))
    detectors_used = list(result.get("detectors_used", []))
    for name, msg in failures.items():
        print(f"warn: PII / {name}: {msg}", file=sys.stderr)

    if not detectors_used:
        failures.setdefault(
            "no_pii_detector",
            "Neither Presidio nor GLiNER loaded inside the subprocess venv; contains_pii=False reported by default.",
        )
        print(f"warn: {failures['no_pii_detector']}", file=sys.stderr)
        return PiiPassReport(
            pass_label=pass_label,
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=failures,
            detector_used=None,
        )

    # Materialize PiiSpan objects with the asr_model + pass_label fields
    # the subprocess can't fill in (it doesn't know which pass it's running
    # against), and dedupe ``(category, normalized_text, source)`` per ASR
    # model so a single entity detected by both Presidio and GLiNER counts
    # once per detector rather than once per phrasing.
    spans_by_asr: dict[str, list[PiiSpan]] = {}
    spans: list[PiiSpan] = []
    for asr_model, raw_spans in spans_by_asr_raw.items():
        seen_per_detector: set[tuple[str, str, str]] = set()
        per_model: list[PiiSpan] = []
        for raw in raw_spans:
            text: str = raw.get("text") or ""
            category: str = raw.get("category") or ""
            source: str = raw.get("source") or "unknown"
            normalized = text.strip().lower()
            dedup_key: tuple[str, str, str] = (category, normalized, source)
            if not normalized or dedup_key in seen_per_detector:
                continue
            seen_per_detector.add(dedup_key)
            score = raw.get("score")
            per_model.append(
                PiiSpan(
                    text=text,
                    category=category,
                    source=source,
                    asr_model=asr_model,
                    pass_label=pass_label,
                    score=float(score) if score is not None else None,
                )
            )
        spans_by_asr[asr_model] = per_model
        spans.extend(per_model)

    # Cross-ASR-model corroboration. A ``(category, normalized_text)``
    # pair detected by at least two ASR backends is treated as real PII;
    # everything else is a candidate that might be an ASR hallucination.
    # When the workflow only invokes one ASR backend, any single hit
    # counts (the corroboration check is informative, not load-bearing,
    # in that case).
    if not spans:
        contains_pii = False
    elif not require_cross_model_corroboration:
        contains_pii = True
    else:
        norm_keys: Counter[tuple[str, str]] = Counter()
        for asr_model, per_model in spans_by_asr.items():
            seen_in_model: set[tuple[str, str]] = set()
            for s in per_model:
                key = (s.category, s.text.strip().lower())
                if key in seen_in_model:
                    continue
                seen_in_model.add(key)
                norm_keys[key] += 1
        contains_pii = any(count >= 2 for count in norm_keys.values())
        if not contains_pii and len(spans_by_asr) < 2 and spans:
            contains_pii = True

    categories = sorted({s.category for s in spans})
    # detection_confidence is computed only on the happy path where at
    # least one detector ran. The early-return branches above already
    # leave it as ``None`` (the dataclass default), so a caller can
    # distinguish "detectors ran, found nothing" (0.0) from "detectors
    # did not actually run" (None).
    detection_confidence = _compute_detection_confidence(spans, n_asr_models=len(spans_by_asr))
    return PiiPassReport(
        pass_label=pass_label,
        contains_pii=contains_pii,
        n_spans=len(spans),
        categories=categories,
        spans=spans,
        failures=failures,
        detector_used=",".join(detectors_used) if detectors_used else None,
        detection_confidence=detection_confidence,
    )


def report_to_dict(report: PiiPassReport) -> dict[str, Any]:
    """Convert a ``PiiPassReport`` into a JSON-serializable dict."""
    return {
        "pass_label": report.pass_label,
        "contains_pii": report.contains_pii,
        "n_spans": report.n_spans,
        "categories": report.categories,
        "detector_used": report.detector_used,
        "detection_confidence": report.detection_confidence,
        "spans": [asdict(s) for s in report.spans],
        "failures": report.failures,
    }
