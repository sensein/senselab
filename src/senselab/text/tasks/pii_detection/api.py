"""Standalone PII detection over plain text, ``ScriptLine``, and ASR transcripts.

One public entry point, :func:`detect_pii`, over a ``str``, a ``ScriptLine`` (optionally
with nested word/segment ``chunks``), or a sequence of either. No dependency on any
audio-workflow concept (a run "pass", a per-pass tag, an ASR-model ensemble) — this is
what a caller reaches for to check arbitrary text or a transcript line for PII on its own.
A caller that needs to merge several ASR backends' transcripts for one recording and
corroborate findings across them wants a workflow-level adapter built on top of this
module, not a feature of it -- ``audio_analysis`` keeps its own for exactly that reason.

Detection runs in an isolated subprocess venv (Presidio + GLiNER + the rules
cascade on Python 3.13) so the host process doesn't need ``presidio-analyzer`` /
``spacy`` / ``gliner`` installed — see ``subprocess_backend.py`` for the
venv contents and worker. Three detectors run inside the venv:

1. **Microsoft Presidio Analyzer** — regex + spaCy-NER orchestrator with
   purpose-built recognizers for emails, phone numbers, SSNs, credit
   cards, IP addresses, dates, and locations.
2. **GLiNER PII** (``nvidia/gliner-pii`` by default) — a transformer-
   based zero-shot NER model fine-tuned on ~100k synthetic PII / PHI
   records. Defaults to the HIPAA Safe Harbor 18 identifiers (matches
   b2aiprep #256) — catches PHI categories Presidio doesn't natively
   recognize (medical record number, health plan number, account
   number, fax number, URL, biometric / device / vehicle identifiers,
   unique identifier, ...).
3. **Rules cascade** (``rules.py``, ported from PR #542) — regex,
   gazetteers, spaCy NER, self-disclosed demographics, age-over-90 and a
   combinatorial re-identification window, with precision guards (a
   Zipf-frequency name hard-gate, structured-identifier format validation)
   that make it worth running alongside the two model-based detectors
   rather than in place of them. Its category vocabulary is its own; see
   ``subprocess_backend.py`` for how it is reconciled with the others.

GLiNER's lowercase labels are normalized to Presidio's uppercase scheme
inside the worker so the cross-detector corroboration logic below — which
keys on ``(category, text.lower())`` — sees two detectors' hits on
the same entity as the same finding.

When the subprocess fails to start or every detector fails to load, the
report records an explicit failure reason
(``failures["pii_subprocess"]``) and ``contains_pii`` defaults to
``False`` — the caller learns the check didn't actually run rather
than getting a silent all-clear.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from senselab.text.tasks.pii_detection.subprocess_backend import (
    _KNOWN_DETECTORS,
    DETECTOR_GLINER,
    DETECTOR_PRESIDIO,
    detect_pii_via_subprocess,
)
from senselab.utils.data_structures import ScriptLine

# Re-export the canonical detector names so ``analyze_audio.py`` (and
# any other caller wiring up a ``--pii-detectors`` flag) can reference
# them as ``pii.DETECTOR_PRESIDIO`` / ``pii.DETECTOR_GLINER`` rather
# than reaching into the subprocess-specific module.
__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_PRESIDIO",
    "PiiReport",
    "PiiSpan",
    "detect_pii",
    "flatten_script_line",
]


@dataclass
class PiiSpan:
    """One PII detection in a transcript."""

    text: str
    category: str  # presidio entity_type, e.g. "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"
    source: str  # "presidio" or "gliner/<original_label>"
    asr_model: str
    score: float | None = None  # detector confidence in [0, 1]


@dataclass
class PiiReport:
    """Aggregated PII findings for one scanned input.

    Carries no notion of a run "pass" or a per-pass tag — that is audio-workflow vocabulary
    that belongs to the caller, not to a standalone text/ScriptLine detector. A caller that
    needs that association (e.g. tagging findings by which pass produced them) keys its own
    ``{tag: PiiReport}`` mapping on top of this type instead (see the ``audio_analysis``
    workflow's per-pass PII adapter).
    """

    contains_pii: bool
    n_spans: int
    categories: list[str]
    spans: list[PiiSpan] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    # Comma-joined list of detectors that successfully ran inside the
    # subprocess venv for this report — e.g. ``"gliner,presidio,rules"`` when
    # all three loaded cleanly, ``"presidio,rules"`` when GLiNER failed but
    # the others worked, or ``None`` when no detector ran.
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


def flatten_script_line(line: ScriptLine) -> str:
    """Join a ScriptLine's text with its nested chunks', depth-first.

    Backends differ in where they put the words: Whisper returns segment text
    plus word-level ``chunks``, a forced aligner returns segment-level lines with
    word-level chunks nested underneath, and a diarization line carries a speaker
    and no text at all. Scanning only ``text`` would make PII coverage silently
    depend on which backend produced the transcript, so the whole tree is
    flattened instead.

    Args:
        line: The ``ScriptLine`` to flatten. Its own ``text`` (if any) comes
            first, followed by each child's flattened text in order.

    Returns:
        The concatenated, whitespace-normalized text. Empty string when the
        line and all its descendants carry no text (e.g. a diarization line
        with only a ``speaker`` label) — this is a normal, documented result,
        not an error.
    """
    parts: list[str] = []
    own = (line.text or "").strip()
    if own:
        parts.append(own)
    for child in line.chunks or []:
        nested = flatten_script_line(child)
        if nested:
            parts.append(nested)
    return " ".join(parts)


def _corroborated_contains_pii(
    spans: list[PiiSpan],
    detectors_used: list[str],
    require_cross_source_corroboration: bool,
) -> bool:
    """Decide the ``contains_pii`` boolean for one scanned input's spans.

    A caller merging several ASR backends' transcripts for one recording can corroborate
    across those *transcripts* (a span only one ASR hallucinated is the classic false
    positive) -- that is what the ``audio_analysis`` workflow's per-pass PII adapter does on
    top of this function. A standalone ``str``/``ScriptLine`` input has exactly one
    transcript, so there is nothing to corroborate across sources in that sense. The
    redundancy that *does* exist at this layer is cross-detector agreement — two of
    Presidio, GLiNER and the rules cascade independently flagging the same ``(category,
    normalized_text)`` — so that becomes the corroboration signal here.

    Args:
        spans: Materialized spans for this single input.
        detectors_used: Detector names that actually loaded for this call.
        require_cross_source_corroboration: When ``True`` and ≥2 detectors ran, only a
            ``(category, normalized_text)`` pair flagged by ≥2 detectors counts. When
            fewer than 2 detectors ran, corroboration cannot apply and any hit counts —
            the same single-witness fallback the cross-ASR-model corroboration above it
            uses.

    Returns:
        Whether the input should be flagged as containing PII.
    """
    if not spans:
        return False
    if not require_cross_source_corroboration or len(detectors_used) < 2:
        return True
    groups: dict[tuple[str, str], set[str]] = {}
    for s in spans:
        normalized = s.text.strip().lower()
        if not normalized:
            continue
        root = s.source.split("/", 1)[0] if s.source else "unknown"
        groups.setdefault((corroboration_family(s.category), normalized), set()).add(root)
    return any(len(roots) >= 2 for roots in groups.values())


# Cross-detector agreement is keyed on a COARSE family, not the reported category.
#
# The detectors disagree about granularity, not about entities. Presidio splits contact
# details into EMAIL_ADDRESS / PHONE_NUMBER / IP_ADDRESS and identifiers into US_SSN /
# CREDIT_CARD / ...; the rules cascade emits one CONTACT and one IDNUM. Keying agreement on
# the raw category therefore made rules structurally unable to corroborate anything, while
# still counting toward the denominator -- so adding a third detector pushed a finding both
# Presidio and GLiNER agreed on from 2/2 = 1.0 down to 2/3 = 0.667. Adding a detector must
# not make a well-corroborated finding look less certain.
#
# The reduction is fine -> coarse, which is deterministic many-to-one. The inverse (mapping
# rules' CONTACT onto one of Presidio's three) would be a guess, which is why it is not done
# here: an unmapped category falls through to itself, so an unknown label simply agrees only
# with its own kind rather than being silently merged into the wrong family.
#
# Only the agreement key is coarsened. PiiSpan.category keeps the finest label any detector
# supplied, because that is what a reader acting on a finding needs.
_CORROBORATION_FAMILY: dict[str, str] = {
    # people
    "PERSON": "NAME",
    "NAME": "NAME",
    # contact details
    "EMAIL_ADDRESS": "CONTACT",
    "PHONE_NUMBER": "CONTACT",
    "IP_ADDRESS": "CONTACT",
    "CONTACT": "CONTACT",
    # government / financial / clinical identifiers
    "US_SSN": "IDNUM",
    "US_DRIVER_LICENSE": "IDNUM",
    "US_PASSPORT": "IDNUM",
    "US_ITIN": "IDNUM",
    "US_BANK_NUMBER": "IDNUM",
    "CREDIT_CARD": "IDNUM",
    "IBAN_CODE": "IDNUM",
    "MEDICAL_LICENSE": "IDNUM",
    "IDNUM": "IDNUM",
    # places
    "LOCATION": "LOC",
    "ADDRESS": "LOC",
    "LOC": "LOC",
    # dates and ages
    "DATE_TIME": "DATE",
    "DATE": "DATE",
    "AGE": "AGE",
    # organisations
    "ORGANIZATION": "ORG",
    "ORG": "ORG",
    "NRP": "ORG",
}


def corroboration_family(category: str) -> str:
    """Return the coarse family a detector category belongs to, for agreement only.

    An unmapped category returns itself, so a detector emitting a label this map has not
    seen corroborates only with the same label rather than being folded into a family it
    may not belong to.
    """
    return _CORROBORATION_FAMILY.get(category.upper(), category.upper())


def _materialize_spans(raw_spans: list[dict[str, Any]], source_id: str) -> list[PiiSpan]:
    """Build deduped ``PiiSpan`` objects from one input's raw subprocess spans.

    Dedupe key is ``(category, normalized_text, source)`` so a single entity detected by
    both Presidio and GLiNER counts once per detector rather than once per phrasing —
    mirrors the per-ASR-model dedup the ``audio_analysis`` workflow's per-pass PII adapter
    does on top of this.

    Args:
        raw_spans: Raw span dicts for one input, as returned under one
            ``spans_by_asr`` key by ``detect_pii_via_subprocess``.
        source_id: Identifier for the scanned input, stored on ``PiiSpan.asr_model``.
            Standalone text has no ASR backend; this reuses that field as the
            per-input identifier ``_compute_detection_confidence`` groups by, rather
            than adding a parallel field for a single-input-per-report caller.

    Returns:
        Deduped ``PiiSpan`` list for the input.
    """
    seen: set[tuple[str, str, str]] = set()
    spans: list[PiiSpan] = []
    for raw in raw_spans:
        text: str = raw.get("text") or ""
        category: str = raw.get("category") or ""
        source: str = raw.get("source") or "unknown"
        normalized = text.strip().lower()
        dedup_key: tuple[str, str, str] = (category, normalized, source)
        if not normalized or dedup_key in seen:
            continue
        seen.add(dedup_key)
        score = raw.get("score")
        spans.append(
            PiiSpan(
                text=text,
                category=category,
                source=source,
                asr_model=source_id,
                score=float(score) if score is not None else None,
            )
        )
    return spans


def _empty_reports(n: int, failures: dict[str, str]) -> list[PiiReport]:
    """Build ``n`` identical "detector did not run" reports, one per input.

    Every early-return branch of ``detect_pii`` (disabled, all-empty, subprocess crash,
    no detector loaded) shares this shape: ``detector_used=None`` and
    ``detection_confidence=None`` so a caller can tell "did not run" apart from "ran and
    found nothing" (``0.0``) — the same distinction the ``audio_analysis`` workflow's
    per-pass PII adapter preserves on top of this.

    Args:
        n: Number of reports to produce (one per input in the caller's batch).
        failures: Failure dict shared by every report in the batch.

    Returns:
        A list of ``n`` ``PiiReport`` objects, each a fresh, independent instance.
    """
    return [
        PiiReport(
            contains_pii=False,
            n_spans=0,
            categories=[],
            spans=[],
            failures=dict(failures),
            detector_used=None,
            detection_confidence=None,
        )
        for _ in range(n)
    ]


def detect_pii(
    inputs: str | ScriptLine | Sequence[str | ScriptLine],
    detectors: list[str] | None = None,
    presidio_score_threshold: float = 0.4,
    gliner_model: str | None = None,
    gliner_labels: list[str] | None = None,
    gliner_threshold: float = 0.5,
    require_cross_source_corroboration: bool = True,
) -> PiiReport | list[PiiReport]:
    """Scan a string or ``ScriptLine`` (or a sequence of either) for PII.

    Standalone entry point: no dependency on any audio-workflow concept (a run "pass", a
    per-pass tag, an ASR-model ensemble). ``str`` and ``ScriptLine`` are two entry shapes
    over one implementation — a ``ScriptLine`` is flattened to text via
    :func:`flatten_script_line` first, then both shapes go through the same
    subprocess-backed detection.

    Every non-empty input is sent to ``detect_pii_via_subprocess`` in a single batched
    call (one entry per input index) rather than one subprocess per input — the venv's
    model loads (~5-10s each for Presidio/GLiNER) dominate cost, so batching a whole
    sequence avoids paying that repeatedly.

    Args:
        inputs: A single ``str``, a single ``ScriptLine``, or a sequence of either
            (may be mixed). A ``ScriptLine`` with no text anywhere in its tree (e.g. a
            diarization line carrying only a ``speaker`` label) flattens to ``""`` and
            is treated exactly like an empty string input — a documented empty result,
            not an error.
        detectors: Subset of ``{"presidio", "gliner", "rules"}`` to run. ``None`` (default)
            runs all three. An empty list (``[]``) is the caller explicitly disabling detection:
            no subprocess is spawned, every report gets ``detector_used=None`` and an
            explicit ``"pii_disabled"`` failure note — the LLM-judge-shaped case: this
            module ships no LLM judge, and there is deliberately no default-on path
            that would require one.
        presidio_score_threshold: Presidio entities below this score are dropped at
            extraction time. 0.4 is permissive enough to catch standard phone-number
            formats; cross-detector corroboration still gates the boolean flag for
            borderline scores.
        gliner_model: HuggingFace model id for GLiNER. ``None`` uses the subprocess
            module's default (``nvidia/gliner-pii``). Ignored when ``"gliner"`` is
            excluded from ``detectors``.
        gliner_labels: Labels passed to GLiNER's ``predict_entities``. ``None`` uses
            the subprocess module's curated HIPAA-18 default set.
        gliner_threshold: Drop GLiNER predictions below this score.
        require_cross_source_corroboration: When ``True`` (default) and ≥2 detectors
            ran, only flip ``contains_pii`` to ``True`` for a ``(category,
            normalized_text)`` pair that ≥2 of them independently
            flagged. A caller corroborating across multiple ASR transcripts of the same
            recording applies the same rationale one layer up (see the ``audio_analysis``
            workflow's per-pass PII adapter); this generalizes it to cross-detector
            agreement since a standalone input has exactly one transcript to corroborate.
            When only one detector ran, any single detection counts (corroboration cannot
            apply with one witness).

    Returns:
        A single ``PiiReport`` when ``inputs`` is a scalar ``str``/``ScriptLine``; a
        list of ``PiiReport``, same length and order as ``inputs``, when it is a
        sequence. ``detector_used=None`` and ``detection_confidence=None`` together
        mean detection did not actually run for that input (disabled, empty input,
        subprocess failure, or no detector loaded) — distinct from ``detection_confidence
        == 0.0``, which means detection ran and found nothing.
    """
    items: Sequence[str | ScriptLine]
    if isinstance(inputs, (str, ScriptLine)):
        is_scalar = True
        items = [inputs]
    else:
        is_scalar = False
        items = list(inputs)

    texts: list[str] = [flatten_script_line(item) if isinstance(item, ScriptLine) else item for item in items]

    # Explicit ``detectors=[]`` means the caller has chosen to disable PII detection.
    # Checked before anything else so it wins regardless of input content, giving a clean
    # three-way distinction between "pii_disabled", "ran, found nothing", and "subprocess
    # failed" that any caller layered on top of this function can rely on.
    if detectors is not None and len(detectors) == 0:
        disabled_reports = _empty_reports(
            len(texts), {"pii_disabled": "PII detection disabled by caller (detectors=[])."}
        )
        return disabled_reports[0] if is_scalar else disabled_reports

    # Index-keyed so a batch of inputs shares one subprocess call; empty/whitespace-only
    # entries never reach the detectors.
    transcripts_by_index: dict[str, str] = {str(i): t for i, t in enumerate(texts) if t and t.strip()}

    if not transcripts_by_index:
        return _quick_return(texts, {}, is_scalar)

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
        result = detect_pii_via_subprocess(transcripts_by_index, **subprocess_kwargs)
    except Exception as exc:  # noqa: BLE001 — caller needs reports to continue, not a crash
        msg = f"PII subprocess failed: {type(exc).__name__}: {exc}"
        print(f"warn: {msg}", file=sys.stderr)
        return _quick_return(texts, {"pii_subprocess": msg}, is_scalar)

    spans_by_index_raw = result.get("spans_by_asr", {})
    subprocess_failures = dict(result.get("failures", {}))
    detectors_used = list(result.get("detectors_used", []))
    for name, msg in subprocess_failures.items():
        print(f"warn: PII / {name}: {msg}", file=sys.stderr)

    if not detectors_used:
        no_detector_msg = (
            "No PII detector (Presidio, GLiNER, rules) loaded inside the subprocess venv; "
            "contains_pii=False reported by default."
        )
        subprocess_failures.setdefault("no_pii_detector", no_detector_msg)
        print(f"warn: {no_detector_msg}", file=sys.stderr)
        return _quick_return(texts, subprocess_failures, is_scalar)

    built_reports: list[PiiReport] = []
    for i, _text in enumerate(texts):
        spans = _materialize_spans(spans_by_index_raw.get(str(i), []), str(i))
        contains_pii = _corroborated_contains_pii(spans, detectors_used, require_cross_source_corroboration)
        built_reports.append(
            PiiReport(
                contains_pii=contains_pii,
                n_spans=len(spans),
                categories=sorted({s.category for s in spans}),
                spans=spans,
                failures=dict(subprocess_failures),
                detector_used=",".join(detectors_used),
                detection_confidence=_compute_detection_confidence(
                    spans, n_asr_models=1, n_detectors_run=len(detectors_used)
                ),
            )
        )
    return built_reports[0] if is_scalar else built_reports


def _quick_return(texts: list[str], failures: dict[str, str], is_scalar: bool) -> PiiReport | list[PiiReport]:
    """Shared tail for ``detect_pii``'s "detector did not run" branches.

    Args:
        texts: The flattened input texts (only its length matters here — one report
            per input).
        failures: Failure dict to attach to every report (empty for the "all inputs
            were empty, nothing to scan" branch).
        is_scalar: Whether the original ``detect_pii`` call took a scalar input.

    Returns:
        A bare ``PiiReport`` when ``is_scalar``, else a list matching ``texts``' length.
    """
    reports = _empty_reports(len(texts), failures)
    return reports[0] if is_scalar else reports


def _compute_detection_confidence(spans: list[PiiSpan], n_asr_models: int, n_detectors_run: int) -> float:
    """Aggregate per-span detector scores into a single ``[0, 1]`` confidence.

    Combines three signals per unique ``(category, normalized_text)`` finding:

    - **max raw detector confidence** on that finding (Presidio's analyzer
      score or GLiNER's prediction probability)
    - **cross-detector agreement** — fraction of the detectors that actually
      ran *for this report* that independently flagged the same
      ``(category, normalized_text)``. The denominator is ``n_detectors_run``,
      not the size of the module's known-detector set: dividing by the known
      set would cap a Presidio-only finding at ``1/len(_KNOWN_DETECTORS)``
      even when GLiNER was never asked to run (failed to load, or excluded
      via ``detectors=``) — as though a second detector had declined to
      corroborate it, when in fact none was consulted. It would also mean
      every detector added to the module silently rescales every confidence
      already published, since the historical denominator changes underfoot.
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
        n_detectors_run: Number of detectors that actually ran for this
            report (e.g. ``len(detectors_used)``), used as the denominator
            for cross-detector agreement. Clamped to at least 1 so a caller
            passing 0 (which should never happen — the caller short-circuits
            before reaching this function when no detector ran) still gets a
            defined score rather than a ``ZeroDivisionError``.

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
        key = (corroboration_family(s.category), normalized)
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
    denom_detectors = max(1, n_detectors_run)
    denom_asrs = max(1, n_asr_models)
    risks: list[float] = []
    for g in groups.values():
        detector_agreement = min(1.0, len(g["detectors"]) / denom_detectors)
        asr_agreement = min(1.0, len(g["asrs"]) / denom_asrs)
        risks.append(g["max_score"] * detector_agreement * asr_agreement)
    return max(risks) if risks else 0.0
