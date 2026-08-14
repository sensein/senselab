"""Standalone PII detection over plain text, ``ScriptLine``, and ASR transcripts.

Two halves, kept separate on purpose. :func:`scan_for_pii` **runs the detectors** and
returns a :class:`PiiScan` -- spans, which detectors ran, what failed -- and decides
nothing. :func:`decide_pii` **aggregates that evidence to a verdict**: how many detectors
must agree, how scores and agreement combine into a confidence, what ``contains_pii``
becomes. :func:`detect_pii` is the convenience composition of the two for the common case.

The split is the point. Running a detector and concluding something from its output are
different jobs with different parameters -- a score floor is a property of the tool, a
corroboration rule is a judgement -- and a caller who needs a different judgement should
be able to keep the scan and replace the decision without reimplementing either.

All three accept a ``str``, a ``ScriptLine`` (optionally
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

A fourth, **the optional local-LLM detector** (``local_llm.py``), is known but never
default-on: it has to be named in ``detectors`` explicitly. It runs in *this* process
rather than the venv — it needs nothing but stdlib ``urllib`` — and refuses any
non-loopback endpoint, so transcript text cannot leave the machine. See
:func:`default_detectors` for why a network detector is opt-in.

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

from senselab.text.tasks.pii_detection.local_llm import LocalLlmConfig, scan_or_fail
from senselab.text.tasks.pii_detection.subprocess_backend import (
    _DEFAULT_DETECTORS,
    _KNOWN_DETECTORS,
    DETECTOR_GLINER,
    DETECTOR_LLM,
    DETECTOR_PRESIDIO,
    DETECTOR_RULES,
    detect_pii_via_subprocess,
)
from senselab.utils.data_structures import ScriptLine

# Re-export the canonical detector names so ``analyze_audio.py`` (and
# any other caller wiring up a ``--pii-detectors`` flag) can reference
# them as ``pii.DETECTOR_PRESIDIO`` / ``pii.DETECTOR_GLINER`` rather
# than reaching into the subprocess-specific module.
__all__ = [
    "DETECTOR_GLINER",
    "DETECTOR_LLM",
    "DETECTOR_PRESIDIO",
    "DETECTOR_RULES",
    "LocalLlmConfig",
    "PiiReport",
    "PiiSpan",
    "PiiScan",
    "decide_pii",
    "default_detectors",
    "detect_pii",
    "flatten_script_line",
    "scan_for_pii",
]


def default_detectors() -> list[str]:
    """Return the detectors ``detectors=None`` runs, in sorted order.

    Narrower than ``_KNOWN_DETECTORS``: the local-LLM detector is accepted by name and
    counted in the cross-detector agreement denominator when a caller turns it on, but
    is never default-on. A default-on network detector would make a scan depend on
    whether a server happened to be listening, so the same corpus would score
    differently on two machines with nothing in the report explaining the gap.

    Returns:
        Sorted detector names, e.g. ``["gliner", "presidio", "rules"]``.
    """
    return sorted(_DEFAULT_DETECTORS)


@dataclass
class PiiSpan:
    """One PII detection in a transcript."""

    text: str
    category: str  # presidio entity_type, e.g. "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"
    source: str  # "presidio" or "gliner/<original_label>"
    asr_model: str
    score: float | None = None  # detector confidence in [0, 1]


@dataclass
class PiiScan:
    """What the detectors found for one input, before anything is concluded from it.

    The output of :func:`scan_for_pii` and the input to :func:`decide_pii`. It carries
    evidence and nothing else: no ``contains_pii``, no confidence, no threshold applied.
    Running the detectors and deciding what their agreement means are separate jobs, and a
    caller that wants the second one done differently — a different corroboration rule, a
    severity ordering this module deliberately does not impose, an aggregation across
    several transcripts of the same recording — needs the evidence without a verdict
    already baked into it.

    Attributes:
        spans: Deduped detections for this input. Deduplication is normalisation, not
            judgement: two detectors reporting the same ``(category, text, source)`` are
            one finding by definition.
        detectors_used: Names of the detectors that actually ran. Load-bearing for any
            agreement rule downstream — it is the denominator, and "two of two agreed" is
            a different claim from "two of four agreed".
        failures: Why a detector did not run, keyed by name. An empty ``spans`` with a
            populated ``failures`` is "we could not check", which must never be read as
            "we checked and it was clean".
    """

    spans: list[PiiSpan] = field(default_factory=list)
    detectors_used: list[str] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)


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


def scan_for_pii(
    inputs: str | ScriptLine | Sequence[str | ScriptLine],
    detectors: list[str] | None = None,
    presidio_score_threshold: float = 0.4,
    gliner_model: str | None = None,
    gliner_labels: list[str] | None = None,
    gliner_threshold: float = 0.5,
    local_llm_config: LocalLlmConfig | None = None,
) -> PiiScan | list[PiiScan]:
    """Run the detectors over a string or ``ScriptLine`` and return what they found.

    Execution only. Every parameter here says *how to run the detectors* — which ones, at
    what per-detector score floor, against which label set, at which endpoint. Nothing here
    decides what the findings mean; that is :func:`decide_pii`'s job, and the two are
    separate functions so a caller can keep one and replace the other.

    Standalone: no dependency on any audio-workflow concept (a run "pass", a per-pass tag,
    an ASR-model ensemble). A ``ScriptLine`` is flattened to text via
    :func:`flatten_script_line` first, then both shapes go through the same
    subprocess-backed detection.

    Every non-empty input is sent to ``detect_pii_via_subprocess`` in a single batched call
    (one entry per input index) rather than one subprocess per input — the venv's model
    loads (~5-10s each for Presidio/GLiNER) dominate cost, so batching a whole sequence
    pays that once.

    Args:
        inputs: A ``str``, a ``ScriptLine``, or a sequence of either.
        detectors: Subset of ``{"presidio", "gliner", "rules", "llm"}`` to run. ``None``
            (default) runs :func:`default_detectors` — the first three; ``"llm"`` is known
            but never default-on, so it has to be named explicitly. An empty list (``[]``)
            is the caller explicitly disabling detection: no subprocess is spawned and the
            scan comes back with an explicit ``"pii_disabled"`` failure note.
        presidio_score_threshold: Drop Presidio detections below this score.
        gliner_model: Override the GLiNER checkpoint.
        gliner_labels: Labels passed to GLiNER's ``predict_entities``. ``None`` uses the
            subprocess module's curated HIPAA-18 default set. Keep it flat — see
            ``doc.md`` on competing-claim interference.
        gliner_threshold: Drop GLiNER predictions below this score.
        local_llm_config: Endpoint for the optional local-LLM detector. Ignored unless
            ``"llm"`` is named in ``detectors``; defaults to
            :class:`~senselab.text.tasks.pii_detection.local_llm.LocalLlmConfig`'s own
            loopback default. A non-loopback URL raises at construction — see that module
            for why that is a checked invariant rather than a documented one.

    Returns:
        A single :class:`PiiScan` when ``inputs`` is a scalar; a list of the same length and
        order otherwise. An empty ``detectors_used`` means no detector ran for that batch,
        and ``failures`` says why — never confuse it with "ran and found nothing", which is
        an empty ``spans`` alongside a populated ``detectors_used``.
    """
    items: Sequence[str | ScriptLine]
    if isinstance(inputs, (str, ScriptLine)):
        is_scalar = True
        items = [inputs]
    else:
        is_scalar = False
        items = list(inputs)

    texts: list[str] = [flatten_script_line(item) if isinstance(item, ScriptLine) else item for item in items]

    def _finish(scans: list[PiiScan]) -> PiiScan | list[PiiScan]:
        return scans[0] if is_scalar else scans

    def _empty(failures: dict[str, str]) -> list[PiiScan]:
        return [PiiScan(spans=[], detectors_used=[], failures=dict(failures)) for _ in texts]

    # Explicit ``detectors=[]`` means the caller has chosen to disable PII detection.
    # Checked before anything else so it wins regardless of input content, giving a clean
    # three-way distinction between "pii_disabled", "ran, found nothing", and "subprocess
    # failed" that any caller layered on top of this function can rely on.
    if detectors is not None and len(detectors) == 0:
        return _finish(_empty({"pii_disabled": "PII detection disabled by caller (detectors=[])."}))

    # Index-keyed so a batch of inputs shares one subprocess call; empty/whitespace-only
    # entries never reach the detectors.
    transcripts_by_index: dict[str, str] = {str(i): t for i, t in enumerate(texts) if t and t.strip()}

    if not transcripts_by_index:
        return _finish(_empty({}))

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
    except Exception as exc:  # noqa: BLE001 — caller needs a scan to continue, not a crash
        msg = f"PII subprocess failed: {type(exc).__name__}: {exc}"
        print(f"warn: {msg}", file=sys.stderr)
        return _finish(_empty({"pii_subprocess": msg}))

    spans_by_index_raw = result.get("spans_by_asr", {})
    failures = dict(result.get("failures", {}))
    detectors_used = list(result.get("detectors_used", []))

    # The local LLM runs here, not in the venv -- it needs nothing but stdlib urllib
    # (see local_llm.py). Merged before the "did anything run?" check below so that
    # `detectors=["llm"]`, which leaves the worker with nothing to do, is still a real
    # scan rather than a no-detector early return.
    if detectors is not None and DETECTOR_LLM in detectors:
        # Keyed off transcripts_by_index, not texts: empty/whitespace-only inputs never
        # reach the venv detectors either, and asking an LLM about an empty string costs
        # a round trip to be told nothing.
        llm_results = {
            i: scan_or_fail(t, local_llm_config or LocalLlmConfig()) for i, t in transcripts_by_index.items()
        }
        first_failure = next((r.failure for r in llm_results.values() if r.failure), None)
        if first_failure is None:
            for index, llm_result in llm_results.items():
                spans_by_index_raw.setdefault(index, []).extend(llm_result.spans)
            detectors_used.append(DETECTOR_LLM)
        else:
            # One failure fails the detector for the whole batch rather than leaving it
            # counted as "ran" for some inputs and not others: `detectors_used` is the
            # agreement denominator, so a per-input denominator would make two scans in
            # one batch incomparable.
            failures["llm"] = first_failure

    for name, msg in failures.items():
        print(f"warn: PII / {name}: {msg}", file=sys.stderr)

    if not detectors_used:
        no_detector_msg = "No PII detector (Presidio, GLiNER, rules, llm) ran."
        failures.setdefault("no_pii_detector", no_detector_msg)
        print(f"warn: {no_detector_msg}", file=sys.stderr)
        return _finish(_empty(failures))

    return _finish(
        [
            PiiScan(
                spans=_materialize_spans(spans_by_index_raw.get(str(i), []), str(i)),
                detectors_used=list(detectors_used),
                failures=dict(failures),
            )
            for i in range(len(texts))
        ]
    )


def decide_pii(
    scans: PiiScan | Sequence[PiiScan],
    require_cross_source_corroboration: bool = True,
    n_sources: int = 1,
) -> PiiReport | list[PiiReport]:
    """Turn detector evidence into a verdict.

    The aggregation-and-threshold half, deliberately separate from :func:`scan_for_pii`.
    Everything here is a *decision*: how many detectors must agree before a finding counts,
    how per-detector scores and agreement combine into one confidence, and what
    ``contains_pii`` ends up being. None of it re-runs a detector, and none of
    :func:`scan_for_pii`'s parameters appear here — that separation is the point. A caller
    with a different corroboration rule, a severity ordering this module declines to
    impose, or an aggregation across several transcripts of one recording replaces this
    function and keeps the scan.

    Args:
        scans: One :class:`PiiScan`, or a sequence of them.
        require_cross_source_corroboration: When ``True`` (default) and ≥2 detectors ran,
            only flip ``contains_pii`` for a ``(category, normalized_text)`` pair that ≥2
            of them independently flagged. Agreement keys on a coarse family rather than
            the raw category, so two detectors naming the same entity slightly differently
            still corroborate. When only one detector ran, corroboration cannot apply and
            any single detection counts — one witness is not a quorum, but it is not
            nothing either.
        n_sources: How many independent transcripts of the same underlying content this
            evidence came from. ``1`` for standalone text. A workflow that scans several
            ASR backends' transcripts of one recording passes the count so cross-source
            agreement can raise confidence — a span only one transcript contains is the
            prototypical ASR hallucination.

    Returns:
        A single ``PiiReport`` for a single scan; a list of the same length and order for a
        sequence. ``detector_used=None`` and ``detection_confidence=None`` together mean
        detection did not actually run — distinct from ``detection_confidence == 0.0``,
        which means it ran and found nothing.
    """
    is_scalar = isinstance(scans, PiiScan)
    scan_list = [scans] if isinstance(scans, PiiScan) else list(scans)

    reports: list[PiiReport] = []
    for scan in scan_list:
        if not scan.detectors_used:
            # Nothing ran, so there is nothing to decide. contains_pii=False is the safe
            # report, and detection_confidence stays None so it cannot be read as 0.0
            # ("checked, clean").
            reports.append(
                PiiReport(
                    contains_pii=False,
                    n_spans=0,
                    categories=[],
                    spans=[],
                    failures=dict(scan.failures),
                    detector_used=None,
                    detection_confidence=None,
                )
            )
            continue

        reports.append(
            PiiReport(
                contains_pii=_corroborated_contains_pii(
                    scan.spans, scan.detectors_used, require_cross_source_corroboration
                ),
                n_spans=len(scan.spans),
                categories=sorted({s.category for s in scan.spans}),
                spans=scan.spans,
                failures=dict(scan.failures),
                detector_used=",".join(scan.detectors_used),
                detection_confidence=_compute_detection_confidence(
                    scan.spans, n_asr_models=n_sources, n_detectors_run=len(scan.detectors_used)
                ),
            )
        )

    return reports[0] if is_scalar else reports


def detect_pii(
    inputs: str | ScriptLine | Sequence[str | ScriptLine],
    detectors: list[str] | None = None,
    presidio_score_threshold: float = 0.4,
    gliner_model: str | None = None,
    gliner_labels: list[str] | None = None,
    gliner_threshold: float = 0.5,
    require_cross_source_corroboration: bool = True,
    local_llm_config: LocalLlmConfig | None = None,
) -> PiiReport | list[PiiReport]:
    """Scan for PII and decide on the result: :func:`scan_for_pii` then :func:`decide_pii`.

    The convenience path for the common case, and deliberately nothing more than the
    composition of the two halves — it exists so a caller who wants the default decision
    does not have to write both calls, not to hide the split. Reach for the halves directly
    when the decision needs to differ from the default: ``scan_for_pii`` gives the evidence,
    and any rule can be applied to it.

    Args:
        inputs: See :func:`scan_for_pii`.
        detectors: See :func:`scan_for_pii`.
        presidio_score_threshold: See :func:`scan_for_pii`.
        gliner_model: See :func:`scan_for_pii`.
        gliner_labels: See :func:`scan_for_pii`.
        gliner_threshold: See :func:`scan_for_pii`.
        require_cross_source_corroboration: See :func:`decide_pii`.
        local_llm_config: See :func:`scan_for_pii`.

    Returns:
        A single ``PiiReport`` when ``inputs`` is a scalar; a list of the same length and
        order otherwise.
    """
    scans = scan_for_pii(
        inputs,
        detectors=detectors,
        presidio_score_threshold=presidio_score_threshold,
        gliner_model=gliner_model,
        gliner_labels=gliner_labels,
        gliner_threshold=gliner_threshold,
        local_llm_config=local_llm_config,
    )
    return decide_pii(scans, require_cross_source_corroboration=require_cross_source_corroboration)


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
