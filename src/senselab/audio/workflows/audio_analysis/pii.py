"""PII detection over ASR transcripts.

Two ML-based detectors, layered with graceful degradation:

1. **Microsoft Presidio Analyzer** (preferred) — industry-standard PII detection
   library combining spaCy NER with purpose-built recognizers for emails, phone
   numbers, SSNs, credit cards, IP addresses, dates, and locations. Each span
   carries a confidence score and an entity type.
2. **spaCy NER** (fallback) — bare ``en_core_web_sm`` / ``en_core_web_trf`` for
   PERSON / GPE / LOC / ORG entities. Less coverage than Presidio (no email /
   phone / SSN-specific recognizers) but available wherever spaCy is installed.

Regex-only detection is intentionally NOT supported here: ASR transcripts
routinely contain digit-heavy hallucinations and ambiguous date strings that
flip a regex-only ``contains_pii`` flag from a single false positive. ML
detectors with confidence scores let the workflow gate on a meaningful
threshold.

When neither detector is available, the report records an explicit failure
reason (``failures["no_pii_detector"]``) and ``contains_pii`` defaults to
``False`` — the user knows the check didn't run rather than getting a silent
all-clear.
"""

from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PiiSpan:
    """One PII detection in a transcript."""

    text: str
    category: str  # presidio entity_type, e.g. "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"
    source: str  # "presidio" or "spacy/<label>"
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
    detector_used: str | None = None  # "presidio" or "spacy" or None


# Default Presidio entity types we treat as PII. Excludes things Presidio
# detects but that aren't PII per se (e.g. URL alone without a personal site).
_PRESIDIO_PII_ENTITIES = {
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",  # nationality / religion / political affiliation
    "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "IBAN_CODE",
    "CRYPTO",
}

# spaCy NER entity types we treat as PII when Presidio is unavailable.
_SPACY_PII_ENTITIES = {"PERSON", "GPE", "LOC", "ORG", "FAC", "NORP"}


def _load_presidio_analyzer() -> tuple[Any, str | None]:  # noqa: ANN401
    """Try to load a Presidio AnalyzerEngine. Returns (analyzer_or_None, failure_msg)."""
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as exc:
        return None, (
            f"Presidio not installed ({exc!r}); install via "
            "'uv sync --extra pii' or 'pip install presidio-analyzer presidio-anonymizer'."
        )
    try:
        # ``AnalyzerEngine()`` constructs a default NLP pipeline + recognizer
        # registry. The first construction downloads / loads the spaCy model
        # Presidio depends on (en_core_web_lg by default).
        return AnalyzerEngine(), None
    except Exception as exc:  # noqa: BLE001 — surface the cause to the caller
        return None, f"Presidio AnalyzerEngine construction failed: {exc!r}"


def _load_spacy_nlp() -> tuple[Any, str | None]:  # noqa: ANN401
    """Try to load a spaCy English NER model. Returns (nlp_or_None, failure_msg)."""
    try:
        import spacy
    except ImportError as exc:
        return None, f"spaCy unavailable: {exc!r}"
    for model_name in ("en_core_web_trf", "en_core_web_md", "en_core_web_sm"):
        try:
            return spacy.load(model_name), None
        except (OSError, IOError):
            continue
    return None, (
        "no spaCy English model installed (tried en_core_web_trf/md/sm); "
        "install via 'uv run python -m spacy download en_core_web_sm'."
    )


def _scan_with_presidio(
    text: str,
    asr_model: str,
    pass_label: str,
    *,
    analyzer: Any,  # noqa: ANN401
    score_threshold: float,
) -> list[PiiSpan]:
    """Run Presidio over a transcript. Returns spans for entities in the PII allowlist."""
    if not text.strip():
        return []
    try:
        results = analyzer.analyze(text=text, language="en", score_threshold=score_threshold)
    except Exception as exc:  # noqa: BLE001
        print(
            f"warn: Presidio analyzer failed on transcript from {asr_model!r}: {exc!r}",
            file=sys.stderr,
        )
        return []
    seen: set[tuple[str, str]] = set()
    spans: list[PiiSpan] = []
    for r in results:
        entity_type = getattr(r, "entity_type", None) or ""
        if entity_type not in _PRESIDIO_PII_ENTITIES:
            continue
        try:
            start = int(r.start)
            end = int(r.end)
            score = float(r.score)
        except (AttributeError, TypeError, ValueError):
            continue
        span_text = text[start:end]
        normalized = span_text.strip().lower()
        key = (entity_type, normalized)
        if key in seen:
            continue
        seen.add(key)
        spans.append(
            PiiSpan(
                text=span_text,
                category=entity_type,
                source="presidio",
                asr_model=asr_model,
                pass_label=pass_label,
                score=score,
            )
        )
    return spans


def _scan_with_spacy(
    text: str,
    asr_model: str,
    pass_label: str,
    *,
    nlp: Any,  # noqa: ANN401
) -> list[PiiSpan]:
    """Run spaCy NER over a transcript when Presidio isn't available."""
    if not text.strip():
        return []
    try:
        doc = nlp(text)
    except Exception as exc:  # noqa: BLE001
        print(
            f"warn: spaCy NER failed on transcript from {asr_model!r}: {exc!r}",
            file=sys.stderr,
        )
        return []
    seen: set[tuple[str, str]] = set()
    spans: list[PiiSpan] = []
    for ent in doc.ents:
        if ent.label_ not in _SPACY_PII_ENTITIES:
            continue
        normalized = ent.text.strip().lower()
        key = (ent.label_, normalized)
        if key in seen:
            continue
        seen.add(key)
        spans.append(
            PiiSpan(
                text=ent.text,
                category=ent.label_,
                source=f"spacy/{ent.label_}",
                asr_model=asr_model,
                pass_label=pass_label,
                score=None,  # spaCy NER doesn't expose per-span confidence
            )
        )
    return spans


def detect_pii_in_pass(
    *,
    pass_label: str,
    asr_resolved: dict[str, Any],
    presidio_score_threshold: float = 0.4,
    require_cross_model_corroboration: bool = True,
) -> PiiPassReport:
    """Scan all ASR transcripts for one pass and return a unified PII report.

    Detector preference order:
    1. Microsoft Presidio (uses internal NLP + dedicated PII recognizers)
    2. spaCy NER (lower coverage, no per-span confidence)
    3. None — both unavailable; report records the failure and ``contains_pii``
       defaults to ``False`` so the workflow can continue.

    Args:
        pass_label: e.g. ``"raw_16k"``.
        asr_resolved: ``{asr_model_id → resolved_asr_result}``.
        presidio_score_threshold: Presidio entities below this score are
            dropped at extraction time. 0.4 is permissive enough to catch
            standard phone-number formats (Presidio assigns ~0.4 to phones
            without surrounding "tel:" / "phone:" context); cross-model
            corroboration still gates the boolean flag for borderline scores.
        require_cross_model_corroboration: When ``True`` (default), only flip
            ``contains_pii`` to ``True`` when a (category, normalized_text)
            pair is detected by ≥2 ASR models. Filters out hallucinated
            entities present in only one ASR's transcript.

    Returns:
        ``PiiPassReport`` with per-span detail, the detector used, and any
        failure reasons.
    """
    failures: dict[str, str] = {}

    analyzer, presidio_msg = _load_presidio_analyzer()
    nlp = None
    spacy_msg: str | None = None
    if analyzer is None:
        if presidio_msg is not None:
            failures["presidio"] = presidio_msg
            print(f"warn: PII / Presidio: {presidio_msg}", file=sys.stderr)
        nlp, spacy_msg = _load_spacy_nlp()
        if nlp is None and spacy_msg is not None:
            failures["spacy_ner"] = spacy_msg
            print(f"warn: PII / spaCy NER: {spacy_msg}", file=sys.stderr)

    detector_used: str | None = None
    if analyzer is not None:
        detector_used = "presidio"
    elif nlp is not None:
        detector_used = "spacy"
    else:
        failures["no_pii_detector"] = (
            "Neither Presidio nor spaCy is available — PII check did not run. "
            "contains_pii=False reported by default; install presidio-analyzer "
            "or spaCy + en_core_web_sm to enable detection."
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

    spans_by_asr: dict[str, list[PiiSpan]] = {}
    spans: list[PiiSpan] = []
    for asr_model, resolved in asr_resolved.items():
        items = resolved if isinstance(resolved, list) else [resolved]
        text_parts: list[str] = []
        for line in items:
            if isinstance(line, dict):
                t = line.get("text") or ""
            else:
                t = getattr(line, "text", "") or ""
            if t.strip():
                text_parts.append(t)
        full_text = " ".join(text_parts)
        if not full_text.strip():
            continue
        per_model: list[PiiSpan]
        if analyzer is not None:
            per_model = _scan_with_presidio(
                full_text,
                asr_model,
                pass_label,
                analyzer=analyzer,
                score_threshold=presidio_score_threshold,
            )
        else:
            per_model = _scan_with_spacy(full_text, asr_model, pass_label, nlp=nlp)
        spans_by_asr[asr_model] = per_model
        spans.extend(per_model)

    # Decide ``contains_pii``. Cross-model corroboration filters out
    # hallucinated entities that only one ASR backend produced.
    if not spans:
        contains_pii = False
    elif not require_cross_model_corroboration:
        contains_pii = True
    else:
        from collections import Counter

        norm_keys: Counter[tuple[str, str]] = Counter()
        for asr_model, per_model in spans_by_asr.items():
            seen_in_model: set[tuple[str, str]] = set()
            for s in per_model:
                key = (s.category, s.text.strip().lower())
                if key not in seen_in_model:
                    seen_in_model.add(key)
                    norm_keys[key] += 1
        contains_pii = any(count >= 2 for count in norm_keys.values())
        # When fewer than 2 ASR models are available, any single detection counts.
        if not contains_pii and len(spans_by_asr) < 2 and spans:
            contains_pii = True

    categories = sorted({s.category for s in spans})
    return PiiPassReport(
        pass_label=pass_label,
        contains_pii=contains_pii,
        n_spans=len(spans),
        categories=categories,
        spans=spans,
        failures=failures,
        detector_used=detector_used,
    )


def report_to_dict(report: PiiPassReport) -> dict[str, Any]:
    """Convert a ``PiiPassReport`` into a JSON-serializable dict."""
    return {
        "pass_label": report.pass_label,
        "contains_pii": report.contains_pii,
        "n_spans": report.n_spans,
        "categories": report.categories,
        "detector_used": report.detector_used,
        "spans": [asdict(s) for s in report.spans],
        "failures": report.failures,
    }
