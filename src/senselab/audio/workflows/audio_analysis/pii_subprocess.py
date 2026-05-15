"""Isolated subprocess venv that runs PII detection on ASR transcripts.

Why a subprocess at all
-----------------------
The host venv may be on a Python version that doesn't yet have wheels for
``spaCy`` (which tops out at cp313 as of writing) or for transformer-backed
GLiNER. Rather than pinning the entire project to an older Python — a heavy
constraint when only one downstream feature needs it — PII detection runs in
its own Python-3.13 venv created via ``ensure_venv``. The host can stay on
whatever recent Python it wants; PII still works.

What runs inside the venv
-------------------------
Two PII detectors run in series on each input transcript and their spans
are merged:

1. **Microsoft Presidio Analyzer** — regex + spaCy-NER orchestrator with
   purpose-built recognizers for emails, phones, SSNs, credit cards, IPs,
   dates, and so on. Each span carries a confidence score and a Presidio
   entity type like ``PERSON`` / ``EMAIL_ADDRESS``.
2. **GLiNER PII** (``nvidia/gliner-pii`` by default) — a transformer-
   based zero-shot NER model fine-tuned on ~100k synthetic PII / PHI
   records. Catches the HIPAA Safe Harbor identifiers Presidio doesn't
   natively recognize (``medical_record_number``,
   ``health_plan_number``, ``account_number``, ``fax_number``, ``url``,
   ``biometric_identifier``, ``unique_identifier``, etc.).

GLiNER's lowercase labels (``"person"``, ``"phone_number"``, ...) are
normalized to Presidio's uppercase scheme inside the worker so the
downstream corroboration logic in ``pii.py`` — which treats ``(category,
text.lower())`` as the dedupe key — sees the two detectors' findings as
referring to the same entity when they do.

What lives in this module
-------------------------
- Venv constants (name, Python version, requirements list).
- Default GLiNER model id and label set, plus the GLiNER → Presidio
  category map.
- The worker script that ``ensure_venv`` runs inside the isolated venv.
- ``detect_pii_via_subprocess`` — the dispatch function called by
  ``pii.py``.

The first call after a fresh venv build pays ~30 s for the GLiNER + spaCy
model loads (in addition to the venv build itself). Subsequent calls pay
the load cost again because each call is a fresh subprocess; batching all
of a subject's transcripts into one call is a worthwhile follow-up
optimization but isn't required for correctness.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any, Optional

from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

_PII_VENV = "pii-detection"
_PII_PYTHON = "3.13"

# en_core_web_lg pinned as a direct wheel URL so uv installs it as part
# of the venv resolution (no separate post-install ``python -m spacy
# download`` step needed). 3.8.0 matches the Presidio default NLP
# config's expected spaCy schema.
_EN_CORE_WEB_LG_WHEEL = (
    "en_core_web_lg @ "
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl"
)

# ``torch`` is named here explicitly so ``ensure_venv``'s auto-detection
# (post-PR #516) triggers Stage 1 routing through the matched CUDA wheel
# index. GLiNER pulls torch transitively; without the explicit pin, the
# transitive resolve in Stage 2 would skip the CUDA-aware routing and
# could land on a CPU-only wheel even on GPU hosts. ``torchaudio`` is
# NOT in this list — neither Presidio nor GLiNER decode audio.
_PII_REQUIREMENTS = [
    "presidio-analyzer>=2.2",
    "spacy>=3.7,<3.9",
    _EN_CORE_WEB_LG_WHEEL,
    "gliner>=0.2",
    "transformers>=4.40",
    "torch>=2.8,<2.9",
]

# nvidia/gliner-pii: 570M-param GLiNER variant fine-tuned on a ~100k
# synthetic PII/PHI dataset. English-only, NVIDIA Open Model License.
# 55+ pre-trained PII categories including medical / health info — a
# better fit for clinical voice data than zero-shotting a general-
# purpose GLiNER model at PII labels. Override via the dispatch
# function's ``gliner_model`` argument if you need multilingual
# coverage (urchade/gliner_multi-v2.1 is the usual swap).
_DEFAULT_GLINER_MODEL = "nvidia/gliner-pii"

# Labels passed to GLiNER's ``predict_entities``. The set below is the
# HIPAA Safe Harbor 18 identifiers (matches b2aiprep PR #256 verbatim).
# Callers can override via ``gliner_labels``.
#
# ──────────────────────────────────────────────────────────────────────
# DO NOT add overlapping labels to this list.
# ──────────────────────────────────────────────────────────────────────
# ``nvidia/gliner-pii`` exhibits competing-claim interference: when two
# labels can plausibly cover the same span, the model commits the span
# to one and silently drops it from consideration under the other —
# *even when the other would have been correct*. We discovered this
# empirically with a diagnostic on ``john.doe@example.com``:
#
#   Labels passed: [person, first_name, last_name, email, email_address, ...]
#   GLiNER output: ``John`` and ``Doe`` classified as first/last_name at
#     score 1.0; the full ``john.doe@example.com`` substring received
#     NO email span at any score above 0.0.
#
#   Labels passed: [name, address, date, phone_number, email, ssn, ...]
#     (a flat HIPAA-style set with one ``name`` and one ``email``)
#   GLiNER output: ``John`` and ``Doe`` as ``name`` at score 1.0,
#     ``john.doe@example.com`` as ``email`` at score 1.0. Both caught.
#
# Same model, same threshold, same input — the only difference was
# label-set granularity. The flat set wins because nothing competes.
# So: keep this list flat (e.g. ``name`` not ``person``+``first_name``+
# ``last_name``; ``address`` not ``address``+``street_address``+``city``;
# ``date`` not ``date``+``date_of_birth``). If you need finer granularity
# downstream, derive it from the matched substring or layer a second
# model — don't fight the GLiNER trainer here.
#
# ──────────────────────────────────────────────────────────────────────
# DO NOT extend this list beyond the HIPAA-18 either.
# ──────────────────────────────────────────────────────────────────────
# An earlier revision of this constant tacked on two clinical-voice
# extensions (``health_condition``, ``medication``) on top of the
# HIPAA list. They aren't semantically overlapping with anything in the
# HIPAA-18 — they cover spans neither ``email`` nor ``name`` would —
# but adding them was enough to drop the email's GLiNER score below
# the 0.5 threshold and cause it to silently disappear. The mechanism
# isn't fully understood (label-list length effect on the model's
# projection space, possibly) but the symptom is robust. Until we
# understand it, ship the HIPAA-18 exactly. Callers who need extra
# labels can pass them via ``gliner_labels=[...]`` and verify that
# their specific labels don't trip the same interference on their
# test corpus.
_DEFAULT_GLINER_LABELS = [
    # HIPAA Safe Harbor 18 identifiers (per 45 CFR §164.514(b)(2),
    # matching b2aiprep's ``hipaa_labels.json`` in PR #256 verbatim so
    # the two projects' PII outputs share a vocabulary).
    "name",
    "address",
    "date",
    "phone_number",
    "fax_number",
    "email",
    "social_security_number",
    "medical_record_number",
    "health_plan_number",
    "account_number",
    "license_number",
    "vehicle_identifier",
    "device_identifier",
    "url",
    "ip_address",
    "biometric_identifier",
    "photographic_image",
    "unique_identifier",
]

# GLiNER returns lowercase labels matching the prompt; Presidio returns
# uppercase entity types like ``PERSON``. Normalize GLiNER → Presidio so
# the downstream cross-model corroboration logic in ``pii.py`` (which
# keys on ``(category, text.lower())``) treats both detectors' hits on
# the same span as the same finding. Labels missing from this map fall
# through to ``str.upper()`` so they remain consistent in scheme even
# if they don't have a direct Presidio analogue.
_GLINER_TO_PRESIDIO_CATEGORY: dict[str, str] = {
    "person": "PERSON",
    "first_name": "PERSON",
    "last_name": "PERSON",
    "full_name": "PERSON",
    "name": "PERSON",
    "email": "EMAIL_ADDRESS",
    "email_address": "EMAIL_ADDRESS",
    "phone_number": "PHONE_NUMBER",
    "phone": "PHONE_NUMBER",
    "address": "LOCATION",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "location": "LOCATION",
    "social_security_number": "US_SSN",
    "ssn": "US_SSN",
    "credit_card_number": "CREDIT_CARD",
    "credit_card": "CREDIT_CARD",
    "card_number": "CREDIT_CARD",
    "date_of_birth": "DATE_TIME",
    "birth_date": "DATE_TIME",
    "date": "DATE_TIME",
    "ip_address": "IP_ADDRESS",
    "ip": "IP_ADDRESS",
    # Medical entities Presidio doesn't cover get their own category
    # names rather than being shoehorned into MEDICAL_LICENSE.
    "medical_record_number": "MEDICAL_RECORD_NUMBER",
    "mrn": "MEDICAL_RECORD_NUMBER",
    "patient_id": "MEDICAL_RECORD_NUMBER",
    "health_condition": "HEALTH_CONDITION",
    "medical_condition": "HEALTH_CONDITION",
    "diagnosis": "HEALTH_CONDITION",
    "medication": "MEDICATION",
    "drug": "MEDICATION",
}

# Default Presidio entity types treated as PII inside the worker. Mirrors
# the set ``pii.py`` historically used in-process so the subprocess-only
# refactor preserves behavior — Presidio detects many entity types
# (URL, etc.) that aren't PII per se and we don't want them flagged.
_PRESIDIO_PII_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "NRP",
    "MEDICAL_LICENSE",
    "US_DRIVER_LICENSE",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "IBAN_CODE",
    "CRYPTO",
]


# Recognized detector names — the public surface for opt-in / opt-out.
# Aliased to the canonical strings ``detect_pii_via_subprocess`` accepts
# in its ``detectors`` kwarg.
DETECTOR_PRESIDIO = "presidio"
DETECTOR_GLINER = "gliner"
_KNOWN_DETECTORS = frozenset({DETECTOR_PRESIDIO, DETECTOR_GLINER})


# Worker script — runs inside the isolated venv. Reads a single JSON
# request from stdin and writes a single JSON response to stdout.
#
# Request shape:
#   {
#     "transcripts": {asr_model: full_text_string, ...},
#     "detectors": [str, ...],   # subset of {"presidio", "gliner"}
#     "presidio_entities": [str, ...],
#     "presidio_score_threshold": float,
#     "gliner_model": str,
#     "gliner_labels": [str, ...],
#     "gliner_threshold": float,
#     "gliner_label_map": {str: str, ...},   # gliner_label → normalized_category
#   }
#
# Detectors not named in the request are NOT loaded — saves ~5-10 s of
# model-load time per skipped detector. A detector that fails to load
# is recorded in ``failures[name]`` and silently sits out; the other
# may still produce results.
#
# Response shape (success):
#   {
#     "spans_by_asr": {asr_model: [{text, category, source, score}, ...]},
#     "failures": {detector_name: error_message},
#     "detectors_used": [str, ...]
#   }
#
# Response shape (catastrophic failure, e.g. unparsable input):
#   {"error": {"type": ..., "message": ..., "traceback": ...}}
_PII_WORKER_SCRIPT = r"""
import json
import sys
import traceback

try:
    args = json.loads(sys.stdin.read())
    transcripts = args["transcripts"]
    detectors_requested = set(args.get("detectors") or [])
    presidio_entities = args.get("presidio_entities") or []
    presidio_score_threshold = float(args.get("presidio_score_threshold", 0.4))
    gliner_model_id = args.get("gliner_model")
    gliner_labels = args.get("gliner_labels") or []
    gliner_threshold = float(args.get("gliner_threshold", 0.5))
    gliner_label_map = args.get("gliner_label_map") or {}

    failures = {}
    detectors_used = []

    # ── Presidio ──────────────────────────────────────────────────
    analyzer = None
    if "presidio" in detectors_requested:
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            detectors_used.append("presidio")
        except Exception as exc:
            failures["presidio"] = f"{type(exc).__name__}: {exc}"

    # ── GLiNER ────────────────────────────────────────────────────
    gliner_model = None
    if "gliner" in detectors_requested and gliner_model_id and gliner_labels:
        try:
            import torch
            from gliner import GLiNER
            gliner_model = GLiNER.from_pretrained(gliner_model_id)
            if torch.cuda.is_available():
                # GLiNER 0.2+ exposes the wrapped HF model on .model; older
                # builds use the wrapper directly. Try both gracefully so
                # the worker doesn't fail just because the API shifts.
                try:
                    gliner_model.model = gliner_model.model.cuda()
                except AttributeError:
                    gliner_model = gliner_model.cuda()
            detectors_used.append("gliner")
        except Exception as exc:
            failures["gliner"] = f"{type(exc).__name__}: {exc}"
            gliner_model = None

    def _normalize_category(label):
        return gliner_label_map.get(label, label.upper())

    def _presidio_scan(text):
        if analyzer is None:
            return []
        # Pass the entity list through unchanged so an empty list means
        # "scan for nothing" (Presidio semantics: entities=None scans for
        # ALL entity types — collapsing [] → None would invert intent).
        results = analyzer.analyze(
            text=text,
            entities=presidio_entities,
            language="en",
            score_threshold=presidio_score_threshold,
        )
        out = []
        for r in results:
            # Belt-and-suspenders filter against Presidio returning a
            # category outside the requested set. `is not None` rather
            # than truthiness so an explicit empty list filters everything.
            if presidio_entities is not None and r.entity_type not in presidio_entities:
                continue
            out.append({
                "text": text[r.start:r.end],
                "category": r.entity_type,
                "source": "presidio",
                "score": float(r.score),
            })
        return out

    def _gliner_scan(text):
        if gliner_model is None:
            return []
        entities = gliner_model.predict_entities(
            text,
            gliner_labels,
            threshold=gliner_threshold,
        )
        out = []
        for ent in entities:
            label_raw = ent.get("label", "")
            out.append({
                "text": ent.get("text", ""),
                "category": _normalize_category(label_raw),
                "source": f"gliner/{label_raw}",
                "score": float(ent.get("score", 0.0)),
            })
        return out

    spans_by_asr = {}
    for asr_model, full_text in transcripts.items():
        if not isinstance(full_text, str) or not full_text.strip():
            spans_by_asr[asr_model] = []
            continue
        merged = []
        merged.extend(_presidio_scan(full_text))
        merged.extend(_gliner_scan(full_text))
        spans_by_asr[asr_model] = merged

    print(json.dumps({
        "spans_by_asr": spans_by_asr,
        "failures": failures,
        "detectors_used": detectors_used,
    }))
except Exception as exc:
    err = {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(limit=8),
    }
    print(json.dumps({"error": err}))
    sys.exit(1)
"""


def detect_pii_via_subprocess(
    transcripts_by_asr: dict[str, str],
    *,
    detectors: Optional[list[str]] = None,
    presidio_score_threshold: float = 0.4,
    presidio_entities: Optional[list[str]] = None,
    gliner_model: str = _DEFAULT_GLINER_MODEL,
    gliner_labels: Optional[list[str]] = None,
    gliner_threshold: float = 0.5,
    timeout: int = 600,
) -> dict[str, Any]:
    """Run Presidio + GLiNER PII detection on per-ASR-model transcripts.

    The call is hermetic: ``ensure_venv`` builds (or reuses) the
    ``pii-detection`` venv at first call, then spawns a fresh worker
    process for each invocation that loads the requested detector(s),
    scans every transcript, and exits. Cold-start cost is dominated by
    the model loads — ~5 s for Presidio + spaCy, ~5 s for GLiNER; the
    ``detectors`` parameter lets callers skip whichever they don't need.

    Args:
        transcripts_by_asr: Mapping ``asr_model_id → concatenated transcript``.
            One entry per ASR backend whose output should be scanned.
            Empty / whitespace-only values are returned with an empty
            spans list and never reach the detectors.
        detectors: Which detectors to run. ``None`` (default) runs both
            ``"presidio"`` and ``"gliner"``. Passing a single-element
            list like ``["presidio"]`` skips the GLiNER load entirely;
            ``["gliner"]`` skips Presidio. An empty list short-circuits
            without invoking the subprocess at all (mirrors the
            "PII detection disabled" case). Unknown detector names raise
            ``ValueError`` immediately.
        presidio_score_threshold: Drop Presidio spans below this score.
            0.4 matches the in-process default tuned to catch standard
            phone-number formats without a "tel:" / "phone:" lead-in.
        presidio_entities: Restrict Presidio to this allow-list of entity
            types (e.g. ``["PERSON", "EMAIL_ADDRESS"]``). ``None`` uses
            the curated default set defined in ``_PRESIDIO_PII_ENTITIES``.
        gliner_model: HuggingFace model id. Default
            ``nvidia/gliner-pii`` (English, PII-fine-tuned, ~570 M params).
            Pass a different id (e.g. ``urchade/gliner_multi-v2.1``) for
            multilingual coverage or smaller-model variants.
        gliner_labels: Labels passed to GLiNER's ``predict_entities``.
            ``None`` uses ``_DEFAULT_GLINER_LABELS`` — the HIPAA Safe
            Harbor 18 identifiers verbatim (matches b2aiprep #256).
            **Keep label sets flat AND don't extend past the HIPAA-18
            unless you've verified your specific additions don't trip
            the competing-claim interference — see the ``DO NOT``
            warnings above the constant for the empirical
            justification.**
        gliner_threshold: Drop GLiNER predictions below this score.
        timeout: Subprocess wall-clock cap. 10 minutes is generous for
            one pass's worth of transcripts; the bulk of the time is the
            initial model loads.

    Returns:
        Dict with three keys:

        - ``"spans_by_asr"`` — ``{asr_model: [{text, category, source, score}, ...]}``.
          Spans from each requested detector are concatenated per ASR
          model; the ``source`` field distinguishes ``"presidio"`` from
          ``"gliner/<original_label>"`` so callers can audit which
          detector produced what.
        - ``"failures"`` — ``{detector_name: error_message}``. Empty when
          every requested detector loaded cleanly. A populated entry
          means that detector silently sat out for this call but any
          other requested detector may still have produced results.
        - ``"detectors_used"`` — list of detector names that successfully
          loaded. ``[]`` means no detector ran and the spans list will
          always be empty for every ASR model.

    Raises:
        ValueError: If ``detectors`` contains an unknown detector name.
        subprocess.CalledProcessError or RuntimeError reconstructed from
            the worker's error JSON. Either case means the subprocess
            couldn't produce ANY result; the caller should record a
            failure and proceed without PII findings rather than crash.
    """
    if detectors is None:
        detectors_resolved = sorted(_KNOWN_DETECTORS)
    else:
        unknown = [d for d in detectors if d not in _KNOWN_DETECTORS]
        if unknown:
            raise ValueError(f"Unknown PII detector(s): {unknown!r}. Known: {sorted(_KNOWN_DETECTORS)!r}.")
        # Preserve caller's order but dedupe; the worker treats the field
        # as a set anyway, so ordering only matters for readability.
        seen: set[str] = set()
        detectors_resolved = []
        for d in detectors:
            if d not in seen:
                seen.add(d)
                detectors_resolved.append(d)

    # An empty detector list is a valid "skip everything" signal — return
    # the same shape as a successful run with no detectors loaded, but
    # don't spend the time spawning the subprocess or building the venv.
    if not detectors_resolved:
        return {
            "spans_by_asr": {asr_model: [] for asr_model in transcripts_by_asr},
            "failures": {},
            "detectors_used": [],
        }

    venv_dir = ensure_venv(_PII_VENV, _PII_REQUIREMENTS, python_version=_PII_PYTHON)
    python = venv_python(venv_dir)

    labels = gliner_labels if gliner_labels is not None else list(_DEFAULT_GLINER_LABELS)
    entities = presidio_entities if presidio_entities is not None else list(_PRESIDIO_PII_ENTITIES)

    input_json = json.dumps(
        {
            "transcripts": transcripts_by_asr,
            "detectors": detectors_resolved,
            "presidio_entities": entities,
            "presidio_score_threshold": float(presidio_score_threshold),
            "gliner_model": gliner_model,
            "gliner_labels": labels,
            "gliner_threshold": float(gliner_threshold),
            "gliner_label_map": _GLINER_TO_PRESIDIO_CATEGORY,
        }
    )

    env = _clean_subprocess_env()
    result = subprocess.run(
        [python, "-c", _PII_WORKER_SCRIPT],
        input=input_json,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    output = parse_subprocess_result(result, "PII subprocess")
    return {
        "spans_by_asr": output.get("spans_by_asr", {}),
        "failures": output.get("failures", {}),
        "detectors_used": output.get("detectors_used", []),
    }
