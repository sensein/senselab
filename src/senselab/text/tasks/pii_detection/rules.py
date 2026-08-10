"""Rule-based PII detection cascade — the third ``rules`` detector.

Ported from PR #542's ``pii_compliance_pipeline.py`` (the compliance-pipeline lines are named
in the porting task's brief; only the PII-detection engines and their precision guards came
across — task compliance, the local-LLM judge, and the report writers are a different tool and
stayed behind). Docstrings and the reasoning inside them are preserved verbatim wherever
possible because the thresholds here (a Zipf cutoff, a confidence floor, a token window) encode
judgements made against real annotated data, not defaults picked by inspection.

Rigidity spectrum
------------------
Every category sits on a "rigid designation" spectrum (Kripke): does this expression, on its
own, pick out one unique individual? ``STRONG_RIGID`` (name, id number, contact, url) does.
``CONTEXTUAL_RIGID`` (org, specific place, specific date) does with supporting context.
``WEAK_RIGID`` (age, time, partial date, profession, generic place) rarely does alone but
combines with other weak signals to re-identify someone — which is what ``combinatorial_scan``
below detects. ``CATEGORY_WEIGHTS`` encodes this spectrum numerically for that scoring.

Precision guards
-----------------
The value of this cascade over a bare regex/NER pass is precision: a common surname-shaped word
("Will", "May", "Grant") is the classic NER false positive, and a digit-free token tagged
"device identifier" is the classic hallucination on ASR'd non-speech (a cough, a filler sound).
``postprocess_entities``, ``_valid_structured_identifier``, and the Zipf-frequency name hard-gate
in ``_name_hard_gate_eligible`` exist specifically to stop those from being reported as PII.
Weakening any of them to simplify this port would defeat the reason this detector is worth
having: a false positive is what makes a PII tool unusable, more than a false negative is.

Adaptations from the source
-----------------------------
1. ``PRECISION_MODE`` was a module-level mutable flag in the source. It is a keyword parameter
   of :func:`postprocess_entities` here instead — a global posture flag is untestable under
   parallel test execution, and was the mechanism behind two of PR #542's review findings
   (structured-identifier format validation and cross-engine corroboration both silently
   switched off when a caller flipped the flag for something unrelated).
2. ``wordfreq`` is optional and lives only in the ``pii-detection`` subprocess venv, never on
   the host — see ``_WORDFREQ_IMPORT`` / :func:`_zipf` below for why a missing import must
   surface as ``None``, not ``0.0``.
3. Gazetteer loading (:func:`load_name_gazetteer`, :func:`load_place_gazetteer`) never raises on
   a network-unavailable NLTK download or a missing ``pycountry``; a failure logs a warning and
   disables just that gazetteer rather than the whole cascade.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rigidity spectrum (proposal Section 2.2 / 5) + canonical guideline labels
# ---------------------------------------------------------------------------
STRONG_RIGID: set[str] = {"NAME", "IDNUM", "CONTACT", "URL"}
CONTEXTUAL_RIGID: set[str] = {"ORG", "LOC_SPECIFIC", "DATE_SPECIFIC"}
WEAK_RIGID: set[str] = {"AGE", "TIME", "DATE_PARTIAL", "PROFESSION", "LOC_GENERIC"}
MISC: set[str] = {"MISC"}

CATEGORY_WEIGHTS: dict[str, float] = {
    **{c: 1.0 for c in STRONG_RIGID},
    **{c: 0.6 for c in CONTEXTUAL_RIGID},
    **{c: 0.25 for c in WEAK_RIGID},
    "MISC": 0.3,
    "COMBINATORIAL": 0.6,
}

# Map the tool's fine-grained categories onto the annotation-guidelines label
# set (AGE, NAME, DATE, TIME, ORG, IDNUM, LOC, PROFESSION, CONTACT, URL, MISC)
# so output is compatible with the label-studio config in the guidelines.
CANONICAL_LABEL: dict[str, str] = {
    "NAME": "NAME",
    "IDNUM": "IDNUM",
    "CONTACT": "CONTACT",
    "URL": "URL",
    "ORG": "ORG",
    "LOC_SPECIFIC": "LOC",
    "LOC_GENERIC": "LOC",
    "DATE_SPECIFIC": "DATE",
    "DATE_PARTIAL": "DATE",
    "AGE": "AGE",
    "TIME": "TIME",
    "PROFESSION": "PROFESSION",
    "MISC": "MISC",
    "COMBINATORIAL": "MISC",
}


def rigidity_tier(category: str) -> str:
    """Classify a category into its position on the rigid-designation spectrum."""
    if category in STRONG_RIGID:
        return "strongly_rigid"
    if category in CONTEXTUAL_RIGID:
        return "contextually_rigid"
    if category in WEAK_RIGID:
        return "weakly_rigid"
    if category == "COMBINATORIAL":
        return "combinatorial"
    return "misc"


def _entity(
    start: int,
    end: int,
    category: str,
    confidence: float,
    method: str,
    **extra: Any,  # noqa: ANN401 -- entity payloads carry heterogeneous extras (age_value, contributing, ...)
) -> dict[str, Any]:
    """Build one raw detection dict shared by every scan function below."""
    e = {"start": start, "end": end, "category": category, "confidence": confidence, "method": method}
    e.update(extra)
    return e


# ---------------------------------------------------------------------------
# Precision guards -- reduce false positives without dropping true positives.
# ---------------------------------------------------------------------------
# High-precision methods whose hits we trust even below MIN_ENGINE_CONFIDENCE
# (they matched an explicit pattern/list, not a soft model score).
_PATTERN_METHODS: set[str] = {"regex", "honorific", "gazetteer", "keyword", "demographic"}
_DIGIT_RE = re.compile(r"\d")

# ---- Precision thresholds (PR #542, tuned against annotated study data) ----
MIN_ENGINE_CONFIDENCE = 0.5  # drop low-confidence model detections (pattern hits exempt)
STRONG_PII_HARD_GATE_CONFIDENCE = 0.85  # a lone strong-PII guess this confident may hard-fail
# A single-token NAME that is also a common English word (zipf >= this) with no real name
# signal is treated as an NER false positive: recorded, but it does not flag the file.
# Lower => stricter (fewer name flags).
NAME_COMMON_WORD_ZIPF = 3.9
RARE_WORD_ZIPF_THRESHOLD = 2.7
RARE_ROLE_QUALIFIER_ZIPF_MAX = 3.9  # soft-qualifier role fires only if it contains an uncommon word
RARE_ROLE_INTRO_ZIPF_MAX = 3.6  # 'I am a <role>' fires only for uncommon occupations
COMBINATORIAL_WINDOW_TOKENS = 15
COMBINATORIAL_THRESHOLD = 0.5
# An age STRICTLY GREATER than this is flagged for review. HIPAA Safe Harbor treats all
# ages over 89 as identifiers, so use 89 for strict HIPAA; 90 matches "any age over 90".
AGE_REVIEW_OVER_YEARS = 90

# Which canonical labels may hard-gate a finding as confirmed direct PII. Only CONFIRMED DIRECT
# identifiers (SSN / MRN / email / phone / URL) are serious enough; a spoken NAME is deliberately
# excluded and routes to review instead (see [HARD GATE] in the source pipeline's notes).
HARD_GATE_PII_LABELS: set[str] = {"IDNUM", "CONTACT", "URL"}
# A lone strong-PII guess never auto-fails a file without corroboration -- deliberately
# independent of any precision/recall posture; see the hard-gate note in merge_pii.
HARD_GATE_REQUIRE_CORROBORATION = True
# Every detected personal name is review-worthy, whoever it belongs to and however weak the
# signal (no auto-fail -- NAME stays out of HARD_GATE_PII_LABELS regardless).
FLAG_ALL_NAMES = True
# High-recall screening widens what reaches review, never what reaches confirmed. Off by
# default: this port ships the precision-first posture; a caller wanting the recall posture
# can pass precision_mode=False to postprocess_entities, which is the one knob PR #542's
# review found safe to expose (see the module docstring's "Adaptations" section).
HIGH_RECALL = False
RECALL_FLAG_ALL_PII = False
# Combinatorial risk needs >=2 DIFFERENT weak/misc categories in the window, not two hits of
# the same weak signal (e.g. two profession mentions), which don't compound identification risk.
COMBINATORIAL_REQUIRE_CATEGORY_DIVERSITY = True
# Self-disclosed gender / race-ethnicity detection (demographic_scan). On by default: these are
# sensitive, protected attributes that, combined with other detail, re-identify a participant.
DEMOGRAPHIC_PII = True

# Holidays / named time-periods. Per the annotation guidelines these are DATES,
# NOT places/orgs/names -- so if any engine tags one as NAME/LOC/ORG we reclassify
# it to a (weak) DATE rather than letting it read as a strong/contextual identifier.
# This is what stops "Easter" surfacing as a LOCATION or ORGANIZATION.
HOLIDAYS: set[str] = {
    "christmas", "christmas eve", "easter", "good friday", "thanksgiving",
    "halloween", "new year", "new year's", "new years", "new year's eve",
    "hanukkah", "chanukah", "passover", "yom kippur", "rosh hashanah",
    "ramadan", "eid", "diwali", "holi", "lent", "advent", "boxing day",
    "independence day", "memorial day", "labor day", "labour day",
    "veterans day", "columbus day", "juneteenth", "mardi gras", "carnival",
    "valentine's day", "valentines day", "st patrick's day", "cinco de mayo",
}  # fmt: skip


# Cues that a nearby capitalized token really is a person's name (high precision
# for ASR speech: "my name is ...", "I'm ...", "this is ...", honorifics, etc.).
NAME_CONTEXT_RE = re.compile(
    r"\b(?:name\s+is|name'?s|my\s+name|named|call(?:ed)?\s+me|i\s*am|i'?m|"
    r"this\s+is|it'?s|mr|mrs|ms|miss|dr|prof(?:essor)?|sir|madam|dame|rev|"
    r"father|speaking\s+with|speaking\s+to|interview(?:ing)?\s+with|"
    r"patient|participant|meet)\b",
    re.IGNORECASE,
)


# ``wordfreq`` ships only in the ``pii-detection`` subprocess venv, never on the host (it is a
# genuinely new dependency this port introduces -- see subprocess_backend.py's requirements
# comment). The import is attempted here, at module level, rather than deferred inside _zipf,
# specifically so a test can `monkeypatch.setattr(rules, "_WORDFREQ_IMPORT", None)` to exercise
# the "unavailable" branch without fighting Python's import cache.
try:  # wordfreq lives in the pii-detection venv, not the host
    from wordfreq import zipf_frequency as _WORDFREQ_IMPORT
except ImportError:
    _WORDFREQ_IMPORT = None


def _zipf(word: str) -> Optional[float]:
    """Zipf word frequency, or None when wordfreq is unavailable.

    None is NOT 0.0. 0.0 means "measured, maximally rare", and every caller reads
    rarity as evidence FOR a PII hit -- so returning 0.0 on a missing dependency
    silently INVERTS the precision guards instead of relaxing them: a common-word
    NAME false positive becomes hard-gate eligible, and the soft rare-role qualifier
    fires on every phrase. Callers must treat None as "unknown" and take their
    precision-safe branch.
    """
    if _WORDFREQ_IMPORT is None:
        return None
    return _WORDFREQ_IMPORT(word.lower(), "en")


def _wordfreq_available() -> bool:
    """Whether word-frequency data is installed at all (wordfreq ships in the pii-detection venv).

    Guards that need frequency evidence must not fire without it.
    """
    return _zipf("the") is not None


def _name_hard_gate_eligible(
    span_text: str,
    start: int,
    source_text: str,
    methods: set[str],
    engines: set[str],
    score: float,
) -> bool:
    """A NAME hard-fails a file only with a real name signal.

    A lone single-token common word tagged as a name by an NER model (Will / May / Grant / Mark
    ...) is the classic false positive -- it drops to needs_review instead of failing.
    """
    tokens = span_text.split()
    multitoken = len(tokens) >= 2
    precise = bool(methods & {"honorific", "gazetteer"})
    corroborated = len(engines) >= 2
    preceding = source_text[max(0, start - 30) : start]
    has_context = bool(NAME_CONTEXT_RE.search(preceding))
    # Unknown frequency (no wordfreq) is treated as "common", i.e. the precision-safe
    # side: a lone single token with no other name signal stays out of the hard gate.
    z = _zipf(span_text)
    common = (not multitoken) and (z is None or z >= NAME_COMMON_WORD_ZIPF)
    if common and not (has_context or precise or multitoken):
        return False
    return multitoken or has_context or precise or corroborated or score >= 0.9


def _valid_structured_identifier(text: str, category: str) -> bool:
    """Structured identifiers must actually look structured.

    A word with no digits / '@' / URL shape is almost always a model hallucination -- e.g. an
    ASR'd cough token flagged as a 'device identifier'. Real SSNs, MRNs, phone numbers, emails,
    and URLs always satisfy these.
    """
    t = (text or "").strip()
    if not t:
        return False
    if category == "URL":
        return bool(re.search(r"(https?://|www\.|\.[a-z]{2,})", t, re.IGNORECASE))
    if category == "CONTACT":  # email / phone / fax / IP
        return ("@" in t) or len(_DIGIT_RE.findall(t)) >= 7 or bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", t))
    if category == "IDNUM":  # ids carry digits
        return _DIGIT_RE.search(t) is not None
    return True


def postprocess_entities(
    entities: list[dict[str, Any]], source_text: str, *, precision_mode: bool = True
) -> list[dict[str, Any]]:
    """Apply the precision guards to a flat list of engine detections.

    Structured-identifier FORMAT VALIDATION is a correctness check, not a precision/recall
    tradeoff: a "device identifier" containing no digits is wrong in either posture, and letting
    it through in high-recall mode promotes a known hallucination class straight to the
    confirmed/hard-gate tier. It therefore runs always. ``precision_mode`` governs only the
    confidence-threshold guard below it -- a keyword parameter rather than the source's
    module-level ``PRECISION_MODE`` flag, since a mutable global posture is untestable under
    parallel execution and was the mechanism behind two of PR #542's review findings.

    Args:
        entities: Raw detections from the scan functions, each a dict with at least
            ``start``, ``end``, ``category``, ``confidence``, and ``method``.
        source_text: The full text the entity spans index into.
        precision_mode: When ``True`` (default), soft (non-pattern-method) detections below
            ``MIN_ENGINE_CONFIDENCE`` are dropped. When ``False``, they survive to be flagged
            downstream instead -- the high-recall posture. Format validation and holiday
            reclassification are unaffected by this flag either way.

    Returns:
        The filtered/reclassified entity list, in the same dict shape as the input.
    """
    out = []
    for e in entities:
        span_text = source_text[e["start"] : e["end"]]
        cat = e["category"]
        # 1) Holiday/named-period reclassification (guidelines: these are DATES).
        low = span_text.lower().strip().strip(".,'’")
        if low in HOLIDAYS and cat in ("NAME", "ORG", "LOC_SPECIFIC", "LOC_GENERIC", "MISC"):
            e = dict(
                e,
                category="DATE_PARTIAL",
                confidence=min(e["confidence"], 0.6),
                method=e["method"] + "+holiday-reclass",
            )
            cat = "DATE_PARTIAL"
        # 2) Structured-identifier format validation (kills word-only "identifiers").
        if cat in ("IDNUM", "CONTACT", "URL") and not _valid_structured_identifier(span_text, cat):
            continue
        # 3) Low-confidence soft detections dropped (pattern/list hits are exempt).
        #    This one IS the precision/recall knob, so high-recall keeps them.
        if precision_mode:
            method_root = e["method"].split(":")[0].split("+")[0]
            if e["confidence"] < MIN_ENGINE_CONFIDENCE and method_root not in _PATTERN_METHODS:
                continue
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Gazetteers (Stage 1 of the rigidity cascade)
# ---------------------------------------------------------------------------
def load_name_gazetteer() -> set[str]:
    """~8000 common given names (NLTK 'names' corpus), fetched lazily on first use.

    Never raises: a blocked download (no network, a sandboxed worker) logs a warning and
    disables gazetteer-based NAME detection rather than crashing the whole cascade over one
    optional corpus -- the same "degrade honestly, don't crash" contract as ``_zipf``.
    """
    try:
        import nltk
        from nltk.corpus import names

        try:
            names.words()
        except LookupError:
            nltk.download("names", quiet=True)
            from nltk.corpus import names  # noqa: PLC0414 -- reimport after download, matches source
        return {n.lower() for n in names.words()}
    except Exception as exc:  # noqa: BLE001 -- any failure disables the gazetteer, not the cascade
        logger.warning(
            "PII name gazetteer unavailable (%s: %s); gazetteer-based NAME detection disabled.",
            type(exc).__name__,
            exc,
        )
        return set()


US_STATES: set[str] = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}  # fmt: skip
MAJOR_CITIES: set[str] = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "boston", "seattle",
    "denver", "washington", "nashville", "detroit", "portland", "las vegas",
    "baltimore", "atlanta", "miami", "toronto", "vancouver", "montreal", "london",
    "paris", "berlin", "madrid", "rome", "amsterdam", "dublin", "cairo", "lagos",
    "nairobi", "johannesburg", "mumbai", "delhi", "bangalore", "beijing", "shanghai",
    "hong kong", "tokyo", "seoul", "singapore", "bangkok", "manila", "jakarta",
    "sydney", "melbourne", "auckland", "mexico city", "sao paulo", "buenos aires",
    "lima", "bogota", "moscow", "istanbul", "damascus", "baghdad", "tehran",
}  # fmt: skip


def load_place_gazetteer() -> set[str]:
    """Country names (via ``pycountry``, if installed) plus the US-states / major-cities sets.

    A missing ``pycountry`` (it is not in the pii-detection venv's requirements -- only the
    host's ``nlp`` extra) narrows this to the built-in state/city sets rather than raising;
    the calling scans already treat an empty/partial place set as "no match", not an error.
    """
    places: set[str] = set()
    try:
        import pycountry

        places = {c.name.lower() for c in pycountry.countries}
    except Exception as exc:  # noqa: BLE001 -- a missing/broken pycountry narrows, doesn't crash
        logger.warning(
            "pycountry unavailable (%s: %s); place gazetteer limited to US states and major cities.",
            type(exc).__name__,
            exc,
        )
    return places | US_STATES | MAJOR_CITIES


ORG_SUFFIX_RE = re.compile(
    r"\b([A-Z][\w&.,'-]*(?:\s+[A-Z][\w&.,'-]*){0,4}\s+"
    r"(?:Hospital|University|Clinic|Center|Centre|Foundation|Institute|"
    r"Inc\.?|LLC|Corp\.?|Ltd\.?|Co\.))\b"
)
CALENDAR_WORDS: set[str] = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "spring", "summer", "autumn", "fall", "winter",
}  # fmt: skip


def gazetteer_scan(text: str, name_set: set[str], place_set: set[str]) -> list[dict[str, Any]]:
    """Match capitalized phrases against the name / place gazetteers, and org-suffix patterns."""
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2}\b", text):
        phrase = m.group(0)
        words = phrase.split()
        if phrase.lower() in place_set:
            entities.append(_entity(m.start(), m.end(), "LOC_SPECIFIC", 0.9, "gazetteer"))
            continue
        if len(words) >= 2 and any(w.lower() in name_set and w.lower() not in CALENDAR_WORDS for w in words):
            entities.append(_entity(m.start(), m.end(), "NAME", 0.85, "gazetteer"))
    for m in ORG_SUFFIX_RE.finditer(text):
        entities.append(_entity(m.start(1), m.end(1), "ORG", 0.8, "gazetteer"))
    return entities


# ---------------------------------------------------------------------------
# Stage 1 regex over structured identifiers
# ---------------------------------------------------------------------------
REGEX_PATTERNS: list[tuple[str, float, re.Pattern[str]]] = [
    ("CONTACT", 0.95, re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),  # email
    ("CONTACT", 0.90, re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),  # phone
    ("IDNUM", 0.95, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),  # SSN
    ("URL", 0.95, re.compile(r"\b(?:https?://|www\.)[^\s,]+", re.IGNORECASE)),
    ("CONTACT", 0.90, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),  # IPv4
    (
        "IDNUM",
        0.85,
        re.compile(
            r"\b(?:MRN|medical record(?:\s+number)?|account number|patient id|"
            r"certificate number|license number|policy number|member id|health plan number)"
            r"\s*[:#]?\s*([A-Za-z0-9-]{4,})\b",
            re.IGNORECASE,
        ),
    ),
    ("DATE_SPECIFIC", 0.90, re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("DATE_SPECIFIC", 0.90, re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("AGE", 0.85, re.compile(r"\b\d{1,3}[\s-]year[\s-]old\b", re.IGNORECASE)),
]
HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Fr|Sr|Sir|Dame)\.?\s+([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?)"
)
PROFESSION_KEYWORDS: set[str] = {
    "doctor", "nurse", "surgeon", "physician", "therapist", "psychologist",
    "psychiatrist", "teacher", "professor", "engineer", "lawyer", "attorney",
    "technician", "officer", "manager", "director", "researcher", "scientist",
    "student", "resident", "pharmacist", "counselor", "social worker", "paramedic",
    "lieutenant", "captain", "sergeant", "president",
}  # fmt: skip


def regex_scan(text: str) -> list[dict[str, Any]]:
    """Match every pattern in ``REGEX_PATTERNS`` and emit one entity per hit."""
    entities = []
    for category, confidence, pattern in REGEX_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span(1) if m.groups() else m.span()
            entities.append(_entity(span[0], span[1], category, confidence, "regex"))
    return entities


# ---------------------------------------------------------------------------
# Stage 2 weak signals (spaCy NER, honorific, profession, rare word)
# ---------------------------------------------------------------------------
_SPACY_LABEL_MAP: dict[str, str] = {"PERSON": "NAME", "ORG": "ORG", "TIME": "TIME"}


def ner_scan(nlp: Any, text: str, place_set: set[str]) -> list[dict[str, Any]]:  # noqa: ANN401 -- spaCy Language
    """Run a loaded spaCy pipeline and map its entity labels onto this cascade's categories."""
    entities = []
    for ent in nlp(text).ents:
        if ent.label_ in ("GPE", "LOC"):
            cat = "LOC_SPECIFIC" if ent.text.lower() in place_set else "LOC_GENERIC"
            entities.append(_entity(ent.start_char, ent.end_char, cat, 0.75, "ner"))
        elif ent.label_ == "DATE":
            cat = "DATE_SPECIFIC" if re.search(r"\b(19|20)\d{2}\b", ent.text) else "DATE_PARTIAL"
            entities.append(_entity(ent.start_char, ent.end_char, cat, 0.7, "ner"))
        elif ent.label_ in _SPACY_LABEL_MAP:
            entities.append(_entity(ent.start_char, ent.end_char, _SPACY_LABEL_MAP[ent.label_], 0.75, "ner"))
    return entities


def honorific_scan(text: str) -> list[dict[str, Any]]:
    """Match ``Mr./Dr./Prof. <Name>`` patterns -- a high-precision pattern method."""
    return [_entity(m.start(1), m.end(1), "NAME", 0.7, "honorific") for m in HONORIFIC_RE.finditer(text)]


_MULTIWORD_PROFESSIONS = sorted((p for p in PROFESSION_KEYWORDS if " " in p), key=len, reverse=True)


def profession_scan(text: str) -> list[dict[str, Any]]:
    """Flag profession keywords.

    Checks every single word individually (so an adjacent word never swallows the keyword) plus
    known multi-word professions.
    """
    entities = []
    taken = []  # spans already claimed by a multi-word match
    for phrase in _MULTIWORD_PROFESSIONS:
        for m in re.finditer(rf"\b{re.escape(phrase)}\b", text, re.IGNORECASE):
            entities.append(_entity(m.start(), m.end(), "PROFESSION", 0.6, "keyword"))
            taken.append((m.start(), m.end()))
    for m in re.finditer(r"\b[a-zA-Z]+\b", text):
        if m.group(0).lower() in PROFESSION_KEYWORDS:
            if any(s <= m.start() < e for s, e in taken):
                continue
            entities.append(_entity(m.start(), m.end(), "PROFESSION", 0.6, "keyword"))
    return entities


def rareword_scan(text: str, zipf_threshold: float) -> list[dict[str, Any]]:
    """Flag capitalized, non-sentence-initial words below a rarity threshold as weak MISC signals.

    Routed through :func:`_zipf` (rather than importing ``wordfreq`` directly, as the source
    did at this call site) so the module-level ``_WORDFREQ_IMPORT`` indirection is the single
    place "is wordfreq available" is decided -- a second independent import path here would
    silently bypass the monkeypatch guard :func:`_zipf`'s docstring exists to protect.
    """
    if not _wordfreq_available():
        return []
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z'-]{3,}\b", text):
        word = m.group(0)
        preceding = text[: m.start()].rstrip()
        if preceding == "" or preceding[-1] in ".!?":  # skip sentence-initial caps
            continue
        z = _zipf(word)
        if z is not None and z <= zipf_threshold:
            entities.append(_entity(m.start(), m.end(), "MISC", 0.5, "rareword"))
    return entities


# A role/activity phrase of 1-2 words that does not run past a stopword (so
# "figure skater and I ..." captures just "figure skater").
_ROLE_STOP = (
    r"(?:and|or|but|who|that|which|because|so|when|while|the|a|an|to|of|"
    r"in|on|for|is|was|are|were|be|i|my|me|he|she|they|we|you|it|this)"
)
_ROLE_WORDS = r"[a-z][a-z'-]+(?:\s+(?!" + _ROLE_STOP + r"\b)[a-z][a-z'-]+){0,1}"
# STRONG qualifiers are identifying on their own ("olympic <x>"). SOFT qualifiers
# ("professional <x>") flag only when the role phrase contains an uncommon word --
# so "professional figure skater" fires but "professional development" does not.
_STRONG_QUALIFIER_RE = re.compile(
    r"\b(olympic|paralympic|championship|champion|world[-\s]?class|elite|decorated|"
    r"renowned|award[-\s]?winning)\s+(" + _ROLE_WORDS + r")\b",
    re.IGNORECASE,
)
_SOFT_QUALIFIER_RE = re.compile(
    r"\b(professional|semi[-\s]?professional|competitive|national|international|"
    r"former|retired|amateur)\s+(" + _ROLE_WORDS + r")\b",
    re.IGNORECASE,
)
# A self-described occupation ("I am a <x>", "I work as a <x>").
_ROLE_INTRO_RE = re.compile(
    r"\b(?:i'm|i\s+am|i\s+was|i\s+used\s+to\s+be|worked\s+as|work\s+as|employed\s+as)\s+"
    r"(?:an?\s+|the\s+)?(" + _ROLE_WORDS + r")\b",
    re.IGNORECASE,
)


def _phrase_min_zipf(phrase: str) -> Optional[float]:
    """Minimum word-frequency across a role phrase (a rare word anywhere signals specificity).

    Returns None if no word is known to wordfreq.
    """
    zs = [z for z in (_zipf(t) for t in re.findall(r"[a-z\'-]+", phrase.lower())) if z is not None and z > 0.0]
    return min(zs) if zs else None


def rare_role_scan(text: str) -> list[dict[str, Any]]:
    """Nuanced quasi-identifiers: unusually specific roles/activities/statuses.

    These can re-identify a person on their own (annotation-guidelines MISC). Example that
    motivated this: a participant mentioning being a "professional figure skater". These are
    surfaced to REVIEW (never an auto-fail); the LLM and a human reviewer make the final call.
    Kept high-precision to avoid flagging common jobs.
    """
    ents = []
    for m in _STRONG_QUALIFIER_RE.finditer(text):
        ents.append(_entity(m.start(), m.end(), "MISC", 0.6, "rare_role"))
    # The soft qualifier ("professional <x>") is only discriminating WITH frequency
    # data -- it is what separates "professional figure skater" from "professional
    # development". Without wordfreq it would fire on every match, so it is skipped:
    # a missing optional dependency must not manufacture flags.
    have_freq = _wordfreq_available()
    for m in _SOFT_QUALIFIER_RE.finditer(text):
        if not have_freq:
            continue
        z = _phrase_min_zipf(m.group(2))
        if z is None or z <= RARE_ROLE_QUALIFIER_ZIPF_MAX:  # skip 'professional development'
            ents.append(_entity(m.start(), m.end(), "MISC", 0.55, "rare_role"))
    for m in _ROLE_INTRO_RE.finditer(text):
        z = _phrase_min_zipf(m.group(1))
        if z is not None and z <= RARE_ROLE_INTRO_ZIPF_MAX:  # only uncommon occupations
            ents.append(_entity(m.start(1), m.end(1), "MISC", 0.5, "rare_role"))
    return ents


# ---------------------------------------------------------------------------
# Demographic quasi-identifiers: self-disclosed gender & race/ethnicity.
# High precision by design -- everything requires a SELF/person anchor so bare
# nouns, pronouns and incidental words ("black coffee", "Chinese food") never fire.
# All hits are MISC with method "demographic:<race|gender>" -> merge_pii marks them
# review_worthy (like rare_role); they never hard-FAIL a file.
# ---------------------------------------------------------------------------
_PERSON_NOUN = r"man|woman|male|female|guy|girl|boy|lady|gentleman|person|individual|folk|folks|kid|child"
# Race / ethnicity terms distinctive enough to flag when self/person-anchored. Bare
# nationalities (Chinese, Mexican, ...) are intentionally excluded -- they fire far too
# often on food/places -- and are caught only via the explicit "my ethnicity is <X>".
_RACE_TERMS = (
    r"black|white|caucasian|african[-\s]?american|afro[-\s]?caribbean|"
    r"asian(?:[-\s]?american)?|south[-\s]?asian|east[-\s]?asian|southeast[-\s]?asian|desi|"
    r"hispanic|latin[oax]|latine|"
    r"native[-\s]?american|american[-\s]?indian|indigenous|alaska[-\s]?native|"
    r"pacific[-\s]?islander|native[-\s]?hawaiian|"
    r"middle[-\s]?eastern|arab|"
    r"biracial|multiracial|mixed[-\s]?race|brown"
)
# Gender-identity terms (sensitive/protected); distinctive enough to flag on sight.
_GENDER_IDENTITY_TERMS = (
    r"transgender|transsexual|trans[-\s]?(?:man|woman|male|female|masc\w*|fem\w*)|"
    r"cis[-\s]?gender(?:ed)?|cisgender(?:ed)?|"
    r"non[-\s]?binary|nonbinary|gender[-\s]?queer|genderqueer|gender[-\s]?fluid|genderfluid|"
    r"agender|bigender|two[-\s]?spirit|inter[-\s]?sex|intersex|"
    r"gender[-\s]?non[-\s]?conforming|gender[-\s]?dysphoria|gender[-\s]?identity|"
    r"mtf|ftm|afab|amab"
)
# Binary sex/gender words -- flagged ONLY when self-disclosed (a first/third-person
# copular anchor or "as a"/"my gender is"), never as a bare noun or pronoun.
_BINARY_GENDER = r"man|woman|male|female|guy|girl|boy|lady|gentleman|nonbinary|non[-\s]?binary"
_ANY_GENDER = r"(?:" + _GENDER_IDENTITY_TERMS + r"|" + _BINARY_GENDER + r")"
# In the strongest self-ID contexts only ("I am ___", "they identify as ___") also accept
# bare trans/cis (lookahead keeps it out of trans-atlantic / cis-regulatory).
_ANY_GENDER_SELF = r"(?:" + _GENDER_IDENTITY_TERMS + r"|" + _BINARY_GENDER + r"|(?:trans|cis)(?![-\w]))"
# Subject pronoun + copula ("I'm", "I am", "she is", "they were", ...).
_SUBJ = r"(?:i|we|he|she|they)(?:\s*'m|\s*'re|\s*'s|\s+am|\s+are|\s+is|\s+was|\s+were)"

_RACE_SELF_RE = re.compile(
    r"\b" + _SUBJ + r"\s+(?:an?\s+)?((?:" + _RACE_TERMS + r")(?:\s+(?:" + _PERSON_NOUN + r"))?)\b", re.IGNORECASE
)
_RACE_ADJ_RE = re.compile(r"\b((?:" + _RACE_TERMS + r")\s+(?:" + _PERSON_NOUN + r"))\b", re.IGNORECASE)
_RACE_EXPLICIT_RE = re.compile(
    r"\bmy\s+(?:race|ethnicity|ethnic\s+background|racial\s+background|heritage|ancestry)\s+"
    r"(?:is|are|was|were|being|:)?\s*([a-z][\w'-]+(?:\s+[a-z][\w'-]+)?)",
    re.IGNORECASE,
)
_POC_RE = re.compile(r"\b((?:person|people|woman|man|women|men)\s+of\s+colou?r)\b", re.IGNORECASE)
_GENDER_IDENTITY_RE = re.compile(r"\b(" + _GENDER_IDENTITY_TERMS + r")\b", re.IGNORECASE)
_GENDER_SELF_RE = re.compile(
    r"\b(?:" + _SUBJ + r"|(?:i|we|he|she|they)\s+identif(?:y|ies)\s+as)\s+"
    r"(?:an?\s+)?(" + _ANY_GENDER_SELF + r")\b",
    re.IGNORECASE,
)
_GENDER_ASA_RE = re.compile(r"\bas\s+an?\s+(" + _ANY_GENDER + r"(?:\s+(?:" + _PERSON_NOUN + r"))?)\b", re.IGNORECASE)
_GENDER_EXPLICIT_RE = re.compile(
    r"\bmy\s+(?:gender(?:\s+identity)?|sex|pronouns?)\s+(?:is|are|was|were|being|:)?\s*([a-z][\w'-]+)",
    re.IGNORECASE,
)

_RACE_RES = (_RACE_SELF_RE, _RACE_ADJ_RE, _RACE_EXPLICIT_RE, _POC_RE)
_GENDER_RES = (_GENDER_IDENTITY_RE, _GENDER_SELF_RE, _GENDER_ASA_RE, _GENDER_EXPLICIT_RE)


def demographic_scan(text: str) -> list[dict[str, Any]]:
    """Nuanced demographic quasi-identifiers: SELF-disclosed race/ethnicity and gender/gender-identity.

    These are sensitive, protected attributes that (with other detail) re-identify a
    participant, so every hit is surfaced to REVIEW as MISC (never a hard FAIL). Precision
    comes from requiring a self/person anchor, so bare nouns, pronouns and incidental words
    ('black coffee', 'Chinese food') do not fire.
    """
    if not DEMOGRAPHIC_PII:
        return []
    ents: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def _add(a: int, b: int, subtype: str, conf: float) -> None:
        if b <= a or (a, b) in seen:
            return
        seen.add((a, b))
        ents.append(_entity(a, b, "MISC", conf, "demographic:" + subtype))

    for rx in _RACE_RES:
        strong = rx in (_RACE_EXPLICIT_RE,)
        for m in rx.finditer(text):
            _add(m.start(1), m.end(1), "race", 0.6 if strong else 0.55)
    for rx in _GENDER_RES:
        strong = rx in (_GENDER_IDENTITY_RE, _GENDER_EXPLICIT_RE)
        for m in rx.finditer(text):
            _add(m.start(1), m.end(1), "gender", 0.6 if strong else 0.55)
    return ents


# ---------------------------------------------------------------------------
# Very high age (> AGE_REVIEW_OVER_YEARS) -- an always-flag quasi-identifier.
# HIPAA Safe Harbor treats ages over 89 as identifiers: at the top of the age
# distribution the population thins out so much that the age alone can single a
# participant out. The general AGE regex above only catches "95-year-old" and is
# too weak to flag a file on its own, so this dedicated scan handles the phrasings
# ASR actually produces -- "95 years old", "aged 102", "my age is 94", spelled-out
# numbers ("ninety-five", "a hundred and two"), and "in her nineties" -- and marks
# the hit `age_over_threshold` so merge_pii always routes it to REVIEW.
# ---------------------------------------------------------------------------
_AGE_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_AGE_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}  # fmt: skip
_UNIT_RE = "|".join(_AGE_UNITS)
_TENS_RE = "|".join(_AGE_TENS)
# A spelled-out number up to ~199: "a/one hundred [and] [tens] [units]" | "ninety-five" | "ninety"
_AGE_WORD_NUM = (
    rf"(?:(?:one|a)\s+hundred(?:\s+and)?(?:\s+(?:{_TENS_RE}))?(?:[-\s](?:{_UNIT_RE}))?"
    rf"|(?:{_TENS_RE})(?:[-\s](?:{_UNIT_RE}))?)"
)
_AGE_VALUE = rf"(\d{{1,3}}|{_AGE_WORD_NUM})"
# Each pattern anchors the number to an explicit AGE meaning, so bare numbers
# ("I'm 100 percent sure", "100 dollars") never register as an age.
_AGE_CONTEXT_RES = (
    re.compile(rf"\b{_AGE_VALUE}\s*-?\s*years?[\s-]*old\b", re.IGNORECASE),
    re.compile(rf"\b{_AGE_VALUE}\s*-?\s*y[\s.]?o\.?\b", re.IGNORECASE),
    re.compile(rf"\bage[d]?\s*(?:is|of|:)?\s*{_AGE_VALUE}\b", re.IGNORECASE),
    re.compile(rf"\b(?:turned|turning|turns)\s+{_AGE_VALUE}\b", re.IGNORECASE),
)
# Decade/status phrasings that imply >90 without naming a number.
_AGE_OVER_PHRASE_RE = re.compile(
    r"\b(?:in\s+(?:his|her|their|my|the)\s+(?:nineties|hundreds)"
    r"|centenarian|centenarians|nonagenarian|nonagenarians"
    r"|over\s+(?:ninety|90|the\s+age\s+of\s+(?:ninety|90)))\b",
    re.IGNORECASE,
)


def _age_word_to_int(phrase: str) -> Optional[int]:
    """Convert a spelled-out age ('ninety-five', 'a hundred and two') to an int, or None.

    Digits are handled by the caller.
    """
    words = re.findall(r"[a-z]+", phrase.lower())
    if not words:
        return None
    total = 0
    if "hundred" in words:
        total = 100
        words = words[words.index("hundred") + 1 :]
    for w in words:
        if w in _AGE_TENS:
            total += _AGE_TENS[w]
        elif w in _AGE_UNITS:
            total += _AGE_UNITS[w]
        elif w in ("and", "a", "one") and total >= 100:
            continue
    return total or None


def age_scan(text: str, over_years: Optional[int] = None) -> list[dict[str, Any]]:
    """Flag ages STRICTLY GREATER than `over_years` (default AGE_REVIEW_OVER_YEARS).

    Only the flaggable (very high) ages are emitted, so ordinary ages keep behaving exactly
    as before. Every hit carries age_over_threshold=True, which makes merge_pii mark it
    review_worthy regardless of the precision guards.
    """
    limit = AGE_REVIEW_OVER_YEARS if over_years is None else over_years
    ents: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def _add(a: int, b: int, conf: float, value: Optional[int] = None) -> None:
        if b <= a or (a, b) in seen:
            return
        seen.add((a, b))
        ents.append(_entity(a, b, "AGE", conf, "regex:age_over", age_over_threshold=True, age_value=value))

    for rx in _AGE_CONTEXT_RES:
        for m in rx.finditer(text):
            raw = m.group(1)
            value = int(raw) if raw.isdigit() else _age_word_to_int(raw)
            # Ignore implausible ages (typos / non-age numbers that slipped through).
            if value is None or value > 125:
                continue
            if value > limit:
                _add(m.start(), m.end(), 0.9, value)
    for m in _AGE_OVER_PHRASE_RE.finditer(text):
        _add(m.start(), m.end(), 0.8, None)
    return ents


# ---------------------------------------------------------------------------
# Stage 3 combinatorial rigidity (sliding window)
# ---------------------------------------------------------------------------
def combinatorial_scan(
    text: str, weak_and_misc_entities: list[dict[str, Any]], window_tokens: int, threshold: float
) -> list[dict[str, Any]]:
    """Slide a token window across the text and flag clusters of >=2 distinct weak/misc categories.

    Flags when their combined, weighted confidence exceeds ``threshold`` -- individually
    non-identifying details (age, profession, generic place) that together can re-identify
    someone (the "mosaic effect").
    """
    tokens = list(re.finditer(r"\S+", text))
    if not tokens or not weak_and_misc_entities:
        return []
    flags = []
    n = len(tokens)
    for i in range(n):
        w_start = tokens[i].start()
        w_end = tokens[min(i + window_tokens, n) - 1].end()
        contributing = [e for e in weak_and_misc_entities if e["start"] >= w_start and e["end"] <= w_end]
        score = sum(CATEGORY_WEIGHTS.get(e["category"], 0.3) * e["confidence"] for e in contributing)
        diverse = len({e["category"] for e in contributing}) >= 2
        if COMBINATORIAL_REQUIRE_CATEGORY_DIVERSITY and not diverse:
            continue  # two of the same weak signal (e.g. two professions) don't combine
        if score >= threshold and len(contributing) >= 2:
            flags.append(
                {
                    "start": min(e["start"] for e in contributing),
                    "end": max(e["end"] for e in contributing),
                    "category": "COMBINATORIAL",
                    "confidence": min(1.0, score),
                    "method": "combinatorial",
                    "contributing": contributing,
                }
            )
    flags.sort(key=lambda e: e["start"])
    merged: list[dict[str, Any]] = []
    for f in flags:
        if merged and f["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], f["end"])
            merged[-1]["confidence"] = max(merged[-1]["confidence"], f["confidence"])
            ids = {id(c) for c in merged[-1]["contributing"]}
            merged[-1]["contributing"].extend(c for c in f["contributing"] if id(c) not in ids)
        else:
            merged.append(f)
    return merged


# ---------------------------------------------------------------------------
# PII merge / cross-validation across engines
# ---------------------------------------------------------------------------
def merge_pii(entities: list[dict[str, Any]], text: str) -> list[dict[str, Any]]:
    """Cluster overlapping detections from all engines.

    Agreement across distinct engines/models boosts confidence; single-engine hits are kept
    but flagged for review (poster: 'Disagreements surface to reviewers').
    """
    entities = sorted(entities, key=lambda e: e["start"])
    clusters: list[list[dict[str, Any]]] = []
    for e in entities:
        placed = False
        for c in clusters:
            if any(max(e["start"], m["start"]) < min(e["end"], m["end"]) for m in c):
                c.append(e)
                placed = True
                break
        if not placed:
            clusters.append([e])

    merged = []
    for c in clusters:
        start = min(m["start"] for m in c)
        end = max(m["end"] for m in c)
        categories = sorted({m["category"] for m in c})
        base = max(CATEGORY_WEIGHTS.get(m["category"], 0.3) * m["confidence"] for m in c)
        engines = sorted({m["method"].split(":")[0].split("+")[0] for m in c})
        methods = {m["method"].split(":")[0].split("+")[0] for m in c}
        # agreement bonus: more independent engines -> higher confidence
        agreement_bonus = min(0.15, 0.05 * (len(engines) - 1))
        best = max(c, key=lambda m: CATEGORY_WEIGHTS.get(m["category"], 0.3) * m["confidence"])
        score = round(min(1.0, base + agreement_bonus), 3)
        is_strong = best["category"] in STRONG_RIGID

        # Hard-gate eligibility: a strong-PII hit auto-FAILS a file only when it is
        # trustworthy on its own -- a high-precision pattern match (SSN/email/phone/
        # URL regex), OR corroborated by >=2 independent engines, OR very high
        # confidence, OR a name via honorific/gazetteer. A lone low-confidence model
        # guess is NOT auto-failed; it still surfaces as needs_review (recall kept).
        # Deliberately independent of the precision/recall posture: high recall widens
        # what reaches REVIEW, never what reaches CONFIRMED. Tying the two together let a
        # single uncorroborated engine guess auto-confirm as hard-gate PII in recall mode.
        if not HARD_GATE_REQUIRE_CORROBORATION:
            hard_gate_eligible = is_strong
        elif not is_strong:
            hard_gate_eligible = False
        elif best["category"] == "NAME":
            hard_gate_eligible = _name_hard_gate_eligible(text[start:end], start, text, methods, set(engines), score)
        else:
            # IDNUM / CONTACT / URL: already format-validated in postprocess; trust a
            # pattern match, cross-engine agreement, or very high confidence.
            hard_gate_eligible = len(engines) >= 2 or score >= STRONG_PII_HARD_GATE_CONFIDENCE or "regex" in methods
        canonical = sorted({CANONICAL_LABEL.get(x, "MISC") for x in categories})
        cross_validated = len(engines) >= 2
        confirmed = hard_gate_eligible and bool(set(canonical) & HARD_GATE_PII_LABELS)
        # review_worthy is the file-flagging bar: a file is flagged for PII only if it
        # has at least one review_worthy span. This is the main lever against
        # over-flagging -- a lone single-engine weak guess, or a common word tagged
        # NAME with no real name signal (even if two engines share the false positive),
        # is recorded below but does NOT flag the file.
        best_cat = best["category"]
        # FLAG_ALL_NAMES: every detected name is review-worthy, whoever it belongs to
        # and however weak the signal (no auto-fail -- NAME stays out of the hard gate).
        name_signal = best_cat == "NAME" and (FLAG_ALL_NAMES or hard_gate_eligible)
        # An age above AGE_REVIEW_OVER_YEARS always flags, despite AGE's weak weight.
        elderly_age = any(m.get("age_over_threshold") for m in c)
        contextual_or_id = best_cat in ((STRONG_RIGID | CONTEXTUAL_RIGID) - {"NAME"})
        quasi_identifier = best_cat == "MISC" and bool({"rare_role", "demographic"} & methods)
        if best_cat == "COMBINATORIAL":
            review_worthy = True
        elif confirmed or name_signal or quasi_identifier or elderly_age:
            review_worthy = True  # identifier, real-name signal, or nuanced quasi-id
        elif contextual_or_id:
            review_worthy = cross_validated  # ORG/LOC/DATE/ID: only if engines agree
        else:
            review_worthy = False  # AGE/TIME/PROFESSION/plain-MISC alone: informational
        # High-recall screening: flag ANY real detection for human review. Only the
        # classic lone common-word NAME false positive (no name signal) stays quiet,
        # unless RECALL_FLAG_ALL_PII forces even those. Minimizes false negatives.
        if HIGH_RECALL:
            lone_common_name = best_cat == "NAME" and not hard_gate_eligible
            if RECALL_FLAG_ALL_PII or not lone_common_name:
                review_worthy = True
        merged.append(
            {
                "start": start,
                "end": end,
                "text": text[start:end],
                "categories": categories,
                "canonical_labels": canonical,
                "rigidity_tier": rigidity_tier(best["category"]),
                "score": score,
                "engines": engines,
                "cross_validated": cross_validated,
                "hard_gate_eligible": hard_gate_eligible,
                "confirmed": confirmed,
                "review_worthy": review_worthy,
                "needs_review": review_worthy,
                "detections": c,
            }
        )
    return merged


def build_masked_preview(text: str, entities: list[dict[str, Any]]) -> str:
    """Reviewer-convenience preview only.

    Never written back to the source -- nothing is auto-redacted (poster).
    """
    preview = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        tag = f"[{ent['canonical_labels'][0]}]"
        preview = preview[: ent["start"]] + tag + preview[ent["end"] :]
    return preview
