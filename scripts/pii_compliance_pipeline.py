#!/usr/bin/env python3
"""
Local, fully-offline PII Detection + Task Compliance pipeline for ASR
transcripts stored in .pt files.

STANDALONE, as a PAIR OF FILES that travel together:

    pii_compliance_pipeline.py   this script
    task_reference.json          the 797-task reference table it checks against

Copy both into the same folder and it runs anywhere -- it imports nothing from
this repository and reads no config file. The JSON is resolved relative to the
SCRIPT's folder, not the working directory, so the pair works from any cwd. If
it is missing the run stops with a message rather than quietly skipping the
scripted-task checks. Everything else the script can read -- a protocol spec --
is genuinely optional.

This tool implements the two verification tasks from the poster
"An AI Pipeline for Ensuring Audio Data Quality" (Ng, Wilke, Johnson,
Ghosh) --

    (3) PII Detection
    (4) Task Compliance

-- operationalised with the theory from the companion proposal
"Unified Framework for PII Detection and Task Compliance Verification
in Clinical Voice Recordings". Both tasks are treated as one *reference
problem* under Kripke's rigid-designation framework: for any span, the
question is not "is this a listed type?" but "does this expression fix a
unique individual / refer to a required action?".

WHY MORE THAN ONE MODEL (per your request + the poster):
  * PII engines run IN PARALLEL and are CROSS-VALIDATED: a rigidity
    rule cascade, GLiNER (HIPAA identifiers), and Presidio (NER). Where
    engines agree, confidence is boosted; where they disagree, the span
    is surfaced for human review. Nothing is ever auto-redacted.
  * A LOCAL LLM (gemma4 via Ollama, one warm model) reasons over the
    residual/ambiguous PII cases and over open-ended compliance. The
    confidence it reports on each judgement widens the human-review band
    exactly as the poster's confidence formula prescribes.

EVERYTHING RUNS LOCALLY. Transcript text is only ever sent to
localhost (Ollama) or processed in-process (spaCy / GLiNER / Presidio /
sentence-transformers). No transcript text leaves the machine. The only
network traffic is the one-time download of model *weights* and
gazetteers, never your data.

SCOPE: this tool reads TEXT (the "transcription" field of each .pt).
Poster steps that need raw audio -- (1) Audio & Environment Quality,
(2) Unconsented Speaker Detection, and Compliance Tier B (acoustic
features) -- cannot run on text alone and are reported as
"skipped: requires audio". The text-level tasks (PII, Compliance
Tier A + Tier C, composite scoring over the available checks) are
fully implemented.

Each .pt file is loaded as:
    data = torch.load(path, map_location="cpu", weights_only=False)
    transcription = data["transcription"]

CONFIGURATION lives in ONE place: the CONFIG block a little further down. Set
the values there and run the file -- there are no environment variables to
export and no config file to locate. Command-line arguments override the block
when you want a one-off change. The only setting you MUST provide is the input
folder; with no input path the script stops and says so rather than guessing.

Usage (IDE): set INPUT_FOLDER in the CONFIG block below and hit Run.
Usage (terminal):
    python pii_compliance_pipeline.py /path/to/pt_folder
    python pii_compliance_pipeline.py /path/to/pt_folder --protocol my_protocol.json
    python pii_compliance_pipeline.py /path/to/pt_folder --flagged-list flagged.txt
    python pii_compliance_pipeline.py --selftest

OUTPUT: a JSON report, an optional CSV summary, and -- the point of the run --
the list of flagged file names, printed and available as summary.flagged_files
in the JSON (or its own text file via --flagged-list).

DEPENDENCIES: only ``torch`` is required (to read the .pt files). Every
detection engine is optional and degrades gracefully with a printed notice if
it is absent -- run with none of them and you still get the rule cascade plus
task compliance. In this repository they come from the extras:
    uv sync --extra pii --extra nlp        # presidio + spacy, nltk
    uv run python -m spacy download en_core_web_lg
    uv run pip install gliner wordfreq     # optional: HIPAA engine, word frequencies
The local LLM (off by default) additionally needs Ollama on localhost.

FULL DOCUMENTATION: scripts/pii_compliance_pipeline.md (beside this file) -- how each engine and tier
works, the precision/recall modes and what they cost, the known blind spots, and
data-safety notes. Read the "Modes" section before a first real run.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch


def _importable(module_name, timeout=60):
    """Check whether `import module_name` succeeds -- in a SUBPROCESS.

    Some optional engines (e.g. gliner / sentence-transformers when they pull a
    TensorFlow build compiled for unavailable CPU instructions) abort the whole
    process at the C level on import; a normal try/except cannot catch a SIGABRT.
    Probing in a throwaway subprocess isolates that crash so the pipeline keeps
    running with the engine simply skipped."""
    try:
        r = subprocess.run([sys.executable, "-c", f"import {module_name}"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=timeout)
        return r.returncode == 0
    except Exception:  # noqa: BLE001
        return False

# =============================================================================
# MODES -- flip these on or off, then hit Run. Nothing else needed.
#
# This block and SETTINGS below are the whole configuration: no environment
# variables, no config file, no command line required. (Every mode is also a
# CLI flag if you ever want a one-off override from a terminal, but running the
# file straight from an IDE uses exactly what is written here.)
#
# The active modes are printed at the start of every run, so you can always see
# what was on without re-reading this block.
# =============================================================================

# ---- What to run ------------------------------------------------------------
SELFTEST = False               # run the built-in self-test on synthetic data, not a real scan
COMPLIANCE_ONLY = False        # skip PII detection entirely: task compliance only (much faster)
RECURSIVE = False              # also scan sub-folders of INPUT_FOLDER

# ---- PII detection engines (each is skipped with a notice if not installed) --
ENABLE_SPACY = True            # spaCy NER
ENABLE_GLINER = True           # GLiNER -- the HIPAA-identifier engine
ENABLE_PRESIDIO = True         # Microsoft Presidio NER

# ---- Local LLM (needs Ollama running -- see OLLAMA_* in SETTINGS) -----------
# The only thing that can judge open-ended (Tier C) compliance: "did this free-speech
# answer actually address the prompt?" is a semantic question no rule reaches. Roughly
# DOUBLES the catch rate, at roughly 10x the review queue, and it is slow (~1-2s per
# file once warm). Off is a deliberate precision choice, not a capability gap.
# --> NOTES [LLM] at the end of this block for the full tradeoff.
ENABLE_LLM = True
ENABLE_LLM_COMPLIANCE_JUDGE = True   # also judge BORDERLINE transcripts; needs ENABLE_LLM

# ---- Posture: precision vs recall -------------------------------------------
PRECISION_MODE = True          # false-positive guards. --> NOTES [PRECISION]
HIGH_RECALL = False            # screening mode: catch more, review more. Forces PRECISION_MODE off.
# The next three only do anything while HIGH_RECALL is on (setting any of them
# switches it on for you, with a printed note). --> NOTES [RECALL]
RECALL_FLAG_ALL_PII = False    # flag even lone common-word NAME guesses (Will / May / Grant)
RECALL_FLAG_ALL_OPEN = False   # route EVERY open free-speech task to review
RECALL_FLAG_ACOUSTIC = False   # route acoustic tasks with unexpected speech to review

# ---- Always-flag identifiers ------------------------------------------------
FLAG_ALL_NAMES = True          # every detected name goes to the redaction queue. --> NOTES [NAMES]
DEMOGRAPHIC_PII = True         # detect self-disclosed gender / race-ethnicity. --> NOTES [DEMOGRAPHIC]

# ---- Hard-gate behaviour (what may FAIL a file rather than route to review) --
HARD_GATE_MISSING_REQUIRED_STEP = True           # a missing REQUIRED protocol step is a real failure
HARD_GATE_REQUIRE_CORROBORATION = True           # a lone weak strong-PII guess never auto-fails
COMBINATORIAL_REQUIRE_CATEGORY_DIVERSITY = True  # combinatorial risk needs 2 DIFFERENT categories


# =============================================================================
# SETTINGS -- paths, models and thresholds. Same deal: edit and run.
# =============================================================================

# ---- Paths. INPUT_FOLDER is the one value you MUST set ----------------------
# With no input folder here (and none on the command line) the run stops immediately
# and says so: it does not guess, and it does not scan the working directory.
INPUT_FOLDER = "/Users/varunthvar1/Desktop/sample_data"   # >>> SET THIS <<< folder of .pt files
MAX_FILES = 10                 # cap the number of files, or None for all. Handy for a pilot run.
# The task table this pipeline checks every .pt against, shipped NEXT TO this script.
# Keep the two together. A relative path resolves against the SCRIPT's folder, never
# the working directory. Missing -> the run STOPS. --> NOTES [TASK REFERENCE]
TASK_REFERENCE = "task_reference.json"
# OPTIONAL. None = zero-config: task info is read from each .pt's own metadata. Set a
# JSON path only for interview-style protocol-step checking (consent sequences etc.).
PROTOCOL_SPEC = None                             # e.g. "protocol_spec.example.json", or None

# ---- Outputs ----------------------------------------------------------------
OUTPUT_JSON = "pii_compliance_report.json"       # full JSON report
OUTPUT_CSV = "pii_compliance_summary.csv"        # flat CSV summary, or "" to skip
# Just the flagged file paths, one per line -- the plain-text list of what needs review.
# "" means don't write it; the list is printed either way and is in the JSON report
# under summary.flagged_files.
OUTPUT_FLAGGED_LIST = ""                         # e.g. "flagged_files.txt"

# ---- Engine models and thresholds -------------------------------------------
SPACY_MODEL = "en_core_web_lg"
GLINER_MODEL = "urchade/gliner_multi_pii-v1"
GLINER_THRESHOLD = 0.6         # raise to cut GLiNER false positives (was 0.4)
# GLiNER caps input at ~384 tokens and SILENTLY TRUNCATES longer text, dropping any PII
# past the cap, so long transcripts are windowed through it instead. Keep max words
# safely under the cap; overlap so entities on a chunk boundary are not split.
GLINER_MAX_WORDS = 320
GLINER_OVERLAP_WORDS = 40

# ---- Local LLM connection ---------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
# Any tag Ollama can serve, e.g. "gemma4:12b", "phi4", "llama3.1". Pull it first:
#   ollama pull gemma4
OLLAMA_MODEL = "gemma4"
LLM_TIMEOUT_SECONDS = 180
# Below this reported confidence the model's yes/no is treated as UNDECIDED rather than
# as an answer: the file routes to review instead of being scored pass or hard-fail.
LLM_CONFIDENT_AT = 0.5

# ---- Precision thresholds ---------------------------------------------------
MIN_ENGINE_CONFIDENCE = 0.5          # drop low-confidence model detections (pattern hits exempt)
STRONG_PII_HARD_GATE_CONFIDENCE = 0.85   # a lone strong-PII guess this confident may hard-fail
# A single-token NAME that is also a common English word (zipf >= this) with no real name
# signal is treated as an NER false positive: recorded, but it does not flag the file.
# Lower => stricter (fewer name flags).
NAME_COMMON_WORD_ZIPF = 3.9

# ---- Rigidity cascade thresholds (PII, from the prior proposal) --------------
COMBINATORIAL_WINDOW_TOKENS = 15
COMBINATORIAL_THRESHOLD = 0.5
INDIVIDUAL_FLAG_THRESHOLD = 0.4
RARE_WORD_ZIPF_THRESHOLD = 2.7
RARE_ROLE_QUALIFIER_ZIPF_MAX = 3.9   # soft-qualifier role fires only if it contains an uncommon word
RARE_ROLE_INTRO_ZIPF_MAX = 3.6       # 'I am a <role>' fires only for uncommon occupations

# ---- Which PII labels may HARD-FAIL a file ----------------------------------
# Canonical labels (AGE/NAME/DATE/TIME/ORG/IDNUM/LOC/PROFESSION/CONTACT/URL/MISC) whose
# CONFIRMED presence hard-fails. Default = the DIRECT identifiers only. Add "NAME" to
# make names fail too; use set() to never fail on PII. --> NOTES [HARD GATE]
HARD_GATE_PII_LABELS = {"IDNUM", "CONTACT", "URL"}
# An age STRICTLY GREATER than this is flagged for review. HIPAA Safe Harbor treats all
# ages over 89 as identifiers, so use 89 for strict HIPAA; 90 matches "any age over 90".
AGE_REVIEW_OVER_YEARS = 90

# ---- High-recall tuning (only bites while HIGH_RECALL is on) ----------------
# An ACOUSTIC task with MORE content words than this routes to REVIEW. Raise it if
# legitimate reps ("pa-ta-ka", counting) trip it. -1 routes EVERY acoustic task to
# review -- the only way to reach acoustic misses, at the cost of the entire acoustic
# half of the corpus in queue.
RECALL_ACOUSTIC_MAX_CONTENT_WORDS = 4
RECALL_OPEN_MIN_CONTENT_WORDS = 12   # an OPEN task thinner than this routes to REVIEW

# ---- Composite scoring (poster Section 5) -----------------------------------
# Q = sum(w_i * c_i * s_i);  C = weighted_mean(c_i) * (1 - LAMBDA * std(c_i))
CHECK_WEIGHTS = {"pii": 0.5, "compliance": 0.5}
CONFIDENCE_LAMBDA = 0.5
Q_PASS_CUTOFF = 0.9
Q_FAIL_CUTOFF = 0.5

# ---- Transcript quality (compliance signal that needs NO task metadata) -----
# A degenerate transcript is a genuine compliance failure whatever the task was:
# empty / non-speech markers only / filler vocalizations only. Near-degenerate goes
# to review. --> NOTES [QUALITY]
COMPLIANCE_HARD_FAIL_MIN_CONTENT_WORDS = 1   # < this many content words -> hard FAIL
COMPLIANCE_REVIEW_MIN_CONTENT_WORDS = 3      # < this many (but >= hard-fail) -> REVIEW
COMPLIANCE_MAX_FILLER_RATIO = 0.6            # filler words / all words above this -> REVIEW
COMPLIANCE_SHORT_RESPONSE_WORDS = 8          # filler ratio only judged for responses this short

# ---- Tier A (scripted read) thresholds --------------------------------------
# Calibrated to ASR, not to a clean text diff. --> NOTES [TIER A]
#     WER <= TIER_A_WER_REVIEW              -> score 1.0..0.9   -> PASS   (read it; ASR noise)
#     TIER_A_WER_REVIEW < WER <= ..._FAIL   -> score 0.9..0.5   -> REVIEW (doubtful read)
#     WER >  TIER_A_WER_FAIL                -> score 0.5..0.0   -> FAIL   (wrong text / no read)
TIER_A_WER_REVIEW = 0.50
TIER_A_WER_FAIL = 0.80
TIER_A_MIN_WORD_ERRORS = 4     # never flag a read with fewer word errors, whatever the ratio
TIER_A_COVERAGE_PASS = 1.1     # 1.1 = the coverage guard is OFF. --> NOTES [TIER A]


# =============================================================================
# NOTES -- why the defaults are what they are. Reference only; nothing to edit.
# =============================================================================
#
# [LLM]  ENABLE_LLM
#   ONE model, kept warm for the whole run. Ollama holds a limited set resident, so
#   cycling calls across several large models forces a disk reload on nearly every
#   call; that reload tax, not generation, dominates runtime. A single warm model is
#   several times faster, and multi-model voting never justified its cost here.
#   Measured on the study corpus the LLM roughly DOUBLES the catch rate (12 -> 20 of
#   42 real problems). It also flags ~79 compliant open-task recordings to do it,
#   taking the review queue from 11 files to 100 and the share of the queue that is
#   real from ~92% to ~20%. Which way that trades depends on what a reviewer's time
#   is worth against a missed problem -- a per-study decision, not a default. Off
#   gives a small, dense, high-precision queue; on gives ~2x the recall for ~10x the
#   queue.
#   ENABLE_LLM_COMPLIANCE_JUDGE asks the model, task-agnostically, whether a
#   BORDERLINE transcript reads like a completed spoken response. It runs only on
#   files already in the review band -- never on every file, never on acoustic tasks,
#   never where a Tier A/C score is authoritative -- and can lower a score into review
#   but never rescue a hard fail. It runs THROUGH the same model, so it is inert
#   unless ENABLE_LLM is on. Left on so that enabling the LLM gets the whole LLM
#   configuration rather than silently omitting the borderline judge.
#
# [PRECISION]  PRECISION_MODE
#   Turns on all the false-positive guards. Everything a guard removes from the
#   auto-FAIL path is still surfaced as needs_review -- true positives are never
#   dropped, only re-routed away from hard rejection. HIGH_RECALL forces this off, so
#   low-confidence detections survive to be flagged.
#
# [RECALL]  HIGH_RECALL and the RECALL_* modes
#   Posture: catch as many problems as possible and route them to human REVIEW (never
#   a hard fail), accepting more false positives. Turn on when a MISSED problem costs
#   more than an extra review. Every recall knob only ADDS review flags; none of them
#   can auto-reject a file.
#   RECALL_FLAG_ACOUSTIC is off by default because acoustic non-compliance (too few
#   reps, no attempt) yields a near-empty transcript indistinguishable from a
#   compliant one, while compliant DDK/counting tasks often DO transcribe as words --
#   so flagging acoustic is high-false-positive for almost no recall.
#   RECALL_FLAG_ALL_OPEN is the no-LLM fallback for "answered the wrong question",
#   which otherwise needs ENABLE_LLM. More false positives.
#
# [NAMES]  FLAG_ALL_NAMES
#   Every detected personal name is review-worthy, no matter whose it is or how weak
#   the signal -- a spoken name always belongs in the human redaction queue. This
#   deliberately overrides the "lone common word tagged NAME" precision guard, so NER
#   false positives (Will / May / Grant / Mark) WILL flag files. Names still route to
#   REVIEW, never an auto-fail: NAME stays out of HARD_GATE_PII_LABELS.
#
# [DEMOGRAPHIC]  DEMOGRAPHIC_PII
#   Self-disclosed gender / gender-identity and race / ethnicity are sensitive,
#   protected attributes; combined with other detail they help re-identify a
#   participant. Mentions surface to REVIEW as MISC quasi-identifiers, never an
#   auto-FAIL -- a human/ethics reviewer makes the call. Kept HIGH-PRECISION: only
#   SELF- or person-anchored mentions fire ("I'm Black", "as a trans man", "my
#   ethnicity is ..."), never bare nouns, pronouns or incidental words ("black
#   coffee", "Chinese food", "the man over there").
#   The same machinery covers unusually SPECIFIC roles ("professional figure skater",
#   which ethics flagged in the pediatric study as "a bit too much info") via
#   RARE_ROLE_*: identifying without being a listed PII type.
#
# [HARD GATE]  HARD_GATE_PII_LABELS
#   The decision is three-way (Pass / Needs Review / Fail), then FAIL is collapsed
#   into REVIEW at the very end so nothing is ever auto-rejected. Most detected PII
#   means "send to the redaction queue", NOT "reject the file": nothing is
#   auto-redacted, and in voice data participants naturally speak names, dates and
#   places, which are redactable rather than reject-worthy. Only CONFIRMED DIRECT
#   identifiers (SSN / MRN / email / phone / URL) are serious enough to hard-fail. A
#   spoken NAME is deliberately excluded.
#
# [QUALITY]  COMPLIANCE_* word counts
#   The Task-Compliance step needs to know a task actually happened. When a .pt
#   carries only a transcript, the strongest ground-truth-free signal is whether the
#   transcript is a real spoken response at all: empty, non-speech markers only, or
#   filler-only means nothing was said, whatever the task. A transcript with real
#   content is treated as a valid attempt -- it cannot be failed on text alone
#   without a reference or rubric, which is what Tier A / Tier C add.
#
# [TIER A]  TIER_A_*
#   Tier A asks ONE question: "did the participant read the script we gave them?" --
#   not "how good is the ASR?". A raw score = 1 - WER fed into the generic Q cutoffs
#   sends everything above WER 0.1 to review, far too tight: on clinical read-aloud
#   speech a 0.1-0.4 WER is the NORMAL cost of dysphonia, accent, room noise and ASR
#   error on someone who read the script perfectly well. Someone who read the WRONG
#   text lands much higher (0.6-1.0+).
#   TIER_A_MIN_WORD_ERRORS exists because WER is a ratio and punishes short
#   references hardest: three misheard words in a 5-word CAPE-V sentence is WER 0.6,
#   while the same three errors in the 64-word rainbow passage is WER 0.05. The short
#   sentence is not three times less compliant.
#   TIER_A_COVERAGE_PASS is OFF by default (1.1 is unreachable) because it is an
#   assumption about the ANNOTATORS, not about ASR. It only helps if your reviewers
#   treat extra speech around a correct read as compliant. In the study corpus they do
#   NOT: two recordings containing every reference word plus 5-7 extra were both
#   labelled `partial`, and the guard turned both into misses while rescuing no false
#   positives. Measure before enabling -- set ~0.90 and re-run the evaluator.
#
# [TASK REFERENCE]  TASK_REFERENCE
#   Each .pt filename encodes its task in a BIDS-style `task-<Name>` field, e.g.
#   sub-XXXX_ses-YYYY_task-Word-color-Stroop_features.pt -> "word-color-stroop".
#   The JSON maps every task to its instructions + read-aloud prompts, which is how
#   the pipeline learns each file's MODALITY -- and the modality decides how, or
#   whether, compliance can be judged from text at all:
#     * scripted (read a sentence/passage) -> Tier A: word error rate vs the prompt.
#     * open     (free speech, naming)     -> Tier C LLM judge, else "speech expected".
#     * acoustic (vowel, DDK, breath, cough, glides, phonation, loudness)
#                                          -> Tier B: NOT text-assessable. A near-empty
#                                             transcript here is EXPECTED, never a failure.
#   A missing file stops the run rather than silently disabling every scripted check.
# =============================================================================


# ---------------------------------------------------------------------------
# Rigidity spectrum (proposal Section 2.2 / 5) + canonical guideline labels
# ---------------------------------------------------------------------------
STRONG_RIGID = {"NAME", "IDNUM", "CONTACT", "URL"}
CONTEXTUAL_RIGID = {"ORG", "LOC_SPECIFIC", "DATE_SPECIFIC"}
WEAK_RIGID = {"AGE", "TIME", "DATE_PARTIAL", "PROFESSION", "LOC_GENERIC"}
MISC = {"MISC"}

CATEGORY_WEIGHTS = {
    **{c: 1.0 for c in STRONG_RIGID},
    **{c: 0.6 for c in CONTEXTUAL_RIGID},
    **{c: 0.25 for c in WEAK_RIGID},
    "MISC": 0.3,
    "COMBINATORIAL": 0.6,
}

# Map the tool's fine-grained categories onto the annotation-guidelines label
# set (AGE, NAME, DATE, TIME, ORG, IDNUM, LOC, PROFESSION, CONTACT, URL, MISC)
# so output is compatible with the label-studio config in the guidelines.
CANONICAL_LABEL = {
    "NAME": "NAME", "IDNUM": "IDNUM", "CONTACT": "CONTACT", "URL": "URL",
    "ORG": "ORG", "LOC_SPECIFIC": "LOC", "LOC_GENERIC": "LOC",
    "DATE_SPECIFIC": "DATE", "DATE_PARTIAL": "DATE", "AGE": "AGE", "TIME": "TIME",
    "PROFESSION": "PROFESSION", "MISC": "MISC", "COMBINATORIAL": "MISC",
}


def rigidity_tier(category: str) -> str:
    if category in STRONG_RIGID:
        return "strongly_rigid"
    if category in CONTEXTUAL_RIGID:
        return "contextually_rigid"
    if category in WEAK_RIGID:
        return "weakly_rigid"
    if category == "COMBINATORIAL":
        return "combinatorial"
    return "misc"


def _entity(start, end, category, confidence, method, **extra):
    e = {"start": start, "end": end, "category": category,
         "confidence": confidence, "method": method}
    e.update(extra)
    return e


# ---------------------------------------------------------------------------
# Precision guards -- reduce false positives without dropping true positives.
# ---------------------------------------------------------------------------
# High-precision methods whose hits we trust even below MIN_ENGINE_CONFIDENCE
# (they matched an explicit pattern/list, not a soft model score).
_PATTERN_METHODS = {"regex", "honorific", "gazetteer", "keyword", "demographic"}
_DIGIT_RE = re.compile(r"\d")

# Holidays / named time-periods. Per the annotation guidelines these are DATES,
# NOT places/orgs/names -- so if any engine tags one as NAME/LOC/ORG we reclassify
# it to a (weak) DATE rather than letting it read as a strong/contextual identifier.
# This is what stops "Easter" surfacing as a LOCATION or ORGANIZATION.
HOLIDAYS = {
    "christmas", "christmas eve", "easter", "good friday", "thanksgiving",
    "halloween", "new year", "new year's", "new years", "new year's eve",
    "hanukkah", "chanukah", "passover", "yom kippur", "rosh hashanah",
    "ramadan", "eid", "diwali", "holi", "lent", "advent", "boxing day",
    "independence day", "memorial day", "labor day", "labour day",
    "veterans day", "columbus day", "juneteenth", "mardi gras", "carnival",
    "valentine's day", "valentines day", "st patrick's day", "cinco de mayo",
}


# Cues that a nearby capitalized token really is a person's name (high precision
# for ASR speech: "my name is ...", "I'm ...", "this is ...", honorifics, etc.).
NAME_CONTEXT_RE = re.compile(
    r"\b(?:name\s+is|name'?s|my\s+name|named|call(?:ed)?\s+me|i\s*am|i'?m|"
    r"this\s+is|it'?s|mr|mrs|ms|miss|dr|prof(?:essor)?|sir|madam|dame|rev|"
    r"father|speaking\s+with|speaking\s+to|interview(?:ing)?\s+with|"
    r"patient|participant|meet)\b",
    re.IGNORECASE,
)


def _zipf(word):
    try:
        from wordfreq import zipf_frequency
        return zipf_frequency(word.lower(), "en")
    except Exception:  # noqa: BLE001
        return 0.0     # no wordfreq -> don't suppress (favor recall)


def _name_hard_gate_eligible(span_text, start, source_text, methods, engines, score):
    """A NAME hard-fails a file only with a real name signal. A lone single-token
    common word tagged as a name by an NER model (Will / May / Grant / Mark ...) is
    the classic false positive -- it drops to needs_review instead of failing."""
    tokens = span_text.split()
    multitoken = len(tokens) >= 2
    precise = bool(methods & {"honorific", "gazetteer"})
    corroborated = len(engines) >= 2
    preceding = source_text[max(0, start - 30):start]
    has_context = bool(NAME_CONTEXT_RE.search(preceding))
    common = (not multitoken) and _zipf(span_text) >= NAME_COMMON_WORD_ZIPF
    if common and not (has_context or precise or multitoken):
        return False
    return multitoken or has_context or precise or corroborated or score >= 0.9


def _valid_structured_identifier(text, category):
    """Structured identifiers must actually look structured. A word with no
    digits / '@' / URL shape is almost always a model hallucination -- e.g. an
    ASR'd cough token flagged as a 'device identifier'. Real SSNs, MRNs, phone
    numbers, emails, and URLs always satisfy these."""
    t = (text or "").strip()
    if not t:
        return False
    if category == "URL":
        return bool(re.search(r"(https?://|www\.|\.[a-z]{2,})", t, re.IGNORECASE))
    if category == "CONTACT":                      # email / phone / fax / IP
        return ("@" in t) or len(_DIGIT_RE.findall(t)) >= 7 or \
            bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", t))
    if category == "IDNUM":                         # ids carry digits
        return _DIGIT_RE.search(t) is not None
    return True


def postprocess_entities(entities, source_text):
    """Apply the precision guards to a flat list of engine detections."""
    if not PRECISION_MODE:
        return entities
    out = []
    for e in entities:
        span_text = source_text[e["start"]:e["end"]]
        cat = e["category"]
        # 1) Holiday/named-period reclassification (guidelines: these are DATES).
        low = span_text.lower().strip().strip(".,'’")
        if low in HOLIDAYS and cat in ("NAME", "ORG", "LOC_SPECIFIC", "LOC_GENERIC", "MISC"):
            e = dict(e, category="DATE_PARTIAL", confidence=min(e["confidence"], 0.6),
                     method=e["method"] + "+holiday-reclass")
            cat = "DATE_PARTIAL"
        # 2) Structured-identifier format validation (kills word-only "identifiers").
        if cat in ("IDNUM", "CONTACT", "URL") and not _valid_structured_identifier(span_text, cat):
            continue
        # 3) Low-confidence soft detections dropped (pattern/list hits are exempt).
        method_root = e["method"].split(":")[0].split("+")[0]
        if e["confidence"] < MIN_ENGINE_CONFIDENCE and method_root not in _PATTERN_METHODS:
            continue
        out.append(e)
    return out


# ===========================================================================
# PART A -- PII DETECTION ENGINES (run in parallel, cross-validated)
# ===========================================================================

# --- Gazetteers (Stage 1 of the rigidity cascade) --------------------------
def load_name_gazetteer():
    """~8000 common given names (NLTK 'names' corpus); downloaded once."""
    import nltk
    try:
        from nltk.corpus import names
        names.words()
    except LookupError:
        nltk.download("names", quiet=True)
        from nltk.corpus import names
    return {n.lower() for n in names.words()}


US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york", "north carolina",
    "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania", "rhode island",
    "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont",
    "virginia", "washington", "west virginia", "wisconsin", "wyoming",
}
MAJOR_CITIES = {
    "new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia",
    "san antonio", "san diego", "dallas", "san jose", "austin", "boston", "seattle",
    "denver", "washington", "nashville", "detroit", "portland", "las vegas",
    "baltimore", "atlanta", "miami", "toronto", "vancouver", "montreal", "london",
    "paris", "berlin", "madrid", "rome", "amsterdam", "dublin", "cairo", "lagos",
    "nairobi", "johannesburg", "mumbai", "delhi", "bangalore", "beijing", "shanghai",
    "hong kong", "tokyo", "seoul", "singapore", "bangkok", "manila", "jakarta",
    "sydney", "melbourne", "auckland", "mexico city", "sao paulo", "buenos aires",
    "lima", "bogota", "moscow", "istanbul", "damascus", "baghdad", "tehran",
}


def load_place_gazetteer():
    places = set()
    try:
        import pycountry
        places = {c.name.lower() for c in pycountry.countries}
    except Exception:  # noqa: BLE001
        pass
    return places | US_STATES | MAJOR_CITIES


ORG_SUFFIX_RE = re.compile(
    r"\b([A-Z][\w&.,'-]*(?:\s+[A-Z][\w&.,'-]*){0,4}\s+"
    r"(?:Hospital|University|Clinic|Center|Centre|Foundation|Institute|"
    r"Inc\.?|LLC|Corp\.?|Ltd\.?|Co\.))\b"
)
CALENDAR_WORDS = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "spring", "summer", "autumn", "fall", "winter",
}


def gazetteer_scan(text, name_set, place_set):
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+){0,2}\b", text):
        phrase = m.group(0)
        words = phrase.split()
        if phrase.lower() in place_set:
            entities.append(_entity(m.start(), m.end(), "LOC_SPECIFIC", 0.9, "gazetteer"))
            continue
        if len(words) >= 2 and any(
            w.lower() in name_set and w.lower() not in CALENDAR_WORDS for w in words
        ):
            entities.append(_entity(m.start(), m.end(), "NAME", 0.85, "gazetteer"))
    for m in ORG_SUFFIX_RE.finditer(text):
        entities.append(_entity(m.start(1), m.end(1), "ORG", 0.8, "gazetteer"))
    return entities


# --- Stage 1 regex over structured identifiers -----------------------------
REGEX_PATTERNS = [
    ("CONTACT", 0.95, re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),                                   # email
    ("CONTACT", 0.90, re.compile(r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)")),  # phone
    ("IDNUM",   0.95, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),                                           # SSN
    ("URL",     0.95, re.compile(r"\b(?:https?://|www\.)[^\s,]+", re.IGNORECASE)),
    ("CONTACT", 0.90, re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),                                     # IPv4
    ("IDNUM",   0.85, re.compile(
        r"\b(?:MRN|medical record(?:\s+number)?|account number|patient id|"
        r"certificate number|license number|policy number|member id|health plan number)"
        r"\s*[:#]?\s*([A-Za-z0-9-]{4,})\b", re.IGNORECASE)),
    ("DATE_SPECIFIC", 0.90, re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b")),
    ("DATE_SPECIFIC", 0.90, re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("AGE",           0.85, re.compile(r"\b\d{1,3}[\s-]year[\s-]old\b", re.IGNORECASE)),
]
HONORIFIC_RE = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Rev|Fr|Sr|Sir|Dame)\.?\s+"
    r"([A-Z][a-zA-Z'-]+(?:\s+[A-Z][a-zA-Z'-]+)?)"
)
PROFESSION_KEYWORDS = {
    "doctor", "nurse", "surgeon", "physician", "therapist", "psychologist",
    "psychiatrist", "teacher", "professor", "engineer", "lawyer", "attorney",
    "technician", "officer", "manager", "director", "researcher", "scientist",
    "student", "resident", "pharmacist", "counselor", "social worker", "paramedic",
    "lieutenant", "captain", "sergeant", "president",
}


def regex_scan(text):
    entities = []
    for category, confidence, pattern in REGEX_PATTERNS:
        for m in pattern.finditer(text):
            span = m.span(1) if m.groups() else m.span()
            entities.append(_entity(span[0], span[1], category, confidence, "regex"))
    return entities


# --- Stage 2 weak signals (spaCy NER, honorific, profession, rare word) ----
_SPACY_LABEL_MAP = {"PERSON": "NAME", "ORG": "ORG", "TIME": "TIME"}


def ner_scan(nlp, text, place_set):
    entities = []
    for ent in nlp(text).ents:
        if ent.label_ in ("GPE", "LOC"):
            cat = "LOC_SPECIFIC" if ent.text.lower() in place_set else "LOC_GENERIC"
            entities.append(_entity(ent.start_char, ent.end_char, cat, 0.75, "ner"))
        elif ent.label_ == "DATE":
            cat = "DATE_SPECIFIC" if re.search(r"\b(19|20)\d{2}\b", ent.text) else "DATE_PARTIAL"
            entities.append(_entity(ent.start_char, ent.end_char, cat, 0.7, "ner"))
        elif ent.label_ in _SPACY_LABEL_MAP:
            entities.append(_entity(ent.start_char, ent.end_char,
                                    _SPACY_LABEL_MAP[ent.label_], 0.75, "ner"))
    return entities


def honorific_scan(text):
    return [_entity(m.start(1), m.end(1), "NAME", 0.7, "honorific")
            for m in HONORIFIC_RE.finditer(text)]


_MULTIWORD_PROFESSIONS = sorted((p for p in PROFESSION_KEYWORDS if " " in p),
                                key=len, reverse=True)


def profession_scan(text):
    """Flag profession keywords. Checks every single word individually (so an
    adjacent word never swallows the keyword) plus known multi-word professions."""
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


def rareword_scan(text, zipf_threshold):
    try:
        from wordfreq import zipf_frequency
    except Exception:  # noqa: BLE001
        return []
    entities = []
    for m in re.finditer(r"\b[A-Z][a-zA-Z'-]{3,}\b", text):
        word = m.group(0)
        preceding = text[: m.start()].rstrip()
        if preceding == "" or preceding[-1] in ".!?":  # skip sentence-initial caps
            continue
        if zipf_frequency(word.lower(), "en") <= zipf_threshold:
            entities.append(_entity(m.start(), m.end(), "MISC", 0.5, "rareword"))
    return entities


# A role/activity phrase of 1-2 words that does not run past a stopword (so
# "figure skater and I ..." captures just "figure skater").
_ROLE_STOP = (r"(?:and|or|but|who|that|which|because|so|when|while|the|a|an|to|of|"
              r"in|on|for|is|was|are|were|be|i|my|me|he|she|they|we|you|it|this)")
_ROLE_WORDS = (r"[a-z][a-z'-]+(?:\s+(?!" + _ROLE_STOP + r"\b)[a-z][a-z'-]+){0,1}")
# STRONG qualifiers are identifying on their own ("olympic <x>"). SOFT qualifiers
# ("professional <x>") flag only when the role phrase contains an uncommon word --
# so "professional figure skater" fires but "professional development" does not.
_STRONG_QUALIFIER_RE = re.compile(
    r"\b(olympic|paralympic|championship|champion|world[-\s]?class|elite|decorated|"
    r"renowned|award[-\s]?winning)\s+(" + _ROLE_WORDS + r")\b", re.IGNORECASE)
_SOFT_QUALIFIER_RE = re.compile(
    r"\b(professional|semi[-\s]?professional|competitive|national|international|"
    r"former|retired|amateur)\s+(" + _ROLE_WORDS + r")\b", re.IGNORECASE)
# A self-described occupation ("I am a <x>", "I work as a <x>").
_ROLE_INTRO_RE = re.compile(
    r"\b(?:i'm|i\s+am|i\s+was|i\s+used\s+to\s+be|worked\s+as|work\s+as|employed\s+as)\s+"
    r"(?:an?\s+|the\s+)?(" + _ROLE_WORDS + r")\b", re.IGNORECASE)


def _phrase_min_zipf(phrase):
    """Minimum word-frequency across a role phrase (a rare word anywhere signals
    specificity). Returns None if no word is known to wordfreq."""
    zs = [z for z in (_zipf(t) for t in re.findall(r"[a-z'-]+", phrase.lower())) if z > 0.0]
    return min(zs) if zs else None


def rare_role_scan(text):
    """Nuanced quasi-identifiers: unusually specific roles/activities/statuses that can
    re-identify a person on their own (annotation-guidelines MISC). Example that
    motivated this: a participant mentioning being a "professional figure skater".
    These are surfaced to REVIEW (never an auto-fail); the LLM and a human
    reviewer make the final call. Kept high-precision to avoid flagging common jobs."""
    ents = []
    for m in _STRONG_QUALIFIER_RE.finditer(text):
        ents.append(_entity(m.start(), m.end(), "MISC", 0.6, "rare_role"))
    for m in _SOFT_QUALIFIER_RE.finditer(text):
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
_PERSON_NOUN = (r"man|woman|male|female|guy|girl|boy|lady|gentleman|person|"
                r"individual|folk|folks|kid|child")
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
    r"\b" + _SUBJ + r"\s+(?:an?\s+)?((?:" + _RACE_TERMS + r")(?:\s+(?:" + _PERSON_NOUN + r"))?)\b",
    re.IGNORECASE)
_RACE_ADJ_RE = re.compile(
    r"\b((?:" + _RACE_TERMS + r")\s+(?:" + _PERSON_NOUN + r"))\b", re.IGNORECASE)
_RACE_EXPLICIT_RE = re.compile(
    r"\bmy\s+(?:race|ethnicity|ethnic\s+background|racial\s+background|heritage|ancestry)\s+"
    r"(?:is|are|was|were|being|:)?\s*([a-z][\w'-]+(?:\s+[a-z][\w'-]+)?)", re.IGNORECASE)
_POC_RE = re.compile(r"\b((?:person|people|woman|man|women|men)\s+of\s+colou?r)\b", re.IGNORECASE)
_GENDER_IDENTITY_RE = re.compile(r"\b(" + _GENDER_IDENTITY_TERMS + r")\b", re.IGNORECASE)
_GENDER_SELF_RE = re.compile(
    r"\b(?:" + _SUBJ + r"|(?:i|we|he|she|they)\s+identif(?:y|ies)\s+as)\s+"
    r"(?:an?\s+)?(" + _ANY_GENDER_SELF + r")\b", re.IGNORECASE)
_GENDER_ASA_RE = re.compile(
    r"\bas\s+an?\s+(" + _ANY_GENDER + r"(?:\s+(?:" + _PERSON_NOUN + r"))?)\b", re.IGNORECASE)
_GENDER_EXPLICIT_RE = re.compile(
    r"\bmy\s+(?:gender(?:\s+identity)?|sex|pronouns?)\s+(?:is|are|was|were|being|:)?\s*"
    r"([a-z][\w'-]+)", re.IGNORECASE)

_RACE_RES = (_RACE_SELF_RE, _RACE_ADJ_RE, _RACE_EXPLICIT_RE, _POC_RE)
_GENDER_RES = (_GENDER_IDENTITY_RE, _GENDER_SELF_RE, _GENDER_ASA_RE, _GENDER_EXPLICIT_RE)


def demographic_scan(text):
    """Nuanced demographic quasi-identifiers: SELF-disclosed race/ethnicity and gender/
    gender-identity. These are sensitive, protected attributes that (with other detail)
    re-identify a participant, so every hit is surfaced to REVIEW as MISC (never a hard
    FAIL). Precision comes from requiring a self/person anchor, so bare nouns, pronouns
    and incidental words ('black coffee', 'Chinese food') do not fire."""
    if not DEMOGRAPHIC_PII:
        return []
    ents, seen = [], set()

    def _add(a, b, subtype, conf):
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
_AGE_UNITS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
              "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
              "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_AGE_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_UNIT_RE = "|".join(_AGE_UNITS)
_TENS_RE = "|".join(_AGE_TENS)
# A spelled-out number up to ~199: "a/one hundred [and] [tens] [units]" | "ninety-five" | "ninety"
_AGE_WORD_NUM = (rf"(?:(?:one|a)\s+hundred(?:\s+and)?(?:\s+(?:{_TENS_RE}))?(?:[-\s](?:{_UNIT_RE}))?"
                 rf"|(?:{_TENS_RE})(?:[-\s](?:{_UNIT_RE}))?)")
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
    r"|over\s+(?:ninety|90|the\s+age\s+of\s+(?:ninety|90)))\b", re.IGNORECASE)


def _age_word_to_int(phrase):
    """Convert a spelled-out age ('ninety-five', 'a hundred and two') to an int,
    or None if it isn't one. Digits are handled by the caller."""
    words = re.findall(r"[a-z]+", phrase.lower())
    if not words:
        return None
    total = 0
    if "hundred" in words:
        total = 100
        words = words[words.index("hundred") + 1:]
    for w in words:
        if w in _AGE_TENS:
            total += _AGE_TENS[w]
        elif w in _AGE_UNITS:
            total += _AGE_UNITS[w]
        elif w in ("and", "a", "one") and total >= 100:
            continue
    return total or None


def age_scan(text, over_years=None):
    """Flag ages STRICTLY GREATER than `over_years` (default AGE_REVIEW_OVER_YEARS).
    Only the flaggable (very high) ages are emitted, so ordinary ages keep behaving
    exactly as before. Every hit carries age_over_threshold=True, which makes
    merge_pii mark it review_worthy regardless of the precision guards."""
    limit = AGE_REVIEW_OVER_YEARS if over_years is None else over_years
    ents, seen = [], set()

    def _add(a, b, conf, value=None):
        if b <= a or (a, b) in seen:
            return
        seen.add((a, b))
        ents.append(_entity(a, b, "AGE", conf, "regex:age_over",
                            age_over_threshold=True, age_value=value))

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


# --- Engine: GLiNER (poster's HIPAA-identifier detector) -------------------
# Label set mirrors the poster's "GLINER TARGETS: HIPAA IDENTIFIERS" panel.
GLINER_LABELS = [
    "person name", "date", "phone number", "fax number", "email address",
    "social security number", "medical record number", "health plan number",
    "account number", "certificate or license number", "vehicle identifier",
    "device identifier", "url", "ip address", "biometric identifier",
    "geographic location", "organization", "profession", "age",
]
_GLINER_TO_CATEGORY = {
    "person name": "NAME", "date": "DATE_SPECIFIC", "phone number": "CONTACT",
    "fax number": "CONTACT", "email address": "CONTACT",
    "social security number": "IDNUM", "medical record number": "IDNUM",
    "health plan number": "IDNUM", "account number": "IDNUM",
    "certificate or license number": "IDNUM", "vehicle identifier": "IDNUM",
    "device identifier": "IDNUM", "url": "URL", "ip address": "CONTACT",
    "biometric identifier": "IDNUM", "geographic location": "LOC_SPECIFIC",
    "organization": "ORG", "profession": "PROFESSION", "age": "AGE",
}


def _gliner_chunks(text, max_words, overlap_words):
    """Yield (chunk_text, char_offset) windows of <= max_words words, with
    overlap, so GLiNER never has to truncate and boundary entities are covered."""
    words = list(re.finditer(r"\S+", text))
    n = len(words)
    if n == 0:
        return
    step = max(1, max_words - overlap_words)
    i = 0
    while i < n:
        chunk = words[i:i + max_words]
        c_start = chunk[0].start()
        c_end = chunk[-1].end()
        yield text[c_start:c_end], c_start
        if i + max_words >= n:
            break
        i += step


def gliner_scan(gliner_model, text, threshold=GLINER_THRESHOLD,
                max_words=GLINER_MAX_WORDS, overlap_words=GLINER_OVERLAP_WORDS):
    if gliner_model is None:
        return []
    entities = []
    for chunk, offset in _gliner_chunks(text, max_words, overlap_words):
        try:
            preds = gliner_model.predict_entities(chunk, GLINER_LABELS, threshold=threshold)
        except Exception:  # noqa: BLE001
            continue
        for p in preds:
            cat = _GLINER_TO_CATEGORY.get(p["label"], "MISC")
            entities.append(_entity(p["start"] + offset, p["end"] + offset, cat,
                                    float(p.get("score", 0.6)), "gliner"))
    return entities


# --- Engine: Presidio (poster's NER detector) ------------------------------
_PRESIDIO_TO_CATEGORY = {
    "PERSON": "NAME", "PHONE_NUMBER": "CONTACT", "EMAIL_ADDRESS": "CONTACT",
    "US_SSN": "IDNUM", "US_DRIVER_LICENSE": "IDNUM", "US_PASSPORT": "IDNUM",
    "MEDICAL_LICENSE": "IDNUM", "CREDIT_CARD": "IDNUM", "IBAN_CODE": "IDNUM",
    "US_BANK_NUMBER": "IDNUM", "URL": "URL", "IP_ADDRESS": "CONTACT",
    "LOCATION": "LOC_SPECIFIC", "DATE_TIME": "DATE_SPECIFIC", "NRP": "MISC",
    "ORGANIZATION": "ORG",
}


def presidio_scan(analyzer, text):
    if analyzer is None:
        return []
    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception:  # noqa: BLE001
        return []
    entities = []
    for r in results:
        cat = _PRESIDIO_TO_CATEGORY.get(r.entity_type, "MISC")
        entities.append(_entity(r.start, r.end, cat, float(r.score), "presidio"))
    return entities


# --- Stage 3 combinatorial rigidity (sliding window) -----------------------
def combinatorial_scan(text, weak_and_misc_entities, window_tokens, threshold):
    tokens = list(re.finditer(r"\S+", text))
    if not tokens or not weak_and_misc_entities:
        return []
    flags = []
    n = len(tokens)
    for i in range(n):
        w_start = tokens[i].start()
        w_end = tokens[min(i + window_tokens, n) - 1].end()
        contributing = [e for e in weak_and_misc_entities
                        if e["start"] >= w_start and e["end"] <= w_end]
        score = sum(CATEGORY_WEIGHTS.get(e["category"], 0.3) * e["confidence"]
                    for e in contributing)
        diverse = len({e["category"] for e in contributing}) >= 2
        if COMBINATORIAL_REQUIRE_CATEGORY_DIVERSITY and not diverse:
            continue  # two of the same weak signal (e.g. two professions) don't combine
        if score >= threshold and len(contributing) >= 2:
            flags.append({
                "start": min(e["start"] for e in contributing),
                "end": max(e["end"] for e in contributing),
                "category": "COMBINATORIAL", "confidence": min(1.0, score),
                "method": "combinatorial", "contributing": contributing,
            })
    flags.sort(key=lambda e: e["start"])
    merged = []
    for f in flags:
        if merged and f["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], f["end"])
            merged[-1]["confidence"] = max(merged[-1]["confidence"], f["confidence"])
            ids = {id(c) for c in merged[-1]["contributing"]}
            merged[-1]["contributing"].extend(c for c in f["contributing"] if id(c) not in ids)
        else:
            merged.append(f)
    return merged


# ===========================================================================
# PART B -- LOCAL LLM ENSEMBLE (more than one model; votes are aggregated)
# ===========================================================================
PII_SYSTEM_PROMPT = (
    "You are screening a clinical-research transcript for identification risk, "
    "using Kripke's theory of rigid designation: for any span the question is not only "
    "'is this a listed identifier type?' but also 'does this expression fix a unique "
    "individual, on its own or combined with other details?'. Be conservative -- "
    "flag anything that could plausibly contribute to identification. This includes "
    "nuanced quasi-identifiers that are not classic PII: an unusually specific "
    "occupation, achievement, or activity (e.g. 'professional figure skater', "
    "'Olympic swimmer', 'the only heart-transplant patient at my clinic') can single "
    "someone out even with no name -- flag these as MISC. Also flag self-disclosed "
    "SENSITIVE DEMOGRAPHICS -- a speaker's gender / gender identity (e.g. 'I identify "
    "as nonbinary', 'as a trans man') and race / ethnicity (e.g. \"I'm Black\", 'my "
    "ethnicity is ...') -- as MISC, since these are protected attributes that aid "
    "re-identification; but do NOT flag incidental uses ('black coffee', 'Chinese "
    "food'). A false negative is far worse than a false positive."
)
COMPLIANCE_SYSTEM_PROMPT = (
    "You are auditing whether a researcher completed a required protocol/task step "
    "in a clinical-research transcript. A step can be completed without using its "
    "exact wording (e.g. consent obtained via 'is it okay if I record this?'). "
    "Judge whether expressions in the transcript REFER TO the required action, not "
    "whether they reproduce it verbatim. Be conservative and explain your reasoning."
)


class LocalLLM:
    """One local Ollama model, queried for the judgements no rule can make.

    Deliberately a SINGLE model rather than a voting ensemble. Ollama keeps a limited
    set of models resident, so rotating calls across several large ones forces a disk
    reload on nearly every call; that reload tax, not generation, dominates runtime.
    One warm model is several times faster, and the confidence it reports carries the
    uncertainty that cross-model disagreement used to stand in for."""

    def __init__(self, host, model, timeout):
        self.host = host
        self.model = model
        self.timeout = timeout
        self.available = self._check()

    def _check(self):
        """Confirm the model is actually served before the run starts, so a missing
        pull surfaces immediately instead of as silently absent LLM judgements."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as resp:
                installed = {m["name"] for m in json.loads(resp.read()).get("models", [])}
        except Exception as e:  # noqa: BLE001
            print(f"  [LLM] Ollama not reachable at {self.host} ({e}); LLM disabled.")
            return False
        # Accept "gemma4" against an installed "gemma4:latest" and vice versa.
        base = self.model.split(":")[0]
        if self.model in installed or any(m.split(":")[0] == base for m in installed):
            print(f"  [LLM] active, model: {self.model}")
            return True
        print(f"  [LLM] model {self.model!r} is not installed; LLM disabled. "
              f"Pull it with `ollama pull {base}`.")
        return False

    @property
    def active(self):
        return self.available

    def _ask(self, system, prompt):
        """One JSON-mode call. Returns the parsed object, or None on any failure --
        unreachable server, timeout, or a reply that isn't the JSON we asked for."""
        payload = json.dumps({"model": self.model, "prompt": prompt, "system": system,
                              "format": "json", "stream": False}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(json.loads(resp.read())["response"])
        except Exception:  # noqa: BLE001
            return None

    # ---- PII: span detection ----
    def pii_spans(self, text):
        if not self.active:
            return []
        data = self._ask(PII_SYSTEM_PROMPT, (
            "Passage:\n" + text + "\n\n"
            'Return JSON {"flags":[{"text":"...","start":int,"end":int,'
            '"category":"NAME|IDNUM|CONTACT|URL|ORG|LOC|DATE|TIME|AGE|PROFESSION|MISC",'
            '"reasoning":"...","confidence":0-1}]}. start/end are char offsets into the passage.'
        ))
        out = []
        for f in (data or {}).get("flags", []):
            try:
                out.append(_entity(int(f["start"]), int(f["end"]),
                                   f.get("category", "MISC").upper(),
                                   float(f.get("confidence", 0.5)),
                                   f"llm:{self.model}", reasoning=f.get("reasoning", "")))
            except (KeyError, ValueError, TypeError):
                continue
        return out

    # ---- PII: does the COMBINATION of spans identify someone? ----
    def pii_combinatorial_vote(self, text, spans):
        if not spans or not self.active:
            return None
        listing = "\n".join(f'- {s["categories"][0]}: "{text[s["start"]:s["end"]]}"'
                            for s in spans)
        d = self._ask(PII_SYSTEM_PROMPT, (
            "These spans were flagged in a transcript:\n" + listing + "\n\n"
            "Taken together, do they fix a unique individual within the plausible "
            "population of a clinical study? Return JSON "
            '{"identifies_unique_individual":true|false,"reasoning":"..."}'
        ))
        if d is None:
            return None
        return {"model": self.model,
                "identifies_unique_individual": bool(d.get("identifies_unique_individual")),
                "reasoning": d.get("reasoning", "")}

    # ---- Compliance: was a required protocol step completed? ----
    def compliance_vote(self, transcript, step):
        d = self._ask(COMPLIANCE_SYSTEM_PROMPT, (
            "Transcript:\n" + transcript + "\n\n"
            f'Required step: "{step["step_description"]}"\n'
            f'Required speaker: {step.get("required_speaker", "either")}\n'
            'Did the transcript show this step being completed? Return JSON '
            '{"completed":true|false,"evidence":"quote or \'none\'","confidence":0-1,"reasoning":"..."}'
        )) if self.active else None
        return self._judgement(d, extra=("evidence",))

    # ---- Compliance: open-ended task judge (poster Tier C) ----
    def task_judge_vote(self, transcript, task):
        rubric = task.get("rubric") or task.get("prompt") or task.get("description", "")
        d = self._ask(COMPLIANCE_SYSTEM_PROMPT, (
            "Transcript of a participant's spoken response:\n" + transcript + "\n\n"
            f'Task the participant was asked to complete: "{rubric}"\n'
            'Did the speaker complete the task? Return JSON '
            '{"completed":true|false,"confidence":0-1,"reasoning":"..."}'
        )) if self.active else None
        return self._judgement(d)

    # ---- Compliance: task-agnostic completion judge (no reference/rubric) ----
    def transcript_completion_vote(self, transcript):
        """Used when the .pt carries no task metadata: ask whether the transcript reads
        like a real, completed, coherent spoken response rather than silence, a cut-off
        fragment, or non-responsive noise. Corroborates the deterministic quality check
        on borderline files only."""
        d = self._ask(COMPLIANCE_SYSTEM_PROMPT, (
            "Transcript of a single spoken response from a clinical voice study:\n"
            + transcript + "\n\n"
            "Ignoring what the specific task was, does this read like a COMPLETE, "
            "coherent spoken response (someone actually spoke and finished a thought), "
            "as opposed to silence, a cut-off fragment, or non-responsive noise? "
            'Return JSON {"completed":true|false,"confidence":0-1,"reasoning":"..."}'
        )) if self.active else None
        return self._judgement(d)

    def _judgement(self, d, extra=()):
        """Normalise one completed/confidence reply into the compliance-score shape."""
        if d is None:
            return None
        try:
            confidence = min(1.0, max(0.0, float(d.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5      # unparsable confidence -> maximally undecided
        completed = bool(d.get("completed"))
        out = {"model": self.model, "completed": completed,
               "confidence": round(confidence, 3),
               "score": llm_judgement_score(completed, confidence),
               "reasoning": d.get("reasoning", ""),
               # The model is unsure enough that a human should decide.
               "uncertain": confidence < 0.5}
        for k in extra:
            out[k] = d.get(k, "")
        return out


def llm_judgement_score(completed, confidence):
    """Put one LLM yes/no + confidence onto the same 0-1 axis as every other
    compliance check, banded so it lands on the composite cutoffs exactly like
    Tier A does:

        confident "completed"      -> 0.9 .. 1.0   PASS
        UNSURE, either way         -> 0.5 .. 0.9   REVIEW
        confident "not completed"  -> 0.0 .. 0.5   FAIL (-> review)

    The middle band is the point. A model that says "no" but is not confident about
    it has not found a failure -- it has failed to decide, which is a human's job.
    Letting low confidence slide into the fail band would report a guess as a hard
    compliance failure. With one model this reported confidence carries the
    uncertainty that a spread of votes across several models used to express, and
    it does so honestly rather than as a function of how many models are installed.
    """
    confidence = min(1.0, max(0.0, confidence))
    # Signed evidence: +confident yes .. 0 undecided .. -confident no.
    e = confidence if completed else -confidence
    t = LLM_CONFIDENT_AT
    if e >= t:                                            # confident yes -> PASS band
        return round(Q_PASS_CUTOFF + (1.0 - Q_PASS_CUTOFF) * (e - t) / max(1e-9, 1.0 - t), 3)
    if e <= -t:                                           # confident no  -> FAIL band
        return round((Q_FAIL_CUTOFF - 0.001) * (e + 1.0) / max(1e-9, 1.0 - t), 3)
    span = Q_PASS_CUTOFF - Q_FAIL_CUTOFF - 0.001          # undecided    -> REVIEW band
    return round(Q_FAIL_CUTOFF + span * (e + t) / (2 * t), 3)


# ===========================================================================
# PART C -- PII merge / cross-validation across engines
# ===========================================================================
def merge_pii(entities, text):
    """Cluster overlapping detections from all engines; agreement across
    distinct engines/models boosts confidence, single-engine hits are kept
    but flagged for review (poster: 'Disagreements surface to reviewers')."""
    entities = sorted(entities, key=lambda e: e["start"])
    clusters = []
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
        if not (HARD_GATE_REQUIRE_CORROBORATION and PRECISION_MODE):
            hard_gate_eligible = is_strong
        elif not is_strong:
            hard_gate_eligible = False
        elif best["category"] == "NAME":
            hard_gate_eligible = _name_hard_gate_eligible(
                text[start:end], start, text, methods, engines, score)
        else:
            # IDNUM / CONTACT / URL: already format-validated in postprocess; trust a
            # pattern match, cross-engine agreement, or very high confidence.
            hard_gate_eligible = (
                len(engines) >= 2
                or score >= STRONG_PII_HARD_GATE_CONFIDENCE
                or "regex" in methods
            )
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
            review_worthy = True             # identifier, real-name signal, or nuanced quasi-id
        elif contextual_or_id:
            review_worthy = cross_validated  # ORG/LOC/DATE/ID: only if engines agree
        else:
            review_worthy = False            # AGE/TIME/PROFESSION/plain-MISC alone: informational
        # High-recall screening: flag ANY real detection for human review. Only the
        # classic lone common-word NAME false positive (no name signal) stays quiet,
        # unless RECALL_FLAG_ALL_PII forces even those. Minimizes false negatives.
        if HIGH_RECALL:
            lone_common_name = (best_cat == "NAME" and not hard_gate_eligible)
            if RECALL_FLAG_ALL_PII or not lone_common_name:
                review_worthy = True
        merged.append({
            "start": start, "end": end, "text": text[start:end],
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
        })
    return merged


def build_masked_preview(text, entities):
    """Reviewer-convenience preview only. Never written back to the .pt --
    nothing is auto-redacted (poster)."""
    preview = text
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        tag = f"[{ent['canonical_labels'][0]}]"
        preview = preview[: ent["start"]] + tag + preview[ent["end"]:]
    return preview


# ===========================================================================
# PART D -- TASK COMPLIANCE
# ===========================================================================
_WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize_words(text):
    return _WORD_RE.findall(text.lower())


def _align_counts(ref, hyp):
    """Levenshtein alignment of two word lists, broken down by EDIT TYPE.

    Returns (substitutions, deletions, insertions, hits). The plain edit distance
    (= sub + del + ins) is all WER needs, but Tier A needs the breakdown: deletions
    and substitutions mean reference words that were NOT said, while insertions are
    only extra words on top. Conflating them is what makes a compliant read with a
    false start look like an off-script read."""
    n, m = len(ref), len(hyp)
    # Each cell carries (cost, sub, del, ins, hit) for the best path reaching it.
    row = [(j, 0, 0, j, 0) for j in range(m + 1)]
    for i in range(1, n + 1):
        prev, row = row, [(i, 0, i, 0, 0)] + [None] * m
        ref_w = ref[i - 1]
        for j in range(1, m + 1):
            c, s, d, ins, h = prev[j - 1]
            diag = (c, s, d, ins, h + 1) if ref_w == hyp[j - 1] else (c + 1, s + 1, d, ins, h)
            c, s, d, ins, h = prev[j]
            drop = (c + 1, s, d + 1, ins, h)          # reference word not said
            c, s, d, ins, h = row[j - 1]
            add = (c + 1, s, d, ins + 1, h)           # extra word not in the reference
            row[j] = min(diag, drop, add, key=lambda t: t[0])
    return row[m][1], row[m][2], row[m][3], row[m][4]


def alignment_stats(reference, hypothesis):
    """Tier A measurements for one reference prompt against one transcript.

    ``wer``      -- classic word error rate, (sub + del + ins) / len(reference).
    ``coverage`` -- fraction of reference words actually matched, 1 - (sub + del)/len(ref).
                    Insertion-blind: extra speech cannot lower it, so it answers
                    "was the script read?" where WER answers "how cleanly?".
    """
    ref = _normalize_words(reference)
    hyp = _normalize_words(hypothesis)
    if not ref:
        return {"wer": 0.0 if not hyp else 1.0, "coverage": 1.0, "errors": len(hyp),
                "substitutions": 0, "deletions": 0, "insertions": len(hyp),
                "hits": 0, "ref_words": 0, "hyp_words": len(hyp)}
    sub, dele, ins, hits = _align_counts(ref, hyp)
    return {"wer": (sub + dele + ins) / len(ref),
            "coverage": hits / len(ref),
            "errors": sub + dele + ins,
            "substitutions": sub, "deletions": dele, "insertions": ins,
            "hits": hits, "ref_words": len(ref), "hyp_words": len(hyp)}


def word_error_rate(reference, hypothesis):
    """Levenshtein-based WER (poster Tier A). No external dependency."""
    return alignment_stats(reference, hypothesis)["wer"]


def tier_a_score(wer):
    """Map a word error rate onto a compliance score in [0,1] using the Tier A
    thresholds, so the generic composite cutoffs (Q_PASS_CUTOFF / Q_FAIL_CUTOFF)
    land on the WER values that actually mean pass / review / fail for read speech.
    Piecewise linear and monotonically decreasing, with each band kept strictly
    inside its cutoff so a boundary WER never lands on the wrong side."""
    wer = max(0.0, wer)
    if wer <= TIER_A_WER_REVIEW:
        frac = wer / TIER_A_WER_REVIEW if TIER_A_WER_REVIEW > 0 else 1.0
        return round(1.0 - (1.0 - Q_PASS_CUTOFF) * frac, 3)          # 1.0 .. 0.9  -> PASS
    if wer <= TIER_A_WER_FAIL:
        span = max(1e-9, TIER_A_WER_FAIL - TIER_A_WER_REVIEW)
        frac = (wer - TIER_A_WER_REVIEW) / span
        return round(Q_PASS_CUTOFF - (Q_PASS_CUTOFF - Q_FAIL_CUTOFF - 0.001) * frac, 3)
    span = max(1e-9, 1.0 - TIER_A_WER_FAIL)                          # 0.9 .. 0.5  -> REVIEW
    frac = min(1.0, (wer - TIER_A_WER_FAIL) / span)
    return round(max(0.0, (Q_FAIL_CUTOFF - 0.001) * (1.0 - frac)), 3)  # 0.5 .. 0.0 -> FAIL


# Non-lexical vocalizations an ASR system emits for hesitation/backchannel. These
# carry no task content; a transcript made only of them means nothing was said.
# (Deliberately excludes real words like "yeah"/"okay"/"no" -- those are content.)
FILLER_TOKENS = {
    "uh", "um", "umm", "uhh", "uhm", "er", "err", "erm", "ah", "ahh", "hmm", "hm",
    "mm", "mmm", "mhm", "mmhm", "mhmm", "mmhmm", "uhhuh", "uh-huh", "huh", "eh",
    "hmmm", "mmk", "shh", "ugh", "oof", "ooh", "aah",
}
# Bracketed / angle-bracketed ASR non-speech annotations: [noise], (coughing),
# <unk>, [inaudible], {laughter}, etc. Stripped before counting words.
_NONSPEECH_MARKER_RE = re.compile(r"[\[\(<\{][^\]\)>\}]*[\]\)>\}]")


def assess_transcript_quality(transcript):
    """Task-compliance signal that needs NO task metadata or reference text.

    Returns a dict with a compliance-style score in [0,1] and a hard_fail flag.
    A degenerate transcript (empty / non-speech only / filler only) is a genuine
    compliance failure: whatever the task was, no valid spoken response exists to
    have completed it. Near-degenerate transcripts route to review."""
    raw = transcript or ""
    stripped = raw.strip()
    no_markers = _NONSPEECH_MARKER_RE.sub(" ", stripped)
    words = re.findall(r"[A-Za-z0-9']+", no_markers.lower())
    n_words = len(words)
    # content = tokens that carry task meaning (filler vocalizations removed).
    content = [w for w in words if w not in FILLER_TOKENS]
    n_content = len(content)
    filler_ratio = (n_words - n_content) / n_words if n_words else 0.0

    def result(status, score, confidence, hard_fail):
        return {"method": "transcript_quality", "status": status,
                "score": round(score, 3), "confidence": confidence,
                "hard_fail": hard_fail, "completed": (not hard_fail) and score >= Q_PASS_CUTOFF,
                "n_words": n_words, "n_content_words": n_content,
                "filler_ratio": round(filler_ratio, 3)}

    if not stripped or n_words == 0:
        return result("empty", 0.0, 0.99, True)
    if n_content == 0:
        # Only non-speech markers and/or filler vocalizations survived.
        return result("non_speech_only", 0.0, 0.97, True)
    if n_content < COMPLIANCE_HARD_FAIL_MIN_CONTENT_WORDS:
        return result("no_content", 0.0, 0.95, True)
    if n_content < COMPLIANCE_REVIEW_MIN_CONTENT_WORDS:
        return result("too_short", 0.6, 0.7, False)
    if n_content < COMPLIANCE_SHORT_RESPONSE_WORDS and filler_ratio > COMPLIANCE_MAX_FILLER_RATIO:
        return result("mostly_filler", 0.6, 0.7, False)
    return result("ok", 1.0, 0.8, False)


def _recall_compliance_override(transcript, modality, quality):
    """HIGH_RECALL only. Route to REVIEW (never a hard fail) the non-compliance the
    text-level tiers structurally miss: unexpected speech on an ACOUSTIC task, and
    thin / likely-incomplete OPEN free-speech. Returns a review-band quality dict
    (score 0.6, between Q_FAIL_CUTOFF and Q_PASS_CUTOFF) so composite_score flags
    compliance for review; otherwise returns the original `quality` unchanged."""
    q = assess_transcript_quality(transcript)
    ncw = q.get("n_content_words", 0)
    if (modality == "acoustic" and RECALL_FLAG_ACOUSTIC
            and ncw > RECALL_ACOUSTIC_MAX_CONTENT_WORDS):
        return dict(q, applies=True, status="acoustic_unexpected_speech",
                    score=0.6, confidence=0.6, hard_fail=False,
                    method="recall:acoustic_extra_speech")
    if modality == "open" and (RECALL_FLAG_ALL_OPEN or ncw < RECALL_OPEN_MIN_CONTENT_WORDS):
        return dict(q, applies=True, status="open_needs_review",
                    score=0.6, confidence=0.55, hard_fail=False,
                    method="recall:open_review")
    return quality


# ---------------------------------------------------------------------------
# Task identity + modality, read from the .pt FILENAME (BIDS `task-<Name>` field)
# ---------------------------------------------------------------------------
# A task's modality decides whether/how compliance can be judged from text alone.
# Classified by substring patterns on the normalized task token so it is robust to
# the many per-subtype variants (diadochokinesis-PA, cape-V-sentences-3, ...).
_ACOUSTIC_TASK_PATTERNS = (
    "diadochokinesis", "prolongedvowel", "maximumphonation", "phonation", "loudness",
    "glide", "hightolow", "lowtohigh", "respiration", "cough", "breath",
    "longsounds", "sillysounds", "noisysounds",
)
_SCRIPTED_TASK_PATTERNS = (
    "capevsentences", "rainbow", "caterpillar", "harvardsentence", "passage", "sentence",
)


def _norm_task(s):
    """Alphanumeric-only, lowercased -- so 'Cape-V-Sentences', 'cape_v_sentences' and
    'cape-V-sentences-(v2)-3' all normalize compatibly for matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def task_key_from_filename(filename):
    """Extract the BIDS-style task token from a .pt filename, e.g.
    'sub-1_ses-2_task-Word-color-Stroop_features.pt' -> 'Word-color-Stroop'."""
    if not filename:
        return None
    stem = Path(filename).name
    stem = stem[:-3] if stem.endswith(".pt") else stem
    m = re.search(r"task-([^_]+)", stem)
    return m.group(1) if m else None


def classify_modality(task_token):
    """acoustic (non-verbal) / scripted (read a reference) / open (free speech)."""
    n = _norm_task(task_token)
    if not n:
        return None
    if any(p in n for p in _ACOUSTIC_TASK_PATTERNS):
        return "acoustic"
    if any(p in n for p in _SCRIPTED_TASK_PATTERNS):
        return "scripted"
    return "open"


def _clean_prompt(p):
    """Strip the recording boilerplate that trails some reference prompts, e.g.
    'The blue spot is on the key again. When you are ready, press "record".'"""
    p = re.sub(r"when you are ready.*$", "", p or "", flags=re.IGNORECASE)
    return p.strip().strip("“”\"'").strip()


class ComplianceChecker:
    def __init__(self, protocol, llm, task_reference=None):
        self.tasks = (protocol or {}).get("tasks", {})
        self.steps = (protocol or {}).get("protocol_steps", [])
        self.llm = llm
        self.task_reference = task_reference or {}
        # normalized-key index for tolerant filename matching (handles subtype variants)
        self._ref_index = {k: _norm_task(k) for k in self.task_reference}

    def _resolve_reference_keys(self, token):
        """Pick the reference key(s) for a task token, MOST SPECIFIC FIRST.

        Returns (keys, ambiguous). Plain bidirectional prefix matching is wrong here:
        it pulls in SIBLING VARIANTS, because one task's normalized name is often a
        prefix of a different task's ('diadochokinesisPA' is a prefix of
        'diadochokinesisPataka'; 'picturedescription' of 'picturedescriptionoption2').
        That silently mixes another task's prompt/instruction into this file's
        compliance test. So:

          1. an EXACT normalized match wins outright -- nothing else is considered;
          2. otherwise prefer the LONGEST key that is a prefix of the token (the most
             specific reference that still describes this task, e.g. token
             'Rainbow-Passage' -> key 'rainbow');
          3. only if the token is itself a prefix of several keys (i.e. the filename is
             LESS specific than the reference, so we genuinely cannot tell which
             variant it was) do we return all of them, flagged `ambiguous`.
        """
        nt = _norm_task(token)
        if not nt:
            return [], False
        exact = [k for k, nk in self._ref_index.items() if nk == nt]
        if exact:
            return exact, False
        # Keys that are a prefix of the token -> token is the more specific name.
        prefixes = [k for k, nk in self._ref_index.items() if nk and nt.startswith(nk)]
        if prefixes:
            longest = max(len(self._ref_index[k]) for k in prefixes)
            best = [k for k in prefixes if len(self._ref_index[k]) == longest]
            return best, len(best) > 1
        # Token is a prefix of these keys -> genuinely ambiguous between variants.
        supersets = [k for k, nk in self._ref_index.items() if nk and nk.startswith(nt)]
        return supersets, len(supersets) > 1

    # ---- Resolve the task + MODALITY for a file (filename first, then .pt metadata) ----
    def resolve_task_context(self, filename, task_meta):
        """Identify the task from the .pt filename's `task-<Name>` field, look up its
        reference prompts/instructions, and classify its modality. This is what lets
        the pipeline judge compliance correctly per task type. Returns None if the task
        is unknown (compliance then falls back to transcript quality)."""
        ctx = {"task_id": None, "modality": None, "tier": None, "reference_texts": [],
               "instruction": None, "matched_keys": [], "source": None,
               "ambiguous_match": False, "reference_missing": False}
        token = task_key_from_filename(filename)
        if token:
            ctx["task_id"] = token
            ctx["source"] = "filename"
            ctx["modality"] = classify_modality(token)
            matches, ambiguous = self._resolve_reference_keys(token)
            ctx["matched_keys"] = matches
            ctx["ambiguous_match"] = ambiguous
            for k in matches:
                entry = self.task_reference.get(k, {})
                for p in entry.get("prompts", []):
                    cleaned = _clean_prompt(p)
                    if cleaned:
                        ctx["reference_texts"].append(cleaned)
                if not ctx["instruction"] and entry.get("instructions"):
                    ctx["instruction"] = entry["instructions"].strip()
        # A scripted task whose reference text is missing from the task reference can
        # still be scored if the .pt itself carries one -- use it rather than silently
        # giving up on the word-error-rate check.
        if (ctx["modality"] == "scripted" and not ctx["reference_texts"]
                and task_meta and task_meta.get("reference_text")):
            ctx["reference_texts"] = [task_meta["reference_text"]]
            ctx["source"] = (ctx["source"] or "") + "+metadata_reference"
        # Fallback: a .pt that carries its own task metadata (reference_text / prompt / tier).
        if ctx["modality"] is None and task_meta:
            if task_meta.get("reference_text"):
                ctx.update(modality="scripted", source="metadata",
                           reference_texts=[task_meta["reference_text"]],
                           task_id=ctx["task_id"] or task_meta.get("task_id"))
            elif task_meta.get("tier"):
                t = task_meta["tier"].upper()
                ctx.update(modality={"A": "scripted", "B": "acoustic"}.get(t, "open"),
                           source="metadata", instruction=task_meta.get("prompt"),
                           task_id=ctx["task_id"] or task_meta.get("task_id"))
            elif task_meta.get("prompt"):
                ctx.update(modality="open", source="metadata",
                           instruction=task_meta.get("prompt"),
                           task_id=ctx["task_id"] or task_meta.get("task_id"))
        if ctx["modality"] is None:
            return None
        ctx["tier"] = {"acoustic": "B", "scripted": "A", "open": "C"}[ctx["modality"]]
        # A SCRIPTED task with no reference text cannot be word-error-rate checked, so
        # its compliance falls back to the transcript-quality floor only -- which cannot
        # detect reading the WRONG text. Record it so the gap is visible in the report
        # instead of looking like a clean pass. Fix by adding the task's prompts to the
        # built-in table (PART H) or a --task-reference JSON, or by shipping a
        # reference_text inside the .pt.
        if ctx["modality"] == "scripted" and not ctx["reference_texts"]:
            ctx["reference_missing"] = True
        return ctx

    # ---- Transcript quality (gated by modality) ----
    def check_quality(self, transcript, suppress=False):
        """Deterministic transcript-quality signal. When `suppress` is True (acoustic
        tasks, or tasks whose Tier A/C score is the authority), the raw status is still
        reported but it does NOT contribute a compliance score -- so a near-empty
        transcript for a prolonged-vowel/breath task is never counted as a failure.
        Otherwise it is the compliance signal (a degenerate transcript => failure),
        optionally corroborated by the LLM on borderline files."""
        q = assess_transcript_quality(transcript)
        if suppress:
            return dict(q, applies=False, score=None, confidence=None, hard_fail=False,
                        raw_status=q["status"])
        q = dict(q, applies=True)
        if (ENABLE_LLM_COMPLIANCE_JUDGE and self.llm and self.llm.active
                and not q["hard_fail"] and q["status"] != "ok"):
            vote = self.llm.transcript_completion_vote(transcript)
            if vote:
                q["llm_completion"] = {k: vote[k] for k in
                                       ("model", "completed", "confidence", "uncertain")}
                if not vote["completed"] and vote["confidence"] >= 0.7:
                    q = dict(q, status="llm_incomplete", score=0.3,
                             confidence=vote["confidence"], hard_fail=True)
                elif vote["completed"] and vote["confidence"] >= 0.7:
                    q = dict(q, status="llm_ok", score=1.0,
                             confidence=vote["confidence"], completed=True)
        return q

    # ---- Route a task to its tier and score it (poster Tiers A / B / C) ----
    def assess_task(self, transcript, ctx):
        if not ctx:
            return None
        modality = ctx["modality"]
        if modality == "acoustic":
            result = {"tier": "B", "method": "acoustic", "score": None,
                      "confidence": None, "completed": None,
                      "status": "skipped: acoustic/non-verbal task (requires audio)"}
        elif modality == "scripted" and ctx["reference_texts"]:
            result = self._tier_a(transcript, ctx["reference_texts"])
        elif modality == "open":
            result = self._tier_c(transcript, ctx)
            if result is None:
                return None  # no LLM -> transcript quality carries the open-task signal
        else:
            return None      # scripted w/o reference text -> quality carries it
        result["task_id"] = ctx.get("task_id")
        result["modality"] = modality
        result["required"] = True
        return result

    # ---- Tier A: scripted read, best-matching reference prompt ----
    def _tier_a(self, transcript, reference_texts):
        """Did the participant read the script? Scored from the BEST-matching reference
        prompt (lowest WER, ties broken by highest coverage), then passed through the
        Tier A thresholds and two precision guards. Both guards can only move a file
        towards PASS -- neither can create a flag -- so they trade no recall away."""
        cands = [alignment_stats(r, transcript) for r in reference_texts if r]
        if not cands:
            best = {"wer": 1.0, "coverage": 0.0, "errors": 0, "substitutions": 0,
                    "deletions": 0, "insertions": 0, "hits": 0, "ref_words": 0,
                    "hyp_words": 0}
        else:
            best = min(cands, key=lambda a: (a["wer"], -a["coverage"]))
        score = tier_a_score(best["wer"])
        guard = None
        # Guard 1 -- absolute error floor. One or two misheard words in a short CAPE-V /
        # Harvard sentence is ASR noise, not a failure to read it, however bad the ratio.
        if score < Q_PASS_CUTOFF and best["errors"] < TIER_A_MIN_WORD_ERRORS:
            score, guard = Q_PASS_CUTOFF, "below_min_word_errors"
        # Guard 2 -- reference covered. Nearly every reference word was actually said and
        # the errors are dominated by EXTRA words (false start, restart, an aside to the
        # researcher). The script was demonstrably read; WER is just inflated by insertions.
        elif (score < Q_PASS_CUTOFF and best["coverage"] >= TIER_A_COVERAGE_PASS
                and best["insertions"] > best["substitutions"] + best["deletions"]):
            score, guard = Q_PASS_CUTOFF, "reference_covered"
        return {"tier": "A", "method": "wer_min_over_prompts",
                "wer": round(best["wer"], 3), "coverage": round(best["coverage"], 3),
                "word_errors": best["errors"], "substitutions": best["substitutions"],
                "deletions": best["deletions"], "insertions": best["insertions"],
                "ref_words": best["ref_words"], "hyp_words": best["hyp_words"],
                "precision_guard": guard,
                "score": score, "confidence": 0.9, "completed": score >= Q_PASS_CUTOFF,
                "num_reference_prompts": len(reference_texts)}

    # ---- Tier C: open-ended, local LLM judge (None if no LLM) ----
    def _tier_c(self, transcript, ctx):
        if not (self.llm and self.llm.active):
            return None
        task = {"rubric": ctx.get("instruction") or "",
                "prompt": ctx["reference_texts"][0] if ctx["reference_texts"] else ""}
        vote = self.llm.task_judge_vote(transcript, task)
        if vote is None:
            return None
        return {"tier": "C", "method": "llm_judge", "score": vote["score"],
                "confidence": vote["confidence"], "completed": vote["completed"],
                "model": vote["model"], "uncertain": vote["uncertain"],
                "reasoning": vote.get("reasoning", "")}

    # ---- Protocol-step verification (proposal Section 4.1: keyword/phrase
    #      gazetteer -> sequence check; + the LLM on residual/ambiguous steps) ----
    def check_protocol_steps(self, transcript):
        if not self.steps:
            return None
        step_results = []
        for step in self.steps:
            # Stage 1 - keyword/phrase gazetteer
            hit, method, conf, evidence = self._keyword_match(transcript, step)
            # LLM on residual / ambiguous steps
            llm_vote = None
            if not hit and self.llm and self.llm.active:
                llm_vote = self.llm.compliance_vote(transcript, step)
                if llm_vote and llm_vote["completed"]:
                    hit, method, conf = True, "llm", llm_vote["confidence"]
                    evidence = _first_evidence(llm_vote)
            if hit:
                status = "protocol_step_completed"
            elif llm_vote and llm_vote.get("uncertain"):
                # The model said "not completed" but was not confident about it -- the
                # step is arguably present. (This is the single-model stand-in for what
                # used to be cross-model disagreement.)
                status = "protocol_step_attempted"
            elif step.get("required", True):
                status = "protocol_step_absent"
            else:
                status = "protocol_step_optional_absent"
            step_results.append({
                "step_id": step["step_id"], "status": status, "method": method,
                "confidence": round(conf, 3), "evidence": evidence,
                "sequence_position": step.get("sequence_position", "any"),
                "required": step.get("required", True),
                "pii_expected": step.get("pii_expected", False),
                "consent_marker": bool(step.get("consent_marker", False)),
                "llm_vote": llm_vote,
            })
        # Stage 3 - sequence verification
        _flag_sequence_deviations(step_results)
        return step_results

    def _keyword_match(self, transcript, step):
        low = transcript.lower()
        for phrase in step.get("canonical_phrases", []):
            if phrase.lower() in low:
                i = low.find(phrase.lower())
                return True, "keyword", 0.9, transcript[i:i + len(phrase)]
        return False, "none", 0.0, None


def _first_evidence(vote):
    """The quote the model pointed at, if it gave a usable one."""
    ev = (vote or {}).get("evidence") or ""
    return ev if vote and vote.get("completed") and ev.lower() != "none" else None


def _flag_sequence_deviations(step_results):
    """Mark completed steps that appear out of prescribed order (proposal Stage 3)."""
    ordered = [(r["sequence_position"], r) for r in step_results
               if r["status"] == "protocol_step_completed"
               and isinstance(r["sequence_position"], int)]
    last = 0
    for pos, r in ordered:
        if pos < last:
            r["status"] = "protocol_deviation"
            r["deviation"] = "out_of_sequence"
        last = max(last, pos)


# ===========================================================================
# PART E -- INTERACTION ANNOTATIONS (mutual reinforcement, proposal 2.3 / 3.3)
# ===========================================================================
def interaction_annotations(pii_entities, step_results, transcript):
    if not step_results:
        return []
    interactions = []
    for step in step_results:
        if step["status"] not in ("protocol_step_completed", "protocol_deviation"):
            # compliance_gap_pii_adjacent: absent step but PII nearby => maybe off-script
            if step["status"] == "protocol_step_absent" and pii_entities:
                interactions.append({
                    "type": "compliance_gap_pii_adjacent", "step_id": step["step_id"],
                    "note": "Required step not found, but PII present in transcript; "
                            "step may have occurred informally/off-script. Priority review.",
                    "priority": "high",
                })
            continue
        # Locate the evidence span for the completed step (best effort)
        ev = step.get("evidence")
        if not ev:
            continue
        pos = transcript.find(ev)
        if pos < 0:
            continue
        ev_start, ev_end = pos, pos + len(ev)
        within = [e for e in pii_entities
                  if e["start"] >= ev_start - 40 and e["end"] <= ev_end + 40]
        for e in within:
            if step.get("pii_expected"):
                itype = "identity_confirmation" if step.get("consent_marker") or \
                    "identity" in step["step_id"] else "pii_within_protocol_step"
                note = "PII used within a step where identifying info is expected (benign)."
                priority = "low"
            else:
                itype = "pii_within_protocol_step"
                note = ("PII appears within a step that should be anonymised -- review "
                        "whether disclosure was inadvertent.")
                priority = "high"
            interactions.append({
                "type": itype, "step_id": step["step_id"],
                "pii_text": e["text"], "pii_labels": e["canonical_labels"],
                "note": note, "priority": priority,
            })
    return interactions


# ===========================================================================
# PART F -- COMPOSITE SCORING (poster Section 5)
# ===========================================================================
def _pii_signals(pii_entities):
    """Returns (s, c, pii_confirmed, pii_review).

    pii_confirmed = a CONFIRMED strong identifier whose canonical label is in
    HARD_GATE_PII_LABELS (the poster's 'certain PII' hard gate -> FAIL).
    pii_review    = at least one review_worthy span (meaningful PII -> REVIEW).
    A lone single-engine weak guess is neither: it is recorded but does not flag
    the file. s/c feed the composite metric; PII never zeroes s on its own unless
    it is a confirmed hard-gate hit."""
    if not pii_entities:
        return 1.0, 0.9, False, False
    pii_confirmed = any(e.get("confirmed") for e in pii_entities)
    pii_review = any(e.get("review_worthy") for e in pii_entities)
    if pii_confirmed:
        return 0.0, 0.95, True, True
    if pii_review:
        # meaningful but unconfirmed -> uncertain score (routes to REVIEW).
        return 0.6, 0.5, False, True
    # only lone/weak detections present: not review-worthy on their own.
    return 0.9, 0.6, False, False


def _compliance_check_score(quality, task_result, step_results):
    """Returns (s, c, missing_required, hard_fail).

    Combines the always-on transcript-quality signal (works with no task metadata)
    with Tier A/C task results and protocol steps when those exist. The soft score
    is the MIN of the available sub-scores (a real failure in any dimension is not
    masked by a good score elsewhere). hard_fail is the OR of the hard signals."""
    scores, confs, missing_required, hard_fail = [], [], False, False
    # Quality contributes only when it "applies" (score is not None). For acoustic
    # tasks, or tasks whose Tier A/C score is authoritative, quality is suppressed
    # (score None) so a naturally near-empty transcript is not counted as a failure.
    if quality is not None and quality.get("score") is not None:
        scores.append(quality["score"])
        confs.append(quality["confidence"])
        hard_fail = hard_fail or bool(quality.get("hard_fail"))
    if task_result and task_result.get("score") is not None:
        scores.append(task_result["score"])
        confs.append(task_result.get("confidence") or 0.5)
    if step_results:
        required = [s for s in step_results if s["required"]]
        if required:
            completed = sum(1 for s in required
                            if s["status"] in ("protocol_step_completed", "protocol_deviation"))
            scores.append(completed / len(required))
            confs.append(sum(s["confidence"] for s in required) / len(required))
            if completed < len(required):
                missing_required = True
    if not scores:
        return None, None, False, False
    return (round(min(scores), 3),
            round(sum(confs) / len(confs), 3), missing_required, hard_fail)


def composite_score(pii_entities, task_result, step_results, quality=None):
    """Q = sum(w*c*s) / sum(w*c);  C = weighted_mean(c) * (1 - lambda*std(c)).

    Three-way decision (poster: Pass / Needs Review / Fail):
      1. FAIL   - a hard gate fired: CONFIRMED hard-gate PII, a required protocol
                  step absent, a degenerate transcript (task not performed), or a
                  compliance score below the fail cutoff.
      2. REVIEW - flagged but recoverable: meaningful-but-unconfirmed PII (redact),
                  a combinatorial re-identification cluster, or a middling
                  compliance/quality score.
      3. PASS   - no PII worth a reviewer's time and a valid, complete response.

    The three-way logic above is computed first, then FAIL is COLLAPSED into
    REVIEW at the very end, so the returned ``decision`` is two-way (pass / review)
    -- nothing is auto-rejected. The pre-collapse severity is still exposed via the
    ``pii_confirmed`` and ``compliance_fail`` flags in the returned dict.
    """
    checks = {}
    # pii_entities is None (not []) when PII detection was SKIPPED entirely
    # (--compliance-only). Then PII contributes no check at all, so Q reflects
    # compliance alone rather than being diluted by a vacuous "clean" PII score.
    pii_skipped = pii_entities is None
    if pii_skipped:
        s_pii = c_pii = None
        pii_confirmed = pii_review = False
    else:
        s_pii, c_pii, pii_confirmed, pii_review = _pii_signals(pii_entities)
        checks["pii"] = {"s": s_pii, "c": c_pii}
    s_comp, c_comp, missing_required, quality_hard_fail = \
        _compliance_check_score(quality, task_result, step_results)
    if s_comp is not None:
        checks["compliance"] = {"s": s_comp, "c": c_comp}

    num = den = 0.0
    weighted_confs, weights = [], []
    for name, v in checks.items():
        w = CHECK_WEIGHTS.get(name, 0.5)
        num += w * v["c"] * v["s"]
        den += w * v["c"]
        weighted_confs.append(v["c"])
        weights.append(w)
    Q = round(num / den, 3) if den else None

    # There may be NO checks at all: under --compliance-only an acoustic task has its
    # quality signal suppressed, Tier B skipped and no protocol steps, so nothing is
    # assessable from text. Q and C are then None (unknown), not 0 (bad).
    if weights:
        mean_c = sum(w * c for w, c in zip(weights, weighted_confs)) / sum(weights)
        if len(weighted_confs) > 1:
            avg = sum(weighted_confs) / len(weighted_confs)
            std = (sum((c - avg) ** 2 for c in weighted_confs) / len(weighted_confs)) ** 0.5
        else:
            std = 0.0
        C = round(mean_c * (1 - CONFIDENCE_LAMBDA * std), 3)
    else:
        C = None

    missing_step = HARD_GATE_MISSING_REQUIRED_STEP and missing_required
    compliance_score_fail = s_comp is not None and s_comp < Q_FAIL_CUTOFF
    compliance_hard = bool(missing_step or quality_hard_fail or compliance_score_fail)
    compliance_review = s_comp is not None and Q_FAIL_CUTOFF <= s_comp < Q_PASS_CUTOFF

    flagged_by_pii = pii_confirmed or pii_review
    flagged_by_compliance = compliance_hard or compliance_review

    if pii_confirmed or compliance_hard:
        decision = "fail"
    elif pii_review or compliance_review:
        decision = "review"
    else:
        decision = "pass"

    # Final collapse: fold FAIL into REVIEW so the headline verdict is two-way --
    # Pass or Review only (nothing is auto-rejected). The underlying severity is
    # still available via pii_confirmed / compliance_fail in the returned dict.
    if decision == "fail":
        decision = "review"

    return {"Q": Q, "C": C, "decision": decision, "checks": checks,
            "pii_skipped": pii_skipped,
            "flagged_by_pii": bool(flagged_by_pii),
            "flagged_by_compliance": bool(flagged_by_compliance),
            # the "serious" (hard-gate) subset of each reason
            "pii_confirmed": bool(pii_confirmed),
            "compliance_fail": bool(compliance_hard)}


# ===========================================================================
# PART G -- PIPELINE ORCHESTRATION
# ===========================================================================
# .pt metadata keys the pipeline will look for to figure out the task, so no
# external protocol spec is required (poster input = "Audio + task metadata").
TASK_NAME_KEYS = ["task_id", "task", "task_name", "prompt_id"]
REFERENCE_KEYS = ["reference_text", "reference", "target_text", "prompt_text",
                  "script", "expected_text"]
PROMPT_KEYS = ["prompt", "task_prompt", "instruction", "rubric",
               "task_description", "description"]
TIER_KEYS = ["tier", "compliance_tier"]


def extract_task_meta(data):
    """Pull whatever task info the .pt carries. All fields optional."""
    def first(keys):
        for k in keys:
            v = data.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None
    tier = first(TIER_KEYS)
    return {
        "task_id": first(TASK_NAME_KEYS),
        "reference_text": first(REFERENCE_KEYS),
        "prompt": first(PROMPT_KEYS),
        "tier": tier.upper() if tier else None,
    }


class Pipeline:
    def __init__(self, args, protocol, task_reference=None):
        self.args = args
        self.protocol = protocol
        self.task_reference = task_reference or {}
        self._load_engines()
        self.llm = LocalLLM(args.ollama_host, args.ollama_model, LLM_TIMEOUT_SECONDS) \
            if args.enable_llm else _NullLLM()
        self.compliance = ComplianceChecker(protocol, self.llm, self.task_reference)

    def _load_engines(self):
        self.nlp = None
        self.name_set = self.place_set = None
        self.gliner = None
        self.presidio = None
        # --compliance-only: none of the PII engines are needed. Skipping the loads is
        # where nearly all the time is saved (spaCy/GLiNER/Presidio startup + per-file
        # inference); task compliance is pure rules + optional LLM.
        if getattr(self.args, "compliance_only", False):
            print("Compliance-only mode: skipping all PII engines.")
            return

        if self.args.enable_spacy and _importable("spacy"):
            try:
                import spacy
                print(f"Loading spaCy '{self.args.spacy_model}' (rigidity-cascade NER)...")
                self.nlp = spacy.load(self.args.spacy_model)
            except Exception as e:  # noqa: BLE001
                print(f"  [spaCy] unavailable ({e}); NER stage skipped.")
        elif self.args.enable_spacy:
            print("  [spaCy] not importable on this interpreter; NER stage skipped.")
        try:
            print("Loading name/place gazetteers...")
            self.name_set = load_name_gazetteer()
            self.place_set = load_place_gazetteer()
        except Exception as e:  # noqa: BLE001
            print(f"  [gazetteer] unavailable ({e}); using empty sets.")
            self.name_set, self.place_set = set(), load_place_gazetteer()

        if self.args.enable_gliner and _importable("gliner"):
            try:
                from gliner import GLiNER
                print(f"Loading GLiNER '{self.args.gliner_model}' (HIPAA identifiers)...")
                self.gliner = GLiNER.from_pretrained(self.args.gliner_model)
            except Exception as e:  # noqa: BLE001
                print(f"  [GLiNER] unavailable ({e}); GLiNER engine skipped.")
        elif self.args.enable_gliner:
            print("  [GLiNER] not importable on this interpreter (skipped).")

        if self.args.enable_presidio and _importable("presidio_analyzer"):
            try:
                from presidio_analyzer import AnalyzerEngine
                print("Loading Presidio analyzer (NER engine)...")
                self.presidio = AnalyzerEngine()
            except Exception as e:  # noqa: BLE001
                print(f"  [Presidio] unavailable ({e}); Presidio engine skipped.")
        elif self.args.enable_presidio:
            print("  [Presidio] not importable on this interpreter (skipped).")

    # ---- PII: all engines in parallel, then cross-validate ----
    def detect_pii(self, text):
        if not text.strip():
            return [], None
        rule = regex_scan(text) + gazetteer_scan(text, self.name_set, self.place_set)
        rule += honorific_scan(text) + profession_scan(text) + rareword_scan(text, RARE_WORD_ZIPF_THRESHOLD)
        rule += rare_role_scan(text) + demographic_scan(text) + age_scan(text)
        if self.nlp is not None:
            rule += ner_scan(self.nlp, text, self.place_set)

        all_engine = rule + gliner_scan(self.gliner, text, getattr(self.args, "gliner_threshold", GLINER_THRESHOLD)) \
            + presidio_scan(self.presidio, text)
        if self.llm.active:
            all_engine += self.llm.pii_spans(text)
        # Precision guards: drop format-invalid ids, reclassify holidays, cut
        # low-confidence soft hits -- applied uniformly across every engine.
        all_engine = postprocess_entities(all_engine, text)

        weak_misc = [e for e in all_engine if e["category"] in WEAK_RIGID | MISC]
        combinatorial = combinatorial_scan(text, weak_misc,
                                            COMBINATORIAL_WINDOW_TOKENS, COMBINATORIAL_THRESHOLD)
        individually_flagged = [e for e in all_engine
                                if CATEGORY_WEIGHTS.get(e["category"], 0.3) * e["confidence"]
                                >= INDIVIDUAL_FLAG_THRESHOLD]
        # Nuanced quasi-identifiers (rare specific roles, self-disclosed gender/race) are
        # low-weight MISC and would miss the individual-flag bar, so include them
        # explicitly -- they carry a deliberate review signal (merge_pii marks them
        # review_worthy).
        quasi_identifiers = [e for e in all_engine
                             if e["method"] == "rare_role"
                             or e["method"].split(":")[0] == "demographic"]
        # Same for a very high age (> AGE_REVIEW_OVER_YEARS): AGE is a weak category,
        # so it would never clear the individual-flag bar -- include it explicitly.
        elderly_ages = [e for e in all_engine if e.get("age_over_threshold")]

        llm_note = f"{sum(1 for e in all_engine if e['method'].startswith('llm'))} " \
                   f"span(s) from the LLM" if self.llm.active else None

        merged = merge_pii(individually_flagged + combinatorial + quasi_identifiers
                           + elderly_ages, text)

        llm_combo = None
        if self.llm.active and merged:
            llm_combo = self.llm.pii_combinatorial_vote(text, merged)
        return merged, {"llm_span_note": llm_note, "llm_combinatorial_vote": llm_combo}

    def analyze(self, transcript, task_meta, filename=None):
        # --compliance-only: skip PII detection outright (no engines, no scan). pii
        # stays None so composite_score omits the PII check instead of scoring it clean.
        if getattr(self.args, "compliance_only", False):
            pii, pii_llm = None, None
        else:
            pii, pii_llm = self.detect_pii(transcript)
        # Identify the task (from the filename) and route compliance by its modality.
        task_ctx = self.compliance.resolve_task_context(filename, task_meta)
        task_result = self.compliance.assess_task(transcript, task_ctx)
        task_has_score = bool(task_result and task_result.get("score") is not None)
        modality = task_ctx["modality"] if task_ctx else None
        # Transcript quality is the compliance signal ONLY for open/unknown tasks with no
        # tier score. Acoustic tasks (a near-empty transcript is expected) and tasks whose
        # Tier A/C score is authoritative suppress it, so they are not falsely failed.
        suppress_quality = (modality == "acoustic") or task_has_score
        quality = self.compliance.check_quality(transcript, suppress=suppress_quality)
        # High-recall: surface likely non-compliance the text-level tiers miss
        # (unexpected speech on acoustic tasks; thin open responses) as REVIEW.
        # Applied even when Tier C DID score the file: the recall net exists to catch what
        # the tiers structurally miss, and the LLM judge is the thing most likely to have
        # been wrong. _compliance_check_score takes the MIN, so this can only add a flag.
        if HIGH_RECALL and modality in ("acoustic", "open"):
            quality = _recall_compliance_override(transcript, modality, quality)
        step_results = self.compliance.check_protocol_steps(transcript)
        interactions = interaction_annotations(pii or [], step_results, transcript)
        composite = composite_score(pii, task_result, step_results, quality)
        # pii is None only under --compliance-only; report the skip explicitly rather
        # than emitting zero counts that would read as "scanned and found nothing".
        pii_block = ({"skipped": True, "entities": [], "num_entities": None,
                      "cross_validated": None, "review_worthy": None,
                      "confirmed": None, "llm": None}
                     if pii is None else
                     {"skipped": False, "entities": pii, "num_entities": len(pii),
                      "cross_validated": sum(1 for e in pii if e["cross_validated"]),
                      "review_worthy": sum(1 for e in pii if e.get("review_worthy")),
                      "confirmed": sum(1 for e in pii if e.get("confirmed")),
                      "llm": pii_llm})
        return {
            "pii": pii_block,
            "compliance": {"task_context": task_ctx, "quality": quality,
                           "task": task_result, "protocol_steps": step_results},
            "interactions": interactions,
            "composite": composite,
            "masked_preview": build_masked_preview(transcript, pii) if pii else transcript,
        }


class _NullLLM:
    """Stand-in when the LLM is off, so callers never branch on None."""
    active = False

    def pii_spans(self, text):
        return []

    def pii_combinatorial_vote(self, *a):
        return None

    def compliance_vote(self, *a):
        return None

    def task_judge_vote(self, *a):
        return None

    def transcript_completion_vote(self, *a):
        return None


# ---------------------------------------------------------------------------
# .pt loading + folder processing
# ---------------------------------------------------------------------------
def load_pt(path):
    """Returns (transcription, task_meta, error). transcription/error mutually exclusive."""
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        return None, None, f"failed to load .pt file: {e}"
    if not isinstance(data, dict) or "transcription" not in data:
        return None, None, "no 'transcription' key found in loaded object"
    transcription = data["transcription"]
    if transcription is None:
        return None, None, "'transcription' key is None"
    if not isinstance(transcription, str):
        transcription = str(transcription)
    return transcription, extract_task_meta(data), None


def process_file(path, pipeline):
    transcription, task_meta, error = load_pt(path)
    if error:
        return {"file": str(path), "error": error}
    result = pipeline.analyze(transcription, task_meta, filename=Path(path).name)
    task_ctx = result["compliance"].get("task_context") or {}
    return {"file": str(path),
            "task_id": task_ctx.get("task_id") or task_meta.get("task_id"),
            "modality": task_ctx.get("modality"),
            "decision": result["composite"]["decision"], **result}


def process_folder(folder, pipeline, recursive, max_files):
    files = sorted(folder.glob("**/*.pt" if recursive else "*.pt"))
    if max_files:
        files = files[:max_files]
    if not files:
        print(f"No .pt files found in {folder} (recursive={recursive})")

    results, errors = [], []
    for i, path in enumerate(files, 1):
        rec = process_file(path, pipeline)
        if "error" in rec:
            errors.append(rec)
            print(f"[{i}/{len(files)}] {path.name}: ERROR - {rec['error']}")
        else:
            results.append(rec)
            comp = rec["composite"]
            print(f"[{i}/{len(files)}] {path.name}: {rec['decision'].upper()} "
                  f"(Q={comp['Q']}, C={comp['C']}, "
                  f"{'PII skipped' if rec['pii'].get('skipped') else str(rec['pii']['review_worthy']) + '/' + str(rec['pii']['num_entities']) + ' PII review-worthy'}, "
                  f"task={rec.get('task_id')}, modality={rec.get('modality')})")

    def tally(key_fn):
        out = {}
        for r in results:
            k = key_fn(r)
            out[k] = out.get(k, 0) + 1
        return out
    summary = {
        "pass": sum(1 for r in results if r["decision"] == "pass"),
        "review": sum(1 for r in results if r["decision"] == "review"),
        "fail": sum(1 for r in results if r["decision"] == "fail"),
        # where the flags come from (these overlap when a file has both)
        "flagged_by_pii": sum(1 for r in results if r["composite"]["flagged_by_pii"]),
        "flagged_by_compliance": sum(1 for r in results if r["composite"]["flagged_by_compliance"]),
        "pii_confirmed": sum(1 for r in results if r["composite"]["pii_confirmed"]),
        "compliance_fail": sum(1 for r in results if r["composite"]["compliance_fail"]),
        "modality_breakdown": tally(lambda r: r.get("modality") or "unknown"),
        "unresolved_tasks": sorted({r.get("task_id") for r in results
                                    if r.get("modality") is None and r.get("task_id")}),
        # Scripted tasks with no reference text get NO word-error-rate check (so reading
        # the wrong text cannot be detected). Surfaced here because it looks like a
        # clean pass otherwise. Fix: add the task's prompts via --task-reference.
        "scripted_without_reference": sorted({
            r.get("task_id") for r in results
            if (r["compliance"].get("task_context") or {}).get("reference_missing")
            and r.get("task_id")}),
        "ambiguous_task_matches": sorted({
            r.get("task_id") for r in results
            if (r["compliance"].get("task_context") or {}).get("ambiguous_match")
            and r.get("task_id")}),
        "errors": len(errors),
        # THE DELIVERABLE: which files a human actually has to look at. Everything
        # above is a count; this is the list. A file is flagged when its decision is
        # anything other than "pass" -- i.e. PII worth redacting, a compliance
        # problem, or both. (`decision` is two-way by the time it gets here: FAIL is
        # collapsed into REVIEW, since nothing is ever auto-rejected. The severity
        # behind each entry is in `reasons` / the per-file record.)
        "flagged_files": [
            {"file": str(r["file"]), "name": Path(r["file"]).name,
             "task_id": r.get("task_id"), "modality": r.get("modality"),
             "reasons": _flag_reasons(r)}
            for r in results if r["decision"] != "pass"],
    }
    return {"folder": str(folder), "num_files": len(files),
            "summary": summary, "files": results, "errors": errors}


def _flag_reasons(rec):
    """Short human-readable tags for WHY a file was flagged, most serious first."""
    cx = rec["composite"]
    reasons = []
    if cx["pii_confirmed"]:
        reasons.append("pii:confirmed")
    elif cx["flagged_by_pii"]:
        reasons.append("pii:review")
    if cx["compliance_fail"]:
        reasons.append("compliance:fail")
    elif cx["flagged_by_compliance"]:
        reasons.append("compliance:review")
    return reasons


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def write_json(report, path):
    Path(path).write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote JSON report to {path}")


def write_csv(report, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "task_id", "modality", "tier", "decision",
                    "flagged_by_pii", "flagged_by_compliance", "pii_confirmed",
                    "compliance_fail", "quality_status", "Q", "C", "num_pii",
                    "pii_review_worthy", "pii_cross_validated", "num_interactions",
                    "pii_labels", "compliance_summary"])
        for r in report["files"]:
            labels = "skipped" if r["pii"].get("skipped") else ";".join(sorted({l for e in r["pii"]["entities"]
                                      for l in e["canonical_labels"]}))
            comp = r["compliance"]
            cx = r["composite"]
            quality = comp.get("quality") or {}
            task = comp.get("task") or {}
            csummary = [f"quality:{quality.get('status')}"]
            if comp["task"]:
                csummary.append(f"task:{task.get('completed')}")
                if task.get("wer") is not None:
                    csummary.append(f"wer:{task.get('wer')}")
                # Tier A triage detail: coverage says whether the script was read at all,
                # word_errors gives the absolute (length-independent) size of the mismatch.
                if task.get("coverage") is not None:
                    csummary.append(f"coverage:{task.get('coverage')}")
                    csummary.append(f"errs:{task.get('word_errors')}/{task.get('ref_words')}w")
                if task.get("precision_guard"):
                    csummary.append(f"guard:{task['precision_guard']}")
            if comp["protocol_steps"]:
                statuses = [s["status"] for s in comp["protocol_steps"]]
                csummary.append(f"steps:{sum(1 for s in statuses if 'completed' in s)}/{len(statuses)}")
            w.writerow([r["file"], r.get("task_id"), r.get("modality"), task.get("tier"),
                        r["decision"], cx["flagged_by_pii"], cx["flagged_by_compliance"],
                        cx["pii_confirmed"], cx["compliance_fail"],
                        quality.get("status"), cx["Q"], cx["C"],
                        r["pii"]["num_entities"], r["pii"].get("review_worthy"),
                        r["pii"]["cross_validated"], len(r["interactions"]),
                        labels, " ".join(csummary)])
        for r in report["errors"]:
            w.writerow([r["file"]] + [""] * 16 + [r["error"]])
    print(f"Wrote CSV summary to {path}")


def print_summary(report):
    s = report["summary"]
    print("\n--- Summary ---")
    print(f"Files processed: {report['num_files']}")
    print(f"  pass:                  {s['pass']}")
    print(f"  review:                {s['review']}  (soft gate -> human review)")
    print(f"  fail:                  {s['fail']}  (hard gate)")
    print("  -- flag breakdown (a file can be flagged by both) --")
    print(f"  flagged_by_pii:        {s['flagged_by_pii']}   (of which confirmed hard-gate PII: {s['pii_confirmed']})")
    print(f"  flagged_by_compliance: {s['flagged_by_compliance']}   (of which hard compliance fail: {s['compliance_fail']})")
    if s.get("modality_breakdown"):
        parts = ", ".join(f"{k}={v}" for k, v in sorted(s["modality_breakdown"].items()))
        print(f"  -- task modality --  {parts}")
    if s.get("unresolved_tasks"):
        print(f"  -- tasks not in reference (compliance via quality only): {s['unresolved_tasks']}")
    if s.get("scripted_without_reference"):
        print(f"\n  !! {len(s['scripted_without_reference'])} SCRIPTED task(s) have NO reference "
              f"text, so no word-error-rate check ran for them -- reading the WRONG text "
              f"cannot be detected. Add their prompts via --task-reference:")
        for t in s["scripted_without_reference"][:10]:
            print(f"       {t}")
        if len(s["scripted_without_reference"]) > 10:
            print(f"       ... and {len(s['scripted_without_reference']) - 10} more")
    if s.get("ambiguous_task_matches"):
        print(f"  !! ambiguous task->reference matches (prompt may be from a sibling "
              f"variant): {s['ambiguous_task_matches']}")
    print(f"  errors:                {s['errors']}")
    print_flagged_files(report)


def print_flagged_files(report, limit=25):
    """Print the pipeline's actual output: the names of the files to review.

    Bounded so a large corpus does not bury the summary -- the complete list is
    always in the JSON report under summary.flagged_files, and --flagged-list
    writes it to its own plain-text file."""
    flagged = report["summary"].get("flagged_files") or []
    print(f"\n--- Flagged files ({len(flagged)}) ---")
    if not flagged:
        print("  none -- every file passed.")
        return
    width = max(len(f["name"]) for f in flagged[:limit])
    for f in flagged[:limit]:
        print(f"  {f['name']:<{width}}  {', '.join(f['reasons']) or 'flagged'}")
    if len(flagged) > limit:
        print(f"  ... and {len(flagged) - limit} more "
              f"(full list: summary.flagged_files in the JSON report, or --flagged-list)")


def write_flagged_list(report, path):
    """Write just the flagged file paths, one per line -- the plain-text handoff to
    whatever does the redaction/review next (xargs, a work queue, a spreadsheet)."""
    flagged = report["summary"].get("flagged_files") or []
    Path(path).write_text("".join(f["file"] + "\n" for f in flagged))
    print(f"Wrote {len(flagged)} flagged file path(s) to {path}")


# ---------------------------------------------------------------------------
# Protocol spec
# ---------------------------------------------------------------------------
def load_protocol(path):
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        print(f"  [protocol] '{path}' not found; compliance runs task-only/empty.")
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        print(f"  [protocol] failed to parse '{path}' ({e}); continuing without it.")
        return {}


def resolve_task_reference_path(path):
    """Absolute path of the task reference JSON. A relative path resolves NEXT TO THIS
    SCRIPT, not against the working directory -- task_reference.json is a companion of
    the script, so it must be found however the script is invoked."""
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parent / p


def load_task_reference(path):
    """Load the task -> {instructions, prompts} table that drives filename-based task
    compliance.

    This is the companion file the pipeline checks each .pt against: the `task-<Name>`
    field of a filename is looked up here to learn the task's modality and, for a
    scripted task, the reference text the participant was supposed to read.

    A configured path that is missing or unparsable STOPS the run. It is not treated
    as "no reference": without this table every scripted task silently loses its Tier A
    word-error-rate check, so reading the WRONG script stops being detectable and the
    files sail through as clean passes. That failure is invisible in the output, so it
    is refused up front instead. To run deliberately without it, set TASK_REFERENCE = ""
    (or pass --no-task-reference) -- compliance then falls back to transcript quality."""
    if not path:
        print("  [task-reference] disabled -- filename-based task compliance is OFF; "
              "compliance falls back to transcript quality only.")
        return {}
    p = resolve_task_reference_path(path)
    if not p.is_file():
        raise SystemExit(
            f"Task reference not found -- stopping.\n"
            f"  looked for: {p}\n"
            f"'{Path(p).name}' is the companion file this script checks each .pt against; "
            f"keep it in the same folder as the script.\n"
            f"Set TASK_REFERENCE in the CONFIG block (top of this file) to point elsewhere, "
            f"or set it to \"\" to run without filename-based task compliance.")
    try:
        ref = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Task reference {p} could not be parsed -- stopping.\n  {e}")
    if not isinstance(ref, dict) or not ref:
        raise SystemExit(f"Task reference {p} is empty or is not a JSON object -- stopping.")
    print(f"  [task-reference] loaded {len(ref)} task definitions from {p.name}")
    return ref


# ---------------------------------------------------------------------------
# Self-test (synthetic data only -- never reads real transcripts)
# ---------------------------------------------------------------------------
def run_selftest(args):
    print("Running self-test on SYNTHETIC .pt files (no real data touched)...")
    checks = {}

    # ---- Part A: ZERO-CONFIG (no protocol spec). Task info comes from the .pt. ----
    print("\n[Part A] Zero-config run -- task metadata read from each .pt, no spec.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torch.save({"transcription": "My name is John Smith, SSN 123-45-6789, "
                                     "email john.smith@example.com."}, tmp / "strong_pii.pt")
        torch.save({"transcription": "The weather was pleasant and the tea was warm."},
                   tmp / "clean.pt")
        torch.save({"transcription": "The 84-year-old retired engineer admitted last "
                                     "Tuesday described his symptoms to the nurse in Ward 3."},
                   tmp / "combinatorial.pt")
        # scripted task: reference text travels WITH the file -> Tier A, no spec needed
        torch.save({"transcription": "the cat sat on a mat", "task": "read_sentence",
                    "reference_text": "the cat sat on the mat"}, tmp / "scripted_task.pt")
        # open-ended task: only a prompt travels with the file -> Tier C
        torch.save({"transcription": "My name is Jane and my favorite show is The Office.",
                    "task": "favorite_show",
                    "prompt": "State your name and your favorite show."}, tmp / "open_task.pt")
        # DEGENERATE transcripts -> genuine compliance FAILURES, no metadata needed.
        torch.save({"transcription": ""}, tmp / "empty.pt")
        torch.save({"transcription": "uh um uh... [inaudible]"}, tmp / "nonspeech.pt")
        torch.save({"transcription": "yeah"}, tmp / "tooshort.pt")

        args.protocol = None
        pipeline = Pipeline(args, {})
        report = process_folder(tmp, pipeline, False, None)
        print_summary(report)
        by = {Path(r["file"]).name: r for r in report["files"]}
        checks.update({
            "A: strong_pii.pt -> REVIEW, PII-confirmed (real SSN + email)":
                by["strong_pii.pt"]["decision"] == "review"
                and by["strong_pii.pt"]["composite"]["flagged_by_pii"]
                and by["strong_pii.pt"]["composite"]["pii_confirmed"],
            "A: clean.pt -> pass, no PII, valid transcript":
                by["clean.pt"]["pii"]["num_entities"] == 0
                and by["clean.pt"]["decision"] == "pass",
            "A: weak-only combinatorial file -> REVIEW, not confirmed, not failed":
                by["combinatorial.pt"]["decision"] == "review"
                and not by["combinatorial.pt"]["composite"]["pii_confirmed"],
            "A: combinatorial.pt -> PII flagged":
                by["combinatorial.pt"]["pii"]["num_entities"] > 0,
            "A: empty.pt -> REVIEW, hard compliance fail (task not performed)":
                by["empty.pt"]["decision"] == "review"
                and by["empty.pt"]["composite"]["flagged_by_compliance"]
                and by["empty.pt"]["composite"]["compliance_fail"]
                and not by["empty.pt"]["composite"]["flagged_by_pii"],
            "A: non-speech-only transcript -> REVIEW, hard compliance fail":
                by["nonspeech.pt"]["decision"] == "review"
                and by["nonspeech.pt"]["composite"]["compliance_fail"],
            "A: too-short response -> REVIEW by compliance (not a hard fail)":
                by["tooshort.pt"]["decision"] == "review"
                and by["tooshort.pt"]["composite"]["flagged_by_compliance"]
                and not by["tooshort.pt"]["composite"]["compliance_fail"],
            "A: scripted_task.pt -> Tier A from .pt metadata (no spec)":
                (by["scripted_task.pt"]["compliance"]["task"] or {}).get("tier") == "A",
            "A: scripted_task.pt -> WER ~0.17 (one sub in 6 words)":
                abs((by["scripted_task.pt"]["compliance"]["task"] or {}).get("wer", -1)
                    - (1 / 6)) < 0.01,
            "A: open_task.pt -> routed to Tier C (open modality); quality carries it w/o LLM":
                (by["open_task.pt"]["compliance"]["task_context"] or {}).get("tier") == "C"
                and (by["open_task.pt"]["compliance"]["task_context"] or {}).get("modality") == "open",
            "A: no protocol steps run without a spec":
                by["clean.pt"]["compliance"]["protocol_steps"] is None,
            "A: every file now gets a transcript-quality signal":
                all(r["compliance"]["quality"] is not None for r in report["files"]),
        })

    # ---- Part B: OPTIONAL spec enables interview-style protocol-step checking. ----
    print("\n[Part B] With an optional protocol spec -> protocol-step checking.")
    protocol = {
        "protocol_steps": [
            {"step_id": "consent", "step_description": "Obtain verbal consent before beginning.",
             "required_speaker": "researcher", "sequence_position": 1, "required": True,
             "canonical_phrases": ["is it okay if I record this", "do you consent"],
             "pii_expected": False, "consent_marker": True},
            {"step_id": "confirm_identity", "step_description": "Confirm participant identity.",
             "required_speaker": "researcher", "sequence_position": 2, "required": True,
             "canonical_phrases": ["can you confirm your name", "state your date of birth"],
             "pii_expected": True},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        torch.save({"transcription": "Before we start, is it okay if I record this? "
                                     "Great. Can you confirm your name for me?"},
                   tmp / "interview.pt")
        pipeline = Pipeline(args, protocol)
        report = process_folder(tmp, pipeline, False, None)
        steps = report["files"][0]["compliance"]["protocol_steps"]
        checks["B: consent step completed from canonical phrase"] = \
            any(s["step_id"] == "consent" and "completed" in s["status"] for s in (steps or []))

    # ---- Part C: precision guards (deterministic, engine-independent). ----
    print("\n[Part C] Precision guards -- false-positive reduction.")
    # Structured-identifier validation: a word-only "identifier" is dropped.
    txt = "cough Easter 123-45-6789"
    ents = [_entity(0, 5, "IDNUM", 0.8, "gliner"),          # 'cough' -> invalid id, drop
            _entity(6, 12, "LOC_SPECIFIC", 0.8, "gliner"),  # 'Easter' -> reclassify to DATE
            _entity(13, 24, "IDNUM", 0.95, "regex")]        # real SSN -> keep
    post = postprocess_entities(ents, txt)
    kept = {(txt[e["start"]:e["end"]], e["category"]) for e in post}
    checks["C: word-only 'cough' IDNUM dropped"] = ("cough", "IDNUM") not in kept
    checks["C: 'Easter' reclassified away from LOCATION -> DATE"] = \
        ("Easter", "DATE_PARTIAL") in kept and ("Easter", "LOC_SPECIFIC") not in kept
    checks["C: valid SSN kept as IDNUM"] = ("123-45-6789", "IDNUM") in kept
    checks["C: validator basics"] = (
        not _valid_structured_identifier("cough", "IDNUM")
        and _valid_structured_identifier("123-45-6789", "IDNUM")
        and _valid_structured_identifier("a@b.com", "CONTACT")
        and not _valid_structured_identifier("hello", "CONTACT"))
    # GLiNER windowing: long text is chunked (no silent truncation) with correct offsets.
    long_text = " ".join(f"w{i}" for i in range(1000))          # 1000 words >> 384 cap
    chunks = list(_gliner_chunks(long_text, GLINER_MAX_WORDS, GLINER_OVERLAP_WORDS))
    checks["C: long text is split into multiple GLiNER chunks"] = len(chunks) >= 3
    checks["C: every chunk is within the token cap"] = \
        all(len(c.split()) <= GLINER_MAX_WORDS for c, _ in chunks)
    checks["C: chunk offsets map back to the original text exactly"] = \
        all(long_text[off:off + len(c)] == c for c, off in chunks)
    checks["C: chunks cover the whole transcript (start..end)"] = \
        chunks[0][1] == 0 and (chunks[-1][1] + len(chunks[-1][0])) == len(long_text)
    # Decision is 3-way (pass/review/fail); reason split into PII vs compliance.
    ssn = merge_pii([_entity(0, 11, "IDNUM", 0.95, "regex")], "123-45-6789")
    ssn_cx = composite_score(ssn, None, None)
    checks["C: real SSN -> REVIEW, by PII, confirmed"] = (
        ssn_cx["decision"] == "review" and ssn_cx["flagged_by_pii"]
        and ssn_cx["pii_confirmed"] and not ssn_cx["flagged_by_compliance"])
    # A failed scripted task -> FAIL, flagged by compliance only.
    bad_task = {"tier": "A", "score": 0.1, "confidence": 0.9, "completed": False, "required": True}
    ct = composite_score([], bad_task, None)
    checks["C: failed task -> REVIEW, by COMPLIANCE only"] = (
        ct["decision"] == "review" and ct["flagged_by_compliance"]
        and ct["compliance_fail"] and not ct["flagged_by_pii"])

    # ---- Transcript-quality compliance (works with NO task metadata) ----
    q_empty = assess_transcript_quality("")
    q_nonspeech = assess_transcript_quality("uh um [noise] (coughing)")
    q_short = assess_transcript_quality("yeah okay")
    q_ok = assess_transcript_quality("the cat sat on the mat quietly this afternoon")
    checks["C: empty transcript -> hard_fail quality"] = \
        q_empty["status"] == "empty" and q_empty["hard_fail"] and q_empty["score"] == 0.0
    checks["C: non-speech/filler-only -> hard_fail quality"] = \
        q_nonspeech["status"] == "non_speech_only" and q_nonspeech["hard_fail"]
    checks["C: short response -> review-band quality (not hard fail)"] = \
        (not q_short["hard_fail"]) and Q_FAIL_CUTOFF <= q_short["score"] < Q_PASS_CUTOFF
    checks["C: substantive transcript -> ok quality, passes"] = \
        q_ok["status"] == "ok" and (not q_ok["hard_fail"]) and q_ok["score"] >= Q_PASS_CUTOFF
    checks["C: empty transcript -> composite REVIEW, hard compliance fail"] = (
        composite_score([], None, None, q_empty)["decision"] == "review"
        and composite_score([], None, None, q_empty)["compliance_fail"])
    checks["C: ok transcript, no PII -> composite PASS"] = \
        composite_score([], None, None, q_ok)["decision"] == "pass"

    # NAME precision -- the main false-positive lever.
    # (a) A lone common word tagged NAME by NER models, no name context: even when
    #     TWO engines share the false positive it must NOT flag the file.
    will = merge_pii([_entity(0, 4, "NAME", 0.8, "ner"),
                      _entity(0, 4, "NAME", 0.75, "gliner")], "Will you sit down now")
    # (b) The same word WITH a name-introduction cue => a real name.
    will_ctx = merge_pii([_entity(11, 15, "NAME", 0.8, "ner")], "my name is Will and I agree")
    # (c) A multi-token name => real.
    full = merge_pii([_entity(6, 18, "NAME", 0.8, "gliner")], "I met Sarah Connor yesterday.")
    # FLAG_ALL_NAMES (default): even a lone common word tagged NAME with no name signal
    # is now REVIEW-worthy -- every name is flagged. It is still never *confirmed*
    # (no hard-gate eligibility), so it can never auto-fail a file.
    checks["C: common-word NAME -> REVIEW (all names flagged), still not confirmed"] = (
        (not will[0]["hard_gate_eligible"]) and will[0]["review_worthy"] is True
        and composite_score(will, None, None, q_ok)["pii_confirmed"] is False
        and composite_score(will, None, None, q_ok)["decision"] == "review")
    # A real name (cue, or multi-token) is REVIEW by default (redaction queue), NOT
    # a hard fail -- names are redactable PII, not reject-worthy.
    checks["C: NAME with 'my name is' cue -> REVIEW, review-worthy, not confirmed"] = (
        will_ctx[0]["hard_gate_eligible"] and will_ctx[0]["review_worthy"]
        and composite_score(will_ctx, None, None, q_ok)["pii_confirmed"] is False
        and composite_score(will_ctx, None, None, q_ok)["decision"] == "review")
    checks["C: multi-token NAME -> REVIEW, review-worthy, not confirmed"] = (
        full[0]["hard_gate_eligible"] and full[0]["review_worthy"]
        and composite_score(full, None, None, q_ok)["pii_confirmed"] is False
        and composite_score(full, None, None, q_ok)["decision"] == "review")
    # Tightening: ADD NAME back to the gate -> the same real name now hard-FAILS.
    _mod = sys.modules[__name__]
    saved_labels = _mod.HARD_GATE_PII_LABELS
    _mod.HARD_GATE_PII_LABELS = {"NAME", "IDNUM", "CONTACT", "URL"}
    full2 = merge_pii([_entity(6, 18, "NAME", 0.8, "gliner")], "I met Sarah Connor yesterday.")
    gated = composite_score(full2, None, None, q_ok)
    checks["C: adding NAME to gate -> REVIEW, confirmed"] = (
        gated["decision"] == "review" and gated["pii_confirmed"] is True)
    _mod.HARD_GATE_PII_LABELS = saved_labels

    # ---- Nuanced quasi-identifier: rare specific role/activity -> REVIEW ----
    skater_txt = "I am a professional figure skater and I train daily"
    skater = merge_pii(rare_role_scan(skater_txt), skater_txt)
    dev_txt = "we scheduled some professional development this week"
    dev = merge_pii(rare_role_scan(dev_txt), dev_txt)
    checks["C: 'professional figure skater' -> review-worthy MISC (not confirmed)"] = (
        len(skater) >= 1 and skater[0].get("review_worthy") is True
        and skater[0].get("confirmed") is False
        and composite_score(skater, None, None, q_ok)["decision"] == "review"
        and composite_score(skater, None, None, q_ok)["flagged_by_pii"] is True)
    checks["C: 'professional development' -> not flagged (common role word)"] = (len(dev) == 0)

    # ---- Demographic quasi-identifiers: self-disclosed gender & race -> REVIEW ----
    race_txt = "I grew up in a small town and I am Black"
    race = merge_pii(demographic_scan(race_txt), race_txt)
    gender_txt = "to answer your question I identify as nonbinary"
    gender = merge_pii(demographic_scan(gender_txt), gender_txt)
    eth_txt = "my ethnicity is Korean and I moved here in 2001"
    eth = merge_pii(demographic_scan(eth_txt), eth_txt)
    neg_txt = "I drank black coffee and we ate Chinese food last night"
    neg = merge_pii(demographic_scan(neg_txt), neg_txt)
    rainbow_txt = ("when the sunlight strikes raindrops in the air they act as a "
                   "prism and form a rainbow the white light is divided")
    rainbow_demo = demographic_scan(rainbow_txt)
    checks["C: 'I am Black' -> review-worthy MISC race (not confirmed), composite REVIEW"] = (
        len(race) >= 1 and race[0].get("review_worthy") is True
        and race[0].get("confirmed") is False and race[0]["canonical_labels"] == ["MISC"]
        and composite_score(race, None, None, q_ok)["decision"] == "review"
        and composite_score(race, None, None, q_ok)["flagged_by_pii"] is True)
    checks["C: 'identify as nonbinary' -> review-worthy MISC gender"] = (
        len(gender) >= 1 and gender[0].get("review_worthy") is True
        and "demographic" in gender[0]["engines"])
    checks["C: 'my ethnicity is Korean' -> review-worthy MISC"] = (
        len(eth) >= 1 and eth[0].get("review_worthy") is True)
    checks["C: demographic PII never hard-fails (MISC not in HARD_GATE_PII_LABELS)"] = (
        all(e.get("confirmed") is False for e in race + gender + eth)
        and composite_score(race, None, None, q_ok)["decision"] != "fail")
    checks["C: 'black coffee' / 'Chinese food' -> NOT flagged (no self anchor)"] = (len(neg) == 0)
    checks["C: scripted 'white light' passage -> no demographic false positive"] = (len(rainbow_demo) == 0)

    # ---- Filename -> task -> tier routing (engine-independent) ----
    ref = {"rainbow": {"instructions": "Please read the passage out loud.",
                       "prompts": ["When the sunlight strikes raindrops in the air, "
                                   "they act as a prism and form a rainbow."]},
           "maximum-phonation-time": {"instructions": "Hold 'ah' as long as you can.", "prompts": []},
           "free-speech-1": {"instructions": "Answer the question.",
                             "prompts": ["Why are you interested in this study?"]}}
    cc = ComplianceChecker({}, _NullLLM(), task_reference=ref)
    ctx_r = cc.resolve_task_context("sub-1_ses-2_task-rainbow_features.pt", {})
    ctx_m = cc.resolve_task_context("sub-1_ses-2_task-Maximum-Phonation-Time_features.pt", {})
    ctx_f = cc.resolve_task_context("sub-1_ses-2_task-Free-Speech-1_features.pt", {})
    good_read = cc.assess_task("when the sunlight strikes raindrops in the air they act "
                               "as a prism and form a rainbow", ctx_r)
    bad_read = cc.assess_task("i really do not want to do this task today", ctx_r)
    acoustic = cc.assess_task("aaaah", ctx_m)
    checks["C: filename task-rainbow -> scripted Tier A"] = (
        ctx_r and ctx_r["modality"] == "scripted" and ctx_r["tier"] == "A"
        and bool(ctx_r["reference_texts"]))
    checks["C: correct read -> Tier A high score, completed"] = (
        good_read["score"] >= 0.7 and good_read["completed"] is True)
    checks["C: wrong content on scripted task -> Tier A fail -> composite REVIEW"] = (
        bad_read["score"] < 0.5
        and composite_score([], bad_read, None, None)["decision"] == "review"
        and composite_score([], bad_read, None, None)["flagged_by_compliance"] is True)
    checks["C: filename task-Maximum-Phonation-Time -> acoustic Tier B, skipped"] = (
        ctx_m and ctx_m["modality"] == "acoustic" and ctx_m["tier"] == "B"
        and acoustic["score"] is None)
    q_sup = cc.check_quality("aaaah", suppress=True)
    checks["C: acoustic near-empty transcript is NOT a compliance fail"] = (
        q_sup["score"] is None
        and composite_score([], acoustic, None, q_sup)["decision"] == "pass"
        and composite_score([], acoustic, None, q_sup)["compliance_fail"] is False)
    checks["C: filename task-Free-Speech-1 -> open Tier C"] = (
        ctx_f and ctx_f["modality"] == "open" and ctx_f["tier"] == "C")

    # ---- Tier A calibration: WER on read speech is ASR noise until it is not ----
    # These are the dominant false-positive modes on real ASR of clinical read-aloud
    # speech. Each one is a participant who DID read the script.
    capev = {"cape-V-sentences-4": {"instructions": "Read the sentence out loud.",
                                    "prompts": ["We eat eggs every Easter."]}}
    cc_a = ComplianceChecker({}, _NullLLM(), task_reference=capev)
    ctx_a = cc_a.resolve_task_context("sub-1_ses-1_task-Cape-V-sentences-4_features.pt", {})
    one_slip = cc_a.assess_task("we eat eggs every easter", ctx_a)          # perfect
    misheard = cc_a.assess_task("we eat eggs every eastern", ctx_a)         # 1 sub in 5 -> WER 0.2
    mangled = cc_a.assess_task("he eats eggs every eastern", ctx_a)         # 3 subs in 5 -> WER 0.6
    restart = cc_a.assess_task("sorry let me start over we eat eggs every easter", ctx_a)
    off_script = cc_a.assess_task("i do not want to read this one", ctx_a)
    stats_ins = alignment_stats("we eat eggs every easter",
                                "sorry let me start over we eat eggs every easter")

    checks["G: alignment separates insertions from missed reference words"] = (
        stats_ins["coverage"] == 1.0 and stats_ins["insertions"] == 5
        and stats_ins["substitutions"] == 0 and stats_ins["deletions"] == 0
        and stats_ins["wer"] == 1.0)
    checks["G: perfect read -> WER 0, coverage 1, PASS"] = (
        one_slip["wer"] == 0.0 and one_slip["coverage"] == 1.0
        and composite_score([], one_slip, None, None)["decision"] == "pass")
    checks["G: 1 misheard word in a 5-word sentence -> NOT flagged (WER threshold)"] = (
        misheard["wer"] == 0.2 and misheard["word_errors"] == 1
        and misheard["precision_guard"] is None
        and composite_score([], misheard, None, None)["decision"] == "pass")
    checks["G: 3 misheard words in a 5-word sentence -> NOT flagged (min-word-errors guard)"] = (
        mangled["wer"] == 0.6 and mangled["word_errors"] == 3
        and mangled["precision_guard"] == "below_min_word_errors"
        and composite_score([], mangled, None, None)["decision"] == "pass")
    checks["G: false start before a correct read -> flagged while the coverage guard is OFF"] = (
        restart["wer"] >= 1.0 and restart["coverage"] == 1.0
        and restart["precision_guard"] is None
        and composite_score([], restart, None, None)["decision"] == "review")
    # ...and the guard does what it claims when a study's labels call for it.
    _saved_cov, globals()["TIER_A_COVERAGE_PASS"] = TIER_A_COVERAGE_PASS, 0.90
    try:
        restart_guarded = cc_a.assess_task(
            "sorry let me start over we eat eggs every easter", ctx_a)
        off_script_guarded = cc_a.assess_task("i do not want to read this one", ctx_a)
    finally:
        globals()["TIER_A_COVERAGE_PASS"] = _saved_cov
    checks["G: coverage guard, when enabled, clears an insertion-only read"] = (
        restart_guarded["precision_guard"] == "reference_covered"
        and composite_score([], restart_guarded, None, None)["decision"] == "pass")
    checks["G: coverage guard, even enabled, never clears an off-script read"] = (
        off_script_guarded["precision_guard"] is None
        and composite_score([], off_script_guarded, None, None)["decision"] == "review")
    checks["G: read the WRONG text -> still flagged (guards never hide a real miss)"] = (
        off_script["coverage"] < 0.5 and off_script["precision_guard"] is None
        and composite_score([], off_script, None, None)["decision"] == "review"
        and composite_score([], off_script, None, None)["compliance_fail"] is True)
    checks["G: tier_a_score bands land on the composite cutoffs"] = (
        tier_a_score(0.0) == 1.0
        and tier_a_score(TIER_A_WER_REVIEW) >= Q_PASS_CUTOFF          # boundary -> PASS
        and Q_FAIL_CUTOFF <= tier_a_score(TIER_A_WER_FAIL) < Q_PASS_CUTOFF
        and tier_a_score(TIER_A_WER_FAIL + 0.01) < Q_FAIL_CUTOFF
        and tier_a_score(1.5) == 0.0)
    checks["G: tier_a_score is monotonically non-increasing in WER"] = all(
        tier_a_score(w / 100) >= tier_a_score((w + 1) / 100) for w in range(0, 150))
    # ---- Single-model LLM judgement -> compliance score (no network) ----
    _llm = LocalLLM.__new__(LocalLLM)          # bypass __init__: no Ollama probe needed
    _llm.model = "gemma4"
    j_yes = _llm._judgement({"completed": True, "confidence": 0.95, "reasoning": "r"})
    j_no = _llm._judgement({"completed": False, "confidence": 0.95})
    j_meh = _llm._judgement({"completed": False, "confidence": 0.1})
    j_bad = _llm._judgement({"completed": True, "confidence": "not-a-number"})
    checks["H: confident 'completed' -> pass band"] = (
        j_yes["score"] >= Q_PASS_CUTOFF and j_yes["uncertain"] is False
        and composite_score([], dict(j_yes, tier="C"), None, None)["decision"] == "pass")
    checks["H: confident 'not completed' -> fail band -> REVIEW"] = (
        j_no["score"] < Q_FAIL_CUTOFF
        and composite_score([], dict(j_no, tier="C"), None, None)["decision"] == "review"
        and composite_score([], dict(j_no, tier="C"), None, None)["compliance_fail"] is True)
    checks["H: unsure answer lands in the REVIEW band, not fail"] = (
        Q_FAIL_CUTOFF <= j_meh["score"] < Q_PASS_CUTOFF and j_meh["uncertain"] is True
        and composite_score([], dict(j_meh, tier="C"), None, None)["decision"] == "review"
        and composite_score([], dict(j_meh, tier="C"), None, None)["compliance_fail"] is False)
    checks["H: unparsable confidence -> undecided, review band, no crash"] = (
        j_bad["confidence"] == 0.5
        and Q_FAIL_CUTOFF <= j_bad["score"] <= 1.0)
    checks["H: judgement bands land on the composite cutoffs"] = (
        llm_judgement_score(True, 1.0) == 1.0
        and llm_judgement_score(True, LLM_CONFIDENT_AT) >= Q_PASS_CUTOFF
        and llm_judgement_score(False, 1.0) == 0.0
        and llm_judgement_score(False, LLM_CONFIDENT_AT) < Q_FAIL_CUTOFF
        # every sub-threshold confidence, either answer, stays in the review band
        and all(Q_FAIL_CUTOFF <= llm_judgement_score(done, c / 100) < Q_PASS_CUTOFF
                for done in (True, False) for c in range(0, int(LLM_CONFIDENT_AT * 100))))
    checks["H: judgement score is monotonic in confidence"] = (
        all(llm_judgement_score(True, c / 20) <= llm_judgement_score(True, (c + 1) / 20)
            for c in range(20))
        and all(llm_judgement_score(False, c / 20) >= llm_judgement_score(False, (c + 1) / 20)
                for c in range(20)))
    checks["H: a dead/unreachable model yields None, never a bogus score"] = (
        _llm._judgement(None) is None)
    checks["H: _NullLLM covers every method LocalLLM exposes"] = all(
        hasattr(_NullLLM(), m) for m in
        ("active", "pii_spans", "pii_combinatorial_vote", "compliance_vote",
         "task_judge_vote", "transcript_completion_vote"))

    checks["G: word_error_rate unchanged by the alignment rewrite"] = (
        abs(word_error_rate("the cat sat on the mat", "the cat sat on a mat") - 1 / 6) < 1e-9
        and word_error_rate("", "") == 0.0 and word_error_rate("", "hello") == 1.0
        and word_error_rate("a b c", "") == 1.0)

    # ---- Always-flag identifiers: every NAME, and any age over the threshold ----
    def _age_flagged(txt):
        ents = age_scan(txt)
        merged_ = merge_pii(ents, txt)
        return bool(merged_) and merged_[0]["review_worthy"] is True

    checks["E: '95 years old' -> flagged (phrasing the old AGE regex missed)"] = _age_flagged(
        "she is 95 years old and lives alone")
    checks["E: '97-year-old' -> flagged"] = _age_flagged("a 97-year-old participant")
    checks["E: 'aged 102' -> flagged"] = _age_flagged("aged 102 at the time")
    checks["E: spelled-out 'ninety-five years old' -> flagged"] = _age_flagged(
        "he is ninety-five years old")
    checks["E: 'a hundred and two years old' -> flagged"] = _age_flagged(
        "my grandmother is a hundred and two years old")
    checks["E: 'in her nineties' -> flagged"] = _age_flagged("she is in her nineties now")
    checks["E: 'centenarian' -> flagged"] = _age_flagged("he is a centenarian")
    checks["E: age AT/below the threshold is NOT flagged"] = (
        age_scan("she is 90 years old") == [] and age_scan("I am 45 years old") == []
        and age_scan("a 12-year-old boy") == [])
    checks["E: non-age numbers never read as an age"] = (
        age_scan("I'm 100 percent sure") == [] and age_scan("it cost 95 dollars") == []
        and age_scan("we drove 95 miles") == [])
    checks["E: age word->int parsing"] = (
        _age_word_to_int("ninety-five") == 95 and _age_word_to_int("a hundred and two") == 102
        and _age_word_to_int("one hundred") == 100 and _age_word_to_int("ninety") == 90)
    checks["E: threshold is configurable (over_years=89 flags 90)"] = (
        len(age_scan("she is 90 years old", over_years=89)) == 1)
    # Every name flags the file, regardless of who it is or how weak the signal.
    for _label, _txt, _span in (("bare first name", "Will you sit down now", (0, 4)),
                                ("full name", "I met Sarah Connor yesterday.", (6, 18)),
                                ("common word name", "May came to visit", (0, 3))):
        _m = merge_pii([_entity(_span[0], _span[1], "NAME", 0.7, "ner")], _txt)
        checks[f"E: NAME always flagged -- {_label}"] = (
            bool(_m) and _m[0]["review_worthy"] is True
            and composite_score(_m, None, None, q_ok)["decision"] == "review")
    checks["E: a flagged NAME still never auto-fails (not confirmed)"] = (
        composite_score(merge_pii([_entity(0, 4, "NAME", 0.7, "ner")], "Will is here"),
                        None, None, q_ok)["pii_confirmed"] is False)

    # ---- Task -> reference-key resolution must not mix SIBLING VARIANTS ----
    # Plain bidirectional prefix matching pulled another task's prompt into this file's
    # compliance test ('diadochokinesisPA' is a prefix of 'diadochokinesisPataka').
    sib_ref = {"diadochokinesis-PA": {"instructions": "Say puh.", "prompts": []},
               "diadochokinesis-Pataka": {"instructions": "Say pataka.", "prompts": []},
               "picture-description": {"instructions": "Describe picture A.",
                                       "prompts": ["Picture A prompt"]},
               "picture-description-option2": {"instructions": "Describe picture B.",
                                              "prompts": ["Picture B prompt"]},
               "free-speech": {"instructions": "Talk freely.", "prompts": []},
               "free-speech-1": {"instructions": "Question one.", "prompts": []},
               "rainbow": {"instructions": "Read it.", "prompts": ["When the sunlight strikes"]},
               "cape-V-sentences": {"instructions": "Read.", "prompts": ["a"]},
               "cape-V-sentences-1": {"instructions": "Read 1.", "prompts": ["b"]}}
    cc_sib = ComplianceChecker({}, _NullLLM(), task_reference=sib_ref)
    checks["F: exact match wins -- 'Diadochokinesis-PA' does NOT pull in 'Pataka'"] = (
        cc_sib._resolve_reference_keys("Diadochokinesis-PA") == (["diadochokinesis-PA"], False))
    checks["F: 'Picture-description' does NOT pull in '-option2' (a different picture)"] = (
        cc_sib._resolve_reference_keys("Picture-description") == (["picture-description"], False))
    checks["F: 'Free-speech-1' resolves to itself, not the generic 'free-speech'"] = (
        cc_sib._resolve_reference_keys("Free-speech-1") == (["free-speech-1"], False))
    checks["F: no exact key -> most specific (longest) prefix wins"] = (
        cc_sib._resolve_reference_keys("Rainbow-Passage") == (["rainbow"], False))
    checks["F: unknown task -> no keys, no crash"] = (
        cc_sib._resolve_reference_keys("Harvard-Sentences-List-12-1") == ([], False))
    # A filename LESS specific than the reference (no exact key, and no key is a prefix
    # of it) is genuinely ambiguous between the variants -> all returned, flagged.
    cc_amb = ComplianceChecker({}, _NullLLM(), task_reference={
        "cape-V-sentences-1": {"instructions": "Read 1.", "prompts": ["a"]},
        "cape-V-sentences-2": {"instructions": "Read 2.", "prompts": ["b"]}})
    amb_keys, amb_flag = cc_amb._resolve_reference_keys("cape-V-sentences")
    checks["F: filename less specific than reference -> flagged ambiguous"] = (
        amb_flag is True and len(amb_keys) == 2)
    # ...whereas a token MORE specific than the only key resolves cleanly to it.
    checks["F: token more specific than the key -> unambiguous"] = (
        cc_sib._resolve_reference_keys("cape-V-sentences-x") == (["cape-V-sentences"], False))
    # A scripted task with no reference text is RECORDED, not silently skipped.
    ctx_missing = cc_sib.resolve_task_context("sub-1_task-Harvard-Sentences-List-12-1_features.pt", {})
    checks["F: scripted w/o reference -> reference_missing recorded"] = (
        ctx_missing["modality"] == "scripted" and ctx_missing["reference_missing"] is True
        and ctx_missing["reference_texts"] == [])
    # ...but a reference_text carried in the .pt itself is used instead of giving up.
    ctx_meta = cc_sib.resolve_task_context(
        "sub-1_task-Harvard-Sentences-List-12-1_features.pt",
        {"reference_text": "the birch canoe slid on the smooth planks"})
    checks["F: scripted w/o reference falls back to the .pt's own reference_text"] = (
        ctx_meta["reference_texts"] == ["the birch canoe slid on the smooth planks"]
        and ctx_meta["reference_missing"] is False)
    _ta = cc_sib.assess_task("the birch canoe slid on the smooth planks", ctx_meta)
    checks["F: that fallback actually enables the Tier A WER check"] = (
        _ta is not None and _ta["tier"] == "A" and _ta["score"] == 1.0)
    checks["F: correct prompt reaches Tier C for an open task"] = (
        (cc_sib.resolve_task_context("sub-1_task-Picture-description_features.pt", {})
         or {}).get("reference_texts") == ["Picture A prompt"])

    # ---- Compliance-only mode: PII skipped entirely (pii_entities is None) ----
    # The corner case that matters: an ACOUSTIC task in compliance-only mode has NO
    # assessable check at all (quality suppressed, Tier B skipped, no protocol steps),
    # so composite_score must not divide by an empty weight list.
    acoustic_skipped = {"tier": "B", "method": "acoustic", "score": None,
                        "confidence": None, "completed": None, "status": "skipped"}
    q_suppressed = {"applies": False, "score": None, "confidence": None, "hard_fail": False}
    co_acoustic = composite_score(None, acoustic_skipped, None, q_suppressed)
    checks["G: compliance-only + acoustic (no checks at all) -> no crash"] = (
        co_acoustic["decision"] == "pass" and co_acoustic["checks"] == {})
    checks["G: no assessable check -> Q and C are None (unknown), not 0 (bad)"] = (
        co_acoustic["Q"] is None and co_acoustic["C"] is None)
    checks["G: compliance-only marks pii_skipped and never flags PII"] = (
        co_acoustic["pii_skipped"] is True
        and co_acoustic["flagged_by_pii"] is False
        and co_acoustic["pii_confirmed"] is False)
    # Compliance still scores normally when a check IS available.
    co_scored = composite_score(None, {"tier": "A", "score": 0.2, "confidence": 0.9,
                                       "completed": False, "required": True}, None, None)
    checks["G: compliance-only still scores a failing scripted task"] = (
        co_scored["decision"] == "review" and co_scored["flagged_by_compliance"] is True
        and co_scored["pii_skipped"] is True and co_scored["Q"] == 0.2)
    # ...and with PII enabled (entities list, even empty) the PII check is present.
    with_pii = composite_score([], None, None, q_ok)
    checks["G: PII enabled -> pii check present, pii_skipped False"] = (
        with_pii["pii_skipped"] is False and "pii" in with_pii["checks"])

    # ---- Part D: high-recall screening mode (opt-in; defaults above untouched) ----
    _mod.HIGH_RECALL = True
    _mod.RECALL_FLAG_ALL_PII = False
    _mod.RECALL_FLAG_ACOUSTIC = True
    hr_age = merge_pii([_entity(0, 3, "AGE", 0.6, "regex")], "old")          # weak, normally quiet
    hr_name = merge_pii([_entity(0, 4, "NAME", 0.8, "ner"),
                         _entity(0, 4, "NAME", 0.75, "gliner")], "Will you sit down now")
    hr_ac = _recall_compliance_override("well sorry I cannot really do this task today",
                                        "acoustic", {"applies": False, "score": None})
    hr_ac_clean = _recall_compliance_override("ah", "acoustic", {"applies": False, "score": None})
    hr_open = _recall_compliance_override("yes", "open", {"applies": True, "score": 1.0})
    checks["D: high-recall flags a weak AGE-alone detection for review"] = (
        bool(hr_age) and hr_age[0]["review_worthy"] is True
        and composite_score(hr_age, None, None, q_ok)["decision"] == "review")
    checks["D: high-recall flags the lone common-word NAME too (FLAG_ALL_NAMES)"] = (
        bool(hr_name) and hr_name[0]["review_worthy"] is True)
    checks["D: acoustic w/ unexpected speech -> review-band (not fail)"] = (
        hr_ac.get("score") == 0.6 and hr_ac.get("hard_fail") is False
        and composite_score([], None, None, hr_ac)["decision"] == "review")
    checks["D: clean near-empty acoustic stays a pass"] = (hr_ac_clean.get("score") is None)
    checks["D: thin open response -> review-band compliance"] = (hr_open.get("score") == 0.6)
    _mod.HIGH_RECALL = False
    _mod.RECALL_FLAG_ACOUSTIC = False

    # ---- Part I: the COMPANION task_reference.json beside this script ----
    # The script and its task reference are a pair. These checks fail loudly if the
    # JSON is missing, malformed, or has drifted from the shape the pipeline expects
    # -- i.e. exactly the cases where scripted-task checking would otherwise go
    # quietly dead and off-script reads would start passing.
    ref_path = resolve_task_reference_path(TASK_REFERENCE)
    print(f"\n[Part I] Companion task reference: {ref_path.name} (beside the script).")
    checks["I: task_reference.json sits next to the script"] = ref_path.is_file()
    ref = load_task_reference(TASK_REFERENCE)
    checks["I: it loads and carries all 797 task definitions"] = (len(ref) == 797)
    checks["I: every entry has the same {instructions, prompts} shape"] = all(
        set(v) == {"instructions", "prompts"}
        and isinstance(v["instructions"], str) and isinstance(v["prompts"], list)
        and all(isinstance(p, str) for p in v["prompts"])
        for v in ref.values())
    checks["I: read-aloud tasks carry their reference text"] = (
        ref.get("harvard-sentences-list-1-1", {}).get("prompts")
        == ["The birch canoe slid on the smooth planks."]
        and len([k for k in ref if k.startswith("harvard-sentences-list-")]) == 720
        and ref["rainbow"]["prompts"][0].lower().startswith("when the sunlight"))
    checks["I: acoustic tasks carry instructions but no reference text"] = (
        ref["maximum-phonation-time"]["prompts"] == []
        and bool(ref["maximum-phonation-time"]["instructions"]))
    checks["I: acoustic / scripted / open tasks are all present"] = all(
        k in ref for k in ("rainbow", "maximum-phonation-time", "free-speech-1",
                           "cape-V-sentences-1", "caterpillar-passage",
                           "picture-description", "diadochokinesis-Pataka"))
    checks["I: a relative path resolves next to the SCRIPT, not the cwd"] = (
        resolve_task_reference_path("task_reference.json").parent
        == Path(__file__).resolve().parent)

    # The reference file drives filename -> tier routing.
    cc_b = ComplianceChecker({}, _NullLLM(), task_reference=ref)
    ctx_rb = cc_b.resolve_task_context("sub-1_ses-1_task-Rainbow_features.pt", {})
    ctx_hb = cc_b.resolve_task_context("sub-1_ses-1_task-Harvard-Sentences-List-1-1_features.pt", {})
    ctx_mb = cc_b.resolve_task_context("sub-1_ses-1_task-Maximum-Phonation-Time_features.pt", {})
    ctx_ob = cc_b.resolve_task_context("sub-1_ses-1_task-Free-Speech-1_features.pt", {})
    checks["I: the reference gives a scripted task its Tier A reference text"] = (
        ctx_rb and ctx_rb["tier"] == "A" and bool(ctx_rb["reference_texts"])
        and ctx_rb["reference_missing"] is False)
    checks["I: a Harvard sentence resolves to its own prompt"] = (
        ctx_hb and ctx_hb["reference_texts"] == ["The birch canoe slid on the smooth planks."])
    checks["I: acoustic -> Tier B and open -> Tier C"] = (
        ctx_mb and ctx_mb["tier"] == "B" and ctx_ob and ctx_ob["tier"] == "C")
    read_ok = cc_b.assess_task("the birch canoe slid on the smooth planks", ctx_hb)
    read_bad = cc_b.assess_task("i would rather not read anything at all today thanks", ctx_hb)
    checks["I: a correct read against a reference prompt PASSES"] = (
        read_ok["score"] >= Q_PASS_CUTOFF
        and composite_score([], read_ok, None, None)["decision"] == "pass")
    checks["I: an off-script read against a reference prompt is FLAGGED"] = (
        composite_score([], read_bad, None, None)["decision"] == "review"
        and composite_score([], read_bad, None, None)["flagged_by_compliance"] is True)

    # A MISSING reference stops the run -- it is never silently treated as "no tasks",
    # which would turn every scripted check off without saying so.
    missing_stops = False
    try:
        load_task_reference("definitely_not_a_real_file_9f2c.json")
    except SystemExit:
        missing_stops = True
    checks["I: a missing task reference STOPS the run (never a silent skip)"] = missing_stops
    bad_stops = False
    with tempfile.TemporaryDirectory() as tmp:
        broken = Path(tmp) / "broken.json"
        broken.write_text("{not valid json")
        try:
            load_task_reference(str(broken))
        except SystemExit:
            bad_stops = True
        # ...and an explicitly chosen alternative file is used as-is.
        alt = Path(tmp) / "study_tasks.json"
        alt.write_text(json.dumps({
            "rainbow": {"instructions": "Read ours.", "prompts": ["a different passage"]},
            "my-new-task": {"instructions": "Do the new thing.", "prompts": []}}))
        alt_ref = load_task_reference(str(alt))
    checks["I: an unparsable task reference STOPS the run"] = bad_stops
    checks["I: --task-reference points at another study's file"] = (
        len(alt_ref) == 2 and alt_ref["rainbow"]["prompts"] == ["a different passage"])
    checks["I: --no-task-reference runs without filename-based compliance"] = (
        load_task_reference("") == {})

    # ---- Part J: the whole pipeline, end to end, on nothing but built-ins ----
    #   .pt files in  ->  task identified from each filename against the built-in
    #   task reference  ->  list of flagged file names out.
    # This is the contract the script exists to satisfy, so it is asserted here
    # rather than left to a manual run.
    print("\n[Part J] End to end: .pt folder -> task_reference.json -> flagged names.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # A correct read of a Harvard prompt from the reference -> compliant.
        torch.save({"transcription": "the birch canoe slid on the smooth planks"},
                   tmp / "sub-01_ses-1_task-Harvard-Sentences-List-1-1_features.pt")
        # The WRONG text for that same prompt -> only the task reference catches this.
        torch.save({"transcription": "i do not really want to read this one today sorry"},
                   tmp / "sub-02_ses-1_task-Harvard-Sentences-List-1-1_features.pt")
        # Acoustic task: a near-empty transcript is EXPECTED, never a failure.
        torch.save({"transcription": "aaaah"},
                   tmp / "sub-03_ses-1_task-Maximum-Phonation-Time_features.pt")
        # Open task carrying a direct identifier -> flagged on PII.
        torch.save({"transcription": "You can reach me any time at sarah.connor@example.com."},
                   tmp / "sub-04_ses-1_task-Free-Speech-1_features.pt")

        args.protocol = None
        e2e = Pipeline(args, {}, ref)
        e2e_report = process_folder(tmp, e2e, False, None)
        print_flagged_files(e2e_report)
        flagged = e2e_report["summary"]["flagged_files"]
        by_name = {f["name"]: f for f in flagged}
        e2e_by = {Path(r["file"]).name: r for r in e2e_report["files"]}
        listing = tmp / "flagged.txt"
        write_flagged_list(e2e_report, listing)
        lines = [l for l in listing.read_text().split("\n") if l]

    n_off, n_pii = ("sub-02_ses-1_task-Harvard-Sentences-List-1-1_features.pt",
                    "sub-04_ses-1_task-Free-Speech-1_features.pt")
    n_ok, n_ac = ("sub-01_ses-1_task-Harvard-Sentences-List-1-1_features.pt",
                  "sub-03_ses-1_task-Maximum-Phonation-Time_features.pt")
    checks["J: all four .pt files were read and scored"] = (
        e2e_report["num_files"] == 4 and e2e_report["summary"]["errors"] == 0)
    checks["J: filenames resolved via the reference (scripted/acoustic/open)"] = (
        e2e_by[n_ok]["modality"] == "scripted" and e2e_by[n_ac]["modality"] == "acoustic"
        and e2e_by[n_pii]["modality"] == "open"
        and e2e_report["summary"]["scripted_without_reference"] == [])
    checks["J: reading the WRONG script is caught -- only the reference reveals it"] = (
        n_off in by_name and "compliance:fail" in by_name[n_off]["reasons"])
    checks["J: a direct identifier flags its file on PII"] = (
        n_pii in by_name and "pii:confirmed" in by_name[n_pii]["reasons"])
    # Engine-independent: these two must never be flagged BY COMPLIANCE, whatever
    # PII engines happen to be installed on the machine running the self-test.
    checks["J: a correct read is not a compliance flag"] = (
        e2e_by[n_ok]["composite"]["flagged_by_compliance"] is False)
    checks["J: a near-empty ACOUSTIC transcript is not a compliance flag"] = (
        e2e_by[n_ac]["composite"]["flagged_by_compliance"] is False)
    checks["J: flagged_files is exactly the set of non-pass decisions"] = (
        {f["name"] for f in flagged}
        == {n for n, r in e2e_by.items() if r["decision"] != "pass"})
    checks["J: every flagged entry says WHY, and names a real file"] = (
        all(f["reasons"] and Path(f["file"]).name == f["name"] for f in flagged))
    checks["J: --flagged-list writes one flagged path per line"] = (
        lines == [f["file"] for f in flagged] and len(lines) == len(flagged))

    print()
    for label, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    overall = all(checks.values())
    print("\nSELF-TEST PASSED" if overall else "\nSELF-TEST FAILED")
    sys.exit(0 if overall else 1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folder", type=Path, nargs="?",
                   default=(Path(INPUT_FOLDER) if INPUT_FOLDER else None),
                   help="Folder of .pt files. Optional only if INPUT_FOLDER is set in "
                        "the CONFIG block at the top of this file; with neither, the "
                        "script stops.")
    p.add_argument("--protocol", default=PROTOCOL_SPEC,
                   help="OPTIONAL protocol spec JSON. Omit for zero-config: task info "
                        "is read from each .pt's own metadata. Provide only for "
                        "interview-style protocol-step checking.")
    p.add_argument("--task-reference", dest="task_reference", default=TASK_REFERENCE,
                   help="Task reference JSON (task -> instructions/prompts) driving "
                        "filename-based Tier A/B/C compliance. Default %(default)r, the "
                        "companion file beside this script; a relative path resolves "
                        "there. Missing -> the run stops.")
    p.add_argument("--no-task-reference", dest="task_reference", action="store_const",
                   const="",
                   help="Run WITHOUT the task reference, disabling filename-based task "
                        "compliance (compliance falls back to transcript quality only).")
    p.add_argument("--output", "-o", type=Path, default=Path(OUTPUT_JSON),
                   help="Full JSON report path. Default %(default)s.")
    # Kept as a plain string rather than type=Path: Path("") is Path("."), so an empty
    # --csv would otherwise sail through as a directory and blow up on write, at the very
    # end of a long run. Empty (or --no-csv) means "skip the CSV" instead.
    p.add_argument("--csv", default=OUTPUT_CSV or "",
                   help="Flat CSV summary path. Pass '' or --no-csv to skip it. "
                        "Default %(default)r.")
    p.add_argument("--no-csv", dest="csv", action="store_const", const="",
                   help="Skip the CSV summary (the JSON report is still written).")
    p.add_argument("--flagged-list", dest="flagged_list", default=OUTPUT_FLAGGED_LIST or "",
                   help="Also write JUST the flagged file paths, one per line, to this "
                        "path -- the plain-text list of what needs human review. The same "
                        "list is always printed and is in the JSON report.")
    p.add_argument("--recursive", action="store_true", default=RECURSIVE)
    p.add_argument("--max-files", type=int, default=MAX_FILES)
    p.add_argument("--spacy-model", default=SPACY_MODEL)
    p.add_argument("--gliner-model", default=GLINER_MODEL)
    p.add_argument("--enable-spacy", action="store_true", default=ENABLE_SPACY)
    p.add_argument("--no-spacy", dest="enable_spacy", action="store_false")
    p.add_argument("--enable-gliner", action="store_true", default=ENABLE_GLINER)
    p.add_argument("--no-gliner", dest="enable_gliner", action="store_false")
    p.add_argument("--enable-presidio", action="store_true", default=ENABLE_PRESIDIO)
    p.add_argument("--no-presidio", dest="enable_presidio", action="store_false")
    p.add_argument("--no-precision", dest="precision", action="store_false", default=PRECISION_MODE,
                   help="Disable the false-positive precision guards (revert to recall-max).")
    p.add_argument("--gliner-threshold", type=float, default=GLINER_THRESHOLD,
                   help="GLiNER detection threshold; raise to cut GLiNER false positives.")
    p.add_argument("--tier-a-wer-review", type=float, default=TIER_A_WER_REVIEW,
                   help="Tier A: word error rate at or below which a scripted read counts "
                        "as compliant (ASR noise on a correctly-read script). Above it the "
                        "file goes to REVIEW. Default %(default)s.")
    p.add_argument("--tier-a-wer-fail", type=float, default=TIER_A_WER_FAIL,
                   help="Tier A: word error rate above which the read is treated as a hard "
                        "failure (wrong text / not read). Default %(default)s.")
    p.add_argument("--tier-a-min-word-errors", type=int, default=TIER_A_MIN_WORD_ERRORS,
                   help="Tier A: never flag a scripted read with fewer than this many word "
                        "errors, however bad the ratio -- stops one misheard word flagging a "
                        "short sentence. Default %(default)s.")
    p.add_argument("--tier-a-coverage-pass", type=float, default=TIER_A_COVERAGE_PASS,
                   help="Tier A: reference-word coverage at or above which an insertion-heavy "
                        "read counts as compliant (the script WAS read, plus extra speech). "
                        "Set to 1.1 to disable the guard. Default %(default)s.")
    p.add_argument("--enable-llm", dest="enable_llm", action="store_true",
                   default=ENABLE_LLM,
                   help="Local LLM via Ollama: enables open-ended (Tier C) task compliance "
                        "plus extra PII judgement. Roughly DOUBLES the catch rate on "
                        "open tasks, at ~10x the review queue. Off by default; slow.")
    p.add_argument("--no-llm", dest="enable_llm", action="store_false",
                   help="Force the LLM off (the default) -- a fast rules-only pass. "
                        "Open-ended (Tier C) compliance is then not assessed.")
    p.add_argument("--llm-compliance", dest="llm_compliance", action="store_true",
                   default=ENABLE_LLM_COMPLIANCE_JUDGE,
                   help="Let the LLM also judge borderline transcripts for task completion. "
                        "Runs only on review-band files, and only with --enable-llm.")
    p.add_argument("--no-llm-compliance", dest="llm_compliance", action="store_false",
                   help="Disable the borderline-transcript LLM judge (Tier C is unaffected).")
    p.add_argument("--ollama-host", default=OLLAMA_HOST)
    p.add_argument("--ollama-model", dest="ollama_model", default=OLLAMA_MODEL,
                   help="The single Ollama model used for every LLM call "
                        "(default: %(default)s). Any tag Ollama can serve.")
    p.add_argument("--high-recall", dest="high_recall", action="store_true", default=HIGH_RECALL,
                   help="SCREENING mode: maximize recall (catch PII / non-compliance), routing "
                        "more files to REVIEW. Fewer false negatives, more false positives. Also "
                        "keeps low-confidence detections (implies no-precision).")
    p.add_argument("--flag-all-open", dest="flag_all_open", action="store_true", default=RECALL_FLAG_ALL_OPEN,
                   help="High-recall: route EVERY open free-speech task to REVIEW (max open-task "
                        "compliance recall).")
    p.add_argument("--flag-all-pii", dest="flag_all_pii", action="store_true", default=RECALL_FLAG_ALL_PII,
                   help="High-recall: flag even lone common-word NAME guesses (absolute max PII recall).")
    p.add_argument("--flag-acoustic", dest="flag_acoustic", action="store_true", default=RECALL_FLAG_ACOUSTIC,
                   help="High-recall: also route acoustic tasks with unexpected speech to REVIEW "
                        "(high false-positive; off by default).")
    p.add_argument("--acoustic-max-content-words", dest="acoustic_max_content_words",
                   type=int, default=RECALL_ACOUSTIC_MAX_CONTENT_WORDS,
                   help="With --flag-acoustic: an acoustic task with MORE content words than "
                        "this goes to REVIEW. Use -1 to route EVERY acoustic task to review -- "
                        "the only way to reach acoustic misses, at the cost of the entire "
                        "acoustic half of the corpus. Default %(default)s.")
    p.add_argument("--open-min-content-words", dest="open_min_content_words",
                   type=int, default=RECALL_OPEN_MIN_CONTENT_WORDS,
                   help="High-recall: an open free-speech task with FEWER content words than "
                        "this goes to REVIEW. Default %(default)s.")
    p.add_argument("--compliance-only", dest="compliance_only", action="store_true",
                   default=COMPLIANCE_ONLY,
                   help="TASK COMPLIANCE ONLY: skip PII detection entirely (no spaCy/GLiNER/"
                        "Presidio load, no PII scan). Much faster. PII fields are reported as "
                        "skipped rather than clean, and the composite score reflects "
                        "compliance alone.")
    p.add_argument("--selftest", action="store_true", default=SELFTEST)
    return p.parse_args()


def _apply_runtime_config(args):
    """Let CLI flags override the CONFIG-block globals that functions read at runtime."""
    global PRECISION_MODE, ENABLE_LLM_COMPLIANCE_JUDGE
    global HIGH_RECALL, RECALL_FLAG_ALL_OPEN, RECALL_FLAG_ALL_PII, RECALL_FLAG_ACOUSTIC
    global RECALL_ACOUSTIC_MAX_CONTENT_WORDS, RECALL_OPEN_MIN_CONTENT_WORDS
    RECALL_ACOUSTIC_MAX_CONTENT_WORDS = getattr(args, "acoustic_max_content_words",
                                                RECALL_ACOUSTIC_MAX_CONTENT_WORDS)
    RECALL_OPEN_MIN_CONTENT_WORDS = getattr(args, "open_min_content_words",
                                            RECALL_OPEN_MIN_CONTENT_WORDS)
    global TIER_A_WER_REVIEW, TIER_A_WER_FAIL, TIER_A_MIN_WORD_ERRORS, TIER_A_COVERAGE_PASS
    TIER_A_WER_REVIEW = getattr(args, "tier_a_wer_review", TIER_A_WER_REVIEW)
    TIER_A_WER_FAIL = max(TIER_A_WER_REVIEW, getattr(args, "tier_a_wer_fail", TIER_A_WER_FAIL))
    TIER_A_MIN_WORD_ERRORS = getattr(args, "tier_a_min_word_errors", TIER_A_MIN_WORD_ERRORS)
    TIER_A_COVERAGE_PASS = getattr(args, "tier_a_coverage_pass", TIER_A_COVERAGE_PASS)
    PRECISION_MODE = getattr(args, "precision", PRECISION_MODE)
    # The borderline-transcript judge runs THROUGH the same model, so --no-llm switches it
    # off too; keeping it "on" with no LLM would only misreport what actually ran.
    ENABLE_LLM_COMPLIANCE_JUDGE = (getattr(args, "llm_compliance", ENABLE_LLM_COMPLIANCE_JUDGE)
                                   and getattr(args, "enable_llm", True))
    HIGH_RECALL = getattr(args, "high_recall", HIGH_RECALL)
    RECALL_FLAG_ALL_OPEN = getattr(args, "flag_all_open", RECALL_FLAG_ALL_OPEN)
    RECALL_FLAG_ALL_PII = getattr(args, "flag_all_pii", RECALL_FLAG_ALL_PII)
    RECALL_FLAG_ACOUSTIC = getattr(args, "flag_acoustic", RECALL_FLAG_ACOUSTIC)
    if HIGH_RECALL:
        # Recall over precision: keep low-confidence detections so more spans exist
        # to flag (the precision guards otherwise drop them -> false negatives).
        PRECISION_MODE = False
    else:
        # The acoustic / open recall nets live inside the high-recall override, so on
        # their own they do nothing. Silently ignoring them would look like "I turned
        # that on and it caught nothing" -- say so instead.
        stray = [f for f, on in (("--flag-acoustic", RECALL_FLAG_ACOUSTIC),
                                 ("--flag-all-open", RECALL_FLAG_ALL_OPEN)) if on]
        if stray:
            print(f"  [note] {', '.join(stray)} has no effect without --high-recall "
                  f"(the recall nets run inside high-recall mode); enabling it now.")
            HIGH_RECALL = True
            PRECISION_MODE = False


def print_active_modes(args):
    """Echo the modes actually in force at the top of a run.

    Written for IDE use: hitting Run gives no visible command line, so without this
    there is nothing in the output saying which toggles were set. Reads the EFFECTIVE
    values (the MODES block as overridden by any CLI flags and by the HIGH_RECALL
    interlock), not the literals in the file, so it cannot drift from what really ran."""
    def sw(flag):
        return "ON" if flag else "off"

    scope = [str(args.folder)]
    if args.recursive:
        scope.append("recursive")
    if args.max_files:
        scope.append(f"first {args.max_files} files")
    print("--- Modes ---")
    print(f"  input           {'  |  '.join(scope)}")
    if getattr(args, "compliance_only", False):
        print("  PII detection   SKIPPED entirely (compliance-only mode)")
    else:
        print(f"  PII engines     spaCy={sw(args.enable_spacy)}  GLiNER={sw(args.enable_gliner)}"
              f"  Presidio={sw(args.enable_presidio)}   (missing ones are skipped with a notice)")
        print(f"  always flag     names={sw(FLAG_ALL_NAMES)}  demographics={sw(DEMOGRAPHIC_PII)}"
              f"  age>{AGE_REVIEW_OVER_YEARS}   hard-fail labels: "
              f"{', '.join(sorted(HARD_GATE_PII_LABELS)) or 'none'}")
    if args.enable_llm:
        print(f"  local LLM       ON -- {args.ollama_model!r} at {args.ollama_host}"
              f"{'  + borderline judge' if ENABLE_LLM_COMPLIANCE_JUDGE else ''}")
        print("                  Tier C open-task compliance + extra PII judgement; "
              "~1-2s per file once warm.")
    else:
        print("  local LLM       off -- open-ended (Tier C) task compliance is NOT assessed, so "
              "open tasks")
        print("                  are judged only on whether a real response was spoken. "
              "Set ENABLE_LLM = True.")
    print(f"  posture         precision guards={sw(PRECISION_MODE)}  high-recall={sw(HIGH_RECALL)}"
          + (f"  (all-PII={sw(RECALL_FLAG_ALL_PII)} all-open={sw(RECALL_FLAG_ALL_OPEN)} "
             f"acoustic={sw(RECALL_FLAG_ACOUSTIC)})" if HIGH_RECALL else ""))
    ref = getattr(args, "task_reference", TASK_REFERENCE)
    print(f"  task reference  {ref if ref else 'DISABLED -- no filename-based task compliance'}")
    outs = [str(args.output)] + [x for x in (args.csv, getattr(args, "flagged_list", "")) if x]
    print(f"  outputs         {', '.join(outs)}")


def main():
    args = build_args()
    if args.selftest:
        # The self-test must stay hermetic, fast and deterministic: synthetic data, no
        # network, no model server. Force the LLM off regardless of the defaults so a
        # machine with Ollama running gets the same result as one without.
        args.enable_llm = args.llm_compliance = False
    _apply_runtime_config(args)
    if args.selftest:
        run_selftest(args)
        return
    # STOP if there is no input path. Checked BEFORE any status is printed, so a
    # misconfigured run ends with just this message rather than an unrelated notice
    # above it. Nothing is guessed: no default folder, no scanning the working
    # directory -- the script would otherwise silently report on the wrong data.
    if args.folder is None:
        raise SystemExit(
            "No input folder set -- stopping.\n"
            "Set INPUT_FOLDER in the CONFIG block at the top of this file:\n"
            '    INPUT_FOLDER = "/path/to/your/pt_files"\n'
            "or pass the folder as the first argument:\n"
            f"    python {Path(__file__).name} /path/to/your/pt_files")
    if not args.folder.is_dir():
        raise SystemExit(
            f"{str(args.folder)!r} is not a directory -- stopping.\n"
            "Point INPUT_FOLDER (CONFIG block, top of this file) or the first argument "
            "at a folder of .pt files.")
    print_active_modes(args)
    protocol = load_protocol(args.protocol)
    task_reference = load_task_reference(getattr(args, "task_reference", TASK_REFERENCE))
    pipeline = Pipeline(args, protocol, task_reference)
    report = process_folder(args.folder, pipeline, args.recursive, args.max_files)
    print_summary(report)
    write_json(report, args.output)
    if args.csv:
        write_csv(report, Path(args.csv))
    if getattr(args, "flagged_list", ""):
        write_flagged_list(report, Path(args.flagged_list))


if __name__ == "__main__":
    main()
