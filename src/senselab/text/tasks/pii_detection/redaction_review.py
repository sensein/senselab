"""An instruction-tuned LLM reading one recording's transcript, in its own isolated venv.

The fifth PII engine, and the only one asked to *explain* rather than to mark. It reads the original
words and, where one exists, the text an applied redaction produced; it does not depend on the
detectors having run, which is what lets it read the population they never saw. It is off unless a
run turns it on.

Its product is a chain of thought, three independent judgments — whether an applied redaction
removed what identifies the speaker, whether the original words carry anything identifying at all,
and whether the words show more than one person speaking in the recording — and a proposal: the
redaction it would apply instead, as text spans, in either direction. What it can do that the
detectors cannot is say *why*: a date plus a street plus an employer that no single detector flags,
a duration that identifies nobody, a diagnosis nothing here looks for.

The loop is the caller's, not the model's: one review per call, and a caller that wants the model to
see the effect of its own concerns masks them and calls again. See
``specs/20260817-triage-workflow-dag/llm-check.md``.

Heavy dependencies (``transformers``, ``torch``, ``accelerate``, ``compressed-tensors``) live in an
isolated venv built by :func:`senselab.utils.subprocess_venv.ensure_venv`, the same way the detector
cascade's own venv does, so the host stays off a fixed torch stack.

The model is served by one of two engines (:data:`~senselab.text.tasks.pii_detection.redaction_review_vllm.ENGINES`):
the transformers worker below, or a pinned vLLM server fed one request at a time
(:mod:`~senselab.text.tasks.pii_detection.redaction_review_vllm`). Each reading records which.

The venv holds **one long-lived worker per process**, started on the first review and reused by every
later one, so the weights are loaded once rather than once per review. :func:`shutdown_review_worker`
ends it; it also ends at interpreter exit and when its stdin reaches EOF, so a killed parent does not
leave weights resident. See ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
"""

from __future__ import annotations

import atexit
import difflib
import hashlib
import json
import logging
import queue
import re
import subprocess
import threading
import time
import unicodedata
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

from senselab.text.tasks.pii_detection.redaction_policy import country_runs, date_positions, state_positions
from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python

logger = logging.getLogger("senselab")

REVIEW_VENV = "pii-redaction-review"
REVIEW_PYTHON = "3.12"

REVIEW_REQUIREMENTS = [
    "transformers>=5.8",
    "torch>=2.8,<2.9",
    "accelerate>=1.0",
    "compressed-tensors>=0.15",
]

DEFAULT_MODEL = "google/gemma-4-31B-it-qat-w4a16-ct"
DEFAULT_REF = "main"
_REASONING_HEADING = "REASONING:"
_PROPOSAL_HEADING = "PROPOSAL:"
_REDACTION_HEADING = "REDACTION:"
_ORIGINAL_HEADING = "ORIGINAL:"
_SPEAKERS_HEADING = "SPEAKERS:"
_CONDITIONS_HEADING = "CONDITIONS:"
_OTHER_SPEAKERS_HEADING = "OTHER_SPEAKERS:"
_INSTRUCTIONS_SPOKEN_HEADING = "INSTRUCTIONS_SPOKEN:"
_OTHER_SPEAKER_ROLE_HEADING = "OTHER_SPEAKER_ROLE:"
_OTHER_SPEAKER_QUOTES_HEADING = "OTHER_SPEAKER_QUOTES:"
_OFF_TASK_SPEECH_HEADING = "OFF_TASK_SPEECH:"
_OFF_TASK_QUOTES_HEADING = "OFF_TASK_QUOTES:"
_PHRASES_HEADING = "PHRASES_INSTEAD_OF_ITEMS:"
_PHRASE_QUOTES_HEADING = "PHRASE_QUOTES:"
_TASK_CONTENT_QUOTES_HEADING = "TASK_CONTENT_QUOTES:"

PROMPT_VERSION = 14
"""The prompt and its parse, as one number: it changes whenever either changes what a reading holds."""

REVIEW_INPUTS_VERSION = 3
"""The shape of the ``review_inputs`` record a reading carries (:func:`review_inputs`)."""

REQUIRED_REVIEW_INPUTS = ("task", "full_transcript", "pii_annotations", "asr_readings", "instructions")
"""The ``review_inputs`` keys a reading must have had for its releases to be weighed."""

CONSENSUS_TRANSCRIPT = "consensus_transcript"
"""The context key holding the whole consensus transcript, word by word, with its PII and variation."""

WORD_SCANNED = ""
WORD_TASK = "task"
WORD_NON_LEXICAL = "non_lexical"
WORD_KINDS = (WORD_SCANNED, WORD_TASK, WORD_NON_LEXICAL)
"""What a consensus word is to the PII scan: a word it read, the task's own content, or non-lexical."""

CONSENSUS_HAYSTACK = "consensus"
"""The reading a finding was found on when it was found on the consensus transcript itself."""

REDACTION_STATES = ("complete", "incomplete", "not_applicable")
"""Whether the redaction, where one was applied, removed the identifying content."""

ORIGINAL_STATES = ("clean", "carries_pii")
"""Whether the recording's own words carry identifying content, redaction aside."""

SPEAKER_STATES = ("one", "more_than_one", "unclear")
"""How many people the transcript's own words show speaking in the recording."""

OFF_TASK_STATES = ("none", "some", "extensive")
"""How much of the recording the participant spends off the task."""

PHRASES_NONE = "none"
PHRASES_OCCASIONAL = "occasional"
PHRASES_PREDOMINANT = "predominant"
PHRASE_STATES = (PHRASES_NONE, PHRASES_OCCASIONAL, PHRASES_PREDOMINANT)
"""In an item-set task, whether the participant utters phrases instead of items."""

ITEM_SET = "item_set"
"""The context key that is True where the declared task is an item-set task: it is asked about phrases instead
of items, not about off-task speech (:func:`review_prompt`)."""

OTHER_SPEAKER_ROLES = ("none", "assistant", "background", "unclear")
"""Who besides the participant speaks: nobody, an assistant or examiner, someone in the background, or unclear."""

REDACT = "redact"
RELEASE = "release"
PROPOSAL_ACTIONS = (REDACT, RELEASE)
"""What a proposal entry asks for: remove this text, or stop removing text already removed."""

RELABEL_WORK_TITLE = "work_title"
RELABEL_BRAND = "brand_or_product"
RELABEL_ORGANIZATION = "organization"
RELABEL_PLACE = "place"
RELABEL_OTHER = "other_non_person"
RELABELS = (RELABEL_WORK_TITLE, RELABEL_BRAND, RELABEL_ORGANIZATION, RELABEL_PLACE, RELABEL_OTHER)
"""What a ``release`` entry may say a removed name is instead of a person (its ``relabel`` key)."""

PLACE_HISTORICAL = "historical"
PLACE_FICTIONAL = "fictional"
PLACE_GENERAL = "public_landmark_or_general_knowledge"
PLACE_TASK = "task_content"
PLACE_REASONS = (PLACE_HISTORICAL, PLACE_FICTIONAL, PLACE_GENERAL, PLACE_TASK)
"""Why a ``release`` entry may let a place smaller than a country through (its ``place_reason`` key)."""

SAFE_HARBOR_PATH = Path(__file__).parent / "data" / "safe_harbor.yaml"
"""The HIPAA Safe Harbor identifiers, with their citation; the reviewer applies (D)-(R) as written and the
redaction policy (:mod:`~senselab.text.tasks.pii_detection.redaction_policy`) in place of (A)-(C)."""


TASK_GUIDANCE_PATH = Path(__file__).parent / "data" / "task_guidance.yaml"
"""What counts as the task's own content in each task family, and what kind of task each family is."""

NATURE_OPEN_RESPONSE = "open_response"
NATURE_ITEM_GENERATION = "item_generation"
NATURE_READ_OR_RECALL = "read_or_recall"
NATURE_NON_LEXICAL = "non_lexical"
TASK_NATURES = (NATURE_OPEN_RESPONSE, NATURE_ITEM_GENERATION, NATURE_READ_OR_RECALL, NATURE_NON_LEXICAL)
"""The kinds of task a declared family is, as ``task_guidance.yaml`` names them under ``natures``."""


@lru_cache(maxsize=1)
def _task_guidance() -> tuple[tuple[tuple[str, ...], str], ...]:
    """The packaged guidance, as ``(family names, guidance)`` pairs in file order."""
    raw = yaml.safe_load(TASK_GUIDANCE_PATH.read_text()) or {}
    return tuple(
        (tuple(str(name) for name in entry.get("names") or ()), " ".join(str(entry.get("guidance") or "").split()))
        for entry in raw.get("families") or ()
    )


def task_guidance(family: str | None) -> str:
    """The guidance on task content for one declared task family.

    Args:
        family: The declared family (``productive-vocabulary``, ``picture-description-option1`` ...), or None.

    Returns:
        The first entry whose names include the family, or a prefix of it followed by ``-``; the empty
        string where none does.
    """
    for names, guidance in _task_guidance():
        if _family_matches(family, names):
            return guidance
    return ""


def _family_matches(family: str | None, names: Sequence[str]) -> bool:
    """Whether a declared family equals one of the names, or starts with one followed by ``-``."""
    key = str(family or "").strip().lower()
    return bool(key) and any(key == name or key.startswith(f"{name}-") for name in names)


@lru_cache(maxsize=1)
def _task_natures() -> tuple[tuple[str, tuple[str, ...], str], ...]:
    """The packaged natures, as ``(nature, family names, description)`` in file order."""
    raw = yaml.safe_load(TASK_GUIDANCE_PATH.read_text()) or {}
    natures = tuple(
        (
            str(entry.get("nature") or ""),
            tuple(str(name) for name in entry.get("names") or ()),
            " ".join(str(entry.get("description") or "").split()),
        )
        for entry in raw.get("natures") or ()
    )
    unknown = [nature for nature, _names, _description in natures if nature not in TASK_NATURES]
    if unknown:
        raise ValueError(f"{TASK_GUIDANCE_PATH.name}: unknown natures {unknown}; known: {TASK_NATURES}")
    return natures


def task_nature(family: str | None) -> str:
    """What kind of task a declared family is.

    Args:
        family: The declared family, or None.

    Returns:
        One of :data:`TASK_NATURES` for the first ``natures`` entry whose names match the family, or the
        empty string where none does.
    """
    for nature, names, _description in _task_natures():
        if _family_matches(family, names):
            return nature
    return ""


def task_nature_description(family: str | None) -> str:
    """The description of a declared family's nature, as both prompts send it.

    Args:
        family: The declared family, or None.

    Returns:
        The matching ``natures`` entry's description, or the empty string where none matches.
    """
    for _nature, names, description in _task_natures():
        if _family_matches(family, names):
            return description
    return ""


@lru_cache(maxsize=1)
def task_guidance_digest() -> str:
    """The sha256 of the packaged guidance file, which a reading's cache identity carries."""
    return hashlib.sha256(TASK_GUIDANCE_PATH.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def safe_harbor() -> dict[str, Any]:
    """The Safe Harbor standard as packaged: identifiers, residual clause, examples, category map.

    Returns:
        The parsed ``data/safe_harbor.yaml``.
    """
    return dict(yaml.safe_load(SAFE_HARBOR_PATH.read_text()) or {})


def safe_harbor_codes(category: str) -> tuple[str, ...]:
    """The Safe Harbor identifiers a reviewer category falls under.

    Args:
        category: A reviewer category (``PERSON``, ``LOCATION`` ...), upper-cased or not.

    Returns:
        Their codes (``A`` .. ``R``); empty where the category is no identifier by itself.
    """
    return tuple(str(code) for code in (safe_harbor().get("categories") or {}).get(str(category).upper(), ()) or ())


def _identifier_rule() -> str:
    """The Safe Harbor number, contact and code identifiers (D-R), as the prompt lists them."""
    standard = safe_harbor()
    lines = [f"Also remove every one of these HIPAA Safe Harbor identifiers ({standard['citation']}):"]
    for identifier in standard["identifiers"]:
        if str(identifier["code"]) in ("A", "B", "C"):
            continue
        lines.append(f"  ({identifier['code']}) {identifier['text']}")
    return "\n".join(lines)


_POLICY = (
    "REDACTION POLICY. Apply these rules; they are stricter than HIPAA Safe Harbor in places.\n"
    "- NAMES. Remove the name of the participant and of every person they know: family, friends, "
    "neighbours, doctors, coworkers, anyone in their life. A public figure -- a celebrity, an author, an "
    'artist, a politician, a historical figure, a fictional character ("Ray Bradbury", "Taylor Swift", '
    '"Cinderella") -- may be proposed for release with the reason; a person will confirm it before it is '
    'released. When a removed name is not a person at all, release it with the key "relabel" saying what '
    'it is: work_title (a film, book, song, show or game: "Star Wars", "Harry Potter", "The Green Mile"), '
    'brand_or_product ("Botox", "M&M"), organization, place, or other_non_person (an ordinary word, or a '
    "character in the task's picture or story, that a detector mistook for a name). A relabelled "
    "organization then follows the organization rule and a relabelled place the place rule. Relationship "
    'words are never removed: release "mom", "my brother", "grandpa", "my wife", "my son", "aunt", '
    '"cousin", "mamá", "mi hermano", "abuela".\n'
    '- DATES. Remove every absolute date element, the year included: a year ("2021", "\'98", "two '
    'thousand twenty-one"), a month ("October"), a holiday ("Halloween", "Christmas"), a day of the '
    'month ("the 14th"). These are never released: in "I had COVID in 2021" remove "2021". Keep a '
    'season ("summer", "invierno"), a day of the week ("Monday"), a time of day ("this morning"), and every '
    'relative or length-of-time expression ("2-3 weeks ago", "last year", "a few months", '
    '"yesterday", "for two years") -- release these.\n'
    '- AGES. Remove every age ("I\'m 73", "seventy-three years old", "in my sixties", "my 50th '
    'birthday"); never release one.\n'
    "- PLACES. Remove every place smaller than a country that the speaker connects to themselves or to "
    "people they know -- where they live, lived, work, travel, are treated or were born: a street, a "
    "neighbourhood, a city, a county, a state or province, a region, a named venue. A place that "
    'identifies nobody may be released with the key "place_reason" set to one of: historical ("the '
    'Roman Empire", "the Civil War in Virginia"), fictional (a setting in a book or film), '
    'public_landmark_or_general_knowledge ("the Colosseum", "a New York winter", "California poppies"), '
    "task_content (the task's own stimulus, or a description of it). A state or province is released "
    "only as historical. A place the speaker connects to themselves or their people stays removed, "
    'whatever else it is. A country may stay ("we moved from Mexico") unless, together with other details '
    "left in the text, it would help single the speaker out; then propose redacting it.\n"
    "- ORGANIZATIONS. Remove a specific named organization: an employer, a company, a school or "
    "university, a military unit, a hospital or clinic with a proper name, a church "
    '("USF voice center", "the United States Marine Corps", "Vanderbilt"). Keep a generic description '
    '("the voice center", "the hospital", "my school", "work") -- release it.\n'
    "- HEALTH CONDITIONS. Never remove a health condition and never propose one: conditions are released "
    "and never hold a recording back. List diagnoses under CONDITIONS for the record only.\n"
    "- LANGUAGE. The transcript may be in Spanish or another language; judge it in that language. The "
    'automatic detectors were built for English and often mark ordinary Spanish words ("pero", '
    '"estado", "familia") as names; release every removed word that is not a name, a place, a date, '
    "an age, an organization or an identifier.\n\n"
)


def redaction_policy_text() -> str:
    """The redaction policy as the reviewer is given it: the rules, then the remaining Safe Harbor identifiers.

    Returns:
        The policy text, the one source every prompt that asks what the policy removes is built from.
    """
    return _POLICY + _identifier_rule()


_POINT_8_OFF_TASK = (
    "8. Whether the participant stopped doing the task and talked about something else: a remark, a "
    "question, a story or a conversation that is not what the instructions ask for. Asking whether "
    'they are done ("Is that enough?") is some; talking at length about something else is extensive. In a '
    "non-lexical task (breathing, coughing, phonation, repeated syllables) any words the participant says "
    "are off-task.\n"
)
_OFF_TASK_PARTS = (
    "OFF_TASK_SPEECH: one of none, some, extensive (point 8).\n"
    "OFF_TASK_QUOTES: a JSON array of strings, each an exact passage, quoted from the ORIGINAL, where the "
    "participant talks off-task. Return [] when OFF_TASK_SPEECH is none; otherwise quote at least one.\n"
)
_POINT_8_PHRASES = (
    "8. This task asks for a list of items. Find every phrase in the recording. An item is a short unit, "
    "typically one to three words: a name, a number, a letter, an animal, whatever the instructions ask for. "
    "A phrase is clausal or non-item speech: a clause, a sentence, a comment, a question or a story rather "
    'than an item ("I don\'t know what else to say", "my dog\'s name is ...", "is that enough?"). Judge each '
    "stretch of the transcript by what it says; neither a pause nor the number of words decides it. Then judge "
    "whether the participant is uttering phrases instead of items: none where they list items and say no "
    'phrase, or only brief asides about the task itself ("um, let me think", "is that enough?"); occasional '
    "where a few phrases come between the items; predominant where phrases make up much of the recording, the "
    "list giving way to narration or conversation.\n"
)
_PHRASE_PARTS = (
    "PHRASES_INSTEAD_OF_ITEMS: one of none, occasional, predominant (point 8).\n"
    "PHRASE_QUOTES: a JSON array of strings quoting every phrase under point 8, one entry per phrase, each "
    "exactly as it appears in the ORIGINAL and never an item. Return [] only when there is no phrase at all; "
    "when PHRASES_INSTEAD_OF_ITEMS is occasional or predominant, quote at least one.\n"
)


_PROMPT = (
    "You are auditing one recording's transcript before it is released. You are given the "
    "ORIGINAL words as transcribed, and where an automatic redaction has already been applied, "
    "the RELEASED text it produced from the words the detectors read (every word outside the <task> and "
    "<nonlex> tags), in which every [CATEGORY] token marks removed text. Where no "
    "redaction was applied the RELEASED section says so, and nothing has been removed.\n\n"
    "Where the ORIGINAL is headed TRANSCRIPT, it is the whole recording as the speech recognisers heard it: "
    "every consensus word in order, written as plain words with annotations attached. The heading names "
    "each recogniser by a letter; C is the consensus itself. Words inside <task>...</task> are the task's "
    "own content (its stimulus, target words or syllables) and words inside <nonlex>...</nonlex> are "
    "non-lexical (fillers, sounds, bracketed markers); the automatic detectors did not read either, the "
    "redaction never removes them, and they are there so you read every word in its place. A word with "
    "something to say about it carries it in braces right after it, parts separated by ' | ':\n"
    "  pii CATEGORY by DETECTOR@READING, ... -- an automatic detector found that category on this word, "
    "reading the transcript the letter names (several categories are separated by ' ; ');\n"
    "  variant A=word B=word -- the recognisers read different words here;\n"
    "  insertion A=word B=- -- only some recognisers heard a word here ('-' is none);\n"
    "  agreement A=word B=word -- they agree on the word but spell it differently.\n"
    'For example, "my sister Maria{pii PERSON by gliner@C, presidio@A | variant A=Maria B=Mario} lives '
    'in <task>the cookie jar</task> <nonlex>um</nonlex> Ohio{pii LOCATION by presidio@C}" has a name two '
    "detectors found and the recognisers read differently, a place one detector found, three task words and "
    "a filler. A word without braces is one every recogniser read the same way and no detector marked. Use "
    "the readings to judge whether a word was said at all and what it was: a name only one recogniser read, "
    "where another read an ordinary word, may be a misreading. Quote words as the ORIGINAL writes them, "
    "without the braces, their contents or the tags.\n\n"
    + redaction_policy_text()
    + "\n\nJudge nine things independently.\n"
    "1. Whether the redaction, where one was applied, actually removed everything the policy removes.\n"
    "2. Whether the ORIGINAL words carry anything the policy removes at all, which is a separate "
    "question and the one no automatic detector here has asked.\n"
    "3. Whether the redaction removed more than the policy requires: a relationship word, a weekday, a "
    "relative or length-of-time expression, a country, a generic description, a public figure, a "
    "health condition, or an ordinary word a detector mistook for a name. A word or phrase that is the "
    "task's own stimulus, or a definition or description of it (in a vocabulary task, saying what a "
    '"gladiator" is: "a Roman fighter"), is not about the speaker and identifies nobody.\n'
    "4. Whether more than one person is speaking in this recording, judged from the words alone "
    "(turn-taking, instructions given, questions asked and answered) and weighed against the task's "
    'instructions. The participant talking to the examiner ("Is that enough?", "Should I keep '
    'going?") is still one speaker. Any words spoken by someone other than the participant make '
    "more_than_one, including a voice the task expects: an examiner giving or repeating instructions "
    '("You were given the text... you have up to five minutes") in a story recall or an interview, or '
    "someone saying a sentence first where the instructions allow the participant to repeat after them. "
    "Whenever you answer more_than_one, quote the words you attribute to each other person and say "
    "whether the task's instructions expect that voice; an expected voice is still another voice. Words "
    "addressed to the participant that give, restate or enforce the task's instructions (\"You were given "
    'the text", "I said you have up to five minutes", "Go ahead and start") are another person\'s '
    "voice unless the words themselves show the participant reading them aloud; do not assume the "
    "participant is repeating the instructions. When you cannot tell, answer unclear and still quote "
    "those words in OTHER_SPEAKERS.\n"
    "5. For the record only, every specific medical diagnosis the speaker attributes to themselves: a "
    "named disease, disorder or syndrome (Parkinson's disease, spasmodic dysphonia, multiple sclerosis, "
    "thyroid cancer, a synovial joint cyst, sleep apnea, asthma). Do not list symptoms or sensations (a "
    "change in my voice, coughing, tiredness, pain, a sore throat), feelings or moods (anxious, stressed, "
    "sad), everyday events or procedures (surgery, a fall, an accident, an allergy shot) or a medication "
    'or treatment on its own; list a treatment only through the diagnosis it names ("levodopa for my '
    "Parkinson's\" lists Parkinson's disease). A diagnosis is never removed and never goes in the "
    "PROPOSAL. Leave out one that is part of the task's own stimulus.\n"
    "6. Whether the task's instructions -- the INSTRUCTIONS GIVEN TO THE PARTICIPANT above -- are spoken "
    "in the recording, by anyone: an examiner giving or repeating them, or the participant reading them "
    'aloud or saying them back. Count a paraphrase as well as a verbatim reading: "You were given the '
    "test, read the text, have you familiarized? I said you have up to five minutes to read it as many "
    'times as you want" speaks the instructions "You are given a text. Read the text so you '
    'familiarize yourself with it. You have up to 5 minutes to read it as many times as you want". '
    'Words that only mention the task ("is that enough?", "I\'ll describe the picture", "five '
    "minutes\" in passing) do not. The task's stimulus -- what the participant was asked to say or "
    "recall -- is never instructions. Judge this apart from point 4.\n"
    "7. For every word an automatic detector marked (each word carrying a pii annotation), whether it is "
    "task content or a personal disclosure. Task content is what the INSTRUCTIONS ask the participant to "
    "say, or part of the STIMULUS, or an item the task asks for; a personal disclosure is the participant "
    "telling something about their own life. Weigh the NATURE OF THIS TASK: an open response invites "
    "personal content, so a name or a place said there is a real candidate for removal; in item "
    "generation or fluency the task asks for names, places and numbers, so an item said as part of the "
    "list usually is not; in reading or recall the given text's words are task content; in a non-lexical "
    "task no word is task content.\n"
    + _POINT_8_OFF_TASK
    + "9. Whether anyone besides the participant speaks, and who: an assistant, examiner or clinician "
    "(reading the instructions aloud, prompting, encouraging, counting, asking the participant to start "
    "or stop), or someone in the background (a conversation, a television, another person in the room "
    "not addressing the participant). Judge it from the words, as under point 4.\n\n"
    "Answer in exactly thirteen parts, each on its own line or block, in this order.\n"
    "REASONING: your full reasoning, in prose, including what you considered and rejected.\n"
    "REDACTION: one of complete, incomplete, not_applicable (use not_applicable when no "
    "redaction was applied).\n"
    "ORIGINAL: one of clean, carries_pii.\n"
    "SPEAKERS: one of one, more_than_one, unclear.\n"
    "OTHER_SPEAKERS: a JSON array, one element per other person you heard under point 4. Each element is an "
    'object with keys "text" (their exact words, quoted from the ORIGINAL), "expected" (true where the '
    'task\'s instructions expect that voice, false where they do not) and "why" (one sentence). Return [] '
    "when SPEAKERS is one; when it is more_than_one the array must quote at least one person, and when it is "
    "unclear it quotes the words you could not attribute to the participant.\n"
    "OTHER_SPEAKER_ROLE: one of none, assistant, background, unclear (point 9: none when only the "
    "participant speaks; assistant for an examiner, clinician or anyone addressing the participant about the "
    "task; background for anyone else).\n"
    "OTHER_SPEAKER_QUOTES: a JSON array of strings, each the exact words another person said, quoted from the "
    "ORIGINAL. Return [] when OTHER_SPEAKER_ROLE is none; otherwise quote at least one passage.\n"
    + _OFF_TASK_PARTS
    + "TASK_CONTENT_QUOTES: a JSON array of strings, each a word or phrase carrying a pii annotation that you "
    "judge under point 7 to be task content, quoted exactly from the ORIGINAL. Return [] when there is none.\n"
    "CONDITIONS: a JSON array of every diagnosis under point 5. Each element is an object with "
    'keys "text" (the exact words, quoted from the ORIGINAL) and "why" (one sentence: which diagnosis it is). '
    "Return [] when the speaker mentions none; the part is required either way.\n"
    "INSTRUCTIONS_SPOKEN: a JSON array of every passage under point 6. Each element is an object with "
    'keys "text" (the exact words, quoted from the ORIGINAL) and "why" (one sentence: which instruction it '
    "speaks). Return [] when the instructions are not spoken; the part is required either way.\n"
    "PROPOSAL: a JSON array giving the redaction you would apply instead. Each element is an "
    'object with keys "text" (the exact substring, quoted from the ORIGINAL), "action" (redact to '
    "remove it, release to stop removing text the current redaction removes unnecessarily), "
    '"category" (one of PERSON, LOCATION, DATE_TIME, ORGANIZATION, ID, CONTACT, OTHER; an age is '
    'DATE_TIME), "safe_harbor" (the letter of the Safe Harbor identifier it falls under, or "" for none), '
    '"why" (one sentence; for a release, why the policy lets it through: a public figure, a '
    "relationship word, a weekday or relative time, a country that singles nobody out, a generic "
    "description, a place that identifies nobody, task content, or an ordinary word mistaken for a name), "
    'and, on a release only, "relabel" (one of work_title, brand_or_product, organization, place, '
    'other_non_person, where a removed name is not a person; otherwise "") and "place_reason" (one of '
    "historical, fictional, public_landmark_or_general_knowledge, task_content, where it releases a place "
    'smaller than a country or anything that is the task\'s own content; otherwise ""). Return [] to leave '
    "the current redaction exactly as it is.\n"
    "Whenever you judge the redaction incomplete, or judge the ORIGINAL clean while the RELEASED text "
    "still removes words, or judge the ORIGINAL to carry something identifying, the PROPOSAL must name "
    "every word or phrase concerned, one entry each, quoted exactly as it appears in the ORIGINAL: "
    "release for removed words the policy lets through, redact for words that must go. A judgment of "
    "that kind with an empty PROPOSAL is not an answer. Never propose releasing a year, a month, a "
    "holiday or an age; those always stay removed (the task's own stimulus is never removed "
    "in the first place). A place smaller than a country is released only with its place_reason. If your "
    "REASONING names a venue, resort, hotel, clinic, hospital, employer, company, school, church or "
    "street, the PROPOSAL must carry an entry quoting it: redact a specific named one, or release a "
    "generic description with the reason.\n\n"
)


_ITEM_SET_PROMPT = _PROMPT.replace(_POINT_8_OFF_TASK, _POINT_8_PHRASES).replace(_OFF_TASK_PARTS, _PHRASE_PARTS)


def review_prompt(item_set: bool) -> str:
    """The reviewer's instructions.

    Args:
        item_set: Whether the declared task is an item-set task (the context's :data:`ITEM_SET`).

    Returns:
        The prompt; for an item-set task point 8 and its parts ask for every phrase the participant says, quoted,
        and whether they utter phrases instead of items (:data:`PHRASE_STATES`), in place of off-task speech.
    """
    return _ITEM_SET_PROMPT if item_set else _PROMPT


_BARE_READING = re.compile(r"[^\s{}|=<>;,]+")


def _reading(text: str | None) -> str:
    """One recogniser's word as an annotation writes it: bare, ``-`` where it read none, else JSON-quoted."""
    if text is None or text == "":
        return "-"
    return text if _BARE_READING.fullmatch(text) else json.dumps(text, ensure_ascii=False)


def _letters(count: int) -> list[str]:
    """The letters naming ``count`` recognisers: ``A``, ``B`` ... skipping ``C``, which names the consensus."""
    letters = [chr(code) for code in range(ord("A"), ord("Z") + 1) if chr(code) != "C"]
    return [letters[i] if i < len(letters) else f"R{i + 1}" for i in range(count)]


def transcript_block(transcript: Mapping[str, Any] | None) -> str:
    """The whole consensus transcript, as the reviewer reads it.

    Args:
        transcript: The context's :data:`CONSENSUS_TRANSCRIPT` mapping: ``recognisers``, the recogniser
            names, and ``words``, one ``[index, word, outcome, [each recogniser's word or None], kind,
            [[category, detector, reading], ...]]`` per consensus word in stream order, where ``kind`` is
            one of :data:`WORD_KINDS` and ``reading`` is a recogniser name or :data:`CONSENSUS_HAYSTACK`.

    Returns:
        A heading naming each recogniser's letter, then the words on one line: each word followed by its
        annotations in braces, and each run of task or non-lexical words inside a ``<task>`` or ``<nonlex>``
        tag. An agreement whose readings differ from the word only in case and punctuation carries no
        variation. Empty where there are no words.
    """
    if not transcript or not transcript.get("words"):
        return ""
    names = [str(name) for name in transcript.get("recognisers") or ()]
    letters = _letters(len(names))
    letter_of = {name: letter for name, letter in zip(names, letters)}
    letter_of[CONSENSUS_HAYSTACK] = "C"
    legend = ", ".join(f"{letter} = {name}" for letter, name in zip(letters, names))
    heading = f"TRANSCRIPT (recognisers: {legend}; C = the consensus):"
    tags = {WORD_TASK: "task", WORD_NON_LEXICAL: "nonlex"}
    pieces: list[str] = []
    open_kind = WORD_SCANNED
    for _index, text, outcome, readings, kind, findings in transcript["words"]:
        if kind != open_kind:
            if open_kind in tags:
                pieces[-1] += f"</{tags[open_kind]}>"
            open_kind = kind if kind in tags else WORD_SCANNED
            opening = f"<{tags[open_kind]}>" if open_kind in tags else ""
        else:
            opening = ""
        parts: list[str] = []
        by_category: dict[str, list[str]] = {}
        for category, detector, haystack in findings or ():
            source = f"{detector}@{letter_of.get(str(haystack), str(haystack))}"
            listed = by_category.setdefault(str(category), [])
            if source not in listed:
                listed.append(source)
        if by_category:
            parts.append("pii " + " ; ".join(f"{cat} by {', '.join(src)}" for cat, src in by_category.items()))
        heard = [None if reading is None else str(reading) for reading in readings]
        same = outcome == "agreement" and all(
            reading is not None and _quote_tokens(reading) == _quote_tokens(str(text)) for reading in heard
        )
        if heard and not same:
            each = " ".join(f"{letters[i]}={_reading(reading)}" for i, reading in enumerate(heard))
            parts.append(f"{outcome or 'unaligned'} {each}")
        note = "{" + " | ".join(parts) + "}" if parts else ""
        pieces.append(f"{opening}{text}{note}")
    if open_kind in tags and pieces:
        pieces[-1] += f"</{tags[open_kind]}>"
    return heading + "\n" + " ".join(pieces)


def transcript_text(transcript: Mapping[str, Any] | None) -> str:
    """The whole consensus transcript as plain words, the text a reviewer quote is checked against.

    Args:
        transcript: The context's :data:`CONSENSUS_TRANSCRIPT` mapping.

    Returns:
        Every word's consensus text, space-joined; empty where there are no words.
    """
    if not transcript:
        return ""
    return " ".join(str(word[1]) for word in transcript.get("words") or () if str(word[1]))


def quotable_texts(original: str, context: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The texts a reviewer quote may come from: the whole transcript where one was given, and the residue.

    Args:
        original: The lexical residue the PII scan read.
        context: The context the reading was composed from.

    Returns:
        The non-empty texts.
    """
    whole = transcript_text((context or {}).get(CONSENSUS_TRANSCRIPT))
    return tuple(text for text in (whole, original) if text)


def review_inputs(context: Mapping[str, Any] | None) -> dict[str, Any]:
    """Which of the inputs the fold weighs a reviewer release by a reading was given.

    Args:
        context: The context the reading was composed from (:func:`_compose`).

    Returns:
        ``version`` (:data:`REVIEW_INPUTS_VERSION`), ``prompt_version``, ``task`` (the declared task and
        its instructions or stimulus), ``instructions`` (the task's instructions), ``full_transcript``
        (every consensus word, task and non-lexical words included), ``pii_annotations`` (each word's
        detector findings with the reading they were found on), ``asr_readings`` (each of two or more
        recognisers' word wherever they differ, with the outcome), and ``recognisers``, ``words_n``,
        ``pii_words_n`` and ``variation_words_n``.
    """
    facts = dict(context or {})
    transcript = facts.get(CONSENSUS_TRANSCRIPT) or {}
    words = list(transcript.get("words") or ())
    recognisers = [str(name) for name in transcript.get("recognisers") or ()]
    return {
        "version": REVIEW_INPUTS_VERSION,
        "prompt_version": PROMPT_VERSION,
        "task": bool(facts.get("task")) and bool(facts.get("instructions") or facts.get("asked_to_say")),
        "instructions": bool(str(facts.get("instructions") or "").strip()),
        "full_transcript": bool(words) and all(len(word) == 6 and word[4] in WORD_KINDS for word in words),
        "pii_annotations": bool(words) and "findings_n" in transcript,
        "asr_readings": bool(words) and len(recognisers) >= 2,
        "recognisers": recognisers,
        "words_n": len(words),
        "pii_words_n": sum(1 for word in words if len(word) == 6 and word[5]),
        "variation_words_n": sum(1 for word in words if word[2] != "agreement"),
    }


def review_inputs_complete(inputs: Mapping[str, Any] | None) -> bool:
    """Whether a reading recorded the task, its instructions, and the whole transcript with its PII and readings.

    Args:
        inputs: The reading's ``review_inputs`` record, or None where it recorded none.

    Returns:
        True only for a record of :data:`REVIEW_INPUTS_VERSION` or later with every one of
        :data:`REQUIRED_REVIEW_INPUTS` set.
    """
    if not isinstance(inputs, Mapping) or int(inputs.get("version") or 0) < REVIEW_INPUTS_VERSION:
        return False
    return all(bool(inputs.get(key)) for key in REQUIRED_REVIEW_INPUTS)


def _compose(
    original: str, redacted: str | None, context: Mapping[str, Any] | None = None, feedback: str | None = None
) -> str:
    """The task's own facts and the two transcripts, as one request body.

    Args:
        original: The transcript as the recording's words were read.
        redacted: The text an applied redaction produced, or None where none was applied.
        context: What the recording declares about itself -- ``task`` (the family, whose nature
            :func:`task_nature_description` sends), ``task_name``, ``speech_type``, ``language``,
            ``instructions``, ``asked_to_say``, ``declared_names``, ``task_words`` and
            :data:`CONSENSUS_TRANSCRIPT`. Any key absent or empty is omitted rather than sent empty. Where
            the transcript is given it is the ORIGINAL (:func:`transcript_block`) and ``original`` is not
            sent.
        feedback: What was wrong with the previous round's answer, which this round must correct; None
            on a first round.

    Returns:
        The body the prompt is prefixed to.
    """
    lines: list[str] = []
    facts = dict(context or {})
    if facts.get("task"):
        lines.append(f"TASK: {facts['task']}")
    if facts.get("task_name") and facts.get("task_name") != facts.get("task"):
        lines.append(f"TASK NAME THE RECORDING DECLARES: {facts['task_name']}")
    nature = task_nature_description(facts.get("task"))
    if nature:
        lines.append(f"NATURE OF THIS TASK: {nature}")
    if facts.get("speech_type"):
        lines.append(f"SPEECH TYPE: {facts['speech_type']}")
    if facts.get("language"):
        lines.append(f"LANGUAGE THE RECORDING DECLARES: {facts['language']}")
    if facts.get("instructions"):
        lines.append(f"INSTRUCTIONS GIVEN TO THE PARTICIPANT: {facts['instructions']}")
    if facts.get("asked_to_say"):
        lines.append(f"STIMULUS THE PARTICIPANT WAS GIVEN TO SAY OR RECALL: {facts['asked_to_say']}")
    names = facts.get("declared_names") or ()
    if names:
        lines.append(f"NAMES THE TASK'S OWN MATERIALS CONTAIN: {', '.join(str(name) for name in names)}")
    words = facts.get("task_words") or ()
    if words:
        lines.append(f"WORDS AND PHRASES THE TASK ITSELF IS MADE OF: {', '.join(str(word) for word in words)}")
    guidance = task_guidance(facts.get("task"))
    if guidance:
        lines.append(f"WHAT IS TASK CONTENT IN THIS TASK: {guidance}")
    if lines:
        lines.append(
            "Those lines are the task. Words that are the task's own stimulus, or that the instructions "
            "ask the participant to say, are not identifying for being said: a character, a number or a "
            "date in the stimulus identifies nobody. Anything the speaker adds of their own -- their "
            "name, where they live, who they know, their health -- is judged as it would be anywhere, "
            "and instructions read aloud by someone else are a sign of more than one speaker."
        )
        lines.append("")
    block = transcript_block(facts.get(CONSENSUS_TRANSCRIPT)) or original
    released = redacted if redacted is not None else "(no redaction was applied to this recording)"
    body = "\n".join(lines) + f"ORIGINAL:\n{block}\n\nRELEASED:\n{released}\n"
    if feedback:
        body += f"\nYOUR PREVIOUS ANSWER HAD A PROBLEM TO CORRECT: {feedback}\n"
    return body


@dataclass
class ReviewProposal:
    """One change the model would make to what is removed from the recording.

    Attributes:
        text: The substring it named, quoted from the original.
        action: :data:`REDACT` to remove it, :data:`RELEASE` to stop removing it.
        category: Its category, uppercased.
        why: Its one-sentence reason.
        safe_harbor: The Safe Harbor identifier letter the model named for it, or the empty string.
        relabel: On a release, what a removed name is instead of a person, lower-cased as written; one
            of :data:`RELABELS` when valid, the empty string where none was given.
        place_reason: On a release, why a place smaller than a country, or the task's own content,
            identifies nobody, lower-cased as written; one of :data:`PLACE_REASONS` when valid, the
            empty string where none was given.
    """

    text: str
    action: str
    category: str
    why: str
    safe_harbor: str = ""
    relabel: str = ""
    place_reason: str = ""


@dataclass
class ReviewCondition:
    """A health condition the reviewer lists for the record. Never masked.

    Attributes:
        text: The words, quoted from the original.
        why: Which diagnosis it is, in one sentence.
    """

    text: str
    why: str = ""


@dataclass
class OtherSpeaker:
    """Words the reviewer attributes to someone other than the participant.

    Attributes:
        text: The words, quoted from the original.
        expected: Whether the task's instructions expect that voice.
        why: Its one-sentence reason.
    """

    text: str
    expected: bool
    why: str = ""


@dataclass
class ReviewResult:
    """One pass of the reviewer over one text.

    Attributes:
        available: Whether the model ran at all. ``False`` with a populated ``failure`` is the only
            honest answer when it did not — an empty ``findings`` under ``available=False`` reads
            identically to "the model found nothing", which is the one wrong answer here.
        reasoning: The model's chain of thought, verbatim. The point of the step.
        redaction: One of :data:`REDACTION_STATES`, or the empty string where it answered nothing
            parsable. Whether the applied redaction removed what identifies the speaker.
        original: One of :data:`ORIGINAL_STATES`, or the empty string. Whether the recording's own
            words carry identifying content, which is a separate question from the redaction's.
        speakers: One of :data:`SPEAKER_STATES`, or the empty string. How many people the words
            show speaking in the recording. A reading about the recording's content, never about
            the processing run.
        proposal: The redaction it would apply instead, as text spans.
        conditions: The health conditions the answer's CONDITIONS part listed, for the record.
        other_speakers: The words it attributes to anyone other than the participant, each with whether
            the task expects that voice.
        instructions_spoken: The passages the INSTRUCTIONS_SPOKEN part quotes, where the task's own
            instructions are spoken in the recording.
        instructions_answered: Whether the answer carried its INSTRUCTIONS_SPOKEN part, an empty list
            included.
        off_task_speech: One of :data:`OFF_TASK_STATES`, or None where the answer gave none of them.
        off_task_quotes: The passages the OFF_TASK_QUOTES part quotes.
        phrases_instead_of_items: One of :data:`PHRASE_STATES`, or None where the answer gave none of them.
        phrase_quotes: The phrases the PHRASE_QUOTES part quotes.
        other_speaker: One of :data:`OTHER_SPEAKER_ROLES`, or None where the answer gave none of them.
        other_speaker_quotes: The passages the OTHER_SPEAKER_QUOTES part quotes.
        task_content_quotes: The PII-annotated words the TASK_CONTENT_QUOTES part judges task content.
        conditions_answered: Whether the answer carried its CONDITIONS part, an empty list included.
            :func:`parse_completion` decides it for every answer the model gave; a result built any
            other way is taken as answered, and a failure is never read for it.
        failure: ``None`` on success; otherwise why the reviewer did not run.
        model_id: The repo the review was asked of.
        revision: The resolved 40-hex commit it loaded, or ``None``.
        raw: The model's completion, unparsed, when parsing recovered nothing.
        elapsed_s: Wall-clock seconds this call took, a load it paid for included.
        load_s: How many of those seconds went on starting the worker and loading the weights.
            ``0.0`` when an already-running worker served the call.
        input_tokens: How many tokens the prompt and the two transcripts came to, or ``None`` when
            the model did not answer. Reported beside ``output_tokens`` because the contract reads
            two texts and the input side is what grew.
        output_tokens: How many tokens the model generated, or ``None`` when it did not answer.
        peak_reserved_mib: Device memory the worker's allocator held at the end of the generation,
            before it was emptied. ``0`` on a CPU worker and on a call that did not answer.
        resident_mib: Device memory it holds between reviews — the weights and nothing else, which
            is what a second process on the same card has to live beside.
        engine: The engine that answered, as the worker reported it
            (:func:`~senselab.text.tasks.pii_detection.redaction_review_vllm.engine_identity`); empty where none
            answered.
    """

    available: bool
    reasoning: str = ""
    redaction: str = ""
    original: str = ""
    speakers: str = ""
    proposal: list[ReviewProposal] = field(default_factory=list)
    conditions: list[ReviewCondition] = field(default_factory=list)
    other_speakers: list[OtherSpeaker] = field(default_factory=list)
    conditions_answered: bool = True
    instructions_spoken: list[str] = field(default_factory=list)
    instructions_answered: bool = True
    off_task_speech: Optional[str] = None
    off_task_quotes: list[str] = field(default_factory=list)
    phrases_instead_of_items: Optional[str] = None
    phrase_quotes: list[str] = field(default_factory=list)
    other_speaker: Optional[str] = None
    other_speaker_quotes: list[str] = field(default_factory=list)
    task_content_quotes: list[str] = field(default_factory=list)
    failure: Optional[str] = None
    model_id: str = ""
    revision: Optional[str] = None
    raw: str = ""
    elapsed_s: float = 0.0
    load_s: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    peak_reserved_mib: int = 0
    resident_mib: int = 0
    engine: dict[str, Any] = field(default_factory=dict)


# A line protocol over the worker's stdin/stdout, one JSON object per line.
#
# Load payload, sent once (stdin):
#   {
#     "model_id": str,
#     "model_path": str | None,   # staged snapshot dir, named by the commit it holds
#     "revision": str | None,     # resolved commit SHA, never a mutable ref
#   }
# Load reply (stdout):        {"ready": True, "revision": str | None, "load_s": float}
#
# Review request (stdin):     {"text": str, "prompt": str, "max_new_tokens": int}
# Review reply (stdout):      {"completion": str, "generate_s": float, "output_tokens": int,
#                              "peak_reserved_mib": int, "resident_mib": int}
# Stop request (stdin):       {"stop": True}
#
# The worker empties the CUDA caching allocator after each generation, so ``resident_mib`` is what
# it holds *between* reviews rather than its high-water mark, and reports both. A worker that
# outlives a call also outlives its memory, and it does not share an address space with the graph
# that wants the rest of the card, so the steady-state figure is the one a second process has to
# live beside. What that figure is for this checkpoint, and why emptying the cache moves it far
# less than it looks like it should, is measured in
# ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
#
# Every reply carries the ``_WORKER_MARKER`` prefix, so a library writing to the real stdout cannot
# be mistaken for one; ``sys.stdout`` is redirected to stderr before any heavy import for the same
# reason. A raised exception is reported as {"error": {"type": str, "message": str}} and ends the
# worker: a failure mid-generation is an out-of-memory or a dead CUDA context far more often than it
# is a bad request, and a process in that state cannot be trusted with the next one.
_WORKER_MARKER = "@@SENSELAB_REVIEW@@"

_REVIEW_WORKER_SCRIPT = (
    r"""
import json, os, re, sys, time

MARKER = "%s"
_replies = sys.stdout
sys.stdout = sys.stderr


def emit(payload):
    _replies.write(MARKER + json.dumps(payload) + "\n")
    _replies.flush()


def load(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = args["model_id"]
    revision = args.get("revision")
    target = args.get("model_path") or model_id
    loaded_revision = revision
    if args.get("model_path"):
        basename = os.path.basename(os.path.normpath(args["model_path"]))
        loaded_revision = basename if re.fullmatch(r"[0-9a-f]{40}", basename) else revision
        tokenizer = AutoTokenizer.from_pretrained(target)
        model = AutoModelForCausalLM.from_pretrained(target, dtype="auto", device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(target, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(target, revision=revision, dtype="auto", device_map="auto")
    model.eval()
    return torch, tokenizer, model, loaded_revision


def main():
    started = time.monotonic()
    torch, tokenizer, model, loaded_revision = load(json.loads(sys.stdin.readline()))
    emit({"ready": True, "revision": loaded_revision, "load_s": round(time.monotonic() - started, 3)})
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        if not line.strip():
            continue
        request = json.loads(line)
        if request.get("stop"):
            return
        began = time.monotonic()
        messages = [{"role": "user", "content": request["prompt"] + request["text"]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs, max_new_tokens=int(request["max_new_tokens"]), do_sample=False
            )
        answer = generated[0][inputs["input_ids"].shape[-1]:]
        completion = tokenizer.decode(answer, skip_special_tokens=True)
        tokens = int(answer.shape[-1])
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        generate_s = round(time.monotonic() - began, 3)
        peak, resident = 0, 0
        if torch.cuda.is_available():
            peak = int(torch.cuda.memory_reserved() / 2**20)
            del answer, generated, inputs
            torch.cuda.empty_cache()
            resident = int(torch.cuda.memory_reserved() / 2**20)
        emit(
            {
                "completion": completion,
                "generate_s": generate_s,
                "input_tokens": prompt_tokens,
                "output_tokens": tokens,
                "peak_reserved_mib": peak,
                "resident_mib": resident,
            }
        )


try:
    main()
except Exception as exc:
    emit({"error": {"type": type(exc).__name__, "message": str(exc)}})
    sys.exit(1)
"""
    % _WORKER_MARKER
)


def _staged_snapshot(repo_id: str, revision: str) -> Optional[str]:
    """The local snapshot directory for a staged commit, or ``None`` when it could not be staged.

    Args:
        repo_id: The HuggingFace repo id.
        revision: The resolved 40-hex commit SHA.

    Returns:
        The ``snapshots/<sha>/`` directory as a string, or ``None`` — in which case the worker keeps
        the repo id as its load target and loads online with the same SHA.
    """
    from senselab.utils.dependencies import resolve_model

    try:
        _sha, snapshot = resolve_model(repo_id, revision)
    except Exception as exc:  # noqa: BLE001 — an unstageable model is a fallback, not a crash
        logger.warning(f"redaction review: {repo_id}@{revision} could not be staged ({exc}); loading online.")
        return None
    return str(snapshot)


@dataclass(frozen=True)
class ParsedCompletion:
    """The model's answer, split into its six parts.

    Attributes:
        reasoning: The chain of thought, verbatim.
        redaction: One of :data:`REDACTION_STATES`, or the empty string where none was stated.
        original: One of :data:`ORIGINAL_STATES`, or the empty string.
        speakers: One of :data:`SPEAKER_STATES`, or the empty string.
        proposal: The parsed proposal array; empty where it was missing or malformed.
        conditions: The CONDITIONS part's entries that quote words.
        conditions_answered: Whether the answer carried a CONDITIONS part whose array parsed, an
            empty one included.
        other_speakers: The OTHER_SPEAKERS part's entries that quote words.
        instructions_spoken: The INSTRUCTIONS_SPOKEN part's quoted passages.
        instructions_answered: Whether the answer carried an INSTRUCTIONS_SPOKEN part whose array parsed.
        off_task_speech: One of :data:`OFF_TASK_STATES`, or None.
        off_task_quotes: The OFF_TASK_QUOTES part's quoted passages.
        phrases_instead_of_items: One of :data:`PHRASE_STATES`, or None.
        phrase_quotes: The PHRASE_QUOTES part's quoted phrases.
        other_speaker: One of :data:`OTHER_SPEAKER_ROLES`, or None.
        other_speaker_quotes: The OTHER_SPEAKER_QUOTES part's quoted passages.
        task_content_quotes: The TASK_CONTENT_QUOTES part's quoted words.
    """

    reasoning: str
    redaction: str = ""
    original: str = ""
    speakers: str = ""
    proposal: list[ReviewProposal] = field(default_factory=list)
    conditions: list[ReviewCondition] = field(default_factory=list)
    conditions_answered: bool = False
    other_speakers: list[OtherSpeaker] = field(default_factory=list)
    instructions_spoken: list[str] = field(default_factory=list)
    instructions_answered: bool = False
    off_task_speech: Optional[str] = None
    off_task_quotes: list[str] = field(default_factory=list)
    phrases_instead_of_items: Optional[str] = None
    phrase_quotes: list[str] = field(default_factory=list)
    other_speaker: Optional[str] = None
    other_speaker_quotes: list[str] = field(default_factory=list)
    task_content_quotes: list[str] = field(default_factory=list)


CONDITION = "CONDITION"
"""The category a health condition carried as a proposal entry under prompt version 6 and earlier."""


def _labelled(completion: str, heading: str, allowed: Sequence[str]) -> str:
    """The one-word answer under a heading, where it is one of the allowed words.

    Args:
        completion: The model's raw text.
        heading: The heading to read, including its colon.
        allowed: The words this heading accepts.

    Returns:
        The word, lowercased, or the empty string where the heading is missing or the word is not
        one of ``allowed``. An unrecognised word is not guessed at: the empty string says the model
        did not answer this question, which is different from any of the answers it could give.
    """
    marker = -1
    for found in re.finditer(re.escape(heading), completion):
        if found.start() == 0 or not completion[found.start() - 1].isalnum() and completion[found.start() - 1] != "_":
            marker = found.start()
    if marker == -1:
        return ""
    line = completion[marker + len(heading) :].splitlines()[0] if completion[marker:].splitlines() else ""
    word = line.strip().strip(".").strip().lower()
    return word if word in allowed else ""


def _array(segment: str) -> list[Any] | None:
    """The JSON array a segment of the answer holds, or None where it holds none that parses.

    Args:
        segment: The text after one heading and before the next.

    Returns:
        The parsed list, or None.
    """
    start, end = segment.find("["), segment.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        parsed = json.loads(segment[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, list) else None


def _heading_at(completion: str, heading: str) -> int:
    """The position of a heading's last occurrence that is not the tail of a longer heading, or -1."""
    marker = -1
    for found in re.finditer(re.escape(heading), completion):
        before = completion[found.start() - 1] if found.start() else ""
        if not before or (not before.isalnum() and before != "_"):
            marker = found.start()
    return marker


def _quotes(parsed: list[Any] | None) -> list[str]:
    """The non-empty strings of a parsed array, an element's ``text`` read where it is an object."""
    quotes: list[str] = []
    for item in parsed or ():
        text = item.get("text") if isinstance(item, dict) else item
        if isinstance(text, str) and text.strip():
            quotes.append(text)
    return quotes


_ARRAY_HEADINGS = (
    _PROPOSAL_HEADING,
    _CONDITIONS_HEADING,
    _OTHER_SPEAKERS_HEADING,
    _INSTRUCTIONS_SPOKEN_HEADING,
    _OTHER_SPEAKER_QUOTES_HEADING,
    _OFF_TASK_QUOTES_HEADING,
    _PHRASE_QUOTES_HEADING,
    _TASK_CONTENT_QUOTES_HEADING,
)
_LABEL_HEADINGS = (
    _REDACTION_HEADING,
    _ORIGINAL_HEADING,
    _SPEAKERS_HEADING,
    _OTHER_SPEAKER_ROLE_HEADING,
    _OFF_TASK_SPEECH_HEADING,
    _PHRASES_HEADING,
)


def parse_completion(completion: str) -> ParsedCompletion:
    """Split the model's answer into its reasoning, its judgments, its quotes and its proposal.

    Each array is looked for after the last occurrence of its heading, up to the next heading of any
    part, rather than at the first ``[`` in the completion, because the reasoning routinely quotes the
    transcript's own ``[CATEGORY]`` placeholders. The reasoning ends at the first judgment heading, so
    a one-word answer never reads as prose.

    Args:
        completion: The model's raw text.

    Returns:
        The parts. The reasoning is returned even when everything else is missing or unparsable,
        because the reasoning is what the step exists to capture. A malformed array yields no entries
        rather than raising -- a caller reads ``available`` to tell that from a reviewer that read the
        text and would change nothing, and ``conditions_answered`` to tell a missing CONDITIONS part
        from one that listed none. A one-word part outside its allowed words reads as the empty string
        for ``redaction``, ``original`` and ``speakers``, and as None for ``off_task_speech``,
        ``phrases_instead_of_items`` and ``other_speaker``.
    """
    at = {heading: _heading_at(completion, heading) for heading in _ARRAY_HEADINGS + _LABEL_HEADINGS}
    found = sorted(position for position in at.values() if position != -1)

    def segment(heading: str) -> str:
        start = at[heading]
        end = min((position for position in found if position > start), default=len(completion))
        return completion[start + len(heading) : end]

    def parsed(heading: str) -> list[Any] | None:
        return _array(segment(heading)) if at[heading] != -1 else None

    proposal_at = at[_PROPOSAL_HEADING]
    if proposal_at != -1:
        proposal_text = segment(_PROPOSAL_HEADING)
        head = completion[:proposal_at]
    else:
        start = completion.find("[")
        cut = min(
            (at[heading] for heading in _ARRAY_HEADINGS if heading != _PROPOSAL_HEADING and at[heading] != -1),
            default=-1,
        )
        proposal_text = completion if cut == -1 else completion[:cut]
        head = completion if start == -1 else completion[:start]
    conditions_parsed = parsed(_CONDITIONS_HEADING)
    others_parsed = parsed(_OTHER_SPEAKERS_HEADING)
    spoken_parsed = parsed(_INSTRUCTIONS_SPOKEN_HEADING)
    cuts = [head.find(heading) for heading in _LABEL_HEADINGS + _ARRAY_HEADINGS]
    first = min((cut for cut in cuts if cut != -1), default=-1)
    reasoning = (head if first == -1 else head[:first]).replace(_REASONING_HEADING, " ")

    proposal: list[ReviewProposal] = []
    for item in _array(proposal_text) or ():
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        action = str(item.get("action") or REDACT).strip().lower()
        if not isinstance(text, str) or not text.strip() or action not in PROPOSAL_ACTIONS:
            continue
        proposal.append(
            ReviewProposal(
                text=text,
                action=action,
                category=str(item.get("category") or "OTHER").upper(),
                why=str(item.get("why") or ""),
                safe_harbor=str(item.get("safe_harbor") or "").strip().upper()[:1],
                relabel=str(item.get("relabel") or "").strip().lower(),
                place_reason=str(item.get("place_reason") or "").strip().lower(),
            )
        )
    conditions: list[ReviewCondition] = []
    listed: set[str] = set()
    for item in conditions_parsed or ():
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip() or _normalised(text) in listed:
            continue
        listed.add(_normalised(text))
        conditions.append(ReviewCondition(text=text, why=str(item.get("why") or "") if isinstance(item, dict) else ""))
    other_speakers = [
        OtherSpeaker(text=str(item["text"]), expected=item.get("expected") is True, why=str(item.get("why") or ""))
        for item in others_parsed or ()
        if isinstance(item, dict) and isinstance(item.get("text"), str) and str(item["text"]).strip()
    ]
    return ParsedCompletion(
        reasoning=reasoning.strip(),
        redaction=_labelled(completion, _REDACTION_HEADING, REDACTION_STATES),
        original=_labelled(completion, _ORIGINAL_HEADING, ORIGINAL_STATES),
        speakers=_labelled(completion, _SPEAKERS_HEADING, SPEAKER_STATES),
        proposal=proposal,
        conditions=conditions,
        conditions_answered=conditions_parsed is not None,
        other_speakers=other_speakers,
        instructions_spoken=_quotes(spoken_parsed),
        instructions_answered=spoken_parsed is not None,
        off_task_speech=_labelled(completion, _OFF_TASK_SPEECH_HEADING, OFF_TASK_STATES) or None,
        off_task_quotes=_quotes(parsed(_OFF_TASK_QUOTES_HEADING)),
        phrases_instead_of_items=_labelled(completion, _PHRASES_HEADING, PHRASE_STATES) or None,
        phrase_quotes=_quotes(parsed(_PHRASE_QUOTES_HEADING)),
        other_speaker=_labelled(completion, _OTHER_SPEAKER_ROLE_HEADING, OTHER_SPEAKER_ROLES) or None,
        other_speaker_quotes=_quotes(parsed(_OTHER_SPEAKER_QUOTES_HEADING)),
        task_content_quotes=_quotes(parsed(_TASK_CONTENT_QUOTES_HEADING)),
    )


_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_+]*\]")
_SPACES = re.compile(r"\s+")


def _normalised(text: str) -> str:
    return _SPACES.sub(" ", text.replace("\u2019", "'")).strip().casefold()


_TOKEN = re.compile(r"[a-z0-9]+")


def _quote_tokens(text: str) -> list[str]:
    """A quote's words as compared: lower-cased, apostrophes dropped, other punctuation and hyphens as breaks."""
    folded = unicodedata.normalize("NFKD", _normalised(text)).encode("ascii", "ignore").decode()
    return _TOKEN.findall(folded.replace("'", ""))


def _token_close(a: str, b: str) -> bool:
    if a == b:
        return True
    if min(len(a), len(b)) < 5:
        return False
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def quote_occurs(quote: str, original: str | Sequence[str]) -> bool:
    """Whether a quote occurs in the ORIGINAL, allowing case, punctuation, apostrophe, hyphen and spacing.

    A word of five letters or more may also differ by a transcription spelling (similarity 0.8).

    Args:
        quote: The quoted words.
        original: The ORIGINAL text, or several texts any one of which may hold the quote
            (:func:`quotable_texts`).

    Returns:
        True where the quote's words occur, in order and contiguously, in one text's words.
    """
    wanted = _quote_tokens(quote)
    if not wanted:
        return True
    span = len(wanted)
    for text in (original,) if isinstance(original, str) else original:
        words = _quote_tokens(text)
        if any(
            all(_token_close(word, want) for word, want in zip(words[start : start + span], wanted))
            for start in range(len(words) - span + 1)
        ):
            return True
    return False


_IDENTIFIER_CUE = re.compile(
    r"\b(venue|resort|hotel|hotel chain|motel|clinic|hospital|employer|company|workplace|school|"
    r"university|college|church|street|avenue|road|restaurant)\b",
    re.IGNORECASE,
)
_QUOTED = re.compile(r"[\"\u201c]([^\"\u201d]{2,60})[\"\u201d]")


def _named_unproposed(reasoning: str, proposal: Sequence[ReviewProposal], original: str | Sequence[str]) -> list[str]:
    """Quoted names the reasoning calls a venue, employer, school or street with no PROPOSAL entry quoting them."""
    proposed = [_quote_tokens(entry.text) for entry in proposal]
    missing: list[str] = []
    for sentence in re.split(r"(?<=[.;])\s+", reasoning):
        for match in _QUOTED.finditer(sentence):
            following = sentence[match.end() : match.end() + 30].split('"')[0].split("\u201c")[0]
            preceding = sentence[max(0, match.start() - 30) : match.start()].split('"')[-1]
            if not (_IDENTIFIER_CUE.search(following) or _IDENTIFIER_CUE.search(preceding)):
                continue
            name = match.group(1).strip()
            tokens = _quote_tokens(name)
            if not tokens or not any(ch.isupper() for ch in name[:1]) or not quote_occurs(name, original):
                continue
            covered = any(
                len(entry) >= len(tokens)
                and any(entry[i : i + len(tokens)] == tokens for i in range(len(entry) - len(tokens) + 1))
                for entry in proposed
            )
            if not covered and name not in missing:
                missing.append(name)
    return missing


def answer_problem(result: "ReviewResult", original: str | Sequence[str], redacted: str | None) -> str | None:
    """What makes an answer unusable under the prompt's own rule, as feedback for another round.

    Args:
        result: One round's result.
        original: The ORIGINAL the round read, or the texts its quotes may come from (:func:`quotable_texts`).
        redacted: The RELEASED text it read, or None where no redaction was applied.

    Returns:
        None where the answer is usable. Otherwise one sentence naming the problem: a judgment that
        asks for the redaction to change (incomplete, clean original over removed words, or words
        that carry something identifying where no complete redaction covers them) with an empty
        proposal, a release of something the policy always removes (:func:`dated`, :func:`stated`), a
        relabel or place reason outside the allowed values, a place released without a place reason, a
        name or a place released without a reason, proposal quotes that do not occur in the ORIGINAL, or
        an answer without its CONDITIONS or INSTRUCTIONS_SPOKEN part, or an off-task or other-speaker
        judgment that quotes nothing.
    """
    if not result.available:
        return None
    if not result.conditions_answered:
        return (
            "your answer had no CONDITIONS part; add it as a JSON array listing every specific medical "
            "diagnosis the speaker attributes to themselves, quoted exactly from the ORIGINAL, or [] when "
            "there is none"
        )
    if not result.instructions_answered:
        return (
            "your answer had no INSTRUCTIONS_SPOKEN part; add it as a JSON array quoting, exactly from the "
            "ORIGINAL, every passage where the task's instructions are spoken, verbatim or paraphrased, by "
            "anyone, or [] when there is none"
        )
    if result.off_task_speech in ("some", "extensive") and not result.off_task_quotes:
        return (
            f"you judged OFF_TASK_SPEECH {result.off_task_speech} but OFF_TASK_QUOTES quoted nothing; quote each "
            "off-task passage exactly from the ORIGINAL"
        )
    if result.phrases_instead_of_items in (PHRASES_OCCASIONAL, PHRASES_PREDOMINANT) and not result.phrase_quotes:
        return (
            f"you judged PHRASES_INSTEAD_OF_ITEMS {result.phrases_instead_of_items} but PHRASE_QUOTES quoted "
            "nothing; quote each phrase exactly from the ORIGINAL"
        )
    if result.other_speaker in ("assistant", "background") and not result.other_speaker_quotes:
        return (
            f"you judged OTHER_SPEAKER_ROLE {result.other_speaker} but OTHER_SPEAKER_QUOTES quoted nothing; quote "
            "what the other person said exactly from the ORIGINAL"
        )
    if result.speakers == "more_than_one" and not result.other_speakers:
        return (
            "you judged more_than_one speaker but OTHER_SPEAKERS quoted no words; quote, exactly from the "
            "ORIGINAL, the words you attribute to each other person and say whether the task's instructions "
            "expect that voice"
        )
    masked = redacted is not None and bool(_PLACEHOLDER.search(redacted))
    judged = []
    if result.redaction == "incomplete":
        judged.append("the redaction incomplete")
    if result.original == "carries_pii" and result.redaction != "complete":
        judged.append("the ORIGINAL to carry something identifying")
    if result.original == "clean" and masked:
        judged.append("the ORIGINAL clean while the RELEASED text still removes words")
    if judged and not result.proposal:
        return (
            f"you judged {' and '.join(judged)} but the PROPOSAL listed no words; list each word or phrase "
            "to release (unmask) or redact (mask), quoted exactly from the ORIGINAL"
        )
    releases = [entry for entry in result.proposal if entry.action == RELEASE]
    unknown = [
        entry.text
        for entry in releases
        if (entry.relabel and entry.relabel not in RELABELS)
        or (entry.place_reason and entry.place_reason not in PLACE_REASONS)
    ]
    if unknown:
        quoted = ", ".join(json.dumps(text) for text in unknown)
        return (
            f"these releases give a relabel or place_reason outside the allowed values: {quoted}; relabel is one "
            f"of {', '.join(RELABELS)}, and place_reason one of {', '.join(PLACE_REASONS)}"
        )
    forbidden = [
        entry.text
        for entry in releases
        if dated(entry.text) or (stated(entry.text) and entry.place_reason != PLACE_HISTORICAL)
    ]
    if forbidden:
        quoted = ", ".join(json.dumps(text) for text in forbidden)
        return (
            f"these releases name a year, a month, a holiday, an age, or a state or province not "
            f"released as historical: {quoted}; the policy always removes those, so drop each release or narrow "
            "it to the words the policy lets through"
        )
    placeless = [
        entry.text
        for entry in releases
        if (entry.category == "LOCATION" or entry.relabel == RELABEL_PLACE)
        and entry.place_reason not in PLACE_REASONS
        and not only_countries(entry.text)
    ]
    if placeless:
        quoted = ", ".join(json.dumps(text) for text in placeless)
        return (
            f"these releases let a place smaller than a country through without a place_reason: {quoted}; give "
            f"one of {', '.join(PLACE_REASONS)}, or keep it removed when the speaker connects it to themselves "
            "or their people"
        )
    unreasoned = [
        entry.text
        for entry in result.proposal
        if entry.action == RELEASE and entry.category in ("PERSON", "LOCATION") and not entry.why.strip()
    ]
    if unreasoned:
        quoted = ", ".join(json.dumps(text) for text in unreasoned)
        return (
            f"these releases give no reason: {quoted}; say which public figure each name is, or why the "
            "country singles nobody out given the rest of the transcript, or keep it masked"
        )
    quotes = (
        [entry.text for entry in result.proposal]
        + [entry.text for entry in result.other_speakers]
        + list(result.instructions_spoken)
        + list(result.off_task_quotes)
        + list(result.phrase_quotes)
        + list(result.other_speaker_quotes)
        + list(result.task_content_quotes)
    )
    missing = [text for text in quotes if not quote_occurs(text, original)]
    if missing:
        quoted = ", ".join(json.dumps(text) for text in missing)
        return f"these quotes do not occur in the ORIGINAL: {quoted}; quote the exact words from the ORIGINAL"
    unproposed = _named_unproposed(result.reasoning, result.proposal, original)
    if unproposed:
        quoted = ", ".join(json.dumps(text) for text in unproposed)
        return (
            f"your REASONING names {quoted} as a venue, employer, school or street but the PROPOSAL has no "
            "entry quoting it; add a redact entry, or a release entry saying why the standard lets it through"
        )
    return None


def dated(quote: str) -> bool:
    """Whether a quote writes a year, a month, a holiday or an age.

    Args:
        quote: The words a ``release`` entry quotes.

    Returns:
        True where :func:`~senselab.text.tasks.pii_detection.redaction_policy.date_positions` finds one.
    """
    return bool(date_positions(quote.split()))


def stated(quote: str) -> bool:
    """Whether a quote names a state or a province.

    Args:
        quote: The words a ``release`` entry quotes.

    Returns:
        True where :func:`~senselab.text.tasks.pii_detection.redaction_policy.state_positions` finds one.
    """
    return bool(state_positions(quote.split()))


def only_countries(quote: str) -> bool:
    """Whether every capitalised word of a quote belongs to a country's name.

    Args:
        quote: The words a ``release`` entry quotes.

    Returns:
        True where the quote holds a country and no other capitalised word.
    """
    words = quote.split()
    inside = {position for start, end in country_runs(words) for position in range(start, end)}
    capitals = [position for position, word in enumerate(words) if word.strip("\"'([{")[:1].isupper()]
    return bool(inside) and all(position in inside for position in capitals)


class ReviewWorkerError(RuntimeError):
    """A worker that did not start, did not answer, or reported an exception of its own.

    Its message is already in the shape a ``ReviewResult.failure`` carries — the worker-side
    exception type and message, or what the host observed instead — so a caller reports ``str(exc)``
    rather than prefixing its own type name and nesting one report inside another.
    """


def _failure(exc: BaseException) -> str:
    """One exception as the string a recorded absence carries.

    Args:
        exc: What went wrong.

    Returns:
        The message alone for a :class:`ReviewWorkerError`, which already names the failing type;
        otherwise the type and the message, because nothing else would say what raised.
    """
    return str(exc) if isinstance(exc, ReviewWorkerError) else f"{type(exc).__name__}: {exc}"


class _ReviewWorker:
    """One venv subprocess with the weights loaded, answering review requests until it is stopped.

    Attributes:
        model_id: The repo whose weights it holds.
        revision: The 40-hex commit those weights are, as the worker read it back.
        load_s: How long starting it and loading the weights took.
        engine: The engine's identity: as expected before :meth:`start`, as the worker reported it after.
        engine_details: Everything the worker reported about its engine.
    """

    def __init__(self, model_id: str, revision: str, engine: Mapping[str, Any] | None = None) -> None:
        """Record what the worker will be asked to load. Starting it is :meth:`start`.

        Args:
            model_id: The HuggingFace repo.
            revision: The resolved 40-hex commit.
            engine: The expected engine identity; the transformers engine where None.
        """
        from senselab.text.tasks.pii_detection.redaction_review_vllm import ENGINE_TRANSFORMERS, engine_identity

        self.model_id = model_id
        self.revision: Optional[str] = revision
        self.load_s = 0.0
        self.engine: dict[str, Any] = dict(engine or engine_identity(ENGINE_TRANSFORMERS))
        self.engine_details: dict[str, Any] = dict(self.engine)
        self._process: Optional[subprocess.Popen[str]] = None
        self._replies: queue.Queue[dict[str, Any]] = queue.Queue()
        self._noise: deque[str] = deque(maxlen=40)

    def start(self, timeout_s: int) -> None:
        """Build the venv if needed, stage the commit, spawn the worker and wait for its weights.

        Args:
            timeout_s: Wall-clock ceiling on the load.

        Raises:
            ReviewWorkerError: If the worker died, raised, or did not report ready in time.
        """
        from senselab.text.tasks.pii_detection.redaction_review_vllm import ENGINE_VLLM, start_vllm_worker

        if self.engine.get("name") == ENGINE_VLLM:
            start_vllm_worker(self, timeout_s)
            return
        venv_dir = ensure_venv(REVIEW_VENV, REVIEW_REQUIREMENTS, python_version=REVIEW_PYTHON)
        model_path = _staged_snapshot(self.model_id, str(self.revision))
        env = hf_subprocess_env(self.model_id, str(self.revision), base_env=_clean_subprocess_env())
        self._process = subprocess.Popen(  # noqa: S603 — the interpreter is this repo's own venv
            [venv_python(venv_dir), "-c", _REVIEW_WORKER_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        threading.Thread(target=self._pump_replies, daemon=True).start()
        threading.Thread(target=self._pump_noise, daemon=True).start()
        began = time.monotonic()
        self._send({"model_id": self.model_id, "model_path": model_path, "revision": self.revision})
        reply = self._await(timeout_s)
        self.revision = reply.get("revision") or self.revision
        self.load_s = round(time.monotonic() - began, 3)

    @property
    def alive(self) -> bool:
        """Whether the worker process is still running and still holding its weights."""
        return self._process is not None and self._process.poll() is None

    def review(self, text: str, prompt: str, max_new_tokens: int, timeout_s: int) -> dict[str, Any]:
        """Ask the loaded model for one review.

        Args:
            text: The redacted transcript.
            prompt: The instructions (:func:`review_prompt`).
            max_new_tokens: Generation ceiling.
            timeout_s: Wall-clock ceiling on this generation alone.

        Returns:
            The worker's reply: ``completion``, ``generate_s`` and ``output_tokens``.

        Raises:
            ReviewWorkerError: If the worker died, raised, or did not answer in time.
        """
        self._send({"text": text, "prompt": prompt, "max_new_tokens": int(max_new_tokens)})
        return self._await(timeout_s)

    def close(self) -> None:
        """End the worker, releasing the weights. Safe to call on a worker that never started."""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.write(json.dumps({"stop": True}) + "\n")
                process.stdin.flush()
                process.stdin.close()
            process.wait(timeout=5)
        except Exception:  # noqa: BLE001 — a worker that will not stop politely is killed
            process.kill()
            try:
                process.wait(timeout=30)
            except Exception:  # noqa: BLE001, S110 — nothing further is owed to an unreapable child
                pass

    def _send(self, payload: dict[str, Any]) -> None:
        """Write one request line, turning a closed pipe into the failure the caller reports.

        Args:
            payload: The request.

        Raises:
            ReviewWorkerError: If the worker is gone or its stdin will not take the line.
        """
        if self._process is None or self._process.stdin is None:
            raise ReviewWorkerError("redaction review worker is not running")
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ReviewWorkerError(f"redaction review worker closed its input: {exc}\n{self._tail()}") from exc

    def _await(self, timeout_s: int) -> dict[str, Any]:
        """Wait for one reply, killing the worker on anything that is not one.

        Args:
            timeout_s: How long to wait.

        Returns:
            The reply.

        Raises:
            ReviewWorkerError: On a timeout, a dead worker, or a worker-reported exception. Only a reply the
                worker marks ``recoverable`` (a request it could not take) leaves it running.
        """
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise ReviewWorkerError(f"redaction review worker did not answer in {timeout_s}s\n{self._tail()}") from None
        if "error" in reply:
            if not reply.get("recoverable"):
                self.close()
            error = reply["error"] or {}
            raise ReviewWorkerError(f"{error.get('type', 'RuntimeError')}: {error.get('message', 'unknown error')}")
        if reply.get("eof"):
            self.close()
            raise ReviewWorkerError(f"redaction review worker exited without answering\n{self._tail()}")
        return reply

    def _pump_replies(self) -> None:
        """Drain stdout into the reply queue, forwarding anything unmarked to the noise tail."""
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            if line.startswith(_WORKER_MARKER):
                try:
                    self._replies.put(json.loads(line[len(_WORKER_MARKER) :]))
                except ValueError:
                    self._noise.append(line.rstrip())
            elif line.strip():
                self._noise.append(line.rstrip())
        self._replies.put({"eof": True})

    def _pump_noise(self) -> None:
        """Drain stderr so a chatty loader cannot fill its pipe and deadlock the worker."""
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in process.stderr:
            if line.strip():
                self._noise.append(line.rstrip())

    def _tail(self) -> str:
        """The worker's last lines of output, for a failure message that says what it was doing."""
        return "\n".join(self._noise)


_WORKER_LOCK = threading.Lock()
_WORKER: Optional[_ReviewWorker] = None
_WORKER_KEY: Optional[tuple[str, str, str]] = None
_WORKER_REFUSED: Optional[str] = None


def shutdown_review_worker(*, forget_failure: bool = True) -> None:
    """End the process's review worker, releasing its weights.

    A later :func:`review_transcript` starts a new one. Registered to run at interpreter exit, so
    a caller only needs this to hand the memory back earlier than that, or to retry a load that
    failed.

    Args:
        forget_failure: Whether to also clear a recorded start-up refusal, so the next review
            attempts a load again. ``True`` is the deliberate retry. A caller releasing the weights
            between recordings passes ``False``: the reason a load failed on one recording is a
            property of the host, not of the transcript, and re-attempting it on every recording is
            the per-recording stall the refusal record exists to prevent.
    """
    global _WORKER, _WORKER_KEY, _WORKER_REFUSED
    with _WORKER_LOCK:
        worker, _WORKER, _WORKER_KEY = _WORKER, None, None
        if forget_failure:
            _WORKER_REFUSED = None
    if worker is not None:
        worker.close()


atexit.register(shutdown_review_worker)


def _worker_for(
    model_id: str, revision: str, timeout_s: int, engine: Mapping[str, Any] | None = None
) -> tuple[_ReviewWorker, float]:
    """The running worker holding this commit on this engine, started if there is not a live one already.

    Args:
        model_id: The HuggingFace repo.
        revision: The resolved 40-hex commit.
        timeout_s: Wall-clock ceiling on a load, when one is needed.
        engine: The expected engine identity; the transformers engine where None.

    Returns:
        ``(worker, load_s)`` — the worker with its weights loaded, and the seconds this call spent
        loading them, which is ``0.0`` when an already-running worker was reused.

    Raises:
        ReviewWorkerError: If a worker could not be started, now or earlier in this process.
    """
    global _WORKER, _WORKER_KEY, _WORKER_REFUSED
    if _WORKER_REFUSED is not None:
        raise ReviewWorkerError(_WORKER_REFUSED)
    key = (model_id, revision, json.dumps(dict(engine or {}), sort_keys=True))
    if _WORKER is not None and _WORKER_KEY == key and _WORKER.alive:
        return _WORKER, 0.0
    if _WORKER is not None:
        _WORKER.close()
        _WORKER, _WORKER_KEY = None, None
    worker = _ReviewWorker(model_id, revision, engine)
    try:
        worker.start(timeout_s)
    except Exception as exc:
        worker.close()
        _WORKER_REFUSED = _failure(exc)
        raise
    _WORKER, _WORKER_KEY = worker, key
    return worker, worker.load_s


def review_transcript(
    original: str,
    *,
    redacted: str | None = None,
    context: Mapping[str, Any] | None = None,
    model_id: str = DEFAULT_MODEL,
    ref: str = DEFAULT_REF,
    max_new_tokens: int = 1024,
    timeout_s: int = 1800,
    feedback: str | None = None,
    engine: Mapping[str, Any] | None = None,
) -> ReviewResult:
    """Ask the reviewer to read one recording's transcript, reporting failure rather than raising.

    The ref is resolved to a commit SHA before the worker starts and only the SHA reaches it, so a
    load can never go back through a pointer that may have moved. The worker outlives the call: the
    first review in a process pays the load and every later one does not, and ``load_s`` on the
    result says which this was.

    Calls are serialised — one worker holds one copy of the weights — so a concurrent caller waits.

    Args:
        original: The transcript as the recording's words were read.
        redacted: The text an applied redaction produced, or None where none was applied.
        context: What the recording declares about itself; see :func:`_compose`.
        model_id: The HuggingFace repo. Defaults to the QAT w4a16 Gemma-4 31B checkpoint, which is
            the variant that fits one ordinary GPU; see
            ``specs/20260817-triage-workflow-dag/config-derivations.md``.
        ref: The ref to resolve. Never passed to a load.
        max_new_tokens: Generation ceiling. The reasoning is the product, so this is not small.
        timeout_s: Wall-clock ceiling, applied to the load and to the generation separately.
        feedback: The problem with the previous round's answer (:func:`answer_problem`), sent with the
            texts so this round corrects it; None on a first round.
        engine: The engine to serve it
            (:func:`~senselab.text.tasks.pii_detection.redaction_review_vllm.engine_identity`); the transformers
            worker where None.

    Returns:
        The review. ``available`` is ``True`` only when the worker answered; every other path —
        venv build failure, missing weights, out of memory, timeout, a worker that raised — returns
        ``available=False`` with a populated ``failure`` and no proposal. A worker that could not be
        started is recorded once and reported without a retry for the rest of the process, so a host
        with no reachable GPU costs one load attempt rather than one per recording.
    """
    began = time.monotonic()
    try:
        from senselab.utils.model_revision import resolve_revision

        revision = resolve_revision(model_id, ref)
    except Exception as exc:  # noqa: BLE001 — an unresolvable ref is a recorded absence, not a crash
        return ReviewResult(
            available=False,
            failure=f"revision: {type(exc).__name__}: {exc}",
            model_id=model_id,
            elapsed_s=round(time.monotonic() - began, 3),
        )

    with _WORKER_LOCK:
        try:
            worker, load_s = _worker_for(model_id, revision, timeout_s, engine)
            output = worker.review(
                _compose(original, redacted, context, feedback),
                review_prompt(bool((context or {}).get(ITEM_SET))),
                max_new_tokens,
                timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 — every failure mode becomes a recorded absence
            return ReviewResult(
                available=False,
                failure=_failure(exc),
                model_id=model_id,
                revision=revision,
                elapsed_s=round(time.monotonic() - began, 3),
            )
        loaded_revision = worker.revision
        answered_by = dict(worker.engine)

    completion = str(output.get("completion") or "")
    parsed = parse_completion(completion)
    return ReviewResult(
        available=True,
        reasoning=parsed.reasoning,
        redaction=parsed.redaction,
        original=parsed.original,
        speakers=parsed.speakers,
        proposal=parsed.proposal,
        conditions=parsed.conditions,
        other_speakers=parsed.other_speakers,
        conditions_answered=parsed.conditions_answered,
        instructions_spoken=parsed.instructions_spoken,
        instructions_answered=parsed.instructions_answered,
        off_task_speech=parsed.off_task_speech,
        off_task_quotes=parsed.off_task_quotes,
        phrases_instead_of_items=parsed.phrases_instead_of_items,
        phrase_quotes=parsed.phrase_quotes,
        other_speaker=parsed.other_speaker,
        other_speaker_quotes=parsed.other_speaker_quotes,
        task_content_quotes=parsed.task_content_quotes,
        model_id=model_id,
        revision=loaded_revision or revision,
        raw="" if parsed.reasoning else completion,
        elapsed_s=round(time.monotonic() - began, 3),
        load_s=load_s,
        input_tokens=output.get("input_tokens"),
        output_tokens=output.get("output_tokens"),
        peak_reserved_mib=int(output.get("peak_reserved_mib") or 0),
        resident_mib=int(output.get("resident_mib") or 0),
        engine=answered_by,
    )


def review_payload(result: ReviewResult) -> dict[str, Any]:
    """One review as the mapping a provenance store records.

    Args:
        result: The review.

    Returns:
        The mapping. ``reasoning`` is the chain of thought verbatim, which is what the step is for.
        ``elapsed_s`` and ``load_s`` are what the round cost and how much of that was the weights,
        and the two ``_mib`` fields are what it held on the device at its peak and between reviews,
        so a store answers the time and the memory without a stopwatch outside the graph.
        ``off_task_speech`` (one of :data:`OFF_TASK_STATES` or None), ``off_task_quotes``,
        ``phrases_instead_of_items`` (one of :data:`PHRASE_STATES` or None), ``phrase_quotes``,
        ``other_speaker`` (one of :data:`OTHER_SPEAKER_ROLES` or None), ``other_speaker_quotes`` and
        ``task_content_quotes`` (lists of quoted strings) are the task reading.
    """
    return {
        "available": result.available,
        "reasoning": result.reasoning,
        "redaction": result.redaction,
        "original": result.original,
        "speakers": result.speakers,
        "proposal": [
            {
                "text": entry.text,
                "action": entry.action,
                "category": entry.category,
                "why": entry.why,
                "safe_harbor": entry.safe_harbor,
                "relabel": entry.relabel,
                "place_reason": entry.place_reason,
            }
            for entry in result.proposal
        ],
        "conditions": [{"text": entry.text, "why": entry.why} for entry in result.conditions],
        "other_speakers": [
            {"text": entry.text, "expected": entry.expected, "why": entry.why} for entry in result.other_speakers
        ],
        "conditions_answered": result.conditions_answered,
        "instructions_spoken": list(result.instructions_spoken),
        "instructions_answered": result.instructions_answered,
        "off_task_speech": result.off_task_speech,
        "off_task_quotes": list(result.off_task_quotes),
        "phrases_instead_of_items": result.phrases_instead_of_items,
        "phrase_quotes": list(result.phrase_quotes),
        "other_speaker": result.other_speaker,
        "other_speaker_quotes": list(result.other_speaker_quotes),
        "task_content_quotes": list(result.task_content_quotes),
        "prompt_version": PROMPT_VERSION,
        "failure": result.failure,
        "model_id": result.model_id,
        "revision": result.revision,
        "elapsed_s": result.elapsed_s,
        "load_s": result.load_s,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "peak_reserved_mib": result.peak_reserved_mib,
        "resident_mib": result.resident_mib,
        "engine": dict(result.engine),
    }
