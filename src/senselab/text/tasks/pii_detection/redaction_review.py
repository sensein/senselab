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

The venv holds **one long-lived worker per process**, started on the first review and reused by every
later one, so the weights are loaded once rather than once per review. :func:`shutdown_review_worker`
ends it; it also ends at interpreter exit and when its stdin reaches EOF, so a killed parent does not
leave weights resident. See ``specs/20260817-triage-workflow-dag/llm-check-amortised-load.md``.
"""

from __future__ import annotations

import atexit
import json
import logging
import queue
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import yaml

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

PROMPT_VERSION = 4
"""The prompt and its parse, as one number: it changes whenever either changes what a reading holds."""

REDACTION_STATES = ("complete", "incomplete", "not_applicable")
"""Whether the redaction, where one was applied, removed the identifying content."""

ORIGINAL_STATES = ("clean", "carries_pii")
"""Whether the recording's own words carry identifying content, redaction aside."""

SPEAKER_STATES = ("one", "more_than_one", "unclear")
"""How many people the transcript's own words show speaking in the recording."""

REDACT = "redact"
RELEASE = "release"
PROPOSAL_ACTIONS = (REDACT, RELEASE)
"""What a proposal entry asks for: remove this text, or stop removing text already removed."""

SAFE_HARBOR_PATH = Path(__file__).parent / "data" / "safe_harbor.yaml"
"""The HIPAA Safe Harbor identifiers the reviewer applies, with their citation."""


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


def _safe_harbor_rule() -> str:
    """The standard as the prompt states it, rendered from the packaged data."""
    standard = safe_harbor()
    lines = [
        f"Apply the HIPAA Safe Harbor standard ({standard['citation']}). These must stay removed:",
    ]
    for identifier in standard["identifiers"]:
        release = f" May be released: {identifier['release']}" if identifier.get("release") else ""
        lines.append(f"  ({identifier['code']}) {identifier['text']}{release}")
    lines.append(str(standard["residual"]))
    lines.append("For example:")
    lines.extend(f"  - {example}" for example in standard.get("examples") or ())
    return "\n".join(lines)


_PROMPT = (
    "You are auditing one recording's transcript before it is released. You are given the "
    "ORIGINAL words as transcribed, and where an automatic redaction has already been applied, "
    "the RELEASED text it produced, in which every [CATEGORY] token marks removed text. Where no "
    "redaction was applied the RELEASED section says so, and nothing has been removed.\n\n"
    + _safe_harbor_rule()
    + "\n\nJudge five things independently.\n"
    "1. Whether the redaction, where one was applied, actually removed what identifies the "
    "speaker under that standard.\n"
    "2. Whether the ORIGINAL words carry anything identifying at all, which is a separate "
    "question and the one no automatic detector here has asked.\n"
    "3. Whether the redaction removed more than the standard requires. A time expression with no "
    "calendar anchor -- a duration, a bare time-of-day noun, a relative reference -- usually "
    "identifies nobody.\n"
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
    "5. Separately from the standard, every specific medical diagnosis the speaker attributes to "
    "themselves: a named disease, disorder or syndrome (Parkinson's disease, spasmodic dysphonia, "
    "multiple sclerosis, thyroid cancer, a synovial joint cyst, sleep apnea, asthma). Do not list "
    "symptoms or sensations (a change in my voice, coughing, tiredness, pain, a sore throat), feelings "
    "or moods (anxious, stressed, sad), everyday events or procedures (surgery, a fall, an accident, an "
    "allergy shot) or a medication or treatment on its own; list a treatment only through the "
    "diagnosis it names (\"levodopa for my Parkinson's\" lists Parkinson's disease). None of these is "
    "a Safe Harbor identifier by itself, and none goes in the PROPOSAL; they are listed so that a "
    "person can judge whether, with the rest of what is released, one could single the speaker out. "
    "List a diagnosis whatever you judge about it; leave out only one that is part of the task's own "
    "stimulus.\n\n"
    "Answer in exactly seven parts, each on its own line or block, in this order.\n"
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
    "CONDITIONS: a JSON array of every diagnosis under point 5. Each element is an object with "
    'keys "text" (the exact words, quoted from the ORIGINAL) and "why" (one sentence: which diagnosis it is). '
    "Return [] when the speaker mentions none; the part is required either way.\n"
    "PROPOSAL: a JSON array giving the redaction you would apply instead. Each element is an "
    'object with keys "text" (the exact substring, quoted from the ORIGINAL), "action" (redact to '
    "remove it, release to stop removing text the current redaction removes unnecessarily), "
    '"category" (one of PERSON, LOCATION, DATE_TIME, ORGANIZATION, ID, CONTACT, OTHER), '
    '"safe_harbor" (the letter of the Safe Harbor identifier it falls under, or "" for none) and '
    '"why" (one sentence; for a release, why the standard lets it through here, weighed against the '
    "rest of the transcript). Return [] to leave the current redaction exactly as it is.\n"
    "Whenever you judge the redaction incomplete, or judge the ORIGINAL clean while the RELEASED text "
    "still removes words, or judge the ORIGINAL to carry something identifying, the PROPOSAL must name "
    "every word or phrase concerned, one entry each, quoted exactly as it appears in the ORIGINAL: "
    "release for removed words that identify nobody, redact for words that must go. A judgment of that "
    "kind with an empty PROPOSAL is not an answer, and a place released without a reason is not one "
    "either.\n\n"
)


def _compose(
    original: str, redacted: str | None, context: Mapping[str, Any] | None = None, feedback: str | None = None
) -> str:
    """The task's own facts and the two transcripts, as one request body.

    Args:
        original: The transcript as the recording's words were read.
        redacted: The text an applied redaction produced, or None where none was applied.
        context: What the recording declares about itself -- ``task``, ``speech_type``,
            ``instructions``, ``asked_to_say``, ``declared_names`` and ``task_words``. Any key absent or empty is
            omitted rather than sent empty.
        feedback: What was wrong with the previous round's answer, which this round must correct; None
            on a first round.

    Returns:
        The body the prompt is prefixed to.
    """
    lines: list[str] = []
    facts = dict(context or {})
    if facts.get("task"):
        lines.append(f"TASK: {facts['task']}")
    if facts.get("speech_type"):
        lines.append(f"SPEECH TYPE: {facts['speech_type']}")
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
    if lines:
        lines.append(
            "Those lines are the task. Words that are the task's own stimulus, or that the instructions "
            "ask the participant to say, are not identifying for being said: a character, a number or a "
            "date in the stimulus identifies nobody. Anything the speaker adds of their own -- their "
            "name, where they live, who they know, their health -- is judged as it would be anywhere, "
            "and instructions read aloud by someone else are a sign of more than one speaker."
        )
        lines.append("")
    released = redacted if redacted is not None else "(no redaction was applied to this recording)"
    body = "\n".join(lines) + f"ORIGINAL:\n{original}\n\nRELEASED:\n{released}\n"
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
    """

    text: str
    action: str
    category: str
    why: str
    safe_harbor: str = ""


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
        proposal: The redaction it would apply instead, as text spans, with every health condition the
            answer's CONDITIONS part listed as a ``redact`` entry of category :data:`CONDITION`.
        other_speakers: The words it attributes to anyone other than the participant, each with whether
            the task expects that voice.
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
    """

    available: bool
    reasoning: str = ""
    redaction: str = ""
    original: str = ""
    speakers: str = ""
    proposal: list[ReviewProposal] = field(default_factory=list)
    other_speakers: list[OtherSpeaker] = field(default_factory=list)
    conditions_answered: bool = True
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
        proposal: The parsed proposal array, with every health condition the CONDITIONS part listed
            appended as a ``redact`` entry of category :data:`CONDITION`; empty where both were
            missing or malformed.
        conditions_answered: Whether the answer carried a CONDITIONS part whose array parsed, an
            empty one included.
        other_speakers: The OTHER_SPEAKERS part's entries that quote words.
    """

    reasoning: str
    redaction: str = ""
    original: str = ""
    speakers: str = ""
    proposal: list[ReviewProposal] = field(default_factory=list)
    conditions_answered: bool = False
    other_speakers: list[OtherSpeaker] = field(default_factory=list)


CONDITION = "CONDITION"
"""The category a listed health condition carries as a proposal entry."""


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


def parse_completion(completion: str) -> ParsedCompletion:
    """Split the model's answer into its reasoning, its three judgments, its conditions and its proposal.

    The two arrays are looked for after the last ``PROPOSAL:`` and ``CONDITIONS:`` headings, each up to
    the other heading where that one follows it, rather than at the first ``[`` in the completion,
    because the reasoning routinely quotes the transcript's own ``[CATEGORY]`` placeholders. The
    reasoning ends at the first judgment heading, so a one-word answer never reads as prose.

    Args:
        completion: The model's raw text.

    Returns:
        The parts. The reasoning is returned even when everything else is missing or unparsable,
        because the reasoning is what the step exists to capture. A malformed array yields no entries
        rather than raising -- a caller reads ``available`` to tell that from a reviewer that read the
        text and would change nothing, and ``conditions_answered`` to tell a missing CONDITIONS part
        from one that listed none.
    """
    proposal_at = completion.rfind(_PROPOSAL_HEADING)
    conditions_at = completion.rfind(_CONDITIONS_HEADING)
    others_at = completion.rfind(_OTHER_SPEAKERS_HEADING)

    def segment(at: int, heading: str, *others: int) -> str:
        end = min((other for other in others if other > at), default=len(completion))
        return completion[at + len(heading) : end]

    if proposal_at != -1:
        proposal_text = segment(proposal_at, _PROPOSAL_HEADING, conditions_at, others_at)
        head = completion[:proposal_at]
    else:
        start = completion.find("[")
        cut = min((at for at in (conditions_at, others_at) if at != -1), default=-1)
        proposal_text = completion if cut == -1 else completion[:cut]
        head = completion if start == -1 else completion[:start]
    conditions_parsed = (
        _array(segment(conditions_at, _CONDITIONS_HEADING, proposal_at, others_at)) if conditions_at != -1 else None
    )
    others_parsed = (
        _array(segment(others_at, _OTHER_SPEAKERS_HEADING, proposal_at, conditions_at)) if others_at != -1 else None
    )
    cuts = [
        head.find(heading)
        for heading in (
            _REDACTION_HEADING,
            _ORIGINAL_HEADING,
            _SPEAKERS_HEADING,
            _OTHER_SPEAKERS_HEADING,
            _CONDITIONS_HEADING,
        )
    ]
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
            )
        )
    listed = {_normalised(entry.text) for entry in proposal if entry.action == REDACT and entry.category == CONDITION}
    for item in conditions_parsed or ():
        text = item.get("text") if isinstance(item, dict) else item
        if not isinstance(text, str) or not text.strip() or _normalised(text) in listed:
            continue
        listed.add(_normalised(text))
        why = str(item.get("why") or "") if isinstance(item, dict) else ""
        proposal.append(ReviewProposal(text=text, action=REDACT, category=CONDITION, why=why))
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
        conditions_answered=conditions_parsed is not None,
        other_speakers=other_speakers,
    )


_PLACEHOLDER = re.compile(r"\[[A-Z][A-Z_+]*\]")
_SPACES = re.compile(r"\s+")


def _normalised(text: str) -> str:
    return _SPACES.sub(" ", text.replace("\u2019", "'")).strip().casefold()


def answer_problem(result: "ReviewResult", original: str, redacted: str | None) -> str | None:
    """What makes an answer unusable under the prompt's own rule, as feedback for another round.

    Args:
        result: One round's result.
        original: The ORIGINAL the round read.
        redacted: The RELEASED text it read, or None where no redaction was applied.

    Returns:
        None where the answer is usable. Otherwise one sentence naming the problem: a judgment that
        asks for the redaction to change (incomplete, clean original over removed words, or words
        that carry something identifying where no complete redaction covers them) with an empty
        proposal, a place released without a reason, or proposal quotes that do not occur in the
        ORIGINAL, or an answer without its CONDITIONS part.
    """
    if not result.available:
        return None
    if not result.conditions_answered:
        return (
            "your answer had no CONDITIONS part; add it as a JSON array listing every specific medical "
            "diagnosis the speaker attributes to themselves, quoted exactly from the ORIGINAL, or [] when "
            "there is none"
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
    unreasoned = [
        entry.text
        for entry in result.proposal
        if entry.action == RELEASE and entry.category == "LOCATION" and not entry.why.strip()
    ]
    if unreasoned:
        quoted = ", ".join(json.dumps(text) for text in unreasoned)
        return (
            f"these place releases give no reason: {quoted}; say why Safe Harbor lets each through here, "
            "given the rest of the transcript, or keep it masked"
        )
    haystack = _normalised(original)
    quotes = [entry.text for entry in result.proposal] + [entry.text for entry in result.other_speakers]
    missing = [text for text in quotes if _normalised(text) not in haystack]
    if missing:
        quoted = ", ".join(json.dumps(text) for text in missing)
        return f"these quotes do not occur in the ORIGINAL: {quoted}; quote the exact words from the ORIGINAL"
    return None


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
    """

    def __init__(self, model_id: str, revision: str) -> None:
        """Record what the worker will be asked to load. Starting it is :meth:`start`.

        Args:
            model_id: The HuggingFace repo.
            revision: The resolved 40-hex commit.
        """
        self.model_id = model_id
        self.revision: Optional[str] = revision
        self.load_s = 0.0
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

    def review(self, text: str, max_new_tokens: int, timeout_s: int) -> dict[str, Any]:
        """Ask the loaded model for one review.

        Args:
            text: The redacted transcript.
            max_new_tokens: Generation ceiling.
            timeout_s: Wall-clock ceiling on this generation alone.

        Returns:
            The worker's reply: ``completion``, ``generate_s`` and ``output_tokens``.

        Raises:
            ReviewWorkerError: If the worker died, raised, or did not answer in time.
        """
        self._send({"text": text, "prompt": _PROMPT, "max_new_tokens": int(max_new_tokens)})
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
            ReviewWorkerError: On a timeout, a dead worker, or a worker-reported exception.
        """
        try:
            reply = self._replies.get(timeout=timeout_s)
        except queue.Empty:
            self.close()
            raise ReviewWorkerError(f"redaction review worker did not answer in {timeout_s}s\n{self._tail()}") from None
        if "error" in reply:
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
_WORKER_KEY: Optional[tuple[str, str]] = None
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


def _worker_for(model_id: str, revision: str, timeout_s: int) -> tuple[_ReviewWorker, float]:
    """The running worker holding this commit, started if there is not a live one already.

    Args:
        model_id: The HuggingFace repo.
        revision: The resolved 40-hex commit.
        timeout_s: Wall-clock ceiling on a load, when one is needed.

    Returns:
        ``(worker, load_s)`` — the worker with its weights loaded, and the seconds this call spent
        loading them, which is ``0.0`` when an already-running worker was reused.

    Raises:
        ReviewWorkerError: If a worker could not be started, now or earlier in this process.
    """
    global _WORKER, _WORKER_KEY, _WORKER_REFUSED
    if _WORKER_REFUSED is not None:
        raise ReviewWorkerError(_WORKER_REFUSED)
    if _WORKER is not None and _WORKER_KEY == (model_id, revision) and _WORKER.alive:
        return _WORKER, 0.0
    if _WORKER is not None:
        _WORKER.close()
        _WORKER, _WORKER_KEY = None, None
    worker = _ReviewWorker(model_id, revision)
    try:
        worker.start(timeout_s)
    except Exception as exc:
        worker.close()
        _WORKER_REFUSED = _failure(exc)
        raise
    _WORKER, _WORKER_KEY = worker, (model_id, revision)
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
            worker, load_s = _worker_for(model_id, revision, timeout_s)
            output = worker.review(_compose(original, redacted, context, feedback), max_new_tokens, timeout_s)
        except Exception as exc:  # noqa: BLE001 — every failure mode becomes a recorded absence
            return ReviewResult(
                available=False,
                failure=_failure(exc),
                model_id=model_id,
                revision=revision,
                elapsed_s=round(time.monotonic() - began, 3),
            )
        loaded_revision = worker.revision

    completion = str(output.get("completion") or "")
    parsed = parse_completion(completion)
    return ReviewResult(
        available=True,
        reasoning=parsed.reasoning,
        redaction=parsed.redaction,
        original=parsed.original,
        speakers=parsed.speakers,
        proposal=parsed.proposal,
        other_speakers=parsed.other_speakers,
        conditions_answered=parsed.conditions_answered,
        model_id=model_id,
        revision=loaded_revision or revision,
        raw="" if parsed.reasoning else completion,
        elapsed_s=round(time.monotonic() - began, 3),
        load_s=load_s,
        input_tokens=output.get("input_tokens"),
        output_tokens=output.get("output_tokens"),
        peak_reserved_mib=int(output.get("peak_reserved_mib") or 0),
        resident_mib=int(output.get("resident_mib") or 0),
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
            }
            for entry in result.proposal
        ],
        "other_speakers": [
            {"text": entry.text, "expected": entry.expected, "why": entry.why} for entry in result.other_speakers
        ],
        "conditions_answered": result.conditions_answered,
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
    }
