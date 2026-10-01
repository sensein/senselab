"""The questions triage asks a decision model about one transcript, and how the answers are read.

Four judgments the reviewer also makes, phrased as typed questions: whether anyone besides the
participant speaks, whether the task's instructions are spoken, whether the speaker names a specific
medical diagnosis of their own, and whether a HIPAA Safe Harbor identifier is present. Each answer is
kept as a probability; ``other_voice`` also keeps the chosen class.

:data:`QUESTION_SET_VERSION` changes whenever a question's wording, type or criteria change, because
the answers are a function of the wording.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

QUESTION_SET_VERSION = 1
"""Bump on any change to :data:`QUESTIONS` or to the state :func:`ask_second_opinion` sends."""

OTHER_VOICE = "other_voice"
INSTRUCTIONS_SPOKEN = "instructions_spoken"
NAMED_DIAGNOSIS = "named_diagnosis"
SAFE_HARBOR_IDENTIFIER = "safe_harbor_identifier_present"
MORE_THAN_ONE = "more_than_one"

QUESTIONS: Mapping[str, Mapping[str, Any]] = {
    OTHER_VOICE: {
        "type": "choice",
        "instructions": (
            "Is any of this transcript spoken by someone other than the participant (for example an "
            "examiner giving instructions, or a second person talking)? The participant addressing the "
            "examiner (for example 'Is that enough?') is still the participant."
        ),
        "criteria": {
            "one": "Only the participant speaks",
            MORE_THAN_ONE: "Someone else also speaks",
            "unclear": "Cannot tell",
        },
    },
    INSTRUCTIONS_SPOKEN: {
        "type": "noul",
        "instructions": (
            "Are the task instructions (given in state.instructions) spoken in the transcript, verbatim "
            "or paraphrased, by anyone? The stimulus (state.stimulus) is not the instructions."
        ),
    },
    NAMED_DIAGNOSIS: {
        "type": "noul",
        "instructions": (
            "Does the speaker name a specific medical diagnosis they themselves have (a named disease, "
            "disorder or syndrome; not a symptom, feeling, procedure or medication alone)?"
        ),
    },
    SAFE_HARBOR_IDENTIFIER: {
        "type": "noul",
        "instructions": (
            "Does the transcript contain an identifier under HIPAA Safe Harbor: a person's name, a place "
            "smaller than a state (city, street, venue, employer, school), a calendar date element other "
            "than a year, an age over 89, or a contact or ID number? States, countries, and words that are "
            "part of the task or its stimulus do not count."
        ),
    },
}
"""The question set, in the decision model's typed-question form."""

QUESTION_NAMES = tuple(QUESTIONS)


@dataclass(frozen=True)
class SecondOpinion:
    """One transcript's answers.

    Attributes:
        probabilities: Question name to the probability of "yes" -- for ``other_voice``, of
            ``more_than_one``.
        other_voice_choice: The class the model chose for ``other_voice``.
        raw: The answers as the model returned them.
    """

    probabilities: dict[str, float]
    other_voice_choice: str
    raw: dict[str, Any] = field(default_factory=dict)


def read_answers(answers: Mapping[str, Any]) -> SecondOpinion:
    """Read a decision model's answers to :data:`QUESTIONS`.

    Args:
        answers: The ``answers`` mapping, question name to typed answer.

    Returns:
        The probabilities and the ``other_voice`` choice.

    Raises:
        ValueError: If a question is unanswered or an answer has no probability.
    """
    probabilities: dict[str, float] = {}
    choice = ""
    for name in QUESTION_NAMES:
        answer = answers.get(name)
        if not isinstance(answer, Mapping):
            raise ValueError(f"no answer to {name!r}")
        if QUESTIONS[name]["type"] == "choice":
            classes = answer.get("probabilities")
            if not isinstance(classes, Mapping) or MORE_THAN_ONE not in classes:
                raise ValueError(f"{name!r} answered without class probabilities")
            probabilities[name] = float(classes[MORE_THAN_ONE])
            choice = str(answer.get("choice") or "")
        else:
            if answer.get("noul") is None:
                raise ValueError(f"{name!r} answered without a probability")
            probabilities[name] = float(answer["noul"])
    return SecondOpinion(probabilities=probabilities, other_voice_choice=choice, raw=dict(answers))


def opinion_state(transcript: str, context: Mapping[str, Any]) -> dict[str, str]:
    """The state the questions are asked about: the task's facts and the transcript.

    Args:
        transcript: The text the reviewer reads.
        context: The reviewer's task context (``task``, ``instructions``, ``asked_to_say``, ...).

    Returns:
        The state.
    """
    return {
        "task": str(context.get("task") or ""),
        "speech_type": str(context.get("speech_type") or ""),
        "instructions": str(context.get("instructions") or ""),
        "stimulus": str(context.get("asked_to_say") or ""),
        "transcript": transcript,
    }


def ask_second_opinion(
    ask: Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]]], Mapping[str, Any]],
    transcript: str,
    context: Mapping[str, Any],
) -> SecondOpinion:
    """Ask the question set about one transcript.

    Args:
        ask: Sends a state and questions to the model and returns its ``answers``.
        transcript: The text the reviewer reads.
        context: The reviewer's task context.

    Returns:
        The answers.
    """
    return read_answers(ask(opinion_state(transcript, context), QUESTIONS))
