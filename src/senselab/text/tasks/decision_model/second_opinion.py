"""The questions triage asks a decision model about one transcript, and how the answers are read.

Five judgments, each of which can be compared with the redaction reviewer's reading: whether anyone
besides the participant speaks, whether the task's instructions are spoken, whether the transcript holds
anything the redaction policy removes, whether the masked text as it would ship is free of all of it, and
whether the participant talks off the task. The policy is the reviewer's own
(:func:`~senselab.text.tasks.pii_detection.redaction_review.redaction_policy_text`), sent in the state, and
the task's nature and content guidance come from the reviewer's ``task_guidance.yaml``.

Each answer is kept as a probability; each choice question also keeps its chosen class and its class
probabilities.

:data:`QUESTION_SET_VERSION` changes whenever a question's wording, type or criteria change, or the state
sent with them, because the answers are a function of both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from senselab.text.tasks.pii_detection.redaction_review import (
    redaction_policy_text,
    task_guidance,
    task_nature_description,
)

QUESTION_SET_VERSION = 4
"""Bump on any change to :data:`QUESTIONS` or to the state :func:`ask_second_opinion` sends."""

OTHER_VOICE = "other_voice"
INSTRUCTIONS_SPOKEN = "instructions_spoken"
POLICY_IDENTIFIER = "policy_identifier_present"
MASKED_TEXT_FREE = "masked_text_free_of_identifiers"
OFF_TASK_SPEECH = "off_task_speech"
MORE_THAN_ONE = "more_than_one"

OFF_TASK_NONE = "none"
OFF_TASK_SOME = "some"
OFF_TASK_EXTENSIVE = "extensive"


def _policy_question(subject: str) -> str:
    """A question asking whether a text holds anything the reviewer's redaction policy removes.

    Args:
        subject: The state field the question is about, and how to read it.

    Returns:
        The question's instructions, which defer to ``state.redaction_policy`` for what is removed.
    """
    return (
        f"{subject} Apply the redaction policy in state.redaction_policy, the same policy the redaction "
        "reviewer applies, as the definition of what must be removed: judge only what it removes, not its "
        "instructions about proposing releases. Anything the policy keeps or releases does not count, and "
        "neither does the task's own content (state.stimulus, state.task_content)."
    )


QUESTIONS: Mapping[str, Mapping[str, Any]] = {
    OTHER_VOICE: {
        "type": "choice",
        "instructions": (
            "Is any of this transcript spoken by someone other than the participant (for example an "
            "examiner giving or repeating instructions, or a second person talking)? The participant "
            "addressing the examiner (for example 'Is that enough?') is still the participant. Judge it "
            "against the task (state.task, state.task_nature, state.instructions, state.task_content)."
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
    POLICY_IDENTIFIER: {
        "type": "noul",
        "instructions": _policy_question(
            "Does the transcript (state.transcript) contain anything that must be removed before release?"
        ),
    },
    MASKED_TEXT_FREE: {
        "type": "noul",
        "instructions": _policy_question(
            "state.masked_text is the transcript exactly as it would be released: every removed span is "
            "replaced by a [CATEGORY] token, and where nothing was removed it equals state.transcript. Is "
            "state.masked_text free of everything that must be removed, so that nothing the policy removes "
            "can still be read in it?"
        ),
    },
    OFF_TASK_SPEECH: {
        "type": "choice",
        "instructions": (
            "Did the participant stop doing the task or talk about something other than what the "
            "instructions ask for (state.task, state.task_nature, state.instructions)? In a non-lexical task "
            "(breathing, coughing, phonation, repeated syllables) any words the participant says are off-task."
        ),
        "criteria": {
            OFF_TASK_NONE: "The participant only does the task",
            OFF_TASK_SOME: "A remark or question outside the task",
            OFF_TASK_EXTENSIVE: "Much of the recording is outside the task",
        },
    },
}
"""The question set, in the decision model's typed-question form."""

QUESTION_NAMES = tuple(QUESTIONS)

POSITIVE_CLASSES: Mapping[str, tuple[str, ...]] = {
    OTHER_VOICE: (MORE_THAN_ONE,),
    OFF_TASK_SPEECH: (OFF_TASK_SOME, OFF_TASK_EXTENSIVE),
}
"""For each choice question, the classes whose probabilities sum to its "yes"."""


@dataclass(frozen=True)
class SecondOpinion:
    """One transcript's answers.

    Attributes:
        probabilities: Question name to the probability of "yes": for a ``noul`` question its own
            probability, for a choice question the sum of its :data:`POSITIVE_CLASSES` (``other_voice``:
            ``more_than_one``; ``off_task_speech``: ``some`` plus ``extensive``).
        choices: Choice question name to the class the model chose.
        class_probabilities: Choice question name to its class probabilities as the model gave them.
        raw: The answers as the model returned them.
    """

    probabilities: dict[str, float]
    choices: dict[str, str] = field(default_factory=dict)
    class_probabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def read_answers(answers: Mapping[str, Any]) -> SecondOpinion:
    """Read a decision model's answers to :data:`QUESTIONS`.

    Args:
        answers: The ``answers`` mapping, question name to typed answer.

    Returns:
        The probabilities, the choices and the class probabilities.

    Raises:
        ValueError: If a question is unanswered, an answer has no probability, or a choice answer gives
            no probability for any of its classes.
    """
    probabilities: dict[str, float] = {}
    choices: dict[str, str] = {}
    classes_of: dict[str, dict[str, float]] = {}
    for name in QUESTION_NAMES:
        answer = answers.get(name)
        if not isinstance(answer, Mapping):
            raise ValueError(f"no answer to {name!r}")
        if QUESTIONS[name]["type"] == "choice":
            classes = answer.get("probabilities")
            criteria = QUESTIONS[name]["criteria"]
            if not isinstance(classes, Mapping) or not any(label in classes for label in criteria):
                raise ValueError(f"{name!r} answered without class probabilities")
            held = {str(label): float(p) for label, p in classes.items() if label in criteria}
            probabilities[name] = sum(held.get(label, 0.0) for label in POSITIVE_CLASSES[name])
            choices[name] = str(answer.get("choice") or "")
            classes_of[name] = held
        else:
            if answer.get("noul") is None:
                raise ValueError(f"{name!r} answered without a probability")
            probabilities[name] = float(answer["noul"])
    return SecondOpinion(
        probabilities=probabilities, choices=choices, class_probabilities=classes_of, raw=dict(answers)
    )


def opinion_state(transcript: str, context: Mapping[str, Any], masked_text: str) -> dict[str, str]:
    """The state the questions are asked about: the task's facts, the policy, and both texts.

    Args:
        transcript: The text the reviewer reads.
        context: The reviewer's task context (``task``, ``instructions``, ``asked_to_say``, ...).
        masked_text: The transcript exactly as the release would ship it under the final mask plan.

    Returns:
        The state, with the task family's nature
        (:func:`~senselab.text.tasks.pii_detection.redaction_review.task_nature_description`) as
        ``task_nature``, its task-content guidance as ``task_content``, and the reviewer's policy
        (:func:`~senselab.text.tasks.pii_detection.redaction_review.redaction_policy_text`) as
        ``redaction_policy``.
    """
    family = str(context.get("task") or "") or None
    return {
        "task": str(context.get("task") or ""),
        "task_nature": task_nature_description(family),
        "speech_type": str(context.get("speech_type") or ""),
        "instructions": str(context.get("instructions") or ""),
        "stimulus": str(context.get("asked_to_say") or ""),
        "task_content": task_guidance(family),
        "redaction_policy": redaction_policy_text(),
        "transcript": transcript,
        "masked_text": masked_text,
    }


def ask_second_opinion(
    ask: Callable[[Mapping[str, Any], Mapping[str, Mapping[str, Any]]], Mapping[str, Any]],
    transcript: str,
    context: Mapping[str, Any],
    masked_text: str,
) -> SecondOpinion:
    """Ask the question set about one transcript.

    Args:
        ask: Sends a state and questions to the model and returns its ``answers``.
        transcript: The text the reviewer reads.
        context: The reviewer's task context.
        masked_text: The transcript as the release would ship it.

    Returns:
        The answers.
    """
    return read_answers(ask(opinion_state(transcript, context, masked_text), QUESTIONS))
