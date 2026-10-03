"""Decision models: small local models that answer typed questions about a text with probabilities.

Triage uses Cloudflare's Clef, served by Ollama, as a second opinion beside its instruction-tuned
reviewer. :mod:`.ollama` runs a pinned model; :mod:`.second_opinion` holds the
questions triage asks and reads the answers back.
"""

from senselab.text.tasks.decision_model.ollama import (
    OllamaPin,
    OllamaServer,
    PinMismatchError,
    ServerUnusableError,
    ask_decisions,
    verify_pin,
)
from senselab.text.tasks.decision_model.second_opinion import (
    QUESTION_SET_VERSION,
    QUESTIONS,
    SecondOpinion,
    ask_second_opinion,
    read_answers,
)

__all__ = [
    "QUESTIONS",
    "QUESTION_SET_VERSION",
    "OllamaPin",
    "OllamaServer",
    "PinMismatchError",
    "ServerUnusableError",
    "SecondOpinion",
    "ask_decisions",
    "ask_second_opinion",
    "read_answers",
    "verify_pin",
]
