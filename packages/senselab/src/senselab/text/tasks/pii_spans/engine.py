"""The prompt and parsing engine of `pii_spans`, shared by its backends; imports only numpy and `senselab_worker`.

`build_prompt` turns the profile slice, the optional task family and a text into one prompt; `parse_completion`
turns a model's JSON answer into spans located in the text; `spans_wire_form` writes them as `text-spans@1.0`.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy
from senselab_worker.faults import ContentFault

MAX_NEW_TOKENS = 2048
"""Generation bound of every backend: the longest answer read; a longer one is cut and fails to parse."""

TEXT_OPEN = "<text>\n"
TEXT_CLOSE = "\n</text>"

_ANSWER_FORMAT = (
    'Answer with one JSON object and nothing else: {"spans": [{"quote": <exact text>, "category": <one of the '
    'categories>, "score": <your confidence, a number from 0 to 1>}]}. An empty list when nothing identifies a person.'
)


def build_prompt(profile: Mapping[str, Any], text: str, family: str | None) -> str:
    """Return the prompt: instruction, categories, the guidance entries matching `family`, answer format, text."""
    categories = "\n".join(["Categories:"] + [f"- {c['name']}: {c['description']}" for c in profile["categories"]])
    guidance = [f["guidance"] for f in profile["families"] if family is not None and _matches(family, f["names"])]
    parts = [str(profile["instruction"]), categories, *guidance, _ANSWER_FORMAT, f"{TEXT_OPEN}{text}{TEXT_CLOSE}"]
    return "\n\n".join(parts)


def _matches(family: str, names: Sequence[str]) -> bool:
    return any(family == n or family.startswith(f"{n}-") for n in names)


def text_of(prompt: str) -> str:
    """Return the text a prompt of `build_prompt` carries."""
    return prompt.split(TEXT_OPEN, 1)[1].rsplit(TEXT_CLOSE, 1)[0]


def parse_completion(completion: str, text: str, categories: Sequence[str]) -> list[tuple[int, int, str, float]]:
    """Return `(start, end, category, score)` for every occurrence in `text` of every span the answer quotes.

    Spans are in code points, ordered by start; overlapping spans of one
    category merge to their union at the higher score.

    Raises:
        ContentFault: if the answer holds no JSON object of the format, a span's category is not in `categories`,
            its score is not a number, its quote does not occur in `text`, or spans of two categories overlap.
    """
    start = completion.find("{")
    if start < 0:
        raise ContentFault("the answer is not the span object asked for: it holds no JSON object")
    try:
        answer, _ = json.JSONDecoder().raw_decode(completion[start:])
        entries = [(str(s["quote"]), str(s["category"]), s["score"]) for s in answer["spans"]]
    except (ValueError, KeyError, TypeError) as exc:
        raise ContentFault(f"the answer is not the span object asked for ({type(exc).__name__}: {exc})") from exc
    found: list[tuple[int, int, str, float]] = []
    for quote, category, score in entries:
        if category not in categories:
            raise ContentFault(f"category {category!r} is not one of {sorted(categories)}")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or score != score:  # noqa: PLR0124 - NaN
            raise ContentFault(f"score {score!r} of {quote!r} is not a number")
        at = text.find(quote) if quote else -1
        if at < 0:
            raise ContentFault(f"quoted span {quote!r} does not occur in the text")
        while at >= 0:
            found.append((at, at + len(quote), category, min(1.0, max(0.0, float(score)))))
            at = text.find(quote, at + len(quote))
    return _merged(sorted(found), text)


def _merged(found: list[tuple[int, int, str, float]], text: str) -> list[tuple[int, int, str, float]]:
    kept: list[tuple[int, int, str, float]] = []
    for start, end, category, score in found:
        if not kept or start >= kept[-1][1]:
            kept.append((start, end, category, score))
            continue
        last = kept[-1]
        if category != last[2]:
            raise ContentFault(
                f"spans {text[last[0] : last[1]]!r} ({last[2]}) and {text[start:end]!r} ({category}) overlap"
            )
        kept[-1] = (last[0], max(last[1], end), category, max(last[3], score))
    return kept


def spans_wire_form(text: Mapping[str, Any], spans: Sequence[tuple[int, int, str, float]]) -> dict[str, Any]:
    """Return the `text-spans@1.0` wire form of `spans` over the item's `text@1.0` wire form."""
    return {
        "format": "text-spans@1.0",
        "text": text["identity"],
        "length": len(text["content"]),
        "start": numpy.array([s[0] for s in spans], dtype=numpy.int64),
        "end": numpy.array([s[1] for s in spans], dtype=numpy.int64),
        "label": numpy.array([s[2] for s in spans], dtype=str),
        "score": numpy.array([s[3] for s in spans], dtype=numpy.float64),
    }
