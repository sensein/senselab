"""Token normalisation and phrase matching shared by the rule layers."""

from __future__ import annotations

from typing import Collection, Iterable, List, Sequence, Tuple

_EDGE = "\"'`.,;:!?()…¿¡“”‘’*"
_BRACKETS = "[]<>{}"
_TERMINAL = ("?", ".", "!", "…")


def normalize_token(text: str) -> str:
    """Lower-case a word, fold curly apostrophes and strip edge punctuation.

    A trailing hyphen (a cut-off) and surrounding brackets (``[UH]``) are kept.

    Args:
        text: The word as the recogniser wrote it.

    Returns:
        The normalised token; empty when the word was punctuation only.
    """
    token = text.strip().lower().replace("’", "'").replace("‘", "'")
    return token.strip(_EDGE).strip()


def bare_token(token: str) -> str:
    """Drop surrounding brackets and edge punctuation inside them from a normalised token.

    Args:
        token: A token from :func:`normalize_token`.

    Returns:
        ``"uh"`` for ``"[uh]"``; the token unchanged when it has no brackets.
    """
    return token.strip(_BRACKETS).strip(_EDGE)


def tokens_of(text: str) -> List[str]:
    """Split free text on whitespace and normalise each piece, dropping empty tokens.

    Args:
        text: Free text.

    Returns:
        Bare normalised tokens.
    """
    out = [bare_token(normalize_token(piece)) for piece in text.split()]
    return [t for t in out if t]


def terminal_punctuation(text: str) -> str | None:
    """The sentence-final punctuation a word or segment ends with, if any.

    Args:
        text: Raw text.

    Returns:
        ``"?"``, ``"."``, ``"!"`` or ``"…"``, else ``None``.
    """
    stripped = text.rstrip().rstrip("\"'”’)]")
    for mark in _TERMINAL:
        if stripped.endswith(mark):
            return "." if mark == "…" else mark
    return None


def count_phrases(tokens: Sequence[str], phrases: Iterable[Tuple[str, ...]]) -> int:
    """Count non-overlapping occurrences of any phrase, longest match first.

    Args:
        tokens: Token sequence.
        phrases: Phrases as token tuples.

    Returns:
        Number of matches.
    """
    by_length = sorted(set(phrases), key=len, reverse=True)
    count = 0
    i = 0
    n = len(tokens)
    while i < n:
        for phrase in by_length:
            k = len(phrase)
            if k and tuple(tokens[i : i + k]) == phrase:
                count += 1
                i += k
                break
        else:
            i += 1
    return count


def starts_with(tokens: Sequence[str], phrases: Iterable[Tuple[str, ...]]) -> bool:
    """Whether the token sequence begins with any phrase.

    Args:
        tokens: Token sequence.
        phrases: Phrases as token tuples.

    Returns:
        True on a match.
    """
    return any(phrase and tuple(tokens[: len(phrase)]) == phrase for phrase in phrases)


def contains_phrase(tokens: Sequence[str], phrases: Iterable[Tuple[str, ...]]) -> bool:
    """Whether any phrase occurs anywhere in the token sequence.

    Args:
        tokens: Token sequence.
        phrases: Phrases as token tuples.

    Returns:
        True on a match.
    """
    return count_phrases(tokens, phrases) > 0


def composed_of(tokens: Sequence[str], phrases: Collection[Tuple[str, ...]]) -> bool:
    """Whether the whole token sequence segments into phrases from the lexicon.

    Args:
        tokens: Non-empty token sequence.
        phrases: Phrases as token tuples.

    Returns:
        True when every token is covered by a lexicon phrase.
    """
    n = len(tokens)
    if n == 0:
        return False
    reachable = [False] * (n + 1)
    reachable[0] = True
    lengths = sorted({len(p) for p in phrases if p})
    for i in range(n):
        if not reachable[i]:
            continue
        for k in lengths:
            if i + k <= n and tuple(tokens[i : i + k]) in phrases:
                reachable[i + k] = True
    return reachable[n]


def strip_lead(tokens: Sequence[str], skip: Collection[str]) -> List[str]:
    """Drop leading tokens that are in ``skip``.

    Args:
        tokens: Token sequence.
        skip: Tokens to drop while they lead.

    Returns:
        The remaining tokens.
    """
    i = 0
    while i < len(tokens) and tokens[i] in skip:
        i += 1
    return list(tokens[i:])
