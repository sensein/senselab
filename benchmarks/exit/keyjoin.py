"""Joins for diagnostics that check their own matching.

A diagnostic that joins two sources by a key reports plausible nonsense when the two spell the key differently: the
T24-10 diagnostic compared JSON keys with hex digests and called every key wrong. `join` prints one key of each side
with its shape first and raises `JoinFailure` when the sides share no key, so a mismatch of spelling stops the
diagnostic instead of producing its findings.
"""

import re
from collections.abc import Callable, Collection
from typing import TypeVar

K = TypeVar("K")
SAMPLE_CHARS = 100
HEX_DIGEST = re.compile(r"[0-9a-f]{64}")


class JoinFailure(AssertionError):
    """Two sources of a diagnostic share no key."""


def shape(key: object) -> str:
    """Return the shape of `key` in a few words: a 64-hex digest, JSON text, another string, or its type."""
    if isinstance(key, str):
        if HEX_DIGEST.fullmatch(key):
            return "64-hex digest"
        if key[:1] in {'"', "{", "["}:
            return f"JSON text, {len(key)} chars"
        return f"string, {len(key)} chars"
    return type(key).__name__


def sample(name: str, keys: Collection[object], out: Callable[[str], None] = print) -> None:
    """Print one key of `keys` with its shape, or that there is none."""
    if not keys:
        out(f"{name}: no keys")
        return
    first = sorted(keys, key=repr)[0]
    out(f"{name}: {len(keys)} keys; one is {repr(first)[:SAMPLE_CHARS]} ({shape(first)})")


def join(
    left_name: str,
    left: Collection[K],
    right_name: str,
    right: Collection[K],
    out: Callable[[str], None] = print,
) -> set[K]:
    """Return the keys both sides have, after printing one key of each; raise when there is none.

    Raises:
        JoinFailure: the sides share no key (naming each side's shape when they differ).
    """
    sample(left_name, left, out)
    sample(right_name, right, out)
    matched = set(left) & set(right)
    out(f"join of {left_name} with {right_name}: {len(matched)} keys match")
    if not matched:
        shapes = {shape(k) for k in left}, {shape(k) for k in right}
        hint = (
            "" if shapes[0] == shapes[1] else f" (key shapes differ: {sorted(shapes[0])} against {sorted(shapes[1])})"
        )
        raise JoinFailure(
            f"{left_name} and {right_name} share no key{hint}: the diagnostic is not matching what it reads"
        )
    return matched
