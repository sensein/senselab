"""Every ``native_confidence`` a producer emits must be confidence in the direction it cast.

``support.presence_probability`` reads a vote as ``P(speech) = c if speaks else 1 − c``. A producer
that reports an *undirected* number — the raw evidence, regardless of which way it voted — is
therefore read backwards by exactly the amount it was sure about: a frame posterior of 0.02 arrives
as ``{speaks: False, native_confidence: 0.02}`` and is folded in as ``P(speech) = 0.98``. A
confident silence becomes strong evidence of speech, on the voters the axis trusts most.

Commit 02340ca2 fixed one producer. The property is checked here for *all* of them, in two parts:

* a **sweep**: as the evidence for speech rises, ``presence_probability`` must never fall. This is
  the direction property stated without reference to any one rule's threshold, and it is what an
  inversion breaks — an undirected producer is high at both ends of its own evidence range.
* a **census**: every dict literal in ``src/senselab`` that sets ``native_confidence`` to anything
  other than ``None`` must belong to a swept producer. A new vote producer that skips the sweep
  fails here rather than silently shipping the third instance of this bug.
"""

from __future__ import annotations

import ast
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

import senselab
from senselab.audio.workflows.audio_analysis import speech_presence_link as link
from senselab.audio.workflows.audio_analysis.speech_presence_link import DEFAULT_POLICY
from senselab.audio.workflows.audio_analysis.support import presence_probability

Sweep = Callable[[float], dict[str, Any] | None]


def _lp(confidence: float) -> float:
    """A per-chunk average log-probability whose ``exp`` is ``confidence``."""
    return math.log(max(1e-9, confidence))


# ── the producers, each with the evidence knob that must move P(speech) up ───────────────
#
# Every sweep takes ``t`` in [0, 1] running from "no speech" to "speech" and returns the payload the
# producer emits for that evidence. Sweeps that hold the verdict fixed and vary only the magnitude
# are the ones that catch an inversion: the verdict alone can look monotone while the confidence
# attached to it runs backwards.

PRODUCER_SWEEPS: dict[str, list[tuple[str, Sweep]]] = {
    "_link_diar": [
        ("coverage", lambda t: link._link_diar({"covered_fraction": t}, DEFAULT_POLICY)),
    ],
    "_link_asr": [
        (
            "transcript confidence",
            lambda t: link._link_asr({"word_overlap_s": 0.4, "avg_logprobs": [_lp(t)]}, DEFAULT_POLICY),
        ),
        (
            "no_speech_prob at fixed transcript confidence",
            lambda t: link._link_asr(
                {"word_overlap_s": 0.4, "avg_logprobs": [_lp(0.9)], "no_speech_probs": [1.0 - t]},
                DEFAULT_POLICY,
            ),
        ),
        # The two branches where the verdict is "no speech" while a transcript confidence exists.
        # Token confidence scores the *transcript*; it is evidence for speech and can never be this
        # voter's confidence in silence, so raising it must not lower P(speech).
        (
            "transcript confidence inside a flagged hallucination",
            lambda t: link._link_asr(
                {"word_overlap_s": 0.4, "avg_logprobs": [_lp(t)], "no_speech_probs": [0.99]},
                DEFAULT_POLICY,
            ),
        ),
        (
            "transcript confidence below the word-overlap threshold",
            lambda t: link._link_asr(
                {"word_overlap_s": 0.0, "avg_logprobs": [_lp(t)]},
                replace(DEFAULT_POLICY, word_overlap_threshold_s=0.1),
            ),
        ),
    ],
    "_link_no_speech_prob": [
        ("1 − no_speech_prob", lambda t: link._link_no_speech_prob({"no_speech_prob": 1.0 - t}, DEFAULT_POLICY)),
    ],
    "_link_label_mass": [
        ("speech label mass", lambda t: link._link_label_mass({"speech_label_mass": t}, DEFAULT_POLICY)),
    ],
    "_link_frame": [
        ("frame posterior", lambda t: link._link_frame({"frame_mean": t}, DEFAULT_POLICY)),
        (
            "frame posterior against a moved cut",
            lambda t: link._link_frame({"frame_mean": t}, replace(DEFAULT_POLICY, frame_speech_threshold=0.8)),
        ),
    ],
    # ``_link_hnr`` is gone: HNR is voicing evidence, but its 2->10 dB ramp was a code literal never
    # fitted to voiced speech, and on a clip whose median HNR is 8.12 dB it read ordinary speech as
    # only partly voiced — making it the largest contributor on the presence axis. The dB measurement
    # still travels in ``L1/signals/acoustic_hnr.parquet``; the unfitted dB->probability step does not.
    "_link_lufs": [
        ("loudness", lambda t: link._link_lufs({"lufs": -90.0 + 80.0 * t}, DEFAULT_POLICY)),
    ],
    "_link_excess": [
        ("level above the measured floor", lambda t: link._link_excess({"excess_db": -3.0 + 25.0 * t}, DEFAULT_POLICY)),
    ],
    "directed_presence_vote": [
        ("P(speech)", lambda t: link.directed_presence_vote(t)),
        ("P(speech) against a moved cut", lambda t: link.directed_presence_vote(t, threshold=0.8)),
    ],
}


# ``_silhouette_vote`` and the two sweeps over it are gone with the voter. A cluster silhouette is
# not presence evidence: it measures cluster geometry over every window including silent ones, it
# carried a near-constant 0.44 of doubt (stdev 0.0227 over 214 buckets), and stability-based weighting
# gave it full weight *for* being constant. The clustering reaches the speaker axis as a first-class
# diarizer instead (D-20). See ``speech_presence_link`` where the builder used to be.


@pytest.mark.parametrize(
    ("producer", "label", "sweep"),
    [(name, label, sweep) for name, sweeps in PRODUCER_SWEEPS.items() for label, sweep in sweeps],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_presence_probability_never_falls_as_evidence_for_speech_rises(producer: str, label: str, sweep: Sweep) -> None:
    """The direction property, stated without reference to any rule's own threshold.

    An undirected confidence is symmetric about the cut: it reports the same number for a 0.02
    posterior and a 0.98 one, differing only in ``speaks``. Read in the direction cast, that is a
    P(speech) curve shaped like a V — high at both ends — so a monotone sweep of the evidence is
    exactly the screen an inversion cannot pass.
    """
    steps = [i / 20.0 for i in range(21)]
    # An **abstention is not an inversion.** A rule whose low end is uninformative returns no vote
    # there rather than a fabricated half-confident one, so ``None`` is a legitimate answer and the
    # property is checked over the votes that exist. Two constraints keep that from weakening the
    # test: at least some point must vote, and the abstentions must be a contiguous *prefix* — a hole
    # in the middle of the sweep would mean the rule stops answering as evidence accumulates, which
    # is a defect, not an abstention.
    curve: list[tuple[float, float]] = []
    abstained: list[float] = []
    for t in steps:
        payload = sweep(t)
        if payload is None:
            abstained.append(t)
            continue
        p = presence_probability(payload)
        assert p is not None, f"{producer} ({label}) produced an unreadable vote at t={t}"
        curve.append((t, p))
    assert curve, f"{producer} ({label}) abstained across the whole sweep"
    if abstained:
        assert abstained == steps[: len(abstained)], (
            f"{producer} ({label}) abstained at {abstained} — not a low-end prefix, so it stopped "
            "answering as the evidence for speech rose"
        )

    for (t_lo, p_lo), (t_hi, p_hi) in zip(curve, curve[1:]):
        assert p_hi >= p_lo - 1e-9, (
            f"{producer} ({label}): P(speech) fell from {p_lo:.4f} to {p_hi:.4f} while the evidence "
            f"for speech rose from {t_lo:.2f} to {t_hi:.2f}. The confidence is being read against "
            "the direction it was cast in."
        )


# ── the census ───────────────────────────────────────────────────────────────────────────


def _package_sources() -> Iterator[Path]:
    root = Path(senselab.__file__).parent
    return (p for p in sorted(root.rglob("*.py")) if "__pycache__" not in p.parts)


def _enclosing_function(tree: ast.AST, line: int) -> str | None:
    """Name of the innermost function whose body spans ``line``."""
    best: str | None = None
    for candidate in ast.walk(tree):
        if not isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        end = candidate.end_lineno
        if end is None:
            continue
        if candidate.lineno <= line <= end:
            best = candidate.name  # innermost wins: ast.walk yields outer defs first
    return best


def _confidence_producers() -> dict[tuple[str, str], int]:
    """``{(file, function) → line}`` for every dict literal setting a non-``None`` confidence."""
    found: dict[tuple[str, str], int] = {}
    for path in _package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key, value in zip(node.keys, node.values):
                if not (isinstance(key, ast.Constant) and key.value == "native_confidence"):
                    continue
                if isinstance(value, ast.Constant) and value.value is None:
                    # A literal ``None`` carries no magnitude, so there is no direction to invert.
                    continue
                fn = _enclosing_function(tree, node.lineno) or "<module>"
                found[(path.name, fn)] = node.lineno
    return found


def test_every_confidence_producer_is_swept() -> None:
    """A new vote producer must state its direction under sweep, not be trusted to have one.

    This is the part that generalises: the inversion has now been found twice, in two different
    modules, and both times the code read plausibly. A census makes the *absence* of a sweep the
    failure rather than the presence of a bug.
    """
    unswept = sorted(
        f"{file}:{line} in {fn}()" for (file, fn), line in _confidence_producers().items() if fn not in PRODUCER_SWEEPS
    )
    assert not unswept, (
        "these emit a native_confidence with no directional sweep in PRODUCER_SWEEPS:\n  "
        + "\n  ".join(unswept)
        + "\nAdd a sweep proving P(speech) rises with the evidence, or emit None."
    )


def test_the_census_finds_the_producers_it_claims_to() -> None:
    """The census is only a guard if it actually reaches the code. Pin what it sees."""
    producers = _confidence_producers()
    assert ("speech_presence_link.py", "_link_frame") in producers
    assert ("speech_presence_link.py", "directed_presence_vote") in producers
    # Was ``>= 8``, lowered with ``_link_hnr``'s removal. Kept as a floor rather than an equality so
    # the census stays a guard against a *new* unaudited producer, not a count that has to be edited
    # every time one is legitimately retired.
    assert len(producers) >= 7
