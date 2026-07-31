"""Invariance probes: perturbations under which a correct model must not change its answer.

The stability factor in :mod:`reliability` compares the raw and enhanced passes. That is a
useful sample, but enhancement is a *genuine transform* — a model is entitled to answer
differently on enhanced audio, so a change there is ambiguous between "this model is unstable"
and "the audio really did change".

These perturbations are different in kind. Each is chosen so that a correct model returns the
same answer, which makes any change in its answer a defect in the model rather than a response
to the signal:

- **Gain scaling.** Changes no signal-to-noise ratio — it moves the source and everything
  around it together. This is the same measurement that reframed background detection away
  from amplification: gain cannot rescue a buried source because it lifts the masker too.
  Speaker count, speaker speaker and transcript are all level-independent facts.
- **Whole-sample time shift.** Padding by an integer number of samples moves the timeline
  without resampling, so no sample value is altered and no interpolation error is introduced.
  A model whose speaker count depends on where its analysis windows happen to land is
  reporting an artifact of framing.
- **Small DC offset.** Speech models operate on mean-removed spectra, so a small constant
  should be invisible. One that is not is leaking a time-domain statistic into its decision.

Measuring these requires re-running inference, so the probe is opt-in rather than part of a
default run. What it buys is an unambiguous reliability signal: unlike the enhanced-pass
comparison, a failure here cannot be explained away as the audio having changed.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

__all__ = [
    "probe_diarization_invariance",
    "INVARIANT_PERTURBATIONS",
    "MIN_INVARIANCE",
    "invariance_score",
    "perturb",
]

INVARIANT_PERTURBATIONS: tuple[str, ...] = ("gain_down_6db", "shift_10ms", "dc_offset")
"""Perturbations a correct model's answer must survive unchanged.

Deliberately all *attenuating* or *additive-constant* rather than amplifying: amplification
risks clipping, and a clipped probe measures the distortion it introduced rather than the
model's invariance."""

MIN_INVARIANCE = 0.05
"""Floor, so a model that fails every probe is attenuated rather than silenced. Mirrors the
reliability and support floors — a lone dissenter may still be the only source that noticed
something."""

_GAIN_DOWN_DB = -6.0
_SHIFT_S = 0.010
_DC_OFFSET = 0.01


def perturb(waveform: np.ndarray, sampling_rate: int, name: str) -> np.ndarray:
    """Apply one output-preserving perturbation.

    Args:
        waveform: Mono samples.
        sampling_rate: Sample rate in Hz.
        name: One of :data:`INVARIANT_PERTURBATIONS`.

    Returns:
        The perturbed waveform.

    Raises:
        ValueError: If ``name`` is not a declared perturbation. Returning the input unchanged
            would score a typo as perfect invariance.
    """
    if name not in INVARIANT_PERTURBATIONS:
        raise ValueError(f"unknown perturbation {name!r}; expected one of {INVARIANT_PERTURBATIONS}")
    arr = np.asarray(waveform, dtype=np.float64).squeeze()
    if name == "gain_down_6db":
        return arr * (10.0 ** (_GAIN_DOWN_DB / 20.0))
    if name == "shift_10ms":
        # Whole samples only: a fractional shift would require resampling, and resampling
        # alters sample values, which is exactly what this probe must not do.
        pad = int(round(_SHIFT_S * float(sampling_rate)))
        return np.concatenate([np.zeros(pad, dtype=np.float64), arr])
    return arr + _DC_OFFSET


def invariance_score(
    reference: float,
    probe_answers: Mapping[str, float],
    *,
    min_invariance: float = MIN_INVARIANCE,
) -> float | None:
    """How far a signal's answer survives the output-preserving probes.

    Graded by how far the answer moved rather than whether it moved at all: one extra speaker
    and three extra speakers are not equally wrong, and a binary measure would treat them the
    same.

    Args:
        reference: The answer on unperturbed audio.
        probe_answers: ``{perturbation → answer}``.
        min_invariance: Floor on the returned score.

    Returns:
        A score in ``(0, 1]``, or ``None`` when no probe ran — never measured must not read as
        measured-and-perfect.
    """
    if not probe_answers:
        return None
    ref = float(reference)
    deviations = []
    for answer in probe_answers.values():
        delta = abs(float(answer) - ref)
        # Relative to the reference so the scale follows the quantity: one speaker's
        # difference matters more when the reference is 1 than when it is 8.
        scale = max(1.0, abs(ref))
        deviations.append(1.0 - math.exp(-delta / scale))
    mean_deviation = sum(deviations) / len(deviations)
    return max(float(min_invariance), 1.0 - mean_deviation)


def probe_diarization_invariance(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    reference_counts: Mapping[str, int],
    run_diarization: object,
    perturbations: tuple[str, ...] = INVARIANT_PERTURBATIONS,
) -> dict[str, float]:
    """Re-run diarization under each probe and score every model's invariance.

    Args:
        waveform: Mono samples of the unmodified pass.
        sampling_rate: Sample rate in Hz.
        reference_counts: ``{model → speaker count}`` on unperturbed audio.
        run_diarization: Callable ``(waveform, sampling_rate) → {model → speaker count}``.
        perturbations: Probes to apply.

    Returns:
        ``{model → invariance score}``, omitting any model that produced no probe answer at
        all — a model that failed to run was not measured, which is not the same as a model
        that answered inconsistently.
    """
    answers: dict[str, dict[str, float]] = {}
    for name in perturbations:
        try:
            perturbed = perturb(waveform, sampling_rate, name)
            counts = run_diarization(perturbed, sampling_rate)  # type: ignore[operator]
        except Exception:  # noqa: BLE001 — a failed probe yields no evidence, not a verdict
            continue
        if not isinstance(counts, Mapping):
            continue
        for model, count in counts.items():
            if isinstance(count, (int, float)):
                answers.setdefault(str(model), {})[name] = float(count)

    out: dict[str, float] = {}
    for model, per_probe in sorted(answers.items()):
        reference = reference_counts.get(model)
        if reference is None:
            continue
        score = invariance_score(float(reference), per_probe)
        if score is not None:
            out[model] = score
    return out
