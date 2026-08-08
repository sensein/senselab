"""Foreground suppression and its depth measurement (T064-T065, FR-018 / FR-018a).

The residual of speech enhancement is the background: subtracting estimated speech from the
original leaves what the enhancer decided was not speech, at no additional model cost.

**Suppression depth, not gain, is the binding constraint.** An oracle experiment settles it:
with 30 dB of suppression and the residual amplified to a healthy level, the reported result
was *identical* whether a faint background source was present or entirely absent — the
leaked foreground dominated either way. Amplification moves the background and the residual
foreground together and changes no signal-to-noise ratio, so no amount of it rescues shallow
suppression. That is why every finding from a suppressed variant carries a depth and a
leakage margin: a null result must be attributable to insufficient suppression rather than
to absence of background content.

Leakage is measured by **projection**, not by level. The component of the residual that is
still correlated with the estimated speech is leaked foreground; the orthogonal component is
what is genuinely not speech. A level-only measure cannot tell a quiet residual that is
mostly leakage from a quiet residual that is mostly background, and those license opposite
conclusions.

Known risk carried from the research: aggressive spectral subtraction generates *musical
noise* — spurious tonal components appearing and disappearing at random time-frequency
locations. That is a synthetic event generator feeding the classifier, so a higher residual
noise floor is preferable to deeper subtraction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ForegroundSuppression",
    "leakage_margin_db",
    "project_onto",
    "suppression_depth_db",
]


def project_onto(signal: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split ``signal`` into components parallel and orthogonal to ``reference``.

    The parallel part is what remains of the reference inside the signal — leaked foreground,
    when the reference is the estimated speech. The orthogonal part is what is genuinely
    something else.

    Returns:
        ``(parallel, orthogonal)``; ``parallel`` is zero when the reference carries no energy.
    """
    sig = np.asarray(signal, dtype=np.float64).squeeze()
    ref = np.asarray(reference, dtype=np.float64).squeeze()
    n = min(sig.size, ref.size)
    sig, ref = sig[:n], ref[:n]
    denom = float(np.dot(ref, ref))
    if denom <= 0.0:
        return np.zeros_like(sig), sig
    scale = float(np.dot(sig, ref)) / denom
    parallel = scale * ref
    return parallel, sig - parallel


def _power_db(x: np.ndarray) -> float:
    arr = np.asarray(x, dtype=np.float64)
    p = float(np.mean(arr**2)) if arr.size else 0.0
    return 10.0 * math.log10(p) if p > 0.0 else -math.inf


def suppression_depth_db(estimated_foreground: np.ndarray, residual: np.ndarray) -> float:
    """How far the foreground was suppressed, in dB (FR-018a).

    Ratio of the estimated foreground's power to the power of the part of the residual still
    correlated with it. A large value means little foreground survived into the residual.

    Returns ``inf`` when no correlated component remains, and ``-inf`` when the foreground
    carries no energy — neither is clamped, because a clamped value would read as a
    measurement.
    """
    fg_db = _power_db(estimated_foreground)
    leak, _orthogonal = project_onto(residual, estimated_foreground)
    leak_db = _power_db(leak)
    if not math.isfinite(fg_db):
        return -math.inf
    if not math.isfinite(leak_db):
        return math.inf
    return fg_db - leak_db


def leakage_margin_db(residual: np.ndarray, estimated_foreground: np.ndarray) -> float:
    """How far the genuinely-not-foreground part of the residual sits above the leakage.

    Positive means the residual is mostly background; negative means it is mostly leaked
    foreground, and any human-sound category read from it is suspect (FR-026, SC-008).

    This is the quantity a consumer needs in order to read a ``speech`` or ``people``
    category from a suppressed variant at all: without it, leaked foreground and genuine
    background human sounds are indistinguishable.
    """
    leak, orthogonal = project_onto(residual, estimated_foreground)
    leak_db, other_db = _power_db(leak), _power_db(orthogonal)
    if not math.isfinite(other_db):
        return -math.inf
    if not math.isfinite(leak_db):
        return math.inf
    return other_db - leak_db


@dataclass(frozen=True)
class ForegroundSuppression:
    """The suppressed variant and the measurements that make it interpretable."""

    residual: np.ndarray
    achieved_depth_db: float
    leakage_margin_db: float
    model: str
    requested: bool = True
    fallback: str | None = None

    def is_deep_enough_for(self, background_below_foreground_db: float) -> bool:
        """Whether suppression reaches far enough to expose a source at this depth.

        The oracle experiment showed 30 dB of suppression leaving the residual foreground
        dominant over a background 30 dB down, so the comparison is against the source's own
        depth below the foreground — not against a fixed threshold.
        """
        return self.achieved_depth_db > float(background_below_foreground_db)

    def to_json(self) -> dict[str, Any]:
        """Serialize per ``contracts/background-sources.md``."""
        return {
            "requested": self.requested,
            "model": self.model,
            "achieved_depth_db": None if not math.isfinite(self.achieved_depth_db) else self.achieved_depth_db,
            "leakage_margin_db": None if not math.isfinite(self.leakage_margin_db) else self.leakage_margin_db,
            "fallback": self.fallback,
        }


def suppress_foreground(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    model: str | None = None,
    device: Any = None,  # noqa: ANN401 — senselab DeviceType
) -> ForegroundSuppression:
    """Produce the foreground-suppressed variant as the residual of speech enhancement.

    Subtracting estimated speech from the original leaves what the enhancer judged not to be
    speech, which is the background — at no additional model cost over the enhancement the
    pipeline already runs.

    Args:
        waveform: Mono samples.
        sampling_rate: Sample rate in Hz.
        model: Enhancement model id, or ``None`` for the senselab default.
        device: Compute device.

    Returns:
        The variant with its depth and leakage measured. On failure, a
        :class:`ForegroundSuppression` whose ``fallback`` names the reason and whose residual
        is the unmodified input — the run continues on the standard variant rather than
        failing (FR-029).
    """
    try:
        import torch

        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.speech_enhancement import enhance_audios

        arr = np.asarray(waveform, dtype=np.float32).squeeze()
        audio = Audio(waveform=torch.tensor(arr).unsqueeze(0), sampling_rate=sampling_rate)
        enhanced = enhance_audios([audio], model=model) if model else enhance_audios([audio])
        speech = np.asarray(enhanced[0].waveform.squeeze().numpy(), dtype=np.float64)
        n = min(speech.size, arr.size)
        residual = arr[:n].astype(np.float64) - speech[:n]
        return ForegroundSuppression(
            residual=residual,
            achieved_depth_db=suppression_depth_db(speech[:n], residual),
            leakage_margin_db=leakage_margin_db(residual, speech[:n]),
            model=model or "speechbrain/sepformer-wham16k-enhancement",
        )
    except Exception as exc:  # noqa: BLE001 — any failure must degrade, not abort (FR-029)
        return ForegroundSuppression(
            residual=np.asarray(waveform, dtype=np.float64).squeeze(),
            achieved_depth_db=-math.inf,
            leakage_margin_db=-math.inf,
            model=model or "speechbrain/sepformer-wham16k-enhancement",
            fallback=f"{type(exc).__name__}: {exc}",
        )
