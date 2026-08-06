"""The open set of perturbations a run measures under (D-17).

A **perturbation is a transform of the recording**. ``raw`` is the identity; speech enhancement
is one more; the set is open, and a future L2 round may propose another — so L1 is re-enterable
and nothing downstream may assume how many there are or what they are called.

Two assumptions used to be spelled into the code instead of declared here, and both were wrong
in the same way:

- **exactly two.** ``PassLabel`` was ``Literal["raw_16k", "enhanced_16k"]``, the driver ran two
  blocks, and ``get_stream_wav`` branched on those two strings. A third perturbation was a code
  edit in every one of those places.
- **the name carries the transform.** ``variant = "speech_enhanced" if label.startswith("enhanced")``
  inferred what had been done to the audio from how the directory happened to be spelled, so a
  perturbation named ``enhanced_lowpass`` would have claimed to be plain enhancement and a
  perturbation named ``sepformer`` would have claimed to be unmodified.

Here the transform is *declared* beside the name, the parameters travel with it, and
``L1/perturbations.json`` records the whole set — so a reader of a finished run can tell what
each ``L1/perturbation/<k>/`` directory contains without knowing which flag produced it.

Adding a perturbation that reuses a known transform (a second enhancement model, say) is a
register entry and no code edit anywhere. Adding a genuinely *new* transform is one entry in
:data:`TRANSFORMS` plus its implementation in :func:`apply` — one edit, in the one place that
knows how to do it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.layout import evidence_dir

if TYPE_CHECKING:  # pragma: no cover — keeps this module importable without torch
    from senselab.audio.data_structures import Audio
    from senselab.utils.data_structures import DeviceType

__all__ = [
    "IDENTITY_NAME",
    "IDENTITY_TRANSFORM",
    "REGISTER_FILENAME",
    "SNR_GATED_TRANSFORMS",
    "TRANSFORMS",
    "Perturbation",
    "apply",
    "identity",
    "read_measurements",
    "read_register",
    "register_payload",
    "speech_enhancement",
    "write_register",
]

IDENTITY_TRANSFORM: Final[str] = "unmodified"
"""The transform that does nothing. Every run has exactly one perturbation carrying it."""

IDENTITY_NAME: Final[str] = "raw"
"""The identity perturbation's name. ``raw_16k`` used to be it — the sampling rate is a property
of the *pipeline*, applied to every perturbation equally, so it never belonged in the name."""

TRANSFORMS: Final[Mapping[str, str]] = {
    IDENTITY_TRANSFORM: "the recording as read, resampled and downmixed by the pipeline",
    "speech_enhanced": "single-channel speech enhancement (SepFormer family)",
    "foreground_suppressed": "foreground speech projected out, leaving the background residual",
}
"""Known transforms → what each does to the recording.

Also the vocabulary ``StageContext.variant`` validates against: a stage that is only meaningful
on unmodified audio (the background mask, most importantly) gates on the transform, and it now
reads the *declared* transform rather than guessing from the directory name.
"""

SNR_GATED_TRANSFORMS: Final[frozenset[str]] = frozenset({"speech_enhanced"})
"""Transforms whose reading only counts where the recording is actually degraded.

**A speech-enhancement model is a repair, and a repair has no standing where nothing is
broken.** Above the SNR floor there is no noise for it to remove, so any change it makes to a
downstream answer is an artifact of the transform rather than evidence about the recording.
Folding it in unconditionally was measured on a clean two-speaker conversation (41–70 dB SNR
throughout): the raw pass placed the speaker axis at exactly 0.0 in 179 of 190 buckets, the
enhanced pass at 0.398 with only 51% zeros, and averaging the two published 0.227 — the
diarizers agreed and the axis said otherwise, in every one of the 178 buckets where nothing was
in dispute.

The gate is on **SNR alone, not on ambiguity.** Admitting the perturbation wherever the raw
sources disagreed was measured too, and it reads better on that clip (0.0202 against 0.0317,
because enhancement resolves five of the seven contested buckets) — but it is the wrong rule:
at genuinely low SNR the raw sources can be unanimously *wrong*, all of them fooled by the same
noise, and that is precisely the case enhancement exists for. An ambiguity requirement locks it
out there. Ambiguity in a high-SNR bucket, meanwhile, is a real disagreement to be resolved on
the recording's own evidence, not arbitrated by a transform.

**Invariance probes are deliberately not listed** (see :mod:`invariance`). Gain scaling, whole-
sample time shift and a small DC offset are chosen so that a *correct* model's answer cannot
change, which makes them meaningful everywhere and at every SNR — gating them by degradation
would remove the only condition under which their disagreement is unambiguously a model defect.
The distinction is the point: enhancement is a transform a model may legitimately answer
differently on, and an invariance probe is one where it may not.
"""

REGISTER_FILENAME: Final[str] = "perturbations.json"


@dataclass(frozen=True)
class Perturbation:
    """One transform of the recording, with the parameters that reproduce it.

    Attributes:
        name: What the run calls it — the ``L1/perturbation/<name>/`` directory and the
            ``perturbation`` column on every ``L1/signals/`` row.
        transform: A key of :data:`TRANSFORMS`. Declared, never inferred from ``name``.
        parameters: What the transform was given (model id, gain, cutoff…). Recorded so the
            perturbation can be reproduced from ``L1/perturbations.json`` alone.
        gain_db: Level applied after the transform. Carried separately because the scene
            classifiers are amplitude-sensitive, so a result is only interpretable alongside it.
    """

    name: str
    transform: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    gain_db: float = 0.0

    def __post_init__(self) -> None:
        """Reject an undeclared transform at construction rather than at read-back."""
        if self.transform not in TRANSFORMS:
            raise ValueError(f"unknown transform {self.transform!r}; declare it in TRANSFORMS first")

    @property
    def is_identity(self) -> bool:
        """Is this the untransformed recording?"""
        return self.transform == IDENTITY_TRANSFORM

    @property
    def admission_requires_low_snr(self) -> bool:
        """Does this perturbation's reading only count where the recording is degraded?

        True for the repair transforms in :data:`SNR_GATED_TRANSFORMS`. Read by
        ``fuse.SnrGate`` to decide, per bucket, whether this perturbation's readings enter the
        fold at all — never to decide whether to *compute* the pass, which stays the run-level
        ``enhancement.mode`` decision. Computing it and then declining to fold it is not waste:
        the perturbation still contributes its cross-pass ``|delta|`` to
        ``reliability.signal_stability``, which is what sets every signal's weight.
        """
        return self.transform in SNR_GATED_TRANSFORMS

    def to_json(self) -> dict[str, Any]:
        """The register entry for this perturbation."""
        return {
            "name": self.name,
            "transform": self.transform,
            "parameters": dict(self.parameters),
            "gain_db": float(self.gain_db),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> Perturbation:
        """Rebuild a perturbation from its register entry."""
        return cls(
            name=str(payload["name"]),
            transform=str(payload["transform"]),
            parameters=dict(payload.get("parameters") or {}),
            gain_db=float(payload.get("gain_db") or 0.0),
        )


def identity() -> Perturbation:
    """The recording itself — the one perturbation every run has."""
    return Perturbation(name=IDENTITY_NAME, transform=IDENTITY_TRANSFORM)


def speech_enhancement(model_id: str, *, name: str = "enhanced") -> Perturbation:
    """Speech enhancement under ``model_id``.

    ``name`` is a parameter because a run may legitimately carry two of these under different
    models; the *transform* is what they share.
    """
    return Perturbation(name=name, transform="speech_enhanced", parameters={"model_id": model_id})


def register_payload(
    perturbations: Sequence[Perturbation],
    *,
    source_audio: str | None = None,
    measured: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ``L1/perturbations.json`` document for a set of perturbations.

    ``measured`` carries what running each one produced — duration, audio signature, status —
    beside the declaration that produced it, so a later stage that needs a perturbation's
    signature does not have to open the perturbation's directory to find it.
    """
    entries = []
    for perturbation in perturbations:
        entry = perturbation.to_json()
        entry["measured"] = dict((measured or {}).get(perturbation.name) or {})
        entries.append(entry)
    return {"source_audio": source_audio, "perturbations": entries}


def write_register(
    run_dir: Path | str,
    perturbations: Sequence[Perturbation],
    *,
    source_audio: str | None = None,
    measured: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Write ``<run>/L1/perturbations.json`` — the index of what L1 measured under.

    Written by L1, once, and never rewritten. ``L1/passes.json`` — its predecessor — was
    rewritten by ``_write_run_summary`` *after* the adaptive loop, which made the file defining
    L1's inputs a back-edge from the deliverable.
    """
    dest = evidence_dir(run_dir) / REGISTER_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = register_payload(perturbations, source_audio=source_audio, measured=measured)
    dest.write_text(json.dumps(payload, indent=2) + "\n")
    return dest


def read_register(run_dir: Path | str) -> tuple[Perturbation, ...]:
    """The perturbations a completed run measured under, in the order L1 ran them.

    Returns an empty tuple when the register is absent or unreadable — a consumer that needs a
    specific perturbation reports *that* rather than a parse error, and a run predating the
    register simply has none to offer.
    """
    path = evidence_dir(run_dir) / REGISTER_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return ()
    return tuple(_parsed_entries(payload))


def read_measurements(run_dir: Path | str) -> dict[str, dict[str, Any]]:
    """``{perturbation → what running it produced}`` — duration, audio signature, status.

    The small index every later stage actually needs, read from the register rather than by
    reaching into ``L1/<perturbation>/``: D-17 makes ``L1/signals/`` L2's only input from L1, and
    a stage that needs a duration should not have to break that to get one.
    """
    path = evidence_dir(run_dir) / REGISTER_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    entries = payload.get("perturbations") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {}
    return {
        str(entry["name"]): dict(entry.get("measured") or {})
        for entry in entries
        if isinstance(entry, dict) and "name" in entry
    }


def _parsed_entries(payload: Any) -> list[Perturbation]:  # noqa: ANN401 — arbitrary JSON
    entries = payload.get("perturbations") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return []
    out: list[Perturbation] = []
    for entry in entries:
        if isinstance(entry, dict):
            try:
                out.append(Perturbation.from_json(entry))
            except (KeyError, ValueError):
                continue
    return out


def apply(
    perturbation: Perturbation,
    audio: Audio,
    *,
    device: DeviceType | None = None,
) -> Audio:
    """Apply one perturbation to the prepared audio.

    Dispatches on the *declared* transform. An unknown one cannot reach here — ``Perturbation``
    rejects it at construction — so the exhaustive branch below is the whole implementation
    surface a new transform has to extend.

    Args:
        perturbation: What to apply.
        audio: The prepared (mono, resampled) recording.
        device: Compute device, or ``None`` for automatic selection.

    Returns:
        The transformed audio. The identity returns its argument unchanged, by reference: a copy
        would give the two a different ``audio_signature`` and silently split the cache.

    Raises:
        NotImplementedError: For a declared transform with no implementation here yet.
    """
    if perturbation.is_identity:
        return audio
    if perturbation.transform == "speech_enhanced":
        from senselab.audio.tasks.speech_enhancement import enhance_audios
        from senselab.utils.data_structures import SpeechBrainModel

        model_id = str(perturbation.parameters.get("model_id") or "speechbrain/sepformer-wham16k-enhancement")
        return enhance_audios([audio], model=SpeechBrainModel(path_or_uri=model_id), device=device)[0]
    raise NotImplementedError(
        f"transform {perturbation.transform!r} is declared in TRANSFORMS but has no implementation in apply()"
    )
