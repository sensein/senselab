"""Per-pass run environment and plan for the analysis stages (T051).

Two frozen types replace the ``(args: argparse.Namespace, ctx: dict[str, Any])``
pair the stages used to take:

- :class:`StageContext` — *where and how* a pass runs (pass label, audio
  signature, device, cache/output dirs) plus the cache-key and provenance
  derivation that used to live in untyped closures on the ctx dict.
- :class:`PassPlan` — *what to run and with which knobs*. Absence means skip, so
  there is no CLI-shaped ``skip`` set in the library.

Deliberately light: no torch, no transformers, no ``argparse``. ``DeviceType`` is
imported only under ``TYPE_CHECKING`` because importing it at runtime pulls in
torch *and* transformers, and :attr:`StageContext.device_label` only reads
``.value``. That keeps "I want a cache key" from dragging in the ML stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Mapping

from senselab.utils.tasks.cached_inference import (
    CACHE_SCHEMA_VERSION,
    align_cache_key,
    cache_key,
    senselab_version,
    write_json,
)

if TYPE_CHECKING:  # pragma: no cover — avoids a runtime torch+transformers import
    from senselab.utils.data_structures import DeviceType

__all__ = ["STAGE_VERSIONS", "PassPlan", "StageContext", "stage_code_version"]


STAGE_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "diarization": 1,
        "ast": 1,
        "yamnet": 1,
        "features": 1,
        "asr": 1,
        "alignment": 1,
        "ppgs": 1,
    }
)
"""Per-stage cache-invalidation counters, keyed by the task string used in cache keys.

**Bump a stage's number when the stored shape of its outcome changes** — new
fields, changed units, different post-processing. That is the counterpart
obligation for having replaced source hashing: a `wrapper_version_hash` over the
CLI script rotated on every comment edit and reformat, invalidating every cached
model result for no reason, and it would have gotten worse once six stages shared
one module. Coarse and deliberate beats automatic and wrong.

**"diarization"'s blast radius grew with VibeVoice-ASR-HF/MOSS-Transcribe-Diarize/
DiariZen/child-adult**: cache keys are per-model_id so entries don't collide, but
this one counter is shared across every diar model — bumping it because one of
these newer backends' outcome shape changed invalidates every other diar model's
cached results too, including expensive Pyannote/Sortformer runs. Six independent
backends now share this fate where two did before. If that shared-invalidation
cost becomes a real problem in practice, the fix is splitting this into
per-backend-family keys (e.g. a distinct entry per backend rather than one
"diarization" for all of them) — not something to do preemptively without a
concrete case forcing a bump.

Library-side changes are already covered by ``senselab_version`` in the key, so
these numbers only need to move for *wrapper-shaped* output changes — mainly
``features`` (composes three backends into a row dict) and ``ppgs`` (attaches
phoneme labels). The rest are thin pass-throughs to a ``tasks/`` API.
"""


def stage_code_version(task: str) -> str:
    """Return the ``"<task>@<n>"`` code-version token for ``task``.

    Self-describing so a cache entry's provenance is readable at a glance.

    Args:
        task: A key of :data:`STAGE_VERSIONS`.

    Returns:
        e.g. ``"diarization@1"``.

    Raises:
        KeyError: If ``task`` has no declared version — a new stage must declare
            one rather than silently defaulting to 1 and sharing another stage's
            invalidation fate.

    Example:
        >>> stage_code_version("asr")
        'asr@1'
    """
    if task not in STAGE_VERSIONS:
        raise KeyError(
            f"stage {task!r} has no entry in STAGE_VERSIONS; add one (and bump it when the outcome shape changes)"
        )
    return f"{task}@{STAGE_VERSIONS[task]}"


@dataclass(frozen=True, slots=True)
class StageContext:
    """Run environment shared by every stage of one pass.

    Attributes:
        pass_label: e.g. ``"raw_16k"`` / ``"enhanced_16k"``.
        audio_signature: From ``cached_inference.audio_signature`` — the join key
            between ``summary.json`` and each cache entry's provenance.
        device: Compute device, or ``None`` for automatic selection.
        cache_dir: Cache directory, or ``None`` to disable caching.
        out_dir: Pass output directory, or ``None`` for a headless run that emits
            no sidecar files (what the adaptive loop wants).
        audio_source: Absolute source path, recorded in provenance only.
        senselab_ver: Installed senselab version; participates in cache keys.
    """

    pass_label: str
    audio_signature: str
    device: DeviceType | None = None
    cache_dir: Path | None = None
    out_dir: Path | None = None
    audio_source: str = ""
    senselab_ver: str = field(default_factory=senselab_version)

    @property
    def device_label(self) -> str:
        """Device string for cache keys and provenance.

        ``None`` maps to ``"auto"``, not ``"cpu"``: the label goes *into* the
        cache key, so collapsing "let senselab choose" into a concrete device
        would both change every key and make provenance claim something the run
        never specified.
        """
        return self.device.value if self.device is not None else "auto"

    def cache_key_for(self, task: str, model_id: str | None, params: Mapping[str, Any]) -> str:
        """Cache key for one (task, model, params) call in this pass."""
        return cache_key(
            audio_sig=self.audio_signature,
            task=task,
            model_id=model_id,
            params=dict(params),
            code_version=stage_code_version(task),
            senselab_ver=self.senselab_ver,
        )

    def align_key_for(
        self,
        *,
        transcript_sha: str,
        language: str | None,
        aligner_model_id: str,
        aligner_params: Mapping[str, Any],
    ) -> str:
        """Cache key for one alignment call, independent of the parent ASR key."""
        return align_cache_key(
            audio_sig=self.audio_signature,
            transcript_sha=transcript_sha,
            language=language,
            aligner_model_id=aligner_model_id,
            aligner_params=dict(aligner_params),
            code_version=stage_code_version("alignment"),
            senselab_ver=self.senselab_ver,
        )

    def provenance_for(self, task: str, model_id: str | None, params: Mapping[str, Any]) -> dict[str, Any]:
        """Provenance block recorded on a fresh (cache-miss) outcome.

        ``audio_signature`` here must match what ``summary.json`` reports for the
        pass — ``adaptive/interventions.py::build_cache_index`` joins on it.
        """
        return {
            "task": task,
            "model_id": model_id,
            "params": dict(params),
            "audio_signature": self.audio_signature,
            "audio_source": self.audio_source,
            "pass": self.pass_label,
            "device": self.device_label,
            "code_version": stage_code_version(task),
            "senselab_version": self.senselab_ver,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

    def write_sidecar(self, relpath: str | Path, payload: Any) -> None:  # noqa: ANN401 — senselab outputs
        """Write a JSON sidecar under :attr:`out_dir`; a no-op when it is ``None``."""
        if self.out_dir is None:
            return
        write_json(self.out_dir / relpath, payload)


@dataclass(frozen=True, slots=True)
class PassPlan:
    """Which stages to run for a pass, and with what knobs.

    Absence means skip — empty model tuples and ``None`` model ids, rather than a
    ``skip`` set. The CLI's ``--skip`` mixes pass-level and post-pass concerns and
    is mutated after parsing (the no-speech triage path), so translating it into
    an explicit plan is the script's job, not the library's.
    """

    diarization_models: tuple[str, ...] = ()
    asr_models: tuple[str, ...] = ()
    ast_model: str | None = None
    yamnet_model: str | None = None
    ast_win_length: float = 10.24
    ast_hop_length: float = 10.24
    yamnet_win_length: float = 0.96
    yamnet_hop_length: float = 0.48
    scene_top_k: int = 50
    features: bool = False
    features_win_length: float = 1.0
    features_hop_length: float = 0.5
    align_asr: bool = True
    aligner: Literal["qwen", "mms"] = "qwen"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    mms_aligner_model: str = "facebook/mms-1b-all"
    asr_language: str = "en"
    qwen_native_timestamps: bool = True
    ppg: bool = False
