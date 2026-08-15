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

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Literal, Mapping

from senselab.audio.workflows.audio_analysis.perturbations import TRANSFORMS
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

# The package logger by name rather than through ``senselab.utils.data_structures``, whose import
# is exactly the torch+transformers pull this module's header refuses.
logger = logging.getLogger("senselab")


_DEFAULT_REVISION_REF: Final[str] = "main"
"""The ref every ``_commit_sha_for`` resolution is made against.

True today for a structural reason, not a hardcoded guess: a model id reaching :class:`StageContext`
is a bare Hub id (``PassPlan.asr_models``, ``.ast_model``, ``.qwen_aligner_model``, ...) with no
per-model revision knob anywhere in this module's plumbing, so there is nowhere for a caller to have
asked for anything other than ``resolve_revision``'s default ref. If ``PassPlan`` ever grows such a
knob, this constant and ``_commit_sha_for``'s call into ``resolve_revision`` both need to start
threading that value through — silently leaving this literal behind would make
:meth:`StageContext.provenance_for`'s ``"revision"`` lie about what was actually requested.
"""


_VARIANT_NAMES: Final[tuple[str, ...]] = tuple(TRANSFORMS)
"""Recognized audio variants — exactly the declared perturbation transforms.

The variant *is* the transform the perturbation declared, so there is one list rather than two
that have to agree. ``level.py`` carries its own copy for import weight (it pulls numpy; this
module is deliberately importable without it), and ``perturbations`` is a plain dataclass module
with no such cost, so this one can be the real thing."""


STAGE_VERSIONS: Final[Mapping[str, int]] = MappingProxyType(
    {
        "diarization": 1,
        "ast": 1,
        "yamnet": 1,
        "features": 1,
        "asr": 1,
        "alignment": 1,
        "background_mask": 1,
        "noise_floor": 1,
        "background_sources": 1,
        "level_probe": 1,
    }
)
"""Per-stage cache-invalidation counters, keyed by the task string used in cache keys.

**Bump a stage's number when the stored shape of its outcome changes** — new
fields, changed units, different post-processing. That is the counterpart
obligation for having replaced source hashing: a `wrapper_version_hash` over the
CLI script rotated on every comment edit and reformat, invalidating every cached
model result for no reason, and it would have gotten worse once six stages shared
one module. Coarse and deliberate beats automatic and wrong.

Library-side changes are already covered by ``senselab_version`` in the key, so
these numbers only need to move for *wrapper-shaped* output changes — mainly
``features`` (composes three backends into a row dict) and the classifiers (attach
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
        perturbation: e.g. ``"raw"`` / ``"enhanced"``.
        audio_signature: From ``cached_inference.audio_signature`` — the join key
            between ``summary.json`` and each cache entry's provenance.
        device: Compute device, or ``None`` for automatic selection.
        cache_dir: Cache directory, or ``None`` to disable caching.
        out_dir: Pass output directory, or ``None`` for a headless run that emits
            no sidecar files (what the adaptive loop wants).
        run_dir: The run root, or ``None`` under the same headless condition. Carried
            explicitly rather than walked back out of :attr:`out_dir`, because a stage whose
            product is a *decision* rather than a measurement writes it under ``L2/`` and must
            not have to guess how many parents up the run root is.
        audio_source: Absolute source path, recorded in provenance only.
        senselab_ver: Installed senselab version; participates in cache keys.
        variant: Which audio variant this pass consumes — ``"unmodified"``,
            ``"speech_enhanced"``, or ``"foreground_suppressed"``. Recorded on every
            stage outcome so no result is unattributed (FR-012, SC-006).
        variant_gain_db: Gain applied to that variant, in dB. Recorded for the same
            reason: the classifiers are amplitude-sensitive, so a result is only
            interpretable alongside the level it was computed at.

    Note:
        ``variant`` and ``variant_gain_db`` are deliberately **not** part of the cache
        key. Cache correctness comes from :attr:`audio_signature`, which must be computed
        on the *post-gain* waveform — amplifying changes the samples, so the signature
        changes with them. Adding the variant to the key would be redundant, and computing
        the signature *pre*-gain would break that guarantee, so keep signature computation
        downstream of the gain.
    """

    perturbation: str
    audio_signature: str
    device: DeviceType | None = None
    cache_dir: Path | None = None
    out_dir: Path | None = None
    run_dir: Path | None = None
    audio_source: str = ""
    senselab_ver: str = field(default_factory=senselab_version)
    variant: str = "unmodified"
    variant_gain_db: float = 0.0

    def __post_init__(self) -> None:
        """Reject an unknown variant name at construction.

        A typo would propagate into provenance and silently break the joins that
        ``level.json`` and the disagreements index rely on, so it fails here instead.
        """
        if self.variant not in _VARIANT_NAMES:
            raise ValueError(f"unknown audio variant {self.variant!r}; expected one of {_VARIANT_NAMES}")

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
            commit_sha=self._commit_sha_for(model_id),
        )

    def _commit_sha_for(self, model_id: str | None) -> str | None:
        """Resolve ``model_id`` to this run's commit SHA, or ``None`` when there is no commit to pin.

        Resolution has to happen here, above the load, because the cache key is computed to decide
        *whether* to load at all — a SHA harvested during loading would arrive too late to key on.

        That placement is also why this is *not* the same decision as
        ``signal.resolved_commit_sha``'s, which degrades every failure to ``None``. That function
        fills in a provenance **record**, where "unknown commit" is an honest and cheap answer.
        Here ``None`` is a **key** component (``cached_inference.cache_key``'s ``commit_sha``), and
        every id that degrades to it shares one bucket — so two different upstream commits of the
        same model would collide, and the second run would be served the first one's result. Three
        outcomes, therefore, three treatments:

        - **Not a Hub id at all** — a bare ``None`` (a model-less stage, e.g. ``features``) or a
          name with no ``/`` (a local backend). Short-circuits with no Hub round-trip.
        - **A definitive not-found** (``RepositoryNotFoundError``) — the Hub has answered, and the
          answer is that no commit exists. ``None`` is then the *correct* value rather than a
          degradation, and it is stable across runs, so no two commits can collide behind it. This
          is the crash being fixed: ``default.yaml`` ships ``yamnet: google/yamnet``, a TensorFlow
          backend whose id happens to contain a ``/`` and so trips the Hub-id heuristic above.
        - **Anything else** — a 429, a network error, a ``GatedRepoError`` — propagates. Those all
          mean "we could not tell", which is unsound for a key: the load may well succeed, and its
          result would be stored under a commit-blind key that a later run cannot distinguish from
          any other commit's. ``GatedRepoError`` **subclasses** ``RepositoryNotFoundError``, so it
          has to be excluded by hand; ``dependencies._ensure_hf_model`` makes the identical split
          for the identical reason.

        Alternatives considered: ``_YAMNET_ALIASES`` already accepts a bare ``"yamnet"``
        (``classification/api.py``), so setting ``default.yaml``'s ``yamnet:`` to ``yamnet`` would
        remove this specific crash without touching cache-key semantics at all — strictly smaller.
        It was not taken because it fixes one config value rather than the class: any non-Hub
        backend whose id carries a ``/`` hits the same abort, and the heuristic cannot tell them
        apart without asking the Hub.
        """
        if not model_id or "/" not in model_id:
            return None
        from senselab.utils.model_revision import RevisionResolutionError, resolve_revision

        try:
            return resolve_revision(model_id, ref=_DEFAULT_REVISION_REF)
        except RevisionResolutionError as exc:
            # Imported here, not at module scope, so the success path never pays for it — and so
            # this module stays importable by a caller that only wants a cache key.
            from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

            cause = exc.__cause__  # resolve_revision wraps the Hub error it failed on.
            if isinstance(cause, RepositoryNotFoundError) and not isinstance(cause, GatedRepoError):
                logger.warning(
                    "%s is not a Hub repository (%s); its cache key carries no commit. "
                    "Expected for a local backend whose id contains a '/'.",
                    model_id,
                    type(cause).__name__,
                )
                return None
            raise

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
            # Same resolution path as cache_key_for's model_id, not a second one: one aligner
            # id, one place that decides whether it's a Hub repo worth pinning.
            aligner_commit_sha=self._commit_sha_for(aligner_model_id),
        )

    def provenance_for(self, task: str, model_id: str | None, params: Mapping[str, Any]) -> dict[str, Any]:
        """Provenance block recorded on a fresh (cache-miss) outcome.

        ``audio_signature`` here must match what ``summary.json`` reports for the
        pass — ``adaptive/interventions.py::build_cache_index`` joins on it.
        """
        # Resolved once and reused for both fields below: "revision" is what was asked for,
        # "commit_sha" is what ran. Recording only the second cannot distinguish a deliberate
        # pin from a tracked ref that happened to resolve there on the day. "revision" tracks
        # commit_sha's None-ness rather than being unconditionally "main": a model-less stage or
        # a non-Hub backend name has nothing pinned, and claiming a ref for it would assert a
        # request that was never made.
        commit_sha = self._commit_sha_for(model_id)
        return {
            "task": task,
            "model_id": model_id,
            "params": dict(params),
            "audio_signature": self.audio_signature,
            "audio_source": self.audio_source,
            "pass": self.perturbation,
            "variant": self.variant,
            "variant_gain_db": self.variant_gain_db,
            "device": self.device_label,
            "code_version": stage_code_version(task),
            "senselab_version": self.senselab_ver,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "revision": _DEFAULT_REVISION_REF if commit_sha is not None else None,
            "commit_sha": commit_sha,
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
    background_mask: bool = True
    background_sources: bool = True
    task_type: str | None = None
    mask_guard_interval_s: float | None = None
    mask_grid: Any = None
    """The grid the background mask is cut on — ``speech_presence``'s, per D-24.

    Defaulted to ``None`` (``BucketGrid()``'s 0.5 s) only so a caller that builds a plan without
    grids still works. The run must pass the presence grid: the mask is *derived from* presence —
    a region is target-free where presence has settled — so on different grids that derivation
    needs a projection, and every projection is a place to lose localisation. On a shared grid
    row *i* of one is row *i* of the other and the coupling is exact.

    Measured cost of not sharing it: presence produced 1070 buckets at 100 ms and the mask 43 at
    0.5 s, so five presence judgements were projected onto each mask bucket before the mask could
    say anything.
    """
    features: bool = False
    features_win_length: float = 1.0
    features_hop_length: float = 0.5
    align_asr: bool = True
    aligner: Literal["qwen", "mms"] = "qwen"
    qwen_aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    mms_aligner_model: str = "facebook/mms-1b-all"
    # ``None`` = not pinned, which ``stage_alignment`` resolves to English. It was ``str = "en"``
    # while the CLI passed its unset value through unchanged, so the annotation described a default
    # that never took effect.
    asr_language: str | None = None
    qwen_native_timestamps: bool = True
