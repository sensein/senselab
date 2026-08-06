"""The run configuration — one versioned file, loaded once, its identity stamped on every artifact.

``scripts/analyze_audio.py`` takes two things: an audio file and an output directory. Everything else
it used to accept as a flag is a *value* rather than a choice, and lives in
``data/run_config/default.yaml`` beside the derivation that produced it. This module loads that file,
deep-merges an optional override over it, validates the handful of relations that a wrong value would
otherwise break silently, and hands back a frozen :class:`RunConfig`.

**Why not per-knob flags.** Seventy of them existed, and the run recipes in the repo's own docs
differed from one another only in flags whose right value a reader had no basis to pick. Worse, the
grid flags were live: a caller could set the four axes to four different spacings, which is exactly
what the shipped defaults did, and the result was that every cross-axis coupling in the pipeline ran
against zero shared bucket keys. A knob that no one can choose between settings for is not
configurability; it is an unmeasured decision with a public interface.

**Why the identity travels.** :attr:`RunConfig.identity` carries ``{name, version, config_hash}``,
hashed over the *merged* mapping rather than the file on disk, so two runs that merged different
overrides can never claim the same hash. It is stamped onto ``final/summary.json``, every
``L1/<pass>/signals/*.parquet``, every ``L2/round<N>/uncertainty/*.parquet``, the disagreements index
and the LS bundle — the same treatment ``data/detection_margin/<version>.json`` already gets, and for
the same reason: a number whose configuration cannot be named cannot be reproduced.

The adaptive loop's policy is a *section* of this file (``adaptive:``), not a second file. It keeps
its own hash (``policy_hash``) because a policy change and a model change are not the same event and
a reader has to be able to tell them apart.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, Mapping, cast

from senselab.audio.workflows.audio_analysis.axes import DEFAULT_TIME_GRID

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ConfigIdentity",
    "RunConfig",
    "deep_merge",
    "load_run_config",
]

DEFAULT_CONFIG_PATH = Path(__file__).parent / "data" / "run_config" / "default.yaml"

AGGREGATORS = ("min", "mean", "harmonic_mean", "disagreement_weighted")
DEFAULT_SNR_FLOOR_DB: Final[float] = 10.0
"""SNR below which the recording counts as degraded — the config default for ``triage.snr_floor_db``.

One declaration, because two things read it and they must agree: whether to *compute* the enhanced
pass (``enhancement.mode: auto``) and whether a computed pass's readings are *admitted to the fold*
(``fuse.SnrGate``). A standalone adaptive-loop run on a finished run directory falls back to this
when the run recorded no floor, so it gates on the same number a fresh run would.
"""

ENHANCEMENT_MODES = ("auto", "always", "never")
ALIGNERS = ("qwen", "mms")
CLUSTERING_ALGORITHMS = ("spectral", "kmeans")
DEVICES = ("cpu", "cuda", "mps", "auto")


@dataclass(frozen=True)
class ConfigIdentity:
    """What a run was configured by, in the three fields an artifact needs to name it.

    Attributes:
        name: The config's own name (``senselab-audio-analysis/default`` for the packaged one).
            Names the *values*, not the schema.
        version: The schema version of the file's shape. Bumped when the shape changes, never to
            record a changed value — the same split ``detection_margin`` keeps between
            ``profile_version`` and ``calibrated_as``.
        config_hash: sha256 over the merged mapping, canonicalised. Computed after merging so an
            override cannot inherit the packaged file's identity.
        sources: Every file that contributed, in merge order. An override that silently failed to
            load would otherwise be indistinguishable from one that changed nothing.
    """

    name: str
    version: str
    config_hash: str
    sources: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        """The mapping stamped into artifact provenance."""
        return {
            "name": self.name,
            "version": self.version,
            "config_hash": self.config_hash,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class RunConfig:
    """Every value ``analyze_audio`` needs that is not the input file or the output directory.

    Flat, and deliberately not a nested dict: an attribute that does not exist is an
    ``AttributeError`` at the call site, whereas ``cfg["grid"]["win_lenght"]`` is a ``KeyError`` at
    best and a silent ``.get(...)`` default at worst. The YAML is nested for a reader; this is flat
    for a caller.

    ``raw`` keeps the merged mapping so the adaptive section and any override key this class does not
    model can still be read, and so the hash in :attr:`identity` is checkable against something.
    """

    identity: ConfigIdentity
    raw: Mapping[str, Any]

    output_dir: Path
    device: str

    grid_win_length: float
    grid_hop_length: float

    cache_enabled: bool
    cache_dir: Path

    # ── stages ──
    run_diarization: bool
    run_ast: bool
    run_yamnet: bool
    run_features: bool
    run_asr: bool
    run_alignment: bool
    run_comparisons: bool
    align_asr: bool
    background_mask: bool
    scene_quality: bool
    sound_sources: bool
    stability: bool
    per_speaker_identity: bool
    adaptive_outputs: bool
    invariance_probe: bool

    # ── models ──
    diarization_models: tuple[str, ...]
    asr_models: tuple[str, ...]
    embeddings_models: tuple[str, ...]
    ast_model: str
    yamnet_model: str
    enhancement_model: str

    # ── alignment ──
    # Narrowed rather than left as ``str``: ``PassPlan.aligner`` is a ``Literal``, and the widening
    # was invisible for as long as the value arrived from an argparse ``Namespace`` (every attribute
    # of which is ``Any``). :func:`_validate` is what makes the narrowing sound.
    aligner: Literal["qwen", "mms"]
    qwen_aligner_model: str
    mms_aligner_model: str
    qwen_native_timestamps: bool
    asr_language: str | None

    # ── scene classification ──
    ast_win_length: float
    ast_hop_length: float
    yamnet_win_length: float
    yamnet_hop_length: float
    scene_top_k: int

    # ── features ──
    features_win_length: float
    features_hop_length: float

    # ── embeddings ──
    embedding_window_s: float
    embedding_hop_s: float

    # ── speaker ──
    speaker_same_floor: float
    speaker_diff_floor: float
    speaker_cluster_cosine_threshold: float
    clustering_algorithm: str

    # ── task ──
    task_type: str | None
    mask_guard_interval_s: float | None

    # ── enhancement + triage ──
    enhancement_mode: str
    triage_speech_threshold: float
    triage_min_speech_s: float
    triage_snr_floor_db: float
    triage_low_snr_fraction: float

    # ── uncertainty ──
    aggregator: str
    disagreements_top_n: int
    speech_presence_labels: tuple[str, ...]
    asr_scene_coupling_w_q: float
    asr_scene_coupling_w_s: float

    # ── profiles ──
    detection_margin_profile: str | None
    influence_profile: str | None
    calibration_profile: Path | None

    # ── rounds ──
    max_rounds: int

    skipped_stages: frozenset[str] = field(default_factory=frozenset)
    """Stage names the run must not execute — derived from the ``stages`` block, and *mutable in
    effect* only through :meth:`with_skipped`, so a triage decision that widens it produces a new
    config rather than editing one every later stage has already read."""

    def with_skipped(self, extra: set[str]) -> RunConfig:
        """A copy with ``extra`` added to :attr:`skipped_stages`.

        Triage widens the skip set when it finds no speech, and it runs *after* the config is built.
        Returning a new config rather than mutating keeps "what was configured" and "what the audio
        turned out to justify" distinguishable — the previous code mutated ``args.skip`` in place, so
        a plan built before triage and one built after read the same field and disagreed.
        """
        return dataclasses.replace(self, skipped_stages=frozenset(self.skipped_stages | set(extra)))

    @property
    def adaptive(self) -> Mapping[str, Any]:
        """The ``adaptive:`` section, as :func:`~.adaptive.policy.load_policy` reads it."""
        section = self.raw.get("adaptive")
        return section if isinstance(section, Mapping) else {}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive dict merge — an override replaces scalars and lists, and descends into mappings.

    Lists replace rather than concatenate: an override naming two ASR models means *those two*, and
    appending would make "run fewer models" unexpressible.
    """
    out: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_run_config(path: Path | None = None, *, overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """Load the packaged config, deep-merge ``path`` and ``overrides``, validate, and freeze.

    Args:
        path: Optional YAML deep-merged over the packaged default — the whole of ``--config``.
        overrides: Optional in-memory mapping merged last. For tests and for the two values the CLI
            genuinely owns; there are deliberately no per-knob flags feeding this.

    Returns:
        A frozen :class:`RunConfig` whose :attr:`~RunConfig.identity` hashes the merged mapping.

    Raises:
        ValueError: For a value that would otherwise fail somewhere far from its cause — a hop above
            its window, floors in the wrong order, an unknown aggregator. Each is checked here rather
            than at the point of use, because at the point of use the run has already spent the
            inference.
    """
    merged = _load_yaml(DEFAULT_CONFIG_PATH)
    sources = [str(DEFAULT_CONFIG_PATH)]
    if path is not None:
        override = _load_yaml(Path(path))
        merged = deep_merge(merged, override)
        sources.append(str(Path(path)))
    if overrides:
        merged = deep_merge(merged, overrides)
        sources.append("<in-memory overrides>")

    _validate(merged)
    canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"), default=str)
    identity = ConfigIdentity(
        name=str(merged.get("name", "unnamed")),
        version=str(merged.get("version", "0")),
        config_hash=hashlib.sha256(canonical.encode()).hexdigest(),
        sources=tuple(sources),
    )
    return _build(merged, identity)


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: run config must be a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _section(merged: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = merged.get(name)
    return value if isinstance(value, Mapping) else {}


def _validate(merged: Mapping[str, Any]) -> None:
    """Reject configurations that would produce a wrong artifact rather than an error."""
    grid = _section(merged, "grid")
    win, hop = float(grid.get("win_length", 0.0)), float(grid.get("hop_length", 0.0))
    if win <= 0:
        raise ValueError(f"grid.win_length must be > 0, got {win}")
    if not 0 < hop <= win:
        raise ValueError(f"grid.hop_length must be in (0, {win}], got {hop}")
    if (win, hop) != DEFAULT_TIME_GRID and hop < win:
        # Not forbidden — a caller with a measured reason may overlap — but it is the failure D-24
        # names, so it is announced rather than accepted in silence.
        import sys

        print(
            f"warn: grid.win_length={win} > grid.hop_length={hop}, so adjacent rows share "
            f"{100 * (1 - hop / win):.0f}% of their audio; N rows are not N independent measurements "
            "(axes.DEFAULT_TIME_GRID sets window == hop for this reason)",
            file=sys.stderr,
        )

    scene = _section(merged, "scene")
    for prefix in ("ast", "yamnet"):
        s_win = float(scene.get(f"{prefix}_win_length", 0.0))
        s_hop = float(scene.get(f"{prefix}_hop_length", 0.0))
        if s_win <= 0 or not 0 < s_hop <= s_win:
            raise ValueError(f"scene.{prefix}_hop_length must be in (0, {prefix}_win_length], got {s_hop}/{s_win}")

    features = _section(merged, "features")
    f_win, f_hop = float(features.get("win_length", 0.0)), float(features.get("hop_length", 0.0))
    if f_win <= 0 or not 0 < f_hop <= f_win:
        raise ValueError(f"features.hop_length must be in (0, win_length], got {f_hop}/{f_win}")

    embeddings = _section(merged, "embeddings")
    e_win, e_hop = float(embeddings.get("window_s", 0.0)), float(embeddings.get("hop_s", 0.0))
    if e_win <= 0 or not 0 < e_hop <= e_win:
        raise ValueError(f"embeddings.hop_s must be in (0, window_s], got {e_hop}/{e_win}")

    speaker = _section(merged, "speaker")
    same, diff = float(speaker.get("same_floor", 0.0)), float(speaker.get("diff_floor", 0.0))
    if not 0.0 <= same < diff <= 2.0:
        raise ValueError(f"speaker.same_floor must be < diff_floor, both in [0, 2]; got {same} / {diff}")
    algorithm = str(speaker.get("clustering_algorithm", ""))
    if algorithm not in CLUSTERING_ALGORITHMS:
        raise ValueError(f"speaker.clustering_algorithm must be one of {CLUSTERING_ALGORITHMS}, got {algorithm!r}")

    uncertainty = _section(merged, "uncertainty")
    aggregator = str(uncertainty.get("aggregator", ""))
    if aggregator not in AGGREGATORS:
        raise ValueError(f"uncertainty.aggregator must be one of {AGGREGATORS}, got {aggregator!r}")
    top_n = int(uncertainty.get("disagreements_top_n", 0))
    if top_n < 0:
        raise ValueError(f"uncertainty.disagreements_top_n must be >= 0, got {top_n}")
    if not [str(label).strip() for label in (uncertainty.get("speech_presence_labels") or []) if str(label).strip()]:
        raise ValueError(
            "uncertainty.speech_presence_labels must name at least one AudioSet label — with none, "
            "AST and YAMNet cannot contribute to the presence axis and their absence is invisible"
        )

    mode = str(_section(merged, "enhancement").get("mode", ""))
    if mode not in ENHANCEMENT_MODES:
        raise ValueError(f"enhancement.mode must be one of {ENHANCEMENT_MODES}, got {mode!r}")

    aligner = str(_section(merged, "alignment").get("aligner", ""))
    if aligner not in ALIGNERS:
        raise ValueError(f"alignment.aligner must be one of {ALIGNERS}, got {aligner!r}")

    device = str(merged.get("device", ""))
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")

    max_rounds = int(_section(merged, "rounds").get("max_rounds", 0))
    if max_rounds < 1:
        raise ValueError(f"rounds.max_rounds must be >= 1 (1 = baseline only), got {max_rounds}")

    if not _section(merged, "adaptive"):
        raise ValueError("the config has no `adaptive:` section — the loop's policy lives in this file")


def _build(merged: Mapping[str, Any], identity: ConfigIdentity) -> RunConfig:
    grid = _section(merged, "grid")
    cache = _section(merged, "cache")
    stages = _section(merged, "stages")
    models = _section(merged, "models")
    alignment = _section(merged, "alignment")
    scene = _section(merged, "scene")
    features = _section(merged, "features")
    embeddings = _section(merged, "embeddings")
    speaker = _section(merged, "speaker")
    task = _section(merged, "task")
    enhancement = _section(merged, "enhancement")
    triage = _section(merged, "triage")
    uncertainty = _section(merged, "uncertainty")
    coupling = uncertainty.get("asr_scene_coupling") or {}
    profiles = _section(merged, "profiles")
    rounds = _section(merged, "rounds")

    def _path(value: Any) -> Path | None:  # noqa: ANN401 — YAML scalar
        return Path(str(value)) if value else None

    skipped = {
        name
        for name, key in (
            ("diarization", "diarization"),
            ("ast", "ast"),
            ("yamnet", "yamnet"),
            ("features", "features"),
            ("asr", "asr"),
            ("alignment", "alignment"),
            ("comparisons", "comparisons"),
        )
        if not bool(stages.get(key, True))
    }

    return RunConfig(
        identity=identity,
        raw=merged,
        output_dir=Path(str(merged.get("output_dir", "artifacts/analyze_audio"))),
        device=str(merged.get("device", "auto")),
        grid_win_length=float(grid["win_length"]),
        grid_hop_length=float(grid["hop_length"]),
        cache_enabled=bool(cache.get("enabled", True)),
        cache_dir=Path(str(cache.get("dir", "artifacts/analyze_audio_cache"))),
        run_diarization=bool(stages.get("diarization", True)),
        run_ast=bool(stages.get("ast", True)),
        run_yamnet=bool(stages.get("yamnet", True)),
        run_features=bool(stages.get("features", True)),
        run_asr=bool(stages.get("asr", True)),
        run_alignment=bool(stages.get("alignment", True)),
        run_comparisons=bool(stages.get("comparisons", True)),
        align_asr=bool(stages.get("align_asr", True)),
        background_mask=bool(stages.get("background_mask", True)),
        scene_quality=bool(stages.get("scene_quality", True)),
        sound_sources=bool(stages.get("sound_sources", True)),
        stability=bool(stages.get("stability", True)),
        per_speaker_identity=bool(stages.get("per_speaker_identity", True)),
        adaptive_outputs=bool(stages.get("adaptive_outputs", True)),
        invariance_probe=bool(stages.get("invariance_probe", False)),
        diarization_models=tuple(str(m) for m in (models.get("diarization") or [])),
        asr_models=tuple(str(m) for m in (models.get("asr") or [])),
        embeddings_models=tuple(str(m) for m in (models.get("embeddings") or [])),
        ast_model=str(models.get("ast", "")),
        yamnet_model=str(models.get("yamnet", "")),
        enhancement_model=str(models.get("enhancement", "")),
        # Safe by construction: ``_validate`` rejected anything outside ``ALIGNERS`` above.
        aligner=cast(Literal["qwen", "mms"], str(alignment.get("aligner", "qwen"))),
        qwen_aligner_model=str(alignment.get("qwen_model", "")),
        mms_aligner_model=str(alignment.get("mms_model", "")),
        qwen_native_timestamps=bool(alignment.get("qwen_native_timestamps", True)),
        asr_language=(str(alignment["language"]) if alignment.get("language") else None),
        ast_win_length=float(scene["ast_win_length"]),
        ast_hop_length=float(scene["ast_hop_length"]),
        yamnet_win_length=float(scene["yamnet_win_length"]),
        yamnet_hop_length=float(scene["yamnet_hop_length"]),
        scene_top_k=int(scene.get("top_k", 50)),
        features_win_length=float(features["win_length"]),
        features_hop_length=float(features["hop_length"]),
        embedding_window_s=float(embeddings["window_s"]),
        embedding_hop_s=float(embeddings["hop_s"]),
        speaker_same_floor=float(speaker["same_floor"]),
        speaker_diff_floor=float(speaker["diff_floor"]),
        speaker_cluster_cosine_threshold=float(speaker.get("cluster_cosine_threshold", 0.5)),
        clustering_algorithm=str(speaker.get("clustering_algorithm", "spectral")),
        task_type=(str(task["type"]) if task.get("type") else None),
        mask_guard_interval_s=(
            float(task["mask_guard_interval_s"]) if task.get("mask_guard_interval_s") is not None else None
        ),
        enhancement_mode=str(enhancement.get("mode", "always")),
        triage_speech_threshold=float(triage.get("speech_threshold", 0.5)),
        triage_min_speech_s=float(triage.get("min_speech_s", 0.3)),
        triage_snr_floor_db=float(triage.get("snr_floor_db", DEFAULT_SNR_FLOOR_DB)),
        triage_low_snr_fraction=float(triage.get("low_snr_fraction", 0.4)),
        aggregator=str(uncertainty.get("aggregator", "min")),
        disagreements_top_n=int(uncertainty.get("disagreements_top_n", 100)),
        speech_presence_labels=tuple(
            str(label).strip() for label in (uncertainty.get("speech_presence_labels") or []) if str(label).strip()
        ),
        asr_scene_coupling_w_q=float(coupling.get("w_q", 0.0)),
        asr_scene_coupling_w_s=float(coupling.get("w_s", 0.0)),
        detection_margin_profile=(str(profiles["detection_margin"]) if profiles.get("detection_margin") else None),
        influence_profile=(str(profiles["influence"]) if profiles.get("influence") else None),
        calibration_profile=_path(profiles.get("calibration")),
        max_rounds=int(rounds.get("max_rounds", 3)),
        skipped_stages=frozenset(skipped),
    )
