"""Pure linking over harvested measurements — the light half of the harvest/link split.

``compute.py`` owns the expensive, model-touching **harvest** phase and produces one
``PassHarvest`` per pass; this module reads a ``PassHarvest`` as beliefs under a named policy and
emits the per-signal L1 rows, without touching any model, waveform, or file. Consequences (spec
``20260723-225523-dynamic-uncertainty-workflow`` FR-006 / research.md D8):

- re-linking under a different policy costs milliseconds, not GPU time;
- the adaptive loop can merge new votes and re-fold only covered buckets;
- everything here is unit-testable with synthetic measurement dicts.

**Nothing here folds an axis.** An axis aggregates across signals *and* across passes, so a
per-pass axis is a category error; the single fold lives in ``fuse.fuse_axis``, which sees every
pass at once. This module used to compute one per pass (``aggregate_pass``) and then subtract two
of them to measure perturbation stability (``compute_pass_deltas``) — stability is now measured
per *signal* by ``reliability.signal_stability``, which needs no axis at all.

Imports stay within the sibling analysis modules. "Pure" here means *deterministic and
model-free*, not dependency-free: the speech-presence link clusters L1 embedding vectors, which
brings numpy and scikit-learn into this path. That is a computation over data already in hand — it
touches no model, no waveform and no file, so re-aggregation is still cheap and repeatable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence

from senselab.audio.workflows.audio_analysis.axes import HARVEST_SOURCES
from senselab.audio.workflows.audio_analysis.degradation import scene_degradation
from senselab.audio.workflows.audio_analysis.speech_presence_link import (
    SpeechPresencePolicy,
    policy_from_params,
    votes_for_harvest,
)
from senselab.audio.workflows.audio_analysis.types import SignalResult, SignalRow


@dataclass
class PassHarvest:
    """Everything the aggregate phase needs about one pass — and nothing model-bound.

    Attributes:
        perturbation: e.g. ``"raw"``.
        speech_presence_evidence: per-bucket ``{"start", "end", "evidence", "frame_dispersion"}``
            dicts from ``harvest_speech_presence_evidence`` — **measurements, not votes**. The
            thresholds that turn them into beliefs live in ``speech_presence_link``, so this field
            can be re-linked under a different policy without re-running a model. Call
            ``speech_presence_link.link_speech_presence`` to get votes from it.
        speaker_votes: per-bucket vote dicts from ``harvest_speaker_votes``.
        asr_votes: per-bucket vote dicts from ``harvest_asr_votes``.
        quality_by_bucket: speech_presence-grid bucket key → L1 scene-quality *measurements* in native
            units (dB, hertz, proportion). Degradation scores are derived here at L2 via
            ``degradation.scene_degradation``, because the anchors that produce them are
            calibration — held at L1 they zeroed the columns on every real recording.
        source_by_bucket: speech_presence-grid bucket key → source-mass dict (US2 columns).
        grids: axis → ``{"win_length", "hop_length"}`` actually used at harvest.
        sampling_rate: The pass audio's sample rate, needed to compare a measured spectral
            roll-off against Nyquist. Carried on the harvest rather than re-derived, so the
            aggregate phase stays pure and model-free.
        per_window_embeddings: ``{embedding_model → [WindowEmbedding]}`` — L1 vectors. Clustering
            them is an L2 derivation (``speech_presence_link.derive_window_clusters``), so the
            vectors travel and the conclusion is drawn where it can be re-drawn.
        background_mask_evidence: per-bucket votes on whether the **target** was active, from VAD /
            ASR words / speaker occupancy under the declared task type
            (``mask_harvest.harvest_background_mask_evidence``). Derived from
            ``speech_presence_evidence`` rather than re-measured: the mask shares the presence grid, so
            one measurement serves both and the two cannot drift.
        diarization_by_model: ``{model → diarization block}``, carried so the fusion phase can bind
            fused speaker ids to each diarizer's own labels (C2) without re-running a model, which
            would defeat the harvest/aggregate split. This replaces ``frame_posteriors`` as the
            binding's input: the channels were permutation-arbitrary and could not name anyone,
            while a tool's labels carry both timing and its own identity (D-19).
        provenance_extras: scene_quality / sound_sources / frame_posteriors blocks.
        synthetic_diarization: optional ``{source_id: diar_block}`` synthesized from
            embedding clustering (kept explicit so callers can opt into the legacy
            pass-summary injection instead of being mutated silently).
    """

    perturbation: str
    speech_presence_evidence: list[dict[str, Any]] = field(default_factory=list)
    speaker_votes: list[dict[str, Any]] = field(default_factory=list)
    asr_votes: list[dict[str, Any]] = field(default_factory=list)
    quality_by_bucket: dict[tuple[float, float], dict[str, Any]] = field(default_factory=dict)
    source_by_bucket: dict[tuple[float, float], dict[str, Any]] = field(default_factory=dict)
    grids: dict[str, dict[str, float]] = field(default_factory=dict)
    sampling_rate: int = 16000
    per_window_embeddings: dict[str, list[Any]] = field(default_factory=dict)
    diarization_by_model: dict[str, Any] = field(default_factory=dict)
    background_mask_evidence: list[dict[str, Any]] = field(default_factory=list)
    provenance_extras: dict[str, Any] = field(default_factory=dict)
    synthetic_diarization: dict[str, Any] | None = None


def buckets_for_axis(
    harvest: Any,  # noqa: ANN401 — PassHarvest, duck-typed like the rest of this module
    axis: str,
    *,
    policy: SpeechPresencePolicy | None = None,
) -> list[dict[str, Any]]:
    """One axis's per-bucket belief buckets, read off ``harvest`` the way that axis declares.

    The single answer to "where does this axis's evidence come from", for the three readers that
    each had their own: :func:`link_pass` (the L1→L2 link), ``fuse.write_final_uncertainty`` (the
    run's fold) and ``adaptive.belief.VoteStore.from_harvests`` (the loop's ingest). Two of them
    read four axes off ``axes.AXES`` while the third enumerated three in a literal tuple, so
    ``background_mask`` was folded per bucket by L2 and rebuilt from one vote per mask *region* by
    the loop — 1070 rows at round 0, one row by round 4, and nothing reporting a loss. A reader
    that cannot name a per-axis field cannot skip an axis.

    Args:
        harvest: One pass's ``PassHarvest``.
        axis: The axis to read. Must be harvested and must declare a
            :class:`~.axes.HarvestSource`.
        policy: Presence-link policy, for an axis whose harvest holds *measurements*. ``None`` uses
            the documented default anchors.

    Returns:
        Bucket dicts in the shape ``fuse.fuse_axis`` consumes — votes already linked, so no
        consumer has to know which axes needed linking.

    Raises:
        KeyError: For an axis with no declared harvest source. An axis marked ``harvested`` whose
            evidence no reader can find is the failure this replaces, and it has to be loud: an
            empty result reads as "this axis had nothing to say".
        NotImplementedError: For a measurements-holding axis other than speech presence. The only
            link that exists reads ``speech_presence_evidence``; returning its votes for a
            different axis would be a silent mislabel.
    """
    source = HARVEST_SOURCES.get(str(axis))
    if source is None:
        raise KeyError(
            f"axis {axis!r} declares no harvest source; add a HarvestSource to its axes.AXES entry "
            f"— known: {sorted(HARVEST_SOURCES)}"
        )
    if source.holds == "measurements":
        if source.field != "speech_presence_evidence":
            raise NotImplementedError(
                f"axis {axis!r} holds measurements in {source.field!r}, and the only link that exists reads "
                "speech_presence_evidence; give the axis its own linker rather than borrowing presence's"
            )
        return votes_for_harvest(harvest, **({"policy": policy} if policy is not None else {}))
    return list(getattr(harvest, source.field, None) or [])


def mask_from_pvoice(p: float) -> float:
    """p_voice → mask weight: 1.0 if >= 0.5, else linear ramp to 0 at p = 0."""
    return 1.0 if p >= 0.5 else max(0.0, min(1.0, p / 0.5))


def intensity_mask(start: float, end: float, speech_presence_pv_intervals: list[tuple[float, float, float]]) -> float:
    """Average speech_presence-derived mask over speech_presence buckets overlapping ``[start, end)``.

    Overlap-averaging (not closest-only) so a query bucket spanning a half-voice /
    half-silence region is masked proportionally; works when speech_presence runs on a
    finer grid. Falls back to 1.0 (no masking) when speech_presence produced no p_voice.
    """
    vals = [mask_from_pvoice(p) for s_p, e_p, p in speech_presence_pv_intervals if s_p < end and e_p > start]
    return sum(vals) / len(vals) if vals else 1.0


DEFAULT_UTTERANCE_SCENE_COUPLING: dict[str, float] = {"w_q": 0.5, "w_s": 0.25}
"""Default scene→asr coupling weights (FR-019).

``w_q`` weights SNR-based quality degradation, ``w_s`` the mass of competing
non-speech sources (machine + environment). Quality is weighted twice as heavily
because a poor SNR degrades the acoustic evidence the ASR actually decoded, while a
competing source may be present without overlapping the speech. At the defaults a
fully degraded bucket with a fully competing background yields a 1.75× multiplier.
Set both to 0.0 to disable coupling entirely.
"""


def _overlap_mean(start: float, end: float, values: list[tuple[float, float, float]]) -> float | None:
    """Mean of scene values whose speech_presence bucket overlaps ``[start, end)``.

    Overlap-averaging (rather than nearest-bucket) so a wide asr bucket
    spanning several finer speech_presence buckets sees their average, matching how
    ``intensity_mask`` bridges the two grids.
    """
    hits = [v for s, e, v in values if s < end and e > start]
    return sum(hits) / len(hits) if hits else None


def scene_quality_coupling(
    start: float,
    end: float,
    *,
    quality_degradation: list[tuple[float, float, float]],
    competing_source_mass: list[tuple[float, float, float]],
    weights: dict[str, float],
) -> float:
    """Multiplier (``>= 1.0``) by which poor scene conditions inflate asr doubt.

    ``1.0 + w_q · quality_degradation + w_s · competing_source_mass``, with each term
    dropped when the corresponding scene column is absent — so a run with
    ``--skip`` scene features (or a pass where the estimators failed) returns exactly
    ``1.0`` and leaves the reported uncertainty untouched (SC-008).

    Args:
        start: Utterance bucket start, seconds.
        end: Utterance bucket end, seconds.
        quality_degradation: ``(bucket_start, bucket_end, quality_snr)`` triples,
            where ``quality_snr`` is 0 = clean, 1 = fully degraded.
        competing_source_mass: ``(bucket_start, bucket_end, machine + environment)``
            triples in ``[0, 1]``.
        weights: ``{"w_q": float, "w_s": float}``.

    Returns:
        The coupling multiplier, at least 1.0.
    """
    coupling = 1.0
    degradation = _overlap_mean(start, end, quality_degradation)
    if degradation is not None:
        coupling += float(weights.get("w_q", 0.0)) * max(0.0, min(1.0, degradation))
    competing = _overlap_mean(start, end, competing_source_mass)
    if competing is not None:
        coupling += float(weights.get("w_s", 0.0)) * max(0.0, min(1.0, competing))
    return max(1.0, coupling)


MAX_PROBABILITY_STD = 0.5
"""Largest possible standard deviation of a value bounded in ``[0, 1]``.

Reached by a half-0 / half-1 split, which is exactly the onset-crossing case the frame dispersion
signal is meant to catch."""


def _dispersion_to_instability(dispersion: float | None) -> float | None:
    """Map within-bucket frame dispersion onto ``[0, 1]`` for folding into a belief.

    L1 reports dispersion in probability units and does not rescale it, because rescaling makes it
    a different statistic and invites reading a dispersion as a probability. The rescale is a
    *modelling choice* about how temporal instability should contribute to doubt, so it happens
    here, once, where it can be seen and changed — rather than being baked into the recorded
    measurement as ``clip(2 * std, 0, 1)`` was.
    """
    if dispersion is None:
        return None
    return max(0.0, min(1.0, float(dispersion) / MAX_PROBABILITY_STD))


def _quality_anchors(params: dict[str, Any]) -> dict[str, float] | None:
    """Extract fitted scene-quality anchors from run params, or ``None`` for the defaults.

    The calibration profile is a versioned artifact (US5), so only the keys ``degradation``
    recognises are forwarded — an unknown key would otherwise silently do nothing while appearing
    to have been applied.
    """
    calibration = params.get("calibration")
    if not isinstance(calibration, dict):
        return None
    anchors = {
        key: float(calibration[key])
        for key in ("snr_clean_db", "snr_floor_db", "c50_clean_db", "c50_floor_db")
        if isinstance(calibration.get(key), (int, float))
    }
    return anchors or None


def apply_scene_coupling(
    asr_rows: Sequence[MutableMapping[str, Any]],
    scene_rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Inflate the asr rows' *policy fold* where the scene degrades the evidence (FR-019).

    Applied to ``triage_score`` only — the policy fold, which exists to rank where to spend budget
    — and never to ``uncertainty``, which is the entropy measure and has no policy in it. The
    multiplier, its weights and the pre-coupling value go onto the row, so the adjustment is
    re-decidable without re-running anything.

    Lives here rather than in ``compute`` because both callers need it. It ran once, on
    ``compute_uncertainty_axes``'s in-memory rows, and then ``write_final_uncertainty``'s rounds
    re-folded every axis from the harvests and overwrote ``triage_score`` and ``coupled_from`` —
    so no persisted row ever carried the coupling, while ``scene_quality_coupling`` and
    ``triage_score_pre_coupling`` stayed behind on the in-memory row asserting an adjustment its
    number did not contain.

    Args:
        asr_rows: The asr axis's rows, mutated in place.
        scene_rows: Presence rows carrying the scene measurements (``quality_snr``,
            ``src_machine``, ``src_environment``). Empty means nothing was measured, and every
            multiplier is then 1.0 — which is the identity, not a claim that the scene is clean.
        params: Comparator params; ``asr_scene_coupling`` overrides the default weights.

    Returns:
        The provenance block naming the weights, the defaults and what the coupling applies to.
    """
    weights = _coupling_weights(dict(params))
    quality_intervals = [
        (float(r["start"]), float(r["end"]), float(r["quality_snr"]))
        for r in scene_rows
        if isinstance(r.get("quality_snr"), (int, float))
    ]
    competing_intervals = [
        (
            float(r["start"]),
            float(r["end"]),
            float(r.get("src_machine") or 0.0) + float(r.get("src_environment") or 0.0),
        )
        for r in scene_rows
        if isinstance(r.get("src_machine"), (int, float)) or isinstance(r.get("src_environment"), (int, float))
    ]
    for row in asr_rows:
        coupling = scene_quality_coupling(
            float(row["start"]),
            float(row["end"]),
            quality_degradation=quality_intervals,
            competing_source_mass=competing_intervals,
            weights=weights,
        )
        row["scene_quality_coupling"] = coupling
        row["triage_score_pre_coupling"] = row.get("triage_score")
        if isinstance(row.get("triage_score"), (int, float)) and coupling != 1.0:
            row["triage_score"] = max(0.0, min(1.0, float(row["triage_score"]) * coupling))
        if coupling != 1.0:
            row["coupled_from"] = sorted({*(row.get("coupled_from") or []), "scene_quality"})
    return {
        "weights": dict(weights),
        "defaults": dict(DEFAULT_UTTERANCE_SCENE_COUPLING),
        "applies_to": "triage_score",
    }


def _coupling_weights(params: dict[str, Any]) -> dict[str, float]:
    """Resolve coupling weights from ``params``, falling back to the documented defaults."""
    raw = params.get("asr_scene_coupling")
    if not isinstance(raw, dict):
        return dict(DEFAULT_UTTERANCE_SCENE_COUPLING)
    weights = dict(DEFAULT_UTTERANCE_SCENE_COUPLING)
    for key in ("w_q", "w_s"):
        if key in raw:
            try:
                weights[key] = float(raw[key])
            except (TypeError, ValueError):
                continue
    return weights


def _signal_rows_from_buckets(
    buckets: Any,  # noqa: ANN401 — sequence of harvested bucket dicts
    key: str,
    into: dict[str, list[SignalRow]],
) -> None:
    """Accumulate one bucket family's per-signal measurements into ``into``.

    No fold and no threshold: the entry a signal reported is copied through as the
    measurement, and the provenance fields the signal declared are lifted onto the row so a
    reader does not have to know which harvester wrote it.
    """
    for bucket in buckets or []:
        if not isinstance(bucket, Mapping):
            continue
        start, end = float(bucket.get("start", 0.0)), float(bucket.get("end", 0.0))
        for name, entry in (bucket.get(key) or {}).items():
            signal = str(name)
            if signal.startswith("__"):
                # Synthetic cross-signal blocks (pairwise distances, quality, sources) are not a
                # signal's own report; they are emitted separately below where they belong.
                continue
            measurement = dict(entry) if isinstance(entry, Mapping) else {"value": entry}
            rows = into.setdefault(signal, [])
            for existing in rows:
                # The same signal can report on two axes (an ASR model votes on presence and on
                # asr). One signal, one file: merge rather than shadow.
                if existing.start == start and existing.end == end:
                    existing.measurement.update(measurement)
                    break
            else:
                rows.append(
                    SignalRow(
                        start=start,
                        end=end,
                        signal=signal,
                        measurement=measurement,
                        units=measurement.get("units"),
                        native_window_s=_as_float(measurement.get("native_window_s")),
                        resolution_s=_as_float(measurement.get("resolution_s")),
                        model_id=measurement.get("model_id") or signal,
                    )
                )


def _as_float(value: Any) -> float | None:  # noqa: ANN401
    """``float(value)`` when it is a finite number, else ``None`` — never a default."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass
class LinkedPass:
    """One pass, linked: L1 rows per signal, and the belief buckets L2 fuses.

    Two products, deliberately separate. ``signal_results`` is what L1 writes — measurements in
    native units, no axis anywhere. ``buckets_by_axis`` is L2's input: the same measurements read
    as beliefs under a *named* policy, which is recorded in ``provenance``. Neither is an axis
    value; folding across signals happens exactly once, in ``fuse.fuse_axis``.
    """

    perturbation: str
    buckets_by_axis: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    signal_results: dict[str, SignalResult] = field(default_factory=dict)
    quality_scores: dict[tuple[float, float], dict[str, float]] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def link_pass(harvest: PassHarvest, *, params: dict[str, Any]) -> LinkedPass:
    """Read one pass's L1 measurements as beliefs, and emit its per-signal L1 rows.

    Args:
        harvest: One pass's harvested measurements.
        params: Run parameters (calibration anchors, presence-policy thresholds).

    Returns:
        A :class:`LinkedPass`. Pure: same harvest + same params ⇒ identical output.

    **No axis is computed here.** An axis aggregates across signals *and* across passes, so it
    cannot be produced from one pass; the fold lives in ``fuse.fuse_axis``, which receives every
    pass at once. What happens here is the *link* — measurements read as beliefs under the policy
    recorded in ``provenance`` — and the emission of the per-signal L1 rows.
    """
    presence_policy = policy_from_params(params)
    # Every harvested axis, read the way it declares — including ``background_mask``, whose
    # per-bucket harvest was absent from this dict while ``reliability`` and ``fuse`` both knew
    # about it. So this pass contributed no mask evidence to ``L2/round/0/derivatives/votes/`` and
    # the driver overwrote that (empty) file with one vote per mask region.
    buckets_by_axis = {axis: buckets_for_axis(harvest, axis, policy=presence_policy) for axis in HARVEST_SOURCES}
    presence_buckets = buckets_by_axis["speech_presence"]

    signal_rows: dict[str, list[SignalRow]] = {}
    _signal_rows_from_buckets(harvest.speech_presence_evidence, "evidence", signal_rows)
    _signal_rows_from_buckets(harvest.speaker_votes, "votes", signal_rows)
    _signal_rows_from_buckets(harvest.asr_votes, "votes", signal_rows)

    # Frame dispersion is a per-bucket L1 measurement of how much the frame posteriors moved
    # inside the bucket. It reached the belief store only through the in-process path, so the
    # artifact-driven path read ``None`` everywhere and one of P2's two triggers was structurally
    # dead. Persisting it as a signal fixes that in both paths.
    dispersion_rows = [
        SignalRow(
            start=float(b["start"]),
            end=float(b["end"]),
            signal="frame_dispersion",
            measurement={"frame_dispersion": float(b["frame_dispersion"]), "units": "probability"},
            units="probability",
        )
        for b in harvest.speech_presence_evidence
        if isinstance(b, Mapping) and isinstance(b.get("frame_dispersion"), (int, float))
    ]
    if dispersion_rows:
        signal_rows["frame_dispersion"] = dispersion_rows

    # Scene quality: L1 keeps the dB / hertz / proportion measurements; the anchored [0, 1]
    # degradation scores are derived here, at L2, where a fitted calibration profile can replace
    # the defaults and where a saturated column is visibly a fusion choice rather than a
    # measurement.
    anchors = _quality_anchors(params)
    quality_rows: list[SignalRow] = []
    quality_scores: dict[tuple[float, float], dict[str, float]] = {}
    for (start, end), quality in sorted(harvest.quality_by_bucket.items()):
        native = {k: v for k, v in quality.items() if k != "provenance"}
        quality_rows.append(
            SignalRow(
                start=float(start),
                end=float(end),
                signal="scene_quality",
                measurement=native,
                units="mixed",
                model_id="scene_quality",
            )
        )
        scores = scene_degradation(quality, sampling_rate=harvest.sampling_rate, calibration=anchors)
        if scores:
            quality_scores[(round(float(start), 6), round(float(end), 6))] = scores
    if quality_rows:
        signal_rows["scene_quality"] = quality_rows

    source_rows = [
        SignalRow(
            start=float(start),
            end=float(end),
            signal="sound_sources",
            measurement=dict(source),
            units="proportion",
            model_id="sound_sources",
        )
        for (start, end), source in sorted(harvest.source_by_bucket.items())
    ]
    if source_rows:
        signal_rows["sound_sources"] = source_rows

    # The scene blocks ride along on the presence buckets under ``__``-prefixed keys, as they
    # always have: they are cross-signal context for the bucket rather than one signal's report,
    # and consumers that weigh evidence per source need them next to the votes they qualify.
    for bucket in presence_buckets:
        key = (round(float(bucket["start"]), 6), round(float(bucket["end"]), 6))
        quality_block = harvest.quality_by_bucket.get(key)
        if quality_block is not None:
            bucket["votes"] = {
                **bucket["votes"],
                "__quality__": {k: v for k, v in quality_block.items() if k != "provenance"},
            }
        source_block = harvest.source_by_bucket.get(key)
        if source_block is not None:
            bucket["votes"] = {**bucket["votes"], "__sources__": dict(source_block.get("_raw") or {})}

    provenance_common = {
        "pass": harvest.perturbation,
        "grids": {k: dict(v) for k, v in harvest.grids.items()},
        "sampling_rate": harvest.sampling_rate,
        "speech_presence_policy": asdict(presence_policy),
        "quality_calibration": anchors,
        **{k: v for k, v in harvest.provenance_extras.items()},
    }
    return LinkedPass(
        perturbation=harvest.perturbation,
        buckets_by_axis=buckets_by_axis,
        signal_results={
            signal: SignalResult(
                perturbation=harvest.perturbation,  # type: ignore[arg-type]
                signal=signal,
                rows=sorted(rows, key=lambda r: (r.start, r.end)),
                provenance=provenance_common,
            )
            for signal, rows in sorted(signal_rows.items())
        },
        quality_scores=quality_scores,
        provenance=provenance_common,
    )
