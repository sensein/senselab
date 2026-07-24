"""Pure aggregation over harvested votes — the light half of the harvest/aggregate split.

``compute.py`` owns the expensive, model-touching **harvest** phase and produces one
``PassHarvest`` per pass; this module folds a ``PassHarvest`` into ``AxisResult`` rows
without touching any model, waveform, or file. Consequences (spec
``20260723-225523-dynamic-uncertainty-workflow`` FR-006 / research.md D8):

- re-aggregating with a different ``aggregator`` costs milliseconds, not GPU time;
- the adaptive loop can merge new votes and re-fold only covered buckets;
- everything here is unit-testable with synthetic vote dicts.

No imports beyond stdlib + the sibling pure modules (``aggregate``, ``types``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_identity,
    aggregate_presence,
    aggregate_utterance,
    presence_p_voice,
)
from senselab.audio.workflows.audio_analysis.types import AxisResult, UncertaintyRow


@dataclass
class PassHarvest:
    """Everything the aggregate phase needs about one pass — and nothing model-bound.

    Attributes:
        pass_label: e.g. ``"raw_16k"``.
        presence_votes: per-bucket ``{"start", "end", "votes", "frame_instability"?}``
            dicts from ``harvest_presence_votes``.
        identity_votes: per-bucket vote dicts from ``harvest_identity_votes``.
        utterance_votes: per-bucket vote dicts from ``harvest_utterance_votes``.
        quality_by_bucket: presence-grid bucket key → quality score dict (US1 columns).
        source_by_bucket: presence-grid bucket key → source-mass dict (US2 columns).
        grids: axis → ``{"win_length", "hop_length"}`` actually used at harvest.
        provenance_extras: scene_quality / sound_sources / frame_posteriors blocks.
        synthetic_diarization: optional ``{source_id: diar_block}`` synthesized from
            embedding clustering (kept explicit so callers can opt into the legacy
            pass-summary injection instead of being mutated silently).
    """

    pass_label: str
    presence_votes: list[dict[str, Any]] = field(default_factory=list)
    identity_votes: list[dict[str, Any]] = field(default_factory=list)
    utterance_votes: list[dict[str, Any]] = field(default_factory=list)
    quality_by_bucket: dict[tuple[float, float], dict[str, Any]] = field(default_factory=dict)
    source_by_bucket: dict[tuple[float, float], dict[str, Any]] = field(default_factory=dict)
    grids: dict[str, dict[str, float]] = field(default_factory=dict)
    provenance_extras: dict[str, Any] = field(default_factory=dict)
    synthetic_diarization: dict[str, Any] | None = None


def mask_from_pvoice(p: float) -> float:
    """p_voice → mask weight: 1.0 if >= 0.5, else linear ramp to 0 at p = 0."""
    return 1.0 if p >= 0.5 else max(0.0, min(1.0, p / 0.5))


def intensity_mask(start: float, end: float, presence_pv_intervals: list[tuple[float, float, float]]) -> float:
    """Average presence-derived mask over presence buckets overlapping ``[start, end)``.

    Overlap-averaging (not closest-only) so a query bucket spanning a half-voice /
    half-silence region is masked proportionally; works when presence runs on a
    finer grid. Falls back to 1.0 (no masking) when presence produced no p_voice.
    """
    vals = [mask_from_pvoice(p) for s_p, e_p, p in presence_pv_intervals if s_p < end and e_p > start]
    return sum(vals) / len(vals) if vals else 1.0


def aggregate_pass(harvest: PassHarvest, *, aggregator: str, params: dict[str, Any]) -> dict[str, AxisResult]:
    """Fold one pass's harvested votes into the three per-axis ``AxisResult``s.

    Pure: same harvest + same aggregator ⇒ identical rows (bit-for-bit). The math is
    the historical compute.py aggregation, moved verbatim: presence keeps
    ``aggregated_uncertainty = aggregate_presence(votes)`` with the temporal-
    instability OR only on the additive ``presence_uncertainty`` column; identity /
    utterance keep the intensity mask OUT of ``aggregated_uncertainty`` and expose it
    as ``intensity_weight``.
    """
    pass_label = harvest.pass_label
    out: dict[str, AxisResult] = {}

    # ── presence ──
    presence_rows: list[UncertaintyRow] = []
    presence_pv_intervals: list[tuple[float, float, float]] = []
    for bucket in harvest.presence_votes:
        u = aggregate_presence(bucket["votes"])
        p_v = presence_p_voice(bucket["votes"])
        if u is None and not bucket["votes"]:
            continue
        bkey = (round(bucket["start"], 6), round(bucket["end"], 6))
        quality = harvest.quality_by_bucket.get(bkey)
        source = harvest.source_by_bucket.get(bkey)
        votes = bucket["votes"]
        if quality is not None:
            votes = {**votes, "__quality__": quality.get("_raw", {})}
        if source is not None:
            votes = {**votes, "__sources__": source.get("_raw", {})}
        instability = bucket.get("frame_instability")
        if u is None:
            presence_uncertainty: float | None = None
        elif instability is None:
            presence_uncertainty = u
        else:
            presence_uncertainty = max(0.0, min(1.0, 1.0 - (1.0 - u) * (1.0 - float(instability))))
        presence_rows.append(
            UncertaintyRow(
                start=bucket["start"],
                end=bucket["end"],
                axis="presence",
                aggregated_uncertainty=u,
                contributing_models=sorted(k for k in votes if not k.startswith("__")),
                model_votes=votes,
                comparison_status="ok" if u is not None else "incomparable",
                raw_aggregated_uncertainty=u,
                intensity_weight=1.0,
                presence_confidence=float(p_v) if p_v is not None else None,
                presence_uncertainty=presence_uncertainty,
                quality_snr=quality.get("quality_snr") if quality else None,
                quality_clip=quality.get("quality_clip") if quality else None,
                quality_reverb=quality.get("quality_reverb") if quality else None,
                quality_bandwidth=quality.get("quality_bandwidth") if quality else None,
                quality_uncertainty=quality.get("quality_uncertainty") if quality else None,
                src_speech=source.get("src_speech") if source else None,
                src_people=source.get("src_people") if source else None,
                src_machine=source.get("src_machine") if source else None,
                src_environment=source.get("src_environment") if source else None,
                src_dominant=source.get("src_dominant") if source else None,
            )
        )
        if p_v is not None:
            presence_pv_intervals.append((float(bucket["start"]), float(bucket["end"]), float(p_v)))

    pres_grid = harvest.grids.get("presence", {})
    out["presence"] = AxisResult(
        pass_label=pass_label,  # type: ignore[arg-type]
        axis="presence",
        rows=presence_rows,
        provenance={
            "axis": "presence",
            "pass": pass_label,
            "grid": dict(pres_grid),
            "comparator_params": params,
            "contributing_model_set": sorted({m for b in harvest.presence_votes for m in b["votes"]}),
            **{k: v for k, v in harvest.provenance_extras.items()},
        },
    )

    # ── identity ──
    identity_rows: list[UncertaintyRow] = []
    for bucket in harvest.identity_votes:
        u_raw = aggregate_identity(bucket["votes"], raw_vs_enh=None, aggregator=aggregator)
        if u_raw is None and not bucket["votes"]:
            continue
        mask = intensity_mask(bucket["start"], bucket["end"], presence_pv_intervals)
        identity_rows.append(
            UncertaintyRow(
                start=bucket["start"],
                end=bucket["end"],
                axis="identity",
                aggregated_uncertainty=u_raw,
                contributing_models=sorted(bucket["votes"].keys()),
                model_votes=bucket["votes"],
                comparison_status="ok" if u_raw is not None else "incomparable",
                raw_aggregated_uncertainty=u_raw,
                intensity_weight=mask,
            )
        )
    out["identity"] = AxisResult(
        pass_label=pass_label,  # type: ignore[arg-type]
        axis="identity",
        rows=identity_rows,
        provenance={
            "axis": "identity",
            "pass": pass_label,
            "grid": dict(harvest.grids.get("identity", {})),
            "comparator_params": params,
            "contributing_model_set": sorted({m for b in harvest.identity_votes for m in b["votes"]}),
        },
    )

    # ── utterance ──
    utterance_rows: list[UncertaintyRow] = []
    for bucket in harvest.utterance_votes:
        u_raw = aggregate_utterance(bucket["votes"], aggregator=aggregator)
        if u_raw is None and not bucket["votes"]:
            continue
        mask = intensity_mask(bucket["start"], bucket["end"], presence_pv_intervals)
        utterance_rows.append(
            UncertaintyRow(
                start=bucket["start"],
                end=bucket["end"],
                axis="utterance",
                aggregated_uncertainty=u_raw,
                contributing_models=sorted(bucket["votes"].keys()),
                model_votes=bucket["votes"],
                comparison_status="ok" if u_raw is not None else "incomparable",
                raw_aggregated_uncertainty=u_raw,
                intensity_weight=mask,
            )
        )
    out["utterance"] = AxisResult(
        pass_label=pass_label,  # type: ignore[arg-type]
        axis="utterance",
        rows=utterance_rows,
        provenance={
            "axis": "utterance",
            "pass": pass_label,
            "grid": dict(harvest.grids.get("utterance", {})),
            "comparator_params": params,
            "contributing_model_set": sorted({m for b in harvest.utterance_votes for m in b["votes"]}),
        },
    )
    return out


def compute_pass_deltas(
    raw_rows: list[UncertaintyRow],
    enh_rows: list[UncertaintyRow],
    axis: str,
    aggregator: str,
) -> list[UncertaintyRow]:
    """Pair raw and enhanced rows by (start, end) and emit a delta row per shared bucket.

    Moved verbatim from ``compute._compute_raw_vs_enhanced_delta`` (pure). The delta
    row's ``aggregated_uncertainty`` is |raw − enhanced| clipped to [0, 1]; buckets in
    one pass only → ``comparison_status="one_sided"`` with ``None`` uncertainty.
    """
    raw_by_bucket = {(r.start, r.end): r for r in raw_rows}
    enh_by_bucket = {(r.start, r.end): r for r in enh_rows}
    bucket_keys = sorted(set(raw_by_bucket) | set(enh_by_bucket))
    out: list[UncertaintyRow] = []
    for key in bucket_keys:
        raw_row = raw_by_bucket.get(key)
        enh_row = enh_by_bucket.get(key)
        votes: dict[str, dict[str, Any]] = {}
        if raw_row is not None:
            for m, v in raw_row.model_votes.items():
                votes[f"raw_16k::{m}"] = v
        if enh_row is not None:
            for m, v in enh_row.model_votes.items():
                votes[f"enhanced_16k::{m}"] = v

        if raw_row is None or enh_row is None:
            present = raw_row if raw_row is not None else enh_row
            iw = present.intensity_weight if present and present.intensity_weight is not None else None
            ra_raw = raw_row.raw_aggregated_uncertainty if raw_row else None
            enh_raw = enh_row.raw_aggregated_uncertainty if enh_row else None
            out.append(
                UncertaintyRow(
                    start=key[0],
                    end=key[1],
                    axis=axis,  # type: ignore[arg-type]
                    aggregated_uncertainty=None,
                    contributing_models=sorted(votes.keys()),
                    model_votes=votes,
                    comparison_status="one_sided",
                    raw_aggregated_uncertainty=ra_raw if ra_raw is not None else enh_raw,
                    intensity_weight=iw,
                )
            )
            continue
        ra = raw_row.aggregated_uncertainty
        ea = enh_row.aggregated_uncertainty
        if ra is None or ea is None:
            delta = None
            status = "incomparable"
        else:
            delta = max(0.0, min(1.0, abs(ra - ea)))
            status = "ok"
        raw_iw = raw_row.intensity_weight if raw_row.intensity_weight is not None else 1.0
        enh_iw = enh_row.intensity_weight if enh_row.intensity_weight is not None else 1.0
        out.append(
            UncertaintyRow(
                start=key[0],
                end=key[1],
                axis=axis,  # type: ignore[arg-type]
                aggregated_uncertainty=delta,
                contributing_models=sorted(votes.keys()),
                model_votes=votes,
                comparison_status=status,  # type: ignore[arg-type]
                raw_aggregated_uncertainty=delta,
                intensity_weight=max(raw_iw, enh_iw),
            )
        )
    return out
