"""Top-level ranked index over the fused axes — ``disagreements.json``.

Ranks over ``L2/round<N>/uncertainty/<axis>.parquet``, on ``triage_score`` — the column that
exists for exactly this question ("where should budget be spent?"). There is no ``pass`` field:
an axis is a fold across passes, so a disagreement belongs to a span of the recording, not to one
transform of it. Per-signal detail comes from the L1 signal rows, joined on ``(bucket, signal)``.
"""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.audio_analysis.labelstudio import HIGH_THRESHOLD
from senselab.audio.workflows.audio_analysis.types import FusedAxis

_AXIS_PRIORITY: dict[str, int] = {"asr": 0, "speaker": 1, "speech_presence": 2}


def _row_summary(row: Mapping[str, Any], axis: str, evidence: Mapping[str, Any] | None) -> str:
    """One-line human-readable explanation of why a bucket scored high.

    The fused row says *how much* doubt there is and which signals carried it; ``evidence`` is the
    L1 per-signal measurement for the same bucket, which says *what* they measured. Keeping them
    separate is the point — the summary reads a measurement, never a second fold.
    """
    weights = row.get("signal_weights") or {}
    signals = list(row.get("contributing_signals") or [])
    parts = [f"signals={signals!r}"]
    if weights:
        parts.append(f"weights={ {k: round(float(v), 3) for k, v in sorted(weights.items())}!r}")
    if axis == "speech_presence":
        for field, fmt in (("snr_brouhaha_db", "snr={:.1f}dB"), ("quality_snr", "quality_snr={:.2f}")):
            value = row.get(field)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                parts.append(fmt.format(float(value)))
        if isinstance(row.get("src_dominant"), str):
            parts.append(f"src={row['src_dominant']}")
    if axis == "asr" and isinstance(row.get("scene_quality_coupling"), (int, float)):
        parts.append(f"scene_coupling={round(float(row['scene_quality_coupling']), 3)}")
    if evidence:
        parts.append(f"evidence={ {k: v for k, v in sorted(evidence.items())}!r}")
    return " ".join(parts)


def _evidence_index(
    signal_results_by_pass: Mapping[str, Mapping[str, Any]] | None,
) -> dict[tuple[float, float], dict[str, Any]]:
    """``{(start, end) → {"<pass>::<signal>": measurement}}`` from the L1 signal rows."""
    out: dict[tuple[float, float], dict[str, Any]] = {}
    for perturbation, by_signal in sorted((signal_results_by_pass or {}).items()):
        for signal, result in sorted(by_signal.items()):
            for row in getattr(result, "rows", []):
                key = (round(float(row.start), 6), round(float(row.end), 6))
                out.setdefault(key, {})[f"{perturbation}::{signal}"] = row.measurement
    return out


def build_disagreements_index(
    *,
    fused_axes: Mapping[str, FusedAxis],
    top_n: int,
    run_dir: Path,
    config: dict[str, Any],
    incomparable_reasons: dict[str, str],
    models_without_native_signal: list[str] | None = None,
    signal_results_by_pass: Mapping[str, Mapping[str, Any]] | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Build the ``disagreements.json`` payload.

    Ranks by ``triage_score`` desc, with axis-priority tiebreak (asr > speaker >
    speech_presence) and start-time secondary tiebreak. Truncated to ``top_n``; ``top_n=0``
    returns an empty entries list (caller should skip writing the file).
    """
    rows_by_axis: dict[str, int] = {"speech_presence": 0, "speaker": 0, "asr": 0}
    total_rows = 0
    high_count = 0
    evidence = _evidence_index(signal_results_by_pass)

    candidates: list[dict[str, Any]] = []
    for axis_raw, result in sorted(fused_axes.items()):
        axis = str(axis_raw)
        rows: Sequence[Mapping[str, Any]] = result.rows
        rows_by_axis[axis] = rows_by_axis.get(axis, 0) + len(rows)
        total_rows += len(rows)
        for row_idx, row in enumerate(rows):
            triage = row.get("triage_score")
            if isinstance(triage, (int, float)) and not math.isnan(float(triage)) and triage >= HIGH_THRESHOLD:
                high_count += 1
            bucket = (round(float(row["start"]), 6), round(float(row["end"]), 6))
            entry = {
                "axis": axis,
                "start": float(row["start"]),
                "end": float(row["end"]),
                "triage_score": triage,
                "uncertainty": row.get("uncertainty"),
                "epistemic_uncertainty": row.get("epistemic_uncertainty"),
                "contributing_signals": list(row.get("contributing_signals") or []),
                "contributing_passes": list(row.get("contributing_passes") or []),
                "signal_weights": dict(row.get("signal_weights") or {}),
                "weight_basis": dict(row.get("weight_basis") or {}),
                "round": row.get("round"),
                "parquet": _parquet_path_for(axis, row.get("round") if round_index is None else round_index),
                "row_idx": row_idx,
                "ls_region_id": f"{_track_name(axis)}__{row_idx}",
                "summary": _row_summary(row, axis, evidence.get(bucket)),
            }
            candidates.append(entry)

    def _sort_key(e: dict[str, Any]) -> tuple[Any, ...]:
        score = e["triage_score"]
        primary = -float(score) if isinstance(score, (int, float)) and not math.isnan(float(score)) else float("inf")
        return (primary, _AXIS_PRIORITY.get(e["axis"], 99), e["start"])

    candidates.sort(key=_sort_key)
    selected = candidates[: max(0, top_n)] if top_n > 0 else []
    for rank, entry in enumerate(selected, start=1):
        entry["rank"] = rank

    return {
        "schema_version": 2,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "wrapper_hash": config.get("wrapper_hash", ""),
        "senselab_version": config.get("senselab_version", ""),
        "config": {
            k: config[k]
            for k in (
                "top_n",
                "aggregator",
                "phoneme_disagreement_threshold",
                "bucket_grid",
                "speech_presence_labels",
            )
            if k in config
        },
        "models_without_native_signal": list(models_without_native_signal or []),
        "incomparable_reasons": dict(incomparable_reasons),
        "totals": {
            "total_rows": total_rows,
            "rows_by_axis": rows_by_axis,
            "high_uncertainty_rows": high_count,
            "high_uncertainty_rate": (high_count / total_rows) if total_rows else 0.0,
        },
        "entries": selected,
    }


def _parquet_path_for(axis: str, round_index: Any) -> str:  # noqa: ANN401
    """Path of the parquet (relative to run_dir) that holds this axis's fused rows."""
    n = int(round_index) if isinstance(round_index, (int, float)) else 0
    return f"L2/round{n}/uncertainty/{axis}.parquet"


def _track_name(axis: str) -> str:
    """Label Studio track carrying this axis. No pass token: an axis has no pass."""
    return f"uncertainty__{axis}"
