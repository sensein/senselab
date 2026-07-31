"""Top-level ranked index over the 9 axis parquets — ``disagreements.json``."""

from __future__ import annotations

import datetime as _dt
import math
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.labelstudio import HIGH_THRESHOLD
from senselab.audio.workflows.audio_analysis.types import AxisResult

_AXIS_PRIORITY: dict[str, int] = {"utterance": 0, "identity": 1, "presence": 2}


def _row_summary(row: Any, axis: str) -> str:  # noqa: ANN401
    """One-line human-readable explanation of why a row scored high."""
    if axis == "presence":
        speaks = [m for m, v in row.model_votes.items() if v.get("speaks")]
        silent = [m for m, v in row.model_votes.items() if v.get("speaks") is False]
        summary = f"speaks={speaks!r} silent={silent!r}"
        # FR-024 (T042): surface the scene sub-signals when present.
        extras = []
        if getattr(row, "presence_uncertainty", None) is not None:
            extras.append(f"presence_unc={round(float(row.presence_uncertainty), 3)}")
        if getattr(row, "quality_uncertainty", None) is not None:
            extras.append(f"quality_unc={round(float(row.quality_uncertainty), 3)}")
        if getattr(row, "src_dominant", None) is not None:
            extras.append(f"src={row.src_dominant}")
        return summary + (" " + " ".join(extras) if extras else "")
    if axis == "identity":
        labels = {
            m: v.get("speaker_label")
            for m, v in row.model_votes.items()
            if isinstance(v, dict) and "speaker_label" in v
        }
        same_unc = {
            m: round(float(v["same_label_uncertainty"]), 3)
            for m, v in row.model_votes.items()
            if isinstance(v, dict) and v.get("same_label_uncertainty") is not None
        }
        change_unc = {
            m: round(float(v["change_inconsistency_uncertainty"]), 3)
            for m, v in row.model_votes.items()
            if isinstance(v, dict) and v.get("change_inconsistency_uncertainty") is not None
        }
        cross_block = row.model_votes.get("__cross_diar_label_disagreement__")
        cross_str = ""
        if isinstance(cross_block, dict) and cross_block.get("value") is not None:
            cross_str = f" cross_diar={round(float(cross_block['value']), 3)}"
        parts = [f"labels={labels!r}"]
        if same_unc:
            parts.append(f"same_unc={same_unc!r}")
        if change_unc:
            parts.append(f"change_unc={change_unc!r}")
        return " ".join(parts) + cross_str
    if axis == "utterance":
        texts = {m: (v.get("text") or "")[:40] for m, v in row.model_votes.items() if v.get("text")}
        return f"transcripts={texts!r}"
    return ""


def build_disagreements_index(
    *,
    axis_results: dict[tuple[Any, Any], AxisResult],
    top_n: int,
    run_dir: Path,
    config: dict[str, Any],
    incomparable_reasons: dict[str, str],
    models_without_native_signal: list[str] | None = None,
) -> dict[str, Any]:
    """Build the ``disagreements.json`` payload per ``contracts/disagreements.json.md``.

    Ranks by ``within_pass_uncertainty`` desc, with axis-priority tiebreak (utterance >
    identity > presence) and start-time secondary tiebreak. Truncated to ``top_n``;
    ``top_n=0`` returns an empty entries list (caller should skip writing the file).
    """
    rows_by_axis: dict[str, int] = {"presence": 0, "identity": 0, "utterance": 0}
    rows_by_pass: dict[str, int] = {"raw_16k": 0, "enhanced_16k": 0, "raw_vs_enhanced": 0}
    total_rows = 0
    high_count = 0

    candidates: list[dict[str, Any]] = []
    for (pass_label_raw, axis_raw), result in axis_results.items():
        pass_label = str(pass_label_raw)
        axis = str(axis_raw)
        rows_by_axis[axis] = rows_by_axis.get(axis, 0) + len(result.rows)
        rows_by_pass[pass_label] = rows_by_pass.get(pass_label, 0) + len(result.rows)
        total_rows += len(result.rows)
        for row_idx, row in enumerate(result.rows):
            au = row.within_pass_uncertainty
            if au is not None and not math.isnan(au) and au >= HIGH_THRESHOLD:
                high_count += 1
            entry = {
                "axis": axis,
                "pass": pass_label,
                "start": float(row.start),
                "end": float(row.end),
                "within_pass_uncertainty": au,
                "contributing_models": list(row.contributing_models),
                "parquet": _parquet_path_for(pass_label, axis),
                "row_idx": row_idx,
                "ls_region_id": f"{_track_name(pass_label, axis)}__{row_idx}",
                "summary": _row_summary(row, axis),
            }
            # FR-024 (T042): presence entries carry the scene sub-signals so the
            # index is filterable/rankable on them (null-safe: omitted when absent).
            if axis == "presence":
                for field in ("presence_uncertainty", "quality_uncertainty", "src_dominant"):
                    value = getattr(row, field, None)
                    if value is not None and not (isinstance(value, float) and math.isnan(value)):
                        entry[field] = value
            candidates.append(entry)

    # Sort: NaN / None last. Primary descending by within_pass_uncertainty; axis
    # priority tiebreak; then (FR-024/T042) the enriched presence sub-signal —
    # among equal-aggregated presence rows, higher presence_uncertainty (the
    # decisiveness + temporal-instability composite) ranks first. Rows where
    # aggregated differs are ordered exactly as before (SC-008: baseline
    # within_pass_uncertainty values and their relative order are unchanged).
    def _sort_key(e: dict[str, Any]) -> tuple[Any, ...]:
        au = e["within_pass_uncertainty"]
        primary = -float(au) if au is not None and not (isinstance(au, float) and math.isnan(au)) else float("inf")
        pu = e.get("presence_uncertainty")
        sub_signal = -float(pu) if isinstance(pu, (int, float)) and not math.isnan(float(pu)) else 0.0
        return (primary, _AXIS_PRIORITY.get(e["axis"], 99), sub_signal, e["start"])

    candidates.sort(key=_sort_key)
    selected = candidates[: max(0, top_n)] if top_n > 0 else []
    for rank, entry in enumerate(selected, start=1):
        entry["rank"] = rank

    return {
        "schema_version": 1,
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
            "rows_by_pass": rows_by_pass,
            "high_uncertainty_rows": high_count,
            "high_uncertainty_rate": (high_count / total_rows) if total_rows else 0.0,
        },
        "entries": selected,
    }


def _parquet_path_for(pass_label: str, axis: str) -> str:
    """Path of the parquet (relative to run_dir) that holds ``(pass_label, axis)`` rows."""
    if pass_label == "raw_vs_enhanced":
        return f"uncertainty/raw_vs_enhanced/{axis}.parquet"
    return f"{pass_label}/uncertainty/{axis}.parquet"


def _track_name(pass_label: str, axis: str) -> str:
    pass_token = "pass_pair" if pass_label == "raw_vs_enhanced" else pass_label
    return f"{pass_token}__uncertainty__{axis}"
