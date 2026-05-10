"""Label Studio bundle integration for the three uncertainty axes.

Per FR-005 the bundle exposes:
    - 6 Labels tracks per pass (3 axes × 2 passes), named ``<pass>__uncertainty__<axis>``.
    - 3 raw_vs_enhanced delta tracks named ``pass_pair__uncertainty__<axis>``.
    - 3 utterance TextArea sibling tracks (one per pass + one for pass_pair), named
      ``<pass>__uncertainty__utterance__text``, carrying the per-bucket transcript
      consensus + dissenting model transcripts.
"""

from __future__ import annotations

import json
import re
from typing import Any

from senselab.audio.workflows.audio_analysis.types import AxisResult, ComparisonStatus

LABEL_VALUES = ("low", "medium", "high", "incomparable", "unavailable")
LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66


def uncertainty_to_label_bin(value: float | None, status: ComparisonStatus | str) -> str:
    """Map ``aggregated_uncertainty`` to one of the LS label values per FR-005."""
    if status in ("incomparable", "unavailable", "one_sided"):
        return "unavailable" if status == "unavailable" else "incomparable"
    if value is None:
        return "incomparable"
    if value < LOW_THRESHOLD:
        return "low"
    if value < HIGH_THRESHOLD:
        return "medium"
    return "high"


def _safe(model_id: str) -> str:
    """Sanitize a model id for use in a track / region name."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", model_id).strip("_") or "model"


def _track_name(pass_label: str, axis: str) -> str:
    pass_token = "pass_pair" if pass_label == "raw_vs_enhanced" else pass_label
    return f"{pass_token}__uncertainty__{axis}"


def _build_labels_xml(track_name: str) -> str:
    inner = "\n".join(f'  <Label value="{v}"/>' for v in LABEL_VALUES)
    return f'<Labels name="{track_name}" toName="audio">\n{inner}\n</Labels>'


def _build_textarea_xml(track_name: str) -> str:
    return (
        f'<TextArea name="{track_name}__text" toName="audio" perRegion="true" '
        f'editable="false" placeholder="Per-bucket transcript consensus + dissenting models"/>'
    )


def _utterance_text_payload(model_votes: dict[str, dict[str, Any]]) -> str:
    """Build the consensus + dissenting-models string for the utterance TextArea."""
    transcripts = [
        (m, str(v.get("text") or "").strip()) for m, v in model_votes.items() if str(v.get("text") or "").strip()
    ]
    if not transcripts:
        return "(no transcripts on this bucket)"
    # Plurality consensus.
    counts: dict[str, int] = {}
    for _, t in transcripts:
        counts[t] = counts.get(t, 0) + 1
    consensus = max(counts.items(), key=lambda kv: kv[1])[0]
    lines = [f"consensus: {consensus!r}"]
    for model_id, t in transcripts:
        lines.append(f"{model_id}: {t!r}")
    return "\n".join(lines)


def attach_uncertainty_tracks_to_ls(
    *,
    ls_tasks: Any,  # noqa: ANN401 — list[dict] or dict, matches build_labelstudio_task variants
    ls_config: str,
    axis_results: dict[tuple[Any, Any], AxisResult],
) -> tuple[Any, str]:
    """Append uncertainty Labels + TextArea tracks to the LS config and tasks payloads.

    Args:
        ls_tasks: Existing LS tasks payload (single dict or list of dicts) — typically
            produced by ``scripts/analyze_audio.py``'s ``build_labelstudio_task``.
        ls_config: Existing LS config XML string.
        axis_results: ``{(pass_label, axis) → AxisResult}`` from ``compute_uncertainty_axes``.

    Returns:
        Updated ``(ls_tasks, ls_config)``.
    """
    # ── Build the new XML blocks ──
    blocks: list[str] = []
    for (pass_label, axis), _result in axis_results.items():
        track = _track_name(str(pass_label), str(axis))
        blocks.append(_build_labels_xml(track))
        if axis == "utterance":
            blocks.append(_build_textarea_xml(track))

    # Inject before the closing </View> tag.
    if "</View>" in ls_config:
        ls_config = ls_config.replace("</View>", "\n".join(blocks) + "\n</View>", 1)
    else:
        ls_config = ls_config + "\n" + "\n".join(blocks)

    # ── Build per-row LS regions and attach to the matching task ──
    tasks_list = ls_tasks if isinstance(ls_tasks, list) else [ls_tasks]
    by_pass_task: dict[str, dict[str, Any]] = {}
    for t in tasks_list:
        pass_label = (t.get("data") or {}).get("pass") or "raw_16k"
        by_pass_task[pass_label] = t

    # raw_vs_enhanced regions ride on the raw_16k task by convention.
    fallback_task = by_pass_task.get("raw_16k") or (tasks_list[0] if tasks_list else None)

    for (pass_label, axis), result in axis_results.items():
        pass_label = str(pass_label)
        axis = str(axis)
        track = _track_name(pass_label, axis)
        target_task = by_pass_task.get(pass_label) or fallback_task
        if target_task is None or not target_task.get("predictions"):
            continue
        result_list = target_task["predictions"][0].setdefault("result", [])
        for row_idx, row in enumerate(result.rows):
            label_value = uncertainty_to_label_bin(row.aggregated_uncertainty, row.comparison_status)
            region_id = f"{track}__{row_idx}"
            result_list.append(
                {
                    "id": region_id,
                    "from_name": track,
                    "to_name": "audio",
                    "type": "labels",
                    "value": {
                        "start": float(row.start),
                        "end": float(row.end),
                        "labels": [label_value],
                    },
                }
            )
            if axis == "utterance":
                result_list.append(
                    {
                        "id": f"{region_id}__text",
                        "from_name": f"{track}__text",
                        "to_name": "audio",
                        "type": "textarea",
                        "value": {
                            "start": float(row.start),
                            "end": float(row.end),
                            "text": [_utterance_text_payload(row.model_votes)],
                        },
                    }
                )

    return ls_tasks, ls_config
