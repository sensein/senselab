"""Label Studio bundle integration for the three uncertainty axes.

Per FR-005 the bundle exposes:
    - 6 Labels tracks per pass (3 axes × 2 passes), named ``<pass>__uncertainty__<axis>``.
    - 3 raw_vs_enhanced delta tracks named ``pass_pair__uncertainty__<axis>``.
    - 3 asr TextArea sibling tracks (one per pass + one for pass_pair), named
      ``<pass>__uncertainty__asr__text``, carrying the per-bucket transcript
      consensus + dissenting model transcripts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.audio_analysis.harvesters import (
    asr_has_timestamps,
    seg_attr,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_window_top1 as _classification_window_top1,
)
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_windows as _classification_windows,
)
from senselab.audio.workflows.audio_analysis.types import AxisResult, ComparisonStatus
from senselab.utils.data_structures import safe_model_id


def load_ls_ground_truth(path: Path) -> dict[str, Any]:
    """Parse a Label Studio export into ``{segments: [{start, end, speaker, text|None}], duration}``.

    The import-side counterpart of this module's export builders (moved here
    from the adaptive workflow — architecture-review T049): reads the standard
    LS JSON export (list of tasks; paired ``labels``/``textarea`` results
    sharing region ids) and yields time-ordered speaker segments with optional
    transcripts — the ground-truth shape consumed by the adaptive loop's
    evaluation harness.
    """
    tasks = json.loads(Path(path).read_text())
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"unexpected LS export shape in {path}")
    ann = (tasks[0].get("annotations") or [{}])[0]
    by_id: dict[str, dict[str, Any]] = {}
    duration = 0.0
    for item in ann.get("result") or []:
        rid = item.get("id")
        val = item.get("value") or {}
        duration = max(duration, float(item.get("original_length") or 0.0))
        seg = by_id.setdefault(rid, {})
        if item.get("type") == "labels":
            seg.update(
                {
                    "start": float(val["start"]),
                    "end": float(val["end"]),
                    "speaker": (val.get("labels") or [None])[0],
                }
            )
        elif item.get("type") == "textarea":
            seg["text"] = " ".join(val.get("text") or []) or None
            seg.setdefault("start", float(val["start"]))
            seg.setdefault("end", float(val["end"]))
    segments = sorted((s for s in by_id.values() if "start" in s), key=lambda s: s["start"])
    for s in segments:
        s.setdefault("text", None)
        s.setdefault("speaker", None)
    return {"segments": segments, "duration": duration}


LABEL_VALUES = ("low", "medium", "high", "incomparable", "unavailable")
LOW_THRESHOLD = 0.33
HIGH_THRESHOLD = 0.66


def uncertainty_to_label_bin(value: float | None, status: ComparisonStatus | str) -> str:
    """Map ``within_pass_uncertainty`` to one of the LS label values per FR-005."""
    if status in ("incomparable", "unavailable", "one_sided"):
        return "unavailable" if status == "unavailable" else "incomparable"
    if value is None:
        return "incomparable"
    if value < LOW_THRESHOLD:
        return "low"
    if value < HIGH_THRESHOLD:
        return "medium"
    return "high"


def _track_name(pass_label: str, axis: str) -> str:
    pass_token = "pass_pair" if pass_label == "raw_vs_enhanced" else pass_label
    return f"{pass_token}__uncertainty__{axis}"


SOURCE_LABEL_VALUES = ("speech", "people", "machine", "environment", "unavailable")


def _build_labels_xml(track_name: str) -> str:
    inner = "\n".join(f'  <Label value="{v}"/>' for v in LABEL_VALUES)
    return f'<Labels name="{track_name}" toName="audio">\n{inner}\n</Labels>'


def _build_source_labels_xml(track_name: str) -> str:
    inner = "\n".join(f'  <Label value="{v}"/>' for v in SOURCE_LABEL_VALUES)
    return f'<Labels name="{track_name}" toName="audio">\n{inner}\n</Labels>'


def _scene_track_name(pass_label: str, kind: str) -> str:
    """FR-024 scene tracks: ``<pass>__speech_presence__quality`` / ``<pass>__speech_presence__sources``."""
    pass_token = "pass_pair" if pass_label == "raw_vs_enhanced" else pass_label
    return f"{pass_token}__speech_presence__{kind}"


def _quality_degradation(row: Any) -> float | None:  # noqa: ANN401 — UncertaintyRow duck-typed
    """Overall degradation for the quality track: max over the four quality columns."""
    values = [
        v
        for v in (row.quality_snr, row.quality_clip, row.quality_reverb, row.quality_bandwidth)
        if v is not None and not (isinstance(v, float) and v != v)
    ]
    return max(values) if values else None


def _build_textarea_xml(track_name: str) -> str:
    return (
        f'<TextArea name="{track_name}__text" toName="audio" perRegion="true" '
        f'editable="false" placeholder="Per-bucket transcript consensus + dissenting models"/>'
    )


def _utterance_text_payload(model_votes: dict[str, dict[str, Any]]) -> str:
    """Build the consensus + dissenting-models string for the asr TextArea."""
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
    for (pass_label, axis), result in axis_results.items():
        track = _track_name(str(pass_label), str(axis))
        blocks.append(_build_labels_xml(track))
        if axis == "asr":
            blocks.append(_build_textarea_xml(track))
        # FR-024 (T040): additive scene tracks on per-pass speech_presence results —
        # emitted only when the pass actually carries the corresponding columns
        # (delta rows never do), so legacy bundles are byte-identical.
        if str(axis) == "speech_presence" and str(pass_label) != "raw_vs_enhanced":
            if any(_quality_degradation(r) is not None for r in result.rows):
                blocks.append(_build_labels_xml(_scene_track_name(str(pass_label), "quality")))
            if any(r.src_dominant is not None for r in result.rows):
                blocks.append(_build_source_labels_xml(_scene_track_name(str(pass_label), "sources")))

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
            label_value = uncertainty_to_label_bin(row.within_pass_uncertainty, row.comparison_status)
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
            if axis == "asr":
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
            # FR-024 (T040): scene tracks ride the same speech_presence rows.
            if axis == "speech_presence" and pass_label != "raw_vs_enhanced":
                degradation = _quality_degradation(row)
                if degradation is not None:
                    q_track = _scene_track_name(pass_label, "quality")
                    result_list.append(
                        {
                            "id": f"{q_track}__{row_idx}",
                            "from_name": q_track,
                            "to_name": "audio",
                            "type": "labels",
                            "value": {
                                "start": float(row.start),
                                "end": float(row.end),
                                "labels": [uncertainty_to_label_bin(degradation, "ok")],
                            },
                        }
                    )
                if row.src_dominant is not None:
                    s_track = _scene_track_name(pass_label, "sources")
                    label = str(row.src_dominant)
                    result_list.append(
                        {
                            "id": f"{s_track}__{row_idx}",
                            "from_name": s_track,
                            "to_name": "audio",
                            "type": "labels",
                            "value": {
                                "start": float(row.start),
                                "end": float(row.end),
                                "labels": [label if label in SOURCE_LABEL_VALUES else "unavailable"],
                            },
                        }
                    )

    return ls_tasks, ls_config


# ── Per-task export builders (moved from scripts/analyze_audio.py, T051b) ──
#     These turn one analyze_audio run into a Label Studio task + config. They
#     were library-grade code sitting in the CLI; the uncertainty-track builders
#     above already lived here, so the whole LS surface is now in one module.


def _new_region_id(prefix: str, idx: int) -> str:
    """Stable per-region ID for Label Studio result entries."""
    return f"{prefix}_{idx:04d}"


def _ls_label_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    label: str,
    score: float | None = None,
) -> dict[str, Any]:
    """Build one Label Studio ``labels`` result entry on the audio timeline."""
    value: dict[str, Any] = {"start": float(start), "end": float(end), "labels": [label]}
    entry: dict[str, Any] = {
        "id": region_id,
        "from_name": from_name,
        "to_name": "audio",
        "type": "labels",
        "value": value,
    }
    if score is not None:
        entry["score"] = float(score)
    return entry


def _ls_textarea_region(
    *,
    region_id: str,
    from_name: str,
    start: float,
    end: float,
    text: str,
) -> dict[str, Any]:
    """Build one Label Studio ``textarea`` per-region transcription entry."""
    return {
        "id": region_id,
        "from_name": from_name,
        "to_name": "audio",
        "type": "textarea",
        "value": {"start": float(start), "end": float(end), "text": [text]},
    }


def _diarization_to_ls(result: Any, prefix: str) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert diarize_audios output (List[List[ScriptLine]]) into LS regions."""
    out: list[dict[str, Any]] = []
    if not result:
        return out
    segments = result[0] if isinstance(result, list) and result else []
    for i, seg in enumerate(segments):
        start = seg_attr(seg, "start")
        end = seg_attr(seg, "end")
        speaker = seg_attr(seg, "speaker") or "SPEAKER_UNKNOWN"
        if start is None or end is None:
            continue
        out.append(
            _ls_label_region(
                region_id=_new_region_id(f"{prefix}_dia", i),
                from_name=prefix,
                start=start,
                end=end,
                label=str(speaker),
            )
        )
    return out


def _classification_to_ls(
    result: Any,  # noqa: ANN401
    prefix: str,
    win_length: float,
    hop_length: float,
) -> list[dict[str, Any]]:
    """Convert classify_audios windowed output into LS regions (top-1 per window).

    Window centers advance by ``hop_length`` and span ``win_length`` seconds, so
    the LS regions reflect each model's own native frame stride.
    """
    out: list[dict[str, Any]] = []
    windows = _classification_windows(result)
    for i, window in enumerate(windows):
        label, score, _entropy = _classification_window_top1(window)
        if label is None:
            continue
        # Prefer per-window start/end when the canonical shape carries them.
        if isinstance(window, dict) and window.get("start") is not None and window.get("end") is not None:
            start = float(window["start"])
            end = float(window["end"])
        else:
            start = i * hop_length
            end = start + win_length
        out.append(
            _ls_label_region(
                region_id=_new_region_id(f"{prefix}_cls", i),
                from_name=prefix,
                start=start,
                end=end,
                label=label,
                score=score if score is not None else 0.0,
            )
        )
    return out


def _asr_to_ls(result: Any, prefix: str, full_duration: float) -> list[dict[str, Any]]:  # noqa: ANN401
    """Convert transcribe_audios output into LS textarea regions, one per ScriptLine.

    Whisper sometimes returns one ScriptLine without timing for a short clip; in
    that case we pin the textarea to the full audio span.
    """
    out: list[dict[str, Any]] = []
    if not result:
        return out
    lines = result if isinstance(result, list) else [result]
    for i, line in enumerate(lines):
        text = seg_attr(line, "text") or ""
        start = seg_attr(line, "start")
        end = seg_attr(line, "end")
        if start is None or end is None:
            start, end = 0.0, full_duration
        if not text:
            continue
        out.append(
            _ls_textarea_region(
                region_id=_new_region_id(f"{prefix}_asr", i),
                from_name=prefix,
                start=start,
                end=end,
                text=text,
            )
        )
    return out


def build_labelstudio_task(
    audio_uri: str,
    pass_label: str,
    duration_s: float,
    pass_summary: dict[str, Any],
    ast_win_length: float,
    ast_hop_length: float,
    yamnet_win_length: float,
    yamnet_hop_length: float,
) -> dict[str, Any]:
    """Build one Label Studio task with predictions for all analyzers in this pass.

    Each analyzer (diarization, ast, yamnet, asr) becomes its own
    ``from_name`` track on the audio timeline, so the importer sees parallel
    annotation rows. When a senselab task was run with multiple models, every
    model's output is exported as its own track (e.g., ``asr_whisper_turbo``,
    ``asr_whisper_small``) so they can be visually compared.
    """
    regions: list[dict[str, Any]] = []

    dia = pass_summary.get("diarization", {})
    for model_id, model_block in (dia.get("by_model") or {}).items():
        if model_block.get("status") == "ok":
            from_name = f"{pass_label}__diarization__{safe_model_id(model_id)}"
            regions.extend(_diarization_to_ls(model_block.get("result"), from_name))

    ast_block = pass_summary.get("ast", {})
    if ast_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                ast_block.get("result"),
                f"{pass_label}__ast",
                win_length=ast_win_length,
                hop_length=ast_hop_length,
            )
        )

    yam_block = pass_summary.get("yamnet", {})
    if yam_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                yam_block.get("result"),
                f"{pass_label}__yamnet",
                win_length=yamnet_win_length,
                hop_length=yamnet_hop_length,
            )
        )

    asr = pass_summary.get("asr", {})
    alignment = pass_summary.get("alignment") or {}
    align_by_model = alignment.get("by_model") or {}
    for model_id, model_block in (asr.get("by_model") or {}).items():
        if model_block.get("status") != "ok":
            continue
        from_name = f"{pass_label}__asr__{safe_model_id(model_id)}"
        # Three-case branch:
        # (a) ASR with native timestamps  -> use the ASR result for per-segment regions.
        # (b) ASR text-only + successful alignment -> use the alignment result.
        # (c) ASR text-only + alignment skipped or failed -> single full-audio TextArea.
        align_block = align_by_model.get(model_id) or {}
        asr_result = model_block.get("result")
        if asr_has_timestamps(asr_result):
            regions.extend(_asr_to_ls(asr_result, from_name, duration_s))
        elif align_block.get("status") == "ok":
            # align_transcriptions returns List[List[ScriptLine | None]] —
            # one inner list per input audio. We always pass a single audio,
            # so unwrap to the inner segment list.
            ar = align_block.get("result")
            inner = ar[0] if isinstance(ar, list) and ar and isinstance(ar[0], list) else ar
            regions.extend(_asr_to_ls(inner, from_name, duration_s))
        else:
            regions.extend(_asr_to_ls(asr_result, from_name, duration_s))

    return {
        "data": {
            "audio": audio_uri,
            "pass": pass_label,
            "duration_s": duration_s,
        },
        "predictions": [
            {
                "model_version": f"senselab-analyze:{pass_label}",
                "score": 1.0,
                "result": regions,
            }
        ],
    }


def build_labelstudio_config(summary: dict[str, Any]) -> str:
    """Build a Label Studio labeling-config XML matching this run's per-task tracks.

    Generates one ``<Labels>`` control per (pass, analyzer, model) and one
    ``<TextArea>`` control per (pass, asr_model). Speakers, scene labels,
    and transcripts each become a stacked timeline annotation row.

    The three-axis uncertainty tracks are appended downstream by
    ``senselab.audio.workflows.audio_analysis.attach_uncertainty_tracks_to_ls``.
    """
    parts: list[str] = ["<View>", '  <Audio name="audio" value="$audio"/>']
    seen_label_sets: dict[str, list[str]] = {}

    for pass_label, pass_summary in summary.get("passes", {}).items():
        # Diarization tracks: one per model, with that model's discovered speaker labels
        dia_by_model = (pass_summary.get("diarization") or {}).get("by_model") or {}
        for model_id, model_block in dia_by_model.items():
            if model_block.get("status") != "ok":
                continue
            speakers = sorted({str(getattr(seg, "speaker", "?")) for seg in (model_block.get("result", [[]])[0] or [])})
            if not speakers:
                speakers = ["SPEAKER_00", "SPEAKER_01"]
            seen_label_sets[f"{pass_label}__diarization__{safe_model_id(model_id)}"] = speakers

        # AST scene labels
        ast = pass_summary.get("ast") or {}
        if ast.get("status") == "ok":
            labels = _collect_classification_labels(ast.get("result"))
            if labels:
                seen_label_sets[f"{pass_label}__ast"] = sorted(labels)

        # YAMNet scene labels
        yam = pass_summary.get("yamnet") or {}
        if yam.get("status") == "ok":
            labels = _collect_classification_labels(yam.get("result"))
            if labels:
                seen_label_sets[f"{pass_label}__yamnet"] = sorted(labels)

        # ASR: each model gets its own TextArea
        asr_by_model = (pass_summary.get("asr") or {}).get("by_model") or {}
        for model_id, model_block in asr_by_model.items():
            if model_block.get("status") != "ok":
                continue
            from_name = f"{pass_label}__asr__{safe_model_id(model_id)}"
            parts.append(
                f'  <TextArea name="{from_name}" toName="audio" perRegion="true" '
                f'editable="true" placeholder="ASR transcript ({model_id})"/>'
            )

    for from_name, label_values in sorted(seen_label_sets.items()):
        parts.append(f'  <Labels name="{from_name}" toName="audio">')
        for v in label_values:
            v_escaped = v.replace('"', "&quot;")
            parts.append(f'    <Label value="{v_escaped}"/>')
        parts.append("  </Labels>")

    parts.append("</View>")
    return "\n".join(parts) + "\n"


def _collect_classification_labels(result: Any) -> set[str]:  # noqa: ANN401
    """Extract the union of label strings observed in a classify_audios output."""
    labels: set[str] = set()
    for window in _classification_windows(result):
        if not isinstance(window, dict):
            continue
        for label in (next(iter(d)) for d in label_scores(window)):
            if label:
                labels.add(str(label))
    return labels


# ── background mask + per-speaker speech_presence tracks (T106) ──────────────

MASK_STATE_VALUES = ("target_free", "target_active", "indeterminate")


def _mask_track_name(pass_label: str) -> str:
    return f"{pass_label}__background__mask"


def _speaker_track_name(pass_label: str) -> str:
    return f"{pass_label}__speaker__speech_presence"


def attach_scene_context_tracks_to_ls(
    *,
    ls_tasks: Any,  # noqa: ANN401 — list[dict] or dict, matching attach_uncertainty_tracks_to_ls
    ls_config: str,
    mask_rows: Sequence[Mapping[str, Any]] = (),
    speaker_rows: Sequence[Mapping[str, Any]] = (),
    pass_label: str = "raw_16k",
) -> tuple[Any, str]:
    """Append the background-mask and per-speaker speech_presence tracks to the LS bundle.

    Both answer questions a human reviewer cannot answer from the uncertainty tracks alone.
    The mask decides which background findings are trustworthy, so a reviewer checking those
    findings needs to see the same intervals the machine used (FR-033). Per-speaker speech_presence
    is labelled by speaker rather than merged, because knowing *who* is contested is the
    entire reason the speaker axis moved off a single scalar — a merged track would put the
    same unreadable number back in front of the annotator.

    Args:
        ls_tasks: Existing LS tasks payload.
        ls_config: Existing LS config XML.
        mask_rows: Background-mask rows with ``start``, ``end``, ``state``.
        speaker_rows: Per-speaker speech_presence rows.
        pass_label: Pass the tracks describe. Both are properties of the recording as
            captured, so they ride on the unmodified pass.

    Returns:
        Updated ``(ls_tasks, ls_config)``. With neither input the bundle is returned
        unchanged rather than gaining empty tracks.
    """
    if not mask_rows and not speaker_rows:
        return ls_tasks, ls_config

    mask_track, speaker_track = _mask_track_name(pass_label), _speaker_track_name(pass_label)
    blocks: list[str] = []
    if mask_rows:
        inner = "\n".join(f'  <Label value="{v}"/>' for v in MASK_STATE_VALUES)
        blocks.append(f'<Labels name="{mask_track}" toName="audio">\n{inner}\n</Labels>')
    if speaker_rows:
        # Label values are the speaker ids actually present, so the config declares exactly
        # the speakers this run hypothesized. An undeclared value is dropped on import.
        ids = sorted({str(r.get("speaker_id")) for r in speaker_rows if r.get("speaker_id")})
        inner = "\n".join(f'  <Label value="{sid}"/>' for sid in ids)
        blocks.append(f'<Labels name="{speaker_track}" toName="audio">\n{inner}\n</Labels>')
        blocks.append(
            f'<TextArea name="{speaker_track}__text" toName="audio" perRegion="true" '
            f'editable="false" placeholder="Per-speaker speech_presence confidence and backing sources"/>'
        )

    if "</View>" in ls_config:
        ls_config = ls_config.replace("</View>", "\n".join(blocks) + "\n</View>", 1)
    else:
        ls_config = ls_config + "\n" + "\n".join(blocks)

    tasks_list = ls_tasks if isinstance(ls_tasks, list) else [ls_tasks]
    target = next(
        (t for t in tasks_list if ((t.get("data") or {}).get("pass") or "raw_16k") == pass_label),
        tasks_list[0] if tasks_list else None,
    )
    if target is None or not target.get("predictions"):
        return ls_tasks, ls_config

    # Built in full before anything is mutated. A half-applied attachment leaves regions
    # pointing at tracks the config never declared, and Label Studio drops those silently —
    # so the bundle would read as successfully annotated with data quietly missing.
    result_list: list[dict[str, Any]] = []

    for i, row in enumerate(mask_rows):
        state = str(row.get("state") or "indeterminate")
        result_list.append(
            {
                "id": f"{mask_track}__{i}",
                "from_name": mask_track,
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": float(row.get("start", 0.0)),
                    "end": float(row.get("end", 0.0)),
                    "labels": [state if state in MASK_STATE_VALUES else "indeterminate"],
                },
            }
        )

    for i, row in enumerate(speaker_rows):
        region_id = f"{speaker_track}__{i}"
        start, end = float(row.get("start", 0.0)), float(row.get("end", 0.0))
        result_list.append(
            {
                "id": region_id,
                "from_name": speaker_track,
                "to_name": "audio",
                "type": "labels",
                "value": {"start": start, "end": end, "labels": [str(row.get("speaker_id"))]},
            }
        )
        # The speaker's own doubt travels with the region: without it a reviewer sees who
        # was claimed but not how doubtful the claim was, which is the actionable part.
        conf, unc = row.get("speech_presence_confidence"), row.get("speech_presence_uncertainty")
        # Parquet list columns read back as numpy arrays, whose truthiness raises rather
        # than falling through to a default.
        raw_sources = row.get("contributing_sources")
        listed = [] if raw_sources is None else list(raw_sources)
        sources = ", ".join(str(s) for s in listed) or "(none recorded)"
        result_list.append(
            {
                "id": f"{region_id}__text",
                "from_name": f"{speaker_track}__text",
                "to_name": "audio",
                "type": "textarea",
                "value": {
                    "start": start,
                    "end": end,
                    "text": [
                        f"confidence: {conf if conf is None else round(float(conf), 2)}\n"
                        f"uncertainty: {unc if unc is None else round(float(unc), 2)}\n"
                        f"backed by: {sources}"
                    ],
                },
            }
        )

    target["predictions"][0].setdefault("result", []).extend(result_list)
    return ls_tasks, ls_config
