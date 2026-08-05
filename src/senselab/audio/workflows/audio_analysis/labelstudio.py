"""Label Studio bundle integration for the three uncertainty axes.

The bundle exposes:
    - one Labels track per fused L2 axis, named ``uncertainty__<axis>``. No pass token: an axis is a
      fold across passes, so there is no per-pass axis to draw.
    - **no transcript text.** There was an ``uncertainty__asr__text`` TextArea rebuilding a
      per-bucket consensus from each model's bucketed transcript; the words are published at word
      resolution in ``final/transcript.json``, and ``adaptive.ls_final`` renders them as
      ``final__consensus_transcript__text`` in the deliverable bundle this one is the input to. Two
      renderings of one transcript at two resolutions is one too many, and the coarse one is what
      forced the asr axis onto a 1.0 s grid of its own.
    - per-pass, per-signal evidence tracks ``<pass>__signal__<signal>`` straight from the L1
      signal rows. That is where "what did each model say on each pass" is legitimately served —
      per pass without being an axis.
    - the scene tracks ``<pass>__presence__{quality,sources}``, which are per-pass
      *measurements* and stay per-pass.
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
from senselab.audio.workflows.audio_analysis.types import ComparisonStatus, FusedAxis, SignalResult
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


BIN_POLICY = {
    "policy": "labelstudio.uncertainty_to_label_bin",
    "low_threshold": LOW_THRESHOLD,
    "high_threshold": HIGH_THRESHOLD,
}
"""The binning thresholds, named so a rendered label can be traced to the rule that produced it.

Recorded on the bundle (``data.uncertainty_bin_policy``) rather than living only as two module
constants: a track that says "high" is a thresholded value, and L2's one-line test is that every
threshold which shaped a value is named in a policy recorded alongside it.
"""


def uncertainty_to_label_bin(value: float | None, status: ComparisonStatus | str) -> str:
    """Bin a fused axis value into one of the LS label values."""
    if status in ("incomparable", "unavailable"):
        return "unavailable" if status == "unavailable" else "incomparable"
    if value is None:
        return "incomparable"
    if value < LOW_THRESHOLD:
        return "low"
    if value < HIGH_THRESHOLD:
        return "medium"
    return "high"


def _track_name(axis: str) -> str:
    """Track carrying one fused axis. No pass token — an axis has no pass."""
    return f"uncertainty__{axis}"


def _signal_track_name(perturbation: str, signal: str) -> str:
    """Per-pass, per-signal evidence track: ``<pass>__signal__<signal>``."""
    return f"{perturbation}__signal__{re.sub(r'[^A-Za-z0-9_.-]+', '_', signal)}"


SOURCE_LABEL_VALUES = ("speech", "people", "machine", "environment", "unavailable")


def _build_labels_xml(track_name: str) -> str:
    inner = "\n".join(f'  <Label value="{v}"/>' for v in LABEL_VALUES)
    return f'<Labels name="{track_name}" toName="audio">\n{inner}\n</Labels>'


def _build_source_labels_xml(track_name: str) -> str:
    inner = "\n".join(f'  <Label value="{v}"/>' for v in SOURCE_LABEL_VALUES)
    return f'<Labels name="{track_name}" toName="audio">\n{inner}\n</Labels>'


def _scene_track_name(perturbation: str, kind: str) -> str:
    """Scene tracks: ``<pass>__presence__quality`` / ``<pass>__presence__sources``.

    Per pass because they carry per-pass *measurements*, not an axis fold.
    """
    return f"{perturbation}__presence__{kind}"


QUALITY_DISPLAY_FOLD = {
    "policy": "labelstudio._quality_degradation",
    "rule": "max over quality_snr / quality_clip / quality_reverb / quality_bandwidth",
    "purpose": "display only — one stripe cannot show four differently-anchored scores",
}
"""The rendering fold behind the quality track, named because it *is* a reduction.

It is a display choice, not a measurement: four scores anchored against four different references
collapse to one stripe so a reviewer can see where to look. The four remain separately on the
fused presence row.
"""


def _quality_degradation(row: Mapping[str, Any]) -> float | None:
    """Overall degradation for the quality track: max over the four quality columns."""
    values = [
        float(row[k])
        for k in ("quality_snr", "quality_clip", "quality_reverb", "quality_bandwidth")
        if isinstance(row.get(k), (int, float)) and row[k] == row[k]
    ]
    return max(values) if values else None


def attach_uncertainty_tracks_to_ls(
    *,
    ls_tasks: Any,  # noqa: ANN401 — list[dict] or dict, matches build_labelstudio_task variants
    ls_config: str,
    fused_axes: Mapping[str, FusedAxis],
    signal_results_by_pass: Mapping[str, Mapping[str, SignalResult]] | None = None,
) -> tuple[Any, str]:
    """Append one Labels track per fused axis, plus the per-pass signal evidence tracks.

    Args:
        ls_tasks: Existing LS tasks payload (single dict or list of dicts) — typically
            produced by ``scripts/analyze_audio.py``'s ``build_labelstudio_task``.
        ls_config: Existing LS config XML string.
        fused_axes: ``{axis → FusedAxis}`` — the L2 answer. One track per axis, attached once.
        signal_results_by_pass: ``{pass → {signal → SignalResult}}`` — the L1 evidence. One
            track per ``(pass, signal)``, which is the reviewer's "what did each model say on
            each pass" question, answered without inventing a per-pass axis.

    Returns:
        Updated ``(ls_tasks, ls_config)``.
    """
    tasks_list = ls_tasks if isinstance(ls_tasks, list) else [ls_tasks]
    by_pass_task: dict[str, dict[str, Any]] = {}
    for t in tasks_list:
        perturbation = (t.get("data") or {}).get("pass") or "raw"
        by_pass_task[perturbation] = t
    # An axis belongs to the recording, not to a transform of it, so its regions attach once —
    # to the as-recorded task.
    axis_task = by_pass_task.get("raw") or (tasks_list[0] if tasks_list else None)

    blocks: list[str] = []
    presence_rows = fused_axes["speech_presence"].rows if "speech_presence" in fused_axes else []
    for axis in sorted(fused_axes):
        blocks.append(_build_labels_xml(_track_name(axis)))
    for perturbation, by_signal in sorted((signal_results_by_pass or {}).items()):
        for signal in sorted(by_signal):
            blocks.append(_build_labels_xml(_signal_track_name(perturbation, signal)))
        if any(_quality_degradation(m) is not None for _s, _e, m in _scene_rows(by_signal, presence_rows)):
            blocks.append(_build_labels_xml(_scene_track_name(perturbation, "quality")))
        if "sound_sources" in by_signal:
            blocks.append(_build_source_labels_xml(_scene_track_name(perturbation, "sources")))

    if "</View>" in ls_config:
        ls_config = ls_config.replace("</View>", "\n".join(blocks) + "\n</View>", 1)
    else:
        ls_config = ls_config + "\n" + "\n".join(blocks)

    for t in tasks_list:
        # The thresholds that turned a number into "high" travel with the bundle, so a label can
        # be traced to the rule that produced it without reading this module.
        (t.setdefault("data", {}))["uncertainty_bin_policy"] = dict(BIN_POLICY)

    if axis_task is not None and axis_task.get("predictions"):
        result_list = axis_task["predictions"][0].setdefault("result", [])
        for axis in sorted(fused_axes):
            track = _track_name(axis)
            for row_idx, row in enumerate(fused_axes[axis].rows):
                value = row.get("uncertainty")
                region_id = f"{track}__{row_idx}"
                result_list.append(
                    {
                        "id": region_id,
                        "from_name": track,
                        "to_name": "audio",
                        "type": "labels",
                        "value": {
                            "start": float(row["start"]),
                            "end": float(row["end"]),
                            "labels": [uncertainty_to_label_bin(value, "ok" if value is not None else "incomparable")],
                        },
                    }
                )

    for perturbation, by_signal in sorted((signal_results_by_pass or {}).items()):
        target_task = by_pass_task.get(perturbation)
        if target_task is None or not target_task.get("predictions"):
            continue
        result_list = target_task["predictions"][0].setdefault("result", [])
        for signal in sorted(by_signal):
            track = _signal_track_name(perturbation, signal)
            for row_idx, signal_row in enumerate(by_signal[signal].rows):
                result_list.append(
                    {
                        "id": f"{track}__{row_idx}",
                        "from_name": track,
                        "to_name": "audio",
                        "type": "labels",
                        "value": {
                            "start": float(signal_row.start),
                            "end": float(signal_row.end),
                            "labels": [signal_row.status if signal_row.status != "ok" else "low"],
                        },
                    }
                )
        _attach_scene_rows(result_list, perturbation, by_signal, presence_rows)

    return ls_tasks, ls_config


def _scene_rows(
    by_signal: Mapping[str, SignalResult], presence_rows: Sequence[Mapping[str, Any]]
) -> list[tuple[float, float, dict[str, Any]]]:
    """``(start, end, measurement+scores)`` per bucket: L1 scene measurements, L2 scores joined on.

    The join is what keeps the two apart on disk. The dB readings come from
    ``L1/<pass>/signals/scene_quality.parquet``; the anchored ``quality_*`` scores come from the
    fused presence rows, where the calibration profile that produced them is recorded.
    """
    result = by_signal.get("scene_quality")
    if result is None:
        return []
    scores = {(round(float(r["start"]), 6), round(float(r["end"]), 6)): dict(r) for r in presence_rows}
    joined: list[tuple[float, float, dict[str, Any]]] = []
    for signal_row in result.rows:
        merged = dict(signal_row.measurement)
        merged.update(scores.get((round(signal_row.start, 6), round(signal_row.end, 6))) or {})
        joined.append((signal_row.start, signal_row.end, merged))
    return joined


def _attach_scene_rows(
    result_list: list[dict[str, Any]],
    perturbation: str,
    by_signal: Mapping[str, SignalResult],
    presence_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Attach the per-pass quality + source stripes from L1 scene rows."""
    q_track = _scene_track_name(perturbation, "quality")
    for row_idx, (start, end, merged) in enumerate(_scene_rows(by_signal, presence_rows)):
        degradation = _quality_degradation(merged)
        if degradation is None:
            continue
        result_list.append(
            {
                "id": f"{q_track}__{row_idx}",
                "from_name": q_track,
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": float(start),
                    "end": float(end),
                    "labels": [uncertainty_to_label_bin(degradation, "ok")],
                    "fold": dict(QUALITY_DISPLAY_FOLD),
                },
            }
        )
    sources = by_signal.get("sound_sources")
    if sources is None:
        return
    s_track = _scene_track_name(perturbation, "sources")
    for row_idx, source_row in enumerate(sources.rows):
        label = source_row.measurement.get("src_dominant") or source_row.measurement.get("dominant")
        if not isinstance(label, str):
            continue
        result_list.append(
            {
                "id": f"{s_track}__{row_idx}",
                "from_name": s_track,
                "to_name": "audio",
                "type": "labels",
                "value": {
                    "start": float(source_row.start),
                    "end": float(source_row.end),
                    "labels": [label if label in SOURCE_LABEL_VALUES else "unavailable"],
                },
            }
        )


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
    perturbation: str,
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
            from_name = f"{perturbation}__diarization__{safe_model_id(model_id)}"
            regions.extend(_diarization_to_ls(model_block.get("result"), from_name))

    ast_block = pass_summary.get("ast", {})
    if ast_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                ast_block.get("result"),
                f"{perturbation}__ast",
                win_length=ast_win_length,
                hop_length=ast_hop_length,
            )
        )

    yam_block = pass_summary.get("yamnet", {})
    if yam_block.get("status") == "ok":
        regions.extend(
            _classification_to_ls(
                yam_block.get("result"),
                f"{perturbation}__yamnet",
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
        from_name = f"{perturbation}__asr__{safe_model_id(model_id)}"
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
            "pass": perturbation,
            "duration_s": duration_s,
        },
        "predictions": [
            {
                "model_version": f"senselab-analyze:{perturbation}",
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

    The per-axis uncertainty tracks are appended downstream by
    ``senselab.audio.workflows.audio_analysis.attach_uncertainty_tracks_to_ls``.
    """
    parts: list[str] = ["<View>", '  <Audio name="audio" value="$audio"/>']
    seen_label_sets: dict[str, list[str]] = {}

    for perturbation, pass_summary in summary.get("passes", {}).items():
        # Diarization tracks: one per model, with that model's discovered speaker labels
        dia_by_model = (pass_summary.get("diarization") or {}).get("by_model") or {}
        for model_id, model_block in dia_by_model.items():
            if model_block.get("status") != "ok":
                continue
            speakers = sorted({str(getattr(seg, "speaker", "?")) for seg in (model_block.get("result", [[]])[0] or [])})
            if not speakers:
                speakers = ["SPEAKER_00", "SPEAKER_01"]
            seen_label_sets[f"{perturbation}__diarization__{safe_model_id(model_id)}"] = speakers

        # AST scene labels
        ast = pass_summary.get("ast") or {}
        if ast.get("status") == "ok":
            labels = _collect_classification_labels(ast.get("result"))
            if labels:
                seen_label_sets[f"{perturbation}__ast"] = sorted(labels)

        # YAMNet scene labels
        yam = pass_summary.get("yamnet") or {}
        if yam.get("status") == "ok":
            labels = _collect_classification_labels(yam.get("result"))
            if labels:
                seen_label_sets[f"{perturbation}__yamnet"] = sorted(labels)

        # ASR: each model gets its own TextArea
        asr_by_model = (pass_summary.get("asr") or {}).get("by_model") or {}
        for model_id, model_block in asr_by_model.items():
            if model_block.get("status") != "ok":
                continue
            from_name = f"{perturbation}__asr__{safe_model_id(model_id)}"
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


def _mask_track_name(perturbation: str) -> str:
    return f"{perturbation}__background__mask"


def _speaker_track_name(perturbation: str) -> str:
    return f"{perturbation}__speaker__speech_presence"


def attach_scene_context_tracks_to_ls(
    *,
    ls_tasks: Any,  # noqa: ANN401 — list[dict] or dict, matching attach_uncertainty_tracks_to_ls
    ls_config: str,
    mask_rows: Sequence[Mapping[str, Any]] = (),
    speaker_rows: Sequence[Mapping[str, Any]] = (),
    perturbation: str = "raw",
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
        perturbation: Pass the tracks describe. Both are properties of the recording as
            captured, so they ride on the unmodified pass.

    Returns:
        Updated ``(ls_tasks, ls_config)``. With neither input the bundle is returned
        unchanged rather than gaining empty tracks.
    """
    if not mask_rows and not speaker_rows:
        return ls_tasks, ls_config

    mask_track, speaker_track = _mask_track_name(perturbation), _speaker_track_name(perturbation)
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
        (t for t in tasks_list if ((t.get("data") or {}).get("pass") or "raw") == perturbation),
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
