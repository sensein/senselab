"""VOICE — the phonation spans PREPROCESS detected, measured. It measures; it does not classify.

The subject is a store read: every live ``span`` whose ``family`` is ``phonation``. Nothing another
branch claimed is removed from it, and no span is refused for being unvoiced. The HNR, F0 and RMS
tracks are measured once over the whole stream and sliced per span by time, each on its own frame
grid; only the point process is queried per span, and it is absent outside voiced and mixed spans.
The onset is a period where one exists and the offset is always a criterion, named apart in the span
attributes; the criterion itself is PREPROCESS's, reported verbatim. A span shorter than the window
the point process needs has no marks measured and is counted in ``marks_skipped_short_n``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import AudioHints
from senselab.audio.tasks.phonation import PeriodMark, f0_track, hnr_track, period_marks
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    clamp_extent,
    find_measurement,
    live_entities,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

NODE = "VOICE"
KIND = "voice"
_PHONATION_FAMILY = "phonation"
_MARK_PERIODS = 3.0  # Praat's point process needs this many periods of f0_min; its own limit, not a choice
_MARKS_UNMEASURED = "shorter_than_mark_window"  # a vocabulary token for the skip, not a threshold
_UNVOICED_SPAN = "unvoiced_span"  # the other reason nobody looked, told apart from the first
_TASK_NOT_EVALUATED = "not_evaluated"
_NO_TASK_DECLARED = "no_task_declared"
_TASK_HAS_NO_RANGE = "task_has_no_declared_range"


def _f0_range(config: TriageConfig, hint: AudioHints | None) -> tuple[float, float]:
    """The F0 search range for this recording's declared population.

    Args:
        config: The triage configuration.
        hint: The caller's hint; ``metadata["population"]`` selects an override.

    Returns:
        ``(f0_min_hz, f0_max_hz)``.

    Raises:
        ValueError: If ``voice.f0_range_hz`` is unmeasured, or if ``f0_max / f0_min`` exceeds
            ``voice.f0_range_ratio_max`` — a period-doubling check over a range that wide flags every
            recording, and a check that fires on everything reports nothing, so the configuration is
            refused rather than run and flagged.
    """
    declared = hint.metadata.get("population") if hint is not None else None
    population = str(declared) if declared else None
    by_population = config.get("voice.f0_range_by_population") or {}
    raw = by_population.get(population) if population is not None else None
    if raw is None:
        raw = config.require("voice.f0_range_hz")
    f0_min_hz, f0_max_hz = float(raw[0]), float(raw[1])
    ratio_max = config.get("voice.f0_range_ratio_max")
    if ratio_max is not None and f0_max_hz / f0_min_hz > float(ratio_max):
        raise ValueError(
            f"voice.f0_range_ratio_max is {float(ratio_max)} and the declared range "
            f"[{f0_min_hz}, {f0_max_hz}] has ratio {f0_max_hz / f0_min_hz:.2f}; the period-doubling "
            "check over that range flags every recording and is refused rather than run"
        )
    return f0_min_hz, f0_max_hz


def _required(config: TriageConfig, hint: AudioHints | None) -> dict[str, Any]:
    """Resolve every ``require()`` key at entry, before the store is touched (N2).

    Args:
        config: The triage configuration.
        hint: The caller's hint, which selects the population's F0 range.

    Returns:
        The resolved analysis parameters and the period-doubling identity factor.

    Raises:
        ValueError: If any key read here is null, or if the declared range is vacuous.
    """
    f0_min_hz, f0_max_hz = _f0_range(config, hint)
    return {
        "f0_min_hz": f0_min_hz,
        "f0_max_hz": f0_max_hz,
        "hop_s": float(config.require("phonation.hop_s")),
        "silence_threshold": float(config.require("phonation.silence_threshold")),
        "periods_per_window": float(config.require("phonation.periods_per_window")),
        "doubling": float(config.require("phonation.period_doubling_factor")),
        "span_hop_s": float(config.require("phonation_spans.hop_s")),
    }


def _rms_track(x: np.ndarray, sr: int, times: np.ndarray, window_s: float) -> np.ndarray:
    """Root-mean-square amplitude at each frame time, over a window centred on it.

    ``window_s`` is ``periods_per_window / f0_min_hz`` — the window Praat's harmonicity uses, an
    identity on existing config keys, so no new constant appears.
    """
    half = window_s / 2.0
    out = np.empty(len(times))
    for k, t in enumerate(times):
        i0 = max(0, int((t - half) * sr))
        i1 = min(len(x), int((t + half) * sr))
        out[k] = float(np.sqrt(np.mean(np.square(x[i0:i1])))) if i1 > i0 else 0.0
    return out


def _alias_in_range(f0_median_hz: float, *, factor: float, f0_min_hz: float, f0_max_hz: float) -> bool:
    """Whether the period-doubling alias of this F0 also lies inside the search range (N21)."""
    return f0_median_hz * factor <= f0_max_hz or f0_median_hz / factor >= f0_min_hz


def _task_range(
    config: TriageConfig, hint: AudioHints | None, longest_span_s: float
) -> tuple[str | dict[str, Any], str | None]:
    """Read the longest span against the range its declared task expects.

    Args:
        config: The triage configuration.
        hint: The caller's hint; ``metadata["task"]`` names the task.
        longest_span_s: The duration to read against the range.

    Returns:
        The verdict's ``task_range`` value and the flag naming the declared range, or None. The value
        is a vocabulary token whenever no reading was possible and the four-field row when one was.
    """
    ranges = config.get("voice.task_duration_ranges")
    if ranges is None:
        return _TASK_NOT_EVALUATED, None
    declared = hint.metadata.get("task") if hint is not None else None
    if not declared:
        return _NO_TASK_DECLARED, None
    task = str(declared)
    raw = ranges.get(task)
    if raw is None:
        return _TASK_HAS_NO_RANGE, None
    low, high = float(raw[0]), float(raw[1])
    within = low <= longest_span_s <= high
    row = {"task": task, "range": [low, high], "longest_span_s": longest_span_s, "within": within}
    if within:
        return row, None
    return (
        row,
        f"task_duration_outside_range: {task} declares [{low}, {high}] and the longest span is {longest_span_s:.3f}s",
    )


def voice(  # noqa: C901 — the store read, the tracks and the per-span assembly, in order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Measure the phonation spans PREPROCESS detected.

    Args:
        store: The provenance store, holding PREPROCESS's streams and phonation spans.
        source: The store-held stream name the audio is sliced from, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; read for the population and the task. A
            declaration this branch's measurements contradict is named by VERDICT's fold.
        run_dir: The run directory sidecar paths are relative to; ``voice_tracks.npz`` goes under
            ``derivatives/``.

    Returns:
        The verdict, the view over the spans and measurements written, and the verdict entity id.

    Raises:
        ValueError: If a key read at entry is null, or the declared F0 range is vacuous (N2) —
            raised before the store is touched.
        LookupError: If the ``source`` stream is absent.
    """
    params = _required(config, hint)
    hnr_interval = config.get("phonation.hnr_floor_interval_db")
    rms_interval = config.get("phonation.rms_floor_interval")
    if hnr_interval is not None and rms_interval is not None:
        gate_interval = "measured"
    elif hnr_interval is None and rms_interval is None:
        gate_interval = "unmeasured"
    else:
        gate_interval = "partial"
    f0_min_hz, f0_max_hz = params["f0_min_hz"], params["f0_max_hz"]
    window_s = params["periods_per_window"] / f0_min_hz
    min_marks_s = _MARK_PERIODS / f0_min_hz
    # A frame stands for a hop-wide interval centred on its time, so a span's measurable extent runs
    # from half a hop before its first frame to half a hop after its last: the tolerance is the hop,
    # an identity of the analysis grid.
    frame_edge_tolerance_s = params["span_hop_s"]

    stream_id, plain = resolve_stream(store, run_dir, source)
    sr = int(plain.sampling_rate)

    software = software_agent(store)
    activity = store.activity(
        node=NODE,
        step="analyze",
        parameters={
            "f0_range_hz": [f0_min_hz, f0_max_hz],
            "hop_s": params["hop_s"],
            "window_s": window_s,
            "min_marks_s": min_marks_s,
            "frame_edge_tolerance_s": frame_edge_tolerance_s,
            "period_doubling_factor": params["doubling"],
            "gate_interval": gate_interval,
        },
    )
    store.was_associated_with(activity, software)
    store.used(activity, stream_id)
    for name in ("energy_envelope", "silence"):
        measurement = find_measurement(store, name)
        if measurement is not None:
            store.used(activity, measurement.id)

    spans = [
        entity
        for entity in live_entities(store, "span")
        if entity.attributes.get("family") == _PHONATION_FAMILY and entity.extent is not None
    ]
    spans.sort(key=lambda entity: entity.extent or (0.0, 0.0))
    for span in spans:
        store.used(activity, span.id)
    if not spans:
        why = "PREPROCESS detected no phonation span"
        outcome = Outcome.FAIL
        verdict_id, verdict = write_verdict(
            store,
            activity,
            software,
            node=NODE,
            outcome=outcome,
            kind=KIND,
            why=why,
            detail={
                "spans_n": 0,
                "phonation_s": 0.0,
                "longest_span_s": 0.0,
                "longest_span_criterion": None,
                "production": {"voiced": 0, "unvoiced": 0, "mixed": 0},
                "ambiguous_spans_n": 0,
                "marks_skipped_short_n": 0,
                "task_range": _TASK_NOT_EVALUATED
                if config.get("voice.task_duration_ranges") is None
                else _NO_TASK_DECLARED,
                "gate_interval": gate_interval,
                "flags": [],
            },
        )
        return NodeResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id)

    mono = plain.waveform.mean(dim=0).numpy().astype(np.float64)
    stream_times, stream_hnr = hnr_track(
        plain,
        f0_min_hz=f0_min_hz,
        hop_s=params["hop_s"],
        silence_threshold=params["silence_threshold"],
        periods_per_window=params["periods_per_window"],
    )
    stream_rms = _rms_track(mono, sr, stream_times, window_s)
    stream_f0_times, stream_f0, stream_strength = f0_track(
        plain, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz, hop_s=params["hop_s"]
    )
    track_times: list[np.ndarray] = []
    track_rms: list[np.ndarray] = []
    track_hnr: list[np.ndarray] = []
    f0_times: list[np.ndarray] = []
    f0_values: list[np.ndarray] = []
    f0_strengths: list[np.ndarray] = []
    span_ids: list[str] = []
    mark_ids: list[str] = []
    all_periods: list[float] = []
    flags: list[str] = []
    production_counts = {"voiced": 0, "unvoiced": 0, "mixed": 0}
    phonation_s = 0.0
    longest_span_s = 0.0
    longest_span_criterion: str | None = None
    ambiguous_spans_n = 0
    marks_skipped_short_n = 0

    for span in spans:
        assert span.extent is not None  # noqa: S101 — the comprehension above admits no other case
        start, end = clamp_extent(span.extent, plain)
        frames = (stream_times >= start) & (stream_times < end)
        track_times.append(stream_times[frames])
        track_rms.append(stream_rms[frames])
        track_hnr.append(stream_hnr[frames])
        pitch_frames = (stream_f0_times >= start) & (stream_f0_times < end)
        f0_times.append(stream_f0_times[pitch_frames])
        f0_values.append(stream_f0[pitch_frames])
        f0_strengths.append(stream_strength[pitch_frames])

        duration_s = float(span.attributes["duration_s"])
        phonation_s += duration_s
        if duration_s > longest_span_s:
            longest_span_s = duration_s
            longest_span_criterion = str(span.attributes["offset_criterion"])
        production = str(span.attributes["production"])
        production_counts[production] = production_counts.get(production, 0) + 1

        measurable_s = (span.extent[1] - span.extent[0]) + frame_edge_tolerance_s
        marks: list[PeriodMark] = []
        marks_attributes: dict[str, Any] = {"name": "period_marks", "signal": source}
        if production == "unvoiced":
            marks_attributes["unmeasured"] = _UNVOICED_SPAN
        elif measurable_s < min_marks_s:
            marks_attributes["unmeasured"] = _MARKS_UNMEASURED
            marks_skipped_short_n += 1
        else:
            marks = period_marks(plain, start, end, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz)
            marks_attributes["n"] = len(marks)
            marks_attributes["marks"] = [
                {"time_s": m.time_s, "period_s": m.period_s, "amplitude": m.amplitude} for m in marks
            ]

        onset_s = float(marks[0].time_s) if marks else start
        span_id = store.entity(
            prov_type="span",
            extent=(onset_s, end),
            attributes={
                "family": _PHONATION_FAMILY,
                "member": str(span.attributes["member"]),
                "production": production,
                "duration_s": duration_s,
                "onset_kind": "period" if marks else "criterion",
                "offset_kind": "criterion",
                "offset_criterion": str(span.attributes["offset_criterion"]),
                "marks_n": len(marks) if "n" in marks_attributes else None,
            },
        )
        store.was_generated_by(span_id, activity)
        store.was_attributed_to(span_id, software)
        store.was_derived_from(span_id, span.id)
        span_ids.append(span_id)

        marks_id = store.entity(prov_type="measurement", extent=(onset_s, end), attributes=marks_attributes)
        store.was_generated_by(marks_id, activity)
        store.was_attributed_to(marks_id, software)
        store.was_derived_from(marks_id, span_id)
        mark_ids.append(marks_id)

        if marks:
            periods = [m.period_s for m in marks]
            all_periods.extend(periods)
            span_f0 = float(1.0 / np.median(periods))
            if _alias_in_range(span_f0, factor=params["doubling"], f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz):
                ambiguous_spans_n += 1
                flags.append(f"period_doubling_alias in range for span at {onset_s:.3f}s")

    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    tracks_path = "derivatives/voice_tracks.npz"
    np.savez(
        run_dir / tracks_path,
        times_s=np.concatenate(track_times),
        rms=np.concatenate(track_rms),
        hnr_db=np.concatenate(track_hnr),
        f0_times_s=np.concatenate(f0_times),
        f0_hz=np.concatenate(f0_values),
        f0_strength=np.concatenate(f0_strengths),
    )
    tracks_id = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "voice_tracks", "signal": source, "path": tracks_path, "hop_s": params["hop_s"]},
    )
    store.was_generated_by(tracks_id, activity)
    store.was_attributed_to(tracks_id, software)

    task_range, task_flag = _task_range(config, hint, longest_span_s)
    if task_flag is not None:
        flags.append(task_flag)

    if flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome, why = Outcome.PASS, "phonation spans measured; nothing contested"

    detail: dict[str, Any] = {
        "spans_n": len(spans),
        "phonation_s": phonation_s,
        "longest_span_s": longest_span_s,
        "longest_span_criterion": longest_span_criterion,
        "production": production_counts,
        "ambiguous_spans_n": ambiguous_spans_n,
        "marks_skipped_short_n": marks_skipped_short_n,
        "task_range": task_range,
        "gate_interval": gate_interval,
        "flags": flags,
    }
    if all_periods:
        detail["f0_median_hz"] = float(1.0 / np.median(all_periods))
        detail["f0_stream"] = source
    verdict_id, verdict = write_verdict(
        store, activity, software, node=NODE, outcome=outcome, kind=KIND, why=why, detail=detail
    )
    view = tuple(span_ids + mark_ids + [tracks_id, verdict_id])
    return NodeResult(verdict=verdict, view=view, verdict_entity_id=verdict_id)
