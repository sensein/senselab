"""VOICE — vocalic activity nobody else claimed. It measures; it does not classify.

The residual is a store fold: contiguous intervals where the envelope exceeds its local floor, minus
airway-labelled spans, minus SPEECH's spans — an unlabelled span is not excluded. The gate is energy
AND periodicity; runs are elementary; period marks are a point process per voiced run, absent outside
runs. The onset is a period and the offset is a criterion, named apart in the span attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.phonation import PeriodMark, f0_track, hnr_track, period_marks
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    find_measurement,
    resolve_stream,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore

NODE = "VOICE"
KIND = "voice_no_words"


def _required(config: TriageConfig) -> dict[str, Any]:
    """Resolve every ``require()`` key at entry, before the store is touched (N2).

    Args:
        config: The triage configuration.

    Returns:
        The resolved phonation parameters, the period-doubling identity factor and the hint tags.

    Raises:
        ValueError: If any ``phonation.*`` key is null — the gate never invents a floor.
    """
    return {
        "f0_min_hz": float(config.require("phonation.f0_min_hz")),
        "f0_max_hz": float(config.require("phonation.f0_max_hz")),
        "hnr_floor_db": float(config.require("phonation.hnr_floor_db")),
        "rms_floor": float(config.require("phonation.rms_floor")),
        "hop_s": float(config.require("phonation.hop_s")),
        "silence_threshold": float(config.require("phonation.silence_threshold")),
        "periods_per_window": float(config.require("phonation.periods_per_window")),
        "doubling": float(config.require("phonation.period_doubling_factor")),
        "hint_tags": [str(tag) for tag in config.require("voice.hint_tags")],
    }


def _generating_node(store: ProvStore, entity_id: str) -> str | None:
    """The node whose activity generated this entity, or None (N19)."""
    activity_id = store.generated_by(entity_id)
    if activity_id is None:
        return None
    return store.get_activity(activity_id).node


def _runs_of_true(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal ``[i0, i1)`` index runs where the mask holds. Elementary: no merging."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]


def _contiguous_true(mask: np.ndarray, rate: int) -> list[tuple[float, float]]:
    """Contiguous True stretches of a boolean track, as ``(start_s, end_s)`` at this rate (N20)."""
    return [(i0 / rate, i1 / rate) for i0, i1 in _runs_of_true(mask)]


def _subtract_intervals(
    intervals: list[tuple[float, float]], claimed: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    """Remove every claimed extent from the intervals, keeping what nobody claimed."""
    out: list[tuple[float, float]] = []
    for start, end in intervals:
        pieces = [(start, end)]
        for c0, c1 in claimed:
            survivors: list[tuple[float, float]] = []
            for p0, p1 in pieces:
                if c1 <= p0 or c0 >= p1:
                    survivors.append((p0, p1))
                    continue
                if p0 < c0:
                    survivors.append((p0, c0))
                if c1 < p1:
                    survivors.append((c1, p1))
            pieces = survivors
        out.extend(pieces)
    return [(p0, p1) for p0, p1 in out if p1 > p0]


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


def _airway_labelled(store: ProvStore) -> list[tuple[tuple[float, float], str, str]]:
    """Spans carrying a live ``label`` assertion authored by an AIRWAY activity (N19).

    Returns:
        ``(extent, span_id, assertion_id)`` per labelled span. A span AIRWAY proposed and declined
        to label is not here — an unlabelled span is exactly where unclaimed vocalic activity sits.
    """
    out: list[tuple[tuple[float, float], str, str]] = []
    for assertion in store.entities("assertion"):
        if assertion.attributes.get("verb") != "label" or store.is_invalidated(assertion.id):
            continue
        if _generating_node(store, assertion.id) != "AIRWAY":
            continue
        for source_id in store.derived_from(assertion.id):
            source = store.get_entity(source_id)
            if source.prov_type == "span" and source.extent is not None:
                out.append((source.extent, source_id, assertion.id))
    return out


def _speech_spans(store: ProvStore) -> list[Entity]:
    """SPEECH's live speech spans, attributed by generating activity (N19)."""
    return [
        e
        for e in store.entities("span")
        if e.extent is not None and not store.is_invalidated(e.id) and _generating_node(store, e.id) == "SPEECH"
    ]


def _alias_in_range(f0_median_hz: float, *, factor: float, f0_min_hz: float, f0_max_hz: float) -> bool:
    """Whether the period-doubling alias of this F0 also lies inside the search range (N21)."""
    return f0_median_hz * factor <= f0_max_hz or f0_median_hz / factor >= f0_min_hz


def _hint_declares_voice(hint: AudioHints | None, hint_tags: list[str]) -> bool:
    """Whether the caller declared phonation content (N25)."""
    if hint is None:
        return False
    return bool({tag.lower() for tag in hint.may_contain} & {tag.lower() for tag in hint_tags})


def _offset_criterion(hnr_db: np.ndarray, rms: np.ndarray, i1: int, *, hnr_floor_db: float, rms_floor: float) -> str:
    """Which gate condition stopped holding at the frame after the run, or ``residual_end``."""
    if i1 >= len(hnr_db):
        return "residual_end"
    hnr_stopped = bool(hnr_db[i1] < hnr_floor_db)
    rms_stopped = bool(rms[i1] < rms_floor)
    if hnr_stopped and rms_stopped:
        return "both"
    return "hnr" if hnr_stopped else "rms"


def voice(  # noqa: C901 — the fold, the gate and the per-run assembly, in order
    store: ProvStore,
    source: str,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> NodeResult:
    """Measure voiced runs over the residual — energetic intervals no other branch claimed.

    Args:
        store: The provenance store, holding PREPROCESS's envelope and spans, AIRWAY's labels and
            SPEECH's spans.
        source: The store-held stream name the audio is sliced from, ``"plain"``.
        config: The triage configuration.
        hint: What the recording was declared to contain; read only to condition an absence.
        run_dir: The run directory sidecar paths are relative to; ``voice_tracks.npz`` goes under
            ``derivatives/``.

    Returns:
        The verdict, the view over the runs and measurements written, and the verdict entity id.

    Raises:
        ValueError: If any ``phonation.*`` key is null (N2) — raised before the store is touched.
        LookupError: If the ``plain`` stream or the ``energy_envelope`` measurement is absent.
    """
    params = _required(config)
    hnr_interval = config.get("phonation.hnr_floor_interval_db")
    rms_interval = config.get("phonation.rms_floor_interval")
    if hnr_interval is not None and rms_interval is not None:
        gate_interval = "measured"
    elif hnr_interval is None and rms_interval is None:
        gate_interval = "unmeasured"
    else:
        gate_interval = "partial"
    window_s = params["periods_per_window"] / params["f0_min_hz"]

    envelope_meas = find_measurement(store, "energy_envelope")
    if envelope_meas is None:
        raise LookupError("no energy_envelope measurement in the store; PREPROCESS has not run")
    stream_id, plain = resolve_stream(store, run_dir, source)
    sr = int(plain.sampling_rate)

    software = software_agent(store)
    activity = store.activity(
        node=NODE,
        step="analyze",
        parameters={
            "f0_min_hz": params["f0_min_hz"],
            "f0_max_hz": params["f0_max_hz"],
            "hnr_floor_db": params["hnr_floor_db"],
            "rms_floor": params["rms_floor"],
            "hop_s": params["hop_s"],
            "window_s": window_s,
            "period_doubling_factor": params["doubling"],
            "gate_interval": gate_interval,
        },
    )
    store.was_associated_with(activity, software)
    store.used(activity, stream_id)
    store.used(activity, envelope_meas.id)
    silence = find_measurement(store, "silence")
    if silence is not None:
        store.used(activity, silence.id)

    labelled = _airway_labelled(store)
    speech = _speech_spans(store)
    for _, span_id, assertion_id in labelled:
        store.used(activity, span_id)
        store.used(activity, assertion_id)
    for span in speech:
        store.used(activity, span.id)

    sidecar = np.load(run_dir / envelope_meas.attributes["path"])
    envelope, floor = sidecar["envelope_dbfs"], sidecar["floor_dbfs"]
    envelope_rate = int(envelope_meas.attributes["sampling_rate"])
    energetic = _contiguous_true(envelope > floor, envelope_rate)
    claimed = [extent for extent, _, _ in labelled] + [span.extent for span in speech if span.extent is not None]
    residual = _subtract_intervals(energetic, claimed)
    hint_declares = _hint_declares_voice(hint, params["hint_tags"])

    if not residual:
        why = "every energetic interval is claimed by another branch" if energetic else "no energy exceeds the floor"
        if hint_declares:
            why += "; a hint declares phonation not found"
        outcome = Outcome.FLAG if hint_declares else Outcome.FAIL
        verdict_id, verdict = write_verdict(
            store,
            activity,
            software,
            node=NODE,
            outcome=outcome,
            kind=KIND,
            why=why,
            detail={
                "runs_n": 0,
                "voiced_s": 0.0,
                "ambiguous_runs_n": 0,
                "flags": [why] if hint_declares else [],
                "gate_interval": gate_interval,
            },
        )
        return NodeResult(verdict=verdict, view=(verdict_id,), verdict_entity_id=verdict_id)

    mono = plain.waveform.mean(dim=0).numpy().astype(np.float64)
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
    runs_n = 0
    voiced_s = 0.0
    ambiguous_runs_n = 0

    for interval_start, interval_end in residual:
        i0, i1 = int(interval_start * sr), int(interval_end * sr)
        segment = Audio(waveform=plain.waveform[:, i0:i1], sampling_rate=sr)
        times_rel, hnr_db = hnr_track(
            segment,
            f0_min_hz=params["f0_min_hz"],
            hop_s=params["hop_s"],
            silence_threshold=params["silence_threshold"],
            periods_per_window=params["periods_per_window"],
        )
        pitch_times_rel, f0_hz, strength = f0_track(
            segment, f0_min_hz=params["f0_min_hz"], f0_max_hz=params["f0_max_hz"], hop_s=params["hop_s"]
        )
        rms = _rms_track(mono[i0:i1], sr, times_rel, window_s)
        times = times_rel + interval_start
        track_times.append(times)
        track_rms.append(rms)
        track_hnr.append(hnr_db)
        f0_times.append(pitch_times_rel + interval_start)
        f0_values.append(f0_hz)
        f0_strengths.append(strength)

        gate_ok = (hnr_db >= params["hnr_floor_db"]) & (rms >= params["rms_floor"])
        for r0, r1 in _runs_of_true(gate_ok):
            gate_start, gate_end = float(times[r0]), float(times[r1 - 1])
            marks: list[PeriodMark] = period_marks(
                plain, gate_start, gate_end, f0_min_hz=params["f0_min_hz"], f0_max_hz=params["f0_max_hz"]
            )
            span_start = float(marks[0].time_s) if marks else gate_start
            attributes = {
                "onset_kind": "period" if marks else "criterion",
                "offset_kind": "criterion",
                "offset_criterion": _offset_criterion(
                    hnr_db, rms, r1, hnr_floor_db=params["hnr_floor_db"], rms_floor=params["rms_floor"]
                ),
                "marks_n": len(marks),
                "hnr_onset_db": float(hnr_db[r0]),
                "rms_onset": float(rms[r0]),
                "hnr_offset_db": float(hnr_db[r1 - 1]),
                "rms_offset": float(rms[r1 - 1]),
            }
            run_id = store.entity(prov_type="span", extent=(span_start, gate_end), attributes=attributes)
            store.was_generated_by(run_id, activity)
            store.was_attributed_to(run_id, software)
            span_ids.append(run_id)
            marks_id = store.entity(
                prov_type="measurement",
                extent=(span_start, gate_end),
                attributes={
                    "name": "period_marks",
                    "signal": source,
                    "n": len(marks),
                    "marks": [{"time_s": m.time_s, "period_s": m.period_s, "amplitude": m.amplitude} for m in marks],
                },
            )
            store.was_generated_by(marks_id, activity)
            store.was_attributed_to(marks_id, software)
            store.was_derived_from(marks_id, run_id)
            mark_ids.append(marks_id)
            runs_n += 1
            voiced_s += gate_end - span_start
            if marks:
                periods = [m.period_s for m in marks]
                all_periods.extend(periods)
                run_f0 = float(1.0 / np.median(periods))
                if _alias_in_range(
                    run_f0, factor=params["doubling"], f0_min_hz=params["f0_min_hz"], f0_max_hz=params["f0_max_hz"]
                ):
                    ambiguous_runs_n += 1
                    flags.append(f"period_doubling_alias in range for run at {span_start:.3f}s")
            if hnr_interval is not None and hnr_interval[0] <= attributes["hnr_onset_db"] <= hnr_interval[1]:
                flags.append(f"near_gate_edge hnr at onset of run at {span_start:.3f}s")
            if rms_interval is not None and rms_interval[0] <= attributes["rms_onset"] <= rms_interval[1]:
                flags.append(f"near_gate_edge rms at onset of run at {span_start:.3f}s")

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

    if runs_n == 0:
        why = "no run passes the energy-and-periodicity gate"
        if hint_declares:
            why += "; a hint declares phonation not found"
            flags.append(why)
            outcome = Outcome.FLAG
        else:
            outcome = Outcome.FAIL
    elif flags:
        outcome, why = Outcome.FLAG, "; ".join(flags)
    else:
        outcome, why = Outcome.PASS, "voiced runs measured; nothing contested"

    detail: dict[str, Any] = {
        "runs_n": runs_n,
        "voiced_s": voiced_s,
        "ambiguous_runs_n": ambiguous_runs_n,
        "flags": flags,
        "gate_interval": gate_interval,
    }
    if all_periods:
        detail["f0_median_hz"] = float(1.0 / np.median(all_periods))
    verdict_id, verdict = write_verdict(
        store, activity, software, node=NODE, outcome=outcome, kind=KIND, why=why, detail=detail
    )
    view = tuple(span_ids + mark_ids + [tracks_id, verdict_id])
    return NodeResult(verdict=verdict, view=view, verdict_entity_id=verdict_id)
