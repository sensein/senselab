"""Per-recording census of what VOICE actually had in hand.

Reads each finished run's store, reconstructs the exact evidence `voice.py` reads, and replays the
qualifier gate by gate so every False and every absent can be attributed to the gate that decided
it. Reads only; writes one JSONL row per recording. No transcript text leaves.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes import voice as V
from senselab.audio.workflows.triage.nodes.branches import (
    VOICE_EXPECTATIONS,
    BranchParams,
    Expectation,
    Pattern,
    amplitude_spans,
    branch_params,
    duration,
    lexical,
    longest_monotone_run,
    max_windowed_spread,
    overlaps,
    robust_spread,
    semitones,
    spans_by_measure,
    trace_slice,
    track_slice,
    word_extent,
)
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

CONFIG = load_triage_config(None)
PARAMS = branch_params(CONFIG)


def window_spreads(values: np.ndarray, hop_s: float, window_s: float) -> np.ndarray:
    """Every window's spread, not only the worst — the statistic the gate reduces to one number."""
    series = np.asarray(values, dtype=float)
    width = max(2, int(round(window_s / hop_s))) if hop_s > 0.0 else 2
    if series.size <= width:
        return np.array([robust_spread(series)])
    return np.array([robust_spread(series[i : i + width]) for i in range(0, series.size - width + 1)])


def gate_census(evidence: V.Evidence, expectation: Expectation, params: BranchParams) -> list[dict]:
    """Every amplitude span, and the first gate that rejected it."""
    out = []
    minimum_s = params.point("production_min_s")
    strength_min = params.point("voiced_strength_min")
    fraction_min = params.point("voiced_fraction_min")
    spread_window_s = params.point("f0_spread_window_s")
    spread_max = params.point("f0_spread_max_semitones")
    continuity_min = params.point("continuity_min")
    words = lexical(evidence.words)
    for span in amplitude_spans(evidence.spans):
        rec = {
            "dur_s": round(duration(span.extent), 3) if span.extent else None,
            "signal": span.attributes.get("signal"),
            "pof_db": span.attributes.get("peak_over_floor_db"),
        }
        if span.extent is None or duration(span.extent) < minimum_s:
            rec["rejected_by"] = "production_min_s"
            out.append(rec)
            continue
        if expectation.lexical_separator and any(overlaps(word_extent(w), span.extent) for w in words):
            rec["rejected_by"] = "lexical_separator"
            out.append(rec)
            continue
        if evidence.tracks is None:
            rec["rejected_by"] = "tracks_absent"
            out.append(rec)
            continue
        track = track_slice(evidence.tracks, span.extent, strength_min)
        if track.strength.size == 0:
            rec["rejected_by"] = "empty_track_slice"
            out.append(rec)
            continue
        vf = float(track.voiced.mean())
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        spread = float("nan") if spread_window_s is None else max_windowed_spread(pitch, track.hop_s, spread_window_s)
        trace = np.empty(0) if evidence.continuity is None else trace_slice(evidence.continuity, span.extent)
        stat = float(np.median(trace)) if trace.size else 0.0
        rec.update(
            voiced_fraction=round(vf, 4),
            f0_spread_semitones=None if not np.isfinite(spread) else round(float(spread), 3),
            stationarity=round(stat, 4),
            continuity_frames=int(trace.size),
        )
        if spread_window_s is not None and np.isfinite(spread):
            per = window_spreads(pitch, track.hop_s, spread_window_s)
            rec.update(
                windows_n=int(per.size),
                spread_median=round(float(np.median(per)), 3),
                spread_p95=round(float(np.percentile(per, 95.0)), 3),
                windows_over_max=int((per > (spread_max or np.inf)).sum()),
            )
        if fraction_min is not None and vf < fraction_min:
            rec["rejected_by"] = "voiced_fraction_min"
        elif spread_max is not None and spread_window_s is not None and not (spread <= spread_max):
            rec["rejected_by"] = "f0_spread_max_semitones"
        elif continuity_min is not None and stat < continuity_min:
            rec["rejected_by"] = "continuity_min"
        else:
            rec["rejected_by"] = None
        out.append(rec)
    return out


def glide_census(evidence: V.Evidence, params: BranchParams) -> list[dict]:
    """Why no sweep was found, span by span, for the GLIDE pattern."""
    out = []
    minimum_s = params.point("production_min_s")
    strength_min = params.point("voiced_strength_min")
    tolerance = params.point("monotone_tolerance_semitones")
    fraction_min = params.point("voiced_fraction_min")
    dominant_min = params.point("dominant_segment_min_fraction")
    for span in amplitude_spans(evidence.spans):
        rec = {"dur_s": round(duration(span.extent), 3) if span.extent else None}
        if span.extent is None or duration(span.extent) < minimum_s:
            rec["rejected_by"] = "production_min_s"
            out.append(rec)
            continue
        if evidence.tracks is None:
            rec["rejected_by"] = "tracks_absent"
            out.append(rec)
            continue
        track = track_slice(evidence.tracks, span.extent, strength_min)
        if track.strength.size == 0:
            rec["rejected_by"] = "empty_track_slice"
            out.append(rec)
            continue
        vf = float(track.voiced.mean())
        rec["voiced_fraction"] = round(vf, 4)
        if fraction_min is not None and vf < fraction_min:
            rec["rejected_by"] = "voiced_fraction_min"
            out.append(rec)
            continue
        pitch = semitones(np.where(track.voiced, track.f0_hz, np.nan))
        run = longest_monotone_run(pitch, tolerance)
        if run is None:
            rec["rejected_by"] = "longest_monotone_run_none"
            out.append(rec)
            continue
        first, last, sign = run
        sweep = (float(track.times_s[first]), float(track.times_s[last]) + track.hop_s)
        rec["sweep_s"] = round(duration(sweep), 3)
        rec["sweep_fraction"] = round(duration(sweep) / max(duration(span.extent), 1e-9), 4)
        rec["direction"] = "up" if sign > 0 else "down"
        if dominant_min is not None and rec["sweep_fraction"] < dominant_min:
            rec["rejected_by"] = "dominant_segment_min_fraction"
        else:
            rec["rejected_by"] = None
        out.append(rec)
    return out


def one(run_dir: Path, store_path: Path, family: str, stem: str) -> dict:
    """One recording's row: the fold's reading of VOICE, and the evidence behind it."""
    store = ProvStore.read_jsonl(store_path)
    row = {"stem_id": hashlib.sha1(stem.encode()).hexdigest()[:12], "family": family}

    ents = [e for e in store.entities("verdict") if not store.is_invalidated(e.id)]
    fv = next((e for e in ents if e.attributes.get("node") == "VERDICT"), None)
    if fv is not None:
        a = fv.attributes
        row.update(
            triage=a.get("outcome"),
            declared_family=a.get("declared_family"),
            routes=a.get("routes"),
            findings=a.get("findings"),
            hints=a.get("hints"),
            agreement=a.get("agreement"),
            conformance=a.get("conformance"),
            route_state=a.get("route_state"),
            reasons=[[r.get("node"), r.get("outcome"), r.get("why")] for r in (a.get("reasons") or [])],
        )
    reps = [e for e in store.entities("branch_report") if not store.is_invalidated(e.id)]
    vr = next((e for e in reps if e.attributes.get("node") == "VOICE"), None)
    if vr is not None:
        d = vr.attributes.get("detail") or vr.attributes
        row["voice_report"] = {
            k: d.get(k)
            for k in (
                "mode",
                "declared_task_family",
                "spans_n",
                "phonation_s",
                "longest_span_s",
                "roles",
                "phonation_tracks",
                "notes",
            )
        }
        row["voice_conformance"] = vr.attributes.get("conformance")
        row["voice_in_family"] = vr.attributes.get("in_family")
        row["voice_unmeasured"] = vr.attributes.get("unmeasured")
        row["voice_deviations"] = vr.attributes.get("deviations")

    tracks_m = find_measurement(store, V.PHONATION_TRACKS)
    if tracks_m is not None:
        row["tracks_params"] = {k: tracks_m.attributes.get(k) for k in ("hop_s", "f0_min_hz", "f0_max_hz", "f0_signal")}

    ev = V.read_evidence(store, run_dir)
    all_spans = ev.spans
    row["spans_by_measure"] = {
        m: len(spans_by_measure(all_spans, m)) for m in ("amplitude", "continuity", "asr", "gap")
    }
    row["amplitude_spans_n"] = len(amplitude_spans(all_spans))
    row["tracks_present"] = ev.tracks is not None
    row["continuity_present"] = ev.continuity is not None
    row["tracks_measurement"] = ev.tracks_id is not None
    row["continuity_measurement"] = ev.continuity_id is not None
    row["consensus_words_n"] = len(ev.words)
    row["lexical_words_n"] = len(lexical(ev.words))
    row["file_s"] = round(duration(ev.file_extent), 3) if ev.file_extent else None
    if ev.tracks is not None:
        vt = ev.tracks.strength >= (PARAMS.point("voiced_strength_min") or 0.45)
        row["file_voiced_fraction"] = round(float(vt.mean()), 4)
        row["track_frames"] = int(ev.tracks.times_s.size)
    if ev.continuity is not None:
        row["file_continuity_median"] = round(float(np.median(ev.continuity.continuity)), 4)

    exp = VOICE_EXPECTATIONS.get(family)
    row["in_family"] = exp is not None
    if exp is not None and exp.pattern is Pattern.GLIDE:
        row["census"] = glide_census(ev, PARAMS)
        row["pattern"] = "glide"
    else:
        row["census"] = gate_census(ev, exp or Expectation(pattern=Pattern.SUSTAINED), PARAMS)
        row["pattern"] = "sustained" if exp is not None else "detect"
    row["carriers_n"] = sum(1 for c in row["census"] if c.get("rejected_by") is None)
    return row


def discover(root: Path) -> list[tuple[Path, Path, str, str]]:
    """Every finished run under a driver output tree, read off the store rather than the row.

    The driver's row is not the sentinel here: a row can be marked failed by a post-run bookkeeping
    error while the graph itself completed and wrote its store, and the store is what this censuses.

    Args:
        root: The driver's output root, holding ``out/<sub>/<ses>/<stem>_<stamp>/run/store.jsonl``.

    Returns:
        One ``(run_dir, store_path, family, stem)`` per finished run, latest stamp per stem.
    """
    from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of

    latest: dict[str, tuple[Path, Path, str, str]] = {}
    for store_path in sorted(root.rglob("run/store.jsonl")):
        run_dir = store_path.parent
        stem = run_dir.parent.name.rsplit("_", 1)[0]
        latest[stem] = (run_dir, store_path, task_family(task_id_of(stem)), stem)
    return [latest[stem] for stem in sorted(latest)]


def main() -> int:
    """Walk a driver output tree, census each finished run, write one JSONL row per recording."""
    root, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    found = discover(root)
    with out_path.open("w") as sink:
        for i, (run_dir, store_path, family, stem) in enumerate(found, 1):
            try:
                rec = one(run_dir, store_path, family, stem)
                rec["ok"] = True
            except Exception as err:  # noqa: BLE001 — a probe failure is a row, never fatal
                rec = {"family": family, "ok": False, "probe_error": f"{type(err).__name__}: {err}"}
            sink.write(json.dumps(rec, default=str) + "\n")
            if i % 25 == 0:
                print(f"{i}/{len(found)}", flush=True)
    print(f"wrote {out_path} ({len(found)} runs)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
