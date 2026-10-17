"""Compare the ways of getting one channel per speaker from a mono mix, by what VAP makes of them.

For each synthetic dialogue made by ``make_dialogue.py`` (clean per-speaker channels + their mix),
run VAP + backchannel prediction on:

- ``clean``          the true per-speaker channels (the reference),
- ``masking/<diar>`` the mix under each speaker's smoothed diarization mask,
- ``mossformer2/<diar>`` MossFormer2_SS_16K separation, windows assigned to speakers by ``<diar>``,
- ``duplicate``      the mix in both channels (what VAP sees with no speaker information at all),

where ``<diar>`` is ``oracle`` (the generator's IPUs) or ``pyannote`` (speaker-diarization-community-1
on the mix, two speakers, labels mapped to A/B by overlap with the oracle). Each input is compared
with ``clean`` frame by frame and by turn-shift events, and every input's hold/shift and shift
projectability readings are scored against the generator's truth.

Usage (one benchmark process, three threads)::

    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 uv run python evaluate_inputs.py DATA_DIR OUT_DIR [--seeds 0 1 2]

Timings are only meaningful on an idle machine; each row records the 1-minute load average.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.turn_taking_projection import project_voice_activity, separate_dyad, summarize_projection
from senselab.audio.tasks.turn_taking_projection.result import VAPResult
from senselab.utils.data_structures import ScriptLine
from senselab.utils.venv_worker import shutdown_venv_workers

TOLERANCE_S = 0.5
HYSTERESIS_FRAMES = 2


def lines(rows: List[dict]) -> List[ScriptLine]:
    """Truth rows to ScriptLines."""
    return [ScriptLine(speaker=r["speaker"], start=r["start"], end=r["end"]) for r in rows]


def pyannote_diarization(mix: Audio, oracle: List[ScriptLine]) -> List[ScriptLine]:
    """community-1 on the mix with two speakers, labels renamed to the oracle's by overlap."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios
    from senselab.utils.data_structures import PyannoteAudioModel

    model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")
    [segments] = diarize_audios([mix], model=model, num_speakers=2, exclusive=False)
    labels = sorted({s.speaker for s in segments if s.speaker})
    overlap: Dict[Tuple[str, str], float] = {}
    for s in segments:
        for o in oracle:
            ov = min(s.end, o.end) - max(s.start, o.start)
            if ov > 0:
                overlap[(s.speaker, o.speaker)] = overlap.get((s.speaker, o.speaker), 0.0) + ov
    best = max(
        ({labels[0]: "A", labels[1]: "B"}, {labels[0]: "B", labels[1]: "A"}),
        key=lambda m: sum(overlap.get((k, v), 0.0) for k, v in m.items()),
    )
    return [ScriptLine(speaker=best.get(s.speaker, s.speaker), start=s.start, end=s.end) for s in segments]


def shift_events(p_now_a: np.ndarray, times: np.ndarray) -> List[Tuple[float, str]]:
    """Times where the projected next speaker changes, with hysteresis."""
    state = "A" if p_now_a[0] >= 0.5 else "B"
    run = 0
    events = []
    for t, p in zip(times, p_now_a):
        leader = "A" if p >= 0.5 else "B"
        run = run + 1 if leader != state else 0
        if run >= HYSTERESIS_FRAMES:
            state = leader
            run = 0
            events.append((float(t), state))
    return events


def event_f1(ref: List[Tuple[float, str]], hyp: List[Tuple[float, str]]) -> Dict[str, float]:
    """One-to-one greedy matching of same-direction events within TOLERANCE_S."""
    used = set()
    hits = 0
    for t, d in hyp:
        best = None
        for k, (rt, rd) in enumerate(ref):
            if k in used or rd != d or abs(rt - t) > TOLERANCE_S:
                continue
            if best is None or abs(rt - t) < abs(ref[best][0] - t):
                best = k
        if best is not None:
            used.add(best)
            hits += 1
    precision = hits / len(hyp) if hyp else 0.0
    recall = hits / len(ref) if ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "n_ref": len(ref), "n_hyp": len(hyp)}


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, NaN when either is constant."""
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare(ref: VAPResult, hyp: VAPResult) -> Dict[str, float]:
    """Frame-level agreement of ``hyp`` with ``ref``."""
    n = min(len(ref.frames), len(hyp.frames))
    r, h = ref.frames.iloc[:n], hyp.frames.iloc[:n]
    out = {
        "p_now_corr": corr(r.p_now_a.to_numpy(), h.p_now_a.to_numpy()),
        "p_now_mae": float(np.mean(np.abs(r.p_now_a.to_numpy() - h.p_now_a.to_numpy()))),
        "p_future_corr": corr(r.p_future_a.to_numpy(), h.p_future_a.to_numpy()),
        "vad_a_corr": corr(r.vad_a.to_numpy(), h.vad_a.to_numpy()),
        "vad_b_corr": corr(r.vad_b.to_numpy(), h.vad_b.to_numpy()),
    }
    if ref.has_backchannel and hyp.has_backchannel:
        out["p_bc_a_corr"] = corr(r.p_bc_a.to_numpy(), h.p_bc_a.to_numpy())
        out["p_bc_b_corr"] = corr(r.p_bc_b.to_numpy(), h.p_bc_b.to_numpy())
    times = r.time.to_numpy()
    f1 = event_f1(shift_events(r.p_now_a.to_numpy(), times), shift_events(h.p_now_a.to_numpy(), times))
    out.update({f"shift_event_{k}": v for k, v in f1.items()})
    return out


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=["masking", "mossformer2"])
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = out_dir / "report.json"
    report: Dict[str, dict] = json.loads(previous.read_text()) if previous.is_file() else {}
    for seed in args.seeds:
        d = Path(args.data) / f"dialogue_seed{seed}"
        truth = json.loads((d / "truth.json").read_text())
        mix = Audio(filepath=str(d / "mix.wav"))
        clean = (Audio(filepath=str(d / "a.wav")), Audio(filepath=str(d / "b.wav")))
        oracle = lines(truth["ipus"])
        turns, ipus, bcs = lines(truth["turns"]), lines(truth["ipus"]), lines(truth["backchannels"])
        cached = out_dir / f"seed{seed}_pyannote_diarization.json"
        if cached.is_file():
            stored = json.loads(cached.read_text())
            predicted, diar_s = lines(stored["segments"]), stored["seconds"]
        else:
            began = time.monotonic()
            predicted = pyannote_diarization(mix, oracle)
            diar_s = time.monotonic() - began
            segments = [{"speaker": x.speaker, "start": x.start, "end": x.end} for x in predicted]
            cached.write_text(json.dumps({"seconds": diar_s, "segments": segments}))
        diar = {"oracle": oracle, "pyannote": predicted}

        inputs: Dict[str, Tuple[Audio, Audio]] = {"clean": clean}
        sep_time: Dict[str, float] = {"pyannote_diarization": diar_s}
        for method in args.methods:
            for name, segments in diar.items():
                began = time.monotonic()
                inputs[f"{method}/{name}"] = separate_dyad(mix, segments, method=method, speakers=("A", "B"))
                sep_time[f"{method}/{name}"] = time.monotonic() - began
            shutdown_venv_workers()
        dup = Audio(waveform=mix.waveform.repeat(2, 1), sampling_rate=mix.sampling_rate)
        inputs["duplicate"] = (
            Audio(waveform=dup.waveform[:1], sampling_rate=16000),
            Audio(waveform=dup.waveform[1:], sampling_rate=16000),
        )

        results: Dict[str, VAPResult] = {}
        rows: Dict[str, dict] = {}
        for name, pair in inputs.items():
            vap = project_voice_activity(pair, speakers=("A", "B"))
            results[name] = vap
            vap.frames.to_csv(out_dir / f"seed{seed}_{name.replace('/', '_')}.csv.gz", index=False)
            summary = summarize_projection(vap, turns, ipus, bcs)
            rows[name] = {
                "timing": vap.timing,
                "load_average_1m": os.getloadavg()[0],
                "separation_s": sep_time.get(name),
                "holdshift": summary["overall"]["pauses"],
                "shift_projectability": {
                    k: summary["overall"][k] for k in ("n_shifts", "mean_p_now_next", "projected_rate")
                },
                "backchannel": {s: summary["speakers"][s]["backchannel"] for s in ("A", "B")},
            }
            print(seed, name, json.dumps(rows[name]["holdshift"]), flush=True)
        for name in inputs:
            if name != "clean":
                rows[name]["vs_clean"] = compare(results["clean"], results[name])
                print(seed, name, json.dumps(rows[name]["vs_clean"]), flush=True)
        report[f"seed{seed}"] = {"duration_s": truth["duration"], "inputs": rows, "separation_timing": sep_time}
        (out_dir / "report.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    main()
