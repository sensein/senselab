"""Force a two-way split and report how well it holds, instead of asking whether one happened.

``select_dominant_vectors``' merge-gap rule answers "did this cloud split on its own", tuned
for a false-split rate under 1 %. That is the wrong question for detection: it will refuse a
real but weak second voice. This script asks the detection question instead — cut the tree at
k=2 unconditionally, then report the separation statistics that a calibration sample can turn
into an operating point.

Also runs the incumbent diarizer with ``num_speakers=2``, which only it honours, so that
"pyannote cannot find a second speaker" and "pyannote was never asked for one" stay distinct.

Counts, seconds and cosines only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

import numpy as np

ECAPA = "speechbrain/spkrec-ecapa-voxceleb"
PYANNOTE = "pyannote/speaker-diarization-community-1"
GRIDS = ((2.0, 1.0), (1.0, 0.5), (0.5, 0.25))


def _device(name: Optional[str]) -> Any:  # noqa: ANN401
    from senselab.utils.data_structures import DeviceType

    return None if name is None else {"cpu": DeviceType.CPU, "cuda": DeviceType.CUDA, "mps": DeviceType.MPS}[name]


def load_audio(path: Path) -> Any:  # noqa: ANN401
    """Mono, 16 kHz ``Audio`` for ``path``."""
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    audio = Audio(filepath=str(path))
    if audio.waveform.shape[0] > 1:
        audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != 16000:
        audio = resample_audios([audio], 16000)[0]
    return audio


def pyannote_run(audio: Any, device: Any, num_speakers: Optional[int]) -> dict[str, Any]:  # noqa: ANN401
    """One diarizer run, optionally with the speaker count pinned."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios
    from senselab.utils.data_structures import PyannoteAudioModel
    from senselab.utils.model_revision import resolve_revision

    sha = resolve_revision(PYANNOTE, "main")
    t0 = time.time()
    lines = diarize_audios(
        [audio],
        model=PyannoteAudioModel(path_or_uri=PYANNOTE, revision=sha),
        device=device,
        num_speakers=num_speakers,
        exclusive=False,
    )[0]
    segs = [(float(x.start), float(x.end), str(x.speaker)) for x in lines]
    by_label: dict[str, float] = {}
    for s, e, lab in segs:
        by_label[lab] = by_label.get(lab, 0.0) + (e - s)
    return {
        "num_speakers_arg": num_speakers,
        "revision": sha,
        "wall_s": round(time.time() - t0, 2),
        "n_segments": len(segs),
        "n_speakers": len(by_label),
        "per_speaker_s": {k: round(v, 3) for k, v in sorted(by_label.items())},
        "segments": [[round(s, 3), round(e, 3), lab] for s, e, lab in segs],
    }


def _speech_keep(
    spans: list[tuple[float, float]], starts: list[float], window_s: float, min_cover: float
) -> list[bool]:
    """Windows with at least ``min_cover`` of their length inside a speech span."""
    out = []
    for a in starts:
        b = a + window_s
        cover = sum(max(0.0, min(b, e) - max(a, s)) for s, e in spans)
        out.append(cover >= min_cover * window_s)
    return out


def _runs(times: list[tuple[float, float]], gap_s: float) -> list[list[float]]:
    """Merge window spans into contiguous runs, bridging gaps up to ``gap_s``."""
    if not times:
        return []
    times = sorted(times)
    runs = [[times[0][0], times[0][1]]]
    for s, e in times[1:]:
        if s - runs[-1][1] <= gap_s:
            runs[-1][1] = max(runs[-1][1], e)
        else:
            runs.append([s, e])
    return [[round(a, 2), round(b, 2)] for a, b in runs]


def forced_split(vectors: np.ndarray, spans: list[tuple[float, float]]) -> dict[str, Any]:
    """Cut the average-linkage angular tree at k=2 and describe the two groups."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    x = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(theta, 0.0)
    z = linkage(squareform(theta, checks=False), method="average")
    lab = np.asarray(fcluster(z, t=2, criterion="maxclust"))

    groups = {}
    for k in sorted(set(lab.tolist())):
        m = lab == k
        c = x[m].mean(axis=0)
        c = c / max(np.linalg.norm(c), 1e-12)
        groups[k] = (int(m.sum()), c, m)
    ks = sorted(groups, key=lambda k: -groups[k][0])
    big, small = groups[ks[0]], groups[ks[1]]
    cos_between = float(np.dot(big[1], small[1]))
    within_big = float(np.mean(x[big[2]] @ big[1]))
    within_small = float(np.mean(x[small[2]] @ small[1]))
    minority_spans = [spans[i] for i in range(len(spans)) if small[2][i]]
    return {
        "n_major": big[0],
        "n_minor": small[0],
        "minority_share": round(small[0] / len(lab), 4),
        "cos_between_centroids": round(cos_between, 4),
        "within_major_cos": round(within_big, 4),
        "within_minor_cos": round(within_small, 4),
        # How much further apart the two groups are than the spread inside the larger one.
        "separation_margin": round(min(within_big, within_small) - cos_between, 4),
        "minority_runs": _runs(minority_spans, gap_s=1.0),
        "minority_total_s": round(sum(e - s for s, e in minority_spans), 2),
    }


def probe(path: Path, *, device_name: Optional[str], min_cover: float, with_hint: bool) -> dict[str, Any]:
    """Baseline + hinted diarization, then a forced two-way split on three window grids."""
    from senselab.audio.tasks.speaker_embeddings.windowing import extract_per_window_embeddings, window_starts
    from senselab.utils.model_revision import resolve_revision

    dev = _device(device_name)
    audio = load_audio(path)
    row: dict[str, Any] = {"audio": str(path), "duration_s": round(audio.waveform.shape[-1] / audio.sampling_rate, 3)}

    try:
        base = pyannote_run(audio, dev, None)
        row["pyannote_free"] = base
        spans = [(s, e) for s, e, _ in base["segments"]]
    except Exception as exc:  # noqa: BLE001
        row["pyannote_free"] = {"error": repr(exc)[:300]}
        spans = []
    if with_hint:
        try:
            row["pyannote_k2"] = pyannote_run(audio, dev, 2)
        except Exception as exc:  # noqa: BLE001
            row["pyannote_k2"] = {"error": repr(exc)[:300]}

    sha = resolve_revision(ECAPA, "main")
    duration_s = audio.waveform.shape[-1] / audio.sampling_rate
    row["grids"] = {}
    for window_s, hop_s in GRIDS:
        key = f"{window_s}/{hop_s}"
        try:
            t0 = time.time()
            entries = extract_per_window_embeddings(
                audio=audio, models=[ECAPA], window_s=window_s, hop_s=hop_s, device=dev, revision=sha
            ).get(ECAPA, [])
            wall = round(time.time() - t0, 2)
            starts = window_starts(duration_s, window_s, hop_s)
            keep = _speech_keep(spans, starts, window_s, min_cover) if spans else [True] * len(entries)
            idx = [i for i, k in enumerate(keep) if k and i < len(entries)]
            if len(idx) < 4:
                row["grids"][key] = {"wall_s": wall, "n_speech_windows": len(idx), "note": "too_few_windows"}
                continue
            vecs = np.vstack([entries[i].vector for i in idx])
            win_spans = [(entries[i].start_s, entries[i].end_s) for i in idx]
            out = forced_split(vecs, win_spans)
            out["wall_s"] = wall
            out["n_speech_windows"] = len(idx)
            row["grids"][key] = out
        except Exception as exc:  # noqa: BLE001
            row["grids"][key] = {"error": repr(exc)[:300], "traceback": traceback.format_exc()[-800:]}
    return row


def main() -> int:
    """Probe every file named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--no-hint", action="store_true", help="skip the num_speakers=2 pyannote run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.out, "w") as fh:
        for p in args.audio:
            row = probe(Path(p), device_name=args.device, min_cover=args.min_cover, with_hint=not args.no_hint)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            g = row.get("grids", {})
            brief = {k: (v.get("cos_between_centroids"), v.get("minority_share")) for k, v in g.items()}
            print(f"{Path(p).name}: free={row.get('pyannote_free', {}).get('n_speakers')} {brief}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
