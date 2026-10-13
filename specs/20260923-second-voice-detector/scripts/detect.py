"""Score every candidate two-way split of one recording, so an operating point can be fitted.

Nothing here decides. Each row carries, for one recording:

* what the incumbent diarizer said when free, and what partition it produces when its count
  is pinned to two (it is the only backend that honours the hint);
* a fixed 2.0 s / 1.0 s window grid over its speech — uniform by construction, so the
  duration contrast that dominates variable-length spans cannot express itself as a split;
* for each candidate partition, the cosine silhouette, the minority share, the centroid
  cosine and the within-group cosines.

The arms of the calibration sample supply the labels these scores get fitted against; this
script never sees them. Counts, seconds and cosines only — no transcript text.
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
WINDOW_S = 2.0
HOP_S = 1.0
MIN_COVER = 0.5
MIN_WINDOWS = 6


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
    per: dict[str, float] = {}
    for s, e, lab in segs:
        per[lab] = per.get(lab, 0.0) + (e - s)
    return {
        "revision": sha,
        "wall_s": round(time.time() - t0, 2),
        "n_segments": len(segs),
        "n_speakers": len(per),
        "per_speaker_s": {k: round(v, 3) for k, v in sorted(per.items())},
        "segments": [[round(s, 3), round(e, 3), lab] for s, e, lab in segs],
    }


def _keep(spans: list[tuple[float, float]], starts: list[float]) -> list[bool]:
    """Windows with at least ``MIN_COVER`` of their length inside a speech span."""
    return [sum(max(0.0, min(a + WINDOW_S, e) - max(a, s)) for s, e in spans) >= MIN_COVER * WINDOW_S for a in starts]


def _label_from_segments(mid: np.ndarray, segments: list[list[Any]]) -> Optional[np.ndarray]:
    """Assign each window to the diarized speaker holding its midpoint, or None if any is unheld."""
    labs: list[str] = []
    for t in mid:
        hit = next((lab for s, e, lab in segments if s <= t < e), None)
        if hit is None:
            return None
        labs.append(hit)
    uniq = sorted(set(labs))
    if len(uniq) != 2:
        return None
    return np.array([uniq.index(x) for x in labs])


def _score(x: np.ndarray, lab: np.ndarray) -> dict[str, Any]:
    """Silhouette and geometry of one two-way labelling of the unit-normalised window cloud."""
    from sklearn.metrics import silhouette_score

    sizes = np.bincount(lab, minlength=2)
    if sizes.min() < 1 or len(set(lab.tolist())) < 2:
        return {"valid": False}

    def unit(v: np.ndarray) -> np.ndarray:
        m = v.mean(axis=0)
        return m / max(float(np.linalg.norm(m)), 1e-12)

    c0, c1 = unit(x[lab == 0]), unit(x[lab == 1])
    w0 = float(np.mean(x[lab == 0] @ c0))
    w1 = float(np.mean(x[lab == 1] @ c1))
    return {
        "valid": True,
        "silhouette": round(float(silhouette_score(x, lab, metric="cosine")), 4),
        "sizes": sizes.tolist(),
        "minority_share": round(float(sizes.min() / sizes.sum()), 4),
        "cos_between_centroids": round(float(c0 @ c1), 4),
        "within_cos_min": round(min(w0, w1), 4),
        "separation_margin": round(min(w0, w1) - float(c0 @ c1), 4),
    }


def detect(path: Path, *, device_name: Optional[str]) -> dict[str, Any]:
    """Every candidate two-way split of one recording, scored."""
    from senselab.audio.tasks.speaker_embeddings.windowing import extract_per_window_embeddings, window_starts
    from senselab.utils.model_revision import resolve_revision

    dev = _device(device_name)
    audio = load_audio(path)
    dur = audio.waveform.shape[-1] / audio.sampling_rate
    row: dict[str, Any] = {"audio": str(path), "name": path.name, "duration_s": round(dur, 3)}

    free = pyannote_run(audio, dev, None)
    row["pyannote_free"] = {k: v for k, v in free.items() if k != "segments"}
    spans = [(s, e) for s, e, _ in free["segments"]]
    row["speech_s"] = round(sum(e - s for s, e in spans), 3)

    k2 = pyannote_run(audio, dev, 2)
    row["pyannote_k2"] = {k: v for k, v in k2.items() if k != "segments"}

    sha = resolve_revision(ECAPA, "main")
    t0 = time.time()
    entries = extract_per_window_embeddings(
        audio=audio, models=[ECAPA], window_s=WINDOW_S, hop_s=HOP_S, device=dev, revision=sha
    ).get(ECAPA, [])
    row["embed_wall_s"] = round(time.time() - t0, 2)
    starts = window_starts(dur, WINDOW_S, HOP_S)
    keep = _keep(spans, starts) if spans else [True] * len(entries)
    idx = [i for i, k in enumerate(keep) if k and i < len(entries)]
    row["n_speech_windows"] = len(idx)
    if len(idx) < MIN_WINDOWS:
        row["note"] = "too_few_windows"
        return row

    x = np.vstack([entries[i].vector for i in idx])
    x = x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
    mid = np.array([0.5 * (entries[i].start_s + entries[i].end_s) for i in idx])

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    from sklearn.cluster import KMeans, SpectralClustering

    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(theta, 0.0)
    cand: dict[str, np.ndarray] = {}
    for method in ("average", "complete", "ward"):
        z = linkage(squareform(theta, checks=False), method=method)
        cand[f"ahc_{method}"] = np.asarray(fcluster(z, t=2, criterion="maxclust")) - 1
    cand["kmeans"] = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(x)
    cand["spectral"] = SpectralClustering(n_clusters=2, affinity="precomputed", random_state=0, n_init=5).fit_predict(
        np.maximum(x @ x.T, 0.0)
    )
    pk2 = _label_from_segments(mid, k2["segments"])
    if pk2 is not None:
        cand["pyannote_k2_labels"] = pk2

    row["candidates"] = {name: _score(x, lab) for name, lab in cand.items()}

    # Where the best-separated candidate puts its minority, in seconds — the "where" the
    # owner's case needs, kept separate from the "whether" above.
    best = max(
        (n for n, s in row["candidates"].items() if s.get("valid")),
        key=lambda n: row["candidates"][n]["silhouette"],
        default=None,
    )
    if best:
        lab = cand[best]
        minor = int(np.argmin(np.bincount(lab, minlength=2)))
        sel = sorted(float(t) for t in mid[lab == minor])
        runs: list[list[float]] = []
        for t in sel:
            if runs and t - runs[-1][1] <= 1.5:
                runs[-1][1] = t
            else:
                runs.append([t, t])
        row["best_candidate"] = best
        row["best_minority_runs_s"] = [[round(a - WINDOW_S / 2, 2), round(b + WINDOW_S / 2, 2)] for a, b in runs]
    return row


def main() -> int:
    """Score every audio file named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    done = set()
    if Path(args.out).exists():
        with open(args.out) as fh:
            done = {json.loads(line)["audio"] for line in fh if line.strip()}
    with open(args.out, "a") as fh:
        for i, p in enumerate(args.audio, 1):
            if p in done:
                continue
            try:
                row = detect(Path(p), device_name=args.device)
            except Exception as exc:  # noqa: BLE001
                row = {"audio": p, "name": Path(p).name, "error": repr(exc)[:300], "tb": traceback.format_exc()[-800:]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            c = row.get("candidates", {})
            sil = {k: v.get("silhouette") for k, v in c.items() if v.get("valid")}
            print(
                f"[{i}/{len(args.audio)}] {Path(p).name[:60]} free_n={row.get('pyannote_free', {}).get('n_speakers')} "
                f"win={row.get('n_speech_windows')} sil={sil}",
                file=sys.stderr,
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
