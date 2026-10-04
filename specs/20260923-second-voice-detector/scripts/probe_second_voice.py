"""Diarizer-independent heterogeneity probe: is one recording's voice cloud one speaker or two?

Every window is exactly ``--window-s`` long, so the batch handed to the embedding model is
duration-homogeneous and the duration confound recorded in
``specs/20260922-the-multi-speaker-instrument/speaker-verification-assessment.md`` cannot
express itself as a split.

Emits counts, seconds and cosines. No transcript text, no waveform, no embedding vector
leaves this script.
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


def _device(name: Optional[str]) -> Any:  # noqa: ANN401 — DeviceType imported lazily
    from senselab.utils.data_structures import DeviceType

    if name is None:
        return None
    return {"cpu": DeviceType.CPU, "cuda": DeviceType.CUDA, "mps": DeviceType.MPS}[name]


def load_audio(path: Path, target_sr: int = 16000) -> Any:  # noqa: ANN401
    """Mono, 16 kHz ``Audio`` for ``path``."""
    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    audio = Audio(filepath=str(path))
    if audio.waveform.shape[0] > 1:
        audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != target_sr:
        audio = resample_audios([audio], target_sr)[0]
    return audio


def pyannote_baseline(audio: Any, device: Any) -> dict[str, Any]:  # noqa: ANN401
    """Run the incumbent diarizer exactly as PREPROCESS does, and report what it said."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios
    from senselab.utils.data_structures import PyannoteAudioModel
    from senselab.utils.model_revision import resolve_revision

    sha = resolve_revision(PYANNOTE, "main")
    t0 = time.time()
    lines = diarize_audios(
        [audio],
        model=PyannoteAudioModel(path_or_uri=PYANNOTE, revision=sha),
        device=device,
        min_speakers=None,
        max_speakers=None,
        exclusive=False,
    )[0]
    wall = time.time() - t0
    segs = [(float(ln.start), float(ln.end), str(ln.speaker)) for ln in lines]
    labels = sorted({s for _, _, s in segs})
    return {
        "model": PYANNOTE,
        "revision": sha,
        "wall_s": wall,
        "n_segments": len(segs),
        "n_speakers": len(labels),
        "labels": labels,
        "speech_s": float(sum(e - s for s, e, _ in segs)),
        "segments": [[round(s, 3), round(e, 3), lab] for s, e, lab in segs],
    }


def speech_mask(spans: list[tuple[float, float]], starts: list[float], window_s: float, min_cover: float) -> list[bool]:
    """Which windows carry at least ``min_cover`` of their length inside a speech span."""
    out: list[bool] = []
    for a in starts:
        b = a + window_s
        cover = 0.0
        for s, e in spans:
            lo, hi = max(a, s), min(b, e)
            if hi > lo:
                cover += hi - lo
        out.append(cover >= min_cover * window_s)
    return out


def _rms(audio: Any, start_s: float, end_s: float) -> float:  # noqa: ANN401
    """Root-mean-square level of one window, as a split-cause diagnostic."""
    sr = audio.sampling_rate
    w = audio.waveform[0, int(start_s * sr) : int(end_s * sr)]
    arr = np.asarray(w, dtype=float)
    return float(np.sqrt(np.mean(arr**2))) if arr.size else 0.0


def _relabel(vectors: np.ndarray, linkage_method: str, cut_theta: float) -> np.ndarray:
    """Cluster membership under the cut ``select_dominant_vectors`` chose.

    That function reports the rule and the group sizes but not which row went where, and the
    split-cause diagnostics below need membership. Re-cutting the same linkage at the same
    angular height reproduces its partition rather than inventing a second one.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    x = vectors / np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12, None)
    theta = np.arccos(np.clip(x @ x.T, -1.0, 1.0))
    np.fill_diagonal(theta, 0.0)
    z = linkage(squareform(theta, checks=False), method=linkage_method)
    return np.asarray(fcluster(z, t=cut_theta, criterion="distance"))


def heterogeneity(
    audio: Any,  # noqa: ANN401
    spans: list[tuple[float, float]],
    *,
    window_s: float,
    hop_s: float,
    min_cover: float,
    device: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Cluster the recording's own window embeddings and report whether they split."""
    from senselab.audio.tasks.speaker_embeddings.windowing import extract_per_window_embeddings, window_starts
    from senselab.utils.model_revision import resolve_revision
    from senselab.utils.tasks.embedding_distribution import describe_embedding_distribution, select_dominant_vectors

    sha = resolve_revision(ECAPA, "main")
    duration_s = audio.waveform.shape[-1] / audio.sampling_rate
    starts = window_starts(duration_s, window_s, hop_s)
    keep = speech_mask(spans, starts, window_s, min_cover) if spans else [True] * len(starts)

    t0 = time.time()
    failures: dict[str, str] = {}
    per_model = extract_per_window_embeddings(
        audio=audio,
        models=[ECAPA],
        window_s=window_s,
        hop_s=hop_s,
        device=device,
        failures=failures,
        revision=sha,
    )
    wall = time.time() - t0
    entries = per_model.get(ECAPA, [])
    if not entries:
        return {"model": ECAPA, "revision": sha, "wall_s": wall, "error": failures.get(ECAPA, "no windows")}

    idx = [i for i, k in enumerate(keep) if k and i < len(entries)]
    vecs = np.vstack([entries[i].vector for i in idx]) if idx else np.zeros((0, 1))
    out: dict[str, Any] = {
        "model": ECAPA,
        "revision": sha,
        "wall_s": wall,
        "window_s": window_s,
        "hop_s": hop_s,
        "min_cover": min_cover,
        "n_windows_total": len(entries),
        "n_windows_speech": int(len(idx)),
        "duration_s": duration_s,
    }
    if len(idx) < 4:
        out["decision"] = "too_few_windows"
        return out

    sel = select_dominant_vectors(vecs)
    rule = sel.rule_used
    summaries = [c.model_dump() for c in sel.clusters]
    out["ahc"] = {
        "n_clusters": len(summaries),
        "rule": rule.model_dump(),
        "clusters": summaries,
        "dominant_cluster_id": sel.dominant_cluster_id,
        "runner_up_cluster_id": sel.runner_up_cluster_id,
        "cos_dominant_to_runner_up": sel.cos_dominant_to_runner_up,
        "n_dropped": len(sel.dropped_indices),
    }
    labels = _relabel(vecs, rule.linkage, float(rule.cut_theta))

    # Split-cause diagnostics: a split that tracks level or clock time is not a speaker split.
    if labels.size == len(idx) and len(summaries) >= 2:
        rms = np.array([_rms(audio, entries[i].start_s, entries[i].end_s) for i in idx])
        mid = np.array([0.5 * (entries[i].start_s + entries[i].end_s) for i in idx])
        per = {}
        for lab in sorted(set(labels.tolist())):
            m = labels == lab
            per[str(lab)] = {
                "n": int(m.sum()),
                "rms_mean": float(rms[m].mean()),
                "time_mean_s": float(mid[m].mean()),
                "time_min_s": float(mid[m].min()),
                "time_max_s": float(mid[m].max()),
            }
        out["ahc"]["per_cluster_diagnostics"] = per

    _, dist = describe_embedding_distribution(
        vecs,
        aggregator="spherical_mean",
        window_s=window_s,
        hop_s=hop_s,
        window_starts_s=[entries[i].start_s for i in idx],
    )
    d = dist.model_dump() if hasattr(dist, "model_dump") else dict(dist)
    out["distribution"] = {
        "similarity": d.get("similarity"),
        "spectrum": d.get("spectrum"),
        "nulls": d.get("nulls"),
        "centroid_robustness": d.get("centroid_robustness"),
    }

    try:
        from senselab.audio.workflows.audio_analysis.embeddings import cluster_pass_speakers

        spectral = cluster_pass_speakers(
            [entries[i] for i in range(len(entries))],
            is_speech_per_window=keep,
        )
        if spectral is not None:
            out["spectral"] = {
                "n_speakers": spectral.get("n_speakers"),
                "best_silhouette": spectral.get("best_silhouette"),
                "empirical_same_speaker_floor": spectral.get("empirical_same_speaker_floor"),
                "empirical_diff_speaker_floor": spectral.get("empirical_diff_speaker_floor"),
            }
    except Exception as exc:  # noqa: BLE001
        out["spectral"] = {"error": repr(exc)[:200]}

    return out


def probe(path: Path, *, device_name: Optional[str], window_s: float, hop_s: float, min_cover: float) -> dict[str, Any]:
    """Baseline diarization plus the embedding heterogeneity reading for one file."""
    dev = _device(device_name)
    row: dict[str, Any] = {"audio": str(path)}
    audio = load_audio(path)
    row["duration_s"] = audio.waveform.shape[-1] / audio.sampling_rate
    try:
        base = pyannote_baseline(audio, dev)
        row["pyannote"] = base
        spans = [(s, e) for s, e, _ in base["segments"]]
    except Exception as exc:  # noqa: BLE001
        row["pyannote"] = {"error": repr(exc)[:300]}
        spans = []
    try:
        row["embedding"] = heterogeneity(audio, spans, window_s=window_s, hop_s=hop_s, min_cover=min_cover, device=dev)
    except Exception as exc:  # noqa: BLE001
        row["embedding"] = {"error": repr(exc)[:300], "traceback": traceback.format_exc()[-1200:]}
    return row


def main() -> int:
    """Probe every audio file named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--hop-s", type=float, default=1.0)
    ap.add_argument("--min-cover", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    fh = open(args.out, "w") if args.out else None
    for p in args.audio:
        row = probe(
            Path(p),
            device_name=args.device,
            window_s=args.window_s,
            hop_s=args.hop_s,
            min_cover=args.min_cover,
        )
        line = json.dumps(row)
        if fh:
            fh.write(line + "\n")
            fh.flush()
        print(line[:400], file=sys.stderr, flush=True)
    if fh:
        fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
