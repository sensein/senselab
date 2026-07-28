"""Windowed (temporal) feature extraction across senselab's feature backends.

Moved out of ``scripts/analyze_audio.py`` (T051): this is a model-touching
capability, not CLI plumbing, so it belongs beside the per-backend extractors —
the same reasoning that moved transcript fusion to ``speech_to_text_ensemble``
and WER to ``speech_to_text_evaluation``.

The public entry point is :func:`extract_temporal_features`. It deliberately
prefers each backend's *native* time grid over one imposed window size, because
openSMILE's ~10 ms LLD frames and torchaudio-squim's inherently global quality
scores are not comparable on a shared grid.
"""

from __future__ import annotations

import sys
from typing import Any

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.preprocessing import extract_segments
from senselab.utils.data_structures import DeviceType

__all__ = ["extract_temporal_features"]


def _flatten_feature_dict(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested feature dict into a single-row dict suitable for a parquet column.

    Keys are joined with ``.``; tensors are coerced to floats (mean of
    last axis when 1-D) or skipped when high-dimensional (we don't want
    per-window MFCC tensors as parquet cells — caller can opt back in
    via the JSON sibling). Lists of scalars are kept as-is so pyarrow
    can store them as a list column.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_feature_dict(v, prefix=f"{key}."))
            continue
        if hasattr(v, "ndim") and hasattr(v, "tolist"):  # torch.Tensor / np.ndarray
            try:
                if v.ndim == 0:
                    out[key] = float(v.item())
                elif v.ndim == 1 and v.shape[0] <= 64:
                    out[key] = [float(x) for x in v.tolist()]
                else:
                    # multi-dim tensor (spectrogram, mfcc) — store mean as a scalar
                    # summary; full tensor stays in the JSON sibling for callers
                    # that want it.
                    out[f"{key}.mean"] = float(v.mean().item())
            except Exception:  # noqa: BLE001 — best effort
                pass
            continue
        if isinstance(v, (int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, str):
            out[key] = v
        # silently drop anything else (callable, opaque object) to keep the row clean
    return out


def extract_temporal_features(
    audio: Audio,
    *,
    win_length: float,
    hop_length: float,
    device: DeviceType | None,
) -> dict[str, list[dict[str, Any]]]:
    """Extract per-backend temporal features, preferring each backend's native time grid.

    - **opensmile**: uses ``LowLevelDescriptors`` (native ~10 ms frame
      grid). One row per opensmile frame.
    - **parselmouth**: aggregates over a sliding window since the
      senselab wrapper currently only exposes the summary form.
    - **torchaudio_squim**: STOI/PESQ/SI-SDR are inherently global
      quality scores — windowed externally so the resulting time series
      is comparable to the rest.

    Returns a dict ``{backend: [rows...]}`` so each backend can be
    written to its own parquet sidecar (different columns + time grids
    don't share a schema).
    """
    duration_s = float(audio.waveform.shape[1]) / float(audio.sampling_rate)
    out: dict[str, list[dict[str, Any]]] = {"opensmile": [], "parselmouth": [], "torchaudio_squim": []}

    # opensmile LLD — native windowing (DataFrame indexed by [start, end]).
    try:
        import opensmile as _os

        smile = _os.Smile(
            feature_set=_os.FeatureSet.eGeMAPSv02,
            feature_level=_os.FeatureLevel.LowLevelDescriptors,
        )
        df = smile.process_signal(audio.waveform.squeeze().numpy(), audio.sampling_rate)
        df = df.reset_index()
        df["start"] = df["start"].dt.total_seconds()
        df["end"] = df["end"].dt.total_seconds()
        out["opensmile"] = df.to_dict(orient="records")
    except Exception as exc:  # noqa: BLE001
        print(f"  [features.opensmile] warn: {exc!r}", file=sys.stderr)

    # External 1 s / 0.5 s loop for the summary-style backends.
    t = 0.0
    idx = 0
    while t + win_length <= duration_s + 1e-6:
        start = round(t, 4)
        end = round(min(t + win_length, duration_s), 4)
        clip = extract_segments([(audio, [(start, end)])])[0][0]
        try:
            pm = extract_features_from_audios(
                [clip], opensmile=False, parselmouth=True, torchaudio=False, torchaudio_squim=False, device=device
            )[0]
            row = _flatten_feature_dict(pm.get("praat_parselmouth", {}))
            row.update({"start": start, "end": end, "win_index": idx})
            out["parselmouth"].append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [features.parselmouth win {idx}] warn: {exc!r}", file=sys.stderr)
        try:
            sq = extract_features_from_audios(
                [clip], opensmile=False, parselmouth=False, torchaudio=False, torchaudio_squim=True, device=device
            )[0]
            row = _flatten_feature_dict(sq.get("torchaudio_squim", {}))
            row.update({"start": start, "end": end, "win_index": idx})
            out["torchaudio_squim"].append(row)
        except Exception as exc:  # noqa: BLE001
            print(f"  [features.torchaudio_squim win {idx}] warn: {exc!r}", file=sys.stderr)
        t += hop_length
        idx += 1

    return out
