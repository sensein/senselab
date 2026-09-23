"""Run each diarization backend over one or more recordings and record what it counted.

A structured sibling of ``scripts/run_diarization_backends.py``: that one prints for a human
and includes the transcript the two joint ASR+diarization backends return. This one writes
JSON for a table and **drops every text field** — the corpus this runs on is human-subject
audio and no transcript may reach an artifact.

Sortformer is included here and is absent from the shipped script's ``BACKENDS``.
Every model is loaded at a resolved commit SHA, never a ref.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional, Sequence

TARGET_SR = 16000

# (key, model_id, kind) — `kind` picks the SenselabModel wrapper `diarize_audios` dispatches on.
BACKENDS: Sequence[tuple[str, str, str]] = (
    ("pyannote", "pyannote/speaker-diarization-community-1", "pyannote"),
    ("sortformer", "nvidia/diar_sortformer_4spk-v1", "hf"),
    ("vibevoice", "microsoft/VibeVoice-ASR-HF", "hf"),
    ("moss", "OpenMOSS-Team/MOSS-Transcribe-Diarize", "hf"),
    ("diarizen", "BUT-FIT/diarizen-wavlm-large-s80-md", "hf"),
    ("child_adult", "AlexXu811/whisper-child-adult", "hf"),
)


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
    if audio.sampling_rate != TARGET_SR:
        audio = resample_audios([audio], TARGET_SR)[0]
    return audio


def _overlap_s(segs: list[tuple[float, float, str]]) -> float:
    """Seconds where two distinct labels are simultaneously active."""
    if len(segs) < 2:
        return 0.0
    import numpy as np

    edges = np.unique(np.array([t for s, e, _ in segs for t in (s, e)]))
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mid = 0.5 * (lo + hi)
        if len({lab for s, e, lab in segs if s <= mid < e}) >= 2:
            total += float(hi - lo)
    return total


def run_one(key: str, model_id: str, kind: str, audio: Any, device: Any) -> dict[str, Any]:  # noqa: ANN401
    """One backend on one recording. A refusal is a result, not a crash."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios
    from senselab.utils.data_structures import HFModel, PyannoteAudioModel
    from senselab.utils.model_revision import resolve_revision

    row: dict[str, Any] = {"backend": key, "model": model_id}
    try:
        sha = resolve_revision(model_id, "main")
        row["revision"] = sha
        model = (
            PyannoteAudioModel(path_or_uri=model_id, revision=sha)
            if kind == "pyannote"
            else HFModel(path_or_uri=model_id, revision=sha)
        )
        t0 = time.time()
        kwargs: dict[str, Any] = {"device": device}
        if kind == "pyannote":
            kwargs["exclusive"] = False
        lines = diarize_audios([audio], model=model, **kwargs)[0]
        row["wall_s"] = round(time.time() - t0, 2)
        segs = [
            (float(ln.start), float(ln.end), str(getattr(ln, "speaker", None) or "-"))
            for ln in lines
            if getattr(ln, "start", None) is not None and getattr(ln, "end", None) is not None
        ]
        labels = sorted({lab for _, _, lab in segs})
        row.update(
            {
                "status": "ok",
                "n_segments": len(segs),
                "n_speakers": len(labels),
                "labels": labels,
                "speech_s": round(sum(e - s for s, e, _ in segs), 3),
                "overlap_s": round(_overlap_s(segs), 3),
                # Timings and labels only. Any `text` the backend returned is discarded here.
                "segments": [[round(s, 3), round(e, 3), lab] for s, e, lab in segs],
            }
        )
    except Exception as exc:  # noqa: BLE001
        row.update({"status": "failed", "error": repr(exc)[:400], "traceback": traceback.format_exc()[-1500:]})
    return row


def main() -> int:
    """Run the selected backends over every audio file named on the command line."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--only", nargs="*", default=None, help="backend keys to run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = _device(args.device)
    chosen = [b for b in BACKENDS if args.only is None or b[0] in args.only]
    with open(args.out, "w") as fh:
        for p in args.audio:
            audio = load_audio(Path(p))
            for key, model_id, kind in chosen:
                row = run_one(key, model_id, kind, audio, device)
                row["audio"] = str(p)
                row["duration_s"] = round(audio.waveform.shape[-1] / audio.sampling_rate, 3)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                print(
                    f"{Path(p).name} {key}: {row.get('status')} "
                    f"n_speakers={row.get('n_speakers')} n_seg={row.get('n_segments')} "
                    f"wall={row.get('wall_s')} {str(row.get('error', ''))[:160]}",
                    file=sys.stderr,
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
