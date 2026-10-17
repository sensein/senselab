"""Run one backend on one device over a test recording and write its outputs and timings to JSON.

One process per (backend, device), so only one model is resident at a time. ``--compare`` reads two
such files and reports agreement.

Example:
    uv run python scripts/benchmark_device_parity.py pyannote cpu --repeat 8 --out /tmp/p_cpu.json
    uv run python scripts/benchmark_device_parity.py pyannote mps --repeat 8 --out /tmp/p_mps.json
    uv run python scripts/benchmark_device_parity.py --compare /tmp/p_cpu.json /tmp/p_mps.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_AUDIO = (
    Path(__file__).resolve().parents[1] / "src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav"
)

BACKENDS = ("pyannote", "ecapa", "audeering", "ehcalabres", "speechbrain_ser", "emotion2vec_base", "emotion2vec_large")


def _audio(path: Path, repeat: int) -> Any:  # noqa: ANN401
    import torch

    from senselab.audio.data_structures import Audio
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    audio = resample_audios(downmix_audios_to_mono([Audio(filepath=str(path))]), 16000)[0]
    if repeat > 1:
        audio = Audio(waveform=torch.cat([audio.waveform] * repeat, dim=-1), sampling_rate=16000)
    return audio


def _clips(audio: Any, seconds: float = 3.0) -> List[Any]:  # noqa: ANN401
    from senselab.audio.data_structures import Audio

    n = int(seconds * audio.sampling_rate)
    total = audio.waveform.shape[-1]
    return [
        Audio(waveform=audio.waveform[:, i : i + n], sampling_rate=audio.sampling_rate)
        for i in range(0, total - n + 1, n)
    ]


def run(backend: str, device_name: str, audio_path: Path, repeat: int) -> Dict[str, Any]:
    """Run ``backend`` on ``device_name`` and return outputs, load time and inference time."""
    import torch

    from senselab.utils.data_structures import DeviceType

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    device = DeviceType(device_name)
    audio = _audio(audio_path, repeat)
    duration = audio.waveform.shape[-1] / audio.sampling_rate
    out: Dict[str, Any] = {"backend": backend, "device": device_name, "audio_s": duration}

    def timed(fn: Any) -> Any:  # noqa: ANN401
        start = time.perf_counter()
        result = fn()
        if device_name == "mps":
            torch.mps.synchronize()
        return result, time.perf_counter() - start

    if backend == "pyannote":
        from senselab.audio.tasks.speaker_diarization.pyannote import diarize_audios_with_pyannote

        warm = _audio(audio_path, 1)
        _, out["load_s"] = timed(lambda: diarize_audios_with_pyannote([warm], device=device))
        for view, exclusive in (("exclusive", True), ("overlapping", False)):
            lines, seconds = timed(lambda: diarize_audios_with_pyannote([audio], device=device, exclusive=exclusive))
            out[f"{view}_s"] = seconds
            out[view] = [(s.speaker, float(s.start), float(s.end)) for s in lines[0]]
        out["infer_s"] = out["exclusive_s"]
    elif backend == "ecapa":
        from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings

        clips = _clips(audio)
        extract = SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios
        _, out["load_s"] = timed(lambda: extract(clips[:1], device=device))
        vectors, out["infer_s"] = timed(lambda: extract(clips, device=device))
        out["vectors"] = [v.detach().cpu().flatten().tolist() for v in vectors]
        out["n_clips"] = len(clips)
    else:
        from senselab.audio.tasks.classification.speech_emotion_recognition import classify_emotions_from_speech
        from senselab.utils.data_structures import HFModel, SpeechBrainModel

        model: Any = {
            "audeering": lambda: HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"),
            "ehcalabres": lambda: HFModel(path_or_uri="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"),
            "speechbrain_ser": lambda: SpeechBrainModel(path_or_uri="speechbrain/emotion-recognition-wav2vec2-IEMOCAP"),
            "emotion2vec_base": lambda: HFModel(path_or_uri="emotion2vec/emotion2vec_plus_base"),
            "emotion2vec_large": lambda: HFModel(path_or_uri="emotion2vec/emotion2vec_plus_large"),
        }[backend]()
        import logging
        import re

        worker: List[str] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                if "emotion2vec timing" in record.getMessage():
                    worker.append(record.getMessage())

        logging.getLogger("senselab").addHandler(_Capture())
        clips = _clips(audio, seconds=5.0)
        _, out["load_s"] = timed(lambda: classify_emotions_from_speech(clips[:1], model, device=device))
        results, out["infer_s"] = timed(lambda: classify_emotions_from_speech(clips, model, device=device))
        if worker:
            fields = dict(re.findall(r"(\w+)=(\S+)", worker[-1]))
            out["wall_s_incl_worker_start"] = out["infer_s"]
            out["worker_load_s"] = float(fields["load_s"])
            out["infer_s"] = float(fields["infer_s"])
            out["worker_device"] = fields["device"]
        out["labels"] = [r.labels for r in results]
        out["scores"] = [[float(x) for x in r.scores] for r in results]
        out["n_clips"] = len(clips)
        out["clip_s"] = 5.0
    out["x_real_time"] = duration / out["infer_s"] if out["infer_s"] else None
    out["loadavg"] = os.getloadavg()
    return out


def compare(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Agreement between two runs of the same backend on different devices."""
    import numpy as np

    report: Dict[str, Any] = {"backend": a["backend"], "devices": [a["device"], b["device"]]}
    for key in ("load_s", "infer_s", "x_real_time", "loadavg"):
        report[key] = [a.get(key), b.get(key)]
    if a["backend"] == "pyannote":
        for view in ("exclusive", "overlapping"):
            sa, sb = a[view], b[view]
            report[f"{view}_segments"] = [len(sa), len(sb)]
            report[f"{view}_identical"] = sa == sb
            if len(sa) == len(sb):
                report[f"{view}_max_boundary_diff_s"] = max(
                    (max(abs(x[1] - y[1]), abs(x[2] - y[2])) for x, y in zip(sa, sb)), default=0.0
                )
                report[f"{view}_same_labels"] = [x[0] for x in sa] == [y[0] for y in sb]
    elif a["backend"] == "ecapa":
        va, vb = np.asarray(a["vectors"]), np.asarray(b["vectors"])
        cos = (va * vb).sum(1) / (np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1))
        report["cosine_min"] = float(cos.min())
        report["cosine_mean"] = float(cos.mean())
        report["n"] = len(cos)
    else:
        order = sorted(a["labels"][0])
        sa = np.asarray([[dict(zip(lab, sc))[k] for k in order] for lab, sc in zip(a["labels"], a["scores"])])
        sb = np.asarray([[dict(zip(lab, sc))[k] for k in order] for lab, sc in zip(b["labels"], b["scores"])])
        report["labels"] = order
        report["top_label_equal"] = [lab[0] for lab in a["labels"]] == [lab[0] for lab in b["labels"]]
        report["max_abs_diff"] = float(np.abs(sa - sb).max())
        report["argmax_agree"] = float((sa.argmax(1) == sb.argmax(1)).mean())
        report["n"] = len(sa)
    return report


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", nargs="?", choices=BACKENDS)
    parser.add_argument("device", nargs="?", choices=("cpu", "mps", "cuda"))
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--repeat", type=int, default=1, help="Tile the recording this many times.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    args = parser.parse_args()
    if args.compare:
        a, b = (json.loads(p.read_text()) for p in args.compare)
        print(json.dumps(compare(a, b), indent=2))
        return
    result = run(args.backend, args.device, args.audio, args.repeat)
    text = json.dumps(result)
    if args.out:
        args.out.write_text(text)
    summary = {
        k: v
        for k, v in result.items()
        if k in ("backend", "device", "audio_s", "load_s", "infer_s", "x_real_time", "loadavg")
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
