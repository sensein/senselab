"""Transcribe WAV files with CrisperWhisper and Qwen3-ASR through senselab, one model resident at a time.

Usage: uv run python specs/20261017-104000-asr-apple-silicon/run_pair.py OUT_DIR WAV [WAV ...]
Writes OUT_DIR/<stem>.<model>.json (ScriptLine dump) and OUT_DIR/timings.jsonl.
"""

import json
import os
import sys
import time
from pathlib import Path

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import HFModel
from senselab.utils.venv_worker import shutdown_venv_workers

MODELS = {
    "crisperwhisper": "nyralabs/CrisperWhisper2.0_turbo",
    "qwen3asr": "Qwen/Qwen3-ASR-1.7B",
}

out_dir = Path(sys.argv[1])
out_dir.mkdir(parents=True, exist_ok=True)
wavs = [Path(p) for p in sys.argv[2:]]
only = os.environ.get("ONLY_MODELS")
for tag, model_id in MODELS.items():
    if only and tag not in only.split(","):
        continue
    model = HFModel(path_or_uri=model_id)
    for wav in wavs:
        target = out_dir / f"{wav.stem}.{tag}.json"
        if target.exists():
            continue
        audio = Audio(filepath=str(wav))
        duration = audio.waveform.shape[-1] / audio.sampling_rate
        load_before = os.getloadavg()
        t0 = time.perf_counter()
        [line] = transcribe_audios([audio], model=model)
        wall = time.perf_counter() - t0
        target.write_text(line.model_dump_json())
        rec = {
            "file": wav.name,
            "model": tag,
            "audio_s": duration,
            "wall_s": wall,
            "rtf": wall / duration,
            "loadavg_before": load_before,
            "loadavg_after": os.getloadavg(),
            "n_words": len(line.chunks or []),
        }
        with open(out_dir / "timings.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    shutdown_venv_workers()
