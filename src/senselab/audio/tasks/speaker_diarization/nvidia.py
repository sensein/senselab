"""NVIDIA Sortformer diarization via isolated subprocess venv.

NeMo toolkit has dependency conflicts with the main senselab environment
(pins older transformers). Runs in an isolated subprocess venv managed by uv.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result

# NeMo venv specification
_NEMO_VENV = "nemo-diarization"
_NEMO_REQUIREMENTS = [
    "nemo_toolkit[asr]",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "soundfile",
]
_NEMO_PYTHON = "3.12"

# Worker script — runs inside the isolated venv
_WORKER_SCRIPT = r"""
import json
import sys
from pathlib import Path

try:
    import nemo.collections.asr as nemo_asr
    import torch

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]
    output_dir = args["output_dir"]

    model = nemo_asr.models.SortformerEncLabelModel.from_pretrained(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    all_results = []
    for audio_path in audio_paths:
        with torch.no_grad():
            diar_output = model.diarize(
                audio=audio_path,
                batch_size=1,
                num_workers=0,
                verbose=False,
            )

        # diar_output is List[List[str]] — format: "start end speaker"
        segments = []
        if diar_output and diar_output[0]:
            for line in diar_output[0]:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start = float(parts[0])
                    end = float(parts[1])
                    speaker = parts[2]
                    segments.append({
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                    })
        all_results.append(segments)

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def diarize_audios_with_nvidia_sortformer(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
) -> List[List[ScriptLine]]:
    """Diarize audios with NVIDIA Sortformer (NeMo) via subprocess venv.

    Args:
        audios: Audio clips to diarize (mono, correct sample rate).
        model: HF model to use (default: "nvidia/diar_sortformer_4spk-v1").
        device: CPU or CUDA.

    Returns:
        One list per input audio with (speaker, start, end), sorted by start time.
    """
    model_name = model.path_or_uri if model is not None else "nvidia/diar_sortformer_4spk-v1"
    device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

    venv_dir = ensure_venv(_NEMO_VENV, _NEMO_REQUIREMENTS, python_version=_NEMO_PYTHON)
    python = str(venv_dir / "bin" / "python")

    with tempfile.TemporaryDirectory(prefix="senselab-nemo-") as tmpdir:
        tmp = Path(tmpdir)

        # Serialize audios to WAV
        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.wav")
            audio.save_to_file(path)
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_name": model_name,
                "device": device_type.value,
                "output_dir": str(tmp),
            }
        )

        env = _clean_subprocess_env()
        result = subprocess.run(
            [python, "-c", _WORKER_SCRIPT],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
            env=env,
        )

        output = parse_subprocess_result(result, "NeMo Sortformer")

        results: List[List[ScriptLine]] = []
        for segments in output.get("results", []):
            script_lines = [
                ScriptLine(
                    speaker=str(seg.get("speaker", "")),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                )
                for seg in segments
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

        return results
