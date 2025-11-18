"""This module implements the NVIDIA Sortformer Diarization task (via Docker)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, ScriptLine, _select_device_and_dtype


def diarize_audios_with_nvidia_sortformer(
    audios: List[Audio],
    model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
    docker_image: str = "nvcr.io/nvidia/nemo:25.09",
) -> List[List[ScriptLine]]:
    """Diarize audios with NVIDIA Sortformer (NeMo) via Docker, returning per-speaker segments.

    Args:
        audios (list[Audio]): Audio clips to diarize (mono, correct sample rate).
        model (HFModel | None): HF model to use (default: "nvidia/diar_sortformer_4spk-v1").
        device (DeviceType | None): CPU or CUDA (default picked by _select_device_and_dtype()).
        docker_image (str): Docker image containing NeMo Sortformer.

    Returns:
        list[list[ScriptLine]]: One list per input audio with (speaker, start, end), sorted by start time.

    Raises:
        RuntimeError: If Docker is not running or the container fails.
        ValueError: If input is invalid.
    """
    model_name = model.path_or_uri if model is not None else "nvidia/diar_sortformer_4spk-v1"
    device = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

    results: List[List[ScriptLine]] = []

    for audio in audios:
        # Isolated temp workdir we mount at /app inside the container
        with tempfile.TemporaryDirectory(prefix="nemo_work_") as workdir:
            workdir = os.path.abspath(workdir)
            wav_rel = "input.wav"
            out_rel = "out.json"
            worker_rel = "nemo_diarization_worker.py"

            # 1) Write audio into workdir
            wav_path = os.path.join(workdir, wav_rel)
            audio.save_to_file(wav_path)

            # 2) Copy worker script into workdir
            worker_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "nemo_diarization_worker.py"))
            if not os.path.exists(worker_src):
                raise FileNotFoundError(f"Worker script not found: {worker_src}")
            worker_dst = os.path.join(workdir, worker_rel)
            shutil.copy2(worker_src, worker_dst)

            # 3) Build docker command
            base_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{workdir}:/app",
                "-w",
                "/app",
                # persist HF cache to speed up repeated runs:
                "-v",
                f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
            ]

            cmd = base_cmd + [
                docker_image,
                "python",
                worker_rel,
                "--audio",
                f"/app/{wav_rel}",
                "--model",
                model_name,
                "--device",
                device.value,
                "--out",
                f"/app/{out_rel}",
            ]

            # 4) Run container
            try:
                proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "NeMo Docker worker failed.\n"
                    f"Exit code: {e.returncode}\n"
                    f"Stdout: {e.stdout}\n"
                    f"Stderr: {e.stderr}"
                ) from e

            # 5) Read JSON from file produced by the worker
            out_path = os.path.join(workdir, out_rel)
            if not os.path.exists(out_path):
                raise RuntimeError(
                    "NeMo Docker worker did not produce output file.\n" f"Stdout: {proc.stdout}\nStderr: {proc.stderr}"
                )

            with open(out_path, "r", encoding="utf-8") as f:
                output = json.load(f)

            if isinstance(output, dict) and "error" in output:
                raise RuntimeError(f"NeMo Docker worker error: {output['error']}")

            segments = output.get("segments", [])
            script_lines: List[ScriptLine] = [
                ScriptLine(
                    speaker=str(seg.get("speaker", "")),
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                )
                for seg in segments
            ]
            results.append(sorted(script_lines, key=lambda x: x.start or 0.0))

    return results
