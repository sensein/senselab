"""NeMo ASR via isolated subprocess venv.

NeMo toolkit has dependency conflicts with the main senselab environment
(pins older transformers). Runs in an isolated subprocess venv managed by uv,
reusing the same venv as NeMo diarization.
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

# Reuse the same NeMo venv as diarization — it already has nemo_toolkit[asr]
_NEMO_VENV = "nemo-diarization"
_NEMO_REQUIREMENTS = [
    "nemo_toolkit[asr]",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
    "soundfile",
]
_NEMO_PYTHON = "3.12"

# Worker script — runs inside the isolated venv
_ASR_WORKER_SCRIPT = r"""
import json
import sys

try:
    import nemo.collections.asr as nemo_asr
    import torch

    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    model_name = args["model_name"]
    device = args["device"]

    # NeMo auto-selects CTC vs RNNT vs hybrid based on model config
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()

    all_results = []
    with torch.no_grad():
        # NeMo transcribe accepts a list of file paths
        transcripts = model.transcribe(audio_paths, batch_size=1, verbose=False)

        # NeMo returns different formats depending on model type:
        # - CTC/RNNT models: list of strings or Hypothesis objects
        # - For Hypothesis objects, .text gives the string
        for transcript in transcripts:
            if hasattr(transcript, "text"):
                text = transcript.text
            else:
                text = str(transcript)
            all_results.append({"text": text.strip()})

    print(json.dumps({"results": all_results}))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


class NeMoASR:
    """NeMo ASR transcription via isolated subprocess venv.

    NeMo models (e.g., ``nvidia/stt_en_conformer_ctc_large``) run in an
    isolated subprocess venv to avoid dependency conflicts with the main
    senselab environment.

    Supported model families:
        - CTC models (EncDecCTCModel / EncDecCTCModelBPE)
        - RNNT/Transducer models (EncDecRNNTModel / EncDecRNNTBPEModel)
        - Hybrid models

    The worker uses ``ASRModel.from_pretrained()`` which auto-selects the
    correct architecture based on the model config.
    """

    @classmethod
    def transcribe_with_nemo(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        device: Optional[DeviceType] = None,
    ) -> List[ScriptLine]:
        """Transcribe audios with NeMo ASR via subprocess venv.

        Args:
            audios: Audio clips to transcribe (mono, correct sample rate).
            model: HF model to use (default: ``nvidia/stt_en_conformer_ctc_large``).
            device: CPU or CUDA.

        Returns:
            One ``ScriptLine`` per input audio with the transcript text.
        """
        model_name = model.path_or_uri if model is not None else "nvidia/stt_en_conformer_ctc_large"
        device_type = device or _select_device_and_dtype(compatible_devices=[DeviceType.CUDA, DeviceType.CPU])[0]

        venv_dir = ensure_venv(_NEMO_VENV, _NEMO_REQUIREMENTS, python_version=_NEMO_PYTHON)
        python = str(venv_dir / "bin" / "python")

        with tempfile.TemporaryDirectory(prefix="senselab-nemo-asr-") as tmpdir:
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
                }
            )

            env = _clean_subprocess_env()
            result = subprocess.run(
                [python, "-c", _ASR_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )

            output = parse_subprocess_result(result, "NeMo ASR")

            results: List[ScriptLine] = []
            for entry in output.get("results", []):
                results.append(ScriptLine(text=entry.get("text", "")))

            return results
