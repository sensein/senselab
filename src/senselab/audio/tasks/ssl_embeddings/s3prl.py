"""S3PRL SSL embedding extraction via isolated subprocess venv.

S3PRL (Self-Supervised Speech Pre-training and Representation Learning)
has dependencies that conflict with the main environment (in particular
it calls ``torchaudio.set_audio_backend()`` which was removed in
torchaudio 2.5+). It runs in an isolated subprocess venv managed by uv.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result

# S3PRL venv specification
_S3PRL_VENV = "s3prl"
_S3PRL_REQUIREMENTS = [
    "s3prl",
    "torch>=2.0,<2.5",
    "torchaudio>=2.0,<2.5",
    "numpy",
    "soundfile",
]
_S3PRL_PYTHON = "3.11"

# Worker script for S3PRL embedding extraction — runs inside the isolated venv
_S3PRL_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
model_name = args["model_name"]
device = args["device"]
output_dir = args["output_dir"]

try:
    import s3prl.hub as hub

    model_cls = getattr(hub, model_name, None)
    if model_cls is None:
        available = [k for k in dir(hub) if not k.startswith("_")]
        print(json.dumps({"error": {
            "type": "ValueError",
            "message": f"Unknown S3PRL model '{model_name}'. Available: {available[:20]}"
        }}))
        sys.exit(1)

    model = model_cls()
    model.eval()
    model = model.to(device)

    output_paths = []
    shapes = []
    for i, audio_path in enumerate(audio_paths):
        data, sr = sf.read(audio_path, dtype="float32")
        waveform = torch.tensor(data.squeeze()).to(device)

        with torch.no_grad():
            output = model([waveform])

        # Extract hidden states — S3PRL returns dict with 'hidden_states'
        hidden_states = output.get("hidden_states", None)
        if hidden_states is None:
            # Some models return 'last_hidden_state' directly
            last_hidden = output.get("last_hidden_state", None)
            if last_hidden is not None:
                embedding = last_hidden.cpu().numpy()
            else:
                print(json.dumps({"error": {
                    "type": "RuntimeError",
                    "message": f"Model '{model_name}' did not return hidden_states or last_hidden_state"
                }}))
                sys.exit(1)
        else:
            # Take the last hidden state
            embedding = hidden_states[-1].cpu().numpy()

        out_path = str(Path(output_dir) / f"embedding_{i}.npy")
        np.save(out_path, embedding)
        output_paths.append(out_path)
        shapes.append(list(embedding.shape))

    print(json.dumps({"output_paths": output_paths, "shapes": shapes}))
except Exception as e:
    print(f"Error in S3PRL extraction: {e}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    print(json.dumps({"error": {"type": type(e).__name__, "message": str(e)}}))
    sys.exit(1)
"""


class S3PRLEmbeddingExtractor:
    """Extracts SSL embeddings using S3PRL models in an isolated subprocess venv."""

    @classmethod
    def extract_s3prl_embeddings(
        cls,
        audios: List[Audio],
        model_name: str,
        device: Optional[DeviceType] = None,
    ) -> List[torch.Tensor]:
        """Extract SSL embeddings from audios using an S3PRL model.

        S3PRL runs in an isolated subprocess venv because it requires
        torchaudio<2.5 (uses the removed ``set_audio_backend()`` API).

        Args:
            audios: List of mono Audio objects.
            model_name: S3PRL model name (e.g., "apc", "tera", "cpc",
                "wav2vec2", "hubert"). These are attributes of ``s3prl.hub``.
            device: Device to use (CUDA or CPU).

        Returns:
            List of tensors, one per audio. Each tensor has shape
            ``(time_frames, embedding_dim)``.

        Raises:
            ValueError: If audio is not mono or model name is invalid.
            RuntimeError: If the subprocess fails.
        """
        if len(audios) == 0:
            return []

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        # Validate inputs
        for i, audio in enumerate(audios):
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")

        venv_dir = ensure_venv(_S3PRL_VENV, _S3PRL_REQUIREMENTS, python_version=_S3PRL_PYTHON)
        python = str(venv_dir / "bin" / "python")

        with tempfile.TemporaryDirectory(prefix="senselab-s3prl-") as tmpdir:
            tmp = Path(tmpdir)

            # Serialize audios to FLAC
            audio_paths = []
            for i, audio in enumerate(audios):
                path = str(tmp / f"audio_{i}.flac")
                audio.save_to_file(path, format="flac")
                audio_paths.append(path)

            # Run worker in isolated venv
            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "model_name": model_name,
                    "device": device.value,
                    "output_dir": str(tmp),
                }
            )

            result = subprocess.run(
                [python, "-c", _S3PRL_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=_clean_subprocess_env(),
            )

            output = parse_subprocess_result(result, "S3PRL")

            # Load results
            embeddings = []
            for out_path in output.get("output_paths", []):
                arr = np.load(out_path)
                # Squeeze batch dimension if present (batch=1)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                embeddings.append(torch.from_numpy(arr))

            logger.info(
                "S3PRL extracted %d embeddings with model '%s' (shapes: %s)",
                len(embeddings),
                model_name,
                output.get("shapes"),
            )
            return embeddings
