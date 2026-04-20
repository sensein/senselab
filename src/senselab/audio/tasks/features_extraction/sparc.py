"""SPARC feature extraction via isolated subprocess venv.

speech-articulatory-coding (SPARC) has dependencies that may conflict
with the main environment. It runs in an isolated subprocess venv
managed by uv, shared with ppgs where possible.
"""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, Language, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import ensure_venv

# SPARC venv specification
_SPARC_VENV = "sparc"
_SPARC_REQUIREMENTS = [
    "speech-articulatory-coding>=0.1",
    "torch~=2.8",
    "torchaudio~=2.8",
    "numpy",
    "soundfile",
    "librosa",
    "transformers",
    "torchcrepe==0.0.23",
    "penn==0.0.14",
    "huggingface-hub",
]
_SPARC_PYTHON = "3.11"

# Worker script for feature extraction — runs inside the isolated venv
_FEATURE_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sparc import load_model

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
language = args["language"]
device = args["device"]
output_dir = args["output_dir"]

coder = load_model(language, device=device)

output_paths = []
for i, audio_path in enumerate(audio_paths):
    data, sr = sf.read(audio_path, dtype="float32")
    waveform = data.squeeze()

    try:
        features = coder.encode(waveform)
        # Convert all tensors/arrays to numpy for serialization
        result = {}
        for key, val in features.items():
            if isinstance(val, torch.Tensor):
                result[key] = val.cpu().numpy().tolist()
            elif isinstance(val, np.ndarray):
                result[key] = val.tolist()
            elif isinstance(val, tuple):
                result[key] = list(val)
            else:
                result[key] = val
    except Exception as e:
        print(f"Error extracting SPARC features for audio {i}: {e}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        result = {"error": str(e)}

    out_path = str(Path(output_dir) / f"sparc_{i}.json")
    with open(out_path, "w") as f:
        json.dump(result, f)
    output_paths.append(out_path)

print(json.dumps({"output_paths": output_paths, "expected_sr": coder.sr}))
"""


class SparcFeatureExtractor:
    """A factory for managing feature extraction pipelines using SPARC."""

    @classmethod
    def extract_sparc_features(
        cls,
        audios: List[Audio],
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
        resample: Optional[bool] = False,
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract SPARC articulatory features from audios.

        The SPARC model runs in an isolated subprocess venv with its own
        Python and dependencies. Audio is transferred via FLAC files.

        Args:
            audios: List of audio objects.
            lang: Language for the SPARC model. None means multi-language.
            device: Device to use (CUDA or CPU).
            resample: Whether to resample audios to the model's expected rate.

        Returns:
            List of feature dicts with keys: ema, loudness, pitch,
            periodicity, pitch_stats, spk_emb, ft_len.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )

        if lang is None:
            used_language = "multi"
        elif lang.name == "english":
            used_language = "en+"
        else:
            raise ValueError(f"Language {lang.name} not supported. Supported: english or None (multi-language).")

        venv_dir = ensure_venv(_SPARC_VENV, _SPARC_REQUIREMENTS, python_version=_SPARC_PYTHON)
        python = str(venv_dir / "bin" / "python")

        with tempfile.TemporaryDirectory(prefix="senselab-sparc-") as tmpdir:
            tmp = Path(tmpdir)

            # Serialize audios to FLAC
            audio_paths = []
            for i, audio in enumerate(audios):
                if audio.waveform.squeeze().dim() != 1:
                    raise ValueError(f"Only mono audio files are supported. Audio index: {i}")
                path = str(tmp / f"audio_{i}.flac")
                audio.save_to_file(path, format="flac")
                audio_paths.append(path)

            # Run worker in isolated venv
            input_json = json.dumps(
                {
                    "audio_paths": audio_paths,
                    "language": used_language,
                    "device": device.value,
                    "output_dir": str(tmp),
                }
            )

            result = subprocess.run(
                [python, "-c", _FEATURE_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
            )

            if result.returncode != 0:
                raise RuntimeError(f"SPARC venv failed:\n{result.stderr}")

            output = json.loads(result.stdout.strip().splitlines()[-1])
            expected_sr = output.get("expected_sr")

            # Check sample rates (after we know what the model expects)
            if not resample:
                for i, audio in enumerate(audios):
                    if expected_sr and audio.sampling_rate != expected_sr:
                        raise ValueError(
                            f"Expected sample rate {expected_sr}, got {audio.sampling_rate}. "
                            f"Audio index: {i}. Set resample=True to auto-resample."
                        )

            # If resampling needed, redo with resampled audio
            if resample and expected_sr:
                needs_resample = any(a.sampling_rate != expected_sr for a in audios)
                if needs_resample:
                    resampled = resample_audios(audios, resample_rate=expected_sr)
                    return cls.extract_sparc_features(resampled, lang=lang, device=device, resample=False)

            # Load results
            features_list = []
            for out_path in output.get("output_paths", []):
                with open(out_path) as f:
                    raw = json.load(f)

                if "error" in raw:
                    features_list.append(
                        {
                            "ema": torch.tensor(torch.nan),
                            "loudness": torch.tensor(torch.nan),
                            "pitch": torch.tensor(torch.nan),
                            "periodicity": torch.tensor(torch.nan),
                            "pitch_stats": torch.tensor(torch.nan),
                            "spk_emb": torch.tensor(torch.nan),
                            "ft_len": torch.nan,
                        }
                    )
                else:
                    features = {}
                    for key, val in raw.items():
                        if isinstance(val, list):
                            features[key] = torch.tensor(val)
                        else:
                            features[key] = val
                    features_list.append(features)

            return features_list
