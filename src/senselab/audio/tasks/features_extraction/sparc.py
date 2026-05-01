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
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, parse_subprocess_result, venv_python

# SPARC venv specification
_SPARC_VENV = "sparc"
_SPARC_REQUIREMENTS = [
    "speech-articulatory-coding>=0.1",
    "torch>=2.8,<2.9",
    "torchaudio>=2.8,<2.9",
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
                result[key] = [float(v) for v in val]
            elif isinstance(val, (np.integer, np.floating)):
                result[key] = val.item()
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


_DECODE_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sparc import load_model

args = json.loads(sys.stdin.read())
language = args["language"]
device = args["device"]
output_dir = args["output_dir"]
feature_dir = args["feature_dir"]

coder = load_model(language, device=device)

try:
    # Load features from numpy files
    ema = np.load(str(Path(feature_dir) / "ema.npy"))
    pitch = np.load(str(Path(feature_dir) / "pitch.npy"))
    loudness = np.load(str(Path(feature_dir) / "loudness.npy"))
    spk_emb = np.load(str(Path(feature_dir) / "spk_emb.npy"))

    waveform = coder.decode(ema, pitch, loudness, spk_emb)

    # Save output as FLAC
    out_path = str(Path(output_dir) / "decoded.flac")
    if isinstance(waveform, torch.Tensor):
        wav_np = waveform.detach().cpu().numpy().squeeze()
    else:
        wav_np = np.asarray(waveform).squeeze()
    sf.write(out_path, wav_np, coder.output_sr)

    print(json.dumps({"output_path": out_path, "sample_rate": coder.output_sr}))
except Exception as e:
    print(f"Error decoding SPARC features: {e}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    print(json.dumps({"error": {"type": type(e).__name__, "message": str(e)}}))
    sys.exit(1)
"""

_CONVERT_WORKER_SCRIPT = r"""
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sparc import load_model

args = json.loads(sys.stdin.read())
language = args["language"]
device = args["device"]
output_dir = args["output_dir"]
source_path = args["source_path"]
target_path = args["target_path"]

coder = load_model(language, device=device)

try:
    waveform = coder.convert(src_wav=source_path, trg_wav=target_path)

    # Save output as FLAC
    out_path = str(Path(output_dir) / "converted.flac")
    if isinstance(waveform, torch.Tensor):
        wav_np = waveform.detach().cpu().numpy().squeeze()
    else:
        wav_np = np.asarray(waveform).squeeze()
    sf.write(out_path, wav_np, coder.output_sr)

    print(json.dumps({"output_path": out_path, "sample_rate": coder.output_sr}))
except Exception as e:
    print(f"Error in SPARC voice conversion: {e}", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    print(json.dumps({"error": {"type": type(e).__name__, "message": str(e)}}))
    sys.exit(1)
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
        python = venv_python(venv_dir)

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

            output = parse_subprocess_result(result, "SPARC")
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

    @classmethod
    def decode_sparc_features(
        cls,
        features: Dict[str, torch.Tensor],
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
    ) -> Audio:
        """Resynthesize audio from SPARC articulatory features.

        Takes the feature dict returned by ``extract_sparc_features`` and
        runs ``coder.decode(ema, pitch, loudness, spk_emb)`` in the isolated
        subprocess venv to produce a waveform.

        Args:
            features: Feature dict with keys ``ema``, ``pitch``, ``loudness``,
                and ``spk_emb`` (torch.Tensor values, as returned by
                ``extract_sparc_features``).
            lang: Language for the SPARC model. None means multi-language.
            device: Device to use (CUDA or CPU).

        Returns:
            Resynthesized Audio object.
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

        required_keys = {"ema", "pitch", "loudness", "spk_emb"}
        missing = required_keys - set(features.keys())
        if missing:
            raise ValueError(f"Missing required feature keys: {missing}")

        venv_dir = ensure_venv(_SPARC_VENV, _SPARC_REQUIREMENTS, python_version=_SPARC_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-sparc-decode-") as tmpdir:
            tmp = Path(tmpdir)

            # Serialize feature tensors as numpy files
            feature_dir = tmp / "features"
            feature_dir.mkdir()
            for key in required_keys:
                tensor = features[key]
                if isinstance(tensor, torch.Tensor):
                    np.save(str(feature_dir / f"{key}.npy"), tensor.cpu().numpy())
                else:
                    np.save(str(feature_dir / f"{key}.npy"), np.asarray(tensor))

            input_json = json.dumps(
                {
                    "language": used_language,
                    "device": device.value,
                    "output_dir": str(tmp),
                    "feature_dir": str(feature_dir),
                }
            )

            result = subprocess.run(
                [python, "-c", _DECODE_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=_clean_subprocess_env(),
            )

            output = parse_subprocess_result(result, "SPARC decode")

            audio_path = output["output_path"]
            sample_rate = output["sample_rate"]
            logger.info("SPARC decode produced audio at %d Hz: %s", sample_rate, audio_path)

            tmp_audio = Audio(filepath=audio_path)
            # Force lazy-load while the temp file still exists, then
            # create a clean Audio without a filepath pointing at the
            # (about-to-be-deleted) temp directory.
            return Audio(waveform=tmp_audio.waveform, sampling_rate=tmp_audio.sampling_rate)

    @classmethod
    def convert_voice(
        cls,
        source_audio: Audio,
        target_audio: Audio,
        lang: Optional[Language] = None,
        device: Optional[DeviceType] = None,
    ) -> Audio:
        """Convert the voice of source audio to match the target speaker.

        Uses ``coder.convert(src_wav, trg_wav)`` in the isolated subprocess
        venv.  Both audios are saved as WAV files and passed by path.

        Args:
            source_audio: Audio whose content to preserve.
            target_audio: Audio whose speaker identity to adopt.
            lang: Language for the SPARC model. None means multi-language.
            device: Device to use (CUDA or CPU).

        Returns:
            Voice-converted Audio object.
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

        for label, audio in [("source", source_audio), ("target", target_audio)]:
            if audio.waveform.squeeze().dim() != 1:
                raise ValueError(f"Only mono audio is supported ({label}).")

        venv_dir = ensure_venv(_SPARC_VENV, _SPARC_REQUIREMENTS, python_version=_SPARC_PYTHON)
        python = venv_python(venv_dir)

        with tempfile.TemporaryDirectory(prefix="senselab-sparc-convert-") as tmpdir:
            tmp = Path(tmpdir)

            src_path = str(tmp / "source.wav")
            trg_path = str(tmp / "target.wav")
            source_audio.save_to_file(src_path, format="wav")
            target_audio.save_to_file(trg_path, format="wav")

            input_json = json.dumps(
                {
                    "language": used_language,
                    "device": device.value,
                    "output_dir": str(tmp),
                    "source_path": src_path,
                    "target_path": trg_path,
                }
            )

            result = subprocess.run(
                [python, "-c", _CONVERT_WORKER_SCRIPT],
                input=input_json,
                capture_output=True,
                text=True,
                timeout=600,
                env=_clean_subprocess_env(),
            )

            output = parse_subprocess_result(result, "SPARC convert")

            audio_path = output["output_path"]
            sample_rate = output["sample_rate"]
            logger.info("SPARC convert produced audio at %d Hz: %s", sample_rate, audio_path)

            tmp_audio = Audio(filepath=audio_path)
            # Force lazy-load while the temp file still exists, then
            # create a clean Audio without a filepath pointing at the
            # (about-to-be-deleted) temp directory.
            return Audio(waveform=tmp_audio.waveform, sampling_rate=tmp_audio.sampling_rate)
