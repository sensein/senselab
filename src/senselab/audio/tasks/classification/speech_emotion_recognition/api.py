"""Speech emotion recognition API.

Supports discrete (categorical) and continuous (dimensional) SER models.
Continuous SER models that have broken config.json fields (e.g., vocab_size: null)
are run in an isolated subprocess venv with pinned huggingface-hub<1.0.
"""

import json
import subprocess
import tempfile
import warnings
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, _select_device_and_dtype, logger
from senselab.utils.subprocess_venv import ensure_venv, parse_subprocess_result


class SERType(Enum):
    """SER types for determining model output behaviors."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    VALENCE = "valence"
    NOT_RECOGNIZED = "not_recognized"


# Subprocess venv for continuous SER models with broken configs
_CONT_SER_VENV = "continuous-ser"
_CONT_SER_REQUIREMENTS = [
    "transformers>=4.40,<5",
    "huggingface-hub>=0.34,<1.0",  # 1.x strict validation breaks some model configs
    "torch>=2.8",
    "soundfile",
    "numpy",
    "safetensors",
]
_CONT_SER_PYTHON = "3.12"

_CONT_SER_WORKER = r"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, pipeline

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
model_id = args["model_id"]
revision = args["revision"]
device = args["device"]
output_dir = args["output_dir"]

pipe = pipeline(
    task="audio-classification",
    model=model_id,
    revision=revision,
    device=device,
    function_to_apply="none",
)

results = []
for audio_path in audio_paths:
    data, sr = sf.read(audio_path, dtype="float32")
    output = pipe({"array": data, "sampling_rate": sr})
    results.append(output)

print(json.dumps({"results": results}))
"""


def classify_emotions_from_speech(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify all audios using the given speech emotion recognition model.

    Args:
        audios: The list of audio objects to be classified.
        model: The model used for classification, should be trained for recognizing emotions.
        device: The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the classification function.

    Returns:
        List of speech emotion recognition results.
    """
    if isinstance(model, HFModel):
        model_info = model.get_model_info()
        tags = model_info.tags or []

        ser_type = _get_ser_type(model)

        if (
            not ("speech-emotion-recognition" in tags or "emotion-recognition" in tags)
            and ser_type == SERType.NOT_RECOGNIZED
        ):
            raise ValueError(
                f"The model '{model.path_or_uri}' is not suitable for speech emotion recognition. Please "
                "validate that it has the correct tags or use the more generic "
                "'audio_classification_with_hf_models' function."
            )

        if ser_type == SERType.CONTINUOUS:
            return _classify_continuous_ser_venv(audios, model, device)
        if ser_type == SERType.CONTINUOUS:
            output_function_to_apply = kwargs.get("function_to_apply", None)
            if output_function_to_apply:
                if output_function_to_apply != "none":
                    warnings.warn(
                        "Senselab predicts that you are using a continuous SER model but have "
                        "specified the parameter `function_to_apply` as something other than none. This "
                        "might create side effects when dealing with continuous values that do not "
                        "necessarily represent probabilities."
                    )
            else:
                kwargs["function_to_apply"] = "none"
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError(
            "Only Hugging Face models are supported for now. We aim to support more models in the future."
        )


def _classify_continuous_ser_venv(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
) -> List[AudioClassificationResult]:
    """Run continuous SER in an isolated subprocess venv.

    Some continuous SER models have broken config.json (e.g., vocab_size: null)
    that huggingface-hub >=1.0 strict validation rejects. This runs them in a
    venv with pinned huggingface-hub<1.0.
    """
    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    venv_dir = ensure_venv(_CONT_SER_VENV, _CONT_SER_REQUIREMENTS, python_version=_CONT_SER_PYTHON)
    python = str(venv_dir / "bin" / "python")

    with tempfile.TemporaryDirectory(prefix="senselab-ser-") as tmpdir:
        tmp = Path(tmpdir)

        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.flac")
            audio.save_to_file(path, format="flac")
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_id": str(model.path_or_uri),
                "revision": model.revision or "main",
                "device": device_type.value,
                "output_dir": str(tmp),
            }
        )

        result = subprocess.run(
            [python, "-c", _CONT_SER_WORKER],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = parse_subprocess_result(result, "Continuous SER")

        results = []
        for classification_list in output.get("results", []):
            labels = [c["label"] for c in classification_list]
            scores = [float(c["score"]) for c in classification_list]
            results.append(AudioClassificationResult(labels=labels, scores=scores))

        return results


def _get_ser_type(model: HFModel) -> SERType:
    """Get the type of SER the model is likely used for based on the labels it is set to predict."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
    except Exception:
        # Fall back to raw config dict for models with invalid fields
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(model.path_or_uri, "config.json", revision=model.revision)
        with open(config_path) as f:
            config_dict = json.load(f)
        config = SimpleNamespace(id2label=config_dict.get("id2label", {}))
    id2label = config.id2label
    if id2label:
        labels = list(id2label.values())
        if "positive" in labels and "negative" in labels and "neutral" in labels:
            return SERType.VALENCE
        if "valence" in labels or "arousal" in labels or "dominance" in labels:
            return SERType.CONTINUOUS
        if (
            "happy" in labels
            or "happiness" in labels
            or "joy" in labels
            or "angry" in labels
            or "anger" in labels
            or "sad" in labels
            or "sadness" in labels
            or "fear" in labels
            or "disgust" in labels
            or "surprise" in labels
        ):
            return SERType.DISCRETE
    return SERType.NOT_RECOGNIZED
