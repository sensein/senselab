"""This module implements some utilities for the voice cloning task."""

from typing import Any, Dict, Optional, Tuple

import pydra
import torch
from datasets import Dataset

from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def clone_voice_in_dataset_with_KNNVC(
    source_dataset: Dict[str, Any],
    target_dataset: Dict[str, Any],
    source_audio_column: str = "audio",
    target_audio_column: str = "audio",
    model_id: str = "bshall/knn-vc",
    model_revision: str = "master",
    prematched_vocoder: bool = True,
    topk: int = 4,
    device: Optional[DeviceType] = None,
) -> Dict[str, Any]:
    """Clones the voice in the dataset using KNNVC."""

    def _setup_knn_vc_model(
        model_id: str,
        model_revision: str,
        prematched_vocoder: bool,
        device: Optional[DeviceType] = None,
    ) -> Tuple[object, DeviceType, torch.dtype]:
        """Prepare a KNNVC pipeline."""
        repo_id = f"{model_id}:{model_revision}"
        device, torch_dtype = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        knn_vc = torch.hub.load(
            repo_id,
            "knn_vc",
            prematched=prematched_vocoder,
            trust_repo=True,
            pretrained=True,
            device=device.value,
        )
        return knn_vc, device, torch_dtype

    def _clone_voice_in_row_with_KNNVC(
        source_row: Dataset,
        target_dataset: Dataset,
        knn_vc_model: Any,  # noqa: ANN401
        torch_dtype: torch.dtype,
        source_audio_column: str = "audio",
        target_audio_column: str = "audio",
    ) -> Dict[str, torch.Tensor]:
        def _get_waveform(dataset: Dataset, column: str) -> torch.Tensor:
            audio = dataset[column]
            waveform = torch.tensor(audio["array"], dtype=torch_dtype)
            sampling_rate = audio["sampling_rate"]
            if sampling_rate != 16000:
                raise ValueError(
                    f"{column} sampling rate {sampling_rate} is not supported. "
                    "Only 16kHz sampling rates are supported."
                )
            return waveform

        source_waveform = _get_waveform(source_row, source_audio_column)
        target_waveform = _get_waveform(target_dataset[0], target_audio_column)

        query_seq = knn_vc_model.get_features(source_waveform)
        matching_set = knn_vc_model.get_matching_set([target_waveform])
        out_wav = knn_vc_model.match(query_seq, matching_set, topk=topk)
        return {"cloned_waveform": out_wav}

    hf_source_dataset = _from_dict_to_hf_dataset(source_dataset, audio_columns=[source_audio_column])
    hf_target_dataset = _from_dict_to_hf_dataset(target_dataset, audio_columns=[target_audio_column])

    knn_vc, device, torch_dtype = _setup_knn_vc_model(
        model_id=model_id,
        model_revision=model_revision,
        prematched_vocoder=prematched_vocoder,
        device=device,
    )

    cloned_dataset = hf_source_dataset.map(
        lambda x: _clone_voice_in_row_with_KNNVC(
            x,
            hf_target_dataset,
            knn_vc,
            torch_dtype,
            source_audio_column,
            target_audio_column,
        )
    )
    cloned_dataset = cloned_dataset.remove_columns([source_audio_column])

    return _from_hf_dataset_to_dict(cloned_dataset)

clone_voice_in_dataset_with_KNNVC_pt = pydra.mark.task(
    clone_voice_in_dataset_with_KNNVC
)
