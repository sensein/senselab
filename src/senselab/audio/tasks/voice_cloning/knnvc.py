"""This module contains functions for voice cloning using KNNVC."""

from typing import Any, Dict, List, Optional

import torch

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import TorchModel


class KNNVC:
    """A factory for managing KNNVC pipelines."""

    _pipelines: Dict[str, Any] = {}

    @classmethod
    def _get_knnvc_pipeline(
        cls,
        model: TorchModel,
        prematched_vocoder: bool,
        topk: int,
        device: Optional[DeviceType] = None,
    ) -> Any:  # noqa: ANN401
        """Get or create a KNNVC pipeline."""
        key = f"{model.path_or_uri}-{model.revision}-{prematched_vocoder}-{topk}-{device}"
        if key not in cls._pipelines:
            device, _ = _select_device_and_dtype(
                user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
            )
            knn_vc = torch.hub.load(
                model="knn_vc",
                repo_or_dir=model.path_or_uri,
                prematched=prematched_vocoder,
                trust_repo=True,
                pretrained=True,
                device=device.value,
            )
            cls._pipelines[key] = knn_vc
        return cls._pipelines[key]

    @classmethod
    def clone_voices_with_knn_vc(
        cls,
        source_audios: List[Audio],
        target_audios: List[Audio],
        model: TorchModel = TorchModel(path_or_uri="bshall/knn-vc", revision="master"),
        prematched_vocoder: bool = True,
        topk: int = 4,
        device: Optional[DeviceType] = None,
    ) -> List[Audio]:
        """Clones voices from source audios to target audios using KNNVC."""
        knn_vc = cls._get_knnvc_pipeline(model=model, prematched_vocoder=prematched_vocoder, topk=topk, device=device)

        cloned_audios = []
        for source_audio, target_audio in zip(source_audios, target_audios):
            if source_audio.waveform.shape[0] > 1 or target_audio.waveform.shape[0] > 1:
                raise ValueError("Only mono audio files are supported.")
            source_sampling_rate = source_audio.sampling_rate
            target_sampling_rate = target_audio.sampling_rate
            # 16kHz is the only supported sampling rate for KNNVC
            supported_sampling_rate = 16000
            if source_sampling_rate != supported_sampling_rate or target_sampling_rate != supported_sampling_rate:
                raise ValueError(
                    f"Sampling rates {source_sampling_rate} and/or {target_sampling_rate} are not supported."
                    f"Only {supported_sampling_rate} sampling rate is supported."
                )

            source_waveform = source_audio.waveform
            target_waveform = target_audio.waveform

            query_seq = knn_vc.get_features(source_waveform)
            matching_set = knn_vc.get_matching_set([target_waveform])
            out_wav = knn_vc.match(query_seq, matching_set, topk=topk)

            cloned_audios.append(
                Audio(
                    waveform=out_wav,
                    sampling_rate=source_sampling_rate,
                    orig_path_or_id=source_audio.orig_path_or_id,  # TODO: this should be customized
                )
            )

        return cloned_audios
