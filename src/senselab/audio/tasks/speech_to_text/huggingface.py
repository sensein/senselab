"""This module provides a factory for managing Hugging Face ASR pipelines."""
from typing import Any, Dict, List, Optional

from transformers import pipeline

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from senselab.utils.data_structures.script_line import ScriptLine


class HuggingFaceASR:
    """A factory for managing Hugging Face ASR pipelines."""

    _pipelines: Dict[str, pipeline] = {}

    @classmethod
    def _get_hf_asr_pipeline(
        cls,
        model: HFModel,
        return_timestamps: Optional[str],
        max_new_tokens: int,
        chunk_length_s: int,
        batch_size: int,
        device: Optional[DeviceType] = None,
    ) -> pipeline:
        """Get or create a Hugging Face ASR pipeline.

        Args:
            model (HFModel): The Hugging Face model.
            return_timestamps (Optional[str]): The level of timestamp details.
            max_new_tokens (int): The maximum number of new tokens.
            chunk_length_s (int): The length of audio chunks in seconds.
            batch_size (int): The batch size for processing.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            pipeline: The ASR pipeline.
        """
        device, torch_dtype = _select_device_and_dtype(
                user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
            )
        key = (
            f"{model.path_or_uri}-{model.revision}-{return_timestamps}-"
            f"{max_new_tokens}-{chunk_length_s}-{batch_size}-{device.value}"
        )
        if key not in cls._pipelines:
            cls._pipelines[key] = pipeline(
                "automatic-speech-recognition",
                model=model.path_or_uri,
                revision=model.revision,
                return_timestamps=return_timestamps,
                max_new_tokens=max_new_tokens,
                chunk_length_s=chunk_length_s,
                batch_size=batch_size,
                device=device.value,
                torch_dtype=torch_dtype,
            )
        return cls._pipelines[key]

    @classmethod
    def transcribe_audios_with_transformers(
        cls,
        audios: List[Audio],
        model: HFModel = HFModel(path_or_uri="openai/whisper-tiny"),
        language: Optional[Language] = None,
        return_timestamps: Optional[str] = "word",
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        batch_size: int = 16,
        device: Optional[DeviceType] = None,
    ) -> List[ScriptLine]:
        """Transcribes all audio samples in the dataset.

        Args:
            audios (List[Audio]): The list of audio objects to be transcribed.
            model (HFModel): The Hugging Face model used for transcription.
            language (Optional[Language]): The language of the audio (default is None).
            return_timestamps (Optional[str]): The level of timestamp details (default is "word").
            max_new_tokens (int): The maximum number of new tokens (default is 128).
            chunk_length_s (int): The length of audio chunks in seconds (default is 30).
            batch_size (int): The batch size for processing (default is 16).
            device (Optional[DeviceType]): The device to run the model on (default is None).

        Returns:
            List[ScritpLine]: The list of script lines.
        """

        def _audio_to_huggingface_dict(audio: Audio) -> Dict:
            """Convert an Audio object to a dictionary that can be used by the transformers pipeline.

            Args:
                audio (Audio): The audio object.

            Returns:
                Dict: The dictionary representation of the audio object.
            """
            return {
                "array": audio.waveform.squeeze().numpy(),
                "sampling_rate": audio.sampling_rate,
            }
        
        def _rename_key_recursive(obj: Dict[str, Any], old_key: str, new_key: str) -> Dict[str, Any]:
            """Recursively rename keys in a dictionary."""
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if key == old_key:
                        obj[new_key] = obj.pop(old_key)
                    elif isinstance(obj[key], (dict, list)):
                        obj[key] = _rename_key_recursive(obj[key], old_key, new_key)
            elif isinstance(obj, list):
                obj = [_rename_key_recursive(item, old_key, new_key) for item in obj]
            return obj

        # Check that all audio objects are mono
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(
                    f"Stereo audio is not supported. Got {audio.waveform.shape[0]} channels"
                )
        
        pipe = HuggingFaceASR._get_hf_asr_pipeline(
            model=model,
            return_timestamps=return_timestamps,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            device=device,
        )

        formatted_audios = [_audio_to_huggingface_dict(audio) for audio in audios]
        transcriptions = pipe(
            formatted_audios, generate_kwargs={"language": f"{language.name.lower()}"} if language else {}
        )
        transcriptions = _rename_key_recursive(transcriptions, "timestamp", "timestamps")
        return [ScriptLine.from_dict(t) for t in transcriptions]
