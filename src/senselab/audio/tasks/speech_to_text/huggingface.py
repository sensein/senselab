"""This module provides a factory for managing Hugging Face ASR pipelines.

To ensure correct functionality, call `transcribe_audios_with_transformers` serially or,
if you need to process multiple audios in parallel, pass the entire list of audios to the
function at once, rather than calling the function with one audio at a time.
"""

import time
from typing import Any, Dict, List, Optional

from transformers import pipeline

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel, Language, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


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
        device, _ = _select_device_and_dtype(
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
            )
        return cls._pipelines[key]

    @classmethod
    def transcribe_audios_with_transformers(
        cls,
        audios: List[Audio],
        model: Optional[HFModel] = None,
        language: Optional[Language] = None,
        return_timestamps: Optional[str] = "word",
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        batch_size: int = 1,
        device: Optional[DeviceType] = None,
    ) -> List[ScriptLine]:
        """Transcribes all audio samples in the dataset.

        Args:
            audios (List[Audio]): The list of audio objects to be transcribed.
            model (HFModel): The Hugging Face model used for transcription.
                If None, the default model "openai/whisper-tiny" is used.
            language (Optional[Language]): The language of the audio (default is None).
            return_timestamps (Optional[str]): The level of timestamp details (default is "word").
            max_new_tokens (int): The maximum number of new tokens (default is 128).
            chunk_length_s (int): The length of audio chunks in seconds (default is 30).
            batch_size (int): The batch size for processing (default is 1).
                Note: Issues have been observed with long audio recordings and timestamped transcript
                if the batch_size is high - not exactly clear what high means
                (https://github.com/huggingface/transformers/issues/2615#issuecomment-656923205).
            device (Optional[DeviceType]): The device to run the model on (default is None).

        Returns:
            List[ScritpLine]: The list of script lines.
        """
        if model is None:
            model = HFModel(path_or_uri="openai/whisper-tiny")

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

        # Take the start time of the pipeline initialization
        start_time_pipeline = time.time()
        # Get the Hugging Face pipeline
        pipe = HuggingFaceASR._get_hf_asr_pipeline(
            model=model,
            return_timestamps=return_timestamps,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            device=device,
        )

        # Take the end time of the pipeline initialization
        end_time_pipeline = time.time()
        # Print the time taken for initialize the hugging face ASR pipeline
        elapsed_time_pipeline = end_time_pipeline - start_time_pipeline
        logger.info(f"Time taken to initialize the hugging face ASR pipeline: {elapsed_time_pipeline:.2f} seconds")

        # Retrieve the expected sampling rate from the Hugging Face model
        expected_sampling_rate = pipe.feature_extractor.sampling_rate

        # Check that all audio objects are mono
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Stereo audio is not supported. Got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sampling_rate:
                raise ValueError(
                    f"Incorrect sampling rate. Expected {expected_sampling_rate}" f", got {audio.sampling_rate}"
                )

        # Convert the audio objects to dictionaries that can be used by the pipeline
        formatted_audios = [_audio_to_huggingface_dict(audio) for audio in audios]

        # Take the start time of the transcription
        start_time_transcription = time.time()
        # Run the pipeline
        transcriptions = pipe(
            formatted_audios, generate_kwargs={"language": f"{language.name.lower()}"} if language else {}
        )

        # Take the end time of the transcription
        end_time_transcription = time.time()
        # Print the time taken for transcribing the audios
        elapsed_time_transcription = end_time_transcription - start_time_transcription
        logger.info(f"Time taken for transcribing the audios: {elapsed_time_transcription:.2f} seconds")

        # Rename the "timestamp" key to "timestamps"
        transcriptions = _rename_key_recursive(transcriptions, "timestamp", "timestamps")

        # Convert the pipeline output to ScriptLine objects
        return [ScriptLine.from_dict(t) for t in transcriptions]
