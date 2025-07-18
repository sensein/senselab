"""This module provides a factory for managing Hugging Face audio classification pipelines.

To ensure correct functionality, call `classify_audios_with_transformers` serially or,
if you need to process multiple audios in parallel, pass the entire list of audios to the
function at once, rather than calling the function with one audio at a time.
"""

import time
from typing import Dict, List, Literal, Optional

from transformers import pipeline

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger


class HuggingFaceAudioClassifier:
    """A factory for managing Hugging Face audio classification pipelines."""

    _pipelines: Dict[str, pipeline] = {}

    @classmethod
    def _get_hf_audio_classification_pipeline(
        cls,
        model: HFModel,
        top_k: Optional[int],
        function_to_apply: Literal["softmax", "sigmoid", "none"],
        batch_size: int,
        device: Optional[DeviceType] = None,
    ) -> pipeline:
        """Get or create a Hugging Face audio classification pipeline.

        Args:
            model (HFModel): The Hugging Face model.
            top_k (int): Number of top labels that will be returned by the model.
            function_to_apply (str): Function to apply to the model's output, defaults to softmax.
            batch_size (int): The batch size for processing.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            pipeline: The Audio Classification pipeline.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{top_k}-" f"{function_to_apply}-{batch_size}-{device.value}"
        if key not in cls._pipelines:
            cls._pipelines[key] = pipeline(
                "audio-classification",
                model=model.path_or_uri,
                revision=model.revision,
                # top_k=top_k, #TODO: this causes a bug in the pipeline that has been reported to transformers
                # https://github.com/huggingface/transformers/issues/35736
                function_to_apply=function_to_apply,  # TODO: parameter ignored in transformer code, bug reported
                # https://github.com/huggingface/transformers/issues/35739
                device=device.value,
            )
        return cls._pipelines[key]

    @classmethod
    def classify_audios_with_transformers(
        cls,
        audios: List[Audio],
        model: HFModel,
        top_k: Optional[int] = None,
        function_to_apply: Literal["softmax", "sigmoid", "none"] = "softmax",
        batch_size: int = 16,
        device: Optional[DeviceType] = None,
    ) -> List[AudioClassificationResult]:
        """Transcribes all audio samples in the dataset.

        Args:
            audios (List[Audio]): The list of audio objects to be classified.
            model (HFModel): The Hugging Face model used for classification.
                There is no default model since this task covers a wide range of models.
            top_k (Optional[int]): the number of top labels for the model to return.
                If no top_k, the model returns all of the labels in the configuration file.
            function_to_apply (str): The function to apply to the model's outputs (default is softmax).
            batch_size (int): The batch size for processing (default is 16).
            device (Optional[DeviceType]): The device to run the model on (default is None).

        Returns:
            List[AudioClassificationResult]: The list of classification results,
                where each result contains the classified label and their score.
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

        # Take the start time of the pipeline initialization
        start_time_pipeline = time.time()
        # Get the Hugging Face pipeline
        pipe = HuggingFaceAudioClassifier._get_hf_audio_classification_pipeline(
            model=model,
            top_k=top_k,
            function_to_apply=function_to_apply,
            batch_size=batch_size,
            device=device,
        )

        # Take the end time of the pipeline initialization
        end_time_pipeline = time.time()
        # Print the time taken for initialize the hugging face ASR pipeline
        elapsed_time_pipeline = end_time_pipeline - start_time_pipeline
        logger.info(
            f"Time taken to initialize the hugging face audio classification \
            pipeline: {elapsed_time_pipeline:.2f} seconds"
        )

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
        classifications = pipe(formatted_audios, top_k=top_k, function_to_apply=function_to_apply)

        # Take the end time of the classification
        end_time_transcription = time.time()
        # Print the time taken for classifying the audios
        elapsed_time_transcription = end_time_transcription - start_time_transcription
        logger.info(f"Time taken for classifying the audios: {elapsed_time_transcription:.2f} seconds")

        # Convert the pipeline output to AudioClassificationResult objects
        return [AudioClassificationResult.from_hf_classification_pipeline(t) for t in classifications]
