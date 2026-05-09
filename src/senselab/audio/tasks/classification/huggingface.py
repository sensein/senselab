"""This module provides a factory for managing Hugging Face audio classification pipelines.

To ensure correct functionality, call `classify_audios_with_transformers` serially or,
if you need to process multiple audios in parallel, pass the entire list of audios to the
function at once, rather than calling the function with one audio at a time.
"""

import os
import time
from typing import Dict, List, Literal, Optional, cast

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, Pipeline, pipeline

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger

# Phase 2: head-load diagnostic for the standard pipeline path.
# The PR-#511 family of bugs (random-init head silently produces ~uniform softmax)
# can also fire here for any audio-classification model whose checkpoint head doesn't
# match the standard ``Wav2Vec2ForSequenceClassification``-style flat ``classifier``.
# We surface that as a warning by default; ``SENSELAB_STRICT_HEAD_LOAD=1`` promotes
# it to a hard ``RuntimeError``. Key prefixes that count as "head" weights — anything
# else (encoder/feature_extractor) we ignore because transformers correctly skips
# unused weights and these aren't the silent-failure surface.
_SUSPECT_HEAD_PREFIXES: tuple[str, ...] = ("classifier.", "head.", "score.", "out_proj.", "projector.")


def _check_head_loaded_cleanly(loading_info: dict, model: HFModel) -> None:
    """Warn (or raise, in strict mode) if the loaded checkpoint left head weights random.

    transformers' loader silently random-initializes any module that doesn't have a
    matching key in the checkpoint. For audio classification this is the difference
    between meaningful scores and ~uniform softmax — the bug that motivated PR #511
    on the SER-specific path. This guard surfaces the same condition for the generic
    audio-classification path. Default: ``logger.warning``. Set
    ``SENSELAB_STRICT_HEAD_LOAD=1`` to promote to a hard error.
    """
    missing_head = sorted(
        k for k in loading_info.get("missing_keys", set()) if any(k.startswith(p) for p in _SUSPECT_HEAD_PREFIXES)
    )
    mismatched_head = sorted(
        (k[0] if isinstance(k, tuple) else k)
        for k in loading_info.get("mismatched_keys", set())
        if any((k[0] if isinstance(k, tuple) else k).startswith(p) for p in _SUSPECT_HEAD_PREFIXES)
    )
    if not (missing_head or mismatched_head):
        return

    msg = (
        f"Audio classifier head loaded with suspect weights for {model.path_or_uri} "
        f"(revision={model.revision or 'main'}): missing={missing_head}, "
        f"mismatched_shape={mismatched_head}. The pipeline may emit ~uniform softmax. "
        f"If this checkpoint has a custom head, use the SER-specific path "
        f"(senselab.audio.tasks.classification.speech_emotion_recognition.api) or extend "
        f"its head registry. Set SENSELAB_STRICT_HEAD_LOAD=1 to make this a hard error."
    )
    if os.environ.get("SENSELAB_STRICT_HEAD_LOAD", "0") == "1":
        raise RuntimeError(msg)
    logger.warning(msg)


class HuggingFaceAudioClassifier:
    """A factory for managing Hugging Face audio classification pipelines."""

    _pipelines: Dict[str, Pipeline] = {}

    @classmethod
    def _get_hf_audio_classification_pipeline(
        cls,
        model: HFModel,
        top_k: Optional[int],
        function_to_apply: Literal["softmax", "sigmoid", "none"],
        batch_size: int,
        device: Optional[DeviceType] = None,
    ) -> Pipeline:
        """Get or create a Hugging Face audio classification pipeline.

        Args:
            model (HFModel): The Hugging Face model.
            top_k (int): Number of top labels that will be returned by the model.
            function_to_apply (str): Function to apply to the model's output, defaults to softmax.
            batch_size (int): The batch size for processing.
            device (Optional[DeviceType]): The device to run the model on.

        Returns:
            Pipeline: The Audio Classification pipeline.
        """
        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        # Cache by model identity + device only. top_k, function_to_apply,
        # and batch_size are call-time parameters passed at pipe() invocation.
        key = f"{model.path_or_uri}-{model.revision}-{device.value}"
        if key not in cls._pipelines:
            # Phase 2: load the model explicitly so we can inspect ``loading_info``
            # for the silent-random-head failure mode that motivated PR #511.
            # ``pipeline()`` accepts a pre-loaded model + feature_extractor, so this
            # only adds the inspection without a second download.
            loaded, loading_info = AutoModelForAudioClassification.from_pretrained(  # type: ignore[call-overload]
                str(model.path_or_uri),
                revision=model.revision,
                output_loading_info=True,
            )
            _check_head_loaded_cleanly(loading_info, model)
            feature_extractor = AutoFeatureExtractor.from_pretrained(str(model.path_or_uri), revision=model.revision)
            loaded = loaded.to(device.value)
            cls._pipelines[key] = cast(
                Pipeline,
                pipeline(  # type: ignore[call-overload]
                    task="audio-classification",
                    model=loaded,
                    feature_extractor=feature_extractor,
                    device=device.value,
                ),
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
        if not audios:
            return []

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

        feature_extractor = getattr(pipe, "feature_extractor", None)
        if feature_extractor is None:
            raise ValueError("Internal error: The Hugging Face pipeline does not have a feature extractor.")

        # Retrieve the expected sampling rate from the Hugging Face model
        expected_sampling_rate = cast(int, getattr(feature_extractor, "sampling_rate", None))  # type: ignore[attr-defined]
        if expected_sampling_rate is None:
            raise ValueError("Internal error: The Hugging Face model does not specify an expected sampling rate.")

        # Validate mono and resample to expected rate inside the loop
        # to avoid holding all resampled audios in memory at once
        formatted_audios = []
        for audio in audios:
            if audio.waveform.shape[0] != 1:
                raise ValueError(f"Stereo audio is not supported. Got {audio.waveform.shape[0]} channels")
            if audio.sampling_rate != expected_sampling_rate:
                audio = resample_audios([audio], resample_rate=expected_sampling_rate)[0]
            formatted_audios.append(_audio_to_huggingface_dict(audio))

        # Take the start time of the transcription
        start_time_transcription = time.time()
        # Run the pipeline
        classifications = pipe(formatted_audios, top_k=top_k, function_to_apply=function_to_apply)  # type: ignore[call-arg]

        # Take the end time of the classification
        end_time_transcription = time.time()
        # Print the time taken for classifying the audios
        elapsed_time_transcription = end_time_transcription - start_time_transcription
        logger.info(f"Time taken for classifying the audios: {elapsed_time_transcription:.2f} seconds")

        # Convert the pipeline output to AudioClassificationResult objects
        return [AudioClassificationResult.from_hf_classification_pipeline(t) for t in classifications]
