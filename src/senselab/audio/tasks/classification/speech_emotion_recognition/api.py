"""This module represents the API for the running speech emotion recognition within the senselab package.

Currently, it supports only Hugging Face models, with plans to include more in the future.
Users can specify the audio clips to classify, the classification model, the preferred device,
and the model-specific parameters, and senselab handles the rest.
"""

import warnings
from enum import Enum
from typing import Any, List, Optional

from transformers import AutoConfig

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel, logger


class SERType(Enum):
    """SER types for determining model output behaviors."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    VALENCE = "valence"
    NOT_RECOGNIZED = "not recognized"


def classify_emotions_from_speech(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify all audios using the given speech emotion recognition model.

    Args:
        audios (List[Audio]): The list of audio objects to be classified.
        model (SenselabModel): The model used for classification, should be trained for recognizing emotions.
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the classification function.

    Returns:
        List[AudioClassificationResult]: The list of speech emotion recognition results.

    Todo:
        - Include more models (e.g., speechbrain, nvidia)
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
                f"The model '{model.path_or_uri}' is not suitable for speech emotion recognition. Please +"
                "validate that it has the correct tags or use the more generic "
                + "'audio_classification_with_hf_models' function."
            )

        if ser_type == SERType.CONTINUOUS:
            output_function_to_apply = kwargs.get("function_to_apply", None)
            if output_function_to_apply:
                if output_function_to_apply != "none":
                    warnings.warn("""Senselab predicts that you are using a continuous SER model but have
                                  specified the parameter `function_to_apply` as something other than none. This
                                  might create side effects when dealing with continuous values that do not
                                  necessarily represent probabilities.""")
            else:
                kwargs["function_to_apply"] = "none"
                logger.info("""Senselab predicts that you are using a continuous SER and have not specified the
                            parameter `function_to_apply`. We are setting this to none such that the model
                            outputs are reported directly rather than being passed through a softmax or sigmoid.""")
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError(
            "Only Hugging Face models are supported for now. We aim to support more models in the future."
        )


def _get_ser_type(model: HFModel) -> SERType:
    """Get the type of SER the model is likely used for based on the labels it is set to predict."""
    config = AutoConfig.from_pretrained(model.path_or_uri)
    id2label = config.id2label
    # print(id2label)
    if id2label:
        # id2label = model.model_info.config['id2label']
        labels = list(id2label.values())
        if "positive" in labels and "negative" in labels and "neutral" in labels:  # Simple, discrete
            return SERType.VALENCE
        if "valence" in labels or "arousal" in labels or "dominance" in labels:  # conitnuous emotions
            return SERType.CONTINUOUS
        if (
            "happy" in labels
            or "happiness" in labels
            or "joy" in labels
            or "sad" in labels
            or "sadness" in labels
            or "fear" in labels
            or "fearful" in labels
            or "anger" in labels
            or "angry" in labels
            or "disgust" in labels
            or "neutral" in labels
            or "calm" in labels
            or "surprised" in labels
        ):
            return SERType.DISCRETE
    return SERType.NOT_RECOGNIZED
