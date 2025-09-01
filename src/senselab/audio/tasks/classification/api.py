"""This module represents the API for the speech classification task within the senselab package.

Currently, it supports only Hugging Face models, with plans to include more in the future.
Users can specify the audio clips to classify, the classification model, the preferred device,
and the model-specific parameters, and senselab handles the rest.
"""

from typing import Any, List, Optional

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.utils.data_structures import DeviceType, HFModel, SenselabModel


def classify_audios(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify all audios using the given model.

    Args:
        audios (List[Audio]): The list of audio objects to be classified.
        model (SenselabModel): The model used for classification.
        device (Optional[DeviceType]): The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the classification function.

    Returns:
        List[AudioClassificationResult]: The list of classification results.

    Todo:
        - Include more models (e.g., speechbrain, nvidia)
    """
    if isinstance(model, HFModel):
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )
    else:
        raise NotImplementedError(
            "Only Hugging Face models are supported for now. We aim to support more models in the future."
        )
