"""This module defines APIs for the user to run Speech Emotion Recognition (SER) on sets of audios."""

import warnings
from typing import Dict, List, Optional, Tuple

from transformers import AutoConfig, pipeline

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType, _select_device_and_dtype
from senselab.utils.data_structures.model import HFModel


def audio_classification_with_hf_models(
    audios: List[Audio], model: HFModel, device: Optional[DeviceType] = None
) -> List[List[Dict]]:
    """General audio classification functionality utilitzing HuggingFace pipelines.

    Classifies all audios, with no underlying assumptions on what the classification labels are,
    and returns the output that the pipeline gives.

    Args:
        audios: List of Audio objects that we want to run classification on
        model: The HuggingFace model that will be used for running the inference
        device: The device to run inference on

    Returns:
        List of Lists of Dictionaries where each corresponds to the audio that it was ran on and the List of
            Dictionaries are of the form [{'label': 'some_label', 'score': some_value},...]

    Raises:
        ValueError if the given model does not have the audio-classification pipeline tag
        UserWarning if the model tags don't include endpoints_compatible (seen on HuggingFace as Inference Endpoints)
            as the behavior of the model might not output as expected as a result.
    """
    model_info = model.get_model_info()

    if "audio-classification" not in model_info.pipeline_tag:
        raise ValueError(f"The model '{model.path_or_uri}' is not suitable for audio classification. SORRY!")
    if "endpoints_compatible" not in model_info.tags:
        warnings.warn(
            UserWarning(f"The model '{model.path_or_uri}' has not been tagged as an Inference Endpoint and \
                                  so we cannot guarantee its input and outputs are as expected")
        )

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])

    classification_pipeline = pipeline(
        task="audio-classification",
        model=model.path_or_uri,
        revision=model.revision,
        device=device.value,
    )
    classification_outputs = []

    # TODO: figure out adding batching and GPU support
    for audio in audios:
        classification_outputs.append(classification_pipeline(audio.waveform.numpy().squeeze()))

    return classification_outputs


def speech_emotion_recognition_with_hf_models(
    audios: List[Audio], model: HFModel, device: Optional[DeviceType] = None
) -> List[Tuple[str, Dict]]:
    """Function for running speech emotion recognition tasks using HuggingFace models.

    Uses an audio classification pipeline to run speech emotion recognition for every audio in audios using the
    specified model. For each audio, we return the emotion that the model suggested has the highest probability
    for single label classification as well as the full emotion output for multi-label classification. Function can
    also be used for continuous speech emotion recognition, where the second element of each tuple will contain a
    dictionary of the predicted continuous values.

    Args:
        audios: List of Audio objects that we want to run speech emotion recognition on. If you wish to get the
            emotion at different segments of the audio (e.g. when different speaker talk) please run the audios
            through an appropriate segmentation task.
        model: The HuggingFace model that will be used for running the inference
        device: The device to run inference on

    Returns:
        List of tuples where the first value is the single label classification of the audio (if appropriate) and the
            second value is the full model output, which might be the probabilities for each evaluated emotion
            (for discrete SER) or the continuous emotion predictions.

    Raises:
        ValueError if the given model is not properly tagged with 'speech-emotion-recognition' or 'emotion-recognition'
            or the model configuration does not contain an id2label property that predicts commonly used
            emotions (happy, sad, neutral, positive, negative, etc.)
    """
    model_info = model.get_model_info()

    tags = model_info.tags

    if not ("speech-emotion-recognition" in tags or "emotion-recognition" in tags) and not _are_emotions_in_config(
        model
    ):
        raise ValueError(
            f"The model '{model.path_or_uri}' is not suitable for speech emotion recognition. Please +"
            "validate that it has the correct tags or use the more generic "
            + "'audio_classification_with_hf_models' function."
        )

    audio_classifications = audio_classification_with_hf_models(audios, model, device)
    # print(audio_classifications)
    ser_output = []
    for classification in audio_classifications:
        classification_output = {}
        for label_score in classification:
            classification_output[label_score["label"]] = label_score["score"]
        single_classification = max(classification_output, key=lambda x: classification_output[x])
        ser_output.append((single_classification, classification_output))
    return ser_output


def _are_emotions_in_config(model: HFModel) -> bool:
    config = AutoConfig.from_pretrained(model.path_or_uri)
    id2label = config.id2label
    # print(id2label)
    if id2label:
        # id2label = model.model_info.config['id2label']
        labels = list(id2label.values())
        if "positive" in labels and "negative" in labels and "neutral" in labels:  # Simple, discrete
            return True
        if "valence" in labels or "arousal" in labels or "dominance" in labels:  # conitnuous emotions
            return True
        if (
            "happy" in labels
            or "happiness" in labels
            or "sad" in labels
            or "sadness" in labels
            or "fear" in labels
            or "anger" in labels
            or "disgust" in labels
        ):
            return True
    return False
