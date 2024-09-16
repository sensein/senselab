"""API module for emotional analysis."""

from typing import Any, Dict, List, Optional, Union

from senselab.text.tasks.emotion_analysis.constants import MODEL_TYPE_TO_UTILS
from senselab.text.tasks.emotion_analysis.emotion_analysis import EmotionAnalysis
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def analyze_emotion(
    pieces_of_text: List[str],
    model: SenselabModel = HFModel(path_or_uri="j-hartmann/emotion-english-distilroberta-base", revision="main"),
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[Dict[str, Union[str, float]]]:
    """Analyze emotion of given text pieces.

    Args:
        pieces_of_text (List[str]): List of text strings to analyze.
        model (SenselabModel): The model to use for emotional analysis.
        device (Optional[DeviceType]): The device to use for computation.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing emotional analysis results.
    """
    model_type = type(model)
    model_utils = MODEL_TYPE_TO_UTILS.get(model_type)

    if model_utils is None:
        raise NotImplementedError(f"The specified model '{model_type}' is not supported for now.")

    return EmotionAnalysis.analyze(
        input_data=pieces_of_text, model_utils=model_utils.get_instance(model), device=device, **kwargs
    )
