"""API module for sentiment analysis."""

from typing import Dict, List, Optional, Union

from senselab.text.tasks.sentiment_analysis.constants import MODEL_TYPE_TO_UTILS
from senselab.text.tasks.sentiment_analysis.sentiment_analysis import SentimentAnalysis
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def analyze_sentiment(
    pieces_of_text: List[str],
    model: SenselabModel = HFModel(path_or_uri="distilbert-base-uncased-finetuned-sst-2-english", revision="main"),
    device: Optional[DeviceType] = None,
    **kwargs: Union[str, int, float, bool],
) -> List[Dict[str, Union[str, float]]]:
    """Analyze sentiment of given text pieces.

    Args:
        pieces_of_text (List[str]): List of text strings to analyze.
        model (SenselabModel): The model to use for sentiment analysis.
        device (Optional[DeviceType]): The device to use for computation.
        **kwargs (Union[str, int, float, bool]): Additional keyword arguments.

    Returns:
        List[Dict[str, Union[str, float]]]: A list of dictionaries containing sentiment analysis results.
    """
    model_type = type(model)
    model_utils = MODEL_TYPE_TO_UTILS.get(model_type)

    if model_utils is None:
        raise NotImplementedError(f"The specified model '{model_type}' is not supported for now.")

    return SentimentAnalysis.analyze(
        input_data=pieces_of_text, model_utils=model_utils.get_instance(model), device=device, **kwargs
    )
