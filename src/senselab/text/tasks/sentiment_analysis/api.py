"""API module for sentiment analysis."""

from typing import Any, Dict, List, Optional

from senselab.text.tasks.sentiment_analysis.constants import MODEL_TYPE_TO_UTILS
from senselab.text.tasks.sentiment_analysis.sentiment_analysis import SentimentAnalysis
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel, SenselabModel


def analyze_sentiment(
    pieces_of_text: List[str],
    model: SenselabModel = HFModel(path_or_uri="cardiffnlp/twitter-xlm-roberta-base-sentiment", revision="main"),
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[Dict[str, Any]]:
    """Analyze sentiment of given text pieces.

    Args:
        pieces_of_text (List[str]): List of text strings to analyze.
        model (SenselabModel): The model to use for sentiment analysis.
        device (Optional[DeviceType]): The device to use for computation.
        **kwargs (Any): Additional keyword arguments.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing sentiment analysis results.
    """
    model_type = type(model)
    model_utils = MODEL_TYPE_TO_UTILS.get(model_type)

    if model_utils is None:
        raise NotImplementedError(f"The specified model '{model_type}' is not supported for now.")

    return SentimentAnalysis.analyze(
        input_data=pieces_of_text, model_utils=model_utils.get_instance(model), device=device, **kwargs
    )
