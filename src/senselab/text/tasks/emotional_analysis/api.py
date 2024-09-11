"""API module for emtoional analysis."""

from typing import Dict, List, Optional, Union

from senselab.text.tasks.emotional_analysis.emotional_analysis import EmotionalAnalysis
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import HFModel


def analyze_emotion(
    pieces_of_text: List[str],
    device: Optional[DeviceType] = None,
    model: Optional[HFModel] = None,
    max_length: int = 512,
    overlap: int = 128,
    **kwargs: Union[str, int, float, bool],
) -> List[Dict[str, Union[str, float]]]:
    """Analyze emotion of given text pieces.

    Args:
        pieces_of_text: List of text strings to analyze.
        device: The device to use for computation.
        model: The model to use for emotional analysis.
        max_length: The maximum length of input text.
        overlap: The amount of overlap between text pieces.
        **kwargs: Additional keyword arguments.

    Returns:
        A list of dictionaries containing emotional analysis results.
    """
    return EmotionalAnalysis.analyze(
        pieces_of_text=pieces_of_text, device=device, model=model, max_length=max_length, overlap=overlap, **kwargs
    )
