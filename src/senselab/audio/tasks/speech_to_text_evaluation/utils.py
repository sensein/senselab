"""This module implements some utilities for evaluating a transcription."""

try:
    import jiwer

    JIWER_AVAILABLE = True
except ModuleNotFoundError:
    JIWER_AVAILABLE = False

# TODO: add more metrics which take into account the meaning/intention


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate the Word Error Rate (WER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WER score.

    Examples:
        >>> calculate_wer("hello world", "hello duck")
        0.5
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )

    return jiwer.wer(reference, hypothesis)


def calculate_mer(reference: str, hypothesis: str) -> float:
    """Calculate the Match Error Rate (MER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The MER score.

    Examples:
        >>> calculate_mer("hello world", "hello duck")
        0.5
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )
    return jiwer.mer(reference, hypothesis)


def calculate_wil(reference: str, hypothesis: str) -> float:
    """Calculate the Word Information Lost (WIL) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WIL score.

    Examples:
        >>> calculate_wil("hello world", "hello duck")
        0.75
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )
    return jiwer.wil(reference, hypothesis)


def calculate_wip(reference: str, hypothesis: str) -> float:
    """Calculate the Word Information Preserved (WIP) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The WIP score.

    Examples:
        >>> calculate_wip("hello world", "hello duck")
        0.25
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )
    return jiwer.wip(reference, hypothesis)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate the Character Error Rate (CER) between the reference and hypothesis.

    Args:
        reference (str): The ground truth text.
        hypothesis (str): The predicted text.

    Returns:
        float: The CER score.

    Examples:
        >>> calculate_cer("hello world", "hello duck")
        0.45454545454545453
    """
    if not JIWER_AVAILABLE:
        raise ModuleNotFoundError(
            "`jiwer` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab['audio']`."
        )
    cer_value = jiwer.cer(reference, hypothesis)
    if isinstance(cer_value, dict):
        return float(cer_value.get("cer", 0.0))  # Extract CER if returned in a dictionary
    return float(cer_value)  # Ensure output is always a float
