"""This module implements some utilities for evaluating a transcription."""

import jiwer

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
    return jiwer.cer(reference, hypothesis)
