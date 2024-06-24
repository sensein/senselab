"""Module for testing speech-to-text evaluation."""

from senselab.audio.tasks.speech_to_text.speech_to_text_evaluation import (
    calculate_cer,
    calculate_mer,
    calculate_wer,
    calculate_wil,
    calculate_wip,
)


def test_calculate_wer() -> None:
    """Tests the calculation of Word Error Rate (WER)."""
    reference = "hello world"
    hypothesis = "hello duck"
    expected_wer = 0.5

    wer = calculate_wer(reference, hypothesis)

    assert wer == expected_wer, f"Expected WER: {expected_wer}, but got: {wer}"


def test_calculate_mer() -> None:
    """Tests the calculation of Match Error Rate (MER)."""
    reference = "hello world"
    hypothesis = "hello duck"
    expected_mer = 0.5

    mer = calculate_mer(reference, hypothesis)

    assert mer == expected_mer, f"Expected MER: {expected_mer}, but got: {mer}"


def test_calculate_wil() -> None:
    """Tests the calculation of Word Information Lost (WIL)."""
    reference = "hello world"
    hypothesis = "hello duck"
    expected_wil = 0.75

    wil = calculate_wil(reference, hypothesis)

    assert wil == expected_wil, f"Expected WIL: {expected_wil}, but got: {wil}"


def test_calculate_wip() -> None:
    """Tests the calculation of Word Information Preserved (WIP)."""
    reference = "hello world"
    hypothesis = "hello duck"
    expected_wip = 0.25

    wip = calculate_wip(reference, hypothesis)

    assert wip == expected_wip, f"Expected WIP: {expected_wip}, but got: {wip}"


def test_calculate_cer() -> None:
    """Tests the calculation of Character Error Rate (CER)."""
    reference = "hello world"
    hypothesis = "hello duck"
    expected_cer = 0.45454545454545453

    cer = calculate_cer(reference, hypothesis)

    assert cer == expected_cer, f"Expected CER: {expected_cer}, but got: {cer}"
