"""This module contains unit tests for the EER function."""

import pytest
import torch

from senselab.utils.tasks.eer import compute_eer

try:
    import speechbrain

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


@pytest.mark.skipif(SPEECHBRAIN_AVAILABLE, reason="SpeechBrain is installed")
def test_compute_eer_import_error() -> None:
    """Test that a ModuleNotFoundError is raised when SpeechBrain is not installed."""
    with pytest.raises(ModuleNotFoundError):
        compute_eer(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5, 0.6]))


@pytest.mark.skipif(not SPEECHBRAIN_AVAILABLE, reason="SpeechBrain is not installed")
def test_compute_eer() -> None:
    """Test that the EER is computed correctly for perfectly separable data."""
    predictions = torch.tensor([0.6, 0.7, 0.8, 0.5])
    targets = torch.tensor([0.4, 0.3, 0.2, 0.1])
    eer, threshold = compute_eer(predictions, targets)
    # Since we expect perfect separation, the EER should be 0
    assert eer == 0.0, "EER should be 0 for perfectly separable data"
    assert 0 <= threshold <= 1, "Threshold should be between 0 and 1"


@pytest.mark.skipif(not SPEECHBRAIN_AVAILABLE, reason="SpeechBrain is not installed")
def test_compute_eer_random() -> None:
    """Test that the EER is computed correctly for random predictions and targets."""
    # Set random seed for reproducibility
    torch.manual_seed(42)
    predictions = torch.rand(100)
    targets = torch.randint(0, 2, (100,))
    eer, threshold = compute_eer(predictions, targets)
    assert isinstance(eer, float), "EER should be a float"
    assert isinstance(threshold, float), "Threshold should be a float"
    assert 0 <= eer <= 1, "EER should be between 0 and 1"
    assert 0 <= threshold <= 1, "Threshold should be between 0 and 1"
