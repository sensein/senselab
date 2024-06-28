"""Tests for forced alignment functions."""

from typing import Optional, TypedDict

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.forced_alignment.data_structures import (
    Point,
    SingleSegment,
)
from senselab.audio.tasks.forced_alignment.forced_alignment import (
    _align_segments,
    _align_transcription,
    _can_align_segment,
    _get_prediction_matrix,
    _interpolate_nans,
    _merge_repeats,
    _prepare_audio,
    _prepare_waveform_segment,
    _preprocess_segments,
    align_transcriptions,
)
from senselab.utils.data_structures.script_line import ScriptLine


class SingleCharSegment(TypedDict):
    """A single char of a speech."""

    char: str
    start: Optional[float]
    end: Optional[float]
    score: float


@pytest.fixture
def dummy_audio() -> Audio:
    """Fixture for dummy audio."""
    waveform = np.random.rand(1, 16000)
    return Audio(waveform=waveform, sampling_rate=16000)


@pytest.fixture
def dummy_segment() -> SingleSegment:
    """Fixture for a dummy segment."""
    return SingleSegment(
        start=0.0,
        end=1.0,
        text="test",
        clean_char=["t", "e", "s", "t"],
        clean_cdx=[0, 1, 2, 3],
        clean_wdx=[0],
        sentence_spans=None,
    )


@pytest.fixture
def dummy_model() -> tuple:
    """Fixture for a dummy model and processor."""
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return model, processor


def test_converts_numpy_to_tensor(dummy_audio: Audio) -> None:
    """Test conversion of numpy array to tensor."""
    prepared_audio = _prepare_audio(dummy_audio)
    assert torch.is_tensor(prepared_audio.waveform)
    assert prepared_audio.waveform.shape == (1, 16000)


def test_preprocess_segments() -> None:
    """Test preprocessing of segments."""
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    preprocessed_segments = _preprocess_segments(transcript, model_dictionary, "en", False, False)
    assert preprocessed_segments[0]["clean_char"] == ["t", "e", "s", "t"]


def test_can_align_segment(dummy_segment: SingleSegment) -> None:
    """Test if a segment can be aligned."""
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    assert _can_align_segment(dummy_segment, model_dictionary, 0.0, 10.0)


def test_prepare_waveform_segment(dummy_audio: Audio) -> None:
    """Test preparation of waveform segment."""
    waveform_segment, lengths = _prepare_waveform_segment(dummy_audio, 0.0, 1.0, "cpu")
    assert waveform_segment.shape == (1, 16000)


def test_get_prediction_matrix(dummy_model: tuple) -> None:
    """Test generation of prediction matrix."""
    model, _ = dummy_model
    waveform_segment = torch.randn(1, 16000)
    prediction_matrix = _get_prediction_matrix(model, waveform_segment, None, "huggingface", "cpu")
    assert prediction_matrix.shape[0] > 0


def test_merge_repeats() -> None:
    """Test merging of repeated tokens."""
    path = [Point(0, 0, 1.0), Point(0, 1, 1.0), Point(1, 2, 1.0)]
    transcript = "test"
    segments = _merge_repeats(path, transcript)
    assert len(segments) == 2


def test_interpolate_nans() -> None:
    """Test interpolation of NaN values."""
    series = pd.Series([0.0, np.nan, 2.0])
    interpolated_series = _interpolate_nans(series)
    assert interpolated_series.isnull().sum() == 0


def test_align_segments(dummy_audio: Audio, dummy_model: tuple) -> None:
    """Test alignment of segments."""
    model, processor = dummy_model
    model_dictionary = processor.tokenizer.get_vocab()

    # Create a sample transcript
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]

    # Preprocess the transcript segments
    preprocessed_transcript = _preprocess_segments(
        transcript, model_dictionary, model_lang="en", print_progress=False, combined_progress=False
    )

    # Ensure the model dictionary has the necessary keys
    for char in "test":
        if char not in model_dictionary:
            model_dictionary[char] = len(model_dictionary)

    aligned_segments, word_segments = _align_segments(
        transcript=preprocessed_transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang="en",
        model_type="huggingface",
        audio=dummy_audio,
        device="cpu",
        max_duration=10.0,
        return_char_alignments=False,
        interpolate_method="nearest",
    )
    assert isinstance(aligned_segments, list)
    assert isinstance(word_segments, list)


def test_align_transcription(dummy_audio: Audio, dummy_model: tuple) -> None:
    """Test alignment of transcription."""
    model, processor = dummy_model
    transcript = [
        SingleSegment(
            start=0.0,
            end=1.0,
            text="test",
            clean_char=["t", "e", "s", "t"],
            clean_cdx=[0, 1, 2, 3],
            clean_wdx=[0],
            sentence_spans=None,
        )
    ]
    aligned_result = _align_transcription(
        transcript=transcript,
        model=model,
        align_model_metadata={
            "dictionary": processor.tokenizer.get_vocab(),
            "language": "en",
            "type": "huggingface",
        },
        audio=dummy_audio,
        device="cpu",
    )
    assert "segments" in aligned_result
    assert "word_segments" in aligned_result


def test_align_transcriptions(dummy_audio: Audio) -> None:
    """Test alignment of transcriptions."""
    audios = [dummy_audio]
    transcriptions = [ScriptLine(text="test", start=0.0, end=1.0)]
    aligned_transcriptions = align_transcriptions(audios=audios, transcriptions=transcriptions)
    assert len(aligned_transcriptions) == 1
    assert len(aligned_transcriptions[0]) == 1
    assert aligned_transcriptions[0][0].text == "test"


if __name__ == "__main__":
    pytest.main()
