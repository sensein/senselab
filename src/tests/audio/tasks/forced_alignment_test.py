"""Tests for forced alignment functions."""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.forced_alignment.data_structures import (
    Point,
    SingleSegment,
)
from senselab.audio.tasks.forced_alignment.forced_alignment import (
    _align_segments,
    _can_align_segment,
    _get_prediction_matrix,
    _interpolate_nans,
    _merge_repeats,
    _preprocess_segments,
    align_transcriptions,
)
from senselab.utils.data_structures import DeviceType, Language, ScriptLine


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


@pytest.fixture
def script_line_fixture() -> ScriptLine:
    """Pytest fixture to create a ScriptLine object.

    Returns:
        ScriptLine: An instance of ScriptLine.
    """
    data = {
        "text": "test",
        "speaker": "Speaker Name",
        "start": 0.0,
        "end": 10.0,
        "chunks": [
            {"text": "Chunk 1 text", "speaker": "Chunk 1 speaker", "start": 0.0, "end": 5.0},
            {"text": "Chunk 2 text", "speaker": "Chunk 2 speaker", "start": 5.0, "end": 10.0},
        ],
    }
    return ScriptLine.from_dict(data)


@pytest.fixture
def script_line_fixture_curiosity() -> ScriptLine:
    """Fixture for a ScriptLine of: "I had that curiosity beside me at this moment".

    Source: https://pytorch.org/audio/main/tutorials/ctc_forced_alignment_api_tutorial.html
    Note: This example was not manually aligned; the timings are from automatic alignment.
    """
    return ScriptLine(
        text="I had that curiosity beside me at this moment",
        start=0.644,
        end=3.138,
        chunks=[
            ScriptLine(text="i", start=0.644, end=0.664),
            ScriptLine(text="had", start=0.704, end=0.845),
            ScriptLine(text="that", start=0.885, end=1.026),
            ScriptLine(text="curiosity", start=1.086, end=1.790),
            ScriptLine(text="beside", start=1.871, end=2.314),
            ScriptLine(text="me", start=2.334, end=2.414),
            ScriptLine(text="at", start=2.495, end=2.575),
            ScriptLine(text="this", start=2.595, end=2.756),
            ScriptLine(text="moment", start=2.837, end=3.138),
        ],
    )


def test_preprocess_segments() -> None:
    """Test preprocessing of segments."""
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]
    model_dictionary = {"T": 0, "E": 1, "S": 2}
    preprocessed_segments = _preprocess_segments(
        transcript, model_dictionary, Language(language_code="en"), False, False
    )
    assert preprocessed_segments[0]["clean_char"] == ["T", "E", "S", "T"]


def test_can_align_segment(dummy_segment: SingleSegment) -> None:
    """Test if a segment can be aligned."""
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    assert _can_align_segment(dummy_segment, model_dictionary, 0.0, 10.0)


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


def test_get_prediction_matrix(dummy_model: tuple) -> None:
    """Test generation of prediction matrix."""
    model, _ = dummy_model
    waveform_segment = torch.randn(1, 16000)
    prediction_matrix = _get_prediction_matrix(model, waveform_segment, None, "huggingface", DeviceType.CPU)
    assert prediction_matrix.shape[0] > 0


def test_align_segments(mono_audio_sample: Audio, dummy_model: tuple) -> None:
    """Test alignment of segments."""
    model, processor = dummy_model
    model_dictionary = processor.tokenizer.get_vocab()

    # Create a sample transcript and preprocess
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]
    model_dictionary = {"T": 0, "E": 1, "S": 2}
    transcript = _preprocess_segments(transcript, model_dictionary, Language(language_code="en"), False, False)

    # Ensure the model dictionary has the necessary keys
    for char in "test":
        if char not in model_dictionary:
            model_dictionary[char] = len(model_dictionary)

    # Call the alignment function
    aligned_segments = _align_segments(
        transcript=transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang=Language(language_code="en"),
        model_type="huggingface",
        audio=mono_audio_sample,
        device=DeviceType.CPU,
        max_duration=10.0,
    )

    # Validate results
    assert isinstance(aligned_segments, list)
    assert all(isinstance(segment, (ScriptLine, type(None))) for segment in aligned_segments)


def test_align_transcriptions_fixture(resampled_mono_audio_sample: Audio, script_line_fixture: ScriptLine) -> None:
    """Test alignment of transcriptions."""
    audios_and_transcriptions_and_language = [
        (resampled_mono_audio_sample, script_line_fixture, Language(language_code="en")),
        (resampled_mono_audio_sample, script_line_fixture, Language(language_code="fr")),
    ]
    aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
    assert len(aligned_transcriptions) == 2
    assert len(aligned_transcriptions[0]) == 1
    if aligned_transcriptions[0][0]:
        assert aligned_transcriptions[0][0].text == "test"


def test_align_transcriptions_multilingual(resampled_mono_audio_sample: Audio, script_line_fixture: ScriptLine) -> None:
    """Test alignment of transcriptions."""
    languages = ["de", "es"]
    expected_text = "test"  # Replace with the appropriate expected text for your fixtures

    for lang in languages:
        audios_and_transcriptions_and_language = [
            (resampled_mono_audio_sample, script_line_fixture, Language(language_code=lang))
        ]
        aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
        assert len(aligned_transcriptions) == 1, f"Failed for language: {lang}"
        assert len(aligned_transcriptions[0]) == 1, f"Failed for language: {lang}"
        if aligned_transcriptions[0][0]:
            assert aligned_transcriptions[0][0].text == expected_text, f"Failed for language: {lang}"


def test_align_transcriptions_curiosity_audio_fixture(
    resampled_had_that_curiosity_audio_sample: Audio, script_line_fixture_curiosity: ScriptLine
) -> None:
    """Test alignment of transcriptions using the 'had that curiosity' audio sample and fixture."""
    audios_and_transcriptions_and_language = [
        (resampled_had_that_curiosity_audio_sample, script_line_fixture_curiosity, Language(language_code="en"))
    ]
    aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
    assert len(aligned_transcriptions[0]) == 1
    if aligned_transcriptions[0][0] is not None:
        assert aligned_transcriptions[0][0].text == script_line_fixture_curiosity.text


if __name__ == "__main__":
    pytest.main()
