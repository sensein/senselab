"""Tests for forced alignment functions."""

import os
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
    _preprocess_segments,
    align_transcriptions,
)
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.script_line import ScriptLine

MONO_AUDIO_PATH = "src/tests/data_for_testing/audio_48khz_mono_16bits.wav"


@pytest.fixture
def mono_audio_sample() -> Audio:
    """Fixture for sample mono audio."""
    return Audio.from_filepath(MONO_AUDIO_PATH)


class SingleCharSegment(TypedDict):
    """A single char of a speech."""

    char: str
    start: Optional[float]
    end: Optional[float]
    score: float


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


if os.getenv("GITHUB_ACTIONS") != "true":

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
            audio=mono_audio_sample,
            device=DeviceType.CPU,
            max_duration=10.0,
            return_char_alignments=False,
            interpolate_method="nearest",
        )
        assert isinstance(aligned_segments, list)
        assert isinstance(word_segments, list)

    def test_align_transcription(mono_audio_sample: Audio, dummy_model: tuple) -> None:
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
            audio=mono_audio_sample,
            device=DeviceType.CPU,
        )
        assert "segments" in aligned_result
        assert "word_segments" in aligned_result

    def test_align_transcriptions(mono_audio_sample: Audio, script_line_fixture: ScriptLine) -> None:
        """Test alignment of transcriptions."""
        audios_and_transcriptions = [(mono_audio_sample, script_line_fixture)]
        aligned_transcriptions = align_transcriptions(audios_and_transcriptions)
        assert len(aligned_transcriptions) == 1
        assert len(aligned_transcriptions[0]) == 1
        assert aligned_transcriptions[0][0].text == "test"


if __name__ == "__main__":
    pytest.main()
