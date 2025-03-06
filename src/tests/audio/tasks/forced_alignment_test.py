"""Tests for forced alignment functions."""

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, Language, ScriptLine

try:
    import nltk  # noqa: F401

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import torchaudio  # noqa: F401

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

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False


@pytest.fixture
def dummy_segment() -> "SingleSegment":
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


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
def test_preprocess_segments_nltk_not_installed() -> None:
    """Test preprocessing of segments when nltk is not installed."""
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    with pytest.raises(ImportError):
        _preprocess_segments(transcript, model_dictionary, Language(language_code="en"), False, False)


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
def test_preprocess_segments() -> None:
    """Test preprocessing of segments."""
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    preprocessed_segments = _preprocess_segments(
        transcript, model_dictionary, Language(language_code="en"), False, False
    )
    assert preprocessed_segments[0]["clean_char"] == ["t", "e", "s", "t"]


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_can_align_segment(dummy_segment: "SingleSegment") -> None:
    """Test if a segment can be aligned."""
    model_dictionary = {"t": 0, "e": 1, "s": 2}
    assert _can_align_segment(dummy_segment, model_dictionary, 0.0, 10.0)


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_merge_repeats() -> None:
    """Test merging of repeated tokens."""
    path = [Point(0, 0, 1.0), Point(0, 1, 1.0), Point(1, 2, 1.0)]
    transcript = "test"
    segments = _merge_repeats(path, transcript)
    assert len(segments) == 2


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_interpolate_nans() -> None:
    """Test interpolation of NaN values."""
    series = pd.Series([0.0, np.nan, 2.0])
    interpolated_series = _interpolate_nans(series)
    assert interpolated_series.isnull().sum() == 0


@pytest.mark.skipif(not TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")
def test_get_prediction_matrix(dummy_model: tuple) -> None:
    """Test generation of prediction matrix."""
    model, _ = dummy_model
    waveform_segment = torch.randn(1, 16000)
    prediction_matrix = _get_prediction_matrix(model, waveform_segment, None, "huggingface", DeviceType.CPU)
    assert prediction_matrix.shape[0] > 0


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
def test_align_segments(mono_audio_sample: Audio, dummy_model: tuple) -> None:
    """Test alignment of segments."""
    model, processor = dummy_model
    model_dictionary = processor.tokenizer.get_vocab()

    # Create a sample transcript
    transcript = [SingleSegment(start=0.0, end=1.0, text="test")]

    # Preprocess the transcript segments
    preprocessed_transcript = _preprocess_segments(
        transcript,
        model_dictionary,
        model_lang=Language(language_code="en"),
        print_progress=False,
        combined_progress=False,
    )

    # Ensure the model dictionary has the necessary keys
    for char in "test":
        if char not in model_dictionary:
            model_dictionary[char] = len(model_dictionary)

    aligned_segments, word_segments = _align_segments(
        transcript=preprocessed_transcript,
        model=model,
        model_dictionary=model_dictionary,
        model_lang=Language(language_code="en"),
        model_type="huggingface",
        audio=mono_audio_sample,
        device=DeviceType.CPU,
        max_duration=10.0,
        return_char_alignments=False,
        interpolate_method="nearest",
    )
    assert isinstance(aligned_segments, list)
    assert isinstance(word_segments, list)


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
def test_align_transcription_faked(resampled_mono_audio_sample: Audio, dummy_model: tuple) -> None:
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
            "language": Language(language_code="en"),
            "type": "huggingface",
        },
        audio=resampled_mono_audio_sample,
        device=DeviceType.CPU,
    )
    assert "segments" in aligned_result
    assert "word_segments" in aligned_result


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
def test_align_transcriptions_fixture(resampled_mono_audio_sample: Audio, script_line_fixture: ScriptLine) -> None:
    """Test alignment of transcriptions."""
    audios_and_transcriptions_and_language = [
        (resampled_mono_audio_sample, script_line_fixture, Language(language_code="en")),
        (resampled_mono_audio_sample, script_line_fixture, Language(language_code="fr")),
    ]
    aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
    assert len(aligned_transcriptions) == 2
    assert len(aligned_transcriptions[0]) == 1
    assert aligned_transcriptions[0][0].text == "test"


@pytest.mark.skipif(not NLTK_AVAILABLE or not TORCHAUDIO_AVAILABLE, reason="nltk or torchaudio are not installed")
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
        assert aligned_transcriptions[0][0].text == expected_text, f"Failed for language: {lang}"


if __name__ == "__main__":
    pytest.main()
