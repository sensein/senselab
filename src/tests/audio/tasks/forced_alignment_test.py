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
from senselab.audio.tasks.forced_alignment.evaluation import compare_alignments
from senselab.audio.tasks.forced_alignment.forced_alignment import (
    _align_segments,
    _can_align_segment,
    _get_prediction_matrix,
    _interpolate_nans,
    _merge_repeats,
    _preprocess_segments,
    align_transcriptions,
    remove_chunks_by_level,
)
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import DeviceType, Language, ScriptLine
from senselab.utils.data_structures.model import HFModel


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


@pytest.fixture
def aligned_scriptline_fixture_resampled_mono_audio() -> ScriptLine:
    """Fixture for an aligned ScriptLine of resampled_mono_audio_fixture.

    Generated with the align_transcriptions function.
    """
    return ScriptLine(
        text="This is Peter. This is Johnny. Kenny. And Joe. We just wanted to take a minute to thank you.",
        start=0.10,
        end=4.88,
        chunks=[
            ScriptLine(
                text="This is Peter.",
                start=0.10,
                end=0.70,
                chunks=[
                    ScriptLine(
                        text="This",
                        start=0.10,
                        end=0.22,
                        chunks=[
                            ScriptLine(text="T", start=0.10, end=0.12),
                            ScriptLine(text="h", start=0.12, end=0.14),
                            ScriptLine(text="i", start=0.14, end=0.16),
                            ScriptLine(text="s", start=0.16, end=0.22),
                        ],
                    ),
                    ScriptLine(
                        text="is",
                        start=0.22,
                        end=0.34,
                        chunks=[
                            ScriptLine(text="i", start=0.26, end=0.30),
                            ScriptLine(text="s", start=0.30, end=0.34),
                        ],
                    ),
                    ScriptLine(
                        text="Peter",
                        start=0.34,
                        end=0.70,
                        chunks=[
                            ScriptLine(text="P", start=0.40, end=0.46),
                            ScriptLine(text="e", start=0.46, end=0.54),
                            ScriptLine(text="t", start=0.54, end=0.56),
                            ScriptLine(text="e", start=0.56, end=0.64),
                            ScriptLine(text="r", start=0.64, end=0.70),
                        ],
                    ),
                ],
            ),
            ScriptLine(
                text="This is Johnny.",
                start=0.70,
                end=1.67,
                chunks=[
                    ScriptLine(
                        text="This",
                        start=0.70,
                        end=1.14,
                        chunks=[
                            ScriptLine(text="T", start=1.00, end=1.02),
                            ScriptLine(text="h", start=1.02, end=1.06),
                            ScriptLine(text="i", start=1.06, end=1.08),
                            ScriptLine(text="s", start=1.08, end=1.14),
                        ],
                    ),
                    ScriptLine(
                        text="is",
                        start=1.14,
                        end=1.28,
                        chunks=[
                            ScriptLine(text="i", start=1.21, end=1.25),
                            ScriptLine(text="s", start=1.25, end=1.28),
                        ],
                    ),
                    ScriptLine(
                        text="Johnny",
                        start=1.28,
                        end=1.67,
                        chunks=[
                            ScriptLine(text="J", start=1.32, end=1.47),
                            ScriptLine(text="o", start=1.47, end=1.49),
                            ScriptLine(text="h", start=1.49, end=1.51),
                            ScriptLine(text="n", start=1.51, end=1.55),
                            ScriptLine(text="n", start=1.55, end=1.59),
                            ScriptLine(text="y", start=1.59, end=1.67),
                        ],
                    ),
                ],
            ),
            ScriptLine(
                text="Kenny.",
                start=1.67,
                end=2.11,
                chunks=[
                    ScriptLine(
                        text="Kenny",
                        start=1.67,
                        end=2.11,
                        chunks=[
                            ScriptLine(text="K", start=1.83, end=1.91),
                            ScriptLine(text="e", start=1.91, end=1.93),
                            ScriptLine(text="n", start=1.93, end=1.99),
                            ScriptLine(text="n", start=1.99, end=2.05),
                            ScriptLine(text="y", start=2.05, end=2.11),
                        ],
                    ),
                ],
            ),
            ScriptLine(
                text="And Joe.",
                start=2.11,
                end=2.89,
                chunks=[
                    ScriptLine(
                        text="And",
                        start=2.11,
                        end=2.53,
                        chunks=[
                            ScriptLine(text="A", start=2.45, end=2.47),
                            ScriptLine(text="n", start=2.47, end=2.51),
                            ScriptLine(text="d", start=2.51, end=2.53),
                        ],
                    ),
                    ScriptLine(
                        text="Joe",
                        start=2.53,
                        end=2.89,
                        chunks=[
                            ScriptLine(text="J", start=2.57, end=2.69),
                            ScriptLine(text="o", start=2.69, end=2.87),
                            ScriptLine(text="e", start=2.87, end=2.89),
                        ],
                    ),
                ],
            ),
            ScriptLine(
                text="We just wanted to take a minute to thank you.",
                start=2.89,
                end=4.88,
                chunks=[
                    ScriptLine(
                        text="We",
                        start=2.89,
                        end=3.51,
                        chunks=[
                            ScriptLine(text="W", start=3.41, end=3.43),
                            ScriptLine(text="e", start=3.43, end=3.51),
                        ],
                    ),
                    ScriptLine(
                        text="just",
                        start=3.51,
                        end=3.67,
                        chunks=[
                            ScriptLine(text="j", start=3.53, end=3.58),
                            ScriptLine(text="u", start=3.58, end=3.62),
                            ScriptLine(text="s", start=3.62, end=3.63),
                            ScriptLine(text="t", start=3.63, end=3.67),
                        ],
                    ),
                    ScriptLine(
                        text="wanted",
                        start=3.67,
                        end=3.92,
                        chunks=[
                            ScriptLine(text="w", start=3.69, end=3.73),
                            ScriptLine(text="a", start=3.73, end=3.75),
                            ScriptLine(text="n", start=3.75, end=3.79),
                            ScriptLine(text="t", start=3.79, end=3.86),
                            ScriptLine(text="e", start=3.86, end=3.88),
                            ScriptLine(text="d", start=3.88, end=3.92),
                        ],
                    ),
                    ScriptLine(
                        text="to",
                        start=3.92,
                        end=4.00,
                        chunks=[
                            ScriptLine(text="t", start=3.94, end=3.96),
                            ScriptLine(text="o", start=3.96, end=4.00),
                        ],
                    ),
                    ScriptLine(
                        text="take",
                        start=4.00,
                        end=4.20,
                        chunks=[
                            ScriptLine(text="t", start=4.04, end=4.10),
                            ScriptLine(text="a", start=4.10, end=4.14),
                            ScriptLine(text="k", start=4.14, end=4.16),
                            ScriptLine(text="e", start=4.16, end=4.20),
                        ],
                    ),
                    ScriptLine(
                        text="a",
                        start=4.20,
                        end=4.26,
                        chunks=[
                            ScriptLine(text="a", start=4.24, end=4.26),
                        ],
                    ),
                    ScriptLine(
                        text="minute",
                        start=4.26,
                        end=4.50,
                        chunks=[
                            ScriptLine(text="m", start=4.30, end=4.34),
                            ScriptLine(text="i", start=4.34, end=4.36),
                            ScriptLine(text="n", start=4.36, end=4.40),
                            ScriptLine(text="u", start=4.40, end=4.44),
                            ScriptLine(text="t", start=4.44, end=4.46),
                            ScriptLine(text="e", start=4.46, end=4.50),
                        ],
                    ),
                    ScriptLine(
                        text="to",
                        start=4.50,
                        end=4.60,
                        chunks=[
                            ScriptLine(text="t", start=4.52, end=4.56),
                            ScriptLine(text="o", start=4.56, end=4.60),
                        ],
                    ),
                    ScriptLine(
                        text="thank",
                        start=4.60,
                        end=4.80,
                        chunks=[
                            ScriptLine(text="t", start=4.66, end=4.68),
                            ScriptLine(text="h", start=4.68, end=4.72),
                            ScriptLine(text="a", start=4.72, end=4.74),
                            ScriptLine(text="n", start=4.74, end=4.78),
                            ScriptLine(text="k", start=4.78, end=4.80),
                        ],
                    ),
                    ScriptLine(
                        text="you",
                        start=4.80,
                        end=4.88,
                        chunks=[
                            ScriptLine(text="y", start=4.82, end=4.84),
                            ScriptLine(text="o", start=4.84, end=4.86),
                            ScriptLine(text="u", start=4.86, end=4.88),
                        ],
                    ),
                ],
            ),
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


def test_align_transcriptions_multilingual(
    resampled_mono_audio_sample: Audio, aligned_scriptline_fixture_resampled_mono_audio: ScriptLine
) -> None:
    """Test alignment of transcriptions."""
    model = HFModel(path_or_uri="openai/whisper-tiny")
    transcription_en = transcribe_audios(
        [resampled_mono_audio_sample], model=model, language=Language(language_code="en")
    )[0]
    transcription_fr = transcribe_audios(
        [resampled_mono_audio_sample], model=model, language=Language(language_code="fr")
    )[0]
    audios_and_transcriptions_and_language = [
        (resampled_mono_audio_sample, transcription_en, Language(language_code="en")),
        (resampled_mono_audio_sample, transcription_fr, Language(language_code="fr")),
    ]
    aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
    assert len(aligned_transcriptions) == 2
    assert len(aligned_transcriptions[0]) == 5
    assert len(aligned_transcriptions[1]) == 5
    aligned_transcription_en = aligned_transcriptions[0][0] or None
    if isinstance(aligned_transcription_en, ScriptLine):
        compare_alignments(
            aligned_scriptline_fixture_resampled_mono_audio, aligned_transcription_en, difference_tolerance=0.001
        )
    else:
        raise ValueError("aligned_transcription_en is not a ScriptLine")


def test_align_transcriptions_curiosity_audio_fixture(
    resampled_had_that_curiosity_audio_sample: Audio, script_line_fixture_curiosity: ScriptLine
) -> None:
    """Test alignment of transcriptions using the 'had that curiosity' audio sample and fixture."""
    audios_and_transcriptions_and_language = [
        (resampled_had_that_curiosity_audio_sample, script_line_fixture_curiosity, Language(language_code="en"))
    ]
    aligned_transcriptions = align_transcriptions(audios_and_transcriptions_and_language)
    assert len(aligned_transcriptions[0]) == 1
    if aligned_transcriptions[0][0] is not None and aligned_transcriptions[0][0].chunks:
        aligned_transcription = aligned_transcriptions[0][0].chunks[0]  # fixture corresponds subsegment
        if aligned_transcription.text:
            aligned_transcription.text = aligned_transcription.text.strip(".")
        compare_alignments(aligned_transcription, script_line_fixture_curiosity, difference_tolerance=0.1)


@pytest.fixture
def nested_scriptline() -> ScriptLine:
    """Creates a nested ScriptLine structure for testing."""
    return ScriptLine(
        text="root",
        chunks=[
            ScriptLine(
                text="level 1 - chunk 1",
                chunks=[
                    ScriptLine(text="level 2 - chunk 1"),
                    ScriptLine(text="level 2 - chunk 2"),
                ],
            ),
            ScriptLine(
                text="level 1 - chunk 2",
                chunks=[
                    ScriptLine(
                        text="level 2 - chunk 3",
                        chunks=[ScriptLine(text="level 3 - chunk 1")],
                    )
                ],
            ),
        ],
    )


def test_remove_chunks_by_level_level_0(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at level 0 (should return chunks directly)."""
    result = remove_chunks_by_level(nested_scriptline, level=0)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].text == "level 1 - chunk 1"
    assert result[1].text == "level 1 - chunk 2"


def test_remove_chunks_by_level_level_1(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at level 1 (should flatten level 2 chunks into root)."""
    result = remove_chunks_by_level(nested_scriptline, level=1)
    assert isinstance(result, ScriptLine)
    assert result.chunks is not None
    assert len(result.chunks) == 3
    assert result.chunks[0].text == "level 2 - chunk 1"
    assert result.chunks[1].text == "level 2 - chunk 2"
    assert result.chunks[2].text == "level 2 - chunk 3"


def test_remove_chunks_by_level_level_2(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at level 2 (should flatten level 3 chunks into level 1)."""
    result = remove_chunks_by_level(nested_scriptline, level=2)
    assert isinstance(result, ScriptLine)
    assert result.chunks is not None
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "level 1 - chunk 1"
    assert result.chunks[1].text == "level 1 - chunk 2"
    assert result.chunks[1].chunks is not None
    assert len(result.chunks[1].chunks) == 1
    assert result.chunks[1].chunks[0].text == "level 3 - chunk 1"


def test_remove_chunks_by_level_no_scriptline() -> None:
    """Test passing None as scriptline (should return None)."""
    assert remove_chunks_by_level(None, level=1) is None


def test_remove_chunks_by_level_keep_lower_false(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks with keep_lower=False (should return None at level 0)."""
    result = remove_chunks_by_level(nested_scriptline, level=0, keep_lower=False)
    assert result is None


def test_remove_chunks_by_level_keep_lower_false_at_level_1(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at level 1 with keep_lower=False (should remove all level 2 chunks)."""
    result = remove_chunks_by_level(nested_scriptline, level=1, keep_lower=False)
    assert isinstance(result, ScriptLine)
    assert result.chunks is not None
    assert len(result.chunks) == 0
    assert result.text == "root"


def test_remove_chunks_by_level_keep_lower_false_at_level_2(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at level 2 with keep_lower=False (should remove all level 2 chunks)."""
    result = remove_chunks_by_level(nested_scriptline, level=2, keep_lower=False)
    assert isinstance(result, ScriptLine)
    assert result.chunks is not None
    assert len(result.chunks) == 2
    assert result.chunks[0].text == "level 1 - chunk 1"
    assert result.chunks[1].text == "level 1 - chunk 2"
    assert all(chunk.chunks is not None and len(chunk.chunks) == 0 for chunk in result.chunks)


def test_remove_chunks_by_level_deep_removal(nested_scriptline: ScriptLine) -> None:
    """Test removing chunks at a level deeper than existing structure (should not modify)."""
    result = remove_chunks_by_level(nested_scriptline, level=5)
    assert result == nested_scriptline  # Should be unchanged


if __name__ == "__main__":
    pytest.main()
