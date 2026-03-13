"""Tests for the conversation analysis workflow."""

from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.utils.data_structures import ScriptLine


def _word_chunk(text: str, start: float, end: float) -> ScriptLine:
    """Create a word-level script chunk for tests."""
    return ScriptLine(text=text, start=start, end=end)


def test_estimate_transcript_accuracy_uses_consensus() -> None:
    """Multi-model agreement should yield a high consensus estimate."""
    from senselab.audio.workflows.conversation_analysis import _estimate_transcript_accuracy

    estimate = _estimate_transcript_accuracy(
        {
            "asr-a": "please help me with this",
            "asr-b": "please help me with this",
            "asr-c": "please help with this",
        }
    )

    assert estimate["method"] == "multi_model_consensus"
    assert 0.8 <= estimate["score"] <= 1.0


def test_extract_dialogue_acts_and_engagement_markers() -> None:
    """Question, request, politeness, and backchannel cues should be surfaced."""
    from senselab.audio.workflows.conversation_analysis import (
        _extract_dialogue_acts,
        _extract_engagement_markers,
        _extract_linguistic_features,
    )

    question_text = "Please can you help me with this?"
    backchannel_text = "yeah right"

    linguistic = _extract_linguistic_features(question_text)
    dialogue_acts = _extract_dialogue_acts(question_text)
    engagement = _extract_engagement_markers(backchannel_text)

    assert "please" in linguistic["politeness_cues"]
    assert "question" in dialogue_acts
    assert "request" in dialogue_acts
    assert engagement["is_backchannel"] is True
    assert "yeah" in engagement["backchannel_cues"]


def test_summarize_turn_taking_counts_interruptions() -> None:
    """Overlapping turns should contribute to interruption statistics."""
    from senselab.audio.workflows.conversation_analysis import _summarize_turn_taking

    turns = [
        {"speaker_id": "spk0", "start": 0.0, "end": 1.0, "duration_seconds": 1.0},
        {"speaker_id": "spk1", "start": 0.9, "end": 1.7, "duration_seconds": 0.8},
        {"speaker_id": "spk0", "start": 2.0, "end": 2.5, "duration_seconds": 0.5},
    ]

    summary = _summarize_turn_taking(turns)

    assert summary["turn_count"] == 3
    assert summary["speaker_switch_count"] == 2
    assert summary["interruption_count"] == 1
    assert summary["overlap_count"] == 1
    assert summary["mean_gap_duration_seconds"] == pytest.approx(0.15, abs=1e-3)


def test_summarize_ppg_segments_builds_sparse_segments() -> None:
    """PPG summaries should collapse frame-wise peaks into sparse onset/offset segments."""
    from senselab.audio.workflows.conversation_analysis import _summarize_ppg_segments

    audio = Audio(waveform=torch.ones(1, 16000), sampling_rate=16000)
    posteriorgram = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )

    summary = _summarize_ppg_segments(audio=audio, posteriorgram=posteriorgram)

    assert summary["frame_count"] == 4
    assert len(summary["segments"]) == 2
    assert summary["segments"][0]["label_index"] == 0
    assert summary["segments"][1]["label_index"] == 1
    assert summary["segments"][0]["start_seconds"] == pytest.approx(0.0)
    assert summary["segments"][1]["end_seconds"] == pytest.approx(1.0, abs=1e-6)


def test_analyze_conversation_recordings_returns_structured_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The workflow should assemble recording, turn, and check outputs from reused tasks."""
    from senselab.audio.workflows import conversation_analysis as workflow_module

    recording_path = tmp_path / "zoom_session.wav"
    recording_path.write_bytes(b"audio")

    recording_audio = Audio(waveform=torch.ones(1, 32000), sampling_rate=16000)
    segment_a = Audio(waveform=torch.ones(1, 12800), sampling_rate=16000)
    segment_b = Audio(waveform=torch.ones(1, 11200), sampling_rate=16000)

    asr_models = [SimpleNamespace(path_or_uri="asr-a"), SimpleNamespace(path_or_uri="asr-b")]
    speaker_embedding_model = [SimpleNamespace(path_or_uri="speaker-emb-a")]
    ssl_embedding_model = [SimpleNamespace(path_or_uri="ssl-emb-a")]
    emotion_model = SimpleNamespace(path_or_uri="emotion-a")

    diarization = [
        [
            ScriptLine(speaker="spk0", start=0.0, end=0.8),
            ScriptLine(speaker="spk1", start=0.75, end=1.45),
        ]
    ]

    def fake_transcribe_audios(
        audios: List[Audio],
        model: SimpleNamespace,
        **kwargs: object,
    ) -> List[ScriptLine]:
        if model.path_or_uri == "asr-a":
            return [
                ScriptLine(
                    text="Please can you help me?",
                    start=0.0,
                    end=0.8,
                    chunks=[
                        _word_chunk("Please", 0.0, 0.2),
                        _word_chunk("can", 0.2, 0.3),
                        _word_chunk("you", 0.3, 0.4),
                        _word_chunk("help", 0.4, 0.6),
                        _word_chunk("me", 0.6, 0.75),
                    ],
                ),
                ScriptLine(
                    text="Yeah thanks",
                    start=0.75,
                    end=1.45,
                    chunks=[_word_chunk("Yeah", 0.8, 1.0), _word_chunk("thanks", 1.0, 1.2)],
                ),
            ]
        return [
            ScriptLine(text="Please can you help me", start=0.0, end=0.8),
            ScriptLine(text="Yeah, thanks", start=0.75, end=1.45),
        ]

    monkeypatch.setattr(
        workflow_module,
        "_prepare_recording_audio_files",
        lambda recording_file_paths: recording_file_paths,
    )
    monkeypatch.setattr(workflow_module, "read_audios", lambda paths: [recording_audio])
    monkeypatch.setattr(workflow_module, "downmix_audios_to_mono", lambda audios: audios)
    monkeypatch.setattr(workflow_module, "resample_audios", lambda audios, resample_rate: audios)
    monkeypatch.setattr(workflow_module, "diarize_audios", lambda audios, model=None, device=None: diarization)
    monkeypatch.setattr(workflow_module, "extract_segments", lambda segments_info: [[segment_a, segment_b]])
    monkeypatch.setattr(workflow_module, "transcribe_audios", fake_transcribe_audios)
    monkeypatch.setattr(
        workflow_module,
        "extract_features_from_audios",
        lambda audios, **kwargs: [
            {
                "praat_parselmouth": {
                    "speaking_rate": 3.4,
                    "articulation_rate": 4.0,
                    "pause_rate": 0.2,
                    "mean_pause_duration": 0.15,
                    "mean_f0_hertz": 180.0,
                    "mean_intensity_db": 63.0,
                },
                "opensmile": {"loudness_sma3_amean": 0.51},
            },
            {
                "praat_parselmouth": {
                    "speaking_rate": 2.1,
                    "articulation_rate": 2.6,
                    "pause_rate": 0.1,
                    "mean_pause_duration": 0.05,
                    "mean_f0_hertz": 155.0,
                    "mean_intensity_db": 58.0,
                },
                "opensmile": {"loudness_sma3_amean": 0.32},
            },
        ],
    )
    monkeypatch.setattr(
        workflow_module,
        "classify_emotions_from_speech",
        lambda audios, model, device=None: [
            AudioClassificationResult(labels=["joy", "neutral"], scores=[0.8, 0.2]),
            AudioClassificationResult(labels=["neutral", "joy"], scores=[0.7, 0.3]),
        ],
    )
    monkeypatch.setattr(
        workflow_module,
        "extract_speaker_embeddings_from_audios",
        lambda audios, model, device=None: [torch.tensor([0.1, 0.2]), torch.tensor([0.3, 0.4])],
    )
    monkeypatch.setattr(
        workflow_module,
        "extract_ssl_embeddings_from_audios",
        lambda audios, model, device=None: [torch.tensor([[0.1, 0.2]]), torch.tensor([[0.3, 0.4]])],
    )
    monkeypatch.setattr(
        workflow_module,
        "extract_ppgs_from_audios",
        lambda audios, device=None: [
            torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8]]),
            torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]),
        ],
    )
    monkeypatch.setattr(
        workflow_module,
        "_build_recording_checks",
        lambda audio: {
            "input": {"passed": True},
            "preprocessing": {"passed": True},
            "environment": {"passed": True, "snr_db": 22.5},
        },
    )

    result = workflow_module.analyze_conversation_recordings(
        recording_file_paths=[recording_path],
        transcription_models=asr_models,
        emotion_model=emotion_model,
        speaker_embeddings_models=speaker_embedding_model,
        ssl_embeddings_models=ssl_embedding_model,
        include_ppgs=True,
    )

    assert len(result) == 1
    recording = result[0]
    assert recording["source_file"] == str(recording_path.resolve())
    assert recording["turn_taking"]["interruption_count"] == 1
    assert recording["speaker_summary"]["speaker_count"] == 2
    assert recording["checks"]["environment"]["snr_db"] == pytest.approx(22.5)
    assert len(recording["turns"]) == 2

    first_turn = recording["turns"][0]
    assert first_turn["speaker_id"] == "spk0"
    assert "question" in first_turn["dialogue_acts"]
    assert "request" in first_turn["dialogue_acts"]
    assert first_turn["transcript_accuracy_estimate"] > 0.9
    assert first_turn["emotion"]["top_label"] == "joy"
    assert first_turn["engagement_markers"]["politeness_cues"] == ["please"]
    assert first_turn["ppg_summary"]["segment_count"] >= 1


def test_analyze_conversation_recordings_rejects_missing_files(tmp_path: Path) -> None:
    """Missing recordings should fail fast before any model work starts."""
    from senselab.audio.workflows.conversation_analysis import analyze_conversation_recordings

    missing_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        analyze_conversation_recordings([missing_path])
