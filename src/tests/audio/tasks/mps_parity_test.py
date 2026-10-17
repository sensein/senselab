"""MPS gives the CPU result for pyannote diarization, SpeechBrain embeddings and in-process SER.

Skipped where MPS is unavailable. Each test loads one model, runs it on CPU and on MPS over the
same recording, and compares.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.utils.data_structures import DeviceType, HFModel, SpeechBrainModel
from tests.audio.conftest import MONO_AUDIO_PATH

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")

CONVERSATION = Path(MONO_AUDIO_PATH).parent / "english_conversation_higgs_audio_v2.wav"


def _conversation() -> Audio:
    return resample_audios(downmix_audios_to_mono([Audio(filepath=str(CONVERSATION))]), 16000)[0]


def _clips(audio: Audio, seconds: float) -> list[Audio]:
    n = int(seconds * audio.sampling_rate)
    return [
        Audio(waveform=audio.waveform[:, i : i + n], sampling_rate=audio.sampling_rate)
        for i in range(0, audio.waveform.shape[-1] - n + 1, n)
    ]


def test_pyannote_segments_identical_on_mps() -> None:
    """Exclusive and overlapping diarization on MPS equal the CPU segments."""
    from senselab.audio.tasks.speaker_diarization import diarize_audios

    audio = _conversation()
    for exclusive in (True, False):
        cpu = diarize_audios([audio], device=DeviceType.CPU, exclusive=exclusive)[0]
        mps = diarize_audios([audio], device=DeviceType.MPS, exclusive=exclusive)[0]
        assert [(s.speaker, s.start, s.end) for s in mps] == [(s.speaker, s.start, s.end) for s in cpu]


def test_ecapa_embeddings_match_on_mps() -> None:
    """ECAPA embeddings on MPS have cosine > 0.999 with the CPU embeddings."""
    from senselab.audio.tasks.speaker_embeddings import extract_speaker_embeddings_from_audios

    clips = _clips(_conversation(), 3.0)
    model: SpeechBrainModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")
    cpu = extract_speaker_embeddings_from_audios(clips, model=model, device=DeviceType.CPU)
    mps = extract_speaker_embeddings_from_audios(clips, model=model, device=DeviceType.MPS)
    cosine = torch.stack(
        [torch.nn.functional.cosine_similarity(a.flatten().cpu(), b.flatten().cpu(), dim=0) for a, b in zip(cpu, mps)]
    )
    assert float(cosine.min()) > 0.999


def test_continuous_ser_matches_on_mps() -> None:
    """Arousal/valence/dominance on MPS are within 1e-4 of CPU."""
    from senselab.audio.tasks.classification.speech_emotion_recognition import classify_emotions_from_speech

    clips = _clips(_conversation(), 5.0)
    model: HFModel = HFModel(path_or_uri="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim")
    cpu = classify_emotions_from_speech(clips, model, device=DeviceType.CPU)
    mps = classify_emotions_from_speech(clips, model, device=DeviceType.MPS)
    for a, b in zip(cpu, mps):
        da, db = dict(zip(a.labels, a.scores)), dict(zip(b.labels, b.scores))
        assert set(da) == {"arousal", "valence", "dominance"}
        assert max(abs(da[k] - db[k]) for k in da) < 1e-4
