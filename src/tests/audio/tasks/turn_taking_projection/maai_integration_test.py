"""The real MaAI checkpoints over a constructed two-speaker dialogue.

Skipped unless the ``maai`` venv is already provisioned and the pinned checkpoints are in the HF
cache: these tests never build the venv (~700 MB) or download weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.audio.tasks.turn_taking_projection import maai as backend
from senselab.audio.tasks.turn_taking_projection import project_voice_activity, separate_dyad, summarize_projection
from senselab.utils.data_structures import ScriptLine
from senselab.utils.subprocess_venv import provisioned_venv_dirs

DATA = Path(__file__).resolve().parents[3] / "data_for_testing"
SR = 16000


def _ready() -> bool:
    from huggingface_hub import constants

    checkpoints = [backend.VAP_CHECKPOINTS[backend.DEFAULT_VAP_CHECKPOINT], backend.BC_CHECKPOINTS["vap_bc_en"]]
    cached = all(
        (
            Path(constants.HF_HUB_CACHE)
            / f"models--{c.repo_id.replace('/', '--')}"
            / "snapshots"
            / c.revision
            / c.filename
        ).is_file()
        for c in checkpoints
    )
    return cached and any((d / ".senselab-installed").is_file() for d in provisioned_venv_dirs("maai"))


pytestmark = pytest.mark.skipif(
    not _ready(), reason="needs the provisioned 'maai' venv and the pinned MaAI checkpoints in the HF cache"
)


def _clip(name: str) -> np.ndarray:
    audio = resample_audios([Audio(filepath=str(DATA / name))], SR)[0]
    return audio.waveform.mean(dim=0).numpy()


@pytest.fixture(scope="module")
def dialogue() -> tuple:
    """Speaker a at 0.5 s, speaker b answering 0.4 s after a ends; 12 s in all."""
    a_clip, b_clip = _clip("had_that_curiosity.wav"), _clip("audio_48khz_mono_16bits.wav")
    n = 12 * SR
    a, b = np.zeros(n, np.float32), np.zeros(n, np.float32)
    a_start = int(0.5 * SR)
    a[a_start : a_start + a_clip.size] = a_clip
    b_start = a_start + a_clip.size + int(0.4 * SR)
    b[b_start : b_start + b_clip.size] = b_clip
    turns = [
        ScriptLine(speaker="A", start=a_start / SR, end=(a_start + a_clip.size) / SR),
        ScriptLine(speaker="B", start=b_start / SR, end=(b_start + b_clip.size) / SR),
    ]
    return a, b, turns


def test_vap_follows_who_is_speaking(dialogue: tuple) -> None:
    """Per-channel VAD tracks each speaker, and p_now favours the speaker holding the floor."""
    a, b, turns = dialogue
    pair = (
        Audio(waveform=torch.from_numpy(a)[None], sampling_rate=SR),
        Audio(waveform=torch.from_numpy(b)[None], sampling_rate=SR),
    )
    vap = project_voice_activity(pair, speakers=("A", "B"))
    f = vap.frames
    assert len(f) == 120 and vap.has_backchannel
    assert f[["p_now_a", "p_now_b", "p_bc_a", "p_bc_b"]].to_numpy().min() >= 0.0
    assert np.allclose(f.p_now_a + f.p_now_b, 1.0, atol=1e-3)
    in_a = (f.time > turns[0].start + 0.5) & (f.time < turns[0].end - 0.2)
    in_b = (f.time > turns[1].start + 0.5) & (f.time < turns[1].end - 0.2)
    assert f.vad_a[in_a].mean() > 0.5 > f.vad_b[in_a].mean()
    assert f.vad_b[in_b].mean() > 0.5 > f.vad_a[in_b].mean()
    assert f.p_now_a[in_a].mean() > 0.5 and f.p_now_b[in_b].mean() > 0.5
    assert vap.models["vap"]["revision"] == backend.VAP_CHECKPOINTS[backend.DEFAULT_VAP_CHECKPOINT].revision
    summary = summarize_projection(vap, turns, turns)
    assert summary["overall"]["n_shifts"] == 1


def test_masked_mono_gives_the_same_speaker_tracks(dialogue: tuple) -> None:
    """The mix, split by masking with the true segments, reproduces the clean channels' VAD."""
    a, b, turns = dialogue
    mix = Audio(waveform=torch.from_numpy(a + b)[None], sampling_rate=SR)
    channel_a, channel_b = separate_dyad(mix, turns)
    vap = project_voice_activity((channel_a, channel_b), backchannel_model=None)
    assert vap.speakers == ("A", "B") and not vap.has_backchannel
    assert vap.input["method"] == "masking"
    f = vap.frames
    in_b = (f.time > turns[1].start + 0.5) & (f.time < turns[1].end - 0.2)
    assert f.vad_b[in_b].mean() > 0.5 > f.vad_a[in_b].mean()
