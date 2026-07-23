"""GPU-gated smoke test for the diarization engine.

Skipped unless ``RUN_DIARIZATION_SMOKE=1`` because it downloads the pyannote model (needs an
``HF_TOKEN`` and, realistically, a GPU). It asserts the call contract, not a speaker count.
"""

from __future__ import annotations

import os

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType
from senselab_ls.common import engine

_RUN_SMOKE = os.getenv("RUN_DIARIZATION_SMOKE") == "1"


@pytest.mark.skipif(not _RUN_SMOKE, reason="set RUN_DIARIZATION_SMOKE=1 (needs GPU + HF token + model download)")
def test_diarize_returns_segment_list() -> None:
    """``engine.diarize`` returns a list of segments for a short synthetic clip."""
    waveform = torch.rand(1, 16000 * 4) * 0.1  # 4 s of low-level noise
    audio = Audio(waveform=waveform, sampling_rate=16000)
    device = DeviceType.CUDA if torch.cuda.is_available() else DeviceType.CPU
    segments = engine.diarize(audio, device=device)
    assert isinstance(segments, list)
