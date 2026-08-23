"""Shared fixtures for the triage node tests. Nothing here loads a model."""

from pathlib import Path
from typing import Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.utils.prov_store import ProvStore


@pytest.fixture
def config() -> TriageConfig:
    """The packaged configuration, unmodified."""
    return load_triage_config()


@pytest.fixture
def store() -> ProvStore:
    """An empty store for one test run."""
    return ProvStore(run_id="test-run")


@pytest.fixture
def wav_writer(tmp_path: Path) -> Callable[..., Path]:
    """A writer for mono or stereo float32 WAV fixtures under this test's tmp dir."""

    def _write(name: str, samples: np.ndarray, sampling_rate: int = 16000) -> Path:
        path = tmp_path / name
        sf.write(str(path), samples.astype(np.float32), sampling_rate)
        return path

    return _write


def burst_samples(duration_s: float = 3.0, sampling_rate: int = 16000) -> np.ndarray:
    """A quiet noise bed with one loud 150 ms tone burst at 1.5 s.

    The burst stands far more than 18 dB over the bed, so `propose_spans` at the airway `K`
    proposes exactly one span over it.
    """
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(duration_s * sampling_rate)) * 1e-4).astype(np.float32)
    i0 = int(1.5 * sampling_rate)
    i1 = i0 + int(0.15 * sampling_rate)
    t = np.arange(i1 - i0) / sampling_rate
    x[i0:i1] += (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return x
