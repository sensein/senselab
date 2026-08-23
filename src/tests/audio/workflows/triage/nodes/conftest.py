"""Shared fixtures for the triage node tests. Nothing here loads a model."""

import json
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


@pytest.fixture
def seed_store(tmp_path: Path) -> Callable[..., dict]:
    """A builder writing PREPROCESS-shaped entities into a store — the Task 2 schema, seeded.

    Writes a real plain-stream WAV so nodes that slice audio can, and entities for spans, words,
    YAMNet windows, silence and no_contrast as requested.
    """

    def _seed(
        store: ProvStore,
        *,
        spans: tuple = (),
        words: tuple = (),
        yamnet_windows: list | None = None,
        silence_windows: list | None = None,
        no_contrast_k: float | None = None,
        asr_available: bool = True,
        k_db: float = 18.0,
        duration_s: float = 4.0,
    ) -> dict:
        (tmp_path / "streams").mkdir(exist_ok=True)
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        sf.write(str(tmp_path / "streams" / "plain.wav"), burst_samples(duration_s=duration_s), 16000)
        activity = store.activity(node="PREPROCESS", step="seed", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(activity, agent)
        ids: dict = {"spans": [], "words": []}

        plain_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "plain",
                "path": "streams/plain.wav",
                "sampling_rate": 16000,
                "channels": 1,
                "peak_scale": 1.0,
            },
        )
        store.was_generated_by(plain_id, activity)
        ids["plain"] = plain_id

        for start, end, contrast in spans:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": contrast, "k_db": k_db, "signal": "preemphasised"},
            )
            store.was_generated_by(span_id, activity)
            ids["spans"].append(span_id)

        if no_contrast_k is not None:
            nc_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "spans_no_contrast",
                    "signal": "preemphasised",
                    "k_db": no_contrast_k,
                    "reason": "seeded",
                },
            )
            store.was_generated_by(nc_id, activity)
            ids["no_contrast"] = nc_id

        if yamnet_windows is not None:
            path = tmp_path / "derivatives" / "yamnet_windows.json"
            path.write_text(json.dumps(yamnet_windows))
            yw_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "yamnet_windows",
                    "signal": "plain",
                    "path": "derivatives/yamnet_windows.json",
                    "n_windows": len(yamnet_windows),
                },
            )
            store.was_generated_by(yw_id, activity)
            ids["yamnet_windows"] = yw_id

        if silence_windows is not None:
            s_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
            )
            store.was_generated_by(s_id, activity)
            ids["silence"] = s_id

        if asr_available:
            asr_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "asr_crisperwhisper",
                    "signal": "plain",
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "transcript": " ".join(str(w["text"]) for w in words),
                    "word_ids": [],
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(asr_id, activity)
            ids["asr"] = asr_id

        for word in words:
            word_id = store.entity(
                prov_type="word",
                extent=(float(word["start"]), float(word["end"])),
                attributes={
                    "text": str(word["text"]),
                    "score": 0.9,
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(word_id, activity)
            ids["words"].append(word_id)
        return ids

    return _seed
