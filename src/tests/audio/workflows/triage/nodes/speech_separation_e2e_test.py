"""The multi-speaker instrument against the real MossFormer2 checkpoint, end to end.

Every other test of the instrument stubs ``separate_audios``. This one does not: it builds a real
two-voice mixture, hands it to the packaged configuration's own backend through the ClearerVoice
subprocess venv, and reads back what the branch wrote. It exists to prove that the chain
config -> ``separate_audios`` -> ``separated_*`` streams -> ``localise_speakers`` closes on real
weights, which a stub cannot establish. The separator is the only thing here that is real; the
diarizer, SQUIM and the PII scan are stubbed, because none of them is what this test is about.

It is environment-gated because it downloads a checkpoint, builds a subprocess venv on first use
and costs tens of seconds per call::

    SENSELAB_TRIAGE_SEPARATION_E2E=1 uv run pytest \
        src/tests/audio/workflows/triage/nodes/speech_separation_e2e_test.py -v -s
"""

import os
from pathlib import Path
from typing import Callable, Iterator

import pytest
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.nodes.speech import speech
from senselab.utils.prov_store import ProvStore

from . import speech_test
from .speech_test import _report_entity, _seed_diarization, _seed_speech_store, live_entities

_ENABLED = os.environ.get("SENSELAB_TRIAGE_SEPARATION_E2E")

_VOICE_A = Path("src/tests/data_for_testing/audio_48khz_mono_16bits.wav")
_VOICE_B = Path("src/tests/data_for_testing/had_that_curiosity.wav")

_RATE = 16000
_DURATION_S = 5.0
_INTRUDER = (1.8, 2.6)


def _mono_16k(path: Path) -> torch.Tensor:
    """One test recording as a mono 16 kHz waveform.

    Args:
        path: The WAV.

    Returns:
        A ``(1, n)`` tensor.
    """
    [audio] = resample_audios(downmix_audios_to_mono([Audio(filepath=str(path))]), resample_rate=_RATE)
    return audio.waveform


def _mixture() -> torch.Tensor:
    """Two real voices, one running throughout and one laid over :data:`_INTRUDER`.

    Returns:
        A ``(1, n)`` mixture at :data:`_RATE`.
    """
    total = int(_DURATION_S * _RATE)
    host, guest = _mono_16k(_VOICE_A), _mono_16k(_VOICE_B)
    mix = torch.zeros(1, total)
    for offset in range(0, total, host.shape[-1]):
        width = min(host.shape[-1], total - offset)
        mix[:, offset : offset + width] = host[:, :width]
    start = int(_INTRUDER[0] * _RATE)
    width = min(int(_INTRUDER[1] * _RATE) - start, guest.shape[-1])
    mix[:, start : start + width] += guest[:, :width]
    return mix / mix.abs().max().clamp(min=1e-9) * 0.7


@pytest.fixture
def _everything_but_the_separator(
    monkeypatch: pytest.MonkeyPatch, seed_preprocess_store: Callable[..., None]
) -> Iterator[None]:
    """Bind the shared seeder and stub every model the separator is not.

    Args:
        monkeypatch: The patcher.
        seed_preprocess_store: The shared seeder ``_seed_speech_store`` delegates to.

    Yields:
        Nothing.
    """
    monkeypatch.setattr(speech_test, "_SEEDER", seed_preprocess_store)
    monkeypatch.setattr(
        speech_module,
        "extract_objective_quality_features_from_audios",
        lambda audios, device=None: [{"stoi": 0.9, "pesq": 3.0, "si_sdr": 18.0} for _ in audios],
    )
    monkeypatch.setattr(
        speech_module,
        "scan_for_pii",
        lambda inputs, **kw: [
            speech_test.PiiScan(spans=[], detectors_used=speech_test.default_detectors(), failures={})
            for _ in ([inputs] if isinstance(inputs, str) else list(inputs))
        ],
    )
    yield


@pytest.mark.skipif(not _ENABLED, reason="SENSELAB_TRIAGE_SEPARATION_E2E not set")
@pytest.mark.usefixtures("_everything_but_the_separator")
def test_the_real_separator_produces_streams_the_localisation_reads(store: ProvStore, tmp_path: Path) -> None:
    """The packaged config, real weights: two streams on disk and a localisation that used them."""
    _seed_speech_store(
        store,
        tmp_path,
        words=["one", "two", "three"],
        word_extents=[(0.4, 1.2), (1.6, 2.4), (3.0, 4.2)],
        duration_s=_DURATION_S,
        diarization=False,
    )
    _seed_diarization(
        store,
        tmp_path,
        [
            (0.2, _INTRUDER[0], "SPEAKER_00"),
            (_INTRUDER[0], _INTRUDER[1], "SPEAKER_01"),
            (_INTRUDER[1], 4.6, "SPEAKER_00"),
        ],
    )
    plain = [e for e in live_entities(store, "stream") if e.attributes.get("name") == "plain"][-1]
    Audio(waveform=_mixture(), sampling_rate=_RATE).save_to_file(str(tmp_path / str(plain.attributes["path"])))

    hint = AudioHints(metadata={"task_token": "picture-description"})
    speech(store, "plain", load_triage_config(), hint, run_dir=tmp_path, enrollment=None)

    detail = _report_entity(store, "SPEECH").attributes
    assert detail["separation"]["backend"] == "MossFormer2_SS_16K", detail["separation"]
    separated = sorted(
        (e for e in live_entities(store, "stream") if str(e.attributes.get("name", "")).startswith("separated")),
        key=lambda e: int(e.attributes["source_index"]),
    )
    assert [e.attributes["name"] for e in separated] == ["separated_0", "separated_1"]
    for entity in separated:
        assert entity.attributes["separation_model"] == "alibabasglab/MossFormer2_SS_16K"
        assert len(str(entity.attributes["separation_commit"])) == 40, "the load pinned a commit, not a ref"
        assert (tmp_path / str(entity.attributes["path"])).stat().st_size > 0

    [localise] = [a for a in store.activities() if a.step == speech_module.LOCALISE_STEP]
    assert {e.id for e in separated} <= set(store.uses_of(localise.id)), "the localisation read both streams"

    localisation = detail["source_localisation"]
    assert len(localisation["sources"]) == 2
    assert sum(record["active_s"] for record in localisation["sources"]) > 0.0
    print("\nsource_localisation:", localisation)
    print("notes:", [note for note in detail["notes"] if "loudest" in note])
