"""Speech emotion recognition routes from the local cache with the Hub offline; emotion2vec+ routing."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.speech_emotion_recognition import api as ser_api
from senselab.audio.tasks.classification.speech_emotion_recognition import emotion2vec
from senselab.utils.data_structures import HFModel
from tests.audio.conftest import MONO_AUDIO_PATH

SHA = "6eba34a2485ea31cb03600241787c3a5edab8626"
AUDEERING = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"


@pytest.fixture(autouse=True)
def _reset_ser_module_caches() -> None:
    ser_api._config_memo.clear()
    ser_api._config_memo_locks.clear()
    ser_api._wav2vec2_emotion_models.clear()


def _model(repo: str, sha: str = SHA) -> HFModel:
    """An HFModel without construction-time Hub validation."""
    return HFModel.model_construct(path_or_uri=repo, revision="main", commit_sha=sha, info=None)


def _audio() -> Audio:
    return Audio(filepath=MONO_AUDIO_PATH)


def _offline(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Put huggingface_hub offline and make any Hub metadata call fail loudly."""
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    info = MagicMock(side_effect=AssertionError("Hub model_info called while offline"))
    monkeypatch.setattr(HFModel, "get_model_info", info)
    monkeypatch.setattr("huggingface_hub.HfApi.list_repo_files", MagicMock(side_effect=AssertionError("Hub listing")))
    return info


def test_continuous_model_routes_offline_without_hub_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """An audeering-style config routes to the custom-head path with no Hub call, pinned to the SHA."""
    info = _offline(monkeypatch)
    config = MagicMock(
        model_type="wav2vec2",
        architectures=["Wav2Vec2ForSpeechClassification"],
        id2label={0: "arousal", 1: "dominance", 2: "valence"},
        auto_map=None,
        problem_type=None,
    )
    from_pretrained = MagicMock(return_value=config)
    monkeypatch.setattr("transformers.AutoConfig.from_pretrained", from_pretrained)
    seen: dict = {}

    def fake_custom_head(audios: List[Audio], model: HFModel, device: Any = None, **kwargs: Any) -> list:  # noqa: ANN401
        seen.update(kwargs)
        return [AudioClassificationResult(labels=["arousal", "dominance", "valence"], scores=[0.5, 0.4, 0.3])]

    monkeypatch.setattr(ser_api, "_classify_wav2vec2_speech_cls_ser", fake_custom_head)
    result = ser_api.classify_emotions_from_speech([_audio()], _model(AUDEERING))
    assert result[0].labels == ["arousal", "dominance", "valence"]
    assert seen["apply_softmax"] is False and seen["head"].final_layer == "out_proj"
    info.assert_not_called()
    assert from_pretrained.call_args.kwargs["revision"] == SHA


def test_unrecognised_labels_read_tags_from_the_cached_card(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With the Hub offline, the emotion tag comes from the cached README front matter."""
    info = _offline(monkeypatch)
    card = tmp_path / "README.md"
    card.write_text("---\ntags:\n- speech-emotion-recognition\n---\n# card\n")
    monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda repo, filename, revision: str(card))
    assert ser_api._model_card_tags(_model("org/odd-labels")) == ["speech-emotion-recognition"]

    card.write_text("---\ntags:\n- audio\n---\n")
    assert ser_api._model_card_tags(_model("org/odd-labels")) == ["audio"]
    info.assert_not_called()


def test_not_an_emotion_model_still_raises_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """No emotion labels and no emotion tag: a ValueError, not a Hub error."""
    _offline(monkeypatch)
    monkeypatch.setattr(ser_api, "_load_config_cached", lambda model: (MagicMock(id2label={0: "dog", 1: "cat"}), None))
    monkeypatch.setattr(ser_api, "_model_card_tags", lambda model: [])
    with pytest.raises(ValueError, match="not suitable for speech emotion recognition"):
        ser_api.classify_emotions_from_speech([_audio()], _model("org/pets"))


def test_emotion2vec_routes_before_any_config_read(monkeypatch: pytest.MonkeyPatch) -> None:
    """emotion2vec repositories have no config.json; they route to the FunASR backend first."""
    _offline(monkeypatch)
    monkeypatch.setattr(ser_api, "_load_config_cached", MagicMock(side_effect=AssertionError("config read")))
    called = MagicMock(return_value=[AudioClassificationResult(labels=["happy"], scores=[1.0])])
    monkeypatch.setattr(ser_api, "classify_emotions_with_emotion2vec", called)
    model = _model("emotion2vec/emotion2vec_plus_large", "6c303ba987b86b93193de93e34bb2b077a6bedc4")
    assert ser_api.classify_emotions_from_speech([_audio()], model)[0].labels == ["happy"]
    called.assert_called_once()


def test_emotion2vec_without_a_head_is_refused() -> None:
    """``emotion2vec_base`` is a representation model; asking it for emotions is a ValueError."""
    with pytest.raises(ValueError, match="no emotion classification head"):
        emotion2vec.emotion2vec_checkpoint(_model("emotion2vec/emotion2vec_base", "0" * 40))


def test_emotion2vec_worker_gets_the_staged_snapshot(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The worker payload names the local snapshot directory, mono 16 kHz audio, and no repo id or ref."""
    sha = "6c303ba987b86b93193de93e34bb2b077a6bedc4"
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    monkeypatch.setattr(emotion2vec, "resolve_model", lambda repo, rev: (sha, snapshot))
    monkeypatch.setattr(emotion2vec, "ensure_emotion2vec_venv", lambda: tmp_path)
    monkeypatch.setattr(emotion2vec, "venv_python", lambda venv: sys.executable)
    captured: dict = {}

    def fake_run(cmd: list, input: str, **kwargs: Any) -> subprocess.CompletedProcess:  # noqa: ANN401, A002
        payload = json.loads(input)
        captured.update(payload)
        from senselab.utils.portable_audio_io import audio_info

        captured["rates"] = [int(audio_info(p).sampling_rate) for p in payload["audio_paths"]]
        out = {"results": [{"labels": list(emotion2vec.EMOTION2VEC_LABELS), "scores": [1 / 9] * 9}], "timing": {}}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(emotion2vec.subprocess, "run", fake_run)
    model = _model("emotion2vec/emotion2vec_plus_large", sha)
    result = emotion2vec.classify_emotions_with_emotion2vec([_audio()], model, threads=2)
    assert captured["model_dir"] == str(snapshot)
    assert captured["rates"] == [16000]
    assert captured["threads"] == 2
    assert not any("revision" in key or key == "model_id" for key in captured)
    assert sorted(result[0].labels) == sorted(emotion2vec.EMOTION2VEC_LABELS)


def _cached(repo: str) -> bool:
    from huggingface_hub import constants

    return (Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots").is_dir()


@pytest.mark.skipif(not _cached(AUDEERING), reason="audeering model not in the local HF cache")
def test_continuous_ser_runs_with_hf_hub_offline() -> None:
    """End to end in a fresh interpreter with HF_HUB_OFFLINE=1: three dimensions, no Hub call."""
    code = (
        "from senselab.audio.data_structures import Audio\n"
        "from senselab.audio.tasks.preprocessing import resample_audios\n"
        "from senselab.audio.tasks.classification.speech_emotion_recognition import classify_emotions_from_speech\n"
        "from senselab.utils.data_structures import DeviceType, HFModel\n"
        f"a = resample_audios([Audio(filepath={MONO_AUDIO_PATH!r})], 16000)\n"
        f"r = classify_emotions_from_speech(a, HFModel(path_or_uri={AUDEERING!r}), device=DeviceType.CPU)\n"
        "print('LABELS', sorted(r[0].labels))\n"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "OMP_NUM_THREADS": "3"}
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "LABELS ['arousal', 'dominance', 'valence']" in result.stdout


E2V_BASE = "emotion2vec/emotion2vec_plus_base"


def _emotion2vec_ready() -> bool:
    from senselab.utils.subprocess_venv import provisioned_venv_dirs

    return _cached(E2V_BASE) and bool(provisioned_venv_dirs("emotion2vec"))


@pytest.mark.skipif(not _emotion2vec_ready(), reason="emotion2vec venv or emotion2vec_plus_base not provisioned")
def test_emotion2vec_runs_offline_behind_the_network_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The FunASR worker classifies with SENSELAB_BLOCK_NETWORK=1: nine labels, posteriors summing to one."""
    from senselab.utils.model_revision import resolve_revision

    monkeypatch.setenv("SENSELAB_BLOCK_NETWORK", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    model = _model(E2V_BASE, resolve_revision(E2V_BASE, "main"))
    [result] = emotion2vec.classify_emotions_with_emotion2vec([_audio()], model)
    assert sorted(result.labels) == sorted(emotion2vec.EMOTION2VEC_LABELS)
    assert sum(result.scores) == pytest.approx(1.0, abs=1e-3)
