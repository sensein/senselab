"""Behavior test: the Qwen subprocess ASR worker is launched with an offline env.

Qwen (``Qwen/Qwen3-ASR-1.7B`` + its forced-aligner companion) is the backend that
429'd under parallel batches. Its worker must be handed an env built by
``hf_subprocess_env`` so the child loads both models from cache with no Hub HEAD.
"""

from pathlib import Path

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, HFModel


class _StopBeforeSubprocess(Exception):
    """Raised by the hf_subprocess_env spy to halt before the real subprocess run."""


def test_qwen_worker_env_built_via_hf_subprocess_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """transcribe_with_qwen stages the ASR model + aligner offline via hf_subprocess_env."""
    import senselab.audio.tasks.speech_to_text.qwen as qwen_mod

    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda *a, **k: True)
    # Every ref this backend resolves (the ASR model via HFModel construction, and the
    # forced-aligner companion via a bare resolve_revision call) fakes to the same SHA, so
    # this test observes routing rather than depending on network/local-cache resolution.
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "f" * 40)
    monkeypatch.setattr(qwen_mod, "ensure_venv", lambda *a, **k: tmp_path)
    monkeypatch.setattr(qwen_mod, "venv_python", lambda *a, **k: "python")

    captured: dict = {}

    def spy_env(
        repo_id: str, revision: str = "main", *, also: object = None, base_env: object = None, **k: object
    ) -> None:
        captured["repo_id"] = repo_id
        captured["revision"] = revision
        captured["also"] = also
        raise _StopBeforeSubprocess

    monkeypatch.setattr(qwen_mod, "hf_subprocess_env", spy_env)

    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    model: HFModel = HFModel(path_or_uri="Qwen/Qwen3-ASR-1.7B", revision="main")

    with pytest.raises(_StopBeforeSubprocess):
        qwen_mod.QwenASR.transcribe_with_qwen([audio], model=model, return_timestamps=True, device=DeviceType.CPU)

    assert captured["repo_id"] == "Qwen/Qwen3-ASR-1.7B"
    # A resolved commit SHA, never the mutable "main" ref -- the fix this test guards.
    assert captured["revision"] == "f" * 40
    assert captured["also"] == [("Qwen/Qwen3-ForcedAligner-0.6B", "f" * 40)]
