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
    # The *ref*, not the resolved SHA -- and this assertion is inverted from what it originally
    # claimed, because the property it pinned was the defect. Staging is what makes
    # resolve_model -> _point_ref_at re-point refs/<ref> at the run's pinned commit, and
    # _point_ref_at returns immediately when its ref argument is already a SHA. Since upstream
    # qwen_asr reads its processor config bare (AutoProcessor.from_pretrained(path,
    # fix_mistral_regex=True), no revision passthrough), that read follows refs/<ref> whatever the
    # payload pins: staging by SHA leaves it naming whatever "main" last resolved to on this host,
    # or nothing at all on a cold cache. The pin itself is untouched -- the worker payload still
    # carries the resolved SHA for the loads that accept one, which
    # revision_pinning_guard_test.REVISION_RESOLVED_SUBPROCESS_FILES covers.
    assert captured["revision"] == "main"
    assert captured["also"] == [("Qwen/Qwen3-ForcedAligner-0.6B", "main")]
