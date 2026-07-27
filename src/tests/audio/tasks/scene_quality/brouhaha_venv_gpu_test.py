"""GPU-gated regression: the brouhaha subprocess venv must build on a CUDA host.

Guards the ``pick_torch_index`` routing bug reported on PR #536 (wilke0818): on a
modern-CUDA host the wheel index resolves to ``cu128``, on which brouhaha's
``torch>=2.0,<2.3`` pin has no linux-x86_64 wheel, so ``ensure_venv`` cannot
provision the venv and every scene-quality signal (SNR/reverb/clip/bandwidth)
comes back null — failing the strict scene-quality gate.

Runs only where CUDA is present (the EC2 GPU runner). On CPU hosts brouhaha
resolves to CPU wheels and the bug does not manifest, so the test skips there
(local dev, the CPU CI job). It isolates the venv *build* (torch/torchaudio
import), which fails before the gated ``pyannote/brouhaha`` model is ever
loaded, so it needs no ``HF_TOKEN`` / model access.
"""

from __future__ import annotations

import subprocess

import pytest

torch = pytest.importorskip("torch")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="brouhaha venv-build regression only manifests on CUDA hosts (cu12x wheel index)",
)
def test_brouhaha_subprocess_venv_builds_on_cuda_host() -> None:
    """``ensure_venv`` must provision an importable torch/torchaudio into the brouhaha venv.

    Failure here means the CUDA-aware wheel-index selection picked an index with
    no wheel satisfying brouhaha's ``torch>=2.0,<2.3`` pin (the reported bug).
    """
    from senselab.audio.tasks.scene_quality.brouhaha import (
        _BROUHAHA_MAX_CUDA_VERSION,
        _BROUHAHA_PYTHON,
        _BROUHAHA_REQUIREMENTS,
        _BROUHAHA_VENV,
    )
    from senselab.utils.subprocess_venv import ensure_venv, venv_python

    # Build exactly as extract_brouhaha_frames does — with brouhaha's declared
    # CUDA-index ceiling. This is the regression point: drop/blank the ceiling
    # and a modern-CUDA host routes through cu128 (no torch<2.3 wheel) and this
    # install fails.
    assert _BROUHAHA_MAX_CUDA_VERSION is not None, "brouhaha must declare a CUDA-index ceiling"
    venv_dir = ensure_venv(
        _BROUHAHA_VENV,
        _BROUHAHA_REQUIREMENTS,
        python_version=_BROUHAHA_PYTHON,
        max_cuda_version=_BROUHAHA_MAX_CUDA_VERSION,
    )
    python = venv_python(venv_dir)
    result = subprocess.run(
        [str(python), "-c", "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        "brouhaha venv failed to import torch/torchaudio — the CUDA-aware wheel-index "
        f"selection likely chose an index with no torch>=2.0,<2.3 wheel:\n{result.stderr[-800:]}"
    )
