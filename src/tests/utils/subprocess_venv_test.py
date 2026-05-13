"""Integration tests for ``senselab.utils.subprocess_venv.ensure_venv`` CUDA routing.

All ``subprocess.run`` invocations are mocked. No real venv is created.
Covers: marker-mismatch rebuild paths, install-argv routing through the
chosen PyTorch index, ``SenselabCudaCompatibilityError`` wrapping on wheel
not-found errors, pass-through of unrelated failures, and the same
behavior across the three real subprocess-venv backends.
"""

import json
import subprocess
from pathlib import Path
from typing import Optional

import pytest

from senselab.utils import subprocess_venv
from senselab.utils.cuda_probe import HostCuda, SenselabCudaCompatibilityError, TorchIndex
from senselab.utils.subprocess_venv import _classify_uv_failure, ensure_venv

# ── Fixtures + helpers ─────────────────────────────────────────────


@pytest.fixture
def fake_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``_cache_dir()`` to a tmp dir so we never touch ~/.cache."""
    monkeypatch.setattr(subprocess_venv, "_cache_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def fake_uv(monkeypatch: pytest.MonkeyPatch) -> str:
    """Make ``_find_uv()`` return a deterministic fake path so it doesn't shell out to ``which uv``."""
    monkeypatch.setattr(subprocess_venv, "_find_uv", lambda: "/fake/uv")
    return "/fake/uv"


@pytest.fixture
def force_cu128(monkeypatch: pytest.MonkeyPatch) -> TorchIndex:
    """Pin the resolved index to cu128 so tests don't depend on the host's actual CUDA."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = TorchIndex(
        url="https://download.pytorch.org/whl/cu128",
        tag="cu128",
        cuda_version=(12, 8),
        source="static-map",
    )
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)
    monkeypatch.setattr(subprocess_venv, "pick_torch_index", lambda host_cuda, env_override=None: idx)
    return idx


class _SubprocessRecorder:
    """Replacement for ``subprocess.run`` that records calls and replays canned results.

    ``uv venv`` is simulated by creating the target directory so the rest of
    ``ensure_venv`` (which writes the marker file inside it) doesn't trip on
    the missing parent.
    """

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        # Per-call hook — set to raise/return per recorded call index.
        self.hook = None  # type: Optional[object]

    def __call__(
        self,
        argv: list[str],
        check: bool = False,
        capture_output: bool = False,
        text: bool = False,
        **_: object,
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(argv))
        # Simulate ``uv venv --python X /path/to/venv`` by mkdir-ing the target.
        if len(argv) >= 5 and argv[1] == "venv" and argv[2] == "--python":
            Path(argv[4]).mkdir(parents=True, exist_ok=True)
        # If a hook is set, let it drive the response (raise / return).
        if self.hook is not None:
            return self.hook(self.calls)  # type: ignore[no-any-return]
        return subprocess.CompletedProcess(args=argv, returncode=0, stdout="", stderr="")


# ── Marker mismatch + rebuild paths ────────────────────────────────


def test_marker_without_torch_index_triggers_rebuild(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An existing marker from the pre-fix code (no ``torch_index`` key) must rebuild."""
    name = "t-no-index"
    venv_dir = fake_cache_dir / name
    venv_dir.mkdir(parents=True)
    (venv_dir / ".senselab-installed").write_text(
        json.dumps({"requirements": ["torch>=2.8,<2.9"], "python_version": "3.12"})
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    # Two subprocess invocations: uv venv + uv pip install. If we'd hit the
    # cache fast-path we'd see zero.
    assert len(recorder.calls) == 2
    # New marker now carries the resolved index.
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url
    assert written["torch_index"]["tag"] == "cu128"


def test_marker_with_matching_torch_index_is_cache_hit(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker whose requirements + ``torch_index.url`` match → no install, no rebuild."""
    name = "t-cache-hit"
    venv_dir = fake_cache_dir / name
    venv_dir.mkdir(parents=True)
    (venv_dir / ".senselab-installed").write_text(
        json.dumps(
            {
                "requirements": ["torch>=2.8,<2.9"],
                "python_version": "3.12",
                "torch_index": {
                    "tag": force_cu128.tag,
                    "url": force_cu128.url,
                    "source": force_cu128.source,
                },
            }
        )
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    # Zero subprocess invocations on the cache fast-path.
    assert recorder.calls == []


def test_marker_with_different_torch_index_triggers_rebuild(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A marker with a different ``torch_index.url`` (e.g. resolution changed) must rebuild."""
    name = "t-different-index"
    venv_dir = fake_cache_dir / name
    venv_dir.mkdir(parents=True)
    (venv_dir / ".senselab-installed").write_text(
        json.dumps(
            {
                "requirements": ["torch>=2.8,<2.9"],
                "python_version": "3.12",
                "torch_index": {
                    "tag": "cu121",
                    "url": "https://download.pytorch.org/whl/cu121",
                    "source": "static-map",
                },
            }
        )
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    out = ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    assert out == venv_dir
    assert len(recorder.calls) == 2  # uv venv + uv pip install
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url


# ── Install argv routing ───────────────────────────────────────────


def test_install_argv_routes_through_chosen_torch_index(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uv pip install argv contains both --index-url and --extra-index-url, in that order."""
    name = "t-argv"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"], python_version="3.12")

    # Second call is `uv pip install ...`.
    install_argv = recorder.calls[1]
    assert install_argv[0:3] == [fake_uv, "pip", "install"]
    assert "--index-url" in install_argv
    idx_pos = install_argv.index("--index-url")
    assert install_argv[idx_pos + 1] == force_cu128.url
    assert "--extra-index-url" in install_argv
    extra_pos = install_argv.index("--extra-index-url")
    assert install_argv[extra_pos + 1] == "https://pypi.org/simple"
    # --index-url must come before --extra-index-url so uv's resolver prefers it.
    assert idx_pos < extra_pos


def test_install_argv_appends_default_ipc_dependencies(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ensure_venv keeps appending safetensors, numpy, torchaudio after the caller's reqs."""
    name = "t-ipc-deps"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["coqui-tts~=0.27"], python_version="3.12")

    install_argv = recorder.calls[1]
    for dep in ("safetensors", "numpy", "torchaudio"):
        assert dep in install_argv, f"expected {dep!r} in install argv"


# ── env override ───────────────────────────────────────────────────


def test_env_override_routes_through_override_url_and_still_probes_for_diagnostic(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SENSELAB_TORCH_INDEX_URL routes the install verbatim; probe still runs for diagnostic surface."""
    name = "t-override"
    override_url = "https://pypi.internal.example.com/pytorch/cu128"
    monkeypatch.setenv("SENSELAB_TORCH_INDEX_URL", override_url)

    # Probe still runs so the host-CUDA value is available for any
    # ``SenselabCudaCompatibilityError`` message produced on the override path.
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    install_argv = recorder.calls[1]
    idx_pos = install_argv.index("--index-url")
    assert install_argv[idx_pos + 1] == override_url
    # Marker records the override source.
    written = json.loads((fake_cache_dir / name / ".senselab-installed").read_text())
    assert written["torch_index"]["source"] == "env-override"
    assert written["torch_index"]["tag"] == "override"


def test_env_override_empty_string_does_not_short_circuit(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty SENSELAB_TORCH_INDEX_URL must be ignored (treated as unset)."""
    name = "t-override-empty"
    monkeypatch.setenv("SENSELAB_TORCH_INDEX_URL", "")

    host = HostCuda(version=(12, 4), source="nvidia-smi", raw="CUDA Version: 12.4")
    monkeypatch.setattr(subprocess_venv, "detect_host_cuda", lambda: host)

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    written = json.loads((fake_cache_dir / name / ".senselab-installed").read_text())
    assert written["torch_index"]["source"] == "static-map"
    assert written["torch_index"]["tag"] == "cu124"


# ── Failure handling ───────────────────────────────────────────────


def test_no_matching_distribution_failure_wraps_into_compatibility_error(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uv 'no matching distribution' → SenselabCudaCompatibilityError, venv removed."""
    name = "t-wheel-missing"

    def _hook(calls: list[list[str]]) -> subprocess.CompletedProcess:
        if calls[-1][1] == "venv":
            return subprocess.CompletedProcess(args=calls[-1], returncode=0, stdout="", stderr="")
        # pip install: fail with a "no matching distribution" stderr.
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=calls[-1],
            output="",
            stderr=(
                "error: No solution found when resolving dependencies:\n"
                "  No matching distribution for `torch>=2.8,<2.9`\n"
            ),
        )

    recorder = _SubprocessRecorder()
    recorder.hook = _hook
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(SenselabCudaCompatibilityError) as excinfo:
        ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")

    err = excinfo.value
    assert err.attempted_index.url == force_cu128.url
    assert any("torch>=2.8,<2.9" in p for p in err.failing_packages)
    # Half-built venv must be wiped.
    assert not (fake_cache_dir / name).exists()


def test_unrelated_install_failure_passes_through_as_called_process_error(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Network / permission / unrelated errors → original CalledProcessError, not wrapped."""
    name = "t-network-error"

    def _hook(calls: list[list[str]]) -> subprocess.CompletedProcess:
        if calls[-1][1] == "venv":
            return subprocess.CompletedProcess(args=calls[-1], returncode=0, stdout="", stderr="")
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=calls[-1],
            output="",
            stderr="error: connection timed out while reading https://pypi.org/simple/torch/\n",
        )

    recorder = _SubprocessRecorder()
    recorder.hook = _hook
    monkeypatch.setattr(subprocess, "run", recorder)

    with pytest.raises(subprocess.CalledProcessError):
        ensure_venv(name, ["torch>=2.8,<2.9"], python_version="3.12")
    # The half-built venv is also wiped for unrelated failures so next run starts clean.
    assert not (fake_cache_dir / name).exists()


# ── _classify_uv_failure unit tests ────────────────────────────────


def test_classify_uv_failure_no_matching_distribution() -> None:
    """Uv 'no matching distribution' stderr → returns the failing package list."""
    stderr = (
        "error: No solution found when resolving dependencies:\n  No matching distribution for `torch==2.8.1+cu128`\n"
    )
    failing = _classify_uv_failure(stderr)
    assert failing == ["torch==2.8.1+cu128"]


def test_classify_uv_failure_could_not_find_version() -> None:
    """Uv 'could not find a version that satisfies' stderr → returns the failing package list."""
    stderr = "error: Could not find a version that satisfies the requirement `torchaudio>=2.8,<2.9`\n"
    failing = _classify_uv_failure(stderr)
    assert failing == ["torchaudio>=2.8,<2.9"]


def test_classify_uv_failure_unrelated_returns_none() -> None:
    """Unrelated errors (network, syntax, permission) must return None."""
    assert _classify_uv_failure("error: connection timed out") is None
    assert _classify_uv_failure("error: permission denied while writing to /tmp/foo") is None
    assert _classify_uv_failure("") is None


def test_classify_uv_failure_dedupes_repeated_specs() -> None:
    """Multiple mentions of the same failing spec collapse to one entry."""
    stderr = (
        "error: No matching distribution for `torch>=2.8,<2.9`\nwarning: try setting an index for `torch>=2.8,<2.9`\n"
    )
    failing = _classify_uv_failure(stderr)
    assert failing == ["torch>=2.8,<2.9"]


# ── Cross-backend regression test (US3) ────────────────────────────


def test_all_three_subprocess_backends_route_through_same_torch_index(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Walk the three real backend requirement lists; each must emit the same routed argv.

    Guards against any future backend bypassing ``ensure_venv`` with its own
    install shellout — every subprocess venv in the project must share the
    CUDA-aware resolution.
    """
    from senselab.audio.tasks.speech_to_text.canary_qwen import _CANARY_REQUIREMENTS
    from senselab.audio.tasks.speech_to_text.nemo import _NEMO_REQUIREMENTS
    from senselab.audio.tasks.speech_to_text.qwen import _QWEN_REQUIREMENTS

    for label, reqs in (("canary", _CANARY_REQUIREMENTS), ("nemo", _NEMO_REQUIREMENTS), ("qwen", _QWEN_REQUIREMENTS)):
        recorder = _SubprocessRecorder()
        monkeypatch.setattr(subprocess, "run", recorder)
        ensure_venv(f"t-{label}", list(reqs), python_version="3.12")
        install_argv = recorder.calls[1]
        idx_pos = install_argv.index("--index-url")
        assert install_argv[idx_pos + 1] == force_cu128.url, f"{label} backend did not use cu128 index"
        extra_pos = install_argv.index("--extra-index-url")
        assert install_argv[extra_pos + 1] == "https://pypi.org/simple", f"{label} backend missing --extra-index-url"
