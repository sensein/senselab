"""Integration tests for ``senselab.utils.subprocess_venv.ensure_venv`` CUDA routing.

All ``subprocess.run`` invocations are mocked. No real venv is created.
Covers: marker-mismatch rebuild paths, install-argv routing through the
chosen PyTorch index, ``SenselabCudaCompatibilityError`` wrapping on wheel
not-found errors, pass-through of unrelated failures, and the same
behavior across the three real subprocess-venv backends.
"""

import json
import subprocess
from collections.abc import Callable
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
        self.hook: Optional[Callable[[list[list[str]]], subprocess.CompletedProcess]] = None

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
            return self.hook(self.calls)
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
    # Three subprocess invocations: uv venv + Stage-1 torch install + Stage-2
    # backend install. If we'd hit the cache fast-path we'd see zero.
    assert len(recorder.calls) == 3
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
    assert len(recorder.calls) == 3  # uv venv + Stage-1 torch + Stage-2 backend install
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url


# ── Install argv routing ───────────────────────────────────────────


def test_stage_one_pins_torch_and_torchaudio_to_chosen_index(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-1 install names ONLY the chosen CUDA index and only torch + torchaudio.

    No ``--extra-index-url`` is allowed in Stage 1: uv treats it as having
    higher priority than ``--index-url``, so a PyPI fallback would let
    PyPI's mismatched-tag torch / torchaudio win for these two packages.
    """
    name = "t-stage-one"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"], python_version="3.12")

    # recorder.calls[0] = uv venv; [1] = Stage 1 install; [2] = Stage 2 install.
    stage_one = recorder.calls[1]
    assert stage_one[0:3] == [fake_uv, "pip", "install"]
    idx_pos = stage_one.index("--index-url")
    assert stage_one[idx_pos + 1] == force_cu128.url
    assert "--extra-index-url" not in stage_one
    # Pinned specs from requirements flow through verbatim.
    assert "torch>=2.8,<2.9" in stage_one
    assert "torchaudio>=2.8,<2.9" in stage_one


def test_stage_two_uses_default_pypi_and_includes_ipc_deps(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stage-2 install carries no index flags and includes the IPC serialization deps.

    No ``--index-url`` or ``--extra-index-url`` on Stage 2 — by then torch
    and torchaudio are already installed, so the rest of the resolution
    happens against default PyPI (where setuptools, NeMo, etc. live at
    current versions instead of stale CUDA-index mirrors).
    """
    name = "t-stage-two"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    # Includes a torch pin so Stage 1 runs and we have a Stage 2 to assert against.
    ensure_venv(name, ["coqui-tts~=0.27", "torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"], python_version="3.12")

    stage_two = recorder.calls[2]
    assert stage_two[0:3] == [fake_uv, "pip", "install"]
    assert "--index-url" not in stage_two
    assert "--extra-index-url" not in stage_two
    # IPC deps appended after the backend's own requirements.
    assert "safetensors" in stage_two
    assert "numpy" in stage_two
    # torchaudio was installed in Stage 1 and intentionally NOT re-listed
    # here — if it were, uv would consider re-resolving it from PyPI.
    assert "torchaudio" not in stage_two
    # Backend's own non-torch requirements still flow through.
    assert "coqui-tts~=0.27" in stage_two


def test_stage_two_filters_torch_specs_from_caller_requirements(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backend-pinned torch / torchaudio specs are stripped from Stage 2.

    If ``requirements`` contains e.g. ``torch>=2.8,<2.9``, that spec must
    NOT appear in the Stage-2 argv. Re-listing it without an index flag
    would let uv consider replacing the matched ``+cu128`` wheel installed
    in Stage 1 with whatever PyPI happens to ship at the same public
    version (currently a ``+cu129`` tag) — exactly the split this fix is
    meant to prevent.
    """
    name = "t-stage-two-filter"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(
        name,
        ["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9", "pyarrow<18"],
        python_version="3.12",
    )

    stage_two = recorder.calls[2]
    assert "torch>=2.8,<2.9" not in stage_two
    assert "torchaudio>=2.8,<2.9" not in stage_two
    # Non-torch requirements still flow through.
    assert "pyarrow<18" in stage_two


def test_torch_install_specs_preserves_multiple_constraints_for_same_package(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend passing ``torch>=2.8`` AND ``torch<2.9`` as two specs forwards both.

    The earlier dict-based implementation kept only the last spec it saw
    for a given package name; a caller relying on multiple constraints
    would have ended up with Stage 1 enforcing only one of them, and uv
    could have resolved a version that violated the other. The list-based
    form keeps every matching spec.
    """
    name = "t-multi-constraint"
    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["torch>=2.8", "torch<2.9"], python_version="3.12")

    stage_one = recorder.calls[1]
    assert "torch>=2.8" in stage_one
    assert "torch<2.9" in stage_one


# ── Auto-detection: torch-free venvs skip the probe and Stage 1 ────


def test_no_torch_in_requirements_skips_probe_and_stage_one(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backends without ``torch`` / ``torchaudio`` in ``requirements`` skip the probe entirely.

    ``ensure_venv`` decides the install shape from the caller's
    ``requirements`` itself — no separate flag. A venv that declares
    neither package (yamnet, continuous-ser, or a future pure-Python
    backend) gets the leanest possible install: one ``uv pip install``
    pass against default PyPI, no ``nvidia-smi`` shellout, no
    ``torchaudio`` force-appended.
    """
    name = "t-no-torch"
    # If the probe were invoked it would raise — that's how we confirm
    # auto-detection short-circuits before reaching it.
    monkeypatch.setattr(
        subprocess_venv,
        "detect_host_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("probe should not run when no torch in requirements")),
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    ensure_venv(name, ["some-pure-python-pkg==1.0"], python_version="3.12")

    # Exactly two subprocess calls: uv venv + a single uv pip install.
    assert len(recorder.calls) == 2
    install_argv = recorder.calls[1]
    assert install_argv[0:3] == [fake_uv, "pip", "install"]
    assert "--index-url" not in install_argv
    assert "--extra-index-url" not in install_argv
    # No torch routing → no Stage 1 → torchaudio is NOT force-appended.
    assert "torchaudio" not in install_argv
    assert "torch" not in install_argv
    # Caller's requirements + safetensors + numpy still install.
    assert "some-pure-python-pkg==1.0" in install_argv
    assert "safetensors" in install_argv
    assert "numpy" in install_argv

    # Marker carries no ``torch_index`` field, so a later call whose
    # requirements grow a torch spec will correctly invalidate + rebuild.
    written = json.loads((fake_cache_dir / name / ".senselab-installed").read_text())
    assert "torch_index" not in written


def test_adding_torch_to_requirements_invalidates_torch_free_cache(
    fake_cache_dir: Path,
    fake_uv: str,
    force_cu128: TorchIndex,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A venv built without torch must rebuild if the requirements later grow a torch spec.

    The torch-free marker has no ``torch_index`` field; when a torch-
    requiring call sees its expected ``torch_index.url`` differs from
    the stored ``None``, the marker mismatches and the venv rebuilds.
    """
    name = "t-switch-to-torch"
    venv_dir = fake_cache_dir / name
    venv_dir.mkdir(parents=True)
    (venv_dir / ".senselab-installed").write_text(
        json.dumps({"requirements": ["some-pure-python-pkg==1.0"], "python_version": "3.12"})
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    # Same name, but requirements now include a torch pin → Stage 1 must fire.
    ensure_venv(name, ["some-pure-python-pkg==1.0", "torch>=2.8,<2.9"], python_version="3.12")

    # uv venv + Stage 1 + Stage 2 — full rebuild kicked in.
    assert len(recorder.calls) == 3
    written = json.loads((venv_dir / ".senselab-installed").read_text())
    assert written["torch_index"]["url"] == force_cu128.url


def test_yamnet_style_requirements_omit_torchaudio_from_install(
    fake_cache_dir: Path,
    fake_uv: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Yamnet / continuous-ser-style venvs no longer get ``torchaudio`` force-installed.

    Before this change, every subprocess venv paid for ``torchaudio``
    via an unconditional IPC append, even backends that read audio via
    ``soundfile`` and never imported torchaudio (~200 MB of wheels per
    venv for no functional benefit). Now that the append is gone,
    such a venv ends up with neither torch nor torchaudio in its
    install argv.
    """
    name = "t-yamnet-style"
    monkeypatch.setattr(
        subprocess_venv,
        "detect_host_cuda",
        lambda: (_ for _ in ()).throw(AssertionError("torch-free venv must not invoke the probe")),
    )

    recorder = _SubprocessRecorder()
    monkeypatch.setattr(subprocess, "run", recorder)

    # Mirror yamnet's actual _REQUIREMENTS shape — TF-based, soundfile for audio I/O.
    ensure_venv(
        name,
        ["tensorflow", "tensorflow-hub", "setuptools<70", "numpy", "soundfile"],
        python_version="3.12",
    )

    install_argv = recorder.calls[1]
    assert "torch" not in install_argv
    assert "torchaudio" not in install_argv
    # tensorflow / tensorflow-hub still flow through.
    assert "tensorflow" in install_argv
    assert "tensorflow-hub" in install_argv


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
    """Walk the three real backend requirement lists; each must route Stage 1 through cu128.

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
        # Stage 1 = recorder.calls[1] = torch + torchaudio via the chosen CUDA index.
        stage_one = recorder.calls[1]
        idx_pos = stage_one.index("--index-url")
        assert stage_one[idx_pos + 1] == force_cu128.url, f"{label} backend did not use cu128 index"
        # No --extra-index-url on Stage 1: the precedence quirk would let PyPI win otherwise.
        assert "--extra-index-url" not in stage_one, f"{label} backend leaked --extra-index-url into Stage 1"
        # Every backend must install both torch and torchaudio from the same index.
        assert any(s == "torch" or s.startswith("torch>") or s.startswith("torch=") for s in stage_one), (
            f"{label} backend missing torch in Stage 1"
        )
        assert any(
            s == "torchaudio" or s.startswith("torchaudio>") or s.startswith("torchaudio=") for s in stage_one
        ), f"{label} backend missing torchaudio in Stage 1"
        # Stage 2 must NOT carry the CUDA index — that's the whole point.
        stage_two = recorder.calls[2]
        assert "--index-url" not in stage_two, f"{label} backend leaked --index-url into Stage 2"
        assert "--extra-index-url" not in stage_two, f"{label} backend leaked --extra-index-url into Stage 2"
        # And no torch / torchaudio specs in Stage 2 either: listing them
        # without an index flag would let uv consider replacing the
        # matched wheel from Stage 1 with PyPI's tagless one.
        assert not any(s == "torch" or s.startswith("torch>") or s.startswith("torch=") for s in stage_two), (
            f"{label} backend leaked a torch spec into Stage 2"
        )
        assert not any(
            s == "torchaudio" or s.startswith("torchaudio>") or s.startswith("torchaudio=") for s in stage_two
        ), f"{label} backend leaked a torchaudio spec into Stage 2"
