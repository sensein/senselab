"""ClearVoice's shared machinery: the model table, the pin, the device, the ceiling, the worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import types
from pathlib import Path
from typing import Any, Dict, List

import pytest

from senselab.utils import clearvoice as cv
from senselab.utils.data_structures import DeviceType

SHA_CHARS = set("0123456789abcdef")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


# ── The capability table ──────────────────────────────────────────────


def test_the_table_holds_exactly_the_six_shipped_checkpoints() -> None:
    """clearvoice==0.1.2's network_wrapper dispatches on these six names and no others."""
    assert set(cv.CLEARVOICE_MODELS) == {
        "FRCRN_SE_16K",
        "MossFormerGAN_SE_16K",
        "MossFormer2_SE_48K",
        "MossFormer2_SS_16K",
        "MossFormer2_SR_48K",
        "AV_MossFormer2_TSE_16K",
    }


def test_every_checkpoint_names_the_upstream_task_that_loads_it() -> None:
    """A wrong task string reaches network_wrapper's else branch, which prints and returns None."""
    assert {spec.upstream_task for spec in cv.CLEARVOICE_MODELS.values()} == {
        "speech_enhancement",
        "speech_separation",
        "speech_super_resolution",
        "target_speaker_extraction",
    }


def test_the_four_capabilities_partition_the_table() -> None:
    """Every checkpoint is reachable from exactly one task package."""
    counted = sum(len(cv.clearvoice_models_for_task(task)) for task in cv._TASK_OWNERS)
    assert counted == len(cv.CLEARVOICE_MODELS)
    assert [spec.name for spec in cv.clearvoice_models_for_task("speech_enhancement")] == [
        "FRCRN_SE_16K",
        "MossFormerGAN_SE_16K",
        "MossFormer2_SE_48K",
    ]


def test_only_the_two_models_upstream_normalises_are_marked_as_normalising() -> None:
    """DataReader.extract_feature names exactly these two; the flag must not spread."""
    normalising = {spec.name for spec in cv.CLEARVOICE_MODELS.values() if spec.rms_normalises_input}
    assert normalising == {"FRCRN_SE_16K", "MossFormer2_SS_16K"}


def test_model_ids_are_org_qualified() -> None:
    """The id a caller passes must be the HuggingFace repository id."""
    assert cv.CLEARVOICE_MODELS["FRCRN_SE_16K"].model_id == "alibabasglab/FRCRN_SE_16K"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("alibabasglab/FRCRN_SE_16K", True),
        ("alibabasglab/anything-else", True),
        ("LIANGXU123/DriftSE", False),
        ("speechbrain/sepformer-wham16k-enhancement", False),
    ],
)
def test_recognition_is_organisation_wide(model_id: str, expected: bool) -> None:
    """An unknown model under the org must reach the six-checkpoint message, not "no backend"."""
    assert cv.is_clearvoice_model_id(model_id) is expected


def test_a_bare_model_name_resolves_as_well_as_a_full_id() -> None:
    """Both spellings appear in registries and configs."""
    assert cv.clearvoice_model_spec("FRCRN_SE_16K").name == "FRCRN_SE_16K"
    assert cv.clearvoice_model_spec("alibabasglab/FRCRN_SE_16K").name == "FRCRN_SE_16K"


def test_an_unknown_checkpoint_enumerates_the_real_ones() -> None:
    """A caller inventing a name gets the list, not a shrug."""
    with pytest.raises(ValueError) as exc:
        cv.clearvoice_model_spec("alibabasglab/MossFormer3_SE_96K")
    assert "MossFormer2_SE_48K" in str(exc.value)


def test_a_checkpoint_for_another_capability_is_refused_with_the_right_entry_point() -> None:
    """Running the separator through the enhancer would hand two sources to a one-signal caller."""
    with pytest.raises(ValueError) as exc:
        cv.clearvoice_model_spec("MossFormer2_SS_16K", expected_task="speech_enhancement")
    message = str(exc.value)
    assert "speech separation" in message
    assert "source_separation.separate_audios" in message


# ── The pin ───────────────────────────────────────────────────────────


def test_staging_downloads_only_the_files_the_commit_manifest_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The 1.74 GB optimizer state in the super-resolution repo must not be fetched."""
    sha = "a" * 40
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    # The two-line manifest MossFormer2_SR_48K actually ships.
    (snapshot / "last_best_checkpoint").write_text("last_best_checkpoint_m.pt\nlast_best_checkpoint_g.pt\n")
    requested: List[Dict[str, Any]] = []

    def fake_download(repo_id: str, filename: str, revision: str = "main") -> str:
        requested.append({"repo": repo_id, "file": filename, "revision": revision})
        return str(snapshot / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)

    checkpoint_dir, resolved = cv.stage_clearvoice_checkpoints(cv.CLEARVOICE_MODELS["MossFormer2_SR_48K"])

    assert resolved == sha
    assert checkpoint_dir == snapshot
    assert [entry["file"] for entry in requested] == [
        "last_best_checkpoint",
        "last_best_checkpoint_m.pt",
        "last_best_checkpoint_g.pt",
    ]
    assert all(entry["revision"] == sha for entry in requested), "every file must be fetched at the commit"
    assert not any("do_" in entry["file"] for entry in requested)


def test_staging_resolves_the_ref_to_a_commit_before_downloading(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The loader takes no revision, so the parent's resolution is the only pin there is."""
    sha = "b" * 40
    snapshot = tmp_path / "snapshots" / sha
    snapshot.mkdir(parents=True)
    (snapshot / "last_best_checkpoint").write_text("last_best_checkpoint.pt\n")
    seen: Dict[str, Any] = {}

    def fake_resolve(repo_id: str, ref: str) -> str:
        seen["repo"], seen["ref"] = repo_id, ref
        return sha

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", fake_resolve)
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo_id, filename, revision="main": str(snapshot / filename),
    )

    _, resolved = cv.stage_clearvoice_checkpoints(cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], revision="main")
    assert seen == {"repo": "alibabasglab/FRCRN_SE_16K", "ref": "main"}
    assert len(resolved) == 40 and set(resolved) <= SHA_CHARS


def test_the_s3fd_digest_is_a_full_sha256_at_a_pinned_commit() -> None:
    """This weight has no revision-addressable home, so the digest is its identity."""
    assert len(cv._S3FD_COMMIT) == 40 and set(cv._S3FD_COMMIT) <= SHA_CHARS
    assert len(cv._S3FD_SHA256) == 64 and set(cv._S3FD_SHA256) <= SHA_CHARS
    assert cv._S3FD_COMMIT in cv._S3FD_URL, "the URL must be commit-addressed, not branch-addressed"
    assert "raw.githubusercontent.com" in cv._S3FD_URL


def test_a_wrong_digest_refuses_the_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A silently changed face detector would move every track while reporting success."""
    monkeypatch.setattr(cv, "_cache_dir_path", lambda: tmp_path)

    class _Response:
        def __init__(self, payload: bytes) -> None:
            self._payload = payload

        def read(self, size: int) -> bytes:
            chunk, self._payload = self._payload[:size], self._payload[size:]
            return chunk

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    monkeypatch.setattr(cv.urllib.request, "urlopen", lambda url: _Response(b"not the detector"))
    with pytest.raises(RuntimeError) as exc:
        cv.stage_s3fd_weights()
    assert hashlib.sha256(b"not the detector").hexdigest() in str(exc.value)
    assert cv._S3FD_SHA256 in str(exc.value)
    staged = tmp_path / "clearvoice" / "s3fd" / cv._S3FD_SHA256 / "sfd_face.pth"
    assert not staged.exists(), "a mismatched download must not be left in the cache"


def test_a_matching_digest_is_cached_content_addressed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A digest change is a cache miss, never a stale hit."""
    payload = b"pretend detector"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(cv, "_cache_dir_path", lambda: tmp_path)
    monkeypatch.setattr(cv, "_S3FD_SHA256", digest)

    class _Response:
        def __init__(self) -> None:
            self._payload = payload

        def read(self, size: int) -> bytes:
            chunk, self._payload = self._payload[:size], self._payload[size:]
            return chunk

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    calls = {"n": 0}

    def fake_urlopen(url: str) -> "_Response":
        calls["n"] += 1
        return _Response()

    monkeypatch.setattr(cv.urllib.request, "urlopen", fake_urlopen)
    first = cv.stage_s3fd_weights()
    assert first.read_bytes() == payload
    assert digest in str(first)
    second = cv.stage_s3fd_weights()
    assert second == first
    assert calls["n"] == 1, "a staged file must not be re-downloaded"


# ── The device ────────────────────────────────────────────────────────


def test_no_device_leaves_the_choice_to_the_worker() -> None:
    """Only the venv's own torch can say whether CUDA works there."""
    assert cv.resolve_worker_device(None) is None


def test_cpu_is_sent_as_cpu() -> None:
    """An explicit CPU request must reach the worker, not be re-derived there."""
    assert cv.resolve_worker_device(DeviceType.CPU) == "cpu"


def test_mps_is_refused_rather_than_silently_accepted() -> None:
    """Upstream would select MPS whenever present; none of the six is verified on it."""
    with pytest.raises(ValueError):
        cv.resolve_worker_device(DeviceType.MPS)


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device on this host; the index cannot be resolved")
def test_cuda_resolves_to_an_explicit_index() -> None:
    """A bare "cuda" would let the worker's torch pick, discarding the caller's choice."""
    resolved = cv.resolve_worker_device(DeviceType.CUDA)
    assert resolved is not None and resolved.startswith("cuda:")


# ── The ceiling ───────────────────────────────────────────────────────


def test_the_audio_ceiling_grows_with_the_work() -> None:
    """A fixed ceiling is what discards a legitimate long run."""
    long_run = cv.default_audio_timeout_s(3600.0)
    assert long_run > cv.default_audio_timeout_s(600.0) > 0
    assert long_run == pytest.approx(cv._TIMEOUT_HEADROOM * cv._SECONDS_PER_AUDIO_SECOND * 3600.0)


def test_the_audio_ceiling_has_a_floor_for_the_fixed_costs() -> None:
    """A one-second input still pays for the venv's first torch import and a cold checkpoint read."""
    assert cv.default_audio_timeout_s(1.0) == cv._TIMEOUT_FLOOR_S
    assert cv.default_audio_timeout_s(0.0) == cv._TIMEOUT_FLOOR_S


def test_the_video_ceiling_is_separate_and_larger() -> None:
    """Per-frame face detection at 25 fps is not comparable to decoding audio."""
    assert cv.default_tse_timeout_s(1.0) == cv._TSE_TIMEOUT_FLOOR_S
    assert cv.default_tse_timeout_s(600.0) > cv.default_audio_timeout_s(600.0)


@pytest.mark.parametrize("bad", [0, -1.0])
def test_a_non_positive_ceiling_is_refused(bad: float) -> None:
    """A zero timeout would kill the worker instantly and report a timeout as the cause."""
    spec = cv.CLEARVOICE_MODELS["FRCRN_SE_16K"]
    with pytest.raises(ValueError, match="positive number of seconds"):
        cv.run_clearvoice_audio(spec, [], "/tmp", total_audio_s=1.0, timeout_s=bad)
    with pytest.raises(ValueError, match="positive number of seconds"):
        cv.run_clearvoice_tse(spec, [], "/tmp", total_video_s=1.0, timeout_s=bad)


# ── The worker payload ────────────────────────────────────────────────


def _stub_worker(monkeypatch: pytest.MonkeyPatch, captured: dict, tmp_path: Path, sha: str = "c" * 40) -> None:
    """Replace the venv, the staging and the subprocess with fakes that record what was sent."""
    monkeypatch.setattr(cv, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(cv, "venv_python", lambda venv_dir: "python3")
    monkeypatch.setattr(cv, "stage_clearvoice_checkpoints", lambda spec, revision="main": (tmp_path / "ckpt", sha))
    monkeypatch.setattr(cv, "stage_s3fd_weights", lambda: tmp_path / "sfd_face.pth")

    def fake_run(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        captured["payload"] = json.loads(str(kwargs["input"]))
        captured["timeout"] = kwargs["timeout"]
        payload = captured["payload"]
        body: Dict[str, Any]
        if payload["mode"] == "tse":
            body = {"output_paths": [[] for _ in payload["video_paths"]], "device": "cpu"}
        else:
            body = {
                "output_paths": [[f"{payload['out_dir']}/out_0_s0.wav"] for _ in payload["in_paths"]],
                "input_norm_scalars": [1.0 for _ in payload["in_paths"]],
                "device": "cpu",
            }
        return types.SimpleNamespace(returncode=0, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(cv.subprocess, "run", fake_run)


def test_the_payload_carries_the_resolved_commit_path_and_the_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The worker is pinned by the path it is handed, and told which device to use."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    paths, scalars, sha = cv.run_clearvoice_audio(
        cv.CLEARVOICE_MODELS["FRCRN_SE_16K"],
        ["/tmp/in.wav"],
        str(tmp_path),
        total_audio_s=10.0,
        device=DeviceType.CPU,
    )
    payload = captured["payload"]
    assert payload["checkpoint_dir"] == str(tmp_path / "ckpt")
    assert payload["device"] == "cpu"
    assert payload["expected_version"] == cv.CLEARVOICE_VERSION
    assert payload["rms_normalise"] is True, "FRCRN is one of the two models upstream normalises"
    assert payload["task"] == "speech_enhancement"
    assert sha == "c" * 40
    assert scalars == [1.0] and len(paths) == 1


def test_the_payload_carries_the_staged_io_policy_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The worker imports the range policy from a file the parent copies next to the payload."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    cv.run_clearvoice_audio(cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=1.0)
    io_dir = Path(captured["payload"]["io_dir"])
    assert (io_dir / "portable_audio_io.py").is_file() or io_dir.name.startswith("senselab-clearvoice")


def test_no_device_is_sent_as_null_not_as_a_guess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The host must not resolve CUDA on behalf of a different torch build."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    cv.run_clearvoice_audio(cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=1.0)
    assert captured["payload"]["device"] is None


def test_the_derived_ceiling_reaches_the_subprocess(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A derived ceiling that never reaches subprocess.run is a ceiling in name only."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    cv.run_clearvoice_audio(cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=600.0)
    assert captured["timeout"] == cv.default_audio_timeout_s(600.0)


def test_an_explicit_ceiling_overrides_the_derived_one(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The override exists because the shared term is coarse."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    cv.run_clearvoice_audio(
        cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=600.0, timeout_s=42.0
    )
    assert captured["timeout"] == 42.0


def test_a_timeout_names_the_ceiling_the_work_and_the_way_out(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A discarded run must say what was attempted and which knob raises the limit."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)

    def timing_out(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd, float(kwargs["timeout"]))  # type: ignore[arg-type]

    monkeypatch.setattr(cv.subprocess, "run", timing_out)
    with pytest.raises(RuntimeError) as exc:
        cv.run_clearvoice_audio(
            cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=30.0, timeout_s=5.0
        )
    message = str(exc.value)
    assert "5s ceiling" in message
    assert "30s of audio over 1 input(s)" in message
    assert "timeout_s" in message and "CUDA" in message
    assert "discarded" in message


def test_a_worker_failure_preserves_the_upstream_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The whole point of the blocked downloader is that its message reaches the caller."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    blocked = "clearvoice reached SpeechModel.download_model for FRCRN_SE_16K"

    def failing(cmd: list, **kwargs: object) -> types.SimpleNamespace:
        body = {"error": {"type": "RuntimeError", "message": blocked, "traceback": "..."}}
        return types.SimpleNamespace(returncode=1, stdout=json.dumps(body), stderr="")

    monkeypatch.setattr(cv.subprocess, "run", failing)
    with pytest.raises(RuntimeError, match="download_model"):
        cv.run_clearvoice_audio(cv.CLEARVOICE_MODELS["FRCRN_SE_16K"], ["/tmp/in.wav"], str(tmp_path), total_audio_s=1.0)


def test_the_tse_payload_carries_the_verified_detector_and_the_video_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The audio-visual mode needs a second weight file, and it is the digest-verified one."""
    captured: dict = {}
    _stub_worker(monkeypatch, captured, tmp_path)
    tracks, sha = cv.run_clearvoice_tse(
        cv.CLEARVOICE_MODELS["AV_MossFormer2_TSE_16K"],
        ["/tmp/a.mp4"],
        str(tmp_path),
        total_video_s=12.0,
        device=DeviceType.CPU,
    )
    payload = captured["payload"]
    assert payload["mode"] == "tse"
    assert payload["s3fd_weights"] == str(tmp_path / "sfd_face.pth")
    assert payload["video_paths"] == ["/tmp/a.mp4"]
    assert payload["task"] == "target_speaker_extraction"
    assert captured["timeout"] == cv.default_tse_timeout_s(12.0)
    assert tracks == [[]] and sha == "c" * 40


# ── The worker source ─────────────────────────────────────────────────


def test_the_worker_blocks_the_unpinned_downloader() -> None:
    """Making the download unreachable is the pin; leaving it merely unused is not."""
    assert "cvnet.SpeechModel.download_model = _blocked_download" in cv._WORKER_SCRIPT


def test_the_worker_asserts_the_distribution_version_it_patches() -> None:
    """The device patch reconstructs __init__ field by field, so the version cannot drift silently."""
    assert 'installed != args["expected_version"]' in cv._WORKER_SCRIPT
    assert "clearvoice.__version__" not in cv._WORKER_SCRIPT, "that attribute reports 0.1.0 in 0.1.2"


def test_the_worker_writes_through_the_staged_policy_and_never_soundfile() -> None:
    """Every write must get the same subtype resolution and range policy as an in-process one."""
    assert "from portable_audio_io import read_audio, write_audio" in cv._WORKER_SCRIPT
    assert "sf.write" not in cv._WORKER_SCRIPT
    assert "import soundfile" not in cv._WORKER_SCRIPT


def test_the_worker_refuses_a_cuda_request_the_venv_cannot_honour() -> None:
    """The host's torch and the venv's torch are separate builds; only the venv's answer counts."""
    assert "torch.cuda.is_available()" in cv._WORKER_SCRIPT
    assert 'requested.startswith("cuda")' in cv._WORKER_SCRIPT


def test_the_worker_never_names_a_bare_cuda_device() -> None:
    """An index is always chosen, so a CUDA_VISIBLE_DEVICES mask selects the allocated card."""
    assert 'torch.device("cuda")' not in cv._WORKER_SCRIPT
    assert '"cuda:%d" % torch.cuda.current_device()' in cv._WORKER_SCRIPT


# ── The venv spec ─────────────────────────────────────────────────────


def test_torch_is_named_so_the_cuda_wheel_routing_fires() -> None:
    """ensure_venv decides on the CUDA-aware index by reading this list."""
    named = {req.split("=")[0].split(">")[0].split("<")[0].strip().lower() for req in cv.CLEARVOICE_REQUIREMENTS}
    assert {"torch", "torchaudio"} <= named


def test_the_pinned_distribution_is_exact() -> None:
    """The worker's monkeypatches are written against one version, so it must be pinned to it."""
    assert f"clearvoice=={cv.CLEARVOICE_VERSION}" in cv.CLEARVOICE_REQUIREMENTS


def test_the_speechscore_venv_is_separate_from_the_platform_venv() -> None:
    """Their dependency sets are disjoint; sharing one venv would over-constrain both."""
    assert cv.SPEECHSCORE_VENV != cv.CLEARVOICE_VENV
    assert len(cv.SPEECHSCORE_COMMIT) == 40 and set(cv.SPEECHSCORE_COMMIT) <= SHA_CHARS
