"""Audio-visual target speaker extraction: input validation, dispatch, payload, provenance.

The pipeline itself is never run. It needs a talking-face recording, ffmpeg, the 734 MB checkpoint and
the 86 MB face detector; no verified recording was available on this branch, which is recorded as a
limitation in the task's ``doc.md`` and in design.md D-3 rather than papered over with a fabricated
result. What is tested here is every host-side decision: which containers are accepted, that a
frames-only ``Video`` is refused, what reaches the worker, and what the returned ``Audio`` records.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any, Dict

import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.target_speaker_extraction import (
    DEFAULT_TSE_MODEL,
    extract_target_speakers_from_videos,
)
from senselab.audio.tasks.target_speaker_extraction.clearvoice import (
    SUPPORTED_VIDEO_SUFFIXES,
    _video_path,
    extract_target_speakers_with_clearvoice,
    video_duration_s,
)
from senselab.utils import clearvoice as cv
from senselab.utils.backend_parameters import PARAMETER_RECORD_KEY
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.portable_audio_io import write_audio

STAGED_SHA = "e" * 40


@pytest.fixture
def offline_hub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construct ``HFModel`` without reaching the Hub."""
    monkeypatch.setattr("senselab.utils.data_structures.model.check_hf_repo_exists", lambda **kwargs: True)
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: STAGED_SHA)


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    """A file with an accepted extension. Its contents are never decoded on the host."""
    path = tmp_path / "meeting.mp4"
    path.write_bytes(b"not really a video")
    return path


@pytest.fixture
def worker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Dict[str, Any]:
    """Stub the venv, both staging calls and the subprocess; return ``n_tracks`` extracted WAVs."""
    captured: Dict[str, Any] = {"n_tracks": 2}
    monkeypatch.setattr(cv, "ensure_venv", lambda *a, **k: tmp_path / "venv")
    monkeypatch.setattr(cv, "venv_python", lambda venv_dir: "python3")
    monkeypatch.setattr(
        cv, "stage_clearvoice_checkpoints", lambda spec, revision="main": (tmp_path / "ckpt", STAGED_SHA)
    )
    monkeypatch.setattr(cv, "stage_s3fd_weights", lambda: tmp_path / "sfd_face.pth")

    def fake_run(cmd: list, **kwargs: Any) -> types.SimpleNamespace:
        payload = json.loads(kwargs["input"])
        captured["payload"] = payload
        captured["timeout"] = kwargs["timeout"]
        outputs = []
        for index, _ in enumerate(payload["video_paths"]):
            tracks = []
            for track in range(captured["n_tracks"]):
                out_path = f"{payload['output_dir']}/v{index}_est_{track}.wav"
                write_audio(out_path, __import__("numpy").zeros(1600, dtype="float32"), 16000)
                tracks.append(out_path)
            outputs.append(tracks)
        return types.SimpleNamespace(
            returncode=0, stdout=json.dumps({"output_paths": outputs, "device": "cpu"}), stderr=""
        )

    monkeypatch.setattr(cv.subprocess, "run", fake_run)
    return captured


def _model() -> HFModel:
    return HFModel(path_or_uri=DEFAULT_TSE_MODEL, revision="main")


# ── Input validation ──────────────────────────────────────────────────


def test_the_accepted_containers_are_the_ones_upstream_reads() -> None:
    """read_and_config_file matches these four; anything else falls through to a text read."""
    assert set(SUPPORTED_VIDEO_SUFFIXES) == {".mp4", ".avi", ".mov", ".webm"}


def test_an_unsupported_container_is_refused_with_the_list(tmp_path: Path) -> None:
    """Upstream would try to read an .mkv as a list of paths and fail obscurely."""
    path = tmp_path / "clip.mkv"
    path.write_bytes(b"x")
    with pytest.raises(ValueError) as exc:
        _video_path(path)
    assert ".mp4" in str(exc.value) and "Remux" in str(exc.value)


def test_a_missing_file_is_named(tmp_path: Path) -> None:
    """The pipeline shells out to ffmpeg, which would fail far from the cause."""
    with pytest.raises(FileNotFoundError, match="absent.mp4"):
        _video_path(tmp_path / "absent.mp4")


def test_a_file_backed_video_object_is_accepted(video_file: Path) -> None:
    """``Video`` is the natural input type; its path is what the pipeline needs."""
    from senselab.video.data_structures import Video

    assert _video_path(Video(filepath=str(video_file))) == video_file


def test_a_frames_only_video_is_refused_with_the_reason(mono_audio_sample: Audio) -> None:
    """The pipeline re-encodes the container and extracts its audio track; frames are not enough."""
    import torch

    from senselab.video.data_structures import Video

    frames = torch.zeros((2, 4, 4, 3), dtype=torch.uint8)
    video = Video(frames=frames, frame_rate=25.0)
    with pytest.raises(ValueError) as exc:
        _video_path(video)
    assert "needs the video file" in str(exc.value)
    assert "filepath=" in str(exc.value)


def test_an_unreadable_duration_does_not_refuse_the_video(video_file: Path) -> None:
    """The duration only informs a timeout, so failing to read it must not fail the run."""
    assert video_duration_s(video_file) is None


# ── Dispatch and payload ──────────────────────────────────────────────


def test_extraction_returns_one_audio_per_face_track(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """One speaker per detected track is the capability's output shape."""
    extracted = extract_target_speakers_from_videos([video_file])
    assert len(extracted) == 1 and len(extracted[0]) == 2
    assert all(audio.sampling_rate == 16000 for audio in extracted[0])


def test_each_extracted_speaker_records_its_track_and_the_commit(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """Provenance must survive, and the track index is what identifies the speaker."""
    extracted = extract_target_speakers_from_videos([video_file])[0]
    for index, audio in enumerate(extracted):
        record = audio.metadata["clearvoice"]
        assert record["commit"] == STAGED_SHA
        assert record["model"] == DEFAULT_TSE_MODEL
        assert record["face_track_index"] == index
        assert audio.metadata["source_video"] == str(video_file)


def test_no_detected_face_returns_an_empty_list_not_an_error(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """A video with no trackable face is an outcome; raising would conflate it with a failure."""
    worker["n_tracks"] = 0
    assert extract_target_speakers_from_videos([video_file]) == [[]]


def test_the_verified_face_detector_is_handed_to_the_worker(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """Otherwise the detector shells out to gdown for an unversioned copy."""
    extract_target_speakers_from_videos([video_file])
    assert worker["payload"]["s3fd_weights"].endswith("sfd_face.pth")
    assert worker["payload"]["mode"] == "tse"


def test_the_device_reaches_the_worker(offline_hub: None, worker: Dict[str, Any], video_file: Path) -> None:
    """Face detection is the expensive part, so the device choice matters here most."""
    extract_target_speakers_from_videos([video_file], device=DeviceType.CPU)
    assert worker["payload"]["device"] == "cpu"


def test_the_ceiling_is_the_video_one_not_the_audio_one(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """Per-frame detection at 25 fps plus three ffmpeg passes is not comparable to decoding audio."""
    extract_target_speakers_from_videos([video_file])
    assert worker["timeout"] == cv.default_tse_timeout_s(0.0) == cv._TSE_TIMEOUT_FLOOR_S


def test_an_explicit_ceiling_is_forwarded(offline_hub: None, worker: Dict[str, Any], video_file: Path) -> None:
    """The derived term is unmeasured, so the override is the documented way out."""
    extract_target_speakers_from_videos([video_file], parameters={"timeout_s": 77.0})
    assert worker["timeout"] == 77.0


def test_parameters_are_validated_and_recorded(
    offline_hub: None, worker: Dict[str, Any], video_file: Path
) -> None:
    """Same pathway as the audio tasks: unknown keys raise, chosen ones are recorded."""
    extracted = extract_target_speakers_from_videos([video_file], parameters={"timeout_s": 77.0})
    assert extracted[0][0].metadata[PARAMETER_RECORD_KEY]["explicit"] == ["timeout_s"]
    with pytest.raises(ValueError, match="Unknown parameter"):
        extract_target_speakers_from_videos([video_file], parameters={"variant": "x"})


def test_no_video_means_no_worker(offline_hub: None, worker: Dict[str, Any]) -> None:
    """An empty list must not stage 820 MB of weights."""
    assert extract_target_speakers_from_videos([]) == []
    assert "payload" not in worker


def test_a_checkpoint_for_another_capability_is_refused(offline_hub: None) -> None:
    """The table decides which entry point owns which checkpoint."""
    with pytest.raises(ValueError) as exc:
        extract_target_speakers_with_clearvoice(
            [], HFModel(path_or_uri="alibabasglab/FRCRN_SE_16K", revision="main")
        )
    assert "speech_enhancement.enhance_audios" in str(exc.value)
