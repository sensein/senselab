"""Standardized video encoding: GOP, B-frames, timestamps, frame count, audio, rotation, edge inputs.

Every case encodes a short synthetic clip (moving ``testsrc2``, B-frames and a 250-frame GOP in the source, like
a camera file) with the real settings (``-preset slow``) and checks the contract with ``ffprobe``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from senselab.video.tasks.standardize_encoding import (
    MODEL_INPUT_SPECS,
    EmptyVideoError,
    VideoStandardizationError,
    get_model_spec,
    standardize_video_encoding,
    target_dims,
)
from senselab.video.tasks.standardize_encoding.api import (
    frame_list,
    gop_length,
    is_variable_frame_rate,
    nvenc_available,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")


def _has_x264() -> bool:
    if not FFMPEG:
        return False
    out = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, check=False).stdout
    return b"libx264" in out


pytestmark = pytest.mark.skipif(not (FFMPEG and FFPROBE and _has_x264()), reason="ffmpeg/ffprobe with libx264 needed")


def _make(
    path: Path,
    rate: str = "30",
    seconds: int = 4,
    audio: Optional[str] = "aac",
    size: str = "320x240",
    pix: str = "yuv420p",
    vf: str = "",
) -> Path:
    assert FFMPEG
    argv = [FFMPEG, "-nostdin", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d={seconds}"]
    if audio:
        argv += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}"]
    if vf:
        argv += ["-vf", vf]
    argv += [
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-pix_fmt",
        pix,
        "-g",
        "250",
        "-bf",
        "2",
        "-fps_mode",
        "passthrough",
    ]
    argv += {"aac": ["-c:a", "aac", "-b:a", "96k"], "pcm": ["-c:a", "pcm_s16le", "-ac", "2"], None: []}[audio]
    subprocess.run(argv + [str(path)], capture_output=True, check=True)
    return path


def _streams(path: Path) -> List[Dict[str, Any]]:
    assert FFPROBE
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-show_streams", "-of", "json", str(path)], capture_output=True, check=True
    )
    return json.loads(out.stdout)["streams"]


@pytest.mark.parametrize("rate,keyint", [("25", 25), ("30", 30), ("60", 60), ("30000/1001", 30), ("24000/1001", 24)])
def test_constant_rate_gop_and_timing(tmp_path: Path, rate: str, keyint: int) -> None:
    """Keyframes at exactly every round(fps) frames, no B-frames, same frames and timestamps."""
    src = _make(tmp_path / "src.mp4", rate=rate)
    result = standardize_video_encoding(src, tmp_path / "out" / "o.mp4")
    prov, check = result.provenance, result.provenance["check"]
    assert result.ok, check
    assert prov["gop_frames"] == keyint == prov["settings"]["keyint"]
    assert prov["vfr"] is False
    out_frames = frame_list(result.output_path)
    assert [i for i, f in enumerate(out_frames) if f[2]] == list(range(0, len(out_frames), keyint))
    assert all(f[1] != "B" for f in out_frames)
    assert check["frames"] == prov["frames_decoded"] == len(frame_list(src))
    assert check["pts_max_err_ms"] <= 1.0
    assert (check["width"], check["height"]) == (320, 240)
    assert prov["source"]["sha256"] != prov["output"]["sha256"] and prov["output"]["size"] > 0
    assert prov["ffmpeg_version"].startswith("ffmpeg version")


def test_variable_frame_rate_keeps_timestamps(tmp_path: Path) -> None:
    """Dropped frames keep the source's irregular timestamps; the GOP is round(avg fps) frames."""
    src = _make(tmp_path / "vfr.mp4", vf="select='not(eq(mod(n,7),3))'")
    result = standardize_video_encoding(src, tmp_path / "o.mp4")
    assert result.ok, result.provenance["check"]
    assert result.provenance["vfr"] is True
    assert result.provenance["avg_frame_rate"] != result.provenance["r_frame_rate"]
    assert result.provenance["check"]["pts_max_err_ms"] <= 1.0
    assert result.provenance["gop_frames"] == gop_length(26)


def test_audio_copied_when_aac_and_encoded_otherwise(tmp_path: Path) -> None:
    """AAC is copied; PCM becomes AAC 192k; rate, channels and duration are kept; none stays none."""
    aac = standardize_video_encoding(_make(tmp_path / "a.mp4"), tmp_path / "a_o.mp4")
    assert aac.ok and aac.provenance["settings"]["audio"] == "copy"
    pcm = standardize_video_encoding(_make(tmp_path / "p.mov", audio="pcm"), tmp_path / "p_o.mp4")
    assert pcm.ok and pcm.provenance["settings"]["audio"] == "aac-192k"
    assert pcm.provenance["check"]["audio_dur_err_s"] <= 0.05
    audio_stream = [s for s in _streams(pcm.output_path) if s["codec_type"] == "audio"][0]
    assert (
        audio_stream["codec_name"] == "aac"
        and int(audio_stream["sample_rate"]) == 44100
        and audio_stream["channels"] == 2
    )
    silent = standardize_video_encoding(_make(tmp_path / "s.mp4", audio=None), tmp_path / "s_o.mp4")
    assert silent.ok and silent.provenance["audio"] is None
    assert not [s for s in _streams(silent.output_path) if s["codec_type"] == "audio"]


def test_rotation_is_applied_and_odd_size_cropped(tmp_path: Path) -> None:
    """A display-matrix rotation becomes upright pixels without metadata; odd dimensions are cropped to even."""
    assert FFMPEG
    raw = _make(tmp_path / "raw.mp4")
    rotated = tmp_path / "rot.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-display_rotation",
            "90",
            "-i",
            str(raw),
            "-c",
            "copy",
            str(rotated),
        ],
        check=True,
    )
    result = standardize_video_encoding(rotated, tmp_path / "rot_o.mp4")
    assert result.ok, result.provenance["check"]
    assert result.provenance["rotation"] in (90, 270)
    assert (result.provenance["check"]["width"], result.provenance["check"]["height"]) == (240, 320)
    assert not _streams(result.output_path)[0].get("side_data_list")
    odd = standardize_video_encoding(_make(tmp_path / "odd.mp4", size="321x241", pix="yuv444p"), tmp_path / "odd_o.mp4")
    assert odd.ok
    assert (odd.provenance["check"]["width"], odd.provenance["check"]["height"]) == (320, 240)


def test_metadata_is_stripped(tmp_path: Path) -> None:
    """No title/creation_time/encoder tags survive; the source's tags do not leak."""
    assert FFMPEG and FFPROBE
    src = tmp_path / "tagged.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(_make(tmp_path / "raw.mp4")),
            "-c",
            "copy",
            "-metadata",
            "title=secret-id",
            "-metadata:s:v:0",
            "handler_name=secret-id",
            str(src),
        ],
        check=True,
    )
    result = standardize_video_encoding(src, tmp_path / "o.mp4")
    probe = subprocess.run(
        [FFPROBE, "-v", "error", "-show_format", "-show_streams", "-of", "json", str(result.output_path)],
        capture_output=True,
        check=True,
    )
    assert b"secret-id" not in probe.stdout and b"secret-id" not in result.output_path.read_bytes()
    tags = set(json.loads(probe.stdout)["format"].get("tags", {})) - {
        "major_brand",
        "minor_version",
        "compatible_brands",
    }
    assert not tags


def test_edge_inputs_raise_fixed_codes_and_leave_nothing(tmp_path: Path) -> None:
    """Empty and corrupt sources raise with a fixed code and leave no partial output."""
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with pytest.raises(EmptyVideoError):
        standardize_video_encoding(empty, tmp_path / "o" / "e.mp4")
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(os.urandom(4096))
    with pytest.raises(VideoStandardizationError) as err:
        standardize_video_encoding(corrupt, tmp_path / "o" / "c.mp4")
    assert err.value.code in ("unreadable", "no-video")
    with pytest.raises(VideoStandardizationError):
        standardize_video_encoding(tmp_path / "missing.mp4", tmp_path / "o" / "m.mp4")
    assert not list((tmp_path / "o").glob("*")) if (tmp_path / "o").exists() else True


def test_missing_ffmpeg_is_a_clear_error(tmp_path: Path) -> None:
    """An explicit bad ffmpeg path fails at encode time with a fixed code, never a partial file."""
    src = _make(tmp_path / "src.mp4", seconds=1)
    with pytest.raises(VideoStandardizationError):
        standardize_video_encoding(src, tmp_path / "o.mp4", ffmpeg=tmp_path / "nope")
    assert not (tmp_path / "o.mp4").exists()


def test_stream_copy_window_at_keyframe_is_decodable(tmp_path: Path) -> None:
    """A stream-copy cut between two keyframes holds exactly the frames between them and starts on a key frame."""
    assert FFMPEG
    result = standardize_video_encoding(_make(tmp_path / "src.mp4", seconds=6), tmp_path / "o.mp4")
    frames = frame_list(result.output_path)
    k = result.provenance["gop_frames"]
    start, stop = frames[k][0], frames[3 * k][0]
    assert start is not None and stop is not None
    win = tmp_path / "w.mp4"
    subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            f"{start:.6f}",
            "-i",
            str(result.output_path),
            "-t",
            f"{stop - start - 0.0002:.6f}",
            "-c",
            "copy",
            "-an",
            str(win),
        ],
        check=True,
    )
    cut = frame_list(win)
    assert len(cut) == 2 * k and cut[0][2]


def test_helpers() -> None:
    """GOP clamp and VFR detection."""
    assert gop_length(29.97) == 30 and gop_length(None) == 30 and gop_length(0.2) == 1 and gop_length(5000) == 300
    assert not is_variable_frame_rate([i / 30 for i in range(50)])
    assert is_variable_frame_rate([0, 0.033, 0.066, 0.2, 0.233])


def test_target_dims_modes() -> None:
    """Pure geometry: keep, longest side, fit, pad, crop, stretch, rotation and multiples."""
    assert target_dims(321, 241, 0, None, "fit")["out"] == (320, 240)
    assert target_dims(1920, 1080, 0, None, "fit")["identity"]
    assert target_dims(1920, 1080, 90, None, "fit")["out"] == (1080, 1920)
    assert target_dims(1920, 1080, 0, 960, "fit")["out"] == (960, 540)
    assert target_dims(640, 360, 0, 960, "fit")["identity"]  # never upscaled
    assert target_dims(1920, 1080, 0, (640, 640), "fit")["out"] == (640, 360)
    assert target_dims(1920, 1080, 0, (640, 640), "pad")["out"] == (640, 640)
    assert target_dims(1920, 1080, 0, (640, 640), "crop")["out"] == (640, 640)
    assert target_dims(1920, 1080, 0, (640, 480), "stretch")["scale"] == (640, 480)
    assert target_dims(1280, 720, 0, None, "fit", multiple=32)["out"] == (1280, 704)
    with pytest.raises(ValueError):
        target_dims(1920, 1080, 0, (641, 480), "fit")


def test_resolution_and_audio_targets_are_applied(tmp_path: Path) -> None:
    """Pad/crop reach exact sizes; audio is resampled and downmixed; the GOP still tracks the source rate."""
    src = _make(tmp_path / "src.mp4", seconds=3)
    for mode in ("pad", "crop", "stretch"):
        result = standardize_video_encoding(
            src, tmp_path / f"{mode}.mp4", resolution=(160, 160), resolution_mode=mode, allow_copy=False
        )
        assert result.ok and (result.provenance["check"]["width"], result.provenance["check"]["height"]) == (
            160,
            160,
        ), mode
    fit = standardize_video_encoding(src, tmp_path / "fit.mp4", resolution=(160, 160), profile="transient")
    assert fit.ok and (fit.provenance["check"]["width"], fit.provenance["check"]["height"]) == (160, 120)
    assert fit.provenance["settings"]["preset"] == "veryfast"
    audio = standardize_video_encoding(src, tmp_path / "au.mp4", audio_rate=16000, audio_channels=1)
    assert audio.ok and audio.provenance["settings"]["audio"] == "aac-192k"
    stream = [s for s in _streams(audio.output_path) if s["codec_type"] == "audio"][0]
    assert int(stream["sample_rate"]) == 16000 and stream["channels"] == 1


def test_fps_target_resamples_and_resets_gop(tmp_path: Path) -> None:
    """A target rate gives a constant-rate output, GOP = round(target), frame count by span."""
    src = _make(tmp_path / "src.mp4", rate="30", seconds=4)
    result = standardize_video_encoding(src, tmp_path / "o.mp4", fps=10)
    assert result.ok, result.provenance["check"]
    assert result.provenance["gop_frames"] == 10 and abs(result.provenance["check"]["frames"] - 40) <= 2
    vfr = standardize_video_encoding(
        _make(tmp_path / "v.mp4", vf="select='not(eq(mod(n,7),3))'"), tmp_path / "vo.mp4", fps=15
    )
    assert vfr.ok, vfr.provenance["check"]


def test_fast_path_remuxes_a_compliant_source(tmp_path: Path) -> None:
    """Standardizing an already standard file copies the video (no generation loss); only audio is encoded if needed."""
    first = standardize_video_encoding(_make(tmp_path / "src.mp4"), tmp_path / "std.mp4")
    assert first.provenance["mode"] == "reencode"
    again = standardize_video_encoding(first.output_path, tmp_path / "again.mp4")
    assert again.ok and again.provenance["mode"] == "remux" and again.provenance["backend"] == "copy"
    assert again.provenance["encode_cpu_s"] < first.provenance["encode_cpu_s"]
    retarget = standardize_video_encoding(first.output_path, tmp_path / "re.mp4", audio_rate=16000)
    assert retarget.ok and retarget.provenance["mode"] == "audio-only"
    assert (
        not standardize_video_encoding(first.output_path, tmp_path / "f.mp4", allow_copy=False).provenance["mode"]
        == "remux"
    )


def test_cache_reuses_outputs_by_content(tmp_path: Path) -> None:
    """The same source bytes and parameters hit the cache (even from another path); other parameters miss."""
    src = _make(tmp_path / "src.mp4", seconds=2)
    twin = tmp_path / "twin.mp4"
    twin.write_bytes(src.read_bytes())
    cache = tmp_path / "cache"
    miss = standardize_video_encoding(src, cache_dir=cache, profile="transient")
    hit = standardize_video_encoding(twin, tmp_path / "linked.mp4", cache_dir=cache, profile="transient")
    other = standardize_video_encoding(src, cache_dir=cache, profile="transient", crf=28)
    assert miss.provenance["cache"]["hit"] is False and hit.provenance["cache"]["hit"] is True
    assert (
        other.provenance["cache"]["hit"] is False
        and other.provenance["cache"]["key"] != miss.provenance["cache"]["key"]
    )
    assert hit.output_path.read_bytes() == miss.output_path.read_bytes()


def test_model_specs_drive_targets(tmp_path: Path) -> None:
    """A registered model name supplies patch multiple and audio targets; explicit arguments win; unknown names fail."""
    assert get_model_spec("Qwen3-VL").encode_targets()["resolution_multiple"] == 32
    assert all(spec.source for spec in MODEL_INPUT_SPECS.values())
    src = _make(tmp_path / "src.mp4", size="352x288", seconds=2)
    result = standardize_video_encoding(src, tmp_path / "o.mp4", model="qwen3-vl", allow_copy=False)
    assert result.ok and (result.provenance["check"]["width"], result.provenance["check"]["height"]) == (352, 288)
    asr = standardize_video_encoding(src, tmp_path / "asr.mp4", model="whisper")
    assert asr.ok and asr.provenance["settings"]["targets"]["audio_rate"] == 16000
    with pytest.raises(KeyError):
        get_model_spec("no-such-model")


@pytest.mark.skipif(not (FFMPEG and nvenc_available(FFMPEG)), reason="no NVENC on this host")
def test_nvenc_backend_meets_the_contract(tmp_path: Path) -> None:
    """GPU encode: same GOP/no-B-frame/frame-count contract, backend recorded."""
    result = standardize_video_encoding(
        _make(tmp_path / "src.mp4", size="640x360"), tmp_path / "o.mp4", backend="h264_nvenc", allow_copy=False
    )
    assert result.provenance["backend"] == "h264_nvenc" and result.ok, result.provenance["check"]


def test_max_seconds_truncates_and_still_verifies(tmp_path: Path) -> None:
    """A truncated (sample) encode keeps the first seconds of video and audio and passes verification."""
    result = standardize_video_encoding(_make(tmp_path / "src.mp4", seconds=6), tmp_path / "o.mp4", max_seconds=2)
    assert result.ok, result.provenance["check"]
    assert result.provenance["mode"] == "reencode" and abs(result.provenance["check"]["frames"] - 60) <= 1
