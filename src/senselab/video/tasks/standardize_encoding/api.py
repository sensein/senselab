"""Standardize a video file to one fixed, seek-friendly H.264/AAC encoding, keeping its native timing.

Adapted from the BBQS encoding-helper "recommended" encode (MIT licence,
https://github.com/brain-bbqs/encoding-helper): H.264, a fixed short GOP, ``-preset slow -crf 23``,
``yuv420p``, AAC, ``+faststart`` and stripped metadata. Deviations, each deliberate: the source frame rate
and presentation timestamps are kept (no ``-r``), there are no B-frames, and the GOP is exactly one second
of the source's average frame rate with scene-cut keyframes disabled, so keyframes sit at frame indices
``0, K, 2K, ...`` and a stream-copy cut at a keyframe always starts decodable.

Everything except the codec contract is a configurable target that defaults to "keep the source": resolution,
frame rate, audio sample rate and channels (see ``standardize_video_encoding`` and ``model_specs``). Two profiles:
``archival`` (``-preset slow``, the dataset-wide standard) and ``transient`` (``-preset veryfast``, for on-the-fly
preparation). A source that already meets the contract is remuxed instead of re-encoded; ``backend="h264_nvenc"``
(or ``"auto"``) encodes on the GPU; ``cache_dir`` reuses outputs by content.

Calls the ``ffmpeg`` and ``ffprobe`` executables (PATH, or the explicit ``ffmpeg``/``ffprobe`` arguments);
both must be built with ``libx264``. ``scripts/install-ffmpeg.sh`` installs a suitable build.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from .model_specs import ModelInputSpec

SPEC_VERSION = 1
DEFAULT_CRF = 23
PROFILES: Dict[str, Dict[str, str]] = {
    "archival": {"preset": "slow", "nvenc_preset": "p7"},
    "transient": {"preset": "veryfast", "nvenc_preset": "p4"},
}
RESOLUTION_MODES = ("fit", "pad", "crop", "stretch")
FPS_METHODS = ("drop-dup",)
BACKENDS = ("libx264", "h264_nvenc", "auto")
DEFAULT_THREADS = 4
DEFAULT_AUDIO_KBPS = 192
MAX_KEYINT = 300
PTS_TOLERANCE_MS = 1.0
AUDIO_TOLERANCE_S = 0.05
_TIMEOUT_S = 12 * 3600

PathLike = Union[str, "os.PathLike[str]"]
Resolution = Union[None, int, Tuple[int, int]]
_Frame = Tuple[Optional[float], str, bool]


class VideoStandardizationError(RuntimeError):
    """Standardization failed.

    ``code`` is one of ``ffmpeg-missing``, ``unreadable``, ``no-video``, ``no-frames``, ``no-pts``,
    ``encode-failed``, ``timeout``.
    """

    def __init__(self, code: str) -> None:
        """Keep the fixed failure code (no path or message that could carry an identifier)."""
        super().__init__(code)
        self.code = code


class EmptyVideoError(VideoStandardizationError):
    """The source file is 0 bytes."""

    def __init__(self) -> None:
        """Code ``empty-file``."""
        super().__init__("empty-file")


@dataclass
class StandardizedVideo:
    """The standardized file and how it was made.

    Attributes:
        output_path: The written MP4.
        provenance: Source and output hashes and sizes, fps (``avg_frame_rate``, ``r_frame_rate``, ``fps``),
            ``vfr``, ``duration``, ``frames_decoded``, geometry, rotation, audio, ``settings`` (the exact encode
            parameters), ``ffmpeg_version``, timings and ``check`` (the post-encode verification).
        ok: Whether the verification passed (frame count, timestamps, GOP, no B-frames, audio).
    """

    output_path: Path
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when the output passed every verification check."""
        return bool(self.provenance.get("check", {}).get("ok", False))


def _tool(name: str, explicit: Optional[PathLike]) -> str:
    found = (
        (str(explicit) if explicit and os.access(str(explicit), os.X_OK) else None) if explicit else shutil.which(name)
    )
    if not found:
        raise VideoStandardizationError("ffmpeg-missing")
    return found


def _probe_json(ffprobe: str, path: PathLike) -> Dict[str, Any]:
    try:
        done = subprocess.run(
            [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            capture_output=True,
            check=False,
            timeout=3600,
        )
        return json.loads(done.stdout) if done.returncode == 0 else {}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return {}


def _float(*values: object) -> Optional[float]:
    for value in values:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _rotation(stream: Dict[str, Any]) -> int:
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                return int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    try:
        return int((stream.get("tags") or {}).get("rotate", 0)) % 360
    except ValueError:
        return 0


def probe_streams(path: PathLike, ffprobe: str = "ffprobe") -> Dict[str, Any]:
    """Header-level facts about the first video and first audio stream (no decoding).

    Returns a dict with ``error`` (``None`` or ``unreadable`` / ``no-video``) and, when readable, ``width``,
    ``height``, ``codec``, ``pix_fmt``, ``avg_frame_rate``, ``r_frame_rate``, ``time_base``, ``rotation``,
    ``duration``, ``n_audio`` and ``audio`` (``codec``, ``sample_rate``, ``channels``, ``duration`` or ``None``).
    """
    data = _probe_json(ffprobe, path)
    streams = data.get("streams") or []
    video = [
        s for s in streams if s.get("codec_type") == "video" and (s.get("disposition") or {}).get("attached_pic") != 1
    ]
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    if not video:
        return {"error": "no-video" if streams else "unreadable"}
    v, a = video[0], (audio[0] if audio else None)
    return {
        "error": None,
        "width": v.get("width"),
        "height": v.get("height"),
        "codec": v.get("codec_name"),
        "pix_fmt": v.get("pix_fmt"),
        "avg_frame_rate": v.get("avg_frame_rate"),
        "r_frame_rate": v.get("r_frame_rate"),
        "time_base": v.get("time_base") or "",
        "rotation": _rotation(v),
        "duration": _float(v.get("duration"), (data.get("format") or {}).get("duration")),
        "n_audio": len(audio),
        "audio": None
        if a is None
        else {
            "codec": a.get("codec_name"),
            "sample_rate": int(a["sample_rate"]) if a.get("sample_rate") else None,
            "channels": a.get("channels"),
            "duration": _float(a.get("duration"), (data.get("format") or {}).get("duration")),
        },
    }


def frame_list(path: PathLike, ffprobe: str = "ffprobe") -> List[_Frame]:
    """Every decoded frame of the first video stream in presentation order: ``(pts_s, pict_type, is_key)``."""
    argv = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
        "frame=pts_time,best_effort_timestamp_time,pict_type,key_frame", "-of", "compact=p=0", str(path),
    ]  # fmt: skip
    try:
        done = subprocess.run(argv, capture_output=True, check=False, timeout=24 * 3600)
    except (OSError, subprocess.TimeoutExpired):
        raise VideoStandardizationError("unreadable") from None
    frames: List[_Frame] = []
    for line in done.stdout.decode("utf-8", "replace").splitlines():
        fields = dict(p.split("=", 1) for p in line.split("|") if "=" in p)
        raw = fields.get("pts_time", "N/A")
        if raw == "N/A":
            raw = fields.get("best_effort_timestamp_time", "N/A")
        pts = None if raw == "N/A" else float(raw)
        frames.append((pts, fields.get("pict_type", "?"), fields.get("key_frame") == "1"))
    if done.returncode != 0 and not frames:
        raise VideoStandardizationError("unreadable")
    return frames


def native_fps(info: Dict[str, Any], pts: List[float]) -> Optional[float]:
    """The source's average frame rate: ``avg_frame_rate``, else frames over the timestamp span."""
    try:
        rate = Fraction(info.get("avg_frame_rate") or "")
        if rate > 0:
            return float(rate)
    except (ValueError, ZeroDivisionError):
        pass
    if len(pts) >= 2 and pts[-1] > pts[0]:
        return (len(pts) - 1) / (pts[-1] - pts[0])
    return None


def gop_length(fps: Optional[float]) -> int:
    """``round(fps)`` frames (one second of the native rate), clamped to ``[1, 300]``; 30 when the rate is unknown."""
    if fps is None or fps <= 0:
        return 30
    return max(1, min(MAX_KEYINT, int(round(fps))))


def is_variable_frame_rate(pts: List[float]) -> bool:
    """True when frame spacings differ by more than 1 % of the median spacing (and more than 1 ms)."""
    if len(pts) < 3:
        return False
    gaps = sorted(b - a for a, b in zip(pts, pts[1:]))
    median = gaps[len(gaps) // 2]
    tol = max(0.001, 0.01 * median)
    return gaps[-1] - median > tol or median - gaps[0] > tol


def file_hashes(path: PathLike) -> Dict[str, str]:
    """``sha256`` of the file, plus ``xxh3_128`` when the optional ``xxhash`` package is installed."""
    sha = hashlib.sha256()
    try:
        import xxhash

        fast: Any = xxhash.xxh3_128()
    except ImportError:
        fast = None
    with open(path, "rb") as handle:
        while chunk := handle.read(1 << 22):
            sha.update(chunk)
            if fast is not None:
                fast.update(chunk)
    out = {"sha256": sha.hexdigest()}
    if fast is not None:
        out["xxh3_128"] = fast.hexdigest()
    return out


def _floor_to(value: float, multiple: int) -> int:
    return max(multiple, int(value // multiple) * multiple)


def target_dims(
    width: int, height: int, rotation: int, resolution: Resolution, mode: str, multiple: int = 2
) -> Dict[str, Any]:
    """Output geometry for a source (rotation applied first). Pure; used by the encode and by the tests.

    ``resolution`` is ``None`` (keep), an ``int`` (longest side at most that many pixels, never upscaled) or
    ``(W, H)`` handled per ``mode``: ``fit`` (scale to fit inside, never upscaled, aspect kept), ``pad`` (fit, then
    pad with black to exactly ``(W, H)``), ``crop`` (scale to cover, centre-crop to exactly ``(W, H)``),
    ``stretch`` (scale to ``(W, H)``, aspect not kept). Dimensions are floored to a multiple of ``multiple``
    (default 2, which yuv420p needs).

    Returns:
        ``out`` (final size), ``src`` (rotated source size), ``fit``, ``scale`` (scale target or ``None``), ``pad``,
        ``crop``, ``trim`` (crop an unscaled source to a multiple) and ``identity`` (no filter needed).
    """
    dw, dh = (height, width) if rotation in (90, 270) else (width, height)
    if mode not in RESOLUTION_MODES:
        raise ValueError(f"resolution mode must be one of {RESOLUTION_MODES}")
    src = (dw, dh)
    scale: Optional[Tuple[int, int]] = None
    pad = crop = False
    if resolution is None:
        fit = out = (_floor_to(dw, multiple), _floor_to(dh, multiple))
    elif isinstance(resolution, int):
        s = min(1.0, resolution / max(dw, dh))
        fit = out = (_floor_to(dw * s, multiple), _floor_to(dh * s, multiple))
        scale = fit if s < 1.0 else None
    else:
        tw, th = resolution
        if tw % multiple or th % multiple:
            raise ValueError(f"resolution {tw}x{th} is not a multiple of {multiple}")
        if mode == "stretch":
            fit = out = scale = (tw, th)
        elif mode == "crop":
            s = max(tw / dw, th / dh)
            cover = (max(tw, math.ceil(dw * s)), max(th, math.ceil(dh * s)))
            scale = cover if cover != src else None
            crop = cover != (tw, th)
            fit = out = (tw, th)
        else:  # fit, pad
            s = min(1.0, tw / dw, th / dh)
            fit = (_floor_to(dw * s, multiple), _floor_to(dh * s, multiple))
            scale = fit if s < 1.0 else None
            out = (tw, th) if mode == "pad" else fit
            pad = mode == "pad" and fit != out
    trim = scale is None and not crop and fit != src  # even/multiple crop of an unscaled source
    return {
        "out": out, "src": src, "fit": fit, "scale": scale, "pad": pad, "crop": crop, "trim": trim,
        "identity": scale is None and not pad and not crop and not trim and out == src,
    }  # fmt: skip


def video_filter(dims: Dict[str, Any]) -> Optional[str]:
    """The ``-vf`` chain for ``target_dims`` output (``None`` when no filter is needed)."""
    chain: List[str] = []
    if dims["scale"]:
        chain.append(f"scale={dims['scale'][0]}:{dims['scale'][1]}:flags=lanczos")
    ow, oh = dims["out"]
    if dims["crop"]:
        chain.append(f"crop={ow}:{oh}")
    if dims["trim"]:
        chain.append(f"crop={dims['fit'][0]}:{dims['fit'][1]}")
    if dims["pad"]:
        chain.append(f"pad={ow}:{oh}:(ow-iw)/2:(oh-ih)/2:color=black")
    return ",".join(chain) or None


def _nvenc_args(preset: str, crf: int, keyint: int) -> List[str]:
    return [
        "-c:v", "h264_nvenc", "-preset", preset, "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-profile:v", "high",
        "-bf", "0", "-g", str(keyint), "-forced-idr", "1", "-no-scenecut", "1", "-pix_fmt", "yuv420p",
    ]  # fmt: skip


_NVENC: Dict[str, bool] = {}


def nvenc_available(ffmpeg: str) -> bool:
    """True when ``ffmpeg`` can open ``h264_nvenc`` here (a one-frame test encode; cached per binary)."""
    if ffmpeg not in _NVENC:
        argv = [ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                "color=c=black:s=256x256:d=0.1", "-frames:v", "1",
                "-c:v", "h264_nvenc", "-f", "null", "-"]  # fmt: skip
        try:
            _NVENC[ffmpeg] = subprocess.run(argv, capture_output=True, check=False, timeout=60).returncode == 0
        except (OSError, subprocess.TimeoutExpired):
            _NVENC[ffmpeg] = False
    return _NVENC[ffmpeg]


def build_ffmpeg_command(
    ffmpeg: str,
    src: str,
    dst: str,
    info: Dict[str, Any],
    keyint: int,
    plan: Dict[str, Any],
    max_seconds: Optional[float] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """The ffmpeg argv and the settings record for a resolved ``plan`` (see ``standardize_video_encoding``)."""
    argv = [ffmpeg, "-nostdin", "-v", "error", "-y"]
    backend, copy_video = plan["backend"], plan["mode"] in ("remux",) or plan["video_copy"]
    if backend == "h264_nvenc" and not copy_video:
        argv += ["-hwaccel", "cuda"]
    if max_seconds:
        argv += ["-t", str(max_seconds)]
    argv += ["-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-map_metadata", "-1", "-map_chapters", "-1"]
    settings: Dict[str, Any] = {"spec_version": SPEC_VERSION, "mode": plan["mode"], "backend": backend}
    if copy_video:
        argv += ["-c:v", "copy"]
        settings["video"] = "copy"
    else:
        vf = plan["vf"]
        if plan["fps"]:
            vf = ",".join(x for x in (vf, f"fps={plan['fps']}") if x)
        if vf:
            argv += ["-vf", vf]
        if backend == "h264_nvenc":
            argv += _nvenc_args(plan["nvenc_preset"], plan["crf"], keyint)
            argv += ["-fps_mode", "passthrough"]
        else:
            x264 = f"keyint={keyint}:min-keyint={keyint}:scenecut=0:bframes=0:threads={plan['threads']}"
            argv += [
                "-c:v", "libx264", "-preset", plan["preset"], "-crf", str(plan["crf"]), "-profile:v", "high",
                "-pix_fmt", "yuv420p", "-x264-params", x264, "-fps_mode", "passthrough",
            ]  # fmt: skip
            settings["x264_params"] = x264
        settings.update({"crf": plan["crf"], "preset": plan["preset"] if backend == "libx264" else plan["nvenc_preset"],
                         "profile": "high", "pix_fmt": "yuv420p", "video": backend, "vf": vf})  # fmt: skip
    base = info.get("time_base") or ""
    if not plan["fps"] and base.startswith("1/") and base[2:].isdigit() and 1 <= int(base[2:]) < 2**31:
        argv += ["-video_track_timescale", base[2:]]  # presentation times round-trip exactly
    audio = info.get("audio")
    if audio:
        if plan["audio_copy"]:
            argv += ["-c:a", "copy"]
            settings["audio"] = "copy"
        else:
            channels = plan["audio_channels"] or audio["channels"] or 2
            kbps = DEFAULT_AUDIO_KBPS if channels <= 2 else 96 * channels
            argv += ["-c:a", "aac", "-b:a", f"{kbps}k"]
            if plan["audio_rate"]:
                argv += ["-ar", str(plan["audio_rate"])]
            if plan["audio_channels"]:
                argv += ["-ac", str(plan["audio_channels"])]
            settings["audio"] = f"aac-{kbps}k"
    else:
        settings["audio"] = None
    argv += ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-flags:a", "+bitexact", "-movflags", "+faststart", dst]
    settings.update({"keyint": keyint, "fps_mode": "passthrough", "metadata": "stripped",
                     "rotation_applied": info.get("rotation") or 0, "targets": plan["targets"]})  # fmt: skip
    return argv, settings


def _run(argv: List[str], stderr_path: Path) -> Tuple[int, float, float]:
    """Run to completion: (exit code, wall seconds, child CPU seconds)."""
    start = time.monotonic()
    with open(stderr_path, "wb") as err:
        proc = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=err)
        while True:
            pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            if pid:
                return os.waitstatus_to_exitcode(status), time.monotonic() - start, usage.ru_utime + usage.ru_stime
            if time.monotonic() - start > _TIMEOUT_S:
                proc.kill()
                os.wait4(proc.pid, 0)
                return 124, time.monotonic() - start, 0.0
            time.sleep(0.1)


def verify_output(
    src_pts: List[float],
    dst: PathLike,
    ffprobe: str,
    keyint: int,
    src_audio: Optional[Dict[str, Any]],
    expect: Dict[str, Any],
) -> Dict[str, Any]:
    """Decode the output and test the contract.

    ``expect`` holds ``dims`` (w, h), ``fps`` (resampled target or None), ``src_fps``, ``audio_rate`` and
    ``audio_channels`` (targets or None).
    """
    frames = frame_list(dst, ffprobe)
    out_pts = [f[0] for f in frames]
    info = probe_streams(dst, ffprobe)
    res: Dict[str, Any] = {"frames": len(frames)}
    if expect["fps"]:  # resampled: the count follows the span, timestamps are the target grid
        span = (src_pts[-1] - src_pts[0]) + 1.0 / (expect["src_fps"] or expect["fps"]) if src_pts else 0.0
        res["frames_equal"] = abs(len(frames) - round(span * expect["fps"])) <= 2
        res["pts_max_err_ms"] = None
        res["pts_ok"] = None not in out_pts and all(
            abs((b - a) - 1.0 / expect["fps"]) < 0.001
            for a, b in zip(out_pts, out_pts[1:])
            if a is not None and b is not None
        )
    else:
        res["frames_equal"] = len(frames) == len(src_pts)
        err = None
        if res["frames_equal"] and src_pts and None not in out_pts:
            first = out_pts[0] or 0.0
            err = max(abs((a - first) - (b - src_pts[0])) for a, b in zip(out_pts, src_pts) if a is not None) * 1000
        res["pts_max_err_ms"] = None if err is None else round(err, 4)
        res["pts_ok"] = err is not None and err <= PTS_TOLERANCE_MS
    res["keyint_ok"] = all(key == (i % keyint == 0) for i, (_, _, key) in enumerate(frames))
    res["no_bframes"] = all(kind != "B" for _, kind, _ in frames)
    res["width"], res["height"], res["duration"] = info.get("width"), info.get("height"), info.get("duration")
    res["dims_ok"] = (res["width"], res["height"]) == tuple(expect["dims"])
    audio = info.get("audio")
    res["has_audio"] = audio is not None
    if src_audio and audio:
        gap = abs((audio["duration"] or 0) - (src_audio["duration"] or 0))
        res["audio_dur_err_s"] = round(gap, 4)
        res["audio_ok"] = (
            gap <= AUDIO_TOLERANCE_S
            and audio["sample_rate"] == (expect["audio_rate"] or src_audio["sample_rate"])
            and audio["channels"] == (expect["audio_channels"] or src_audio["channels"])
        )
    else:
        res["audio_dur_err_s"] = None
        res["audio_ok"] = (src_audio is None) == (audio is None)
    res["ok"] = all(res[k] for k in ("frames_equal", "pts_ok", "keyint_ok", "no_bframes", "dims_ok", "audio_ok"))
    return res


def _cache_key(src_sha256: str, request: Dict[str, Any], ffmpeg_version: str) -> str:
    blob = json.dumps(
        {"src": src_sha256, "request": request, "ffmpeg": ffmpeg_version, "spec": SPEC_VERSION}, sort_keys=True
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        os.link(src, tmp)
    except OSError:
        shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def standardize_video_encoding(
    video: PathLike,
    output_path: Optional[PathLike] = None,
    *,
    profile: str = "archival",
    model: Union[None, str, "ModelInputSpec"] = None,
    resolution: Resolution = None,
    resolution_mode: str = "fit",
    resolution_multiple: int = 2,
    fps: Optional[float] = None,
    fps_method: str = "drop-dup",
    audio_rate: Optional[int] = None,
    audio_channels: Optional[int] = None,
    crf: int = DEFAULT_CRF,
    preset: Optional[str] = None,
    backend: str = "libx264",
    threads: int = DEFAULT_THREADS,
    allow_copy: bool = True,
    cache_dir: Optional[PathLike] = None,
    max_seconds: Optional[float] = None,
    verify: bool = True,
    ffmpeg: Optional[PathLike] = None,
    ffprobe: Optional[PathLike] = None,
) -> StandardizedVideo:
    """Re-encode ``video`` to the standard encoding and return the output path with its provenance.

    **Contract (every profile and target).** H.264 High, ``yuv420p``, no B-frames, scene-cut off, keyframes at
    frame indices ``0, K, 2K, ...`` with ``K = round(fps)`` (fps = the target when set, else the source's average),
    presentation timestamps and frame count of the source kept (``-fps_mode passthrough``) unless ``fps`` is set,
    first audio stream kept, ``+faststart``, all metadata stripped. Rotation metadata is applied (upright
    pixels, no display matrix in the output). Only the first video and first audio streams are kept.

    **Targets** (each ``None`` = keep the source). ``resolution``: an ``int`` (longest side at most that many
    pixels, never upscaled) or ``(W, H)`` with ``resolution_mode`` ``fit | pad | crop | stretch`` (see
    ``target_dims``); dimensions are floored to ``resolution_multiple`` (default 2, which yuv420p needs; use 14/16/28/32
    for patch-grid models). ``fps``: resample to a constant rate with ``fps_method="drop-dup"`` (nearest source frame
    for each output tick, ffmpeg's ``fps`` filter), the GOP becoming ``round(fps)``. ``audio_rate`` /
    ``audio_channels``: resample / remix the audio to AAC. ``model``: a ``ModelInputSpec`` or a registered name;
    it supplies the targets above (explicit arguments win).

    **Profiles.** ``archival`` is ``-preset slow`` (the dataset-wide standard with every target ``None``);
    ``transient`` is ``-preset veryfast`` (``p4`` on NVENC) for on-the-fly preparation. ``preset`` overrides.
    ``backend`` is ``libx264`` (CPU, the reproducible reference), ``h264_nvenc`` (GPU encode, NVDEC decode; ``-cq``
    quality, output differs from x264 at the same CRF) or ``auto`` (NVENC when a test encode works, else libx264).
    The backend used is recorded in the provenance.

    **Fast path.** With ``allow_copy`` and no resolution/fps change, a source that is already H.264 ``yuv420p``,
    upright, B-frame free, even-sized and keyframed exactly every ``K`` frames has its video stream copied
    (``mode="remux"``; ``"audio-only"`` when only the audio needs encoding); the source's own quality and bitrate
    are kept. Otherwise ``mode="reencode"``. Sources that never match (long-GOP camera files) are re-encoded.

    **Cache.** With ``cache_dir`` the output is stored as ``<dir>/<key[:2]>/<key>.mp4`` (+ ``.json``) where
    ``key = sha256(source sha256, resolved parameters, ffmpeg version)``; a repeated request returns it without
    encoding (``provenance["cache"]["hit"]``). Nothing is evicted. ``output_path`` is then a hard link (or copy)
    and is optional (the cache path is returned).

    Args:
        video: The source video file.
        output_path: Where to write the MP4 (parents created, written atomically); required without ``cache_dir``.
        profile: ``archival`` or ``transient``.
        model: A ``ModelInputSpec`` or its registered name (``model_specs.MODEL_INPUT_SPECS``).
        resolution: Target resolution (see above).
        resolution_mode: ``fit``, ``pad``, ``crop`` or ``stretch`` for a ``(W, H)`` target.
        resolution_multiple: Dimensions are floored to a multiple of this.
        fps: Target constant frame rate, or ``None`` to keep the source's timestamps.
        fps_method: Resampling method; ``drop-dup`` is the only one declared.
        audio_rate: Target audio sample rate in Hz, or ``None``.
        audio_channels: Target audio channel count, or ``None``.
        crf: x264 CRF / NVENC ``-cq`` (default 23).
        preset: x264 (or NVENC ``p1``-``p7``) preset; ``None`` takes the profile's.
        backend: ``libx264``, ``h264_nvenc`` or ``auto``.
        threads: x264 threads; fixed so the output is reproducible for a given ffmpeg build (default 4).
        allow_copy: Allow the stream-copy fast path.
        cache_dir: Content-addressed cache directory, or ``None``.
        max_seconds: Encode only this many seconds from the start (sample checks); disables the fast path.
        verify: Decode the output and check frame count, timestamps, GOP, B-frames, geometry and audio.
        ffmpeg: Path to ``ffmpeg`` (default: PATH).
        ffprobe: Path to ``ffprobe`` (default: PATH).

    Returns:
        A ``StandardizedVideo``; ``ok`` is False when a verification check failed (the file is still written).

    Raises:
        ValueError: A target is invalid (unknown mode/profile/backend, odd target dimensions, no output path).
        EmptyVideoError: The source is a 0-byte file.
        VideoStandardizationError: ffmpeg is missing, the source is unreadable or has no video or frames, or the
            encode failed (``code`` says which). No partial output is left behind.
    """
    from .model_specs import ModelInputSpec, get_model_spec

    if profile not in PROFILES or backend not in BACKENDS or fps_method not in FPS_METHODS:
        raise ValueError("unknown profile, backend or fps_method")
    if output_path is None and cache_dir is None:
        raise ValueError("output_path is required without cache_dir")
    if model is not None:
        spec = model if isinstance(model, ModelInputSpec) else get_model_spec(model)
        derived = spec.encode_targets()
        resolution = resolution if resolution is not None else derived.get("resolution")
        if resolution is derived.get("resolution") and derived.get("resolution") is not None:
            resolution_mode = derived.get("resolution_mode", resolution_mode)
        resolution_multiple = (
            derived.get("resolution_multiple", resolution_multiple) if resolution_multiple == 2 else resolution_multiple
        )
        fps = fps if fps is not None else derived.get("fps")
        audio_rate = audio_rate if audio_rate is not None else derived.get("audio_rate")
        audio_channels = audio_channels if audio_channels is not None else derived.get("audio_channels")
    ffmpeg_bin, ffprobe_bin = _tool("ffmpeg", ffmpeg), _tool("ffprobe", ffprobe)
    src = Path(video)
    try:
        src_size = src.stat().st_size
    except OSError:
        raise VideoStandardizationError("unreadable") from None
    if src_size == 0:
        raise EmptyVideoError()
    if backend == "auto":
        backend = "h264_nvenc" if nvenc_available(ffmpeg_bin) else "libx264"
    version = _version(ffmpeg_bin)
    targets = {"resolution": resolution, "resolution_mode": resolution_mode, "resolution_multiple": resolution_multiple,
               "fps": fps, "fps_method": fps_method if fps else None, "audio_rate": audio_rate,
               "audio_channels": audio_channels}  # fmt: skip
    request = {"profile": profile, "preset": preset or PROFILES[profile]["preset"], "crf": crf, "backend": backend,
               "threads": threads, "allow_copy": allow_copy, "max_seconds": max_seconds,
               "targets": targets}  # fmt: skip
    src_hashes: Optional[Dict[str, str]] = None
    cached: Optional[Path] = None
    key = ""
    if cache_dir is not None:
        src_hashes = file_hashes(src)
        key = _cache_key(src_hashes["sha256"], request, version)
        cached = Path(cache_dir) / key[:2] / f"{key}.mp4"
        if cached.exists() and cached.with_suffix(".json").exists():
            hit_prov: Dict[str, Any] = json.loads(cached.with_suffix(".json").read_text())
            hit_prov["cache"] = {"hit": True, "key": key}
            final = Path(output_path) if output_path else cached
            if output_path:
                _link_or_copy(cached, final)
            return StandardizedVideo(output_path=final, provenance=hit_prov)
    dst = cached if cached is not None else Path(str(output_path))
    info = probe_streams(src, ffprobe_bin)
    if info["error"]:
        raise VideoStandardizationError(info["error"])
    frames = frame_list(src, ffprobe_bin)
    if max_seconds and frames and frames[0][0] is not None:
        frames = [f for f in frames if f[0] is None or f[0] < frames[0][0] + max_seconds - 1e-9]
    if not frames:
        raise VideoStandardizationError("no-frames")
    if any(f[0] is None for f in frames):
        raise VideoStandardizationError("no-pts")
    pts = [float(f[0]) for f in frames if f[0] is not None]
    src_fps = native_fps(info, pts)
    keyint = gop_length(fps if fps else src_fps)
    dims = target_dims(
        info["width"], info["height"], info["rotation"], resolution, resolution_mode, resolution_multiple
    )
    audio = info["audio"]
    audio_retarget = bool(audio) and (
        (audio_rate and audio_rate != audio["sample_rate"]) or (audio_channels and audio_channels != audio["channels"])
    )
    audio_copy = bool(audio) and audio["codec"] == "aac" and not audio_retarget
    video_ok = (
        allow_copy and not max_seconds and not fps and dims["identity"] and info["codec"] == "h264"
        and info["pix_fmt"] == "yuv420p" and info["rotation"] == 0
        and all(kind != "B" for _, kind, _ in frames)
        and all(key_ == (i % keyint == 0) for i, (_, _, key_) in enumerate(frames))
    )  # fmt: skip
    mode = ("remux" if audio_copy or not audio else "audio-only") if video_ok else "reencode"
    plan = {
        "mode": mode, "video_copy": video_ok, "backend": backend, "crf": crf, "threads": threads,
        "preset": preset or PROFILES[profile]["preset"], "nvenc_preset": preset or PROFILES[profile]["nvenc_preset"],
        "vf": video_filter(dims), "fps": fps, "audio_copy": audio_copy, "audio_rate": audio_rate,
        "audio_channels": audio_channels, "targets": targets,
    }  # fmt: skip
    dst.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".std-", dir=dst.parent))
    tmp = work / "out.mp4"
    try:
        argv, settings = build_ffmpeg_command(ffmpeg_bin, str(src), str(tmp), info, keyint, plan, max_seconds)
        code, wall, cpu = _run(argv, work / "stderr")
        if code != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            raise VideoStandardizationError("timeout" if code == 124 else "encode-failed")
        prov: Dict[str, Any] = {
            "spec_version": SPEC_VERSION,
            "mode": mode,
            "backend": backend if not video_ok else "copy",
            "profile": profile,
            "frames_decoded": len(frames),
            "avg_frame_rate": info["avg_frame_rate"],
            "r_frame_rate": info["r_frame_rate"],
            "fps": src_fps,
            "vfr": is_variable_frame_rate(pts),
            "gop_frames": keyint,
            "gop_seconds_nominal": keyint / rate if (rate := fps or src_fps) else None,
            "duration": info["duration"],
            "width": info["width"],
            "height": info["height"],
            "codec": info["codec"],
            "pix_fmt": info["pix_fmt"],
            "rotation": info["rotation"],
            "n_audio_streams": info["n_audio"],
            "audio": audio,
            "source": {"size": src_size, **(src_hashes or file_hashes(src))},
            "settings": settings,
            "ffmpeg_version": version,
            "encode_wall_s": round(wall, 3),
            "encode_cpu_s": round(cpu, 3),
            "decode_stderr_bytes": (work / "stderr").stat().st_size,
        }
        expect = {
            "dims": dims["out"],
            "fps": fps,
            "src_fps": src_fps,
            "audio_rate": audio_rate,
            "audio_channels": audio_channels,
        }
        prov["check"] = verify_output(pts, tmp, ffprobe_bin, keyint, audio, expect) if verify else {}
        os.replace(tmp, dst)
        prov["output"] = {"size": dst.stat().st_size, **file_hashes(dst)}
        prov["output"]["size_ratio"] = prov["output"]["size"] / src_size
        final = dst
        if cached is not None:
            prov["cache"] = {"hit": False, "key": key}
            if prov.get("check", {}).get("ok", True):  # only a verified output enters the cache
                cached.with_suffix(".json").write_text(json.dumps(prov, sort_keys=True))
            if output_path:
                _link_or_copy(cached, Path(output_path))
                final = Path(output_path)
        return StandardizedVideo(output_path=final, provenance=prov)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _version(ffmpeg: str) -> str:
    try:
        done = subprocess.run([ffmpeg, "-version"], capture_output=True, check=False, timeout=60)
        return done.stdout.decode("utf-8", "replace").splitlines()[0]
    except (OSError, subprocess.TimeoutExpired, IndexError):
        return "unknown"
