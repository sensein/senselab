"""Audio-visual target speaker extraction: AV_MossFormer2_TSE_16K.

Takes a **video file** and returns one extracted 16 kHz speech signal per face track the pipeline
detects, using the speaker's lip motion as the conditioning cue. Runs in an isolated subprocess venv;
see :mod:`senselab.utils.clearvoice` for the venv, the pin and the device contract.

The visual chain (scene detection, S3FD face detection, tracking, cropping) arrives with the
``clearvoice`` distribution, so senselab's ``[video]`` extra is not involved. ffmpeg must be on PATH.
The face detector's weights have no revision-addressable home upstream and are pinned by sha256:
design.md D-7. This capability's numerical output is **not verified** in this repository: design.md
D-3.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Sequence, Union

from senselab.audio.data_structures import Audio
from senselab.utils.clearvoice import clearvoice_model_spec, run_clearvoice_tse
from senselab.utils.data_structures import DeviceType, HFModel
from senselab.utils.data_structures.logging import logger

CLEARVOICE_TSE_TASK = "target_speaker_extraction"

# Containers upstream's reader accepts (dataloader/misc.py:read_and_config_file). Anything else is
# rejected here, because upstream would fall through to treating the path as a list file and fail
# while reading it as text.
SUPPORTED_VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".webm")

VideoInput = Union[str, Path, "object"]


def _video_path(video: VideoInput) -> Path:
    """Return the file path for one input, requiring that it is file-backed.

    Args:
        video: A path, or a ``Video`` carrying one.

    Returns:
        The path.

    Raises:
        ValueError: If a ``Video`` has no file path, or the container is not one upstream reads.
        FileNotFoundError: If the path does not exist.
    """
    from senselab.video.data_structures import Video

    if isinstance(video, Video):
        path = getattr(video, "_file_path", None)
        if not path:
            raise ValueError(
                "audio-visual target speaker extraction needs the video file, not decoded frames: "
                "the pipeline re-encodes the container to 25 fps and extracts its audio track with "
                "ffmpeg. Construct the Video with filepath=..., or write the frames out first."
            )
        resolved = Path(path)
    else:
        resolved = Path(video)

    if not resolved.is_file():
        raise FileNotFoundError(f"video not found: {resolved}")
    if resolved.suffix.lower() not in SUPPORTED_VIDEO_SUFFIXES:
        raise ValueError(
            f"{resolved.suffix!r} is not a container ClearVoice's reader accepts; it takes "
            f"{', '.join(SUPPORTED_VIDEO_SUFFIXES)}. Remux first."
        )
    return resolved


def video_duration_s(path: Path) -> Optional[float]:
    """Return a video's duration in seconds from container metadata, or ``None`` if unreadable.

    Metadata only: decoding the frames to count them would cost more than the timeout it informs.

    Args:
        path: The video file.

    Returns:
        Duration in seconds, or ``None``, in which case the caller falls back to the timeout floor.
    """
    try:
        import av

        with av.open(str(path)) as container:
            if container.duration is not None:
                return float(container.duration) / av.time_base
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is not None and stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except Exception as exc:  # noqa: BLE001 -- an unreadable duration is not a reason to refuse
        logger.warning(f"could not read the duration of {path}: {exc}; using the timeout floor")
    return None


def extract_target_speakers_with_clearvoice(
    videos: Sequence[VideoInput],
    model: HFModel,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
) -> List[List[Audio]]:
    """Extract each detected speaker's voice from every video, conditioned on their lip motion.

    Args:
        videos: Video files, as paths or file-backed ``Video`` objects.
        model: ``HFModel`` naming ``alibabasglab/AV_MossFormer2_TSE_16K``.
        device: CUDA or CPU. ``None`` leaves the choice to the worker. MPS is not offered.
        timeout_s: Ceiling on the worker, in seconds. ``None`` derives one from the total video
            duration; that term is coarse and unmeasured, so a long video may need this raised.

    Returns:
        One list per input video, holding one 16 kHz ``Audio`` per detected face track in track order.
        An empty list means the pipeline found no track long enough to extract — which is an outcome,
        not an error. Each ``Audio`` carries a ``metadata["clearvoice"]`` record.

    Raises:
        ValueError: If ``model`` does not name the ClearVoice extraction checkpoint, if a video is not
            file-backed or is in an unsupported container, or if ``timeout_s`` is not positive.
        FileNotFoundError: If a video path does not exist.
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    spec = clearvoice_model_spec(str(model.path_or_uri), expected_task=CLEARVOICE_TSE_TASK)
    if not videos:
        return []

    paths = [_video_path(video) for video in videos]
    durations = [video_duration_s(path) for path in paths]
    total_video_s = sum(duration for duration in durations if duration is not None)

    with tempfile.TemporaryDirectory(prefix="senselab-clearvoice-tse-out-") as out_dir:
        wav_paths, sha = run_clearvoice_tse(
            spec,
            [str(path) for path in paths],
            out_dir,
            total_video_s=total_video_s,
            device=device,
            timeout_s=timeout_s,
            revision=model.commit_sha or model.revision,
        )

        results: List[List[Audio]] = []
        for path, tracks in zip(paths, wav_paths):
            extracted = []
            for track_index, wav_path in enumerate(tracks):
                audio = Audio(filepath=wav_path)
                # Force the lazy load before the temporary directory holding the file is removed.
                _ = audio.waveform
                audio.metadata = {
                    "clearvoice": {
                        "model": spec.model_id,
                        "commit": sha,
                        "capability": spec.capability,
                        "sampling_rate": spec.sampling_rate,
                        "source_index": track_index,
                        "n_sources": len(tracks),
                        "face_track_index": track_index,
                    },
                    "source_video": str(path),
                }
                extracted.append(audio)
            if not extracted:
                logger.warning(f"no face track was extracted from {path}; returning no speakers for it")
            results.append(extracted)
    return results
