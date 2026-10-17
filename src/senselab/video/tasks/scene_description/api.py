"""Public API: describe the environment visible in a video-call recording."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

from pydantic import BaseModel

from senselab.video.data_structures.video import Video
from senselab.video.tasks.scene_description.aggregation import VideoSceneSummary, aggregate
from senselab.video.tasks.scene_description.backends import VLMBackend, default_backend
from senselab.video.tasks.scene_description.cv_metrics import FrameMetrics, frame_metrics
from senselab.video.tasks.scene_description.describe import FrameDescription, describe_frames
from senselab.video.tasks.scene_description.sampling import FrameReason, iter_sampled_frames


class FrameRecord(BaseModel):
    """Everything measured on one sampled frame.

    Attributes:
        timestamp_s: Presentation time in seconds.
        frame_index: Index in decode order.
        reason: Why the frame was kept (``"interval"`` or ``"scene_cut"``).
        scene_score: Histogram distance to the previous compared frame.
        metrics: Cheap image measurements.
        vlm: The VLM description, or None when the VLM was not run.
    """

    timestamp_s: float
    frame_index: int
    reason: FrameReason
    scene_score: float
    metrics: FrameMetrics
    vlm: Optional[FrameDescription] = None


class VideoSceneReport(BaseModel):
    """Per-frame records and the per-video summary.

    Attributes:
        source: The video's path, or ``"<in-memory>"``.
        model_id: The VLM, or None when it was not run.
        model_revision: The VLM's commit SHA, or None.
        frames: One record per sampled frame.
        summary: The aggregate over frames.
        backend_stats: Resource figures the backend reported (peak memory in bytes).
    """

    source: str
    model_id: Optional[str] = None
    model_revision: Optional[str] = None
    frames: List[FrameRecord]
    summary: VideoSceneSummary
    backend_stats: Dict[str, int] = {}


def describe_video_scene(
    video: Union[str, os.PathLike, Video],
    backend: Optional[VLMBackend] = None,
    use_vlm: bool = True,
    interval_s: float = 60.0,
    scene_threshold: Optional[float] = 0.3,
    max_frames: Optional[int] = None,
    long_side: int = 896,
    batch_size: int = 8,
    detect_faces: bool = True,
) -> VideoSceneReport:
    """Sample frames from a recording, measure them, describe them with a VLM, and summarise.

    Frames are decoded sequentially and processed in batches of ``batch_size``, so memory holds at
    most one batch of downscaled frames.

    Args:
        video: A file path or a ``Video``.
        backend: A loaded VLM backend; defaults to :func:`default_backend` (MLX on Apple silicon,
            transformers elsewhere). Pass the same backend to every call to keep one model resident.
        use_vlm: Run the VLM; when False only the image metrics are computed.
        interval_s: Seconds between interval frames.
        scene_threshold: Histogram distance marking a scene cut; None disables scene-cut frames.
        max_frames: Stop after this many frames.
        long_side: Long side of the frames measured and sent to the VLM.
        batch_size: Frames per VLM request.
        detect_faces: Count faces with OpenCV's Haar cascade.

    Returns:
        The report.
    """
    if use_vlm and backend is None:
        backend = default_backend()
    frames: List[FrameRecord] = []
    pending: List = []

    def flush() -> None:
        descs: Sequence[Optional[FrameDescription]]
        if use_vlm and backend is not None:
            descs = describe_frames([f.image for f in pending], backend, batch_size=batch_size)
        else:
            descs = [None] * len(pending)
        for f, d in zip(pending, descs):
            frames.append(
                FrameRecord(
                    timestamp_s=f.timestamp_s,
                    frame_index=f.frame_index,
                    reason=f.reason,
                    scene_score=f.scene_score,
                    metrics=frame_metrics(f.image, detect_faces=detect_faces),
                    vlm=d,
                )
            )
        pending.clear()

    for sampled in iter_sampled_frames(
        video, interval_s=interval_s, scene_threshold=scene_threshold, max_frames=max_frames, long_side=long_side
    ):
        pending.append(sampled)
        if len(pending) >= batch_size:
            flush()
    if pending:
        flush()

    summary = aggregate(
        [f.timestamp_s for f in frames],
        [f.vlm.description if f.vlm is not None else None for f in frames],
        [f.metrics for f in frames],
    )
    if isinstance(video, Video):
        source = str(video._file_path) if video._file_path is not None else "<in-memory>"
    else:
        source = str(video)
    return VideoSceneReport(
        source=source,
        model_id=backend.model_id if use_vlm and backend is not None else None,
        model_revision=backend.revision if use_vlm and backend is not None else None,
        frames=frames,
        summary=summary,
        backend_stats=backend.stats() if use_vlm and backend is not None else {},
    )


def describe_video_scenes(
    videos: Sequence[Union[str, os.PathLike, Video]],
    backend: Optional[VLMBackend] = None,
    **kwargs: object,
) -> List[VideoSceneReport]:
    """Run :func:`describe_video_scene` over several videos with one resident model.

    Args:
        videos: File paths or ``Video`` objects.
        backend: A loaded VLM backend; defaults to :func:`default_backend`.
        **kwargs: Forwarded to :func:`describe_video_scene`.

    Returns:
        One report per video, in order.
    """
    if kwargs.get("use_vlm", True) and backend is None:
        backend = default_backend()
    return [describe_video_scene(v, backend=backend, **kwargs) for v in videos]  # type: ignore[arg-type]
