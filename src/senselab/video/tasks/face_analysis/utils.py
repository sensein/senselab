"""This module provides utilities for face analysis."""

from typing import Generator, Optional, Tuple

import torch

from senselab.video.data_structures.video import Video


def get_sampled_frames(
    video: Video, frame_sample_rate: Optional[float] = None
) -> Generator[Tuple[int, torch.Tensor], None, None]:
    """Lazily sample frames from a Video at a given frame rate.

    Args:
        video (Video): The video object to sample from.
        frame_sample_rate (Optional[float]): The desired number of frames per second to sample.
            If None or greater than the video's native frame rate, returns all frames.

    Yields:
        Tuple[int, torch.Tensor]: A tuple containing the original frame index and the frame tensor.
    """
    native_fps = video.frame_rate
    total_frames = len(video.frames)

    if frame_sample_rate is None or frame_sample_rate >= native_fps:
        step = 1
    else:
        if frame_sample_rate <= 0:
            raise ValueError("frame_sample_rate must be positive.")
        step = max(int(native_fps // frame_sample_rate), 1)

    for ix in range(0, total_frames, step):
        yield ix, video.frames[ix]
