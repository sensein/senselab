"""Frame sampling on a synthetic video with known scene cuts."""

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from senselab.video.data_structures.video import Video  # noqa: E402
from senselab.video.tasks.scene_description.sampling import histogram_distance, sample_frames  # noqa: E402

FPS = 10
SEGMENT_S = 10
COLOURS = [(200, 40, 40), (40, 160, 220), (30, 30, 30)]
CUTS_S = [10.0, 20.0]


def _segment_frame(colour: tuple, rng: np.random.Generator, width: int = 320, height: int = 240) -> np.ndarray:
    base = np.empty((height, width, 3), dtype=np.int16)
    base[:] = colour
    base += rng.integers(-6, 7, size=base.shape, dtype=np.int16)
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture(scope="module")
def cut_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three 10 s solid-colour segments with sensor-like noise, encoded with FFmpeg's libx264."""
    path = tmp_path_factory.mktemp("scene") / "cuts.mp4"
    rng = np.random.default_rng(0)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = 320, 240, "yuv420p"
        for colour in COLOURS:
            for _ in range(FPS * SEGMENT_S):
                frame = av.VideoFrame.from_ndarray(_segment_frame(colour, rng), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_scene_cuts_found_between_intervals(cut_video: Path) -> None:
    """With a 60 s interval only frame 0 is an interval frame; both cuts are found."""
    frames = sample_frames(cut_video, interval_s=60, scene_threshold=0.3)
    assert [f.reason for f in frames] == ["interval", "scene_cut", "scene_cut"]
    assert frames[0].timestamp_s == pytest.approx(0.0)
    for frame, cut in zip(frames[1:], CUTS_S):
        assert cut <= frame.timestamp_s <= cut + 0.5 + 1e-6
        assert frame.scene_score > 0.3


def test_interval_frames_and_cut_coinciding_with_interval(cut_video: Path) -> None:
    """A 5 s interval puts boundaries on the cuts; the cut frame is then labelled an interval frame."""
    frames = sample_frames(cut_video, interval_s=5, scene_threshold=0.3)
    times = [round(f.timestamp_s, 1) for f in frames]
    assert times == [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    assert all(f.reason == "interval" for f in frames)


def test_scene_detection_disabled(cut_video: Path) -> None:
    """``scene_threshold=None`` keeps only interval frames."""
    frames = sample_frames(cut_video, interval_s=60, scene_threshold=None)
    assert len(frames) == 1


def test_max_frames_and_downscale(cut_video: Path) -> None:
    """``max_frames`` stops early and ``long_side`` bounds the kept image."""
    frames = sample_frames(cut_video, interval_s=1, max_frames=4, long_side=160)
    assert len(frames) == 4
    assert frames[0].image.shape == (120, 160, 3)
    assert frames[0].source_size == (320, 240)
    assert frames[0].image.dtype == np.uint8


def test_colour_preserved(cut_video: Path) -> None:
    """The kept frame decodes to the segment's colour (RGB order)."""
    frames = sample_frames(cut_video, interval_s=60)
    for frame, colour in zip(frames, COLOURS):
        mean = frame.image.reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean, colour, atol=8)


def test_video_object_with_preloaded_frames() -> None:
    """An in-memory ``Video`` is sampled from its tensor."""
    rng = np.random.default_rng(1)
    arrays = [_segment_frame(COLOURS[i // 20], rng, 64, 48) for i in range(40)]
    video = Video(frames=arrays, frame_rate=10.0)
    frames = sample_frames(video, interval_s=60, scene_threshold=0.3)
    assert [f.reason for f in frames] == ["interval", "scene_cut"]
    assert frames[1].timestamp_s == pytest.approx(2.0)


def test_histogram_distance_bounds() -> None:
    """Identical images score 0, disjoint colour distributions score 1."""
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert histogram_distance(black, black) == 0.0
    assert histogram_distance(black, white) == pytest.approx(1.0)


def test_invalid_interval(cut_video: Path) -> None:
    """A non-positive interval is rejected."""
    with pytest.raises(ValueError):
        sample_frames(cut_video, interval_s=0)
