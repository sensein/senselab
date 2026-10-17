"""Facial analysis: tile layout, chunked decoding, head gestures, blinks, gaze proxy, AV linking.

Everything here runs in the main environment on synthetic data; the MediaPipe and py-feat workers
are exercised by ``facial_analysis_integration_test.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

from senselab.video.tasks.facial_analysis import (
    FrameTable,
    detect_gallery_tiles,
    detect_nonverbal_events,
    link_tiles_to_speakers,
    load_defaults,
    segments_to_activity,
    summarize_tiles,
)
from senselab.video.tasks.facial_analysis.frame_stream import crop_tile, iter_chunks, iter_frames, probe_video
from senselab.video.tasks.facial_analysis.mediapipe_worker import euler_from_matrix, eye_offsets
from senselab.video.tasks.facial_analysis.signals import detect_blinks, detect_head_gestures, gaze_proxy
from senselab.video.tasks.facial_analysis.tiles import (
    equalize_tiles,
    estimate_frame_layout,
    layout_from_estimates,
    layout_from_frames,
)

av = pytest.importorskip("av")

CANVAS = 26
Rect = Tuple[int, int, int, int]


def _canvas(h: int, w: int, rects: List[Rect], seed: int = 0) -> np.ndarray:
    """A gallery frame: dark canvas with textured tiles."""
    rng = np.random.default_rng(seed)
    frame = np.full((h, w), CANVAS, np.uint8)
    for x, y, tw, th in rects:
        frame[y : y + th, x : x + tw] = rng.integers(60, 220, size=(th, tw), dtype=np.uint8)
    return frame


def _close(found: List[Rect], truth: List[Rect], tol: int = 3) -> bool:
    return len(found) == len(truth) and all(all(abs(a - b) <= tol for a, b in zip(f, t)) for f, t in zip(found, truth))


class TestFrameLayout:
    """One-frame XY-cut estimates over the canonical Zoom gallery arrangements."""

    def test_two_side_by_side_tiles(self) -> None:
        """Two 16:9 tiles centred vertically, as in a 1080p two-person gallery."""
        truth = [(2, 90, 316, 178), (322, 90, 316, 178)]
        assert _close(estimate_frame_layout(_canvas(360, 640, truth)), truth)

    def test_single_tile_with_letterbox(self) -> None:
        """One tile with dark bars above and below."""
        truth = [(0, 40, 640, 280)]
        assert _close(estimate_frame_layout(_canvas(360, 640, truth)), truth)

    def test_two_by_two_grid(self) -> None:
        """Four tiles in reading order."""
        truth = [(4, 4, 312, 172), (324, 4, 312, 172), (4, 184, 312, 172), (324, 184, 312, 172)]
        assert _close(estimate_frame_layout(_canvas(360, 640, truth)), truth)

    def test_three_tiles_two_over_one(self) -> None:
        """Two tiles over one centred tile, which defeats a single column profile."""
        truth = [(4, 4, 312, 172), (324, 4, 312, 172), (164, 184, 312, 172)]
        assert _close(estimate_frame_layout(_canvas(360, 640, truth)), truth)

    def test_slivers_are_not_tiles(self) -> None:
        """A strip narrower than the minimum tile fraction is ignored."""
        truth = [(0, 40, 600, 280)]
        frame = _canvas(360, 640, truth + [(620, 40, 10, 280)])
        assert _close(estimate_frame_layout(frame), truth)

    def test_letterboxed_tile_is_equalized(self) -> None:
        """A tile trimmed by its own black bars grows back to its neighbour's size."""
        rects = equalize_tiles([(2, 90, 316, 178), (322, 98, 316, 162)])
        assert rects[1] == (322, 90, 316, 178)


class TestLayoutTimeline:
    """Folding per-frame estimates into a reference layout and its timeline."""

    two = [(10, 270, 940, 530), (970, 270, 940, 530)]

    def test_reference_masks_and_ids(self) -> None:
        """Modal layout wins; tiles are inset and carry a bottom-left label mask."""
        estimates = [(float(t), list(self.two)) for t in range(10)]
        layout = layout_from_estimates(estimates, 1920, 1080)
        assert [t.tile_id for t in layout.tiles] == ["T0", "T1"]
        tile = layout.tiles[0]
        assert tile.x > 10 and tile.width < 940
        mx, my, mw, mh = tile.masks[0]
        assert mx == 0 and my + mh == tile.height and 0 < mw < tile.width
        assert layout.reference_fraction == 1.0 and layout.layout_changes() == []

    def test_screen_share_and_camera_off_are_classified(self) -> None:
        """A different arrangement is a change; a missing tile is a subset, not a change."""
        share = [(0, 0, 1500, 1080), (1520, 0, 400, 225)]
        estimates = [(0.0, self.two), (10.0, self.two), (20.0, share), (30.0, share), (40.0, self.two[:1])]
        estimates += [(50.0 + t, self.two) for t in range(5)]
        layout = layout_from_estimates(estimates, 1920, 1080)
        kinds = [(s.start_s, s.kind) for s in layout.timeline]
        assert kinds == [(0.0, "reference"), (20.0, "different"), (40.0, "subset"), (50.0, "reference")]
        changes = layout.layout_changes()
        assert len(changes) == 1 and changes[0].end_s == 30.0 and changes[0].n_samples == 2


def _zoom_frame(rects: List[Rect], means: Tuple[int, ...], bars: Tuple[int, int], seed: int) -> np.ndarray:
    """A 1920x1080 gallery frame: black bars outside rows ``bars``, canvas, noisy tiles, codec-like blur."""
    rng = np.random.default_rng(seed)
    frame = np.full((1080, 1920), CANVAS, np.uint8)
    frame[: bars[0]] = 0
    frame[bars[1] :] = 0
    for (x, y, w, h), mean in zip(rects, means):
        frame[y : y + h, x : x + w] = np.clip(rng.normal(mean, 30, (h, w)), 0, 255).astype(np.uint8)
    blurred = frame.astype(np.float32)
    for axis in (0, 1):
        blurred = 0.25 * np.roll(blurred, 1, axis) + 0.5 * blurred + 0.25 * np.roll(blurred, -1, axis)
    return blurred.round().astype(np.uint8)


def _no_faces(frames: object, rect: object) -> List[List[Rect]]:
    return [[] for _ in frames]  # type: ignore[attr-defined]


def _layout(rects: List[Rect], means: Tuple[int, ...], bars: Tuple[int, int], faces: object = _no_faces) -> list:
    """Tiles found by the analysis-width XY-cut plus the full-resolution split, as (x, y, w, h, derivation)."""
    full = [_zoom_frame(rects, means, bars, seed) for seed in range(12)]
    analysis = [(float(i), f.reshape(540, 2, 960, 2).mean(axis=(1, 3)).astype(np.uint8)) for i, f in enumerate(full)]
    layout = layout_from_frames(analysis, full, 1920, 1080, load_defaults()["tiles"], face_detector=faces)  # type: ignore[arg-type]
    return [(t.x, t.y, t.width, t.height, t.derivation, t.confidence) for t in layout.tiles]


def _matches(found: list, truth: List[Rect], derivation: str, tol: int = 12) -> bool:
    """Each found tile lies inside its true tile, inset by at most ``tol`` px per side."""
    return len(found) == len(truth) and all(
        f[4] == derivation
        and f[5] > 0.5
        and t[0] <= f[0] <= t[0] + tol
        and t[1] <= f[1] <= t[1] + tol
        and t[0] + t[2] - tol <= f[0] + f[2] <= t[0] + t[2]
        and t[1] + t[3] - tol <= f[1] + f[3] <= t[1] + t[3]
        for f, t in zip(found, truth)
    )


class TestMergedTiles:
    """Gutters too narrow for the analysis width, recovered at full resolution or by aspect and faces."""

    @pytest.mark.parametrize("gutter", [0, 1, 2, 4])
    @pytest.mark.parametrize("means", [(90, 150), (120, 120)])
    def test_letterboxed_side_by_side_pair(self, gutter: int, means: Tuple[int, int]) -> None:
        """Two 953x526 tiles between letterbox bars, as in the 1080p Zoom pilot."""
        truth = [(7, 277, 953, 526), (960 + gutter, 277, 953, 526)]
        found = _layout(truth, means, (277, 803))
        if gutter == 0 and means[0] == means[1]:
            assert len(found) == 1
        else:
            assert _matches(found, truth, "gutter"), found

    def test_identical_backgrounds_without_gutter_split_on_faces(self) -> None:
        """No seam at all: one face in each half of a 2 x 16:9 region splits it in two."""
        truth = [(7, 277, 953, 526), (960, 277, 953, 526)]

        def faces(frames: list, rect: Rect) -> List[List[Rect]]:
            return [[(380, 420, 150, 150), (1360, 430, 140, 140)] for _ in frames]

        assert _matches(_layout(truth, (120, 120), (277, 803), faces), truth, "aspect_split")

    @pytest.mark.parametrize("gutter", [0, 1, 2])
    def test_two_by_two_grid(self, gutter: int) -> None:
        """Four 958x538 tiles filling the frame."""
        w, h = 958, 538
        truth = [(0, 0, w, h), (w + gutter, 0, w, h), (0, h + gutter, w, h), (w + gutter, h + gutter, w, h)]
        assert _matches(_layout(truth, (80, 140, 110, 170), (0, 1080)), truth, "gutter")

    @pytest.mark.parametrize("gutter", [0, 1, 2])
    def test_one_by_three_row(self, gutter: int) -> None:
        """Three 638x360 tiles in a letterboxed row."""
        truth = [(i * (638 + gutter), 360, 638, 360) for i in range(3)]
        assert _matches(_layout(truth, (80, 140, 110), (360, 720)), truth, "gutter")

    def test_letterboxed_speaker_view_is_not_split(self) -> None:
        """One 16:9 speaker with a centred face stays one tile."""

        def faces(frames: list, rect: Rect) -> List[List[Rect]]:
            return [[(880, 400, 160, 160)] for _ in frames]

        found = _layout([(0, 140, 1440, 810)], (120,), (140, 950), faces)
        assert len(found) == 1 and found[0][4] == "gutter"


def _write_video(path: Path, frames: List[np.ndarray], fps: int) -> None:
    """Encode grayscale or RGB frames with PyAV."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.height, stream.width = frames[0].shape[:2]
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = 4_000_000
        for image in frames:
            rgb = np.repeat(image[..., None], 3, axis=2) if image.ndim == 2 else image
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def gallery_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """4 s at 15 fps of a 2-tile gallery with a screen-share stretch from 2.0 to 3.0 s."""
    path = tmp_path_factory.mktemp("facial") / "gallery.mp4"
    two = [(4, 92, 308, 176), (328, 92, 308, 176)]
    share = [(0, 0, 640, 360)]
    frames = []
    for i in range(60):
        t = i / 15
        frame = _canvas(360, 640, share if 2.0 <= t < 3.0 else two, seed=i)
        if not 2.0 <= t < 3.0:
            frame[92 + 176 - 16 : 92 + 176 - 4, 8:80] = 250  # a name label
        frames.append(frame)
    _write_video(path, frames, 15)
    return path


class TestGalleryVideo:
    """Tile detection and decoding on an encoded synthetic recording."""

    def test_detects_two_tiles_and_the_screen_share(self, gallery_video: Path) -> None:
        """Two tiles are found and the screen-share window is flagged as a layout change."""
        layout = detect_gallery_tiles(gallery_video, sample_frames=16)
        assert len(layout.tiles) == 2
        t0, t1 = layout.tiles
        assert abs(t0.x - 4) <= 8 and abs(t1.x - 328) <= 8 and abs(t0.width - 308) <= 12
        changes = layout.layout_changes()
        assert changes and all(2.0 <= c.start_s <= 3.0 for c in changes)
        assert 0.6 <= layout.reference_fraction < 1.0

    def test_pilot_format_pair_is_split_in_an_encoded_file(self, tmp_path: Path) -> None:
        """1920x1080 letterboxed pair with a 1 px gutter, through the codec and the file path."""
        truth = [(7, 277, 953, 526), (961, 277, 953, 526)]
        path = tmp_path / "pilot.mp4"
        _write_video(path, [_zoom_frame(truth, (90, 150), (277, 803), seed) for seed in range(16)], 16)
        layout = detect_gallery_tiles(path, sample_frames=8)
        found = [(t.x, t.y, t.width, t.height, t.derivation, t.confidence) for t in layout.tiles]
        assert _matches(found, truth, "gutter", tol=14), found
        assert layout.reference_fraction == 1.0

    def test_probe_and_resampled_decode(self, gallery_video: Path) -> None:
        """A 15 fps stream sampled at 5 fps yields every third frame, evenly spaced."""
        info = probe_video(str(gallery_video))
        assert (info["width"], info["height"]) == (640, 360)
        times = [t for t, _ in iter_frames(str(gallery_video), fps=5.0, scale=0.5)]
        assert len(times) == 20
        assert np.allclose(np.diff(times), 0.2, atol=1e-3)

    def test_chunks_are_bounded_and_complete(self, gallery_video: Path) -> None:
        """Chunks never exceed the chunk size and together cover every sampled frame once."""
        chunks = list(iter_chunks(str(gallery_video), chunk_size=7, fps=10.0, scale=0.5))
        sizes = [len(times) for times, _ in chunks]
        assert max(sizes) <= 7 and sum(sizes) == 40
        times = np.concatenate([t for t, _ in chunks])
        assert np.all(np.diff(times) > 0)
        assert chunks[0][1].shape[1:] == (180, 320, 3)

    def test_crop_tile_masks_the_label(self) -> None:
        """The masked strip takes the tile's mean value and the copy leaves the frame untouched."""
        frame = np.zeros((100, 200, 3), np.uint8)
        frame[10:60, 20:120] = 100
        frame[50:60, 20:50] = 255
        tile = crop_tile(frame, (20, 10, 100, 50), scale=1.0, masks=[(0, 40, 30, 10)])
        assert tile.shape == (50, 100, 3)
        assert tile[45, 5, 0] < 255 and frame[55, 25, 0] == 255


def _oscillation(
    fps: float, seconds: float, freq: float, amp: float, start: float, stop: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1 / fps)
    x = rng.normal(0, 0.3, t.size)
    on = (t >= start) & (t < stop)
    x[on] += amp * np.sin(2 * np.pi * freq * (t[on] - start))
    return x


class TestHeadGestures:
    """Nods and shakes from synthetic pitch/yaw series."""

    fps = 16.0

    def test_a_nod_is_found_where_pitch_oscillates(self) -> None:
        """Two seconds of 2 Hz, 5 degree pitch oscillation is one nod, and nothing else."""
        t = np.arange(0, 10, 1 / self.fps)
        pitch = _oscillation(self.fps, 10, 2.0, 5.0, 4.0, 6.0, seed=1)
        yaw = _oscillation(self.fps, 10, 2.0, 0.0, 0, 0, seed=2)
        events = detect_head_gestures(t, pitch, yaw, self.fps)
        assert [e.kind for e in events] == ["nod"]
        nod = events[0]
        assert 3.5 <= nod.start_s and nod.end_s <= 6.5 and nod.n_extrema >= 5
        assert 1.5 <= nod.frequency_hz <= 2.5 and nod.amplitude_deg >= 3.0

    def test_a_shake_is_found_where_yaw_oscillates(self) -> None:
        """Yaw oscillation is a shake, not a nod."""
        t = np.arange(0, 10, 1 / self.fps)
        yaw = _oscillation(self.fps, 10, 1.5, 8.0, 2.0, 4.0, seed=3)
        pitch = _oscillation(self.fps, 10, 1.0, 0.0, 0, 0, seed=4)
        events = detect_head_gestures(t, pitch, yaw, self.fps)
        assert [e.kind for e in events] == ["shake"]

    def test_noise_and_slow_drift_are_not_gestures(self) -> None:
        """Sub-threshold jitter and a 0.1 Hz head turn produce no events."""
        t = np.arange(0, 20, 1 / self.fps)
        pitch = _oscillation(self.fps, 20, 0.1, 15.0, 0, 20, seed=5)
        yaw = _oscillation(self.fps, 20, 0.1, 0.0, 0, 0, seed=6)
        assert detect_head_gestures(t, pitch, yaw, self.fps) == []

    def test_short_face_dropouts_are_bridged(self) -> None:
        """A 2-sample gap inside a nod does not split or lose it."""
        t = np.arange(0, 10, 1 / self.fps)
        pitch = _oscillation(self.fps, 10, 2.0, 5.0, 4.0, 6.0, seed=7)
        pitch[75:77] = np.nan
        yaw = np.zeros_like(pitch)
        assert [e.kind for e in detect_head_gestures(t, pitch, yaw, self.fps)] == ["nod"]

    def test_eight_fps_still_resolves_a_two_hertz_nod(self) -> None:
        """At the default geometry rate the band edge is clipped below Nyquist and nods survive."""
        fps = 8.0
        t = np.arange(0, 10, 1 / fps)
        pitch = _oscillation(fps, 10, 1.5, 6.0, 3.0, 6.0, seed=8)
        events = detect_head_gestures(t, pitch, np.zeros_like(pitch), fps)
        assert [e.kind for e in events] == ["nod"]


class TestBlinks:
    """Blink detection against a per-face baseline."""

    def test_blinks_above_a_high_resting_level(self) -> None:
        """Three 0.2 s spikes over a 0.37 baseline are blinks; a 1 s closure is not."""
        fps = 16.0
        t = np.arange(0, 12, 1 / fps)
        level = np.full(t.size, 0.37)
        for start in (2.0, 5.0, 8.0):
            level[(t >= start) & (t < start + 0.2)] = 0.85
        level[(t >= 10.0) & (t < 11.0)] = 0.9
        events = detect_blinks(t, level, level)
        assert [round(e.start_s) for e in events] == [2, 5, 8]

    def test_no_face_is_no_blink(self) -> None:
        """NaN samples never start an event."""
        t = np.arange(0, 2, 1 / 16)
        assert detect_blinks(t, np.full(t.size, np.nan), np.full(t.size, np.nan)) == []


class TestGazeProxy:
    """Head plus eye angles."""

    def test_frontal_face_scores_one(self) -> None:
        """Zero head and eye angles at the neutral vertical offset point at the camera."""
        _, _, score = gaze_proxy(np.zeros(1), np.zeros(1), np.zeros(1), np.full(1, -0.1), eye_vertical_neutral=-0.1)
        assert score[0] == pytest.approx(1.0)

    def test_eyes_can_compensate_for_a_turned_head(self) -> None:
        """Head turned to the subject's right with irises toward image right still faces the camera."""
        yaw, _, score = gaze_proxy(np.zeros(1), np.array([15.0]), np.array([0.25]), np.zeros(1), eye_gain_deg=60.0)
        assert yaw[0] == pytest.approx(0.0) and score[0] == pytest.approx(1.0)


class TestWorkerGeometry:
    """Pure helpers of the MediaPipe worker."""

    @staticmethod
    def _rotation(axis: str, degrees: float) -> np.ndarray:
        a = np.radians(degrees)
        c, s = np.cos(a), np.sin(a)
        r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]]}[axis]
        m = np.eye(4)
        m[:3, :3] = r
        return m

    def test_yaw_toward_image_left_is_positive(self) -> None:
        """Rotating the face's forward axis toward camera -x (image left) is positive yaw."""
        pitch, yaw, roll = euler_from_matrix(self._rotation("y", -20.0))
        assert yaw == pytest.approx(20.0, abs=1e-4) and abs(pitch) < 1e-4 and abs(roll) < 1e-4

    def test_face_tilted_up_is_positive_pitch(self) -> None:
        """Rotating the forward axis toward camera +y (up) is positive pitch."""
        pitch, _, _ = euler_from_matrix(self._rotation("x", -10.0))
        assert pitch == pytest.approx(10.0, abs=1e-4)

    def test_centred_iris_has_zero_horizontal_offset(self) -> None:
        """An iris midway between the corners and on their line reads (0, 0)."""
        points = np.zeros((478, 3), np.float32)
        for left, right, iris in ((33, 133, 468), (362, 263, 473)):
            points[left, :2] = (0.2, 0.5)
            points[right, :2] = (0.4, 0.5)
            points[iris, :2] = (0.3, 0.5)
        assert np.allclose(eye_offsets(points), 0.0, atol=1e-6)


def _turns(fps: float, seconds: float, order: List[Tuple[str, float, float]]) -> Tuple[np.ndarray, dict]:
    t = np.arange(0, seconds, 1 / fps)
    activity: dict = {}
    for speaker, start, end in order:
        activity.setdefault(speaker, []).append((start, end))
    return t, {s: segments_to_activity(segs, t) for s, segs in activity.items()}


class TestSpeakerLinking:
    """Mouth-motion to diarization assignment."""

    fps = 8.0

    def _mouth(self, t: np.ndarray, active: np.ndarray, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        talking = 0.25 + 0.2 * np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 6)) * rng.uniform(0.5, 1.0, t.size)
        resting = 0.05 + rng.normal(0, 0.01, t.size)
        return np.where(active > 0, talking, resting)

    def test_assigns_each_tile_to_its_speaker(self) -> None:
        """Alternating turns; T0 moves with SPEAKER_01, T1 with SPEAKER_00."""
        turns = [("SPEAKER_00" if i % 2 else "SPEAKER_01", 5.0 * i, 5.0 * i + 4.0) for i in range(12)]
        t, activity = _turns(self.fps, 60, turns)
        mouths = {"T0": self._mouth(t, activity["SPEAKER_01"], 1), "T1": self._mouth(t, activity["SPEAKER_00"], 2)}
        mouths["T1"][100:140] = np.nan  # the face leaves the frame for a while
        link = link_tiles_to_speakers(mouths, activity, self.fps)
        assert link.assignment == {"T0": "SPEAKER_01", "T1": "SPEAKER_00"}
        assert link.total_margin > 0.5 and all(m > 0.2 for m in link.margins.values())
        assert link.p_value < 0.05
        assert link.correlations["T0"]["SPEAKER_01"] > link.correlations["T0"]["SPEAKER_00"]

    def test_extra_speaker_stays_unassigned(self) -> None:
        """Three diarized speakers, two tiles: the off-screen speaker is left out."""
        turns = [(f"S{i % 3}", 4.0 * i, 4.0 * i + 3.0) for i in range(15)]
        t, activity = _turns(self.fps, 60, turns)
        mouths = {"T0": self._mouth(t, activity["S2"], 3), "T1": self._mouth(t, activity["S0"], 4)}
        link = link_tiles_to_speakers(mouths, activity, self.fps)
        assert link.assignment == {"T0": "S2", "T1": "S0"}

    def test_unrelated_motion_is_not_significant(self) -> None:
        """Mouth motion independent of both speakers: the margin can look large, the shift null is not passed.

        With two strictly alternating speakers the activity tracks are complements, so any chance
        correlation counts twice in the margin; the circular-shift p-value is the confidence.
        """
        turns = [("A" if i % 2 else "B", 5.0 * i, 5.0 * i + 4.0) for i in range(12)]
        t, activity = _turns(self.fps, 60, turns)
        rng = np.random.default_rng(9)
        mouths = {"T0": rng.normal(0.1, 0.05, t.size), "T1": rng.normal(0.1, 0.05, t.size)}
        assert link_tiles_to_speakers(mouths, activity, self.fps).p_value > 0.05

    def test_surrogate_controls_pass_through(self) -> None:
        """``n_shifts=0`` skips the null (p_value 1.0); the signals-module defaults are unchanged."""
        turns = [("A" if i % 2 else "B", 5.0 * i, 5.0 * i + 4.0) for i in range(12)]
        t, activity = _turns(self.fps, 60, turns)
        mouths = {"T0": self._mouth(t, activity["B"], 1), "T1": self._mouth(t, activity["A"], 2)}
        default = link_tiles_to_speakers(mouths, activity, self.fps)
        explicit = link_tiles_to_speakers(mouths, activity, self.fps, n_shifts=200, min_shift_s=10.0)
        assert explicit.p_value == default.p_value
        assert link_tiles_to_speakers(mouths, activity, self.fps, n_shifts=0).p_value == 1.0
        assert link_tiles_to_speakers(mouths, activity, self.fps, n_shifts=10).p_value >= 1 / 11
        assert link_tiles_to_speakers(mouths, activity, self.fps, min_shift_s=1000.0).p_value == 1.0


class TestConfigAndSummaries:
    """Defaults, overrides, and summaries over a synthetic geometry table."""

    def test_overrides_deep_merge(self) -> None:
        """A partial override changes one key and keeps its siblings."""
        merged = load_defaults({"geometry": {"fps": 16.0}})
        assert merged["geometry"]["fps"] == 16.0
        assert merged["geometry"]["tile_width"] == load_defaults()["geometry"]["tile_width"]

    def test_events_and_summary_from_a_geometry_table(self) -> None:
        """A nod, two blinks and half-time face presence come out of the summary as rates."""
        fps = 16.0
        t = np.arange(0, 60, 1 / fps)
        pitch = _oscillation(fps, 60, 2.0, 6.0, 10.0, 12.0, seed=11)
        blink = np.full(t.size, 0.1)
        for start in (20.0, 40.0):
            blink[(t >= start) & (t < start + 0.2)] = 0.9
        present = t < 30.0
        frame = pd.DataFrame(
            {
                "time_s": t,
                "face_present": present,
                "head_pitch_deg": np.where(present, pitch, np.nan),
                "head_yaw_deg": np.where(present, 0.0, np.nan),
                "bs_eyeBlinkLeft": np.where(present, blink, np.nan),
                "bs_eyeBlinkRight": np.where(present, blink, np.nan),
                "bs_jawOpen": np.where(present, 0.2, np.nan),
                "toward_camera": np.where(present, 0.9, np.nan),
                "motion_energy": np.full(t.size, 0.01),
            }
        )
        layout = layout_from_estimates([], 640, 360)
        table = FrameTable(kind="geometry", fps=fps, layout=layout, tiles={"T0": frame})
        events = detect_nonverbal_events(table)
        assert len(events["T0"].nods) == 1 and len(events["T0"].blinks) == 1
        (summary,) = summarize_tiles(table, None, events)
        assert summary.face_present_fraction == pytest.approx(0.5)
        assert summary.blink_rate_per_min == pytest.approx(2.0)
        assert summary.toward_camera_fraction == 1.0
        long = table.to_long()
        assert set(long["tile_id"]) == {"T0"} and len(long) == t.size

    def test_from_long_inverts_to_long(self) -> None:
        """to_long then from_long returns the original tiles, columns and dtypes."""
        t = np.arange(5) / 2.0
        tiles = {
            "T1": pd.DataFrame({"time_s": t, "face_present": t > 0.5, "bs_jawOpen": t * 0.1}),
            "T0": pd.DataFrame({"time_s": t, "face_present": t > 1.0, "bs_jawOpen": t * 0.2}),
        }
        layout = layout_from_estimates([], 640, 360)
        table = FrameTable(kind="geometry", fps=2.0, layout=layout, tiles=tiles)
        back = FrameTable.from_long(table.to_long(), kind="geometry", fps=2.0, layout=layout)
        assert list(back.tiles) == ["T1", "T0"]
        for tile_id, frame in tiles.items():
            pd.testing.assert_frame_equal(back.tiles[tile_id], frame)
