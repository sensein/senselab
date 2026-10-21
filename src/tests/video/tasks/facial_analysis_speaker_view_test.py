"""Facial analysis: face-position splits of merged tiles and active-speaker person tracks.

Face detections and identity embeddings are injected, so everything here runs on synthetic data in
the main environment; ``facial_analysis_integration_test.py`` runs YuNet and SFace on real faces.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from senselab.video.tasks.facial_analysis import (
    FrameTable,
    GalleryLayout,
    extract_face_geometry,
    link_tiles_to_speakers,
    load_defaults,
    segments_to_activity,
)
from senselab.video.tasks.facial_analysis import api as facial_api
from senselab.video.tasks.facial_analysis.backends import _tile_payload
from senselab.video.tasks.facial_analysis.frame_stream import tile_placement, widest_box
from senselab.video.tasks.facial_analysis.identity import Face, cluster_identities
from senselab.video.tasks.facial_analysis.pyfeat_worker import pad_stack
from senselab.video.tasks.facial_analysis.speaker_view import (
    IdentitySample,
    is_speaker_view_candidate,
    speaker_view_layout,
)
from senselab.video.tasks.facial_analysis.tiles import face_position_split, layout_from_estimates, layout_from_frames

Rect = Tuple[int, int, int, int]
CANVAS = 26
BAND = (104, 277, 1712, 526)  # the observed 1712x526 band: two 856x526 tiles, about 1.63:1 each


def _frame(rects: Sequence[Rect], means: Sequence[int], seed: int) -> np.ndarray:
    """A 1920x1080 frame: canvas, letterbox bars above and below the band, noisy tiles, codec-like blur."""
    rng = np.random.default_rng(seed)
    frame = np.full((1080, 1920), CANVAS, np.uint8)
    frame[: BAND[1]] = 0
    frame[BAND[1] + BAND[3] :] = 0
    for (x, y, w, h), mean in zip(rects, means):
        frame[y : y + h, x : x + w] = np.clip(rng.normal(mean, 30, (h, w)), 0, 255).astype(np.uint8)
    blurred = frame.astype(np.float32)
    for axis in (0, 1):
        blurred = 0.25 * np.roll(blurred, 1, axis) + 0.5 * blurred + 0.25 * np.roll(blurred, -1, axis)
    return blurred.round().astype(np.uint8)


def _tiles(rects: Sequence[Rect], means: Sequence[int], faces_per_frame: Sequence[List[Rect]]) -> list:
    """Tiles from the analysis-width XY-cut plus the full-resolution split, with injected faces."""
    full = [_frame(rects, means, seed) for seed in range(len(faces_per_frame))]
    analysis = [(float(i), f.reshape(540, 2, 960, 2).mean(axis=(1, 3)).astype(np.uint8)) for i, f in enumerate(full)]

    def faces(frames: Sequence[np.ndarray], rect: Rect) -> List[List[Rect]]:
        return [list(boxes) for boxes in faces_per_frame]

    layout = layout_from_frames(analysis, full, 1920, 1080, load_defaults()["tiles"], face_detector=faces)
    return [(t.x, t.y, t.width, t.height, t.derivation) for t in layout.tiles]


def _two_faces(n: int, left_cx: float = 532, right_cx: float = 1388, jitter: float = 12.0) -> List[List[Rect]]:
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        a, b = left_cx + rng.normal(0, jitter), right_cx + rng.normal(0, jitter)
        out.append([(int(a - 70), 420, 140, 150), (int(b - 65), 430, 130, 140)])
    return out


class TestFacePositionSplit:
    """Merged regions cut between face clusters regardless of their aspect ratio."""

    TRUTH = [(104, 277, 856, 526), (960, 277, 856, 526)]

    def test_band_without_gutter_splits_between_faces(self) -> None:
        """1712x526 band, two 856-wide tiles, identical backgrounds, no gutter: cut at the face midpoint."""
        found = _tiles([BAND], [120], _two_faces(12))
        assert [f[4] for f in found] == ["face_split", "face_split"], found
        cut = found[0][0] + found[0][2]
        assert abs(cut - 960) <= 20 and found[1][0] >= cut
        assert found[0][0] >= 104 and found[1][0] + found[1][2] <= 104 + 1712

    def test_seam_inside_the_gap_wins_over_the_midpoint(self) -> None:
        """Distinct tile backgrounds and off-centre faces: the cut lands on the seam, not the face midpoint."""
        faces = _two_faces(12, left_cx=420, right_cx=1300)
        found = _tiles(self.TRUTH, [90, 150], faces)
        assert [f[4] for f in found] == ["face_split", "face_split"], found
        assert abs(found[0][0] + found[0][2] - 960) <= 12 and abs(found[1][0] - 960) <= 12

    def test_one_person_moving_across_the_band_is_not_split(self) -> None:
        """A single face drifting across the band forms one cluster."""
        faces = [[(300 + 100 * i, 420, 140, 150)] for i in range(12)]
        assert len(_tiles([BAND], [120], faces)) == 1

    def test_two_people_sharing_one_camera_are_not_split(self) -> None:
        """Two faces in one 16:9 tile: the halves would be 0.89:1, outside the tile-aspect range."""
        rect = (240, 140, 1440, 810)
        faces = [[(600, 400, 150, 150), (1170, 410, 150, 150)] for _ in range(12)]
        assert face_position_split(faces, rect, load_defaults()["tiles"]) is None

    def test_rarely_present_second_face_is_not_split(self) -> None:
        """The right-hand cluster is present in 4 of 12 frames, under the 50 % floor."""
        faces = _two_faces(12)
        faces = [boxes if i < 4 else boxes[:1] for i, boxes in enumerate(faces)]
        assert len(_tiles([BAND], [120], faces)) == 1

    def test_straddling_faces_block_the_split(self) -> None:
        """Faces sitting on the cut in a third of the frames exceed the 20 % straddle limit."""
        faces = _two_faces(12)
        tiles_config = load_defaults()["tiles"]
        for i in range(6):
            faces[i].append((900, 420, 140, 150))
        assert face_position_split(faces, BAND, tiles_config) is None

    def test_vertical_band_splits_along_y(self) -> None:
        """Two 4:3 tiles stacked in one region split between the upper and lower face."""
        rect = (600, 40, 640, 960)
        faces = [[(850, 205, 140, 150), (860, 690, 130, 140)] for _ in range(12)]
        cells = face_position_split(faces, rect, load_defaults()["tiles"])
        assert cells is not None and [c[0][3] for c in cells] == [480, 480]


def _unit(seed: int, dim: int = 128) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim)
    return v / np.linalg.norm(v)


def _embed(identity: np.ndarray, rng: np.random.Generator, noise: float = 0.05) -> np.ndarray:
    v = identity + rng.normal(0, noise, identity.size)
    return v / np.linalg.norm(v)


class TestClusterIdentities:
    """Agglomerative cosine clustering of identity embeddings."""

    def test_two_people_and_a_stray(self) -> None:
        """Two identities in order of appearance; a one-off embedding is dropped as spurious."""
        rng = np.random.default_rng(1)
        a, b = _unit(1), _unit(2)
        vectors = [_embed(b, rng) for _ in range(20)] + [_embed(a, rng) for _ in range(20)] + [_unit(3)]
        labels = cluster_identities(vectors, 0.363, min_fraction=0.05)
        assert (labels[:20] == 0).all() and (labels[20:40] == 1).all() and labels[40] == -1

    def test_subsampled_clustering_assigns_the_rest(self) -> None:
        """With more embeddings than ``max_samples``, the remainder joins the nearest centroid."""
        rng = np.random.default_rng(2)
        a, b = _unit(4), _unit(5)
        vectors = [_embed(a if i % 3 else b, rng) for i in range(300)]
        labels = cluster_identities(vectors, 0.363, max_samples=40)
        truth = np.array([0 if i % 3 else 1 for i in range(300)])
        first = labels[0]
        assert set(labels.tolist()) == {0, 1}
        assert ((labels == first) == (truth == truth[0])).all()


S_OUTER: Rect = (135, 80, 1650, 920)  # the observed speaker tile in 1920x1080
G_RECTS = [(7, 277, 953, 526), (960, 277, 953, 526)]


def _base_layout() -> GalleryLayout:
    return layout_from_estimates([(0.0, [S_OUTER])], 1920, 1080)


def _speaker_face(identity: np.ndarray, rng: np.random.Generator) -> Face:
    return Face(box=(880, 300, 160, 200), score=0.9, embedding=_embed(identity, rng))


def _samples(script: Sequence[Tuple[float, float, str]], fps: float = 2.0) -> List[IdentitySample]:
    """Identity samples following ``(start, end, what)`` with ``what`` in A, B, gallery, band, share, none."""
    rng = np.random.default_rng(3)
    people = {"A": _unit(10), "B": _unit(11)}
    out = []
    for start, end, what in script:
        for t in np.arange(start, end, 1.0 / fps):
            if what in people:
                out.append(IdentitySample(float(t), [S_OUTER], [_speaker_face(people[what], rng)]))
            elif what == "none":
                out.append(IdentitySample(float(t), [S_OUTER], []))
            elif what in ("gallery", "band"):
                faces = [
                    Face((380, 420, 150, 150), 0.9, _embed(people["B"], rng)),
                    Face((1360, 430, 140, 140), 0.9, _embed(people["A"], rng)),
                ]
                rects = G_RECTS if what == "gallery" else [(7, 277, 1906, 526)]
                out.append(IdentitySample(float(t), list(rects), faces))
            else:
                out.append(IdentitySample(float(t), [(0, 0, 1920, 1080)], []))
    return out


def _speaker_layout(script: Sequence[Tuple[float, float, str]]) -> Optional[GalleryLayout]:
    config = load_defaults()
    samples = _samples(script)
    return speaker_view_layout(samples, _base_layout(), config["tiles"], config["speaker_view"], script[-1][1])


class TestSpeakerView:
    """Person tracks from identity samples of an active-speaker recording."""

    SCRIPT = [(0, 10, "A"), (10, 25, "B"), (25, 40, "A"), (40, 50, "gallery"), (50, 60, "B")]

    def test_candidate_needs_one_large_tile(self) -> None:
        """The observed 1650x920 tile qualifies; a two-tile gallery does not."""
        config = load_defaults()["speaker_view"]
        assert is_speaker_view_candidate(_base_layout(), config)
        assert not is_speaker_view_candidate(layout_from_estimates([(0.0, G_RECTS)], 1920, 1080), config)

    def test_switching_speaker_becomes_two_person_tracks(self) -> None:
        """A, B, A in the speaker tile, a gallery stretch, then B: P0 is A throughout, P1 is B."""
        layout = _speaker_layout(self.SCRIPT)
        assert layout is not None and layout.mode == "speaker_view"
        assert [t.tile_id for t in layout.tiles] == ["P0", "P1"]
        assert all(t.derivation == "person_track" for t in layout.tiles)
        assert [r.tile_id for r in layout.regions] == ["S", "G0", "G1"]
        p0, p1 = layout.persons
        assert [(s.start_s, s.end_s, s.region_id) for s in p0.segments] == [
            (0.0, 9.5, "S"),
            (25.0, 39.5, "S"),
            (40.0, 49.5, "G1"),
        ]
        assert [(s.start_s, s.end_s, s.region_id) for s in p1.segments] == [
            (10.0, 24.5, "S"),
            (40.0, 49.5, "G0"),
            (50.0, 60.0, "S"),
        ]
        modes = [(seg.mode, seg.persons) for seg in layout.timeline]
        assert ("gallery", {"P1": "G0", "P0": "G1"}) in modes and ("speaker_view", {"P0": "S"}) in modes
        assert layout.identity["recognizer"]["sha256"].startswith("0ba9fbfa")
        assert 0.8 < layout.reference_fraction < 0.9

    def test_visibility_follows_the_segments(self) -> None:
        """``visible`` is True inside a person's segments only; a gallery tile is always visible."""
        layout = _speaker_layout(self.SCRIPT)
        assert layout is not None
        shown = layout.visible("P0", [5.0, 9.75, 15.0, 30.0, 45.0, 55.0])
        assert shown.tolist() == [True, False, False, True, True, False]
        assert layout_from_estimates([(0.0, G_RECTS)], 1920, 1080).visible("T0", [1.0, 2.0]).all()

    def test_merged_gallery_band_is_split_by_faces(self) -> None:
        """A gallery stretch whose two tiles read as one band is cut between the faces into G0 and G1."""
        layout = _speaker_layout([(0, 20, "A"), (20, 40, "B"), (40, 50, "band"), (50, 60, "A")])
        assert layout is not None
        assert [r.tile_id for r in layout.regions] == ["S", "G0", "G1"]
        g0, g1 = layout.regions[1], layout.regions[2]
        assert g0.x + g0.width < g1.x and abs(g1.x - 960) <= 30

    def test_one_person_throughout_is_not_speaker_view(self) -> None:
        """No identity switch: the single-tile gallery layout stands."""
        assert _speaker_layout([(0, 30, "A"), (30, 35, "none"), (35, 60, "A")]) is None

    def test_mostly_screen_share_is_not_speaker_view(self) -> None:
        """Fewer than half the samples show the speaker tile."""
        assert _speaker_layout([(0, 10, "A"), (10, 20, "B"), (20, 60, "share")]) is None

    def test_unseen_face_samples_leave_a_gap(self) -> None:
        """Samples with no face end a person's segment; the frames between are attributed to nobody."""
        layout = _speaker_layout([(0, 10, "A"), (10, 12, "none"), (12, 30, "B"), (30, 40, "A")])
        assert layout is not None
        p0 = layout.persons[0]
        assert [(s.start_s, s.end_s) for s in p0.segments] == [(0.0, 9.5), (30.0, 40.0)]


class TestWorkerPayload:
    """Person-track segments reach the workers and pick the crop per frame."""

    def test_payload_and_placement(self) -> None:
        """A person track's segments name the region box; outside them the track is skipped."""
        layout = _speaker_layout(TestSpeakerView.SCRIPT)
        assert layout is not None
        payload = _tile_payload(layout)
        p0 = next(p for p in payload if p["tile_id"] == "P0")
        assert len(p0["segments"]) == 3
        region, box, masks = tile_placement(p0, 5.0)  # type: ignore[misc]
        assert region == "S" and tuple(box) == layout.regions[0].box and masks
        assert tile_placement(p0, 15.0) is None
        assert tile_placement(p0, 45.0)[0] == "G1"  # type: ignore[index]
        assert widest_box(payload) == layout.regions[0].width
        gallery = _tile_payload(layout_from_estimates([(0.0, G_RECTS)], 1920, 1080))
        assert "segments" not in gallery[0] and tile_placement(gallery[0], 1e6)[0] == "T0"  # type: ignore[index]

    def test_pad_stack_keeps_every_crop_whole(self) -> None:
        """Crops of different sizes are zero-padded to the largest instead of cut to the smallest."""
        crops = [np.full((4, 6, 3), 7, np.uint8), np.full((2, 3, 3), 9, np.uint8)]
        stacked = pad_stack(crops)
        assert stacked.shape == (2, 4, 6, 3) and (stacked[0] == 7).all() and stacked[1, 1, 2, 0] == 9
        assert stacked[1, 3, 5, 0] == 0


def _fake_mediapipe(times: np.ndarray) -> object:
    """A stand-in for the MediaPipe worker that applies the payload's placement to each time."""

    def run(path: str, layout: object, config: dict, timeout_s: Optional[float] = None) -> tuple:
        arrays: Dict[str, np.ndarray] = {}
        for tile in _tile_payload(layout):  # type: ignore[arg-type]
            kept = np.array([t for t in times if tile_placement(tile, float(t)) is not None])
            n = len(kept)
            key = tile["tile_id"]
            arrays.update(
                {
                    f"{key}/time_s": kept,
                    f"{key}/face_present": np.ones(n, bool),
                    f"{key}/n_faces": np.ones(n, np.int16),
                    f"{key}/pose": np.zeros((n, 3), np.float32),
                    f"{key}/translation": np.zeros((n, 3), np.float32),
                    f"{key}/face_box": np.zeros((n, 4), np.float32),
                    f"{key}/eye_offset": np.zeros((n, 2), np.float32),
                    f"{key}/mouth_aspect": np.zeros(n, np.float32),
                    f"{key}/motion_energy": np.zeros(n, np.float32),
                    f"{key}/blendshapes": np.zeros((n, 1), np.float32),
                }
            )
        return {"blendshape_names": ["jawOpen"], "frames": len(times), "video_s": float(times[-1])}, arrays

    return run


class TestPersonTrackTables:
    """Geometry rows are attributed to the person shown, only where that person is visible."""

    def test_geometry_rows_follow_the_person(self, tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
        """P0 has rows in its three segments, tagged with the region analysed; none while P1 is shown."""
        layout = _speaker_layout(TestSpeakerView.SCRIPT)
        assert layout is not None
        video = tmp_path / "speaker.mp4"  # type: ignore[operator]
        video.write_bytes(b"")
        times = np.arange(0, 60, 1 / 8)
        monkeypatch.setattr(facial_api.backends, "run_mediapipe", _fake_mediapipe(times))
        geometry = extract_face_geometry(video, layout=layout)
        assert set(geometry.tiles) == {"P0", "P1"}
        p0 = geometry.tiles["P0"]
        assert set(p0["region_id"]) == {"S", "G1"}
        assert not ((p0["time_s"] > 10.0) & (p0["time_s"] < 24.5)).any()
        assert len(p0) == len(geometry.tiles["P0"]["time_s"].unique())
        grid = geometry.on_grid("bs_jawOpen", times)
        assert np.isnan(grid["P0"][(times > 10.0) & (times < 24.5)]).all()
        assert np.isfinite(grid["P0"][times < 9.5]).all()
        assert (np.isfinite(grid["P0"]) == layout.visible("P0", times)).all()


class TestPersonTrackLinking:
    """Linking person tracks to diarized speakers in active-speaker view."""

    def test_visibility_is_a_link_cue(self) -> None:
        """The person shown is the one talking: mouth and visibility both pick the right speaker."""
        fps = 8.0
        t = np.arange(0, 120, 1 / fps)
        turns = [("SPK_A" if i % 2 == 0 else "SPK_B", 6.0 * i, 6.0 * i + 5.0) for i in range(20)]
        activity: Dict[str, list] = {}
        for speaker, start, end in turns:
            activity.setdefault(speaker, []).append((start, end))
        speech = {s: segments_to_activity(segs, t) for s, segs in activity.items()}
        shown = {"P0": np.zeros(t.size, bool), "P1": np.zeros(t.size, bool)}
        for i in range(20):
            person = "P1" if i % 2 == 0 else "P0"
            shown[person] |= (t >= 6.0 * i + 0.5) & (t < 6.0 * i + 6.5)
        rng = np.random.default_rng(5)
        mouths = {}
        for person, speaker in (("P1", "SPK_A"), ("P0", "SPK_B")):
            talking = 0.25 + 0.2 * np.sin(2 * np.pi * 4.0 * t + 1.0) * rng.uniform(0.5, 1.0, t.size)
            series = np.where(speech[speaker] > 0, talking, 0.05 + rng.normal(0, 0.01, t.size))
            mouths[person] = np.where(shown[person], series, np.nan)
        link = link_tiles_to_speakers(mouths, speech, fps, visibility_per_tile=shown)
        assert link.assignment == {"P0": "SPK_B", "P1": "SPK_A"}
        corr = link.visibility_activity_corr
        assert corr["P1"]["SPK_A"] > 0.5 and corr["P0"]["SPK_B"] > 0.5 and corr["P0"]["SPK_A"] < 0
        plain = link_tiles_to_speakers(mouths, speech, fps)
        assert plain.visibility_activity_corr == {}


def test_frame_table_on_grid_tolerance() -> None:
    """Rows further than half a frame step from a grid time leave NaN there."""
    import pandas as pd

    layout = layout_from_estimates([(0.0, G_RECTS)], 1920, 1080)
    table = FrameTable(
        kind="geometry",
        fps=4.0,
        layout=layout,
        tiles={"T0": pd.DataFrame({"time_s": [0.0, 0.25, 1.0], "x": [1.0, 2.0, 3.0]})},
    )
    values = table.on_grid("x", [0.0, 0.25, 0.5, 0.75, 1.0])["T0"]
    assert values[[0, 1, 4]].tolist() == [1.0, 2.0, 3.0] and np.isnan(values[[2, 3]]).all()
