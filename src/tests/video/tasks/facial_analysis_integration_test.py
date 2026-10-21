"""Facial analysis end to end on a real talking-head clip, through both subprocess-venv workers.

The clip is "Interview with Barbara Firth" from Wikimedia Commons
(https://commons.wikimedia.org/wiki/File:Interview_with_Barbara_Firth.webm), CC BY 3.0, 656x480,
139 s, about 4 MB. It is downloaded to the test's temporary directory and never committed. Twelve
seconds of it are composed into a two-tile 1280x720 gallery (the second tile shows the clip five
seconds later), with a dark canvas and a name label, as in a Zoom recording.

Each test is skipped when its backend venv has not been provisioned on this host, so CI does not
build them, and when the clip cannot be downloaded.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import List

import numpy as np
import pytest

from senselab.utils.subprocess_venv import provisioned_venv_dirs
from senselab.video.tasks.facial_analysis import (
    detect_gallery_tiles,
    detect_nonverbal_events,
    extract_face_geometry,
    extract_facial_affect,
    summarize_tiles,
)
from senselab.video.tasks.facial_analysis.frame_stream import iter_frames

av = pytest.importorskip("av")

CLIP_URL = "https://upload.wikimedia.org/wikipedia/commons/d/d0/Interview_with_Barbara_Firth.webm"
MEDIAPIPE_READY = bool(provisioned_venv_dirs("mediapipe-face"))
PYFEAT_READY = bool(provisioned_venv_dirs("pyfeat"))


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop to the target aspect ratio, then resize (no squash)."""
    from PIL import Image

    h, w = image.shape[:2]
    keep = int(round(w * height / width))
    top = max(0, (h - keep) // 2)
    image = np.ascontiguousarray(image[top : top + keep])
    return np.array(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))


@pytest.fixture(scope="module")
def gallery(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 12 s, 15 fps, two-tile gallery built from the CC BY clip."""
    directory = tmp_path_factory.mktemp("facial_integration")
    clip = directory / "barbara_firth.webm"
    try:
        request = urllib.request.Request(
            CLIP_URL, headers={"User-Agent": "senselab-tests (https://github.com/sensein/senselab)"}
        )
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            clip.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - offline hosts skip
        pytest.skip(f"clip unavailable: {exc}")
    frames: List[np.ndarray] = [frame for t, frame in iter_frames(str(clip), fps=15.0, start_s=30.0, end_s=47.5)]
    if len(frames) < 250:
        pytest.skip("clip decoded too short")
    out = directory / "gallery.mp4"
    tile_w, tile_h, top = 632, 356, 182
    with av.open(str(out), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=15)
        stream.width, stream.height, stream.pix_fmt, stream.bit_rate = 1280, 720, "yuv420p", 6_000_000
        for i in range(180):
            canvas = np.full((720, 1280, 3), 26, np.uint8)
            for x, offset in ((6, 0), (642, 75)):
                tile = _resize(frames[i + offset], tile_w, tile_h)
                tile[tile_h - 30 : tile_h - 8, 8:150] = (60, 60, 60)
                canvas[top : top + tile_h, x : x + tile_w] = tile
            for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


@pytest.mark.skipif(not MEDIAPIPE_READY, reason="mediapipe-face venv not provisioned on this host")
def test_geometry_finds_the_face_in_both_tiles(gallery: Path) -> None:
    """Two tiles; a face in nearly every frame of each; pose, blendshapes and events come out."""
    layout = detect_gallery_tiles(gallery, sample_frames=12)
    assert [t.tile_id for t in layout.tiles] == ["T0", "T1"]
    geometry = extract_face_geometry(gallery, layout=layout, fps=8.0)
    assert set(geometry.tiles) == {"T0", "T1"}
    for frame in geometry.tiles.values():
        assert len(frame) == 96
        assert frame["face_present"].mean() > 0.8
        assert {"bs_jawOpen", "bs_eyeBlinkLeft", "head_yaw_deg", "toward_camera", "motion_energy"} <= set(frame)
        # She is turned toward her right (image left) throughout the interview.
        assert frame.loc[frame["face_present"], "head_yaw_deg"].median() > 10.0
    assert geometry.timing["x_realtime"] > 0
    events = detect_nonverbal_events(geometry)
    summaries = summarize_tiles(geometry, None, events)
    assert [s.tile_id for s in summaries] == ["T0", "T1"]


@pytest.mark.skipif(not PYFEAT_READY, reason="pyfeat venv not provisioned on this host")
def test_affect_scores_the_face_in_both_tiles(gallery: Path) -> None:
    """Emotion probabilities sum to one, valence/arousal are bounded, and pose agrees in sign."""
    layout = detect_gallery_tiles(gallery, sample_frames=12)
    affect = extract_facial_affect(gallery, layout=layout, fps=1.0)
    emotions = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
    for frame in affect.tiles.values():
        assert len(frame) == 12
        face = frame[frame["face_score"] > 0]
        assert len(face) >= 10
        assert np.allclose(face[emotions].sum(axis=1), 1.0, atol=1e-3)
        assert face["valence"].between(-1, 1).all() and face["arousal"].between(-1, 1).all()
        assert "AU12" in face and face["head_yaw_deg"].median() > 5.0


# Two public-domain official US government portraits (works of the US federal government), fetched
# from Wikimedia Commons at test time and never committed.
PORTRAIT_URLS = {
    "A": "https://commons.wikimedia.org/wiki/Special:FilePath/Official_portrait_of_Barack_Obama.jpg?width=480",
    "B": "https://commons.wikimedia.org/wiki/Special:FilePath/Joe_Biden_official_portrait_2013_cropped.jpg?width=480",
}
SPEAKER_SCRIPT = [(0.0, 5.0, "A"), (5.0, 10.0, "B"), (10.0, 14.0, "A"), (14.0, 17.0, "gallery"), (17.0, 20.0, "B")]


def _place(portrait: np.ndarray, width: int, height: int, grey: int) -> np.ndarray:
    """A tile of ``width x height`` filled with ``grey`` and the portrait centred at full tile height."""
    from PIL import Image

    h, w = portrait.shape[:2]
    scaled = np.array(Image.fromarray(portrait).resize((int(round(w * height / h)), height), Image.Resampling.BILINEAR))
    tile = np.full((height, width, 3), grey, np.uint8)
    left = (width - scaled.shape[1]) // 2
    tile[:, left : left + scaled.shape[1]] = scaled[:, : width - left]
    return tile


@pytest.fixture(scope="module")
def speaker_view_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """20 s at 8 fps, 1280x720: A, B, A in one speaker tile, a two-tile gallery stretch, then B."""
    from PIL import Image

    directory = tmp_path_factory.mktemp("speaker_view")
    portraits = {}
    for key, url in PORTRAIT_URLS.items():
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "senselab-tests (https://github.com/sensein/senselab)"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
                (directory / f"{key}.jpg").write_bytes(response.read())
            portraits[key] = np.array(Image.open(directory / f"{key}.jpg").convert("RGB"))
        except Exception as exc:  # noqa: BLE001 - offline hosts skip
            pytest.skip(f"portrait unavailable: {exc}")
    speaker = {k: _place(p, 1060, 600, 110) for k, p in portraits.items()}
    small = {k: _place(p, 632, 356, 110) for k, p in portraits.items()}
    out = directory / "speaker_view.mp4"
    rng = np.random.default_rng(0)
    with av.open(str(out), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=8)
        stream.width, stream.height, stream.pix_fmt, stream.bit_rate = 1280, 720, "yuv420p", 6_000_000
        for i in range(160):
            t = i / 8
            what = next(w for start, end, w in SPEAKER_SCRIPT if start <= t < end)
            canvas = np.full((720, 1280, 3), 26, np.uint8)
            if what == "gallery":
                canvas[182:538, 6:638] = small["B"]
                canvas[182:538, 642:1274] = small["A"]
            else:
                canvas[60:660, 110:1170] = speaker[what]
            noisy = np.clip(canvas.astype(np.int16) + rng.integers(-3, 4, canvas.shape), 0, 255).astype(np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(noisy, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out


@pytest.mark.slow
def test_speaker_view_person_tracks_on_real_faces(speaker_view_video: Path) -> None:
    """YuNet + SFace separate two real faces into P0 (A) and P1 (B), across the speaker tile and the gallery."""
    try:
        from senselab.video.tasks.facial_analysis.identity import identity_model_paths

        identity_model_paths()
    except Exception as exc:  # noqa: BLE001 - offline hosts skip
        pytest.skip(f"identity models unavailable: {exc}")
    layout = detect_gallery_tiles(speaker_view_video, sample_frames=24)
    assert layout.mode == "speaker_view", layout.timeline
    assert [t.tile_id for t in layout.tiles] == ["P0", "P1"]
    assert [r.tile_id for r in layout.regions] == ["S", "G0", "G1"]
    assert layout.visible("P0", [2.5, 12.0, 15.5]).all() and not layout.visible("P0", [7.5, 18.5]).any()
    assert layout.visible("P1", [7.5, 15.5, 18.5]).all() and not layout.visible("P1", [2.5, 12.0]).any()
    gallery = {s.region_id for p in layout.persons for s in p.segments if s.region_id != "S"}
    assert gallery == {"G0", "G1"}
    p0_gallery = [s.region_id for s in layout.persons[0].segments if s.start_s <= 15.5 <= s.end_s]
    assert p0_gallery == ["G1"]
    if MEDIAPIPE_READY:
        geometry = extract_face_geometry(speaker_view_video, layout=layout, fps=4.0)
        p0 = geometry.tiles["P0"]
        assert set(p0["region_id"]) == {"S", "G1"}
        assert not p0["time_s"].between(5.5, 9.5).any()
        assert p0["face_present"].mean() > 0.8
