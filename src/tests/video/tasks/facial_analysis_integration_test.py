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
