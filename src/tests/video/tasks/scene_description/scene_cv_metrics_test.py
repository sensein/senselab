"""Cheap image metrics on synthetic images."""

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from senselab.video.tasks.scene_description.cv_metrics import (  # noqa: E402
    colourfulness,
    frame_metrics,
    near_black_fraction,
)

REPO = Path(__file__).resolve().parents[5]


def test_uniform_grey() -> None:
    """A flat grey image: mid brightness, no contrast, no edges, no colour, no faces."""
    m = frame_metrics(np.full((120, 160, 3), 128, dtype=np.uint8))
    assert m.brightness == pytest.approx(128 / 255, abs=1e-3)
    assert m.contrast == pytest.approx(0.0, abs=1e-6)
    assert m.blur == pytest.approx(0.0, abs=1e-6)
    assert m.colourfulness == pytest.approx(0.0, abs=1e-6)
    assert m.near_black_fraction == 0.0
    assert m.n_faces == 0


def test_blur_drops_when_smoothed() -> None:
    """Variance of the Laplacian falls when a checkerboard is blurred."""
    board = ((np.indices((128, 128)).sum(axis=0) // 8) % 2 * 255).astype(np.uint8)
    sharp = np.stack([board] * 3, axis=-1)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert frame_metrics(sharp, detect_faces=False).blur > 10 * frame_metrics(blurred, detect_faces=False).blur


def test_contrast_of_half_black_half_white() -> None:
    """Half black, half white has contrast 0.5 and brightness 0.5."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, 32:] = 255
    m = frame_metrics(img, detect_faces=False)
    assert m.contrast == pytest.approx(0.5, abs=1e-3)
    assert m.brightness == pytest.approx(0.5, abs=1e-3)
    assert m.n_faces == -1


def test_colourfulness_orders_grey_below_saturated() -> None:
    """A saturated two-colour image is more colourful than a grey one."""
    grey = np.full((32, 32, 3), 100, dtype=np.uint8)
    vivid = np.zeros((32, 32, 3), dtype=np.uint8)
    vivid[:, :16] = (255, 0, 0)
    vivid[:, 16:] = (0, 0, 255)
    assert colourfulness(vivid) > 50 > colourfulness(grey)


def test_near_black_fraction_of_camera_off_tile() -> None:
    """A gallery with one dark camera-off tile of four reads about a quarter near-black."""
    rng = np.random.default_rng(0)
    img = rng.integers(60, 220, size=(256, 256, 3), dtype=np.uint8)
    img[:128, :128] = 30
    m = frame_metrics(img, detect_faces=False)
    assert m.near_black_fraction == pytest.approx(0.25, abs=0.01)


def test_dark_but_textured_is_not_near_black() -> None:
    """A dark region with texture (a dim room) is not counted as a camera-off area."""
    rng = np.random.default_rng(0)
    luma = rng.integers(0, 80, size=(64, 64)).astype(np.float32)
    assert near_black_fraction(luma) == 0.0


def test_rejects_non_rgb() -> None:
    """A grayscale array is rejected."""
    with pytest.raises(ValueError):
        frame_metrics(np.zeros((10, 10), dtype=np.uint8))


@pytest.mark.skipif(not (REPO / "tutorial_images" / "three_people.jpg").exists(), reason="tutorial image missing")
def test_faces_found_in_public_photo() -> None:
    """The Haar cascade finds at least one face in the public three-person tutorial photo."""
    from PIL import Image

    img = np.array(Image.open(REPO / "tutorial_images" / "three_people.jpg").convert("RGB"))
    assert frame_metrics(img).n_faces >= 1
    assert frame_metrics(np.array(Image.open(REPO / "tutorial_images" / "no_people.jpeg").convert("RGB"))).n_faces == 0


def _zoom_pair(right_tile_dark: bool = False) -> np.ndarray:
    """A 1920x1080 letterboxed gallery with two 953x526 textured tiles, downscaled to 896x504."""
    rng = np.random.default_rng(1)
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[277:803] = 26
    img[277:803, 7:960] = rng.integers(60, 220, size=(526, 953, 3), dtype=np.uint8)
    img[277:803, 961:1914] = rng.integers(60, 220, size=(526, 953, 3), dtype=np.uint8)
    if right_tile_dark:
        img[277:803, 961:1914] = 30
        img[530:550, 1380:1500] = 230  # the participant's name
    return cv2.resize(img, (896, 504), interpolation=cv2.INTER_AREA)


PAIR_TILES = [(3, 129, 445, 245), (448, 129, 445, 245)]


def test_letterbox_bars_are_not_camera_off() -> None:
    """Black bars above and below two live tiles: large near-black area, no camera-off area."""
    m = frame_metrics(_zoom_pair(), detect_faces=False)
    assert m.near_black_fraction > 0.4
    assert m.bar_fraction == pytest.approx((1080 - 526) / 1080, abs=0.02)
    assert m.camera_off_fraction < 0.05
    assert m.tiles_camera_off == []


def test_one_dark_tile_in_a_letterboxed_pair_is_camera_off() -> None:
    """Without tiles about half the content area is near-black; with tiles one of two is off."""
    image = _zoom_pair(right_tile_dark=True)
    m = frame_metrics(image, detect_faces=False)
    assert 0.35 < m.camera_off_fraction < 0.55
    tiled = frame_metrics(image, detect_faces=False, tiles=PAIR_TILES)
    assert tiled.tiles_camera_off == [False, True]
    assert tiled.camera_off_fraction == 0.5


def test_live_tiles_with_bars_are_all_on() -> None:
    """Tile boxes over two live tiles: none camera-off."""
    m = frame_metrics(_zoom_pair(), detect_faces=False, tiles=PAIR_TILES)
    assert m.tiles_camera_off == [False, False] and m.camera_off_fraction == 0.0


def test_pillarbox_bars_are_removed() -> None:
    """A 4:3 picture pillarboxed in 16:9: the side bars are excluded."""
    rng = np.random.default_rng(2)
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:, 80:560] = rng.integers(60, 220, size=(360, 480, 3), dtype=np.uint8)
    m = frame_metrics(img, detect_faces=False)
    assert m.bar_fraction == pytest.approx(0.25, abs=0.01)
    assert m.camera_off_fraction == 0.0


def test_camera_off_speaker_view_keeps_its_area() -> None:
    """A full-frame camera-off card with a centred name is not mistaken for bars around the name."""
    img = np.full((360, 640, 3), 30, dtype=np.uint8)
    img[170:190, 260:380] = 230
    m = frame_metrics(img, detect_faces=False)
    assert m.bar_fraction == 0.0
    assert m.camera_off_fraction > 0.9
