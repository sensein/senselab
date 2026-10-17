"""End-to-end report with a mocked VLM, and the real-VLM integration tests."""

import json
from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
pytest.importorskip("cv2")

from scene_describe_test import FakeBackend, _scene, _tile  # noqa: E402

from senselab.video.tasks.scene_description.api import describe_video_scene, describe_video_scenes  # noqa: E402
from senselab.video.tasks.scene_description.backends import (  # noqa: E402
    DEFAULT_MLX_MODEL,
    GenerationRequest,
    is_apple_silicon,
)
from senselab.video.tasks.scene_description.schema import DESCRIBE_PROMPT, parse_scene_json  # noqa: E402

REPO = Path(__file__).resolve().parents[5]
PUBLIC_IMAGE = REPO / "tutorial_images" / "three_people.jpg"


@pytest.fixture(scope="module")
def two_scene_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Six seconds of grey then six of blue, 5 fps."""
    path = tmp_path_factory.mktemp("api") / "two.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=5)
        stream.width, stream.height, stream.pix_fmt = 160, 120, "yuv420p"
        for colour in ((128, 128, 128), (30, 60, 200)):
            for _ in range(30):
                img = np.empty((120, 160, 3), dtype=np.uint8)
                img[:] = colour
                for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                    container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_report_with_mocked_vlm(two_scene_video: Path) -> None:
    """Two sampled frames, one repaired completion, and a layout change point at the cut."""
    outputs = [
        json.dumps(_scene()),
        "```json\n{bad",
        json.dumps(_scene(layout="gallery", tiles=[_tile(), _tile()])),
    ]
    backend = FakeBackend(outputs)
    report = describe_video_scene(two_scene_video, backend=backend, interval_s=60)
    assert [f.reason for f in report.frames] == ["interval", "scene_cut"]
    assert report.frames[1].vlm is not None and report.frames[1].vlm.repaired
    assert report.summary.n_parsed == 2
    assert [c.field for c in report.summary.change_points] == ["layout", "n_tiles"]
    assert report.model_id == "fake/vlm" and report.model_revision == "0" * 40
    assert report.frames[0].metrics.brightness == pytest.approx(128 / 255, abs=0.02)


def test_report_without_vlm(two_scene_video: Path) -> None:
    """``use_vlm=False`` measures frames and leaves the VLM fields empty."""
    reports = describe_video_scenes([two_scene_video, two_scene_video], use_vlm=False, interval_s=60)
    assert len(reports) == 2
    assert all(f.vlm is None for f in reports[0].frames)
    assert reports[0].model_id is None and reports[0].summary.n_parsed == 0


def _mlx_model_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(DEFAULT_MLX_MODEL, "config.json"), str)
    except Exception:
        return False


@pytest.mark.large_model
@pytest.mark.skipif(
    not (is_apple_silicon() and _mlx_model_cached() and PUBLIC_IMAGE.exists()),
    reason="needs Apple silicon, the cached MLX Qwen3-VL weights, and the tutorial image",
)
def test_mlx_vlm_describes_public_image() -> None:
    """The real MLX Qwen3-VL returns schema-valid JSON for a public photo of three people."""
    from PIL import Image

    from senselab.video.tasks.scene_description.backends import MLXVLMBackend
    from senselab.video.tasks.scene_description.describe import describe_frames

    backend = MLXVLMBackend()
    image = np.array(Image.open(PUBLIC_IMAGE).convert("RGB"))
    try:
        (record,) = describe_frames([image], backend)
    finally:
        backend.close()
    assert record.parse_ok, record.error
    assert record.description is not None
    assert sum(t.n_people_visible for t in record.description.tiles) >= 2
    assert backend.stats()["peak_memory_bytes"] > 0
    assert len(backend.revision) == 40


@pytest.mark.large_model
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(), reason="the transformers Qwen3-VL backend is exercised on CUDA hosts"
)
def test_transformers_vlm_describes_public_image() -> None:
    """The transformers Qwen3-VL backend returns parseable JSON for a public photo."""
    from PIL import Image

    from senselab.video.tasks.scene_description.backends import TransformersVLMBackend

    backend = TransformersVLMBackend()
    image = np.array(Image.open(PUBLIC_IMAGE).convert("RGB"))
    (gen,) = backend.generate([GenerationRequest(prompt=DESCRIBE_PROMPT, image=image)])
    desc, err = parse_scene_json(gen.text)
    assert desc is not None, err
