"""Schema validation, repair and aggregation with canned VLM outputs."""

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest

from senselab.video.tasks.scene_description.aggregation import aggregate, frame_fields
from senselab.video.tasks.scene_description.backends import Generation, GenerationRequest
from senselab.video.tasks.scene_description.cv_metrics import FrameMetrics
from senselab.video.tasks.scene_description.describe import describe_frames
from senselab.video.tasks.scene_description.schema import SceneDescription, parse_scene_json


def _tile(**over: Any) -> Dict[str, Any]:  # noqa: ANN401
    tile = {
        "camera_on": True,
        "n_people_visible": 1,
        "setting": "home",
        "background_type": "real",
        "lighting": "good",
        "other_people_present": False,
        "distractions": ["none"],
        "head_framing": "centered",
    }
    tile.update(over)
    return tile


def _scene(layout: str = "single", tiles: Optional[List[Dict[str, Any]]] = None, notes: str = "ok") -> Dict:
    tiles = tiles if tiles is not None else [_tile()]
    return {"layout": layout, "n_tiles": len(tiles), "tiles": tiles, "notes": notes}


GOOD = json.dumps(_scene())


class FakeBackend:
    """Returns canned completions in order and records the requests."""

    model_id = "fake/vlm"
    revision = "0" * 40

    def __init__(self, outputs: Sequence[str]) -> None:
        """Queue the completions."""
        self.outputs = list(outputs)
        self.requests: List[GenerationRequest] = []

    def generate(self, requests: Sequence[GenerationRequest]) -> List[Generation]:
        """Pop one canned completion per request."""
        self.requests.extend(requests)
        return [
            Generation(text=self.outputs.pop(0), seconds=1.0, prompt_tokens=10, generation_tokens=5) for _ in requests
        ]

    def stats(self) -> Dict[str, Any]:
        """No resources."""
        return {"peak_memory_bytes": 0}


IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def test_parse_valid_json() -> None:
    """Valid JSON parses with no error."""
    desc, err = parse_scene_json(GOOD)
    assert err is None and desc is not None
    assert desc.tiles[0].setting == "home"


def test_parse_strips_fences_and_prose_and_normalises_case() -> None:
    """Code fences, surrounding text and enum case and spacing are tolerated."""
    raw = (
        "Here you go:\n```json\n" + json.dumps(_scene(layout="Speaker View", tiles=[_tile(setting="Office")])) + "\n```"
    )
    desc, err = parse_scene_json(raw)
    assert err is None and desc is not None
    assert desc.layout == "speaker_view"
    assert desc.tiles[0].setting == "office"


def test_parse_distractions_normalised() -> None:
    """``none`` is dropped beside real distractions, and an empty list becomes ``["none"]``."""
    desc, _ = parse_scene_json(json.dumps(_scene(tiles=[_tile(distractions=["Pets", "none", "pets"])])))
    assert desc is not None and desc.tiles[0].distractions == ["pets"]
    desc, _ = parse_scene_json(json.dumps(_scene(tiles=[_tile(distractions=[])])))
    assert desc is not None and desc.tiles[0].distractions == ["none"]


@pytest.mark.parametrize(
    "raw, fragment",
    [
        ("not json at all", "invalid JSON"),
        ('{"layout": "single", "n_tiles": 1, "tiles": [', "invalid JSON"),
        (json.dumps(_scene(layout="mosaic")), "layout"),
        (json.dumps(_scene(tiles=[_tile(distractions=["dragons"])])), "distractions"),
        (json.dumps({**_scene(), "extra": 1}), "extra"),
        (json.dumps(_scene(tiles=[_tile(n_people_visible=-1)])), "n_people_visible"),
    ],
)
def test_parse_rejects(raw: str, fragment: str) -> None:
    """Invalid output yields no description and an error naming the problem."""
    desc, err = parse_scene_json(raw)
    assert desc is None
    assert err is not None and fragment in err


def test_describe_frames_valid_first_time() -> None:
    """A valid completion needs no repair."""
    backend = FakeBackend([GOOD, GOOD])
    out = describe_frames([IMG, IMG], backend)
    assert [r.parse_ok for r in out] == [True, True]
    assert not any(r.repaired for r in out)
    assert len(backend.requests) == 2
    assert all(r.image is not None for r in backend.requests)


def test_describe_frames_repairs_once() -> None:
    """An invalid completion is sent back text-only with its error, and the repair is used."""
    backend = FakeBackend(["{broken", GOOD, GOOD])
    out = describe_frames([IMG, IMG], backend)
    assert [r.parse_ok for r in out] == [True, True]
    assert out[0].repaired and not out[1].repaired
    assert out[0].raw_text == "{broken" and out[0].repair_text == GOOD
    repair = backend.requests[2]
    assert repair.image is None
    assert "{broken" in repair.prompt and "invalid JSON" in repair.prompt
    assert out[0].seconds == pytest.approx(2.0)


def test_describe_frames_records_failure_after_repair() -> None:
    """Two invalid completions leave ``parse_ok`` False with the last error; no third attempt."""
    backend = FakeBackend(["nope", "still nope"])
    out = describe_frames([IMG], backend)
    assert out[0].parse_ok is False and out[0].description is None
    assert out[0].error is not None and "invalid JSON" in out[0].error
    assert len(backend.requests) == 2


def test_describe_frames_batches() -> None:
    """Images are sent in batches of ``batch_size``."""
    backend = FakeBackend([GOOD] * 5)
    out = describe_frames([IMG] * 5, backend, batch_size=2)
    assert len(out) == 5 and all(r.parse_ok for r in out)


def _metrics(brightness: float = 0.5) -> FrameMetrics:
    return FrameMetrics(
        brightness=brightness, contrast=0.2, blur=100.0, colourfulness=10.0, near_black_fraction=0.0, n_faces=1
    )


def test_frame_fields_primary_tile_and_counts() -> None:
    """Primary fields come from the first tile; counts sum over tiles."""
    desc = SceneDescription.model_validate(
        _scene(
            layout="gallery",
            tiles=[_tile(setting="office"), _tile(camera_on=False, n_people_visible=0, other_people_present=False)],
        )
    )
    fields = frame_fields(desc)
    assert fields["primary_setting"] == "office"
    assert fields["n_camera_on"] == 1 and fields["n_people_visible"] == 1
    assert fields["any_other_people_present"] is False


def test_aggregate_modes_consistency_and_change_points() -> None:
    """Modes, consistency, distraction rates and change points over five frames."""
    scenes = [
        _scene(),
        _scene(),
        _scene(tiles=[_tile(n_people_visible=2, other_people_present=True, distractions=["children"])]),
        None,
        _scene(layout="gallery", tiles=[_tile(), _tile()]),
    ]
    descriptions = [SceneDescription.model_validate(s) if s else None for s in scenes]
    summary = aggregate([0, 60, 120, 180, 240], descriptions, [_metrics(b) for b in (0.1, 0.2, 0.3, 0.4, 0.5)])
    assert summary.n_frames == 5 and summary.n_parsed == 4
    assert summary.parse_rate == pytest.approx(0.8)
    layout = summary.categorical["layout"]
    assert layout.mode == "single" and layout.consistency == pytest.approx(0.75)
    assert summary.numeric["n_tiles"].mean == pytest.approx(1.25)
    assert summary.distractions["children"] == pytest.approx(0.25)
    assert summary.distractions["pets"] == 0.0
    assert summary.metrics["brightness"].mean == pytest.approx(0.3)
    changes = [(c.timestamp_s, c.field, c.before, c.after) for c in summary.change_points]
    assert changes == [
        (120, "any_other_people_present", False, True),
        (240, "layout", "single", "gallery"),
        (240, "n_tiles", 1, 2),
        (240, "any_other_people_present", True, False),
    ]


def test_aggregate_empty_and_mismatched() -> None:
    """No frames gives an empty summary; mismatched lengths are rejected."""
    summary = aggregate([], [], [])
    assert summary.n_frames == 0 and summary.parse_rate == 0.0
    assert summary.categorical["layout"].mode is None
    with pytest.raises(ValueError):
        aggregate([0.0], [], [])
