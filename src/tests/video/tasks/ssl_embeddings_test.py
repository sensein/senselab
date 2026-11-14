"""Tests for video SSL embeddings (a tiny randomly initialised V-JEPA2 on CPU; no network)."""

import functools
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import pytest
import torch
import transformers
from transformers import VJEPA2Config, VJEPA2Model

from senselab.utils.data_structures import HFModel
from senselab.video.data_structures import Video
from senselab.video.tasks.ssl_embeddings import api
from senselab.video.tasks.ssl_embeddings.api import extract_ssl_embeddings_from_video

SHA = "875c192b0000000000000000000000000000000a"
HIDDEN = 32


def _tiny() -> Tuple[Any, Callable[[torch.Tensor], torch.Tensor]]:
    """A 4-frame, 16 px V-JEPA2 with random weights, and a preprocess from (T, H, W, C) uint8 to (1, T, C, H, W)."""
    torch.manual_seed(0)
    config = VJEPA2Config(
        crop_size=16, frames_per_clip=4, tubelet_size=2, patch_size=8, hidden_size=HIDDEN, num_hidden_layers=1,
        num_attention_heads=2, mlp_ratio=2, pred_hidden_size=16, pred_num_attention_heads=2, pred_num_hidden_layers=1,
        pred_num_mask_tokens=1, pred_mlp_ratio=2,
    )  # fmt: skip
    return VJEPA2Model(config).eval(), lambda clip: clip.permute(0, 3, 1, 2).float().div(255).unsqueeze(0)


@pytest.fixture()
def video() -> Video:
    """A synthetic 32-frame, 16x16 clip at 8 fps."""
    gen = torch.Generator().manual_seed(1)
    return Video(frames=torch.randint(0, 256, (32, 16, 16, 3), generator=gen, dtype=torch.uint8), frame_rate=8.0)


def _model(revision: str) -> HFModel:
    """An HFModel built without the Hub check (model_construct skips the validators)."""
    return HFModel.model_construct(path_or_uri="facebook/vjepa2-test", revision=revision)


def test_non_hex_revision_raises(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ref name, a short hash and an upper-case hash are refused before anything is loaded."""
    monkeypatch.setattr(api, "_load_model", lambda model, device: pytest.fail("loaded a model with a bad revision"))
    for bad in ("main", "875c192b", SHA.upper(), SHA + "0"):
        with pytest.raises(ValueError, match="40-hex"):
            extract_ssl_embeddings_from_video(video, _model(bad), frames_per_clip=4, fps=8.0, pooling="mean")


def test_tiny_vjepa2_returns_clip_embeddings(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """32 frames at 8 fps in clips of 16 frames give shape (2, hidden), and a provenance record."""
    monkeypatch.setattr(api, "_load_model", lambda model, device: (*_tiny(), "cpu"))
    result = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="mean")
    assert result.embeddings.shape == (2, HIDDEN)
    record = result.provenance
    assert record["model"] == "facebook/vjepa2-test" and record["revision"] == SHA
    assert (record["frames_per_clip"], record["fps"], record["pooling"], record["n_clips"]) == (16, 8.0, "mean", 2)
    assert {"torch", "transformers", "device"} <= set(record)


def test_resampling_changes_the_clip_count(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resampling 8 fps to 4 fps halves the frames, so 32 frames make one clip of 16."""
    monkeypatch.setattr(api, "_load_model", lambda model, device: (*_tiny(), "cpu"))
    result = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=4.0, pooling="mean")
    assert result.embeddings.shape == (1, HIDDEN)


def test_identical_input_gives_identical_output(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two runs on the same input are bit-identical."""
    monkeypatch.setattr(api, "_load_model", lambda model, device: (*_tiny(), "cpu"))
    a = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="mean")
    b = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="mean")
    assert torch.equal(a.embeddings, b.embeddings)


def test_bad_pooling_and_short_video_raise(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown pooling and a video shorter than one clip are refused."""
    monkeypatch.setattr(api, "_load_model", lambda model, device: (*_tiny(), "cpu"))
    with pytest.raises(ValueError, match="pooling"):
        extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="max")
    with pytest.raises(ValueError, match="shorter"):
        extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=64, fps=8.0, pooling="mean")


def test_predictor_is_skipped_for_vjepa2(video: Video, monkeypatch: pytest.MonkeyPatch) -> None:
    """V-JEPA2 is called with `skip_predictor=True`: its predictor never runs, and the embeddings are unchanged."""
    encoder, preprocess = _tiny()
    calls: Dict[str, Any] = {}
    original = encoder.forward

    @functools.wraps(original)
    def spy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        calls.update(kwargs)
        return original(*args, **kwargs)

    encoder.predictor.register_forward_hook(lambda *a: pytest.fail("the predictor ran"))
    monkeypatch.setattr(encoder, "forward", spy)
    monkeypatch.setattr(api, "_load_model", lambda model, device: (encoder, preprocess, "cpu"))
    result = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="mean")
    assert calls["skip_predictor"] is True
    monkeypatch.setattr(api, "_load_model", lambda model, device: (*_tiny(), "cpu"))
    full = extract_ssl_embeddings_from_video(video, _model(SHA), frames_per_clip=16, fps=8.0, pooling="mean")
    assert torch.allclose(result.embeddings, full.embeddings)


class _Stub:
    """Records the revision each `from_pretrained` receives."""

    def __init__(self, kind: str) -> None:
        self.kind = kind

    def to(self, _device: str) -> "_Stub":
        return self

    def eval(self) -> "_Stub":
        return self


def test_load_model_loads_the_resolved_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`_load_model` resolves through `resolve_model` and loads both the model and the processor at that commit."""
    resolved = "b" * 40
    asked: Dict[str, Any] = {}
    seen: Dict[str, str] = {}

    def fake_resolve(repo_id: str, revision: str = "main") -> Tuple[str, Path]:
        asked.update(repo_id=repo_id, revision=revision)
        return resolved, tmp_path

    def loader(kind: str) -> Callable[..., _Stub]:
        def load(name: str, revision: str | None = None, **_: Any) -> _Stub:  # noqa: ANN401
            seen[kind] = f"{name}@{revision}"
            return _Stub(kind)

        return load

    monkeypatch.setattr(api, "resolve_model", fake_resolve)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", loader("model"))
    monkeypatch.setattr(transformers.AutoVideoProcessor, "from_pretrained", loader("processor"))
    api._load_model(_model(SHA), None)
    assert asked == {"repo_id": "facebook/vjepa2-test", "revision": SHA}
    assert seen == {"model": f"facebook/vjepa2-test@{resolved}", "processor": f"facebook/vjepa2-test@{resolved}"}
