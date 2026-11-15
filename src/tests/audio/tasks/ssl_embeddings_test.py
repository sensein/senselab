"""Tests for HF SSL embeddings: a tiny random wav2vec2 and a tiny random Whisper on CPU (no weights, no network).

The Whisper tests use the published `openai/whisper-large-v3` feature extractor config (vendored JSON).
"""

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperModel,
)

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.ssl_embeddings import extract_ssl_embeddings_from_audios
from senselab.audio.tasks.ssl_embeddings import self_supervised_features as ssf
from senselab.utils.data_structures import DeviceType, HFModel

DATA = Path(__file__).parent / "data"
WHISPER_SHA = "06f233fe06e710322aca913c1bc4249a0d71fce1"
SR = 16000
HIDDEN = 16
LAYERS = 2


def _model(name: str, revision: str) -> HFModel:
    """An HFModel built without the Hub check (model_construct skips the validators)."""
    return HFModel.model_construct(path_or_uri=name, revision=revision)


def _audio(seconds: float, seed: int = 0) -> Audio:
    gen = torch.Generator().manual_seed(seed)
    return Audio(waveform=torch.randn(1, int(seconds * SR), generator=gen) * 0.1, sampling_rate=SR)


def _whisper_parts() -> Dict[str, Any]:
    """The real whisper-large-v3 feature extractor and a tiny random WhisperModel with its mel and position sizes."""
    real = json.loads((DATA / "whisper_large_v3_config.json").read_text())
    torch.manual_seed(0)
    config = WhisperConfig(
        vocab_size=64, num_mel_bins=real["num_mel_bins"], max_source_positions=real["max_source_positions"],
        max_target_positions=32, pad_token_id=0, bos_token_id=1, eos_token_id=2, decoder_start_token_id=1,
        d_model=HIDDEN, encoder_layers=LAYERS, decoder_layers=1, encoder_attention_heads=2, decoder_attention_heads=2,
        encoder_ffn_dim=32, decoder_ffn_dim=32,
    )  # fmt: skip
    extractor = WhisperFeatureExtractor.from_json_file(str(DATA / "whisper_large_v3_preprocessor_config.json"))
    return {"extractor": extractor, "model": WhisperModel(config).eval()}


def _wav2vec2_parts() -> Dict[str, Any]:
    torch.manual_seed(0)
    config = Wav2Vec2Config(
        hidden_size=HIDDEN, num_hidden_layers=LAYERS, num_attention_heads=2, intermediate_size=32,
        conv_dim=(16, 16), conv_stride=(5, 4), conv_kernel=(10, 4), num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )  # fmt: skip
    extractor = Wav2Vec2FeatureExtractor(sampling_rate=SR, feature_size=1, return_attention_mask=False)
    return {"extractor": extractor, "model": Wav2Vec2Model(config).eval()}


@pytest.fixture()
def loads(monkeypatch: pytest.MonkeyPatch) -> Dict[str, List[Any]]:
    """Empty the loader caches and record what resolve_model and the two from_pretrained calls are asked for."""
    monkeypatch.setattr(ssf.SSLEmbeddingsFactory, "_feat_extractor", {})
    monkeypatch.setattr(ssf.SSLEmbeddingsFactory, "_model", {})
    calls: Dict[str, List[Any]] = {"resolve": [], "extractor": [], "model": []}
    return calls


def _install(monkeypatch: pytest.MonkeyPatch, calls: Dict[str, List[Any]], parts: Dict[str, Any]) -> None:
    def resolve(repo_id: str, revision: str = "main", **_: Any) -> Any:  # noqa: ANN401
        calls["resolve"].append((repo_id, revision))
        return (revision if len(revision) == 40 else "f" * 40), Path("/nonexistent")

    class _Extractor:
        @staticmethod
        def from_pretrained(name: str, revision: str, cache_dir: Any = None) -> Any:  # noqa: ANN401
            calls["extractor"].append((name, revision))
            return parts["extractor"]

    class _Model:
        @staticmethod
        def from_pretrained(name: str, revision: str, cache_dir: Any = None) -> Any:  # noqa: ANN401
            calls["model"].append((name, revision))
            return parts["model"]

    monkeypatch.setattr(ssf, "resolve_model", resolve)
    monkeypatch.setattr(ssf, "AutoFeatureExtractor", _Extractor)
    monkeypatch.setattr(ssf, "AutoModel", _Model)


def test_wav2vec2_path_returns_all_hidden_states(monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]) -> None:
    """The input_values path is unchanged: a ref revision is accepted and the shape is (layers + 1, frames, hidden)."""
    _install(monkeypatch, loads, _wav2vec2_parts())
    out = extract_ssl_embeddings_from_audios([_audio(1.0)], _model("org/w2v-test", "main"), device=DeviceType.CPU)
    assert len(out) == 1 and out[0].shape[0] == LAYERS + 1 and out[0].shape[2] == HIDDEN
    assert out[0].shape[1] == 1.0 * SR // 20 - 1  # strides 5 * 4, kernels 10 and 4
    assert loads["resolve"] == [("org/w2v-test", "main")] * 2


def test_whisper_short_audio_is_one_window(monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]) -> None:
    """5 s gives one 30 s window cut to the audio it holds: 250 frames of 20 ms."""
    _install(monkeypatch, loads, _whisper_parts())
    out = extract_ssl_embeddings_from_audios(
        [_audio(5.0)], _model("org/whisper-test", WHISPER_SHA), device=DeviceType.CPU
    )
    assert out[0].shape == (LAYERS + 1, 250, HIDDEN)


def test_whisper_long_audio_is_chunked_into_30s_windows(
    monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]
) -> None:
    """70 s gives windows of 30, 30 and 10 s (1500 + 1500 + 500 frames), each encoded on its own."""
    _install(monkeypatch, loads, _whisper_parts())
    model = _model("org/whisper-test", WHISPER_SHA)
    long = _audio(70.0, seed=3)
    out = extract_ssl_embeddings_from_audios([long], model, device=DeviceType.CPU)[0]
    assert out.shape == (LAYERS + 1, 3500, HIDDEN)
    first = Audio(waveform=long.waveform[:, : 30 * SR], sampling_rate=SR)
    third = Audio(waveform=long.waveform[:, 60 * SR :], sampling_rate=SR)
    first_out, third_out = extract_ssl_embeddings_from_audios([first, third], model, device=DeviceType.CPU)
    assert torch.allclose(out[:, :1500], first_out, atol=1e-5)
    assert torch.allclose(out[:, 3000:], third_out, atol=1e-5)


def test_whisper_loads_the_given_commit(monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]) -> None:
    """The extractor and the model are resolved and loaded at the 40-hex commit given."""
    _install(monkeypatch, loads, _whisper_parts())
    extract_ssl_embeddings_from_audios([_audio(1.0)], _model("org/whisper-test", WHISPER_SHA), device=DeviceType.CPU)
    assert loads["resolve"] == [("org/whisper-test", WHISPER_SHA)] * 2
    assert loads["extractor"] == loads["model"] == [("org/whisper-test", WHISPER_SHA)]


def test_whisper_refuses_a_non_hex_revision(monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]) -> None:
    """A ref name, a short hash and an upper-case hash are refused before the model weights load."""
    _install(monkeypatch, loads, _whisper_parts())
    for bad in ("main", WHISPER_SHA[:8], WHISPER_SHA.upper()):
        with pytest.raises(ValueError, match="40-hex"):
            extract_ssl_embeddings_from_audios([_audio(1.0)], _model("org/whisper-test", bad), device=DeviceType.CPU)
    assert loads["model"] == []


def test_whisper_refuses_empty_audio(monkeypatch: pytest.MonkeyPatch, loads: Dict[str, List]) -> None:
    """An audio with no samples has no window to encode."""
    _install(monkeypatch, loads, _whisper_parts())
    empty = Audio(waveform=torch.zeros(1, 0), sampling_rate=SR)
    with pytest.raises(ValueError, match="no samples"):
        extract_ssl_embeddings_from_audios([empty], _model("org/whisper-test", WHISPER_SHA), device=DeviceType.CPU)
