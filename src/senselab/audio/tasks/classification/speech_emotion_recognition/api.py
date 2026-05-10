"""Speech emotion recognition API.

Supports discrete (categorical) and continuous (dimensional) SER models.

Models using the ``Wav2Vec2ForSpeechClassification`` architecture (e.g.
``audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`` and community
fine-tunes) are loaded with a custom ``_Wav2Vec2EmotionModel`` class that
matches the saved weight structure (``classifier.dense`` / ``classifier.out_proj``).
Without this, transformers falls back to ``Wav2Vec2ForSequenceClassification``
whose head expects different key names, leaving the regression head randomly
initialized and producing near-0.33 outputs for every input.

Other continuous SER models with broken config.json fields (e.g., vocab_size:
null) are run in an isolated subprocess venv with pinned huggingface-hub<1.0.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from senselab.audio.data_structures import Audio, AudioClassificationResult
from senselab.audio.tasks.classification.huggingface import HuggingFaceAudioClassifier
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    SenselabModel,
    SpeechBrainModel,
    _select_device_and_dtype,
    logger,
)
from senselab.utils.dependencies import speechbrain_loading_cwd, speechbrain_savedir
from senselab.utils.subprocess_venv import ensure_venv, parse_subprocess_result, venv_python

# Exceptions for which falling back to a raw config.json read is sensible. Tests against
# audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim show ``StrictDataclassError`` is
# the actually-encountered case; ``ValueError`` / ``TypeError`` / ``KeyError`` cover the
# adjacent malformed-field scenarios. ``OSError`` is intentionally excluded — that means
# the network / file is broken, and the fallback path needs the same network hop, so it
# can't recover. ``StrictDataclassError`` only exists in huggingface_hub>=1.0.
try:
    from huggingface_hub.errors import StrictDataclassError

    _CONFIG_LOAD_RECOVERABLE: tuple[type[Exception], ...] = (
        StrictDataclassError,
        ValueError,
        TypeError,
        KeyError,
    )
except ImportError:  # pragma: no cover — older huggingface_hub
    _CONFIG_LOAD_RECOVERABLE = (ValueError, TypeError, KeyError)


class SERType(Enum):
    """SER types for determining model output behaviors."""

    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    VALENCE = "valence"
    NOT_RECOGNIZED = "not_recognized"


# Subprocess venv for continuous SER models with broken configs
_CONT_SER_VENV = "continuous-ser"
_CONT_SER_REQUIREMENTS = [
    "transformers>=4.40,<5",
    "huggingface-hub>=0.34,<1.0",  # 1.x strict validation breaks some model configs
    "torch>=2.8",
    "soundfile",
    "numpy",
    "safetensors",
]
_CONT_SER_PYTHON = "3.12"

_CONT_SER_WORKER = r"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import AutoConfig, pipeline

args = json.loads(sys.stdin.read())
audio_paths = args["audio_paths"]
model_id = args["model_id"]
revision = args["revision"]
device = args["device"]
output_dir = args["output_dir"]

pipe = pipeline(
    task="audio-classification",
    model=model_id,
    revision=revision,
    device=device,
    function_to_apply="none",
)

results = []
for audio_path in audio_paths:
    data, sr = sf.read(audio_path, dtype="float32")
    output = pipe({"array": data, "sampling_rate": sr})
    results.append(output)

print(json.dumps({"results": results}))
"""


# ---------------------------------------------------------------------------
# Wav2Vec2ForSpeechClassification backend
#
# Models like audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim were saved
# with a custom Wav2Vec2ForSpeechClassification class (not in standard
# transformers). Their regression head weights are stored as:
#   classifier.dense.{weight,bias}
#   classifier.out_proj.{weight,bias}
#
# When the standard pipeline loads such a model it falls back to
# Wav2Vec2ForSequenceClassification, whose head expects flat
# classifier.{weight,bias} and projector.{weight,bias}. Neither key exists in
# the checkpoint, so the head is randomly initialized → near-0.33 outputs.
#
# _Wav2Vec2EmotionModel below reproduces the original head structure so that
# from_pretrained loads the saved weights correctly.
# ---------------------------------------------------------------------------


# --- Encoder-family registry ----------------------------------------------------
#
# Each supported speech encoder family stores the inner backbone under a different
# attribute name in its ``ForSequenceClassification`` / ``ForSpeechClassification``
# class — e.g. ``self.wav2vec2`` (Wav2Vec2*), ``self.hubert`` (HuBERT),
# ``self.wavlm`` (WavLM). Checkpoint keys mirror that name, so the dynamically-built
# emotion classifier class must use the right attribute or every base-encoder weight
# will look "missing" and the Phase-1 guard will fire.
_BASE_REGISTRY: dict[str, tuple[str, str, str]] = {
    # config.model_type → (PreTrainedModel base class name, encoder model class name, encoder attribute name)
    # Both class names are explicit (not derived) so future families that don't follow
    # the ``XxxPreTrainedModel``/``XxxModel`` naming convention plug in cleanly.
    #
    # NOTE: ``wav2vec2-bert`` is intentionally NOT registered here. Unlike the other three
    # families it consumes log-mel ``input_features`` (via ``SeamlessM4TFeatureExtractor``)
    # rather than raw ``input_values`` (via ``Wav2Vec2FeatureExtractor``); the loader at
    # ``_classify_wav2vec2_speech_cls_ser`` hardcodes the latter, so adding wav2vec2-bert
    # without a matching feature-extractor adapter would silently feed waveforms into a
    # model expecting features. Add a per-family adapter (FE class + input key) before
    # re-introducing it.
    "wav2vec2": ("Wav2Vec2PreTrainedModel", "Wav2Vec2Model", "wav2vec2"),
    "hubert": ("HubertPreTrainedModel", "HubertModel", "hubert"),
    "wavlm": ("WavLMPreTrainedModel", "WavLMModel", "wavlm"),
}


def _resolve_base(model_type: str) -> Optional[tuple[type, type, str]]:
    """Return ``(base_pretrained_cls, encoder_model_cls, encoder_attr_name)`` or ``None``.

    Imports happen lazily so we don't pay the cost on the SpeechBrain or
    standard-pipeline paths that don't need this registry.
    """
    entry = _BASE_REGISTRY.get(model_type)
    if entry is None:
        return None
    base_name, encoder_name, attr_name = entry
    import transformers

    try:
        base_cls = getattr(transformers, base_name)
        encoder_cls = getattr(transformers, encoder_name)
    except AttributeError:  # pragma: no cover — older transformers without this family
        return None
    return base_cls, encoder_cls, attr_name


# --- Head layout descriptor -----------------------------------------------------
#
# Two known head layouts share the same dense+activation+dropout+linear structure
# but use different attribute names for the final layer and may use different
# activations/dropout config fields. ``_HeadEntry`` keeps the shape here so that
# adding a new family is a registry edit rather than a code change.
@dataclass(frozen=True)
class _HeadEntry:
    """2-layer emotion-classifier head descriptor (dense → activation → dropout → final)."""

    final_layer: str = "out_proj"  # classifier.{final_layer}.{weight,bias}
    activation: str = "tanh"  # one of "tanh", "gelu", "relu"
    dropout_field: str = "final_dropout"  # config attribute name controlling dropout_p


_FINAL_LAYER_NAMES = ("out_proj", "output")
_ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "relu": F.relu,
}


def _make_regression_head_class(head: _HeadEntry) -> type:
    fl = head.final_layer
    activation_fn = _ACTIVATIONS[head.activation]
    dropout_field = head.dropout_field

    class _Head(nn.Module):
        def __init__(self, config: PretrainedConfig) -> None:
            super().__init__()
            dropout_p = getattr(config, dropout_field, 0.0) or 0.0
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(dropout_p)
            setattr(self, fl, nn.Linear(config.hidden_size, config.num_labels))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.dropout(x)
            x = self.dense(x)
            x = activation_fn(x)
            x = self.dropout(x)
            return cast(torch.Tensor, getattr(self, fl)(x))

    return _Head


# Cache for built emotion-model classes, keyed by ``(model_type, final_layer, activation, dropout_field)``
# so that ``isinstance()`` checks stay stable across calls.
_emotion_model_classes: dict[tuple[str, str, str, str], type] = {}


def _make_emotion_model_class(model_type: str, head: _HeadEntry) -> type:
    cache_key = (model_type, head.final_layer, head.activation, head.dropout_field)
    cached = _emotion_model_classes.get(cache_key)
    if cached is not None:
        return cached

    resolved = _resolve_base(model_type)
    if resolved is None:
        raise RuntimeError(
            f"No PreTrainedModel base registered for model_type={model_type!r}. "
            f"Add an entry to _BASE_REGISTRY in "
            f"senselab.audio.tasks.classification.speech_emotion_recognition.api."
        )
    base_cls, encoder_cls, attr_name = resolved
    head_cls = _make_regression_head_class(head)

    class _EmotionModel(base_cls):  # type: ignore[valid-type, misc]
        def __init__(self, config: PretrainedConfig) -> None:
            super().__init__(config)
            setattr(self, attr_name, encoder_cls(config))
            self.classifier = head_cls(config)
            # post_init populates all_tied_weights_keys / parallel plans, which the
            # transformers>=5.0 weight-loader expects when finalizing from_pretrained.
            self.post_init()

        def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            encoder = getattr(self, attr_name)
            hidden = encoder(input_values)[0]
            pooled = hidden.mean(dim=1)
            return pooled, self.classifier(pooled)

    _emotion_model_classes[cache_key] = _EmotionModel
    return _EmotionModel


# Backwards-compatible shim used by existing tests that monkeypatch
# ``_make_wav2vec2_emotion_model_class``. New code should call
# ``_make_emotion_model_class(model_type, head)`` directly.
def _make_wav2vec2_emotion_model_class(final_layer: str = "out_proj") -> type:
    return _make_emotion_model_class("wav2vec2", _HeadEntry(final_layer=final_layer))


_wav2vec2_emotion_models: dict = {}


# Some emotion checkpoints ship only ``pytorch_model.bin`` (no safetensors, no shard
# index) — too large to download just to peek at the head keys. For those we hardcode
# the head layout. Every entry must be safe-to-load via ``_classify_wav2vec2_speech_cls_ser``
# with the listed entry.
_KNOWN_HEAD_LAYOUTS: dict[str, _HeadEntry] = {
    "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition": _HeadEntry(final_layer="output"),
}


def _peek_head_final_layer(model: HFModel, encoder_attr: str) -> Optional[str]:
    """Return the head's final-layer attribute name if the checkpoint manifest discloses one.

    Only inspects files whose contents are bounded in size (index manifests or
    safetensors metadata via HTTP range requests). Single-bin checkpoints without
    a shard index are skipped to avoid pulling hundreds of MB just to read keys.
    """
    name = str(model.path_or_uri)

    try:
        from huggingface_hub import HfApi

        files = set(HfApi().list_repo_files(name, revision=model.revision))
    except Exception:
        return None

    keys: list[str] = []
    try:
        from huggingface_hub import hf_hub_download

        if "model.safetensors.index.json" in files:
            p = hf_hub_download(name, "model.safetensors.index.json", revision=model.revision)
            with open(p) as f:
                keys = list(json.load(f).get("weight_map", {}).keys())
        elif "pytorch_model.bin.index.json" in files:
            p = hf_hub_download(name, "pytorch_model.bin.index.json", revision=model.revision)
            with open(p) as f:
                keys = list(json.load(f).get("weight_map", {}).keys())
        elif "model.safetensors" in files:
            from huggingface_hub import get_safetensors_metadata

            meta = get_safetensors_metadata(name, revision=model.revision)
            wm = getattr(meta, "weight_map", None) if meta is not None else None
            keys = list(wm.keys()) if wm else []
    except Exception:
        return None

    head_keys = {k for k in keys if not k.startswith(f"{encoder_attr}.")}
    if "classifier.dense.weight" not in head_keys:
        # No 2-layer dense+linear head signature → the standard pipeline can load this.
        return None
    for fl in _FINAL_LAYER_NAMES:
        if f"classifier.{fl}.weight" in head_keys:
            return fl
    # The checkpoint has the dense layer (a strong custom-head signature) but no recognized
    # final-layer attribute. Falling through to the standard pipeline here would silently
    # random-initialize the head and emit ~uniform softmax — exactly the bug this PR fixes.
    # Refuse instead: surface the actual classifier-key list so the user (or a maintainer)
    # can either add the final-layer name to ``_FINAL_LAYER_NAMES`` or register the
    # checkpoint in ``_KNOWN_HEAD_LAYOUTS``.
    classifier_keys = sorted(k for k in head_keys if k.startswith("classifier."))
    raise RuntimeError(
        f"Detected a custom 2-layer emotion head on {name} (revision={model.revision or 'main'}) "
        f"with classifier keys {classifier_keys}, but its final-layer attribute name is not in "
        f"_FINAL_LAYER_NAMES={_FINAL_LAYER_NAMES!r}. Routing this through the standard "
        f"transformers pipeline would silently random-initialize the head and emit ~uniform "
        f"softmax. Add the final-layer name to _FINAL_LAYER_NAMES, or register the checkpoint "
        f"in _KNOWN_HEAD_LAYOUTS, in "
        f"senselab.audio.tasks.classification.speech_emotion_recognition.api."
    )


# Backwards-compatible alias used by existing tests.
def _peek_wav2vec2_head_final_layer(model: HFModel) -> Optional[str]:
    name = str(model.path_or_uri)
    known = _KNOWN_HEAD_LAYOUTS.get(name)
    if known is not None:
        return known.final_layer
    return _peek_head_final_layer(model, encoder_attr="wav2vec2")


def _emotion_head_kind(model: HFModel) -> Optional[tuple[str, _HeadEntry]]:
    """Identify if the model uses a 2-layer dense+linear emotion head we can load correctly.

    Returns ``(model_type, head_entry)`` if so, ``None`` if the model should fall through
    to the standard transformers pipeline (or a different SER backend).

    Detection rules:
      - Models with ``auto_map`` (custom code on hub) → defer to ``trust_remote_code``.
      - Architectures string contains ``ForSpeechClassification`` (e.g. audeering MSP-Dim
        on Wav2Vec2): use ``out_proj`` head.
      - Architectures string contains ``ForSequenceClassification``: peek the checkpoint
        manifest for ``classifier.dense`` + ``classifier.{out_proj,output}`` keys; if
        found, use the matching head.
      - Repo registered in ``_KNOWN_HEAD_LAYOUTS``: use the registered entry.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
        auto_map = getattr(config, "auto_map", None)
        architectures = getattr(config, "architectures", None) or []
        model_type = getattr(config, "model_type", None)
    except _CONFIG_LOAD_RECOVERABLE:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(str(model.path_or_uri), "config.json", revision=model.revision)
        with open(config_path) as f:
            config_dict = json.load(f)
        auto_map = config_dict.get("auto_map")
        architectures = config_dict.get("architectures") or []
        model_type = config_dict.get("model_type")

    if auto_map:
        return None
    if not model_type or model_type not in _BASE_REGISTRY:
        return None

    repo_id = str(model.path_or_uri)
    if repo_id in _KNOWN_HEAD_LAYOUTS:
        return (model_type, _KNOWN_HEAD_LAYOUTS[repo_id])

    if any("ForSpeechClassification" in a for a in architectures):
        return (model_type, _HeadEntry(final_layer="out_proj"))

    if any("ForSequenceClassification" in a for a in architectures):
        encoder_attr = _BASE_REGISTRY[model_type][2]
        peeked = _peek_head_final_layer(model, encoder_attr=encoder_attr)
        if peeked is not None:
            return (model_type, _HeadEntry(final_layer=peeked))

    return None


# Backwards-compatible alias for existing call sites that only need the final-layer name.
def _wav2vec2_emotion_head_kind(model: HFModel) -> Optional[str]:
    result = _emotion_head_kind(model)
    return result[1].final_layer if result is not None else None


def _resolve_apply_softmax(model: HFModel, ser_type: "SERType") -> bool:
    """Decide whether to softmax raw logits, preferring ``config.problem_type`` when set.

    The HF convention is that classification configs set
    ``problem_type="single_label_classification"`` (or ``multi_label_classification``)
    and regression configs set ``problem_type="regression"``. When that field is
    populated we trust it directly, which is more reliable than the keyword-based
    label heuristic in ``_get_ser_type``: a regression head with non-AVD axes (e.g.
    ``["energy","pleasantness"]``) would otherwise be misclassified as discrete
    and softmax would corrupt its outputs.

    When ``problem_type`` is absent (audeering and any pre-2022 head fall here), peek
    ``architectures``: ``Wav2Vec2ForSpeechClassification`` is the audeering signature
    and is always a regression head. A regression checkpoint that ships neither
    ``problem_type`` nor that architecture string but has labels resembling discrete
    emotions will still be misrouted; ``_FINAL_LAYER_NAMES``-aware ehcalabres-style
    checkpoints declare ``Wav2Vec2ForSequenceClassification`` so the legacy
    keyword-based heuristic remains correct for them.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
        problem_type = getattr(config, "problem_type", None)
        architectures = getattr(config, "architectures", None) or []
    except _CONFIG_LOAD_RECOVERABLE:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(str(model.path_or_uri), "config.json", revision=model.revision)
        with open(config_path) as f:
            config_dict = json.load(f)
        problem_type = config_dict.get("problem_type")
        architectures = config_dict.get("architectures") or []

    if problem_type == "regression":
        return False
    if problem_type in ("single_label_classification", "multi_label_classification"):
        return True
    # No problem_type — fall back to architecture string + label heuristic. Models declaring
    # ``ForSpeechClassification`` (audeering pattern) are always regression heads.
    if any("ForSpeechClassification" in a for a in architectures):
        return False
    # Legacy keyword-based label heuristic for everything else.
    return ser_type != SERType.CONTINUOUS


def classify_emotions_from_speech(
    audios: List[Audio],
    model: SenselabModel,
    device: Optional[DeviceType] = None,
    **kwargs: Any,  # noqa: ANN401
) -> List[AudioClassificationResult]:
    """Classify all audios using the given speech emotion recognition model.

    Args:
        audios: The list of audio objects to be classified.
        model: The model used for classification, should be trained for recognizing emotions.
        device: The device to run the model on (default is None).
        **kwargs: Additional keyword arguments to pass to the classification function.

    Returns:
        List of speech emotion recognition results.
    """
    if isinstance(model, SpeechBrainModel):
        return _classify_speechbrain_ser(audios, model, device)

    if isinstance(model, HFModel):
        model_info = model.get_model_info()
        tags = model_info.tags or []

        ser_type = _get_ser_type(model)

        if (
            not ("speech-emotion-recognition" in tags or "emotion-recognition" in tags)
            and ser_type == SERType.NOT_RECOGNIZED
        ):
            raise ValueError(
                f"The model '{model.path_or_uri}' is not suitable for speech emotion recognition. Please "
                "validate that it has the correct tags or use the more generic "
                "'audio_classification_with_hf_models' function."
            )

        # Wav2Vec2 / HuBERT / WavLM emotion checkpoints with a 2-layer
        # dense+linear head (continuous audeering-style ``out_proj`` or discrete
        # ehcalabres-style ``output``) would be silently random-headed by the
        # standard transformers pipeline. Route through the in-process custom path,
        # picking softmax based on ``problem_type`` (Phase 5) when present.
        head_match = _emotion_head_kind(model)
        if head_match is not None:
            head_model_type, head_entry = head_match
            apply_softmax = _resolve_apply_softmax(model, ser_type)
            return _classify_wav2vec2_speech_cls_ser(
                audios,
                model,
                device,
                model_type=head_model_type,
                head=head_entry,
                apply_softmax=apply_softmax,
            )

        if ser_type == SERType.CONTINUOUS:
            return _classify_continuous_ser_venv(audios, model, device)
        return HuggingFaceAudioClassifier.classify_audios_with_transformers(
            audios=audios, model=model, device=device, **kwargs
        )

    raise NotImplementedError(
        "Only Hugging Face and SpeechBrain models are supported. Pass an HFModel or SpeechBrainModel instance."
    )


def _classify_continuous_ser_venv(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
) -> List[AudioClassificationResult]:
    """Run continuous SER in an isolated subprocess venv.

    Some continuous SER models have broken config.json (e.g., vocab_size: null)
    that huggingface-hub >=1.0 strict validation rejects. This runs them in a
    venv with pinned huggingface-hub<1.0.
    """
    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    venv_dir = ensure_venv(_CONT_SER_VENV, _CONT_SER_REQUIREMENTS, python_version=_CONT_SER_PYTHON)
    python = venv_python(venv_dir)

    with tempfile.TemporaryDirectory(prefix="senselab-ser-") as tmpdir:
        tmp = Path(tmpdir)

        audio_paths = []
        for i, audio in enumerate(audios):
            path = str(tmp / f"audio_{i}.flac")
            audio.save_to_file(path, format="flac")
            audio_paths.append(path)

        input_json = json.dumps(
            {
                "audio_paths": audio_paths,
                "model_id": str(model.path_or_uri),
                "revision": model.revision or "main",
                "device": device_type.value,
                "output_dir": str(tmp),
            }
        )

        result = subprocess.run(
            [python, "-c", _CONT_SER_WORKER],
            input=input_json,
            capture_output=True,
            text=True,
            timeout=600,
        )

        output = parse_subprocess_result(result, "Continuous SER")

        results = []
        for classification_list in output.get("results", []):
            labels = [c["label"] for c in classification_list]
            scores = [float(c["score"]) for c in classification_list]
            results.append(AudioClassificationResult(labels=labels, scores=scores))

        return results


def _classify_wav2vec2_speech_cls_ser(
    audios: List[Audio],
    model: HFModel,
    device: Optional[DeviceType] = None,
    *,
    model_type: str = "wav2vec2",
    head: Optional[_HeadEntry] = None,
    final_layer: Optional[str] = None,  # legacy kwarg; if set, takes precedence over `head`
    apply_softmax: bool = False,
) -> List[AudioClassificationResult]:
    """Run inference for Wav2Vec2/HuBERT/WavLM emotion checkpoints with a 2-layer head.

    Builds a model whose ``classifier`` matches the saved weight names —
    ``classifier.dense.*`` plus ``classifier.{head.final_layer}.*`` — so the head is
    loaded correctly rather than randomly initialized. Set ``apply_softmax`` for
    discrete-emotion checkpoints to return per-class probabilities; leave it
    False for continuous (arousal/valence/dominance) regression heads.

    The ``final_layer`` kwarg is retained for backwards compatibility with the
    pre-Phase-3 API and short-circuits to ``_HeadEntry(final_layer=...)``.
    """
    from transformers import AutoConfig, Wav2Vec2FeatureExtractor

    if final_layer is not None and head is None:
        head = _HeadEntry(final_layer=final_layer)
    if head is None:
        head = _HeadEntry()  # default: out_proj/tanh/final_dropout (audeering pattern)

    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    key = f"{model.path_or_uri}-{model.revision or 'main'}-{device_type.value}-{model_type}-{head.final_layer}"
    if key not in _wav2vec2_emotion_models:
        EmotionModel = _make_emotion_model_class(model_type, head)
        try:
            config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
        except _CONFIG_LOAD_RECOVERABLE:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(str(model.path_or_uri), "config.json", revision=model.revision)
            with open(config_path) as f:
                config_dict = json.load(f)
            # config.json carries its own "model_type" key, which would collide with the positional
            # arg to AutoConfig.for_model(...). Drop it.
            config_dict.pop("model_type", None)
            # Some checkpoints (e.g. audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim) ship
            # vocab_size: null, which trips huggingface_hub>=1.0 strict-dataclass validators.
            # The regression head doesn't use vocab_size, so let the Config class supply its default.
            if config_dict.get("vocab_size") is None:
                config_dict.pop("vocab_size", None)
            config = AutoConfig.for_model(model_type, **config_dict)

        loaded, loading_info = EmotionModel.from_pretrained(  # type: ignore[attr-defined]
            str(model.path_or_uri),
            revision=model.revision,
            config=config,
            output_loading_info=True,
        )
        # Refuse to silently emit random-head outputs. The guard inspects ``missing_keys``
        # (weight present in the model but absent from the checkpoint → randomly initialized)
        # and ``mismatched_keys`` (weight present but shape-incompatible → also random-init
        # for the affected layer). The check covers BOTH the classifier and the encoder
        # backbone: a bad ``_BASE_REGISTRY`` entry (wrong encoder attribute name for the
        # model_type) would otherwise leave every encoder weight missing while the
        # classifier loads cleanly — silently producing meaningless output. The whitelist
        # below is the small set of buffers HF post_init creates that aren't in checkpoints
        # (e.g. SpecAugment's ``masked_spec_embed``); add to it only with evidence.
        encoder_attr = _BASE_REGISTRY[model_type][2]
        legitimately_missing_suffixes = (".masked_spec_embed",)

        def _is_suspect(key: str) -> bool:
            if key.startswith("classifier."):
                return True
            if key.startswith(f"{encoder_attr}.") and not any(key.endswith(s) for s in legitimately_missing_suffixes):
                return True
            return False

        suspect_missing = sorted(k for k in loading_info.get("missing_keys", set()) if _is_suspect(k))
        suspect_mismatched = sorted(
            (k[0] if isinstance(k, tuple) else k)
            for k in loading_info.get("mismatched_keys", set())
            if _is_suspect(k[0] if isinstance(k, tuple) else k)
        )
        if suspect_missing or suspect_mismatched:
            raise RuntimeError(
                "Custom emotion head failed to load cleanly: "
                f"missing keys (would be randomly initialized)={suspect_missing}, "
                f"shape-mismatched keys={suspect_mismatched}. The output would be "
                f"non-informative. If the missing keys are inside ``classifier.``, add the "
                f"checkpoint's head layout to _KNOWN_HEAD_LAYOUTS / _FINAL_LAYER_NAMES in "
                f"senselab.audio.tasks.classification.speech_emotion_recognition.api, or run "
                f"the model through the standard transformers pipeline if it has a flat head. "
                f"If they are inside ``{encoder_attr}.``, the _BASE_REGISTRY entry for "
                f"model_type={model_type!r} is wrong — fix the (base_cls, encoder_cls, attr) "
                f"tuple. "
                f"Model: {model.path_or_uri} (revision={model.revision or 'main'}), head={head}."
            )
        # Shape-sanity: the final layer's output dimension must match config.num_labels
        # (or len(id2label) when num_labels is absent). A mismatch here means the head
        # loaded but is mis-sized — every result this run would have wrong-length scores.
        final_module = getattr(loaded.classifier, head.final_layer, None)
        actual_out = getattr(final_module, "out_features", None) if final_module is not None else None
        expected_out = int(getattr(config, "num_labels", 0) or len(getattr(config, "id2label", None) or {}))
        if actual_out is not None and expected_out and actual_out != expected_out:
            raise RuntimeError(
                f"Custom emotion head final layer has out_features={actual_out} "
                f"but config declares num_labels={expected_out}. Refusing to run inference. "
                f"Model: {model.path_or_uri} (revision={model.revision or 'main'}), "
                f"model_type={model_type}."
            )
        loaded = loaded.to(device_type.value)
        loaded.eval()
        # Use the feature extractor directly: Wav2Vec2Processor.from_pretrained re-runs
        # AutoConfig.from_pretrained internally (to pick a tokenizer), which would re-trip
        # the vocab_size: null strict-validator on these checkpoints. Regression inference
        # doesn't need a tokenizer.
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(str(model.path_or_uri), revision=model.revision)
        # Resolve labels once: HF stores id2label keys as strings in JSON; AutoConfig usually
        # coerces them to int but the raw-dict fallback above does not. Tolerate either.
        raw_id2label = getattr(config, "id2label", None) or {}
        coerced: dict = {}
        for k, v in raw_id2label.items():
            try:
                coerced[int(k)] = v
            except (TypeError, ValueError):
                continue
        n_labels = int(getattr(config, "num_labels", len(coerced)) or len(coerced))
        labels: List[str] = [coerced.get(i, f"class_{i}") for i in range(n_labels)]
        expected_sr = int(getattr(feature_extractor, "sampling_rate", 16000) or 16000)
        _wav2vec2_emotion_models[key] = (loaded, feature_extractor, labels, expected_sr)

    loaded_model, feature_extractor, labels, expected_sr = _wav2vec2_emotion_models[key]

    results: List[AudioClassificationResult] = []
    for audio in audios:
        if audio.waveform.shape[0] != 1:
            raise ValueError("Only mono audio is supported for this Wav2Vec2 emotion model.")
        if audio.sampling_rate != expected_sr:
            audio = resample_audios([audio], resample_rate=expected_sr)[0]
        data = audio.waveform.squeeze().numpy()
        inputs = feature_extractor(data, sampling_rate=expected_sr, return_tensors="pt", padding=True)
        inputs = {k: v.to(device_type.value) for k, v in inputs.items()}
        with torch.no_grad():
            _, logits = loaded_model(inputs["input_values"])
        if apply_softmax:
            scores = torch.softmax(logits[0], dim=-1).cpu().tolist()
        else:
            scores = logits[0].cpu().tolist()
        # Reconcile label/score lengths defensively: a config/head mismatch can leave
        # `labels` longer or shorter than the actual model output.
        result_labels = labels[: len(scores)]
        if len(result_labels) < len(scores):
            result_labels = result_labels + [f"class_{i}" for i in range(len(result_labels), len(scores))]
        results.append(AudioClassificationResult(labels=result_labels, scores=scores))

    return results


def _get_ser_type(model: HFModel) -> SERType:
    """Get the type of SER the model is likely used for based on the labels it is set to predict."""
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model.path_or_uri, revision=model.revision)
    except _CONFIG_LOAD_RECOVERABLE:
        # Fall back to raw config dict for models with invalid fields
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(str(model.path_or_uri), "config.json", revision=model.revision)
        with open(config_path) as f:
            config_dict = json.load(f)
        config = SimpleNamespace(id2label=config_dict.get("id2label", {}))
    id2label = config.id2label
    if id2label:
        labels = list(id2label.values())
        if "positive" in labels and "negative" in labels and "neutral" in labels:
            return SERType.VALENCE
        if "valence" in labels or "arousal" in labels or "dominance" in labels:
            return SERType.CONTINUOUS
        if (
            "happy" in labels
            or "happiness" in labels
            or "joy" in labels
            or "angry" in labels
            or "anger" in labels
            or "sad" in labels
            or "sadness" in labels
            or "fear" in labels
            or "disgust" in labels
            or "surprise" in labels
        ):
            return SERType.DISCRETE
    return SERType.NOT_RECOGNIZED


# ---------------------------------------------------------------------------
# SpeechBrain SER backend
# ---------------------------------------------------------------------------

# Cache for loaded SpeechBrain SER models
_speechbrain_ser_models: dict = {}


def _classify_speechbrain_ser(
    audios: List[Audio],
    model: SpeechBrainModel,
    device: Optional[DeviceType] = None,
) -> List[AudioClassificationResult]:
    """Classify emotions using a SpeechBrain model.

    SpeechBrain SER models (e.g., ``speechbrain/emotion-recognition-wav2vec2-IEMOCAP``)
    define custom module names (``wav2vec2``, ``avg_pool``, ``output_mlp``) that
    don't match the generic ``EncoderClassifier`` expectations. This function
    loads them via a lightweight ``Pretrained`` subclass.

    Args:
        audios: Audio objects (must be mono, 16 kHz).
        model: A SpeechBrainModel pointing to a HuggingFace repo.
        device: Device to run on.

    Returns:
        List of AudioClassificationResult, one per audio.
    """
    from speechbrain.inference import Pretrained

    device_type, _ = _select_device_and_dtype(
        user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
    )

    key = f"{model.path_or_uri}-{model.revision or 'main'}-{device_type.value}"
    if key not in _speechbrain_ser_models:
        # Discover required modules from the model's hyperparams
        _speechbrain_ser_models[key] = _load_speechbrain_ser_model(model, device_type)

    recognizer, label_list = _speechbrain_ser_models[key]

    if not audios:
        return []

    results: List[AudioClassificationResult] = []
    for audio in audios:
        if audio.waveform.shape[0] != 1:
            raise ValueError("Only mono audio is supported for SpeechBrain SER.")

        wavs = audio.waveform.to(recognizer.device).float()
        wav_lens = torch.ones(wavs.shape[0], device=recognizer.device)

        with torch.no_grad():
            # Walk through the model's modules in order
            out = wavs
            for mod_name in recognizer.MODULES_NEEDED:
                mod = getattr(recognizer.mods, mod_name)
                if mod_name == recognizer.MODULES_NEEDED[0]:
                    # First module (feature extractor) gets wavs + lens
                    out = mod(wavs, wav_lens)
                elif "pool" in mod_name:
                    out = mod(out, wav_lens)
                    out = out.view(out.shape[0], -1)
                else:
                    out = mod(out)

            probs = torch.nn.functional.softmax(out, dim=-1)

        prob_list = probs[0].cpu().tolist()
        results.append(AudioClassificationResult(labels=label_list, scores=prob_list))

    return results


def _load_speechbrain_ser_model(model: SpeechBrainModel, device_type: DeviceType) -> tuple:
    """Load a SpeechBrain SER model and discover its structure.

    Returns:
        Tuple of (recognizer, label_list).
    """
    import yaml  # type: ignore[import-untyped]
    from huggingface_hub import hf_hub_download
    from speechbrain.inference import Pretrained

    # Download hyperparams to discover MODULES_NEEDED
    hp_path = hf_hub_download(str(model.path_or_uri), "hyperparams.yaml", revision=model.revision)

    # SpeechBrain YAML uses custom tags (!new:, !ref, etc.) that yaml.safe_load
    # cannot handle. Use a permissive loader subclass that ignores unknown tags.
    class _PermissiveLoader(yaml.SafeLoader):
        pass

    _PermissiveLoader.add_multi_constructor("", lambda loader, suffix, node: None)
    with open(hp_path) as f:
        hparams = yaml.load(f, Loader=_PermissiveLoader)  # noqa: S506

    modules_needed = hparams.get("MODULES_NEEDED") if hparams else None

    if not modules_needed:
        raise ValueError(
            f"Could not determine MODULES_NEEDED from {model.path_or_uri}/hyperparams.yaml. "
            "This SpeechBrain model may need a custom inference class."
        )

    # Create a dynamic Pretrained subclass with the correct modules
    recognizer_cls = type(
        "SpeechBrainSER",
        (Pretrained,),
        {"MODULES_NEEDED": modules_needed},
    )

    run_opts = {"device": device_type.value}
    savedir = speechbrain_savedir(str(model.path_or_uri), model.revision)
    with speechbrain_loading_cwd(savedir):
        recognizer = recognizer_cls.from_hparams(  # type: ignore[attr-defined]
            source=str(model.path_or_uri),
            savedir=str(savedir),
            run_opts=run_opts,
        )

    # Extract label names from label_encoder if available
    label_list: List[str] = []
    if hasattr(recognizer.hparams, "label_encoder"):
        le = recognizer.hparams.label_encoder
        if hasattr(le, "ind2lab"):
            n_classes = len(le.ind2lab)
            label_list = [le.ind2lab.get(i, f"class_{i}") for i in range(n_classes)]

    if not label_list:
        logger.warning("Could not extract label names from SpeechBrain model; using generic indices.")

    return recognizer, label_list
