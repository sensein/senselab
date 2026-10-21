"""In-process Hugging Face sequence-classification backend for text sentiment and emotion."""

from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from senselab.text.tasks.sentiment_analysis.data_structures import TextClassificationResult
from senselab.utils.data_structures import DeviceType, HFModel, _select_device_and_dtype
from senselab.utils.dependencies import resolve_model

LongText = Literal["truncate", "window"]

_COMPATIBLE_DEVICES = [DeviceType.CUDA, DeviceType.MPS, DeviceType.CPU]


class TextClassifierFactory:
    """Loads and caches sequence-classification models keyed by (model, commit, device)."""

    _tokenizers: Dict[Tuple[str, str], PreTrainedTokenizerBase] = {}
    _models: Dict[Tuple[str, str, str], PreTrainedModel] = {}

    @classmethod
    def _resolve(cls, model: HFModel) -> str:
        """Return the commit SHA this model loads from."""
        if model.commit_sha:
            return model.commit_sha
        sha, _ = resolve_model(str(model.path_or_uri), model.revision or "main")
        return sha

    @classmethod
    def get(cls, model: HFModel, device: DeviceType) -> Tuple[PreTrainedTokenizerBase, PreTrainedModel, str]:
        """Return ``(tokenizer, model, commit_sha)``, loading each at most once per key."""
        repo = str(model.path_or_uri)
        sha = cls._resolve(model)
        if (repo, sha) not in cls._tokenizers:
            cls._tokenizers[(repo, sha)] = AutoTokenizer.from_pretrained(repo, revision=sha)
        key = (repo, sha, device.value)
        if key not in cls._models:
            loaded = AutoModelForSequenceClassification.from_pretrained(repo, revision=sha).eval()
            cls._models[key] = loaded.to(device.value)
        return cls._tokenizers[(repo, sha)], cls._models[key], sha


def _label_names(clf: PreTrainedModel) -> List[str]:
    """Labels in model-index order."""
    id2label = {int(k): str(v) for k, v in (clf.config.id2label or {}).items()}
    return [id2label[i] for i in range(len(id2label))]


def _special_tokens(tokenizer: PreTrainedTokenizerBase) -> Tuple[List[int], List[int]]:
    """Ids the tokenizer adds before and after a single sequence, found by probing."""
    plain = tokenizer("a", add_special_tokens=False)["input_ids"]
    full = tokenizer("a", add_special_tokens=True)["input_ids"]
    start = next(i for i in range(len(full) - len(plain) + 1) if full[i : i + len(plain)] == plain)
    return full[:start], full[start + len(plain) :]


def _encode(
    tokenizer: PreTrainedTokenizerBase, text: str, long_text: LongText, window_stride: int
) -> Tuple[List[List[int]], bool]:
    """Token-id windows (with special tokens) for one non-empty text, and whether text was dropped."""
    prefix, suffix = _special_tokens(tokenizer)
    max_len = int(min(tokenizer.model_max_length, 512))
    body = max_len - len(prefix) - len(suffix)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= body:
        return [prefix + ids + suffix], False
    if long_text == "truncate":
        return [prefix + ids[:body] + suffix], True
    step = max(1, body - window_stride)
    windows = []
    for start in range(0, len(ids), step):
        windows.append(prefix + ids[start : start + body] + suffix)
        if start + body >= len(ids):
            break
    return windows, False


def classify_texts(
    texts: Sequence[str],
    model: HFModel,
    device: Optional[DeviceType],
    batch_size: int,
    long_text: LongText,
    window_stride: int,
) -> List[TextClassificationResult]:
    """Classify texts, returning one full-distribution result per input in input order."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if long_text not in ("truncate", "window"):
        raise ValueError("long_text must be 'truncate' or 'window'")
    dev, _ = _select_device_and_dtype(user_preference=device, compatible_devices=_COMPATIBLE_DEVICES)
    tokenizer, clf, sha = TextClassifierFactory.get(model, dev)
    labels = _label_names(clf)
    model_id = str(model.path_or_uri)

    # One entry per (text index, window); sorted by length so batches pad little.
    pieces: List[Tuple[int, List[int]]] = []
    truncated: Dict[int, bool] = {}
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        windows, cut = _encode(tokenizer, text, long_text, window_stride)
        truncated[i] = cut
        pieces.extend((i, w) for w in windows)
    order = sorted(range(len(pieces)), key=lambda k: len(pieces[k][1]))

    probs: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        batch = tokenizer.pad({"input_ids": [pieces[k][1] for k in idx]}, return_tensors="pt")
        inputs = {k: v.to(dev.value) for k, v in batch.items()}
        with torch.no_grad():
            p = torch.softmax(clf(**inputs).logits.float(), dim=-1).cpu()
        for row, k in enumerate(idx):
            text_i, ids = pieces[k]
            probs.setdefault(text_i, []).append((p[row], len(ids)))

    results: List[TextClassificationResult] = []
    for i, text in enumerate(texts):
        if i not in probs:
            results.append(
                TextClassificationResult(text=text, scores={}, n_windows=0, model_id=model_id, commit_sha=sha)
            )
            continue
        parts = probs[i]
        weights = torch.tensor([float(n) for _, n in parts])
        mean = (torch.stack([p for p, _ in parts]) * weights[:, None]).sum(0) / weights.sum()
        scores = {name: float(mean[j]) for j, name in enumerate(labels)}
        top = max(scores, key=lambda n: scores[n])
        results.append(
            TextClassificationResult(
                text=text,
                scores=scores,
                label=top,
                confidence=scores[top],
                n_windows=len(parts),
                truncated=truncated[i],
                model_id=model_id,
                commit_sha=sha,
            )
        )
    return results
