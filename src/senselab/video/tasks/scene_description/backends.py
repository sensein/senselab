"""Vision-language model backends that turn (image, prompt) requests into completions.

``MLXVLMBackend`` serves ``mlx-community/Qwen3-VL-4B-Instruct-4bit`` from a long-lived worker in
the ``mlx-vlm`` subprocess venv (Apple silicon). ``TransformersVLMBackend`` runs
``Qwen/Qwen3-VL-4B-Instruct`` in process with transformers (CUDA or CPU). Both load a model once
and keep it for every later request, across frames and across videos.
"""

from __future__ import annotations

import platform
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from senselab.utils.dependencies import hf_subprocess_env
from senselab.utils.subprocess_venv import _clean_subprocess_env, ensure_venv, venv_python
from senselab.utils.venv_worker import serve_in_venv, shutdown_venv_workers

DEFAULT_MLX_MODEL = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
DEFAULT_TRANSFORMERS_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

_MLX_VENV = "mlx-vlm"
_MLX_REQUIREMENTS = ["mlx-vlm==0.7.8"]
_MLX_PYTHON = "3.12"
_MLX_LOAD_TIMEOUT_S = 1800
_MLX_REQUEST_TIMEOUT_S = 3600
_MLX_WORKER_SCRIPT = r"""
import resource
import sys
import time


def load(init):
    # Loads once, from the staged snapshots/<sha> directory.
    from mlx_vlm import load as mlx_load
    from mlx_vlm.utils import load_config

    model, processor = mlx_load(init["model_path"])
    config = load_config(init["model_path"])
    return {"model": model, "processor": processor, "config": config}


def _peak_memory_bytes():
    import mlx.core as mx

    getter = getattr(mx, "get_peak_memory", None) or mx.metal.get_peak_memory
    return int(getter())


def _rss_peak_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def handle(state, request):
    # One completion per item {"image_path": str | None, "prompt": str}.
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    results = []
    for item in request["items"]:
        image_path = item.get("image_path")
        images = [image_path] if image_path else []
        prompt = apply_chat_template(state["processor"], state["config"], item["prompt"], num_images=len(images))
        start = time.perf_counter()
        out = generate(
            state["model"],
            state["processor"],
            prompt,
            image=images or None,
            max_tokens=int(request.get("max_tokens", 1024)),
            temperature=float(request.get("temperature", 0.0)),
            verbose=False,
        )
        seconds = time.perf_counter() - start
        text = getattr(out, "text", out)
        results.append(
            {
                "text": text,
                "seconds": seconds,
                "prompt_tokens": int(getattr(out, "prompt_tokens", 0) or 0),
                "generation_tokens": int(getattr(out, "generation_tokens", 0) or 0),
            }
        )
    return {
        "results": results,
        "peak_memory_bytes": _peak_memory_bytes(),
        "peak_rss_bytes": _rss_peak_bytes(),
    }
"""


@dataclass
class GenerationRequest:
    """One prompt, with or without an image.

    Attributes:
        prompt: The text prompt.
        image: RGB ``uint8`` array, or None for a text-only request.
    """

    prompt: str
    image: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class Generation:
    """One completion.

    Attributes:
        text: The generated text.
        seconds: Wall-clock generation time.
        prompt_tokens: Prompt tokens, image tokens included (0 when the backend does not report it).
        generation_tokens: Generated tokens (0 when the backend does not report it).
    """

    text: str
    seconds: float = 0.0
    prompt_tokens: int = 0
    generation_tokens: int = 0


class VLMBackend(Protocol):
    """What :func:`describe_frames` needs from a model."""

    model_id: str
    revision: str

    def generate(self, requests: Sequence[GenerationRequest]) -> List[Generation]:
        """Return one completion per request, in order."""
        ...

    def stats(self) -> Dict[str, Any]:
        """Return resource figures (peak memory) recorded so far."""
        ...


def is_apple_silicon() -> bool:
    """True on macOS arm64, where the MLX backend runs."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _save_png(image: np.ndarray, path: Path) -> None:
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(image.astype(np.uint8))).save(path)


class MLXVLMBackend:
    """Qwen3-VL through ``mlx-vlm`` in a long-lived subprocess-venv worker.

    The model is resolved to a commit SHA and staged in the parent; the worker loads the
    ``snapshots/<sha>`` directory, so the weights it serves are the ones ``revision`` names.
    Workers are pooled by identity, so every instance with the same model and revision shares one
    resident model.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MLX_MODEL,
        revision: str = "main",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> None:
        """Resolve the model's commit and prepare the worker identity.

        Args:
            model_id: Hugging Face repo of an MLX-converted VLM.
            revision: Ref or commit SHA; resolved to a SHA before anything is loaded.
            max_tokens: Generation cap per completion.
            temperature: Sampling temperature; 0 is greedy.
        """
        from senselab_core.runtime.revision import resolve_revision

        from senselab.utils.dependencies import resolve_model

        self.model_id = model_id
        self.revision = resolve_revision(model_id, revision)
        _, snapshot = resolve_model(model_id, self.revision)
        self._snapshot = str(snapshot)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._stats: Dict[str, Any] = {"peak_memory_bytes": 0, "peak_rss_bytes": 0}

    def generate(self, requests: Sequence[GenerationRequest]) -> List[Generation]:
        """Generate one completion per request in the resident worker.

        Args:
            requests: The prompts and images.

        Returns:
            One ``Generation`` per request, in order.
        """
        if not requests:
            return []
        venv_dir = ensure_venv(_MLX_VENV, _MLX_REQUIREMENTS, python_version=_MLX_PYTHON)
        env = hf_subprocess_env(self.model_id, self.revision, base_env=_clean_subprocess_env())
        with tempfile.TemporaryDirectory(prefix="senselab-vlm-") as tmp:
            items = []
            for i, req in enumerate(requests):
                image_path = None
                if req.image is not None:
                    image_path = str(Path(tmp) / f"frame_{i:05d}.png")
                    _save_png(req.image, Path(image_path))
                items.append({"image_path": image_path, "prompt": req.prompt})
            output = serve_in_venv(
                (_MLX_VENV, self.model_id, self.revision, "mlx", "default"),
                python=venv_python(venv_dir),
                script=_MLX_WORKER_SCRIPT,
                init={"model_path": self._snapshot, "model_revision": self.revision},
                request={"items": items, "max_tokens": self.max_tokens, "temperature": self.temperature},
                env=env,
                label="mlx-vlm",
                load_timeout_s=_MLX_LOAD_TIMEOUT_S,
                request_timeout_s=_MLX_REQUEST_TIMEOUT_S,
            )
        for key in ("peak_memory_bytes", "peak_rss_bytes"):
            self._stats[key] = max(self._stats[key], int(output.get(key, 0)))
        return [
            Generation(
                text=str(r["text"]),
                seconds=float(r.get("seconds", 0.0)),
                prompt_tokens=int(r.get("prompt_tokens", 0)),
                generation_tokens=int(r.get("generation_tokens", 0)),
            )
            for r in output["results"]
        ]

    def stats(self) -> Dict[str, Any]:
        """Peak MLX (Metal) memory and peak worker RSS, in bytes, over this backend's requests."""
        return dict(self._stats)

    def close(self) -> None:
        """End the resident worker for this model and revision, releasing its memory.

        A later :meth:`generate` starts a new worker.
        """
        shutdown_venv_workers((_MLX_VENV, self.model_id, self.revision))


_TRANSFORMERS_CACHE: Dict[tuple, tuple] = {}


class TransformersVLMBackend:
    """Qwen3-VL through transformers, in process; the loaded model is cached per identity."""

    def __init__(
        self,
        model_id: str = DEFAULT_TRANSFORMERS_MODEL,
        revision: str = "main",
        device: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> None:
        """Resolve the model's commit and choose a device.

        Args:
            model_id: Hugging Face repo of a transformers image-text-to-text model.
            revision: Ref or commit SHA; resolved to a SHA before anything is loaded.
            device: ``"cuda"``, ``"mps"`` or ``"cpu"``; defaults to CUDA when available.
            max_tokens: Generation cap per completion.
        """
        import torch
        from senselab_core.runtime.revision import resolve_revision

        self.model_id = model_id
        self.revision = resolve_revision(model_id, revision)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_tokens = max_tokens
        self._stats: Dict[str, Any] = {"peak_memory_bytes": 0}

    def _load(self) -> tuple:
        key = (self.model_id, self.revision, self.device)
        if key not in _TRANSFORMERS_CACHE:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            from senselab.utils.dependencies import load_hf_resilient

            dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
            processor = load_hf_resilient(
                AutoProcessor.from_pretrained, self.model_id, repo_id=self.model_id, revision=self.revision
            )
            model = load_hf_resilient(
                AutoModelForImageTextToText.from_pretrained,
                self.model_id,
                repo_id=self.model_id,
                revision=self.revision,
                dtype=dtype,
            )
            model.to(torch.device(self.device)).eval()  # type: ignore[arg-type]
            _TRANSFORMERS_CACHE[key] = (model, processor)
        return _TRANSFORMERS_CACHE[key]

    def generate(self, requests: Sequence[GenerationRequest]) -> List[Generation]:
        """Generate one completion per request with greedy decoding.

        Args:
            requests: The prompts and images.

        Returns:
            One ``Generation`` per request, in order.
        """
        import torch
        from PIL import Image

        model, processor = self._load()
        results = []
        for req in requests:
            content: List[Dict[str, Any]] = []
            if req.image is not None:
                content.append({"type": "image", "image": Image.fromarray(req.image.astype(np.uint8))})
            content.append({"type": "text", "text": req.prompt})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
            seconds = time.perf_counter() - start
            n_prompt = int(inputs["input_ids"].shape[1])
            new_tokens = out[0, n_prompt:]
            results.append(
                Generation(
                    text=processor.decode(new_tokens, skip_special_tokens=True),
                    seconds=seconds,
                    prompt_tokens=n_prompt,
                    generation_tokens=int(new_tokens.shape[0]),
                )
            )
        if self.device == "cuda":
            self._stats["peak_memory_bytes"] = max(self._stats["peak_memory_bytes"], torch.cuda.max_memory_allocated())
        return results

    def stats(self) -> Dict[str, Any]:
        """Peak CUDA memory in bytes (0 off CUDA)."""
        return dict(self._stats)


def default_backend(**kwargs: Any) -> VLMBackend:  # noqa: ANN401
    """The MLX backend on Apple silicon, the transformers backend elsewhere.

    Args:
        **kwargs: Forwarded to the backend's constructor.

    Returns:
        A ready backend.
    """
    if is_apple_silicon():
        return MLXVLMBackend(**kwargs)
    return TransformersVLMBackend(**kwargs)
