"""Label Studio ML backend: speaker diarization via senselab.

Connect this backend to a Label Studio project whose config has a ``<Labels>`` control on an
``<Audio>`` object. For each task it builds an
:class:`~senselab_ls.common.audio_plus.AudioPlus` from the incoming audio reference, runs
pyannote (or Sortformer) diarization, and returns one ``labels`` region per speaker turn.

Runtime configuration (env vars):

* ``DIA_MODEL``  -- diarization model repo id (default: pyannote community-1).
* ``DIA_DEVICE`` -- ``cuda`` / ``cpu`` / ``mps``; unset means auto-select.
* ``HF_TOKEN``, ``LABEL_STUDIO_URL``, ``LABEL_STUDIO_API_KEY`` -- model + data access.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType
from senselab_ls.common import engine
from senselab_ls.common.audio_io import load_audio
from senselab_ls.common.audio_plus import MetadataProvider, build_audio_plus
from senselab_ls.common.b2ai_metadata import B2AIMetadataProvider
from senselab_ls.common.ls_regions import diarization_to_ls

DIARIZATION_MODEL_ID = os.getenv("DIA_MODEL", engine.DEFAULT_PYANNOTE_MODEL)
MODEL_VERSION = f"senselab-diarization:{DIARIZATION_MODEL_ID}"


def _metadata_provider() -> Optional[MetadataProvider]:
    """Build the b2ai metadata provider when ``B2AI_DATASET_ROOT`` is set; else ``None``.

    Returns:
        A :class:`B2AIMetadataProvider` rooted at ``B2AI_DATASET_ROOT``, or ``None`` (which
        makes ``build_audio_plus`` fall back to bytes-only Audio+).
    """
    root = os.getenv("B2AI_DATASET_ROOT")
    return B2AIMetadataProvider(root) if root else None


def _pick_device() -> Optional[DeviceType]:
    """Resolve the device from ``DIA_DEVICE``; ``None`` means auto-select.

    Returns:
        The requested ``DeviceType``, or ``None`` for senselab's auto-detection.
    """
    value = os.getenv("DIA_DEVICE", "").strip().lower()
    if value in ("cuda", "gpu"):
        return DeviceType.CUDA
    if value == "cpu":
        return DeviceType.CPU
    if value == "mps":
        return DeviceType.MPS
    return None


class DiarizationBackend(LabelStudioMLBase):
    """A per-aspect ML backend that pre-annotates speaker segments."""

    def setup(self) -> None:
        """Record the model version shown on predictions in Label Studio."""
        self.set("model_version", MODEL_VERSION)

    def predict(
        self,
        tasks: list[dict[str, Any]],
        context: Optional[dict[str, Any]] = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> ModelResponse:
        """Return diarization predictions for each task.

        Args:
            tasks: Label Studio tasks; each ``task["data"]`` carries the audio reference.
            context: Interactive-labeling context (unused here).
            **kwargs: Ignored extra arguments passed by the SDK.

        Returns:
            A :class:`ModelResponse` with one prediction per task.
        """
        from_name, to_name, value_key = self.label_interface.get_first_tag_occurence("Labels", "Audio")
        device = _pick_device()
        b2ai_provider = _metadata_provider()
        predictions: list[dict[str, Any]] = []
        for task in tasks:
            ref = str(task["data"][value_key])
            task_id = task.get("id")

            def _load(resolve_ref: str, task_id: object = task_id) -> Audio:
                """Load one audio ref, resolving LS-hosted refs via this task's LS credentials.

                Pass host + token explicitly so ``get_local_path`` uses the token directly (legacy
                ``Token`` auth) instead of the PAT-refresh path, which otherwise falls back to the
                machine ``HOSTNAME`` and fails with connection-refused.
                """
                return load_audio(
                    resolve_ref,
                    http_downloader=lambda url: self.get_local_path(
                        url,
                        task_id=task_id,
                        ls_host=os.getenv("LABEL_STUDIO_URL"),
                        ls_access_token=os.getenv("LABEL_STUDIO_API_KEY"),
                    ),
                )

            # Mode is auto-detected from the ref: s3:// -> b2ai mode (Audio+ metadata + related
            # files); anything else (an LS-uploaded/standalone file) -> no dataset metadata.
            provider = b2ai_provider if ref.startswith("s3://") else None
            audio_plus = build_audio_plus(ref, audio_loader=_load, metadata_provider=provider)
            segments = engine.diarize(audio_plus.audio, model_id=DIARIZATION_MODEL_ID, device=device)
            regions = diarization_to_ls(segments, from_name, to_name=to_name)
            predictions.append({"result": regions, "model_version": self.get("model_version"), "score": 1.0})
        return ModelResponse(predictions=predictions)
