"""This module implements some utilities to extract speaker embeddings from an audio clip."""

from typing import List, Optional

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import SpeakerEmbeddingProvenance, TargetSpeakerEmbedding
from senselab.audio.tasks.speaker_embeddings.speechbrain import SpeechBrainEmbeddings
from senselab.utils.compatibility import requires_compatibility
from senselab.utils.data_structures import DeviceType, SenselabModel, SpeechBrainModel
from senselab.utils.tasks.embedding_distribution import (
    describe_embedding_distribution,
    select_dominant_vectors,
)

# ECAPA rather than a pair: PR #543 defaulted to ECAPA+ResNet because analyze_audio scores both
# per-window, and this estimator is deliberately decoupled from that consumer, so a second model
# would be unused cost. provenance.model_id records which one produced a vector.
DEFAULT_SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

# Measured for a profile centroid, not picked: a 2.0 s window on a 1.0 s hop gave cross-file
# centroid stability 0.890 and cross-subject separation 0.168, against 0.331 for a 0.5/0.25 grid
# carrying four times the windows. Deliberately different from windowing.py's 1.0/0.5 detection
# defaults, which are tuned for temporal resolution on a 0.5 s bucket grid.
PROFILE_WINDOW_S = 2.0
PROFILE_HOP_S = 1.0


@requires_compatibility("audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios")
def extract_speaker_embeddings_from_audios(
    audios: List[Audio],
    model: Optional[SpeechBrainModel] = None,
    device: Optional[DeviceType] = None,
) -> List[torch.Tensor]:
    """Compute fixed-dimensional speaker embeddings for a batch of `Audio` objects.

    By default, this uses a SpeechBrain speaker-recognition model
    (``speechbrain/spkrec-ecapa-voxceleb``, revision ``"main"``) to extract
    one embedding per input audio. The order of outputs matches the order of
    the inputs.

    Notes:
        - Many pretrained speaker models are trained on **mono, 16 kHz** audio.
          For best results, consider preprocessing inputs with
          `downmix_audios_to_mono(...)` and `resample_audios(..., 16000)`.
        - `device` must be a `DeviceType` (`CPU` or `CUDA`). If `None`,
          the backend will choose `CUDA` if available, otherwise `CPU`.
        - The embedding dimensionality depends on the model
          (e.g., ECAPA-TDNN often returns 192-D vectors).

    Args:
        audios (list[Audio]):
            Input audio objects to embed.
        model (SpeechBrainModel, optional):
            SpeechBrain model handle (path or HF Hub URI + revision).
            If ``None``, defaults to
            ``SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")``.
        device (DeviceType, optional):
            Target device for inference:
              * ``DeviceType.CPU``
              * ``DeviceType.CUDA`` (GPU, if available)
            If ``None``, a default is chosen by the backend.

    Returns:
        list[torch.Tensor]: One 1-D tensor per input audio (e.g., shape ``[192]`` for ECAPA).

    Raises:
        NotImplementedError:
            If a non-SpeechBrain model is provided (only SpeechBrain is supported currently).

    Example (default model on CPU):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> embs = extract_speaker_embeddings_from_audios([a1, a2], device=DeviceType.CPU)
        >>> embs[0].shape
        torch.Size([192])

    Example (explicit model on CUDA):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, SpeechBrainModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")
        >>> embs = extract_speaker_embeddings_from_audios([a1], model=model, device=DeviceType.CUDA)
        >>> len(embs), embs[0].ndim
        (1, 1)
    """
    if model is None:
        model = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main")

    if isinstance(model, SpeechBrainModel):
        return SpeechBrainEmbeddings.extract_speechbrain_speaker_embeddings_from_audios(
            audios=audios, model=model, device=device
        )
    else:
        raise NotImplementedError(
            "Only SpeechBrain models are supported for now. We aim to support more models in the future."
        )


# Imported here rather than at module top: `windowing.py` itself imports
# `extract_speaker_embeddings_from_audios` back out of this module (it needs the plain
# per-audio extractor to build its per-window grid), so a top-of-file import would run before
# that name is bound and the package's own `__init__` -- still mid-import -- would hand
# `windowing.py` a partially initialized module. By this point in the file the name above is
# already defined, so the cycle closes cleanly.
from senselab.audio.tasks.speaker_embeddings.windowing import (  # noqa: E402
    WindowEmbedding,
    extract_per_window_embeddings,
)


def _resolve_embedding_model(model: Optional[SenselabModel]) -> tuple[str, Optional[str], Optional[str]]:
    """Return ``(model_id, commit_sha, unresolved_reason)`` for provenance.

    A ref in the commit field would be provenance that is confidently wrong, so an unresolvable
    model yields ``None`` plus a stated reason instead.

    Args:
        model: The caller's model, or ``None`` for the default.

    Returns:
        ``(model_id, commit_sha, unresolved_reason)``; exactly one of the last two is ``None``.
    """
    model_id = str(model.path_or_uri) if model is not None else DEFAULT_SPEAKER_EMBEDDING_MODEL
    sha = getattr(model, "commit_sha", None) if model is not None else None
    if sha:
        return model_id, str(sha), None
    try:
        from senselab.utils.model_revision import resolve_revision

        return model_id, resolve_revision(model_id, "main"), None
    except Exception as exc:  # noqa: BLE001 — an unresolved commit must be recorded, not raised
        return model_id, None, f"{type(exc).__name__}: {exc}"


def estimate_speaker_embedding_from_audios(
    audios: List[Audio],
    model: Optional[SenselabModel] = None,
    device: Optional[DeviceType] = None,
    window_s: float = PROFILE_WINDOW_S,
    hop_s: float = PROFILE_HOP_S,
    aggregator: str = "spherical_mean",
    reject_contamination: bool = False,
    created_at: Optional[str] = None,
) -> TargetSpeakerEmbedding:
    """Estimate one speaker embedding from files that *may* contain that speaker.

    Windows each file, embeds every window, pools them, and describes the resulting distribution.
    The returned statistics are what let a caller judge how well-supported the centroid is: this
    function reaches no verdict about whether the file set was clean.

    Args:
        audios: The recordings to enroll from. A file that does not contain the target speaker is
            not an error -- it shows up in the returned per-file statistics, which is how a caller
            curates its input.
        model: Embedding model. Defaults to ECAPA.
        device: CPU or CUDA.
        window_s: Window length. Defaults to the measured profile-centroid value, 2.0 s.
        hop_s: Hop between windows. Defaults to 1.0 s.
        aggregator: ``"spherical_mean"``, ``"trimmed_mean"`` or ``"medoid"``.
        reject_contamination: When ``True``, group the pooled windows and keep only the dominant
            group before describing. Off by default, because selecting a dominant group is a
            decision. What it removed is recorded in ``provenance.method`` and
            ``provenance.n_windows_dropped``.
        created_at: ISO-8601 timestamp for provenance. Not defaulted to "now": a library that
            stamps wall-clock time makes its own output unreproducible.

    Returns:
        A :class:`TargetSpeakerEmbedding` carrying the vector, its provenance, and the
        distribution it was estimated from.

    Raises:
        ValueError: If ``audios`` is empty, or if no window survived extraction.
    """
    if not audios:
        raise ValueError("estimate_speaker_embedding_from_audios needs at least one audio")

    model_id, commit_sha, unresolved = _resolve_embedding_model(model)
    # Deliberately *not* `SpeechBrainModel(path_or_uri=model_id, revision="main")` here: HFModel's
    # validators (`validate_hf_model_id`, `_resolve_commit_sha`) call the Hub over the network on
    # a cache miss, and on a cold cache `check_hf_repo_exists` -> `ensure_hf_model` runs a full
    # `snapshot_download` -- exactly the 20 GB mistake this task was warned against, and it would
    # fire regardless of any patch to `_resolve_embedding_model`, since it is a *separate*
    # construction. The plain `SenselabModel` base class carries `path_or_uri` with no such
    # validator, which is all `extract_per_window_embeddings` and its own tests need from it.
    speaker_model: SenselabModel = model if model is not None else SenselabModel(path_or_uri=model_id)

    vectors: list[np.ndarray] = []
    file_ids: list[str] = []
    starts: list[float] = []
    source_files: list[str] = []
    for idx, audio in enumerate(audios):
        # `Audio.filepath` is a method, not a property (`getattr(audio, "filepath")` alone
        # returns the bound-method object, which is truthy for every audio and stringifies
        # off the object's *field* repr rather than its identity or path -- two audios with
        # equal metadata then collide onto one file id). Calling it is required to get the
        # actual path, or `None` for an in-memory audio with no backing file.
        file_id = str(audio.filepath() or f"audio-{idx}")
        source_files.append(file_id)
        # `models` is typed `list[str]` on the real `extract_per_window_embeddings` (plain model
        # ids), but the model object is passed through here instead: the extraction seam -- real
        # or, in tests, monkeypatched -- reads `.path_or_uri` off it rather than re-deriving an id
        # string, so provenance and the mocked extraction agree on one source of truth. See the
        # comment above on why this can't instead be a network-validated `SpeechBrainModel`.
        per_model: dict[str, list[WindowEmbedding]] = extract_per_window_embeddings(
            audio=audio,
            models=[speaker_model],  # type: ignore[list-item]
            device=device,
            window_s=window_s,
            hop_s=hop_s,
        )
        for windows in per_model.values():
            for w in windows:
                vectors.append(np.asarray(w.vector, dtype=np.float64))
                file_ids.append(file_id)
                starts.append(float(w.start_s))

    if not vectors:
        raise ValueError("no embedding windows were produced; are the inputs shorter than window_s?")

    pooled = np.vstack(vectors)
    n_input = int(pooled.shape[0])
    method = aggregator

    if reject_contamination:
        selection = select_dominant_vectors(pooled, file_ids)
        keep = selection.kept_indices
        pooled = pooled[keep]
        file_ids = [file_ids[i] for i in keep]
        starts = [starts[i] for i in keep]
        method = f"{aggregator}+dominant_cluster"

    centroid, distribution = describe_embedding_distribution(
        pooled,
        file_ids,
        aggregator=aggregator,
        window_s=window_s,
        hop_s=hop_s,
        window_starts_s=starts,
    )

    return TargetSpeakerEmbedding(
        vector=centroid,
        provenance=SpeakerEmbeddingProvenance(
            model_id=model_id,
            model_commit_sha=commit_sha,
            unresolved_reason=unresolved,
            method=method,
            source_files=source_files,
            window_s=window_s,
            hop_s=hop_s,
            n_windows_used=int(distribution.counts.n_scored),
            n_windows_dropped=n_input - int(distribution.counts.n_scored),
            created_at=created_at,
        ),
        distribution=distribution,
    )
