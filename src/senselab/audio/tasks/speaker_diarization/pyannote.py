"""This module implements the Pyannote Diarization task."""

import time
from typing import Dict, List, Optional, Union

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel, ScriptLine, _select_device_and_dtype
from senselab.utils.data_structures.logging import logger
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import resolve_model, retry_on_transient_error

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Annotation

    PYANNOTEAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    PYANNOTEAUDIO_AVAILABLE = False

CAPABILITIES = DiarizationCapabilities(
    populates_text=False,
    speaker_label_kind="identity",  # SPEAKER_00, SPEAKER_01, ...
    labels_stable_across_files=False,  # not measured; False is the conservative default
    max_speakers=None,  # unmeasured — pending the NeMo synthetic-speaker probe
    honors_speaker_hints=True,  # the only backend that acts on num_speakers
)


class PyannoteDiarization:
    """Factory for creating and caching **Pyannote** diarization pipelines.

    Pipelines are cached per *(model.path_or_uri, revision, device)*, so repeated calls
    with the same configuration reuse the initialized pipeline.

    Guidance:
        - Pyannote models typically expect **mono, 16 kHz** audio.
        - If you know the number of speakers, set `num_speakers`; otherwise use
          `min_speakers`/`max_speakers` bounds to help estimation.
        - Supported devices: ``DeviceType.CPU`` and ``DeviceType.CUDA``.
    """

    _pipelines: Dict[str, "Pipeline"] = {}

    @classmethod
    def _get_pyannote_diarization_pipeline(
        cls,
        model: PyannoteAudioModel,
        device: Union[DeviceType, None],
    ) -> "Pipeline":
        """Get or create a Pyannote Diarization pipeline.

        Args:
            model (PyannoteAudioModel): The Pyannote model.
            device (DeviceType): The device to run the model on.

        Returns:
            Pipeline: The diarization pipeline.
        """
        if not PYANNOTEAUDIO_AVAILABLE:
            raise ModuleNotFoundError(
                "`pyannote-audio` is not installed. "
                "Please install senselab audio dependencies using `pip install senselab`."
            )

        device, _ = _select_device_and_dtype(
            user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU]
        )
        key = f"{model.path_or_uri}-{model.revision}-{device}"
        if key not in cls._pipelines:
            # Resolve the ref to an immutable SHA once (download-once via the
            # cross-process heartbeat lock), then pin the pipeline load to it so a
            # cached model makes no per-call Hub HEAD — the 429 source under
            # parallel batch load. ``resolve_model`` takes the token for the gated
            # pyannote repo; the SHA also fixes the latent ``revision=f"{None}"``
            # ("None" string) when ``model.revision`` is unset.
            token = get_huggingface_token()
            sha, _ = resolve_model(str(model.path_or_uri), model.revision or "main", token=token)
            pipeline = retry_on_transient_error(
                Pipeline.from_pretrained,
                checkpoint=f"{model.path_or_uri}",
                revision=sha,
                token=token,
            )
            if not pipeline:
                raise ValueError(f"Pyannote model {model.path_or_uri} not found.")
            pipeline = pipeline.to(torch.device(device.value))
            cls._pipelines[key] = pipeline
        return cls._pipelines[key]


def diarize_audios_with_pyannote(
    audios: List[Audio],
    model: Optional[PyannoteAudioModel] = None,
    device: Optional[DeviceType] = None,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    exclusive: bool = True,
) -> List[List[ScriptLine]]:
    """Diarize audios with **Pyannote**; returns per-speaker segments per audio.

    Requirements:
        - Input must be **mono** (`[1, T]`); stereo/multi-channel is rejected.
        - Sampling rate must be **16 kHz** (per model card for `3.1`).

    Args:
        audios (list[Audio]):
            Audio clips to diarize (mono, 16 kHz).
        model (PyannoteAudioModel | None):
            Pyannote model. Defaults to ``pyannote/speaker-diarization-community-1@main``.
        device (DeviceType | None):
            Inference device (``CPU`` or ``CUDA``).
        num_speakers (int | None):
            If known, fix the number of speakers.
        min_speakers (int | None):
            Minimum speakers when estimating (ignored if `num_speakers` is set).
        max_speakers (int | None):
            Maximum speakers when estimating (ignored if `num_speakers` is set).

        exclusive (bool):
            Which view of the pipeline's output to return. ``True`` (default) returns
            ``exclusive_speaker_diarization``: a **partition**, where every instant belongs to at
            most one speaker and concurrent speech has been resolved away. ``False`` returns the
            pipeline's overlapping view, where two speakers talking at once produce two segments
            covering the same instant.

            This matters more than it looks. With the exclusive view, *no downstream consumer can
            detect overlap at all* — a per-instant speaker count derived from these segments is
            capped at 1 by construction, so it reports "no overlap" as a confident measurement
            rather than as something the input could not express. `community-1` computes overlap
            internally (its local segmentation model is `segmentation-3.0`); the exclusive view
            discards it.

    Returns:
        list[list[ScriptLine]]: One list per input audio with `(speaker, start, end)`. Segments may
        overlap when ``exclusive=False``.

    Raises:
        ModuleNotFoundError:
            If `pyannote-audio` is not installed.
        ValueError:
            If audio is not mono or sampling rate ≠ 16 kHz.

    Example (estimate speakers within bounds):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> mdl = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")
        >>> diar = diarize_audios_with_pyannote(
        ...     [a1],
        ...     model=mdl,
        ...     device=DeviceType.CPU,
        ...     min_speakers=1,
        ...     max_speakers=3,
        ... )
        >>> len(diar[0]) >= 0
        True

    Example (known number of speakers):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> diar = diarize_audios_with_pyannote([a1], num_speakers=2)
        >>> len(diar[0]) >= 0
        True
    """

    def _annotation_to_script_lines(annotation: "Annotation") -> List[ScriptLine]:
        """Convert a Pyannote annotation to a list of script lines.

        Args:
            annotation (Annotation): The Pyannote annotation object.

        Returns:
            List[ScriptLine]: A list of script lines.
        """
        diarization_list: List[ScriptLine] = []
        for segment, label in annotation:
            diarization_list.append(ScriptLine(speaker=label, start=segment.start, end=segment.end))
        return diarization_list

    if not PYANNOTEAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`pyannote-audio` is not installed. "
            "Please install senselab audio dependencies using `pip install senselab`."
        )

    if model is None:
        model = PyannoteAudioModel(path_or_uri="pyannote/speaker-diarization-community-1", revision="main")

    # 16khz comes from the model cards of pyannote/speaker-diarization-community-1
    expected_sample_rate = 16000

    # Check that all audio objects have the correct sampling rate
    for audio in audios:
        if audio.waveform.shape[0] != 1:
            raise ValueError(f"Audio waveform must be mono (1 channel), but got {audio.waveform.shape[0]} channels")
        if audio.sampling_rate != expected_sample_rate:
            raise ValueError(
                "Audio sampling rate "
                + str(audio.sampling_rate)
                + " does not match expected "
                + str(expected_sample_rate)
            )

    # Take the start time of the model initialization
    start_time_model = time.time()
    pipeline = PyannoteDiarization._get_pyannote_diarization_pipeline(model=model, device=device)
    end_time_model = time.time()
    elapsed_time_model = end_time_model - start_time_model
    logger.info(f"Time taken to initialize the pyannote model: {elapsed_time_model:.2f} seconds")

    # Perform diarization
    start_time_diarization = time.time()
    results: List[List[ScriptLine]] = []
    for audio in audios:
        diarization = pipeline(
            {"waveform": audio.waveform, "sample_rate": audio.sampling_rate},
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        # ``getattr`` with an explicit failure rather than a silent fallback to the exclusive view:
        # returning a partition when the caller asked for overlap would be the absent-vs-zero
        # failure again — a structurally-impossible overlap reading as a measured absence of one.
        if exclusive:
            annotation = diarization.exclusive_speaker_diarization
        else:
            annotation = getattr(diarization, "speaker_diarization", None)
            if annotation is None:
                raise AttributeError(
                    "the pipeline output exposes no overlapping speaker_diarization view; "
                    "exclusive=False cannot be honoured, and silently returning the exclusive "
                    "partition would report structural non-overlap as a measurement"
                )
        results.append(_annotation_to_script_lines(annotation))
    end_time_diarization = time.time()
    elapsed_time_diarization = end_time_diarization - start_time_diarization
    logger.info(f"Time taken to perform diarization: {elapsed_time_diarization:.2f} seconds")

    return results
