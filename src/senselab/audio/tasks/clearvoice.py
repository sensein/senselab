"""The ``Audio`` ↔ file-path bridge for ClearVoice's three audio-only capabilities.

``senselab.utils.clearvoice`` owns the venv, the pin, the device and the worker, and cannot touch
``Audio``. This module is the other half: resample, downmix, write, run, read back, carry provenance —
written once, because it is identical for enhancement, separation and super-resolution.

The **output count** is deliberately not decided here. :func:`run_clearvoice_over_audios` reports the
sources it received and each entry point applies its own contract to that, via
:func:`single_source_per_input` or its own check. Never from the model's name: design.md D-5.

Design: ``specs/20260819-clearvoice-integration/design.md``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from senselab.audio.data_structures import Audio
from senselab.utils.clearvoice import ClearVoiceModelSpec, run_clearvoice_audio
from senselab.utils.data_structures import DeviceType


def prepare_audios_for_clearvoice(audios: List[Audio], spec: ClearVoiceModelSpec) -> List[Audio]:
    """Resample to the checkpoint's rate and downmix to mono.

    Every ClearVoice checkpoint is single-channel and rate-specific. senselab does this rather than
    upstream's own reader, whose rescaling heuristic mis-scales a quiet 32-bit input: design.md D-8.

    Args:
        audios: Inputs.
        spec: The checkpoint about to run.

    Returns:
        One mono ``Audio`` per input at ``spec.sampling_rate``, in order.
    """
    from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

    return downmix_audios_to_mono(resample_audios(audios, resample_rate=spec.sampling_rate))


def run_clearvoice_over_audios(
    spec: ClearVoiceModelSpec,
    audios: List[Audio],
    *,
    device: Optional[DeviceType] = None,
    timeout_s: Optional[float] = None,
    revision: str = "main",
) -> List[List[Audio]]:
    """Run one audio-only ClearVoice checkpoint and return whatever sources it produced.

    Args:
        audios: Inputs. Resampled and downmixed as needed.
        spec: Checkpoint to run.
        device: CUDA or CPU. ``None`` leaves the choice to the worker.
        timeout_s: Ceiling on the worker, in seconds; ``None`` derives one from the total duration.
        revision: Ref or commit for the checkpoint repository.

    Returns:
        One list per input, holding as many ``Audio`` objects as the checkpoint actually produced —
        not as many as its name suggests. Each carries the input's metadata plus a ``"clearvoice"``
        entry naming the model, the resolved commit, the source index, and the RMS scalar upstream's
        reader applied.

    Raises:
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    if not audios:
        return []

    prepared = prepare_audios_for_clearvoice(audios, spec)
    total_audio_s = sum(audio.waveform.shape[-1] / audio.sampling_rate for audio in prepared)

    with tempfile.TemporaryDirectory(prefix="senselab-clearvoice-io-") as tmpdir:
        tmp = Path(tmpdir)
        in_paths = []
        for index, audio in enumerate(prepared):
            in_path = str(tmp / f"in_{index}.wav")
            # A plain .wav resolves to FLOAT and round-trips these samples bit-exactly; out-of-range
            # data raises rather than being clipped on the way in.
            audio.save_to_file(in_path)
            in_paths.append(in_path)

        output_paths, scalars, sha = run_clearvoice_audio(
            spec,
            in_paths,
            str(tmp),
            total_audio_s=total_audio_s,
            device=device,
            timeout_s=timeout_s,
            revision=revision,
        )

        results: List[List[Audio]] = []
        for original, paths, scalar in zip(prepared, output_paths, scalars):
            sources = []
            for source_index, path in enumerate(paths):
                produced = Audio(filepath=path)
                # Force the lazy load before the temporary directory holding the file is removed.
                _ = produced.waveform
                produced.metadata = dict(original.metadata)
                produced.metadata["clearvoice"] = {
                    "model": spec.model_id,
                    "commit": sha,
                    "capability": spec.capability,
                    "sampling_rate": spec.sampling_rate,
                    "source_index": source_index,
                    "n_sources": len(paths),
                    "input_norm_scalar": scalar,
                    "input_norm_applied_to_output": len(paths) == 1,
                }
                sources.append(produced)
            results.append(sources)
    return results


def single_source_per_input(
    results: List[List[Audio]],
    spec: ClearVoiceModelSpec,
    caller: str,
) -> List[Audio]:
    """Flatten one-source-per-input results, refusing anything else.

    Args:
        results: What :func:`run_clearvoice_over_audios` returned.
        spec: The checkpoint that ran.
        caller: The public function's name, for the message.

    Returns:
        One ``Audio`` per input.

    Raises:
        RuntimeError: If any input yielded a number of sources other than one — refused rather than
            taking the first or concatenating, on PR #569's reasoning.
    """
    wrong = [(index, len(sources)) for index, sources in enumerate(results) if len(sources) != 1]
    if wrong:
        detail = ", ".join(f"input {index} -> {count} source(s)" for index, count in wrong[:3])
        raise RuntimeError(
            f"{caller} expects one signal per input, but {spec.model_id} produced {detail}. Refusing "
            "to pick one or to concatenate them: if this checkpoint decomposes its input, it belongs "
            "behind senselab.audio.tasks.source_separation.separate_audios, not here."
        )
    return [sources[0] for sources in results]


def clearvoice_provenance(audio: Audio) -> Optional[Tuple[str, str]]:
    """Return ``(model_id, commit)`` for an ``Audio`` a ClearVoice model produced, if it says so.

    Args:
        audio: A returned ``Audio``.

    Returns:
        The model id and the 40-hex commit its weights came from, or ``None`` if this audio did not
        come from ClearVoice.
    """
    record = audio.metadata.get("clearvoice")
    if not isinstance(record, dict):
        return None
    return str(record["model"]), str(record["commit"])
