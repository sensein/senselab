"""The ``Audio`` ↔ file-path bridge for ClearerVoice's three audio-only capabilities.

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

CLEARVOICE_PROCESS = "clearvoice"
"""The result-cache process name every ClearerVoice capability is keyed under."""


def prepare_audios_for_clearvoice(audios: List[Audio], spec: ClearVoiceModelSpec) -> List[Audio]:
    """Resample to the checkpoint's rate and downmix to mono.

    Every ClearerVoice checkpoint is single-channel and rate-specific. senselab does this rather than
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
    """Run one audio-only ClearerVoice checkpoint and return whatever sources it produced.

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
        reader applied, and a ``"cache"`` record: the result-cache key, whether the sources were
        reused, and where a reused result was first computed.

    Raises:
        RuntimeError: If the worker fails or exceeds its ceiling.
    """
    if not audios:
        return []

    from senselab.utils.model_revision import resolve_revision
    from senselab.utils.tasks.cached_inference import audio_signature, result_cache_key, result_lookup, result_store

    prepared = prepare_audios_for_clearvoice(audios, spec)
    sha = resolve_revision(spec.model_id, revision)
    params = {"capability": spec.capability, "sampling_rate": spec.sampling_rate}
    keys = [
        result_cache_key(
            input_signature=audio_signature(audio),
            process=CLEARVOICE_PROCESS,
            model_id=spec.model_id,
            commit_sha=sha,
            params=params,
        )
        for audio in prepared
    ]
    held = [result_lookup(key) for key in keys]
    missing = [index for index, entry in enumerate(held) if entry is None]

    computed: dict[int, tuple[list[Audio], float]] = {}
    if missing:
        run_on = [prepared[index] for index in missing]
        total_audio_s = sum(audio.waveform.shape[-1] / audio.sampling_rate for audio in run_on)
        with tempfile.TemporaryDirectory(prefix="senselab-clearvoice-io-") as tmpdir:
            tmp = Path(tmpdir)
            in_paths = []
            for position, audio in enumerate(run_on):
                in_path = str(tmp / f"in_{position}.wav")
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
                revision=sha,
            )
            for index, paths, scalar in zip(missing, output_paths, scalars):
                read_back = [Audio(filepath=path) for path in paths]
                # Force the lazy load before the temporary directory holding the file is removed.
                waveforms = [audio.waveform for audio in read_back]
                sources = [Audio(waveform=waveform, sampling_rate=spec.sampling_rate) for waveform in waveforms]
                computed[index] = (sources, scalar)
                result_store(
                    keys[index],
                    {"input_norm_scalar": scalar, "n_sources": len(sources)},
                    process=CLEARVOICE_PROCESS,
                    model_id=spec.model_id,
                    commit_sha=sha,
                    arrays={f"source_{n}": waveform.numpy() for n, waveform in enumerate(waveforms)},
                )

    results: List[List[Audio]] = []
    for index, original in enumerate(prepared):
        entry = held[index]
        if entry is None:
            sources, scalar = computed[index]
            cache = {"key": keys[index], "hit": False, "origin": None}
        else:
            count = int(entry["result"]["n_sources"])
            sources = [
                Audio(waveform=entry["arrays"][f"source_{n}"], sampling_rate=spec.sampling_rate) for n in range(count)
            ]
            scalar = entry["result"]["input_norm_scalar"]
            cache = {"key": keys[index], "hit": True, "origin": entry.get("origin")}
        for source_index, produced in enumerate(sources):
            produced.metadata = dict(original.metadata)
            produced.metadata["clearvoice"] = {
                "model": spec.model_id,
                "commit": sha,
                "capability": spec.capability,
                "sampling_rate": spec.sampling_rate,
                "source_index": source_index,
                "n_sources": len(sources),
                "input_norm_scalar": scalar,
                "input_norm_applied_to_output": len(sources) == 1,
                "cache": cache,
            }
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
    """Return ``(model_id, commit)`` for an ``Audio`` a ClearerVoice model produced, if it says so.

    Args:
        audio: A returned ``Audio``.

    Returns:
        The model id and the 40-hex commit its weights came from, or ``None`` if this audio did not
        come from ClearerVoice.
    """
    record = audio.metadata.get("clearvoice")
    if not isinstance(record, dict):
        return None
    return str(record["model"]), str(record["commit"])
