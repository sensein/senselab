"""Public API: split a dyad, project its voice activity, read turn-taking measures off the projection."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from senselab_core.runtime.models import WEIGHTS

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.turn_taking_projection import maai
from senselab.audio.tasks.turn_taking_projection.measures import summarize_events
from senselab.audio.tasks.turn_taking_projection.result import VAPResult
from senselab.audio.tasks.turn_taking_projection.separation import (
    SAMPLING_RATE,
    choose_speakers,
    separate_mix,
)
from senselab.utils.data_structures import DeviceType, ScriptLine

DEFAULT_SEPARATION = "masking"
"""Chosen by measurement on synthetic dyads; see specs/20261017-110000-dyadic-turn-taking/."""


def _mono16k(audio: Audio) -> Audio:
    if audio.waveform.shape[0] > 1:
        audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != SAMPLING_RATE:
        audio = resample_audios([audio], SAMPLING_RATE)[0]
    return audio


def separate_dyad(
    audio: Audio,
    diarization: Sequence[ScriptLine],
    method: str = DEFAULT_SEPARATION,
    speakers: Optional[Sequence[str]] = None,
    device: Optional[DeviceType] = None,
) -> Tuple[Audio, Audio]:
    """Split a mono two-speaker recording into one 16 kHz channel per speaker.

    Args:
        audio: The recording; downmixed and resampled to 16 kHz mono if needed.
        diarization: Speaker segments (``speaker``, ``start``, ``end``) in recording time. These
            define who each channel is; a separator's own speaker labels are not used.
        method: ``"masking"`` (mix under each speaker's smoothed diarization mask), ``"mossformer2"``
            (ClearerVoice ``MossFormer2_SS_16K``) or ``"pyannote"``
            (``pyannote/speech-separation-ami-1.0``, gated).
        speakers: The two speaker labels, in channel order. ``None`` takes the two with the most
            speech, by first onset.
        device: For the separators; masking runs in NumPy.

    Returns:
        ``(channel_a, channel_b)``, each with ``metadata["speaker"]`` and
        ``metadata["turn_taking_projection"]`` describing how it was made.

    Raises:
        ValueError: If the diarization does not hold two usable speakers or ``method`` is unknown.
    """
    mono = _mono16k(audio)
    pair = choose_speakers(diarization, speakers)
    mix = mono.waveform.squeeze(0).numpy().astype(np.float32)
    (a, b), provenance = separate_mix(mix, diarization, pair, method, device)
    channels = []
    for speaker, samples in zip(pair, (a, b)):
        channels.append(
            Audio(
                waveform=torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0),
                sampling_rate=SAMPLING_RATE,
                metadata={"speaker": speaker, "turn_taking_projection": dict(provenance, speakers=list(pair))},
            )
        )
    return channels[0], channels[1]


def _split_input(
    stereo_or_pair: Union[Audio, Tuple[Audio, Audio], List[Audio]],
    speakers: Optional[Sequence[str]],
) -> Tuple[Audio, Audio, Tuple[str, str], Dict[str, Any]]:
    if isinstance(stereo_or_pair, Audio):
        if stereo_or_pair.waveform.shape[0] != 2:
            raise ValueError(
                f"a single Audio must be stereo with one speaker per channel, got "
                f"{stereo_or_pair.waveform.shape[0]} channel(s); use separate_dyad for a mono mix"
            )
        rate = stereo_or_pair.sampling_rate
        a = Audio(waveform=stereo_or_pair.waveform[:1], sampling_rate=rate)
        b = Audio(waveform=stereo_or_pair.waveform[1:], sampling_rate=rate)
        source: Dict[str, Any] = {"kind": "stereo"}
    else:
        if len(stereo_or_pair) != 2:
            raise ValueError(f"expected two channels, got {len(stereo_or_pair)}")
        a, b = stereo_or_pair
        source = dict(a.metadata.get("turn_taking_projection", {"kind": "pair"}))
    if speakers is not None:
        if len(speakers) != 2:
            raise ValueError(f"speakers must be two labels, got {list(speakers)}")
        labels = (str(speakers[0]), str(speakers[1]))
    elif a.metadata.get("speaker") and b.metadata.get("speaker"):
        labels = (str(a.metadata["speaker"]), str(b.metadata["speaker"]))
    else:
        labels = ("a", "b")
    if labels[0] == labels[1]:
        raise ValueError(f"the two channels need distinct speaker labels, got {labels}")
    a, b = _mono16k(a), _mono16k(b)
    n = min(a.waveform.shape[-1], b.waveform.shape[-1])
    a = Audio(waveform=a.waveform[:, :n], sampling_rate=SAMPLING_RATE)
    b = Audio(waveform=b.waveform[:, :n], sampling_rate=SAMPLING_RATE)
    return a, b, labels, source


def project_voice_activity(
    stereo_or_pair: Union[Audio, Tuple[Audio, Audio], List[Audio]],
    model: str = maai.DEFAULT_VAP_CHECKPOINT,
    backchannel_model: Optional[str] = maai.DEFAULT_BC_CHECKPOINT,
    device: DeviceType = DeviceType.CPU,
    speakers: Optional[Sequence[str]] = None,
    num_workers: int = 3,
) -> VAPResult:
    """Run Voice Activity Projection (and backchannel prediction) over a two-speaker dyad.

    Args:
        stereo_or_pair: A stereo ``Audio`` with one speaker per channel, or two mono ``Audio``
            (e.g. from :func:`separate_dyad`). Resampled to 16 kHz; trimmed to the shorter channel.
        model: A key of :data:`maai.VAP_CHECKPOINTS`.
        backchannel_model: A key of :data:`maai.BC_CHECKPOINTS`, or ``None`` to skip the two
            backchannel runs.
        device: CPU or CUDA. Each run is single-threaded; on CPU, ``num_workers`` runs go at once.
        speakers: Labels for the two channels. ``None`` takes ``metadata["speaker"]`` of a pair from
            :func:`separate_dyad`, else ``("a", "b")``.
        num_workers: Concurrent runs (the VAP run and the two backchannel runs).

    Returns:
        A :class:`VAPResult` at 10 frames per second.

    Raises:
        ValueError: On a malformed input or an unknown checkpoint, or if ``device`` is MPS.
    """
    if device not in (DeviceType.CPU, DeviceType.CUDA):
        raise ValueError(f"MaAI runs on CPU or CUDA, not {device}")
    a, b, labels, source = _split_input(stereo_or_pair, speakers)
    jobs = maai.plan_jobs(model, backchannel_model)
    began = time.monotonic()
    arrays, timing = maai.run_maai(a, b, jobs, device=device, num_workers=num_workers)
    wall = time.monotonic() - began
    vap = arrays["vap"]
    n = vap.shape[0]
    frames = pd.DataFrame(vap, columns=list(maai.VAP_COLUMNS))
    frames.insert(0, "time", (np.arange(n, dtype=np.float64) + 1.0) / maai.FRAME_RATE)
    if backchannel_model is not None:
        frames["p_bc_a"] = arrays["bc_a"][:n, 0]
        frames["p_bc_b"] = arrays["bc_b"][:n, 0]
    duration = a.waveform.shape[-1] / SAMPLING_RATE
    models = {
        job.name: {
            "repo_id": job.checkpoint.repo_id,
            "revision": job.checkpoint.revision,
            "filename": job.checkpoint.filename,
            "licence": job.checkpoint.licence,
            "channel_1": "b" if job.swap else "a",
        }
        for job in jobs
    }
    models["encoder"] = {"url": maai.CPC_URL, "sha256": WEIGHTS["maai-cpc"]}
    flat_timing = {f"{name}_{k}": v for name, t in timing.items() for k, v in t.items()}
    flat_timing.update({"wall_s": wall, "audio_s": duration, "x_real_time": duration / wall if wall > 0 else 0.0})
    return VAPResult(
        frame_rate=maai.FRAME_RATE,
        speakers=labels,
        frames=frames,
        models=models,
        timing=flat_timing,
        input=source,
    )


def summarize_projection(
    vap: VAPResult,
    turns: Sequence[ScriptLine],
    ipus: Sequence[ScriptLine],
    backchannels: Optional[Sequence[ScriptLine]] = None,
) -> Dict[str, Any]:
    """Read dyadic turn-taking measures off a projection, at the caller's turns and IPUs.

    Args:
        vap: From :func:`project_voice_activity`; its ``speakers`` must match the labels below.
        turns: Turns (``speaker``, ``start``, ``end``); consecutive turns by different speakers are
            the shifts scored for projectability.
        ipus: Inter-pausal units; mutual silences between them are scored as hold/shift.
        backchannels: Backchannels produced, if annotated. ``None`` derives them from ``ipus``.

    Returns:
        ``{"speakers": {label: {...}}, "overall": {...}, "events": {...}, "parameters": {...}}`` —
        per-speaker summaries (turn-end projectability, gaps when taking the floor, hold/shift
        prediction after their IPUs, backchannel responsiveness), overall summaries, every event
        row behind them, and the constants used.
    """
    return summarize_events(vap, turns, ipus, backchannels)
