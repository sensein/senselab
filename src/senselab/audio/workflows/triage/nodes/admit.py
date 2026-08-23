"""ADMIT — is this recording measurable at all.

The only rejections are decode failure, all samples zero, and a constant signal. No thresholds, no
``flag`` outcome, no models, no derived audio. The measurements behind the threshold-free rule are in
``specs/20260817-triage-workflow-dag/admit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

NODE = "ADMIT"


@dataclass(frozen=True)
class AdmitResult(NodeResult):
    """ADMIT's result.

    Attributes:
        audio: The decoded recording, as supplied, on ``pass``; None on ``fail``.
    """

    audio: Audio | None


def admit(
    store: ProvStore,
    source: str | Path,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> AdmitResult:
    """Decide whether the recording is measurable at all.

    Rejects only decode failure, all-zero samples and a constant signal. Everything else passes, as
    supplied — no resampling, no channel reduction, no models, no quality judgement. ``config``,
    ``hint`` and ``run_dir`` belong to the shared node shape and are not read: ADMIT holds no
    numbers, no hint changes whether a file decodes, and it writes no sidecars.

    Args:
        store: The provenance store.
        source: The recording, as supplied.
        config: The triage configuration. Unread.
        hint: What the recording was declared to contain. Unread.
        run_dir: The run directory. Unused.

    Returns:
        The verdict and, on ``pass``, the decoded audio.
    """
    activity_id = store.activity(node=NODE, step=None, parameters={"audio_file": str(source)})
    agent_id = software_agent(store)
    store.was_associated_with(activity_id, agent_id)

    def _fail(why: str) -> AdmitResult:
        entity_id, verdict = write_verdict(
            store, activity_id, agent_id, node=NODE, outcome=Outcome.FAIL, kind=None, why=why, detail={}
        )
        return AdmitResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, audio=None)

    try:
        audio = Audio(filepath=str(source))
        waveform = audio.waveform
    except Exception as err:  # noqa: BLE001 — every decode failure is the same finding
        return _fail(f"decode failure: {type(err).__name__}")
    if waveform.shape[-1] == 0:
        return _fail("decode returned zero frames")
    if not bool(torch.any(waveform != 0)):
        return _fail("every sample is zero")
    if bool(torch.all(waveform == waveform.reshape(-1)[0])):
        return _fail("constant value; no variance")

    duration_s = waveform.shape[-1] / audio.sampling_rate
    stream_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "recording",
            "path": str(Path(source).resolve()),
            "sampling_rate": int(audio.sampling_rate),
            "channels": int(waveform.shape[0]),
        },
    )
    store.was_generated_by(stream_id, activity_id)
    store.was_attributed_to(stream_id, agent_id)
    verdict_id, verdict = write_verdict(
        store,
        activity_id,
        agent_id,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why="the file decodes and its samples vary",
        detail={"stream": stream_id},
    )
    return AdmitResult(verdict=verdict, view=(stream_id, verdict_id), verdict_entity_id=verdict_id, audio=audio)
