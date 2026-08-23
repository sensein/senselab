"""PREPROCESS — one conditioning pass, every shared derivative written to the store.

The recognizers, the aligner, SQUIM, level and YAMNet silence read the plain resampled signal; the
envelope, spans, spectrograms and gammatone read the pre-emphasised one. A derivative that cannot be
computed is absent from the store, not an error. Every parameter's derivation is in
``data/config/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.envelope.api import hilbert_envelope_dbfs, rolling_floor_dbfs
from senselab.audio.tasks.features_extraction.torchaudio import extract_spectrogram_from_audios
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.forced_alignment.constants import DEFAULT_ALIGN_MODELS_HF
from senselab.audio.tasks.forced_alignment.forced_alignment import align_transcriptions
from senselab.audio.tasks.gammatone.api import gammatone_filterbank
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.spans.api import NoContrast, propose_spans
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.tasks.speech_to_text_ensemble.api import fuse_word_streams, iter_word_leaves
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel, Language, ScriptLine
from senselab.utils.prov_store import ProvStore

NODE = "PREPROCESS"
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_TIMESTAMP_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
ALIGNMENT_LANGUAGE = "en"


def _crisperwhisper_model() -> HFModel:
    """The CrisperWhisper model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=CRISPERWHISPER_ID, revision="main")


def _qwen_model() -> HFModel:
    """The Qwen3-ASR model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=QWEN_ID, revision="main")


@dataclass(frozen=True)
class PreprocessResult(NodeResult):
    """PREPROCESS's result.

    Attributes:
        absent: Names of derivatives that could not be computed and are absent from the store.
    """

    absent: tuple[str, ...]


def _bound_to_duration(start: float, end: float, duration_s: float) -> tuple[float, float] | None:
    """Bound one timed span by the duration of the stream it was timed against.

    Args:
        start: The span's start, in seconds.
        end: The span's end, in seconds.
        duration_s: The duration the stream decoded to, in seconds.

    Returns:
        The span with its end bound by ``duration_s``, or None when its start is at or past
        ``duration_s``, where it names no part of the stream at all.
    """
    if start >= duration_s:
        return None
    return start, min(end, duration_s)


def _measurement(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    name: str,
    signal: str,
    attributes: dict[str, Any],
    derived_from: tuple[str, ...] = (),
) -> str:
    """Write one derivative measurement entity with its provenance."""
    entity_id = store.entity(
        prov_type="measurement", extent=None, attributes={"name": name, "signal": signal, **attributes}
    )
    store.was_generated_by(entity_id, activity_id)
    store.was_attributed_to(entity_id, agent_id)
    for source_id in derived_from:
        store.was_derived_from(entity_id, source_id)
    return entity_id


def preprocess(  # noqa: C901 — one block per derivative, each independent
    store: ProvStore,
    source: Audio,
    config: TriageConfig,
    hint: AudioHints | None = None,
    *,
    run_dir: Path,
) -> PreprocessResult:
    """Condition the admitted audio and write every derivative to the store.

    Args:
        store: The provenance store, already holding ADMIT's ``recording`` stream.
        source: The audio ADMIT returned, as supplied.
        config: The triage configuration.
        hint: Accepted for the shared node shape; not read.
        run_dir: Where the streams and sidecars are written.

    Returns:
        A pass verdict (PREPROCESS has no fail and no flag), the view over what was written, and the
        names of derivatives that are absent.
    """
    software = software_agent(store)
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)

    recording_ids = [
        e.id
        for e in store.entities("stream")
        if e.attributes.get("name") == "recording" and not store.is_invalidated(e.id)
    ]
    target_hz = int(config.require("resample.target_hz"))
    preemph_enabled = bool(config.require("preemphasis.enabled"))
    coefficient = float(config.require("preemphasis.coefficient"))

    condition = store.activity(
        node=NODE,
        step="condition",
        parameters={
            "target_hz": target_hz,
            "downmix": "mean",
            "preemphasis_enabled": preemph_enabled,
            "coefficient": coefficient,
        },
    )
    store.was_associated_with(condition, software)
    for recording_id in recording_ids:
        store.used(condition, recording_id)

    mono = Audio(waveform=source.waveform.mean(dim=0, keepdim=True), sampling_rate=source.sampling_rate)
    [plain] = resample_audios([mono], target_hz)
    peak = float(plain.waveform.abs().max())
    peak_scale = 1.0 if peak <= 1.0 else 1.0 / peak
    if peak_scale != 1.0:
        plain = Audio(waveform=plain.waveform * peak_scale, sampling_rate=target_hz)
    duration_s = plain.waveform.shape[-1] / target_hz
    plain.save_to_file(str(run_dir / "streams" / "plain.wav"))
    plain_id = store.entity(
        prov_type="stream",
        extent=(0.0, duration_s),
        attributes={
            "name": "plain",
            "path": "streams/plain.wav",
            "sampling_rate": target_hz,
            "channels": 1,
            "peak_scale": peak_scale,
        },
    )
    store.was_generated_by(plain_id, condition)
    store.was_attributed_to(plain_id, software)
    for recording_id in recording_ids:
        store.was_derived_from(plain_id, recording_id)

    if preemph_enabled:
        x = plain.waveform
        emphasised = torch.cat([x[:, :1], x[:, 1:] - coefficient * x[:, :-1]], dim=1)
        sharp = Audio(waveform=emphasised, sampling_rate=target_hz)
        sharp.save_to_file(str(run_dir / "streams" / "preemphasised.wav"))
        sharp_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "preemphasised",
                "path": "streams/preemphasised.wav",
                "sampling_rate": target_hz,
                "channels": 1,
                "coefficient": coefficient,
            },
        )
        store.was_generated_by(sharp_id, condition)
        store.was_attributed_to(sharp_id, software)
        store.was_derived_from(sharp_id, plain_id)
        sharp_signal = "preemphasised"
    else:
        sharp, sharp_id, sharp_signal = plain, plain_id, "plain"

    absent: dict[str, str] = {}
    derivatives: dict[str, Any] = {}
    view: list[str] = [plain_id] + ([sharp_id] if sharp_id != plain_id else [])
    state: dict[str, Any] = {}

    def _step(step: str, parameters: dict[str, Any], reads: tuple[str, ...], agent_id: str) -> str:
        """One sub-activity, associated and with its reads recorded."""
        activity_id = store.activity(node=NODE, step=step, parameters=parameters)
        store.was_associated_with(activity_id, agent_id)
        for entity_id in reads:
            store.used(activity_id, entity_id)
        return activity_id

    def _envelope() -> None:
        """`energy_envelope` and its floor, over the pre-emphasised signal, to one npz sidecar."""
        parameters = {
            "lowpass_hz": float(config.require("envelope.lowpass_hz")),
            "filter_order": int(config.require("envelope.filter_order")),
            "floor_window_s": float(config.require("floor.window_s")),
            "floor_percentile": float(config.require("floor.percentile")),
            "floor_eval_grid_s": float(config.require("floor.eval_grid_s")),
        }
        activity = _step("envelope", parameters, (sharp_id,), software)
        envelope = hilbert_envelope_dbfs(
            sharp, lowpass_hz=parameters["lowpass_hz"], filter_order=int(parameters["filter_order"])
        )
        floor = rolling_floor_dbfs(
            envelope,
            target_hz,
            window_s=parameters["floor_window_s"],
            percentile=parameters["floor_percentile"],
            eval_grid_s=parameters["floor_eval_grid_s"],
        )
        np.savez(run_dir / "derivatives" / "energy_envelope.npz", envelope_dbfs=envelope, floor_dbfs=floor)
        entity_id = _measurement(
            store,
            activity,
            software,
            name="energy_envelope",
            signal=sharp_signal,
            attributes={"path": "derivatives/energy_envelope.npz", "sampling_rate": target_hz},
            derived_from=(sharp_id,),
        )
        derivatives["energy_envelope"] = entity_id
        view.append(entity_id)
        state.update(envelope=envelope, floor=floor, envelope_id=entity_id)

    def _spans() -> None:
        """Span proposals at the airway K; `NoContrast` becomes a measurement, never an empty list."""
        if "envelope" not in state:
            raise LookupError("energy_envelope is absent")
        k_db = float(config.require("spans.k_db.airway"))
        parameters: dict[str, Any] = {
            "k_db": k_db,
            "onset_drop_db": float(config.require("spans.onset_drop_db")),
            "offset_fraction": float(config.require("spans.offset_fraction")),
            "hangover_ms": int(config.require("spans.hangover_ms")),
            "min_duration_ms": int(config.require("spans.min_duration_ms")),
            "min_separation_ms": int(config.require("spans.min_separation_ms")),
        }
        activity = _step("spans", parameters, (state["envelope_id"],), software)
        proposed = propose_spans(
            state["envelope"],
            state["floor"],
            target_hz,
            k_db=k_db,
            onset_drop_db=parameters["onset_drop_db"],
            offset_fraction=parameters["offset_fraction"],
            hangover_ms=parameters["hangover_ms"],
            min_duration_ms=parameters["min_duration_ms"],
            min_separation_ms=parameters["min_separation_ms"],
        )
        if isinstance(proposed, NoContrast):
            entity_id = _measurement(
                store,
                activity,
                software,
                name="spans_no_contrast",
                signal=sharp_signal,
                attributes={"k_db": k_db, "reason": proposed.reason},
                derived_from=(state["envelope_id"],),
            )
            derivatives["spans_no_contrast"] = entity_id
            view.append(entity_id)
            return
        span_ids: list[str] = []
        for span in proposed:
            span_id = store.entity(
                prov_type="span",
                extent=(span.start, span.end),
                attributes={"peak_over_floor_db": span.peak_over_floor_db, "k_db": k_db, "signal": sharp_signal},
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, state["envelope_id"])
            span_ids.append(span_id)
        derivatives["spans"] = span_ids
        view.extend(span_ids)
        state["span_ids"] = span_ids

    def _yamnet() -> None:
        """The full YAMNet native windows, to a json sidecar."""
        top_k = int(config.require("yamnet.top_k"))
        agent = store.agent(
            agent_type="model",
            model_id="https://tfhub.dev/google/yamnet/1",
            unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
        )
        activity = _step("yamnet", {"top_k": top_k}, (plain_id,), agent)
        [windows] = classify_audios([plain], model="yamnet", top_k=top_k)
        (run_dir / "derivatives" / "yamnet_windows.json").write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent,
            name="yamnet_windows",
            signal="plain",
            attributes={"path": "derivatives/yamnet_windows.json", "n_windows": len(windows)},
            derived_from=(plain_id,),
        )
        derivatives["yamnet_windows"] = entity_id
        view.append(entity_id)
        state.update(yamnet_windows=windows, yamnet_windows_id=entity_id)

    def _silence() -> None:
        """The Silence projection of the YAMNet windows."""
        if "yamnet_windows" not in state:
            raise LookupError("yamnet_windows is absent")
        threshold = float(config.require("yamnet.silence_threshold"))
        activity = _step("silence", {"threshold": threshold}, (state["yamnet_windows_id"],), software)
        rows = []
        for window in state["yamnet_windows"]:
            score = 0.0
            for pair in label_scores(window):
                if "Silence" in pair:
                    score = float(pair["Silence"])
                    break
            rows.append(
                {"start": window["start"], "end": window["end"], "score": score, "is_silence": score >= threshold}
            )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="silence",
            signal="plain",
            attributes={"threshold": threshold, "windows": rows},
            derived_from=(state["yamnet_windows_id"],),
        )
        derivatives["silence"] = entity_id
        view.append(entity_id)

    def _level() -> None:
        """File-level peak dBFS, RMS dBFS and LUFS on the plain signal."""
        activity = _step("level", {}, (plain_id,), software)
        x = plain.waveform.squeeze(0).numpy()
        peak_dbfs = float(20.0 * np.log10(max(float(np.abs(x).max()), 1e-12)))
        rms_dbfs = float(20.0 * np.log10(max(float(np.sqrt(np.mean(x**2))), 1e-12)))
        lufs = float(integrated_lufs(x, target_hz))
        entity_id = _measurement(
            store,
            activity,
            software,
            name="level",
            signal="plain",
            attributes={"peak_dbfs": peak_dbfs, "rms_dbfs": rms_dbfs, "lufs": lufs},
            derived_from=(plain_id,),
        )
        derivatives["level"] = entity_id
        view.append(entity_id)

    def _squim() -> None:
        """One objective-head measure assertion per envelope span; refusals recorded, never padded."""
        if not state.get("span_ids"):
            raise LookupError("spans are absent")
        agent = store.agent(
            agent_type="model",
            model_id="torchaudio SQUIM_OBJECTIVE",
            unresolved_reason=f"bundled torchaudio weights, version {_dist_version('torchaudio')}",
        )
        activity = _step("squim", {}, tuple(state["span_ids"]), agent)
        assertion_ids: list[str] = []
        for span_id in state["span_ids"]:
            span = store.get_entity(span_id)
            start, end = span.extent or (0.0, 0.0)
            segment = Audio(
                waveform=plain.waveform[:, int(start * target_hz) : int(end * target_hz)],
                sampling_rate=target_hz,
            )
            try:
                [scores] = extract_objective_quality_features_from_audios([segment])
                attributes: dict[str, Any] = {
                    "verb": "measure",
                    "name": "squim",
                    "stoi": float(scores["stoi"]),
                    "pesq": float(scores["pesq"]),
                    "si_sdr": float(scores["si_sdr"]),
                }
            except Exception as err:  # noqa: BLE001 — a span SQUIM refuses is unmeasured, not padded
                attributes = {"verb": "measure", "name": "squim", "unmeasured": type(err).__name__}
            assertion_id = store.entity(prov_type="assertion", extent=span.extent, attributes=attributes)
            store.was_generated_by(assertion_id, activity)
            store.was_attributed_to(assertion_id, agent)
            store.was_derived_from(assertion_id, span_id)
            assertion_ids.append(assertion_id)
        derivatives["squim"] = assertion_ids
        view.extend(assertion_ids)

    def _asr(
        name: str,
        factory: Callable[[], HFModel],
        source_kind: str,
        timing_model: str | None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """One recognizer: transcript measurement plus one word entity per chunk it timed in bounds.

        A chunk missing either bound is counted in ``untimed_chunks_n`` and written as no word; a
        chunk starting at or after the plain stream's duration is counted in
        ``out_of_bounds_chunks_n`` and likewise written as no word. Either way its text stays in the
        transcript. A chunk that starts in bounds and ends past the duration keeps its word, with
        the end bound by the duration.
        """
        model = factory()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        activity = _step(
            name, {"model": str(model.path_or_uri), **{k: str(v) for k, v in kwargs.items()}}, (plain_id,), agent
        )
        [line] = transcribe_audios([plain], model=model, **kwargs)
        word_ids: list[str] = []
        untimed_chunks_n = 0
        out_of_bounds_chunks_n = 0
        for chunk in line.chunks or []:
            if chunk.start is None or chunk.end is None:
                untimed_chunks_n += 1
                continue
            span = _bound_to_duration(float(chunk.start), float(chunk.end), duration_s)
            if span is None:
                out_of_bounds_chunks_n += 1
                continue
            start, end = span
            attributes: dict[str, Any] = {
                "text": chunk.text,
                "score": chunk.score,
                "recognizer": str(model.path_or_uri),
                "timestamp_source": source_kind,
            }
            if end < float(chunk.end):
                attributes["end_clamped_to_duration"] = True
            if timing_model is not None:
                attributes["timestamp_model"] = timing_model
            word_id = store.entity(
                prov_type="word",
                extent=(start, end),
                attributes=attributes,
            )
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, agent)
            word_ids.append(word_id)
        meta: dict[str, Any] = {
            "recognizer": str(model.path_or_uri),
            "transcript": line.text or "",
            "word_ids": word_ids,
            "untimed_chunks_n": untimed_chunks_n,
            "out_of_bounds_chunks_n": out_of_bounds_chunks_n,
            "timestamp_source": source_kind,
        }
        if timing_model is not None:
            meta["timestamp_model"] = timing_model
        entity_id = _measurement(
            store, activity, agent, name=name, signal="plain", attributes=meta, derived_from=(plain_id,)
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        state[name] = line
        state[name + "_id"] = entity_id

    def _agreement() -> None:
        """The fused word list over both recognizers — the derivative SPEECH reads.

        ``iter_word_leaves`` re-reads each recognizer's own output, which the bound on the word
        entities does not reach, so the same bound is applied to the fused list: a word starting at
        or past the plain stream's duration is dropped and counted in ``out_of_bounds_words_n``, and
        an end past the duration is bound by it. ``_alignment`` takes its transcript's end from this
        list, so bounding it here is what keeps a hallucinated timestamp out of the aligner's slice.
        """
        if "asr_crisperwhisper" not in state or "asr_qwen" not in state:
            raise LookupError("both recognizers are needed")
        activity = _step(
            "agreement",
            {"systems": [CRISPERWHISPER_ID, QWEN_ID]},
            (state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
            software,
        )
        streams = {
            CRISPERWHISPER_ID: iter_word_leaves([state["asr_crisperwhisper"].model_dump()]),
            QWEN_ID: iter_word_leaves([state["asr_qwen"].model_dump()]),
        }
        fused: list[dict[str, Any]] = []
        out_of_bounds_words_n = 0
        for word in fuse_word_streams(streams):
            span = _bound_to_duration(float(word["start"]), float(word["end"]), duration_s)
            if span is None:
                out_of_bounds_words_n += 1
                continue
            fused.append({**word, "start": span[0], "end": span[1]})
        entity_id = _measurement(
            store,
            activity,
            software,
            name="asr_agreement",
            signal="plain",
            attributes={
                "words": fused,
                "systems": [CRISPERWHISPER_ID, QWEN_ID],
                "out_of_bounds_words_n": out_of_bounds_words_n,
            },
            derived_from=(state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
        )
        derivatives["asr_agreement"] = entity_id
        view.append(entity_id)
        state.update(fused=fused, asr_agreement_id=entity_id)

    def _alignment() -> None:
        """Forced alignment of the agreed transcript, on the plain signal."""
        if not state.get("fused"):
            raise LookupError("asr_agreement is absent or empty")
        fused = state["fused"]
        agent = store.agent(
            agent_type="model",
            model_id=str(DEFAULT_ALIGN_MODELS_HF[ALIGNMENT_LANGUAGE]["path_or_uri"]),
            unresolved_reason="align_transcriptions loads its aligner internally; the commit is not reported",
        )
        activity = _step("alignment", {"language": ALIGNMENT_LANGUAGE}, (state["asr_agreement_id"],), agent)
        transcript = ScriptLine(
            text=" ".join(word["text"] for word in fused),
            start=min(word["start"] for word in fused),
            end=max(word["end"] for word in fused),
        )
        [aligned] = align_transcriptions([(plain, transcript, Language(language_code=ALIGNMENT_LANGUAGE))])
        payload = [line.model_dump() for line in aligned if line is not None]
        (run_dir / "derivatives" / "alignment.json").write_text(json.dumps(payload, default=str))
        entity_id = _measurement(
            store,
            activity,
            agent,
            name="alignment",
            signal="plain",
            attributes={
                "path": "derivatives/alignment.json",
                "language": ALIGNMENT_LANGUAGE,
                "transcript_source": "asr_agreement",
            },
            derived_from=(state["asr_agreement_id"],),
        )
        derivatives["alignment"] = entity_id
        view.append(entity_id)

    def _spectrogram(name: str, window_key: str) -> None:
        """One STFT magnitude, window and hop from the config, n_fft = win_length (decision N7)."""
        window_ms = float(config.require(window_key))
        hop_ms = float(config.require("spectrogram.hop_ms"))
        win_length = int(target_hz * window_ms / 1000.0)
        hop_length = int(target_hz * hop_ms / 1000.0)
        parameters = {"win_length": win_length, "hop_length": hop_length, "n_fft": win_length}
        activity = _step(name, parameters, (sharp_id,), software)
        [result] = extract_spectrogram_from_audios(
            [sharp], n_fft=win_length, win_length=win_length, hop_length=hop_length
        )
        np.savez(run_dir / "derivatives" / f"{name}.npz", spectrogram=result["spectrogram"].numpy())
        entity_id = _measurement(
            store,
            activity,
            software,
            name=name,
            signal=sharp_signal,
            attributes={"path": f"derivatives/{name}.npz", **parameters},
            derived_from=(sharp_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)

    def _gammatone() -> None:
        """The ERB-spaced filterbank energies, to one npz sidecar."""
        parameters: dict[str, Any] = {
            "n_channels": int(config.require("gammatone.n_channels")),
            "low_hz": float(config.require("gammatone.low_hz")),
            "high_hz": float(config.require("gammatone.high_hz")),
            "hop_s": float(config.require("gammatone.hop_s")),
        }
        activity = _step("gammatone", parameters, (sharp_id,), software)
        centre_frequencies, energy_db = gammatone_filterbank(
            sharp,
            n_channels=parameters["n_channels"],
            low_hz=parameters["low_hz"],
            high_hz=parameters["high_hz"],
            hop_s=parameters["hop_s"],
        )
        np.savez(
            run_dir / "derivatives" / "gammatone.npz",
            centre_frequencies_hz=centre_frequencies,
            energy_db=energy_db,
        )
        entity_id = _measurement(
            store,
            activity,
            software,
            name="gammatone",
            signal=sharp_signal,
            attributes={"path": "derivatives/gammatone.npz", "hop_s": parameters["hop_s"]},
            derived_from=(sharp_id,),
        )
        derivatives["gammatone"] = entity_id
        view.append(entity_id)

    blocks: list[tuple[str, Callable[[], None]]] = [
        ("energy_envelope", _envelope),
        ("spans", _spans),
        ("yamnet_windows", _yamnet),
        ("silence", _silence),
        ("level", _level),
        ("squim", _squim),
        ("asr_crisperwhisper", lambda: _asr("asr_crisperwhisper", _crisperwhisper_model, "native", None)),
        (
            "asr_qwen",
            lambda: _asr("asr_qwen", _qwen_model, "bundled_aligner", QWEN_TIMESTAMP_MODEL, return_timestamps=True),
        ),
        ("asr_agreement", _agreement),
        ("alignment", _alignment),
        ("spectrogram_wideband", lambda: _spectrogram("spectrogram_wideband", "spectrogram.wideband_window_ms")),
        (
            "spectrogram_narrowband",
            lambda: _spectrogram("spectrogram_narrowband", "spectrogram.narrowband_window_ms"),
        ),
        ("gammatone", _gammatone),
    ]
    for name, block in blocks:
        try:
            block()
        except Exception as err:  # noqa: BLE001 — an uncomputable derivative is absent, not an error
            absent[name] = type(err).__name__

    verdict_id, verdict = write_verdict(
        store,
        condition,
        software,
        node=NODE,
        outcome=Outcome.PASS,
        kind=None,
        why="conditioning complete; absent derivatives are listed",
        detail={"absent": dict(sorted(absent.items())), "derivatives": derivatives},
    )
    view.append(verdict_id)
    return PreprocessResult(
        verdict=verdict, view=tuple(view), verdict_entity_id=verdict_id, absent=tuple(sorted(absent))
    )
