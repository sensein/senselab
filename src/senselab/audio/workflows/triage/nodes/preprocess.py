"""PREPROCESS — one conditioning pass, every shared derivative written to the store.

Every model that answers a whole-file question runs here: YAMNet, AST and HeAR alike. No later node
re-runs one. The recognizers, the aligner, SQUIM, level and the window classifiers read the plain
resampled signal; the envelope, spans, spectrograms, gammatone and the phonation pass read the
pre-emphasised one; ``disruptions_file`` reads the original recording. A derivative that cannot be
computed is absent from the store, not an error. Every parameter's derivation is in
``data/config/default.yaml``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.tasks.classification.api import classify_audios
from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.tasks.disruptions.api import detect_disruptions
from senselab.audio.tasks.envelope.api import hilbert_envelope_dbfs, rolling_floor_dbfs
from senselab.audio.tasks.features_extraction.torchaudio import extract_spectrogram_from_audios
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
)
from senselab.audio.tasks.forced_alignment.constants import DEFAULT_ALIGN_MODELS_HF
from senselab.audio.tasks.forced_alignment.forced_alignment import align_transcriptions
from senselab.audio.tasks.gammatone.api import gammatone_filterbank
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.health_acoustics.hear import HEAR_MODEL_ID, HEAR_REVISION
from senselab.audio.tasks.phonation.api import f0_track, formant_track, propose_phonation_spans
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.spans.api import NoContrast, propose_spans
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.audio.workflows.audio_analysis.asr import fuse_consensus_words
from senselab.audio.workflows.audio_analysis.level import integrated_lufs
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    NodeResult,
    describe_exception,
    software_agent,
    write_verdict,
)
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import HFModel, Language, ScriptLine
from senselab.utils.prov_store import ProvStore

NODE = "PREPROCESS"
CRISPERWHISPER_ID = "nyralabs/CrisperWhisper2.0_turbo"
QWEN_ID = "Qwen/Qwen3-ASR-1.7B"
QWEN_TIMESTAMP_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
YAMNET_MODEL_URI = "https://tfhub.dev/google/yamnet/1"
ALIGNMENT_LANGUAGE = "en"


def _crisperwhisper_model() -> HFModel:
    """The CrisperWhisper model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=CRISPERWHISPER_ID, revision="main")


def _qwen_model() -> HFModel:
    """The Qwen3-ASR model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=QWEN_ID, revision="main")


def _ast_model() -> HFModel:
    """The AST model spec; its commit resolves at construction."""
    return HFModel(path_or_uri=AST_ID, revision="main")


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


def _norm_token(token: str) -> str:
    """A token normalised for vocabulary matching: casefolded, edge punctuation stripped."""
    return token.casefold().strip(".,;:!?\"'()")


def _as_non_word(text: str, onomatopoeic: set[str]) -> tuple[str | None, str | None]:
    """The bracketed form of a non-lexical token, or ``(None, None)`` when the token is a word.

    Args:
        text: The token as the recognizer produced it.
        onomatopoeic: The normalised ``words.onomatopoeic_tokens`` vocabulary; empty while it is null.

    Returns:
        ``(bracketed, origin)`` where ``origin`` is ``"bracketed"`` or ``"onomatopoeic"``, or
        ``(None, None)``.
    """
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return stripped, "bracketed"
    normalised = _norm_token(stripped)
    if normalised and normalised in onomatopoeic:
        return f"[{normalised.upper()}]", "onomatopoeic"
    return None, None


def _confident_labels(
    window: dict[str, Any], default_threshold: float, label_thresholds: dict[str, float]
) -> dict[str, float]:
    """The labels this window is confident of, each with the score behind it.

    A label is a member iff its score clears its own threshold — ``label_thresholds[label]`` where
    one exists, ``default_threshold`` otherwise. The result may be empty, which is a window nobody's
    threshold cleared and is a different fact from a window that was never classified.

    Args:
        window: A classifier window, in the shape ``label_scores`` reads.
        default_threshold: The threshold for a label with no entry of its own.
        label_thresholds: Per-label thresholds.

    Returns:
        ``{label: score}`` over the members, in descending score order.
    """
    members: dict[str, float] = {}
    for pair in label_scores(window):
        for label, score in pair.items():
            if float(score) >= float(label_thresholds.get(label, default_threshold)):
                members[label] = float(score)
    return dict(sorted(members.items(), key=lambda item: -item[1]))


def _measurement(
    store: ProvStore,
    activity_id: str,
    agent_id: str,
    *,
    name: str,
    signal: str,
    attributes: dict[str, Any],
    derived_from: tuple[str, ...] = (),
    extent: tuple[float, float] | None = None,
) -> str:
    """Write one derivative measurement entity with its provenance."""
    entity_id = store.entity(
        prov_type="measurement", extent=extent, attributes={"name": name, "signal": signal, **attributes}
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
                attributes={
                    "peak_over_floor_db": span.peak_over_floor_db,
                    "k_db": k_db,
                    "signal": sharp_signal,
                    "merged_proposals": span.merged_proposals,
                },
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, state["envelope_id"])
            span_ids.append(span_id)
        derivatives["spans"] = span_ids
        view.extend(span_ids)
        state["span_ids"] = span_ids

    def _scores(name: str, agent_id: str, activity_step: str, run: Callable[[], list[dict[str, Any]]]) -> None:
        """Run one classifier and store its verbatim windows; no threshold is read here (V3)."""
        activity = _step(activity_step, {}, (plain_id,), agent_id)
        windows = run()
        path = f"derivatives/{name}.json"
        (run_dir / path).write_text(json.dumps(windows))
        entity_id = _measurement(
            store,
            activity,
            agent_id,
            name=name,
            signal="plain",
            attributes={
                "classifier": name.removesuffix("_scores"),
                "path": path,
                "n_windows": len(windows),
                "win_length_s": float(windows[0]["win_length"]) if windows else None,
                "hop_s": float(windows[0]["hop_length"]) if windows else None,
            },
            derived_from=(plain_id,),
        )
        derivatives[name] = entity_id
        view.append(entity_id)
        state[name] = windows
        state[name + "_id"] = entity_id

    def _windows(classifier: str) -> None:
        """Fold the thresholds over one classifier's stored scores into per-window label sets."""
        scores_name = f"{classifier}_scores"
        if scores_name not in state:
            raise LookupError(f"{scores_name} is absent")
        default_threshold = float(config.require(f"windows.{classifier}.default_threshold"))
        label_thresholds = {
            str(label): float(value)
            for label, value in config.require(f"windows.{classifier}.label_thresholds").items()
        }
        activity = _step(
            f"{classifier}_windows",
            {"default_threshold": default_threshold, "label_thresholds": label_thresholds},
            (state[scores_name + "_id"],),
            software,
        )
        raw = state[scores_name]
        window_ids: list[str] = []
        windows_by_label: dict[str, list[str]] = {}
        fired: dict[str, float] = {}
        for raw_window in raw:
            members = _confident_labels(raw_window, default_threshold, label_thresholds)
            window_id = store.entity(
                prov_type="measurement",
                extent=(float(raw_window["start"]), float(raw_window["end"])),
                attributes={
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": list(members),
                    "scores": members,
                },
            )
            store.was_generated_by(window_id, activity)
            store.was_attributed_to(window_id, software)
            store.was_derived_from(window_id, state[scores_name + "_id"])
            window_ids.append(window_id)
            for label in members:
                windows_by_label.setdefault(label, []).append(window_id)
                if label in label_thresholds:
                    fired[label] = label_thresholds[label]
        entity_id = _measurement(
            store,
            activity,
            software,
            name=f"{classifier}_windows",
            signal="plain",
            attributes={
                "classifier": classifier,
                "labels": sorted(windows_by_label),
                "windows_by_label": windows_by_label,
                "n_windows": len(raw),
                "win_length_s": float(raw[0]["win_length"]) if raw else None,
                "hop_s": float(raw[0]["hop_length"]) if raw else None,
                "default_threshold": default_threshold,
                "label_thresholds": fired,
            },
            derived_from=(state[scores_name + "_id"],),
        )
        derivatives[f"{classifier}_windows"] = entity_id
        view.append(entity_id)
        view.extend(window_ids)

    def _yamnet_scores() -> None:
        """YAMNet on its own native grid; `win_length`/`hop_length` are ignored by this backend."""
        _scores(
            "yamnet_scores",
            store.agent(
                agent_type="model",
                model_id=YAMNET_MODEL_URI,
                unresolved_reason="TF-Hub URL pin; no commit exists to resolve",
            ),
            "yamnet",
            lambda: classify_audios([plain], model="yamnet", top_k=int(config.require("yamnet.top_k")))[0],
        )

    def _ast_scores() -> None:
        """AST at the configured window and hop, over its whole label space (C1, C2)."""
        model = _ast_model()
        _scores(
            "ast_scores",
            store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha),
            "ast",
            lambda: classify_audios(
                [plain],
                model=model,
                win_length=float(config.require("windows.ast.win_length_s")),
                hop_length=float(config.require("windows.ast.hop_s")),
                top_k=int(config.require("windows.ast.top_k")),
                function_to_apply="sigmoid",
            )[0],
        )

    def _hear_scores() -> None:
        """HeAR at its model-imposed 2 s window and the configured hop; `top_k=None` keeps all eight."""
        _scores(
            "hear_scores",
            store.agent(agent_type="model", model_id=HEAR_MODEL_ID, commit_sha=HEAR_REVISION),
            "hear",
            lambda: detect_health_acoustic_events(
                [plain], hop_length=float(config.require("windows.hear.hop_s")), top_k=None
            )[0],
        )

    def _silence() -> None:
        """The Silence projection of the stored YAMNet scores."""
        if "yamnet_scores" not in state:
            raise LookupError("yamnet_scores is absent")
        threshold = float(config.require("yamnet.silence_threshold"))
        activity = _step("silence", {"threshold": threshold}, (state["yamnet_scores_id"],), software)
        rows = []
        for window in state["yamnet_scores"]:
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
            derived_from=(state["yamnet_scores_id"],),
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

    def _disruptions_file() -> None:
        """Clipping, dropouts, discontinuities, DC and ZCR over the whole ORIGINAL recording."""
        if not recording_ids:
            raise LookupError("no recording stream in the store")
        parameters: dict[str, Any] = {
            "clip_headroom": float(config.require("disruptions.clip_headroom")),
            "min_clip_run": int(config.require("disruptions.min_clip_run")),
            "min_dropout_ms": float(config.require("disruptions.min_dropout_ms")),
            "discontinuity_local_factor": float(config.require("disruptions.discontinuity_local_factor")),
            "discontinuity_window_ms": float(config.require("disruptions.discontinuity_window_ms")),
        }
        activity = _step("disruptions_file", parameters, (recording_ids[-1],), software)
        original_duration = source.waveform.shape[-1] / int(source.sampling_rate)
        found = detect_disruptions(source, 0.0, original_duration, **parameters)
        counts = {key: value for key, value in asdict(found).items() if key not in ("start", "end")}
        entity_id = _measurement(
            store,
            activity,
            software,
            name="disruptions_file",
            signal="recording",
            attributes={**counts, "sampling_rate": int(source.sampling_rate)},
            derived_from=(recording_ids[-1],),
        )
        derivatives["disruptions_file"] = entity_id
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
        """One recognizer: its transcript and its own word list, retained as the consensus's evidence.

        No ``word`` entity is written here. PREPROCESS writes those once, over the consensus, so a
        consumer never has to disambiguate two populations of ``word`` by generating activity.
        """
        model = factory()
        agent = store.agent(agent_type="model", model_id=str(model.path_or_uri), commit_sha=model.commit_sha)
        activity = _step(
            name, {"model": str(model.path_or_uri), **{k: str(v) for k, v in kwargs.items()}}, (plain_id,), agent
        )
        [line] = transcribe_audios([plain], model=model, **kwargs)
        words: list[dict[str, Any]] = []
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
            words.append({"text": chunk.text, "start": span[0], "end": span[1], "score": chunk.score})
        meta: dict[str, Any] = {
            "recognizer": str(model.path_or_uri),
            "transcript": line.text or "",
            "words": words,
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
        state[name] = line
        state[name + "_id"] = entity_id

    def _consensus() -> None:
        """The consensus over both recognizers, by the audio-analysis routine, plus its word entities."""
        if "asr_crisperwhisper" not in state or "asr_qwen" not in state:
            raise LookupError("both recognizers are needed")
        activity = _step(
            "consensus",
            {"systems": [CRISPERWHISPER_ID, QWEN_ID], "routine": "fuse_consensus_words"},
            (state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
            software,
        )
        fused, provenance = fuse_consensus_words(
            {CRISPERWHISPER_ID: state["asr_crisperwhisper"], QWEN_ID: state["asr_qwen"]}
        )
        if not provenance:
            provenance = {"operator": "consensus_words/resample", "sources": [], "n_words": 0}
        onomatopoeic = {_norm_token(str(token)) for token in (config.get("words.onomatopoeic_tokens") or [])}
        word_ids: list[str] = []
        event_ids: list[str] = []
        kept: list[dict[str, Any]] = []
        for entry in fused:
            span = _bound_to_duration(float(entry["start"]), float(entry["end"]), duration_s)
            if span is None:
                continue
            text = str(entry.get("text") or "")
            recognizers = [str(s) for s in (entry.get("sources") or [])]
            bracketed, origin = _as_non_word(text, onomatopoeic)
            if bracketed is not None:
                event_id = store.entity(
                    prov_type="event",
                    extent=span,
                    attributes={
                        "bracketed": bracketed,
                        "raw": text,
                        "origin": origin,
                        "recognizers": recognizers,
                    },
                )
                store.was_generated_by(event_id, activity)
                store.was_attributed_to(event_id, software)
                event_ids.append(event_id)
                continue
            word_id = store.entity(
                prov_type="word",
                extent=span,
                attributes={
                    "text": text,
                    "confidence": entry.get("confidence"),
                    "existence_confidence": entry.get("existence_confidence"),
                    "temporal_confidence": entry.get("temporal_confidence"),
                    "coverage": entry.get("coverage"),
                    "recognizers": recognizers,
                    "timing_sources": entry.get("timing_sources"),
                    "index": len(kept),
                },
            )
            store.was_generated_by(word_id, activity)
            store.was_attributed_to(word_id, software)
            word_ids.append(word_id)
            kept.append({**entry, "start": span[0], "end": span[1]})
        entity_id = _measurement(
            store,
            activity,
            software,
            name="consensus_transcript",
            signal="plain",
            attributes={
                "words": kept,
                "provenance": provenance,
                "systems": [CRISPERWHISPER_ID, QWEN_ID],
                "word_ids": word_ids,
                "event_ids": event_ids,
                "text": " ".join(str(entry.get("text") or "") for entry in kept),
            },
            derived_from=(state["asr_crisperwhisper_id"], state["asr_qwen_id"]),
        )
        derivatives["consensus_transcript"] = entity_id
        view.append(entity_id)
        view.extend(word_ids)
        view.extend(event_ids)
        state.update(consensus=kept, consensus_id=entity_id)

    def _alignment() -> None:
        """Forced alignment of the consensus transcript, on the plain signal."""
        if not state.get("consensus"):
            raise LookupError("consensus_transcript is absent or empty")
        consensus = state["consensus"]
        agent = store.agent(
            agent_type="model",
            model_id=str(DEFAULT_ALIGN_MODELS_HF[ALIGNMENT_LANGUAGE]["path_or_uri"]),
            unresolved_reason="align_transcriptions loads its aligner internally; the commit is not reported",
        )
        activity = _step("alignment", {"language": ALIGNMENT_LANGUAGE}, (state["consensus_id"],), agent)
        transcript = ScriptLine(
            text=" ".join(str(word["text"]) for word in consensus),
            start=min(float(word["start"]) for word in consensus),
            end=max(float(word["end"]) for word in consensus),
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
                "transcript_source": "consensus_transcript",
            },
            derived_from=(state["consensus_id"],),
        )
        derivatives["alignment"] = entity_id
        view.append(entity_id)

    def _phonation_spans() -> None:
        """Sustained-phonation and glide spans, from tracks computed once over the whole stream."""
        f0_range = config.require("voice.f0_range_hz")
        f0_min_hz, f0_max_hz = float(f0_range[0]), float(f0_range[1])
        parameters: dict[str, Any] = {
            "hop_s": float(config.require("phonation_spans.hop_s")),
            "f0_stability_cents": float(config.require("phonation_spans.f0_stability_cents")),
            "formant_stability_hz": float(config.require("phonation_spans.formant_stability_hz")),
            "glide_min_excursion_cents": float(config.require("phonation_spans.glide_min_excursion_cents")),
            "hangover_ms": float(config.require("phonation_spans.hangover_ms")),
            "voicing_strength_floor": float(config.require("phonation_spans.voicing_strength_floor")),
            "mixed_voiced_fraction": float(config.require("phonation_spans.mixed_voiced_fraction")),
            "max_formants": int(config.require("phonation_spans.max_formants")),
            "formant_max_hz": float(config.require("phonation_spans.formant_max_hz")),
            "formant_window_s": float(config.require("phonation_spans.formant_window_s")),
            "formant_preemphasis_hz": float(config.require("phonation_spans.formant_preemphasis_hz")),
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
        }
        activity = _step("phonation_spans", parameters, (sharp_id,), software)
        times, f0_hz, strength = f0_track(sharp, f0_min_hz=f0_min_hz, f0_max_hz=f0_max_hz, hop_s=parameters["hop_s"])
        formants = formant_track(
            sharp,
            hop_s=parameters["hop_s"],
            max_formants=parameters["max_formants"],
            formant_max_hz=parameters["formant_max_hz"],
            window_s=parameters["formant_window_s"],
            preemphasis_hz=parameters["formant_preemphasis_hz"],
        )
        proposals = propose_phonation_spans(
            times=times,
            f0_hz=f0_hz,
            strength=strength,
            formants=formants,
            hop_s=parameters["hop_s"],
            f0_stability_cents=parameters["f0_stability_cents"],
            formant_stability_hz=parameters["formant_stability_hz"],
            glide_min_excursion_cents=parameters["glide_min_excursion_cents"],
            hangover_ms=parameters["hangover_ms"],
            voicing_strength_floor=parameters["voicing_strength_floor"],
            mixed_voiced_fraction=parameters["mixed_voiced_fraction"],
        )
        span_ids: list[str] = []
        for proposal in proposals:
            span_id = store.entity(
                prov_type="span",
                extent=(proposal.start, proposal.end),
                attributes={
                    "family": "phonation",
                    "member": proposal.member,
                    "duration_s": proposal.end - proposal.start,
                    "production": proposal.production,
                    "voiced_fraction": proposal.voiced_fraction,
                    "f0_median_hz": proposal.f0_median_hz,
                    "f0_start_hz": proposal.f0_start_hz,
                    "f0_end_hz": proposal.f0_end_hz,
                    "glide_direction": proposal.glide_direction,
                    "glide_extent_cents": proposal.glide_extent_cents,
                    "offset_criterion": proposal.offset_criterion,
                    "signal": sharp_signal,
                    "hop_s": parameters["hop_s"],
                },
            )
            store.was_generated_by(span_id, activity)
            store.was_attributed_to(span_id, software)
            store.was_derived_from(span_id, sharp_id)
            span_ids.append(span_id)
            inside = (formants.times_s >= proposal.start) & (formants.times_s < proposal.end)
            track_id = _measurement(
                store,
                activity,
                software,
                name="formant_tracks",
                signal=sharp_signal,
                extent=(proposal.start, proposal.end),
                attributes={
                    "times_s": formants.times_s[inside].tolist(),
                    "hop_s": parameters["hop_s"],
                    **{f"f{order + 1}_hz": formants.f_hz[order][inside].tolist() for order in range(4)},
                    **{f"f{order + 1}_bw_hz": formants.bandwidth_hz[order][inside].tolist() for order in range(4)},
                },
                derived_from=(span_id,),
            )
            view.append(track_id)
        derivatives["phonation_spans"] = span_ids
        view.extend(span_ids)

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
        ("yamnet_scores", _yamnet_scores),
        ("yamnet_windows", lambda: _windows("yamnet")),
        ("silence", _silence),
        ("ast_scores", _ast_scores),
        ("ast_windows", lambda: _windows("ast")),
        ("hear_scores", _hear_scores),
        ("hear_windows", lambda: _windows("hear")),
        ("level", _level),
        ("disruptions_file", _disruptions_file),
        ("squim", _squim),
        ("asr_crisperwhisper", lambda: _asr("asr_crisperwhisper", _crisperwhisper_model, "native", None)),
        (
            "asr_qwen",
            lambda: _asr("asr_qwen", _qwen_model, "bundled_aligner", QWEN_TIMESTAMP_MODEL, return_timestamps=True),
        ),
        ("consensus_transcript", _consensus),
        ("alignment", _alignment),
        ("phonation_spans", _phonation_spans),
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
            absent[name] = describe_exception(err)

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
