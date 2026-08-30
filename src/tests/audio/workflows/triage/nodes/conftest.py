"""Shared fixtures for the triage node tests. Nothing here loads a model."""

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.utils.prov_store import ProvStore


@pytest.fixture
def config() -> TriageConfig:
    """The packaged configuration, unmodified."""
    return load_triage_config()


@pytest.fixture
def store() -> ProvStore:
    """An empty store for one test run."""
    return ProvStore(run_id="test-run")


@pytest.fixture
def wav_writer(tmp_path: Path) -> Callable[..., Path]:
    """A writer for mono or stereo float32 WAV fixtures under this test's tmp dir."""

    def _write(name: str, samples: np.ndarray, sampling_rate: int = 16000) -> Path:
        path = tmp_path / name
        sf.write(str(path), samples.astype(np.float32), sampling_rate)
        return path

    return _write


def burst_samples(duration_s: float = 3.0, sampling_rate: int = 16000) -> np.ndarray:
    """A quiet noise bed with one loud 150 ms tone burst at 1.5 s.

    The burst stands far more than 18 dB over the bed, so `propose_spans` at the airway `K`
    proposes exactly one span over it.
    """
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(duration_s * sampling_rate)) * 1e-4).astype(np.float32)
    i0 = int(1.5 * sampling_rate)
    i1 = i0 + int(0.15 * sampling_rate)
    t = np.arange(i1 - i0) / sampling_rate
    x[i0:i1] += (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    return x


@pytest.fixture
def windows_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with every window threshold supplied, so the folds can run.

    The hops are left at their shipped non-overlapping defaults: they are declared values, not
    open keys, and a fixture that overrode them would stop testing what production reads.
    """
    override = tmp_path / "windows.yaml"
    override.write_text(
        "windows:\n"
        "  yamnet:\n"
        "    default_threshold: 0.5\n"
        "    label_thresholds: {Speech: 0.4}\n"
        "  ast:\n"
        "    default_threshold: 0.3\n"
        "    label_thresholds: {}\n"
        "  hear:\n"
        "    default_threshold: 0.5\n"
        "    label_thresholds: {}\n"
    )
    return load_triage_config(override)


@pytest.fixture
def phonation_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with the F0 range and every phonation-span criterion supplied.

    The values are a test fixture, not a fit: the packaged file leaves each of them null, and this
    is the override mechanism a caller would use to state which population is being measured.
    """
    override = tmp_path / "phonation.yaml"
    override.write_text(
        "voice:\n"
        "  f0_range_hz: [75.0, 500.0]\n"
        "phonation_spans:\n"
        "  f0_stability_cents: 50.0\n"
        "  formant_stability_hz: 50.0\n"
        "  glide_min_excursion_cents: 200.0\n"
        "  hangover_ms: 50.0\n"
        "  voicing_strength_floor: 0.5\n"
        "  mixed_voiced_fraction: 0.6\n"
        "  unvoiced_max_formant_bandwidth_hz: 250.0\n"
        "  word_aligned_min_evidence_fraction: 0.8\n"
    )
    return load_triage_config(override)


@pytest.fixture
def target_spans_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with the clip-grouping and normalization keys supplied.

    The values are a test fixture, not a fit: the packaged file leaves each of them null, and this
    is the override mechanism a caller would use to state them for a real campaign.
    """
    override = tmp_path / "target_spans.yaml"
    override.write_text(
        "clipping:\n"
        "  min_duration_ms: 0.5\n"
        "  merge_gap_ms: 50.0\n"
        "normalization:\n"
        "  macro_smoothing:\n"
        "    window_s: 0.5\n"
        "  micro_smoothing:\n"
        "    window_s: 0.05\n"
        "  envelope_smoothing:\n"
        "    window_s: 0.025\n"
        "  target_dr_db: 15.0\n"
        "  compression_ratio: 2.0\n"
        "  macro_target_dbfs: -6.0\n"
        "  gain_smooth_hz: 10.0\n"
        "  gain_filter_order: 1\n"
        "  floor_dbfs: -100.0\n"
        "  ceiling: 0.95\n"
        "  floor_window_s: 0.3\n"
        "  floor_percentile: 10.0\n"
        "  floor_eval_grid_s: 0.01\n"
        "spans:\n"
        "  k_db:\n"
        "    target: 12.0\n"
    )
    return load_triage_config(override)


def window(start: float, end: float, scores: dict[str, float]) -> dict[str, Any]:
    """One classifier window in the shape ``label_scores`` reads."""
    ordered = sorted(scores.items(), key=lambda pair: -pair[1])
    return {
        "start": start,
        "end": end,
        "label_scores": [{label: score} for label, score in ordered],
        "win_length": end - start,
        "hop_length": end - start,
    }


def _timed(
    entries: list[Any], duration_s: float, slot_s: float = 0.4, first_s: float = 0.5
) -> list[tuple[str, tuple[float, float]]]:
    """Give every bare token an extent, leaving an already-timed one alone."""
    placed: list[tuple[str, tuple[float, float]]] = []
    for index, entry in enumerate(entries):
        if isinstance(entry, tuple):
            placed.append((str(entry[0]), (float(entry[1][0]), float(entry[1][1]))))
            continue
        start = min(first_s + index * slot_s, max(0.0, duration_s - slot_s))
        placed.append((str(entry), (start, min(start + slot_s * 0.75, duration_s))))
    return placed


def _grid(labels: list[list[str]], win_s: float, hop_s: float) -> list[tuple[float, float, list[str]]]:
    """Place one label set per window on a fixed window/hop grid."""
    return [(index * hop_s, index * hop_s + win_s, list(entry)) for index, entry in enumerate(labels)]


@pytest.fixture
def seed_preprocess_store(tmp_path: Path) -> Callable[..., None]:
    """Write the entities PREPROCESS would have left behind, for a node test downstream of it.

    Every argument defaults to ``None``, which writes **nothing** for that derivative — that is how a
    test sets up an ``unavailable`` line, and it is a different state from passing an empty list,
    which writes the derivative and records that it found nothing.

    Args:
        tmp_path: Where the seeded stream WAV and derivative sidecars are written.
        store: The store to seed.
        stream_hz: The ``plain`` stream's rate. A silent mono WAV of ``duration_s`` is written under
            ``tmp_path`` and both the ``recording`` and ``plain`` stream entities point at it.
        duration_s: The streams' duration.
        yamnet_labels: One label list per YAMNet window, on a 0.96 s / 0.48 s grid. ``None`` writes no
            YAMNet measurement at all.
        ast_labels: The same on the owner-directed 10.24 s / 10.24 s grid, for AST.
        hear_labels: The same on a 2 s / 2 s grid, for HeAR.
        scores_only: Classifiers for which only the ``<classifier>_scores`` record is written and the
            threshold fold is left absent -- the state the **packaged config actually produces**,
            where every threshold is null so the model ran but no label set exists. Without this a
            test could not seed the shipped configuration's own store.
        words: The consensus words, as ``[text, ...]`` or ``[(text, (start, end)), ...]``. **An empty
            list still writes a ``consensus_transcript`` measurement carrying no words** — PREPROCESS
            fusing to nothing is not PREPROCESS never having run, and TAXONOMY's lexical line reads
            ``absent`` in the first case and ``unavailable`` in the second. ``None`` writes neither.
        events: Bracketed or onomatopoeic non-words, same shapes as ``words``. ``[]`` writes the
            ``PREPROCESS``/``consensus`` activity and no event, so a reader can tell "the consensus
            ran and produced no event" from "no consensus ran"; ``None`` writes neither.
        phonation: ``[(start, end, production), ...]`` or ``[(start, end, production, member), ...]``
            phonation spans -- ``member`` is ``"sustained"`` by default and ``"glide"`` gives the span
            a direction and an excursion, which T5 and T6 both need. Written with the
            ``PREPROCESS``/``phonation_spans`` activity that says the pass ran. ``[]`` writes the
            activity and no spans; ``None`` writes neither, which is the ``unavailable`` case.
        spans: ``[(start, end, peak_over_floor_db), ...]`` envelope spans at ``span_k_db``. ``[]``
            writes the ``PREPROCESS``/``spans`` activity and no span -- the spans pass ran and
            proposed nothing -- while ``None`` writes neither.
        span_k_db: The ``k_db`` those spans were proposed at.
        span_merged: The ``merged_proposals`` count every seeded envelope span carries.
        disruptions_file: Whether to write the file-level disruption measurement.

    Returns:
        A callable taking ``(store, **the above)`` and writing them. It returns None; a test reads
        what it needs back out of the store, which is what makes these tests behavioural.
    """

    def _seed(  # noqa: C901 — one independent block per derivative, as the node itself has
        store: ProvStore,
        *,
        stream_hz: int = 16000,
        duration_s: float = 5.0,
        yamnet_labels: list[list[str]] | None = None,
        ast_labels: list[list[str]] | None = None,
        hear_labels: list[list[str]] | None = None,
        scores_only: tuple[str, ...] = (),
        words: list[Any] | None = None,
        events: list[Any] | None = None,
        phonation: list[tuple[Any, ...]] | None = None,
        spans: list[tuple[float, float, float]] | None = None,
        span_k_db: float = 18.0,
        span_merged: int = 1,
        disruptions_file: bool = False,
    ) -> None:
        from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID

        (tmp_path / "streams").mkdir(exist_ok=True)
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        name = f"plain-{store.run_id}.wav"
        sf.write(str(tmp_path / "streams" / name), np.zeros(int(duration_s * stream_hz), dtype=np.float32), stream_hz)
        activity = store.activity(node="PREPROCESS", step="seed", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(activity, agent)

        def _write(prov_type: str, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
            """One seeded entity with PREPROCESS's generating activity."""
            entity_id = store.entity(prov_type=prov_type, extent=extent, attributes=attributes)  # type: ignore[arg-type]
            store.was_generated_by(entity_id, activity)
            store.was_attributed_to(entity_id, agent)
            return entity_id

        for stream in ("recording", "plain"):
            _write(
                "stream",
                (0.0, duration_s),
                {
                    "name": stream,
                    "path": f"streams/{name}",
                    "sampling_rate": stream_hz,
                    "channels": 1,
                    **({"peak_scale": 1.0} if stream == "plain" else {}),
                },
            )

        for classifier, labels, win_s, hop_s in (
            ("yamnet", yamnet_labels, 0.96, 0.48),
            ("ast", ast_labels, 10.24, 10.24),
            ("hear", hear_labels, 2.0, 2.0),
        ):
            if labels is None:
                continue
            _write(
                "measurement",
                None,
                {
                    "name": f"{classifier}_scores",
                    "classifier": classifier,
                    "signal": "plain",
                    "path": f"derivatives/{classifier}_scores.json",
                    "n_windows": len(labels),
                    "win_length_s": win_s if labels else None,
                    "hop_s": hop_s if labels else None,
                },
            )
            if classifier in scores_only:
                continue
            windows_by_label: dict[str, list[str]] = {}
            for start, end, members in _grid(labels, win_s, hop_s):
                window_id = _write(
                    "measurement",
                    (start, end),
                    {
                        "name": f"{classifier}_window",
                        "classifier": classifier,
                        "signal": "plain",
                        "labels": list(members),
                        "scores": {label: 0.9 for label in members},
                    },
                )
                for label in members:
                    windows_by_label.setdefault(label, []).append(window_id)
            _write(
                "measurement",
                None,
                {
                    "name": f"{classifier}_windows",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": sorted(windows_by_label),
                    "windows_by_label": windows_by_label,
                    "n_windows": len(labels),
                    "win_length_s": win_s if labels else None,
                    "hop_s": hop_s if labels else None,
                    "default_threshold": 0.5,
                    "label_thresholds": {},
                },
            )

        if words is not None or events is not None:
            store.was_associated_with(store.activity(node="PREPROCESS", step="consensus", parameters={}), agent)
        event_ids: list[str] = []
        for text, extent in _timed(list(events if events is not None else []), duration_s):
            event_ids.append(
                _write(
                    "event",
                    extent,
                    {
                        "bracketed": text if text.startswith("[") else f"[{text.upper()}]",
                        "raw": text,
                        "origin": "bracketed" if text.startswith("[") else "onomatopoeic",
                        "recognizers": [CRISPERWHISPER_ID, QWEN_ID],
                    },
                )
            )

        if words is not None:
            placed = _timed(list(words), duration_s)
            word_ids = [
                _write(
                    "word",
                    extent,
                    {
                        "text": text,
                        "confidence": 0.9,
                        "existence_confidence": 0.9,
                        "temporal_confidence": 0.9,
                        "coverage": 1.0,
                        "recognizers": [CRISPERWHISPER_ID, QWEN_ID],
                        "timing_sources": 2,
                        "index": index,
                    },
                )
                for index, (text, extent) in enumerate(placed)
            ]
            _write(
                "measurement",
                None,
                {
                    "name": "consensus_transcript",
                    "signal": "plain",
                    "words": [
                        {"text": text, "start": extent[0], "end": extent[1], "sources": [CRISPERWHISPER_ID, QWEN_ID]}
                        for text, extent in placed
                    ],
                    "provenance": {
                        "operator": "consensus_words/resample",
                        "sources": [CRISPERWHISPER_ID, QWEN_ID],
                        "n_words": len(placed),
                    },
                    "systems": [CRISPERWHISPER_ID, QWEN_ID],
                    "word_ids": word_ids,
                    "event_ids": event_ids,
                    "text": " ".join(text for text, _ in placed),
                },
            )

        if phonation is not None:
            store.was_associated_with(store.activity(node="PREPROCESS", step="phonation_spans", parameters={}), agent)
            for entry in phonation:
                start, end, production = entry[0], entry[1], entry[2]
                member = entry[3] if len(entry) > 3 else "sustained"
                _write(
                    "span",
                    (start, end),
                    {
                        "family": "phonation",
                        "member": member,
                        "duration_s": end - start,
                        "production": production,
                        "voiced_fraction": 1.0 if production == "voiced" else 0.0,
                        "f0_median_hz": 200.0 if production == "voiced" else None,
                        "f0_start_hz": 200.0 if production == "voiced" else None,
                        "f0_end_hz": 200.0 if production == "voiced" else None,
                        "glide_direction": "rising" if member == "glide" else None,
                        "glide_extent_cents": 900.0 if member == "glide" else None,
                        "offset_criterion": "monotonicity" if member == "glide" else "f0_stability",
                        "signal": "preemphasised",
                        "hop_s": 0.01,
                    },
                )

        if spans is not None:
            store.was_associated_with(store.activity(node="PREPROCESS", step="spans", parameters={}), agent)
        for start, end, peak in spans if spans is not None else []:
            _write(
                "span",
                (start, end),
                {
                    "peak_over_floor_db": peak,
                    "k_db": span_k_db,
                    "signal": "preemphasised",
                    "merged_proposals": span_merged,
                },
            )

        if disruptions_file:
            _write(
                "measurement",
                None,
                {
                    "name": "disruptions_file",
                    "signal": "recording",
                    "clipped_runs": 0,
                    "clipped_s": 0.0,
                    "dropout_runs": 0,
                    "dropout_s": 0.0,
                    "discontinuities": 0,
                    "dc_offset": 0.0,
                    "zero_crossing_rate": 0.0,
                    "sampling_rate": stream_hz,
                },
            )

    return _seed


@pytest.fixture
def seed_store(tmp_path: Path) -> Callable[..., dict]:
    """A builder writing PREPROCESS-shaped entities into a store — the Task 2 schema, seeded.

    Writes a real plain-stream WAV so nodes that slice audio can, and entities for spans, words,
    YAMNet windows, silence and no_contrast as requested.
    """

    def _seed(
        store: ProvStore,
        *,
        spans: tuple = (),
        words: tuple = (),
        yamnet_windows: list | None = None,
        silence_windows: list | None = None,
        no_contrast_k: float | None = None,
        asr_available: bool = True,
        k_db: float = 18.0,
        duration_s: float = 4.0,
    ) -> dict:
        (tmp_path / "streams").mkdir(exist_ok=True)
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        sf.write(str(tmp_path / "streams" / "plain.wav"), burst_samples(duration_s=duration_s), 16000)
        activity = store.activity(node="PREPROCESS", step="seed", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(activity, agent)
        ids: dict = {"spans": [], "words": []}

        plain_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "plain",
                "path": "streams/plain.wav",
                "sampling_rate": 16000,
                "channels": 1,
                "peak_scale": 1.0,
            },
        )
        store.was_generated_by(plain_id, activity)
        ids["plain"] = plain_id

        for start, end, contrast in spans:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": contrast, "k_db": k_db, "signal": "preemphasised"},
            )
            store.was_generated_by(span_id, activity)
            ids["spans"].append(span_id)

        if no_contrast_k is not None:
            nc_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "spans_no_contrast",
                    "signal": "preemphasised",
                    "k_db": no_contrast_k,
                    "reason": "seeded",
                },
            )
            store.was_generated_by(nc_id, activity)
            ids["no_contrast"] = nc_id

        if yamnet_windows is not None:
            path = tmp_path / "derivatives" / "yamnet_windows.json"
            path.write_text(json.dumps(yamnet_windows))
            yw_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "yamnet_windows",
                    "signal": "plain",
                    "path": "derivatives/yamnet_windows.json",
                    "n_windows": len(yamnet_windows),
                },
            )
            store.was_generated_by(yw_id, activity)
            ids["yamnet_windows"] = yw_id

        if silence_windows is not None:
            s_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
            )
            store.was_generated_by(s_id, activity)
            ids["silence"] = s_id

        if asr_available:
            asr_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={
                    "name": "asr_crisperwhisper",
                    "signal": "plain",
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "transcript": " ".join(str(w["text"]) for w in words),
                    "word_ids": [],
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(asr_id, activity)
            ids["asr"] = asr_id

        for word in words:
            word_id = store.entity(
                prov_type="word",
                extent=(float(word["start"]), float(word["end"])),
                attributes={
                    "text": str(word["text"]),
                    "score": 0.9,
                    "recognizer": "nyralabs/CrisperWhisper2.0_turbo",
                    "timestamp_source": "native",
                },
            )
            store.was_generated_by(word_id, activity)
            ids["words"].append(word_id)
        return ids

    return _seed


@pytest.fixture
def seed_voice_store(tmp_path: Path) -> Callable[..., dict]:
    """A seeder writing the store surface VOICE reads, constructed directly.

    A ``plain`` stream WAV carrying a 220 Hz tone inside each ``loud`` interval (defaulting to the
    ``energetic`` ones), the ``energy_envelope`` npz sidecar with both tracks and the envelope
    raised over its floor inside each ``energetic`` interval, PREPROCESS ``span`` entities, AIRWAY
    ``label`` assertions over the ``airway_labelled`` spans, SPEECH ``span`` entities and,
    optionally, the ``silence`` measurement.
    """

    def _seed(
        store: ProvStore,
        *,
        energetic: tuple = (),
        airway_labelled: tuple = (),
        speech_spans: tuple = (),
        unlabelled_spans: tuple = (),
        loud: tuple | None = None,
        silence_windows: list | None = None,
        duration_s: float = 7.0,
    ) -> dict:
        """Seed one store; returns the ids of what it wrote."""
        envelope_rate = 1000
        sampling_rate = 16000
        (tmp_path / "streams").mkdir(exist_ok=True)
        (tmp_path / "derivatives").mkdir(exist_ok=True)
        rng = np.random.default_rng(0)
        x = (rng.standard_normal(int(duration_s * sampling_rate)) * 1e-4).astype(np.float32)
        for start, end in energetic if loud is None else loud:
            i0, i1 = int(start * sampling_rate), int(end * sampling_rate)
            t = np.arange(i1 - i0) / sampling_rate
            x[i0:i1] += (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        wav_name = f"plain-{store.run_id}.wav"
        sf.write(str(tmp_path / "streams" / wav_name), x, sampling_rate)

        n = int(duration_s * envelope_rate)
        floor = np.full(n, -60.0)
        envelope = np.full(n, -70.0)
        for start, end in energetic:
            envelope[int(start * envelope_rate) : int(end * envelope_rate)] = -30.0
        np.savez(tmp_path / "derivatives" / "energy_envelope.npz", envelope_dbfs=envelope, floor_dbfs=floor)

        preprocess = store.activity(node="PREPROCESS", step="seed-voice", parameters={})
        airway_act = store.activity(node="AIRWAY", step="seed-voice", parameters={})
        speech_act = store.activity(node="SPEECH", step="seed-voice", parameters={})
        ids: dict = {"labelled_spans": [], "labels": [], "unlabelled_spans": [], "speech_spans": []}
        stream_id = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": "plain", "path": f"streams/{wav_name}", "sampling_rate": sampling_rate},
        )
        store.was_generated_by(stream_id, preprocess)
        ids["stream"] = stream_id
        envelope_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "energy_envelope",
                "path": "derivatives/energy_envelope.npz",
                "sampling_rate": envelope_rate,
            },
        )
        store.was_generated_by(envelope_id, preprocess)
        ids["envelope"] = envelope_id
        for start, end in airway_labelled:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": 30.0, "k_db": 18.0, "signal": "preemphasised"},
            )
            store.was_generated_by(span_id, preprocess)
            label_id = store.entity(
                prov_type="assertion", extent=(start, end), attributes={"verb": "label", "label": "Cough"}
            )
            store.was_generated_by(label_id, airway_act)
            store.was_derived_from(label_id, span_id)
            ids["labelled_spans"].append(span_id)
            ids["labels"].append(label_id)
        for start, end in unlabelled_spans:
            span_id = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": 25.0, "k_db": 18.0, "signal": "preemphasised"},
            )
            store.was_generated_by(span_id, preprocess)
            ids["unlabelled_spans"].append(span_id)
        for start, end in speech_spans:
            span_id = store.entity(prov_type="span", extent=(start, end), attributes={"source": "words"})
            store.was_generated_by(span_id, speech_act)
            ids["speech_spans"].append(span_id)
        if silence_windows is not None:
            silence_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
            )
            store.was_generated_by(silence_id, preprocess)
            ids["silence"] = silence_id
        return ids

    return _seed


@pytest.fixture
def seed_speech_store(tmp_path: Path) -> Callable[..., tuple[ProvStore, TriageConfig, Path]]:
    """Build ``(store, config, run_dir)`` as ADMIT/PREPROCESS/TAXONOMY/AIRWAY leave them for SPEECH.

    Mirrors the sibling plan's store-schema contract. ``words_cw``/``words_qw`` are
    ``(text, start, end)`` triples per recognizer; ``airway_label_extent`` also writes a PREPROCESS
    span over that extent carrying an AIRWAY ``label`` assertion. ``config_yaml`` is the production
    override mechanism, defaulting to the one unmeasured key every SPEECH test needs.
    """

    def _make(
        words_cw: list[tuple[str, float, float]],
        words_qw: list[tuple[str, float, float]],
        *,
        airway_label_extent: tuple[float, float] | None = None,
        duration_s: float = 6.0,
        config_yaml: str = "speech:\n  word_gap_ms: 300\n",
    ) -> tuple[ProvStore, TriageConfig, Path]:
        from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID

        sr = 16000
        store = ProvStore(run_id="t")
        rng = np.random.default_rng(0)
        n = int(duration_s * sr)
        wave = np.zeros(n, dtype=np.float32)
        for _, start, end in [*words_cw, *words_qw]:
            wave[int(start * sr) : int(end * sr)] = 0.1 * rng.standard_normal(int(end * sr) - int(start * sr))
        if airway_label_extent:
            start, end = airway_label_extent
            wave[int(start * sr) : int(end * sr)] = 0.2 * rng.standard_normal(int(end * sr) - int(start * sr))
        run_dir = tmp_path / "run"
        (run_dir / "streams").mkdir(parents=True, exist_ok=True)
        (run_dir / "derivatives").mkdir(exist_ok=True)
        sf.write(str(run_dir / "streams" / "plain.wav"), wave, sr)

        pre = store.activity(node="PREPROCESS", step=None, parameters={})
        recording = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "recording",
                "path": str(run_dir / "streams" / "plain.wav"),
                "sampling_rate": sr,
                "channels": 1,
            },
        )
        store.was_generated_by(recording, pre)
        plain = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={
                "name": "plain",
                "path": "streams/plain.wav",
                "sampling_rate": sr,
                "channels": 1,
                "peak_scale": 1.0,
            },
        )
        store.was_generated_by(plain, pre)

        # One sidecar (the Task 2 schema): envelope above the floor exactly where the wave is non-zero.
        env = np.full(n, -80.0)
        env[np.abs(wave) > 0] = -30.0
        floor = np.full(n, -60.0)
        np.savez(run_dir / "derivatives" / "energy_envelope.npz", envelope_dbfs=env, floor_dbfs=floor)
        mid = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "energy_envelope",
                "signal": "preemphasised",
                "path": "derivatives/energy_envelope.npz",
                "sampling_rate": sr,
            },
        )
        store.was_generated_by(mid, pre)

        # YAMNet native windows, Speech-positive throughout -- SPEECH reads these from the store.
        windows, t = [], 0.0
        while t < duration_s:
            windows.append(
                {"start": t, "end": t + 0.96, "label_scores": [{"Speech": 0.9}], "win_length": 0.96, "hop_length": 0.48}
            )
            t += 0.48
        (run_dir / "derivatives" / "yamnet_windows.json").write_text(json.dumps(windows))
        yw = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "yamnet_windows",
                "signal": "plain",
                "path": "derivatives/yamnet_windows.json",
                "n_windows": len(windows),
            },
        )
        store.was_generated_by(yw, pre)

        for model_id, triples in ((CRISPERWHISPER_ID, words_cw), (QWEN_ID, words_qw)):
            for text, start, end in triples:
                wid = store.entity(
                    prov_type="word",
                    extent=(start, end),
                    attributes={"text": text, "score": 0.9, "recognizer": model_id, "timestamp_source": "native"},
                )
                store.was_generated_by(wid, pre)

        if airway_label_extent:
            start, end = airway_label_extent
            span = store.entity(
                prov_type="span",
                extent=(start, end),
                attributes={"peak_over_floor_db": 30.0, "k_db": 18.0, "signal": "preemphasised"},
            )
            store.was_generated_by(span, pre)
            air = store.activity(node="AIRWAY", step="classify", parameters={})
            lab = store.entity(
                prov_type="assertion",
                extent=(start, end),
                attributes={
                    "verb": "label",
                    "label": "Cough",
                    "score": 0.97,
                    "scores": {"Cough": 0.97, "Breathe": 0.1},
                    "input": "buffered",
                    "in_certified_silence": None,
                },
            )
            store.was_generated_by(lab, air)
            store.was_derived_from(lab, span)

        override = tmp_path / "override.yaml"
        override.write_text(config_yaml)
        return store, load_triage_config(override), run_dir

    return _make
