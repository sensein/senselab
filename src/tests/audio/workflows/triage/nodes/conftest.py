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
def seed_airway_store(tmp_path: Path, config: TriageConfig) -> Callable[..., dict]:
    """A seeder writing the store surface PREPROCESS leaves behind, constructed directly.

    Spans at the airway ``K``, a ``plain`` stream sidecar, the ``yamnet_windows`` json sidecar,
    CrisperWhisper ``word`` entities and, optionally, a ``spans_no_contrast`` finding at one ``K``
    and a ``silence`` measurement carrying YAMNet's graded windows.
    """

    def _seed(
        store: ProvStore,
        *,
        spans: tuple[tuple[float, float, float], ...],
        yamnet_windows: list[dict[str, Any]],
        words: tuple[dict[str, Any], ...] = (),
        no_contrast_k: float | None = None,
        silence_windows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Seed one store; returns the ids of what it wrote, keyed ``stream``/``spans``/``yamnet``/``words``."""
        from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID

        k_db = float(config.require("spans.k_db.airway"))
        ends = [end for _, end, _ in spans] + [float(w["end"]) for w in words]
        duration = max([3.0, *(end + 0.5 for end in ends)])
        wav = tmp_path / f"plain-{store.run_id}.wav"
        sf.write(str(wav), burst_samples(duration_s=duration), 16000)
        stream_id = store.entity(prov_type="stream", extent=None, attributes={"name": "plain", "path": wav.name})
        span_ids = [
            store.entity(prov_type="span", extent=(start, end), attributes={"k_db": k_db, "peak_over_floor_db": peak})
            for start, end, peak in spans
        ]
        sidecar = tmp_path / f"yamnet-{store.run_id}.json"
        sidecar.write_text(json.dumps(yamnet_windows))
        yamnet_id = store.entity(
            prov_type="measurement", extent=None, attributes={"name": "yamnet_windows", "path": sidecar.name}
        )
        word_ids = [
            store.entity(
                prov_type="word",
                extent=(float(w["start"]), float(w["end"])),
                attributes={"text": w["text"], "recognizer": CRISPERWHISPER_ID},
            )
            for w in words
        ]
        if no_contrast_k is not None:
            store.entity(
                prov_type="measurement", extent=None, attributes={"name": "spans_no_contrast", "k_db": no_contrast_k}
            )
        if silence_windows is not None:
            store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
            )
        return {"stream": stream_id, "spans": span_ids, "yamnet": yamnet_id, "words": word_ids}

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


@pytest.fixture
def seed_redact_store(tmp_path: Path) -> Callable[..., tuple[ProvStore, TriageConfig, Path]]:
    """Build ``(store, config, run_dir)`` as PREPROCESS and SPEECH leave them for REDACT.

    ``findings`` are ``((start, end), category)`` or ``((start, end), category, speaker)`` tuples;
    each writes a ``pii`` entity, a SPEECH word covering it carrying the ``pii`` label assertion
    and, once per store (unless ``scanned`` is False), the ``pii_scan`` measurement — the shapes
    Task 5's node writes. ``scanned_by``/``scan_failed`` are that measurement's own evidence fields,
    so a caller can seed a store scan that did not complete. ``target``, when given, writes SPEECH's
    verdict entity carrying ``target_speaker`` and flagging only that speaker's findings, so a
    speaker-scoped reader has something to scope by. Both recognizers' PREPROCESS word entities and
    model agents are always written so verification can name them; ``commitless`` names recognizers
    whose model agent records no commit and ``wordless`` names recognizers whose ASR died inside
    PREPROCESS, leaving the activity and the agent but no word and a PREPROCESS verdict listing the
    step absent. ``declared`` False writes those activities without the ``model`` parameter, as a
    store whose declaration cannot be read. ``config_yaml`` is the production override mechanism,
    defaulting to the one unmeasured key every REDACT test needs.
    """

    def _make(
        _tmp_path: Path | None = None,
        *,
        findings: tuple = (),
        words: tuple = (("hello", 0.2, 0.5, "SPEAKER_00"), ("world", 2.0, 2.3, "SPEAKER_00")),
        scanned: bool = True,
        scanned_by: tuple[str, ...] = ("gliner", "presidio", "rules"),
        scan_failed: tuple[str, ...] = (),
        target: str | None = None,
        commitless: tuple[str, ...] = (),
        wordless: tuple[str, ...] = (),
        declared: bool = True,
        config_yaml: str = "redaction:\n  padding_ms: 50\n",
    ) -> tuple[ProvStore, TriageConfig, Path]:
        from senselab.audio.workflows.triage.nodes.preprocess import CRISPERWHISPER_ID, QWEN_ID

        sr = 16000
        duration_s = max([5.0, *(float(extent[1]) + 1.0 for extent, *_ in findings)])
        rng = np.random.default_rng(0)
        wave = (0.05 * rng.standard_normal(int(duration_s * sr))).astype(np.float32)
        run_dir = tmp_path / "run"
        (run_dir / "streams").mkdir(parents=True, exist_ok=True)
        sf.write(str(run_dir / "streams" / "plain.wav"), wave, sr)

        store = ProvStore(run_id="t")
        pre = store.activity(node="PREPROCESS", step=None, parameters={})
        recording = store.entity(
            prov_type="stream",
            extent=(0.0, duration_s),
            attributes={"name": "recording", "path": "streams/plain.wav", "sampling_rate": sr, "channels": 1},
        )
        store.was_generated_by(recording, pre)

        for model_id, sha in ((CRISPERWHISPER_ID, "c" * 40), (QWEN_ID, "d" * 40)):
            if model_id in commitless:
                agent = store.agent(agent_type="model", model_id=model_id, unresolved_reason="offline load")
            else:
                agent = store.agent(agent_type="model", model_id=model_id, commit_sha=sha)
            asr_act = store.activity(
                node="PREPROCESS", step=f"asr:{model_id}", parameters={"model": model_id} if declared else {}
            )
            store.was_associated_with(asr_act, agent)
            if model_id in wordless:
                continue
            raw_id = store.entity(
                prov_type="word", extent=(0.2, 0.5), attributes={"text": "hello", "recognizer": model_id}
            )
            store.was_generated_by(raw_id, asr_act)

        if wordless:
            absent_verdict = store.entity(
                prov_type="verdict",
                extent=None,
                attributes={
                    "node": "PREPROCESS",
                    "outcome": "pass",
                    "kind": None,
                    "why": "conditioning complete; absent derivatives are listed",
                    "absent": {f"asr:{model_id}": "RuntimeError" for model_id in wordless},
                    "derivatives": {},
                },
            )
            store.was_generated_by(absent_verdict, pre)

        speech_act = store.activity(node="SPEECH", step="identify", parameters={})
        pii_act = store.activity(node="SPEECH", step="pii", parameters={})
        word_rows = sorted(
            list(words)
            + [
                (f"secret{i}", float(extent[0]), float(extent[1]), (rest[0] if rest else "SPEAKER_00"))
                for i, (extent, _category, *rest) in enumerate(findings)
            ],
            key=lambda row: float(row[1]),
        )
        word_ids: list[str] = []
        for text, start, end, speaker in word_rows:
            word_id = store.entity(
                prov_type="word",
                extent=(float(start), float(end)),
                attributes={"text": str(text), "speaker": speaker, "stream": recording},
            )
            store.was_generated_by(word_id, speech_act)
            word_ids.append(word_id)

        for extent, category, *_rest in findings:
            pii_id = store.entity(
                prov_type="pii",
                extent=(float(extent[0]), float(extent[1])),
                attributes={
                    "category": category,
                    "source": "presidio",
                    "asr_model": CRISPERWHISPER_ID,
                    "detectors_used": ["gliner", "presidio", "rules"],
                    "detectors_failed": [],
                },
            )
            store.was_generated_by(pii_id, pii_act)
            mark_id = store.entity(
                prov_type="assertion",
                extent=(float(extent[0]), float(extent[1])),
                attributes={"verb": "label", "label": "pii", "category": category},
            )
            store.was_generated_by(mark_id, pii_act)
            covering = next(
                (
                    word_id
                    for word_id, row in zip(word_ids, word_rows)
                    if float(row[1]) < float(extent[1]) and float(row[2]) > float(extent[0])
                ),
                None,
            )
            if covering is not None:
                store.was_derived_from(mark_id, covering)

        if scanned:
            scan_id = store.entity(
                prov_type="measurement",
                extent=None,
                attributes={"name": "pii_scan", "scanned_by": list(scanned_by), "failed": list(scan_failed)},
            )
            store.was_generated_by(scan_id, pii_act)

        if target is not None:
            flagged = [
                f"pii ({category}) in the target speaker's speech"
                for _extent, category, *rest in findings
                if (rest[0] if rest else "SPEAKER_00") == target
            ]
            verdict_id = store.entity(
                prov_type="verdict",
                extent=None,
                attributes={
                    "node": "SPEECH",
                    "outcome": "flag" if flagged else "pass",
                    "kind": "speech",
                    "why": "; ".join(flagged) or "words, spans, speakers and quality are in the store",
                    "target_speaker": target,
                    "flags": flagged,
                },
            )
            store.was_generated_by(verdict_id, speech_act)

        override = tmp_path / "redact-override.yaml"
        override.write_text(config_yaml)
        return store, load_triage_config(override), run_dir

    return _make
