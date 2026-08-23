"""PREPROCESS writes every derivative to the store with provenance; an uncomputable one is absent.

Every model call is monkeypatched on the node module (the pii_adapter_test pattern); the DSP —
resample, envelope, spans, spectrograms, gammatone, fuse_word_streams — runs real. No test here
loads weights or touches the network.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as node
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.common import clamp_extent
from senselab.audio.workflows.triage.nodes.preprocess import PreprocessResult, preprocess
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.data_structures import ScriptLine
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import burst_samples


class _FakeModel:
    """A model spec stub carrying exactly what the node reads: path_or_uri and commit_sha."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "a" * 40


@pytest.fixture
def calls() -> dict[str, list]:
    """Captured model-call arguments, per mocked function."""
    return {"classify": [], "transcribe": [], "align": [], "squim": []}


@pytest.fixture
def mock_models(monkeypatch: pytest.MonkeyPatch, calls: dict[str, list]) -> None:
    """Replace every model call PREPROCESS makes; payload shapes mirror the real returns.

    CrisperWhisper/Qwen: a ScriptLine whose ``chunks`` are word ScriptLines with text/start/end/score
    (crisperwhisper.py builds exactly that). YAMNet: windowed dicts with start/end/label_scores/
    win_length/hop_length (classification/api.py). SQUIM: one dict with stoi/pesq/si_sdr
    (torchaudio_squim.py). Alignment: a list per input of ScriptLine | None (forced_alignment.py).
    """
    monkeypatch.setattr(node, "_crisperwhisper_model", lambda: _FakeModel(node.CRISPERWHISPER_ID))
    monkeypatch.setattr(node, "_qwen_model", lambda: _FakeModel(node.QWEN_ID))

    def fake_classify(audios: list, model: object, top_k: int | None = None, **kwargs: object) -> list:
        """YAMNet-shaped windows over the input's real duration."""
        calls["classify"].append({"model": model, "top_k": top_k})
        duration = audios[0].waveform.shape[-1] / audios[0].sampling_rate
        windows, start = [], 0.0
        while start + 0.96 <= duration:
            windows.append(
                {
                    "start": round(start, 2),
                    "end": round(start + 0.96, 2),
                    "label_scores": [{"Silence": 0.7}, {"Speech": 0.1}],
                    "win_length": 0.96,
                    "hop_length": 0.48,
                }
            )
            start += 0.48
        return [windows]

    def fake_transcribe(audios: list, model: _FakeModel, **kwargs: object) -> list:
        """Two words inside the burst, for either recognizer."""
        calls["transcribe"].append({"model": model.path_or_uri, "audio": audios[0], "kwargs": kwargs})
        chunks = [
            ScriptLine(text="hello", start=1.50, end=1.58, score=0.9),
            ScriptLine(text="doctor", start=1.60, end=1.66, score=0.9),
        ]
        return [ScriptLine(text="hello doctor", start=1.50, end=1.66, chunks=chunks, score=0.9)]

    def fake_align(items: list, levels_to_keep: dict | None = None, aligner_model: str | None = None) -> list:
        """One aligned line per input tuple."""
        calls["align"].append({"n": len(items)})
        return [[ScriptLine(text="hello doctor", start=1.50, end=1.66)] for _ in items]

    def fake_squim(audios: list, device: object = None) -> list:
        """One objective-head dict per input."""
        calls["squim"].append({"n_samples": int(audios[0].waveform.shape[-1])})
        return [{"stoi": 0.91, "pesq": 1.8, "si_sdr": 7.5} for _ in audios]

    monkeypatch.setattr(node, "classify_audios", fake_classify)
    monkeypatch.setattr(node, "transcribe_audios", fake_transcribe)
    monkeypatch.setattr(node, "align_transcriptions", fake_align)
    monkeypatch.setattr(node, "extract_objective_quality_features_from_audios", fake_squim)


def _run(
    store: ProvStore,
    config: TriageConfig,
    tmp_path: Path,
    samples: np.ndarray | None = None,
    sampling_rate: int = 16000,
) -> PreprocessResult:
    """Admit a fixture recording, then preprocess it."""
    path = tmp_path / "input.wav"
    sf.write(str(path), (burst_samples() if samples is None else samples).astype(np.float32), sampling_rate)
    admitted = admit(store, path, config, run_dir=tmp_path)
    assert admitted.audio is not None
    return preprocess(store, admitted.audio, config, run_dir=tmp_path)


def _measurement(store: ProvStore, name: str) -> Any:  # noqa: ANN401 — the store's Entity, untyped here
    """The one measurement entity with this name."""
    [entity] = [e for e in store.entities("measurement") if e.attributes.get("name") == name]
    return entity


class TestConditioning:
    """The two retained signals and the overshoot guard."""

    def test_plain_stream_is_mono_16k_with_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A 48 kHz stereo input becomes one mono 16 kHz plain stream derived from the recording."""
        stereo = np.stack([burst_samples(sampling_rate=48000)] * 2, axis=1)
        _run(store, config, tmp_path, samples=stereo, sampling_rate=48000)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        assert plain.attributes["sampling_rate"] == 16000
        assert plain.attributes["channels"] == 1
        assert plain.attributes["peak_scale"] == 1.0
        data, rate = sf.read(str(tmp_path / plain.attributes["path"]))
        assert rate == 16000
        [recording] = [e for e in store.entities("stream") if e.attributes.get("name") == "recording"]
        assert recording.id in store.derived_from(plain.id)

    def test_an_invalidated_recording_is_not_a_source_of_the_plain_stream(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A withdrawn recording must not be recorded as what plain derives from."""
        path = tmp_path / "input.wav"
        sf.write(str(path), burst_samples().astype(np.float32), 16000)
        admitted = admit(store, path, config, run_dir=tmp_path)
        assert admitted.audio is not None
        [withdrawn] = [e for e in store.entities("stream") if e.attributes.get("name") == "recording"]
        store.was_invalidated_by(withdrawn.id, store.activity(node="ADMIT", step="withdraw", parameters={}))
        preprocess(store, admitted.audio, config, run_dir=tmp_path)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        assert withdrawn.id not in store.derived_from(plain.id)

    def test_preemphasised_stream_is_the_first_difference(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """y[n] = x[n] - c * x[n-1], with the coefficient from the config, derived from plain."""
        _run(store, config, tmp_path)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        [sharp] = [e for e in store.entities("stream") if e.attributes.get("name") == "preemphasised"]
        c = float(config.require("preemphasis.coefficient"))
        assert sharp.attributes["coefficient"] == c
        x, _ = sf.read(str(tmp_path / plain.attributes["path"]))
        y, _ = sf.read(str(tmp_path / sharp.attributes["path"]))
        assert np.allclose(y[1:], x[1:] - c * x[:-1], atol=1e-6)
        assert plain.id in store.derived_from(sharp.id)

    def test_disabled_preemphasis_routes_envelope_to_plain(
        self, store: ProvStore, tmp_path: Path, mock_models: None
    ) -> None:
        """With preemphasis.enabled false there is no second stream and derivatives read plain."""
        override = tmp_path / "override.yaml"
        override.write_text("preemphasis:\n  enabled: false\n")
        config = load_triage_config(override)
        _run(store, config, tmp_path)
        assert [e for e in store.entities("stream") if e.attributes.get("name") == "preemphasised"] == []
        assert _measurement(store, "energy_envelope").attributes["signal"] == "plain"


class TestEnvelopeAndSpans:
    """The pre-emphasised derivatives and the span proposals."""

    def test_envelope_reads_the_preemphasised_signal(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """The envelope names its signal and its sidecar holds envelope and floor tracks."""
        _run(store, config, tmp_path)
        envelope = _measurement(store, "energy_envelope")
        assert envelope.attributes["signal"] == "preemphasised"
        sidecar = np.load(tmp_path / envelope.attributes["path"])
        assert sidecar["envelope_dbfs"].shape == sidecar["floor_dbfs"].shape

    def test_spans_carry_the_airway_k_and_no_label(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """The burst yields at least one span at K = spans.k_db.airway, unlabelled."""
        _run(store, config, tmp_path)
        spans = store.entities("span")
        assert spans, "the burst fixture must propose at least one span"
        for span in spans:
            assert span.attributes["k_db"] == float(config.require("spans.k_db.airway"))
            assert "label" not in span.attributes
            assert span.attributes["peak_over_floor_db"] >= float(config.require("spans.k_db.airway"))

    def test_no_contrast_is_recorded_with_its_k(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A burst-free recording writes spans_no_contrast, not an empty span list."""
        rng = np.random.default_rng(1)
        _run(store, config, tmp_path, samples=(rng.standard_normal(48000) * 1e-4))
        assert store.entities("span") == []
        no_contrast = _measurement(store, "spans_no_contrast")
        assert no_contrast.attributes["k_db"] == float(config.require("spans.k_db.airway"))
        assert "reason" in no_contrast.attributes


class TestModelDerivatives:
    """The plain-signal derivatives: YAMNet, level, SQUIM, the recognizers, agreement, alignment."""

    def test_yamnet_is_read_with_the_full_label_space(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """top_k comes from the config, never the function's windowed default of 5."""
        _run(store, config, tmp_path)
        [call] = calls["classify"]
        assert call["model"] == "yamnet"
        assert call["top_k"] == int(config.require("yamnet.top_k"))
        windows_entity = _measurement(store, "yamnet_windows")
        windows = json.loads((tmp_path / windows_entity.attributes["path"]).read_text())
        assert windows_entity.attributes["n_windows"] == len(windows) > 0
        silence = _measurement(store, "silence")
        assert all(row["is_silence"] == (row["score"] >= 0.5) for row in silence.attributes["windows"])

    def test_asr_runs_on_the_plain_signal_not_the_preemphasised_one(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """Both recognizers receive the plain waveform (pre-emphasis changes the peak measurably)."""
        _run(store, config, tmp_path)
        [plain] = [e for e in store.entities("stream") if e.attributes.get("name") == "plain"]
        reference, _ = sf.read(str(tmp_path / plain.attributes["path"]), dtype="float32")
        assert len(calls["transcribe"]) == 2
        for call in calls["transcribe"]:
            received = call["audio"].waveform.squeeze(0).numpy()
            assert np.allclose(received, reference, atol=1e-4)

    def test_words_become_word_entities_with_recognizer_provenance(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """One word entity per recognizer word, stamped with who timed it, attributed to the model."""
        _run(store, config, tmp_path)
        words = store.entities("word")
        assert len(words) == 4  # two words from each recognizer
        recognizers = {w.attributes["recognizer"] for w in words}
        assert recognizers == {node.CRISPERWHISPER_ID, node.QWEN_ID}
        sources = {w.attributes["recognizer"]: w.attributes["timestamp_source"] for w in words}
        assert sources[node.CRISPERWHISPER_ID] == "native"
        assert sources[node.QWEN_ID] == "bundled_aligner"
        for word in words:
            [agent_id] = [a for a in store.associated_with(store.generated_by(word.id) or "")]
            agent = store.get_agent(agent_id)
            assert agent.agent_type == "model"
            assert agent.commit_sha == "a" * 40

    def test_agreement_and_alignment_follow_the_recognizers(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """asr_agreement fuses both streams; alignment aligns the fused transcript on plain."""
        _run(store, config, tmp_path)
        agreement = _measurement(store, "asr_agreement")
        assert agreement.attributes["systems"] == [node.CRISPERWHISPER_ID, node.QWEN_ID]
        assert {w["text"] for w in agreement.attributes["words"]} == {"hello", "doctor"}
        alignment = _measurement(store, "alignment")
        assert alignment.attributes["transcript_source"] == "asr_agreement"
        payload = json.loads((tmp_path / alignment.attributes["path"]).read_text())
        assert payload, "the aligned transcript is serialised to the sidecar"
        assert calls["align"] == [{"n": 1}]

    def test_an_untimed_chunk_is_counted_not_placed_at_zero(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A chunk with no timestamp has no extent; coercing None to 0.0 invents one at the file start.

        The word entity is not written, the count is in the measurement, and the text survives in
        the transcript — the recognizer said it, it just did not say where.
        """

        def untimed_transcribe(audios: list, model: _FakeModel, **kwargs: object) -> list:
            chunks = [
                ScriptLine(text="hello", start=1.50, end=1.58, score=0.9),
                ScriptLine(text="doctor", start=None, end=None, score=0.9),
            ]
            return [ScriptLine(text="hello doctor", start=1.50, end=1.66, chunks=chunks, score=0.9)]

        monkeypatch.setattr(node, "transcribe_audios", untimed_transcribe)
        _run(store, config, tmp_path)
        words = store.entities("word")
        assert [w.extent for w in words] == [(1.50, 1.58), (1.50, 1.58)], "one timed word per recognizer, no more"
        assert all(w.extent != (0.0, 0.0) for w in words)
        measurement = _measurement(store, "asr_crisperwhisper")
        assert measurement.attributes["untimed_chunks_n"] == 1
        assert "doctor" in measurement.attributes["transcript"], "the transcript keeps what the recognizer said"

    def test_the_fused_words_are_bounded_so_alignment_survives_a_hallucination(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """asr_agreement fuses from the recognizer output, which no word-entity bound reaches.

        The production trigger: a hallucinated word at 11.32 s among legitimate ones. The word
        entities are bounded, but `_agreement` re-derives its list from the raw ScriptLine, so the
        fused list carried 11.32 s, `_alignment` set the transcript end from it, and
        `align_transcriptions` refused a slice past the decode. The block loop caught that, so the
        `alignment` derivative vanished from the run instead of failing it.
        """
        received: list[tuple[float, float]] = []

        def hallucinating_transcribe(audios: list, model: _FakeModel, **kwargs: object) -> list:
            chunks = [
                ScriptLine(text="hello", start=1.50, end=1.58, score=0.9),
                ScriptLine(text="doctor", start=1.60, end=1.66, score=0.9),
                ScriptLine(text="podcast", start=11.32, end=11.60, score=0.9),
            ]
            return [ScriptLine(text="hello doctor podcast", start=1.50, end=11.60, chunks=chunks, score=0.9)]

        def bound_checking_align(items: list, **kwargs: object) -> list:
            """Refuse a transcript past the decode, exactly as ``extract_segments`` does."""
            out = []
            for audio, transcript, _language in items:
                duration = audio.waveform.shape[-1] / audio.sampling_rate
                received.append((float(transcript.start or 0.0), float(transcript.end or 0.0)))
                if float(transcript.end or 0.0) > duration:
                    raise ValueError(f"End must be <= duration of the audio ({duration} sec).")
                out.append([ScriptLine(text=transcript.text, start=transcript.start, end=transcript.end)])
            return out

        monkeypatch.setattr(node, "transcribe_audios", hallucinating_transcribe)
        monkeypatch.setattr(node, "align_transcriptions", bound_checking_align)
        result = _run(store, config, tmp_path, samples=burst_samples(duration_s=5.76))

        agreement = _measurement(store, "asr_agreement")
        assert all(float(w["end"]) <= 5.76 for w in agreement.attributes["words"]), "no fused word outruns the decode"
        assert {w["text"] for w in agreement.attributes["words"]} == {"hello", "doctor"}
        assert agreement.attributes["out_of_bounds_words_n"] == 1

        assert "alignment" not in result.absent, "a hallucinated word must not delete the alignment derivative"
        assert received == [(1.50, 1.66)], "the transcript the aligner saw is inside the recording"

    def test_a_chunk_starting_past_the_decode_is_dropped_and_counted(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hallucinated word timed after the file ends is not a word; it is a count.

        Observed in production: on a 5.76 s recording Qwen returned words at 11.32 s and 29.08 s.
        A start at-or-past the decode duration names no sample of this recording, so no word entity
        is written; the text stays in the transcript, as an untimed chunk's does.
        """

        def hallucinating_transcribe(audios: list, model: _FakeModel, **kwargs: object) -> list:
            chunks = [
                ScriptLine(text="hello", start=1.50, end=1.58, score=0.9),
                ScriptLine(text="podcast", start=11.32, end=11.60, score=0.9),
            ]
            return [ScriptLine(text="hello podcast", start=1.50, end=11.60, chunks=chunks, score=0.9)]

        monkeypatch.setattr(node, "transcribe_audios", hallucinating_transcribe)
        _run(store, config, tmp_path, samples=burst_samples(duration_s=5.76))
        words = store.entities("word")
        assert [w.extent for w in words] == [(1.50, 1.58), (1.50, 1.58)], "one in-bounds word per recognizer"
        measurement = _measurement(store, "asr_crisperwhisper")
        assert measurement.attributes["out_of_bounds_chunks_n"] == 1
        assert measurement.attributes["untimed_chunks_n"] == 0
        assert "podcast" in measurement.attributes["transcript"], "the transcript keeps what was said"

    def test_a_chunk_overshooting_the_decode_end_is_clamped_not_dropped(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A word that starts inside the recording and ends past it is plausibly real with a bad end.

        The clamp here is semantic and unbounded in size, which is what distinguishes it from
        ``nodes.common.clamp_extent``: that one bounds float noise within a single sample period and
        raises on anything larger. The raw extent would raise there; the stored one does not, which
        is what stops SPEECH's slice of this word from crashing.
        """

        def overshooting_transcribe(audios: list, model: _FakeModel, **kwargs: object) -> list:
            chunks = [ScriptLine(text="hello", start=0.30, end=6.10, score=0.9)]
            return [ScriptLine(text="hello", start=0.30, end=6.10, chunks=chunks, score=0.9)]

        monkeypatch.setattr(node, "transcribe_audios", overshooting_transcribe)
        _run(store, config, tmp_path, samples=burst_samples(duration_s=5.76))
        plain = Audio(filepath=str(tmp_path / "streams" / "plain.wav"))
        words = store.entities("word")
        assert [w.extent for w in words] == [(0.30, 5.76), (0.30, 5.76)], "the end is bound by the decode"
        measurement = _measurement(store, "asr_crisperwhisper")
        assert measurement.attributes["out_of_bounds_chunks_n"] == 0, "a clamped word was not dropped"
        with pytest.raises(ValueError, match="more than one sample period"):
            clamp_extent((0.30, 6.10), plain)
        assert clamp_extent(words[0].extent or (0.0, 0.0), plain) == (0.30, 5.76)

    def test_an_in_bounds_chunk_keeps_the_timings_the_recognizer_gave(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """Bounding touches nothing that was already inside the recording."""
        _run(store, config, tmp_path, samples=burst_samples(duration_s=5.76))
        words = store.entities("word")
        assert {w.extent for w in words} == {(1.50, 1.58), (1.60, 1.66)}
        for name in ("asr_crisperwhisper", "asr_qwen"):
            assert _measurement(store, name).attributes["out_of_bounds_chunks_n"] == 0

    def test_the_aligner_agent_names_the_model_not_its_whole_spec(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None
    ) -> None:
        """A model_id must be a model id: the language table's value is a mapping, not a name."""
        _run(store, config, tmp_path)
        alignment = _measurement(store, "alignment")
        [agent_id] = store.associated_with(store.generated_by(alignment.id) or "")
        agent = store.get_agent(agent_id)
        assert agent.model_id == "facebook/wav2vec2-base-960h"
        assert agent.commit_sha is None
        assert agent.unresolved_reason is not None, "align_transcriptions reports no commit; the store says so"

    def test_squim_is_measured_per_span_as_a_measure_assertion(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, mock_models: None, calls: dict
    ) -> None:
        """One measure assertion per span, derived from it, on the sliced plain signal."""
        _run(store, config, tmp_path)
        spans = store.entities("span")
        measures = [a for a in store.entities("assertion") if a.attributes.get("name") == "squim"]
        assert len(measures) == len(spans) > 0
        for measure in measures:
            assert measure.attributes["verb"] == "measure"
            assert measure.attributes["stoi"] == 0.91
            assert any(s.id in store.derived_from(measure.id) for s in spans)
        assert all(c["n_samples"] > 0 for c in calls["squim"])

    def test_a_span_squim_refuses_is_unmeasured_not_padded(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SQUIM re-raises on short input; the node records unmeasured and never pads."""

        def refusing_squim(audios: list, device: object = None) -> list:
            """The real function's failure mode for a too-short span."""
            raise RuntimeError("input too short")

        monkeypatch.setattr(node, "extract_objective_quality_features_from_audios", refusing_squim)
        result = _run(store, config, tmp_path)
        measures = [a for a in store.entities("assertion") if a.attributes.get("name") == "squim"]
        assert measures, "the refusal is recorded per span, not dropped"
        assert all(m.attributes["unmeasured"] == "RuntimeError" for m in measures)
        assert result.verdict.outcome is Outcome.PASS


class TestAbsenceIsNotAnError:
    """A derivative that cannot be computed is absent from the store; the node still passes."""

    def test_a_failing_model_leaves_its_derivatives_absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """YAMNet failing removes yamnet_windows and silence; nothing raises; outcome stays pass."""

        def broken_classify(*args: object, **kwargs: object) -> list:
            """A backend crash."""
            raise RuntimeError("subprocess venv failed")

        monkeypatch.setattr(node, "classify_audios", broken_classify)
        result = _run(store, config, tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert "yamnet_windows" in result.absent
        assert "silence" in result.absent
        names = {e.attributes.get("name") for e in store.entities("measurement")}
        assert "yamnet_windows" not in names and "silence" not in names
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["absent"]["yamnet_windows"] == "RuntimeError"

    def test_one_missing_recognizer_takes_agreement_and_alignment_with_it(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        mock_models: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Agreement needs both recognizers; its absence is recorded, not raised."""

        def qwen_only_fails(audios: list, model: _FakeModel, **kwargs: object) -> list:
            """Qwen's venv fails; CrisperWhisper still answers."""
            if model.path_or_uri == node.QWEN_ID:
                raise RuntimeError("qwen venv failed")
            chunks = [ScriptLine(text="hello", start=1.50, end=1.58, score=0.9)]
            return [ScriptLine(text="hello", start=1.50, end=1.58, chunks=chunks, score=0.9)]

        monkeypatch.setattr(node, "transcribe_audios", qwen_only_fails)
        result = _run(store, config, tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert {"asr_qwen", "asr_agreement", "alignment"} <= set(result.absent)
        assert _measurement(store, "asr_crisperwhisper") is not None
