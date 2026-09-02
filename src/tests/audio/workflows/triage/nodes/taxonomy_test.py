"""TAXONOMY v2: a fold over PREPROCESS's stored derivatives.

No models, no hints -- one localising exception: phonation-span detection, over PREPROCESS's own
F0/formant track measurement.
"""

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from scipy.signal import lfilter

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import preprocess as preprocess_module
from senselab.audio.workflows.triage.nodes import taxonomy as taxonomy_module
from senselab.audio.workflows.triage.nodes.common import find_measurements, live_entities
from senselab.audio.workflows.triage.nodes.preprocess import preprocess
from senselab.audio.workflows.triage.nodes.taxonomy import SCREENED_KINDS, taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.nodes.conftest import SR, _audio, _seed_admit, _stub_models


def _floors(tmp_path: Path, **extra: str) -> TriageConfig:
    """The packaged config with every TAXONOMY floor supplied and the speech family named.

    Args:
        tmp_path: Where the override YAML is written.
        **extra: Further YAML fragments appended to the override, for a test that needs one.

    Returns:
        The resolved configuration.
    """
    body = (
        "taxonomy:\n"
        "  presence_floor:\n"
        "    speech: {acoustic: 1, lexical: 1}\n"
        "    airway: {health_acoustic: 1, acoustic: 1}\n"
        "  voice_min_duration_s: 1.0\n"
        "  voice_uncertain_duration_s: 0.3\n"
        "  speech_labels: [Speech, Narration, monologue, Conversation]\n"
    )
    path = tmp_path / "floors.yaml"
    path.write_text(body + "".join(extra.values()))
    return load_triage_config(path)


class TestItRunsNoModels:
    """Every classifier call belongs to PREPROCESS; this node folds what is already there."""

    def test_the_module_imports_no_classifier(self) -> None:
        """A model function reachable from this module is a boundary violation, not a convenience."""
        for name in ("classify_audios", "detect_health_acoustic_events", "transcribe_audios"):
            assert not hasattr(taxonomy_module, name)

    def test_it_writes_no_activity_that_names_a_model(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The only activity is the fold."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert [a.step for a in store.activities("TAXONOMY")] == ["fold"]
        assert not [
            agent
            for activity in store.activities("TAXONOMY")
            for agent in store.associated_with(activity.id)
            if store.get_agent(agent).agent_type == "model"
        ]


class TestTheThreeKinds:
    """speech, airway and voice, each with its own rule and its own evidence."""

    def test_speech_needs_both_lines(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Acoustic windows and lexical words both clearing their floors makes speech present."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two", "three"])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "present"

    def test_an_empty_authoritative_consensus_makes_speech_absent_despite_an_acoustic_label(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One false-positive acoustic window cannot overrule a completed empty consensus."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=[])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "absent"

    def test_consensus_words_make_speech_present_without_an_acoustic_label(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The consensus transcript is the authority for lexical speech, not an acoustic vote."""
        seed_preprocess_store(store, yamnet_labels=[["Music"]], words=["one"])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "present"

    def test_speech_with_neither_line_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both lines below their floors is absence."""
        seed_preprocess_store(store, yamnet_labels=[["Music"]], words=[])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "absent"

    def test_a_bracketed_event_carries_no_lexical_evidence(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """PREPROCESS wrote only events, so the authoritative consensus remains lexically empty."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=[], events=["[COUGH]", "[COUGH]", "ahem"])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "absent"

    def test_airway_is_present_from_hear_alone_without_yamnet_agreement(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """HeAR is the domain-specific detector and decides airway alone; YAMNet need not agree.

        Reads PREPROCESS's own per-span ``span_hear`` measurement, not a whole-file pooled window --
        the same shape AIRWAY's branch now reuses instead of re-deriving.
        """
        seed_preprocess_store(store, spans=[(0.0, 1.0, 20.0)], span_hear_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "present"

    def test_yamnet_alone_does_not_establish_airway_presence(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """YAMNet only corroborates; it cannot decide airway on its own without HeAR agreeing."""
        seed_preprocess_store(store, spans=[(0.0, 1.0, 20.0)], span_hear_labels=[[]], span_yamnet_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "absent"

    def test_a_transcribed_span_is_not_airway_evidence_even_with_a_hear_label(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """ASR outranks HeAR: a span a consensus word overlaps is lexical content, not airway."""
        seed_preprocess_store(
            store,
            spans=[(0.0, 1.0, 20.0)],
            span_hear_labels=[["Cough"]],
            words=[("cough", (0.2, 0.4))],
        )
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "absent"

    def test_voice_is_classified_from_phonation_span_duration_alone(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A 2 s sustain makes voice present, whatever else is in the recording."""
        seed_preprocess_store(store, phonation=[(0.0, 2.0, "voiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_an_unvoiced_sustain_makes_voice_present_too(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A disordered voice sustaining without periodicity is phonation."""
        seed_preprocess_store(store, phonation=[(0.0, 2.0, "unvoiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_a_short_span_is_uncertain_and_a_shorter_one_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Between the two floors is uncertain; below the shorter floor there is nothing to be sure of."""
        seed_preprocess_store(store, phonation=[(0.0, 0.5, "voiced")])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"
        other = ProvStore(run_id="short")
        seed_preprocess_store(other, phonation=[(0.0, 0.1, "voiced")])
        assert taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_span_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The pass ran and found nothing, which is absence."""
        seed_preprocess_store(store, phonation=[])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_pass_at_all_is_uncertain(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A pass that did not run leaves the line unavailable; that is not evidence of absence."""
        seed_preprocess_store(store, phonation=None)
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"


class TestAMissingDerivativeIsNotAbsence:
    """A line whose derivative never reached the store is unavailable, and unavailable is uncertain."""

    def test_a_null_threshold_leaves_every_kind_uncertain(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The packaged config folds no windows, so nothing can be absent."""
        seed_preprocess_store(store, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert set(result.kinds.values()) == {"uncertain"}
        assert result.verdict.outcome is Outcome.FLAG

    def test_the_unavailable_line_says_so_on_the_kind_element(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A reader must see why a kind is uncertain, not only that it is."""
        seed_preprocess_store(store, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        taxonomy(store, "plain", config, run_dir=tmp_path)
        speech = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "speech")
        assert speech.attributes["lines"]["acoustic"]["state"] == "unavailable"


class TestHintsAreNotAnInput:
    """A classification that reads the declaration cannot disagree with it."""

    def test_a_hint_declaring_speech_does_not_move_the_classification(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The same store classifies the same way with and without a hint."""
        seed_preprocess_store(store, yamnet_labels=[["Music"]], words=[])
        hinted = taxonomy(store, "plain", _floors(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        other = ProvStore(run_id="unhinted")
        seed_preprocess_store(other, yamnet_labels=[["Music"]], words=[])
        plain = taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert hinted.kinds == plain.kinds


class TestTheOutcome:
    """fail on all-absent, flag on any-uncertain, pass otherwise."""

    def test_all_absent_fails(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nothing is classified present."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Music"]],
            words=[],
            phonation=[],
            spans=[(0.0, 1.0, 10.0)],
            span_hear_labels=[[]],
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FAIL

    def test_any_uncertain_flags(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One genuinely indeterminate kind is enough."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Music"]],
            hear_labels=[[]],
            ast_labels=[[]],
            words=[],
            phonation=[(0.0, 0.5, "voiced")],
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FLAG

    def test_present_and_absent_together_pass(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Speech present, airway and voice absent, nothing uncertain."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"]],
            ast_labels=[["Speech"]],
            words=["one", "two", "three"],
            phonation=[],
            spans=[(0.0, 1.0, 10.0)],
            span_hear_labels=[[]],
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS

    def test_exactly_three_kind_elements_and_no_residual(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """voice_no_words and not_screened are gone; nothing is a kind by virtue of the others."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two", "three"])
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        kinds = {e.attributes["kind"] for e in live_entities(store, "kind")}
        assert kinds == set(SCREENED_KINDS) == {"airway", "speech", "voice"}
        assert not [e for e in live_entities(store, "kind") if e.attributes["state"] == "not_screened"]

    def test_it_localises_nothing_beyond_phonation(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No new span, no interval, no other extent-bearing element -- phonation spans excepted.

        Phonation-span detection is this node's one localising exception (see the module docstring)
        but it is guarded on a real ``phonation_tracks`` measurement, which this fixture never seeds
        -- spans are seeded directly here, bypassing track computation entirely -- so the set of
        span ids is still unchanged; see ``TestPhonationSpans`` for the case where TAXONOMY does add
        spans.
        """
        seed_preprocess_store(
            store, yamnet_labels=[["Speech"]], words=["one", "two", "three"], phonation=[(0.0, 2.0, "voiced")]
        )
        before = {e.id for e in live_entities(store, "span")}
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert {e.id for e in live_entities(store, "span")} == before
        assert not live_entities(store, "interval")


def _resonate(excitation: np.ndarray, formants: np.ndarray, bandwidth: float = 80.0) -> np.ndarray:
    """Pass an excitation through one two-pole resonator per column of ``formants``."""
    out = excitation.astype(np.float64)
    r = float(np.exp(-np.pi * bandwidth / SR))
    for column in range(formants.shape[1]):
        track = formants[:, column]
        if float(track.min()) == float(track.max()):
            centre = float(track[0])
            out = lfilter([1.0 - r], [1.0, -2 * r * np.cos(2 * np.pi * centre / SR), r * r], out)
            continue
        filtered = np.zeros_like(out)
        previous, before_ = 0.0, 0.0
        for index, centre in enumerate(track):
            filtered[index] = (
                (1.0 - r) * out[index] + 2 * r * np.cos(2 * np.pi * centre / SR) * previous - r * r * before_
            )
            before_, previous = previous, filtered[index]
        out = filtered
    return np.asarray(out / (np.abs(out).max() + 1e-12), dtype=np.float64)


def _pulses(rate_hz: np.ndarray) -> np.ndarray:
    """A pulse train whose instantaneous rate follows ``rate_hz``."""
    out = np.zeros(len(rate_hz))
    phase = 0.0
    for index, rate in enumerate(rate_hz):
        phase += rate / SR
        if phase >= 1.0:
            phase -= 1.0
            out[index] = 1.0
    return out


def _fixed(centres: tuple[float, ...], n_samples: int) -> np.ndarray:
    """A constant formant track, one column per centre frequency."""
    return np.tile(np.array([list(centres)]), (n_samples, 1))


def _steady_vowel() -> np.ndarray:
    """1.5 s of a 200 Hz buzz through fixed resonators: a sustained, voiced production."""
    n_samples = int(1.5 * SR)
    return _resonate(_pulses(np.full(n_samples, 200.0)), _fixed((700.0, 1200.0, 2600.0), n_samples))


def _steady_noise() -> np.ndarray:
    """1.5 s of noise through the same resonators: a sustain with no periodicity at all."""
    n_samples = int(1.5 * SR)
    excitation = np.random.default_rng(0).standard_normal(n_samples)
    return _resonate(excitation, _fixed((700.0, 1200.0, 2600.0), n_samples))


def _broadband_noise() -> np.ndarray:
    """1.5 s of steady broadband noise, deliberately lacking vocal-tract resonances."""
    return np.random.default_rng(0).standard_normal(int(1.5 * SR))


def _rising_glide() -> np.ndarray:
    """0.3 s in which F0 and the resonances both sweep upward faster than either limb tolerates."""
    n_samples = int(0.3 * SR)
    ramp = np.linspace(0.0, 1.0, n_samples)
    resonances = np.stack([300.0 + 3300.0 * ramp, 900.0 + 3500.0 * ramp], axis=1)
    swept = _resonate(_pulses(150.0 * (480.0 / 150.0) ** ramp), resonances)
    return np.concatenate([swept, np.zeros(int(0.2 * SR))])


class TestPhonationSpans:
    """Sustains and glides, voiced, unvoiced or mixed, with duration_s as the primary feature.

    Detection moved here from PREPROCESS this session: PREPROCESS now only measures F0/formant
    tracks (see ``TestPhonationTracks`` in ``preprocess_test.py``); every test here runs PREPROCESS
    first (to produce ``phonation_tracks``) and then TAXONOMY (to localise the spans over it), on the
    same real Praat-backed tracker these fixtures always exercised.
    """

    def test_a_sustained_vowel_yields_a_span_carrying_its_duration(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 1.5 s steady tone is one sustained span whose duration_s is its extent."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        best = max(spans, key=lambda e: e.attributes["duration_s"])
        assert best.attributes["member"] == "sustained"
        assert best.extent is not None
        assert best.attributes["duration_s"] == pytest.approx(best.extent[1] - best.extent[0])
        assert best.attributes["duration_s"] > 1.0

    def test_an_unvoiced_sustain_is_a_span_like_any_other(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Steady band-limited noise sustains with no periodicity and is not refused."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_noise())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert spans
        assert any(e.attributes["production"] in ("unvoiced", "mixed") for e in spans)

    def test_broadband_noise_does_not_become_an_unvoiced_span(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stable broad LPC poles are insufficient to claim phonation from ordinary noise."""
        _seed_admit(store, tmp_path, wav_writer, samples=_broadband_noise())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]

    def test_a_glide_is_a_span_with_a_direction_and_an_excursion(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rising sweep is a glide, not a sustain, and carries where it went."""
        _seed_admit(store, tmp_path, wav_writer, samples=_rising_glide())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        glides = [
            e
            for e in live_entities(store, "span")
            if e.attributes.get("family") == "phonation" and e.attributes["member"] == "glide"
        ]
        assert glides
        assert glides[0].attributes["glide_direction"] == "rising"
        assert glides[0].attributes["glide_extent_cents"] > 0.0

    def test_formant_tracks_are_written_per_span_and_sliced_from_the_stream(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One formant_tracks measurement per span, each derived from the span it covers."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        tracks = find_measurements(store, "formant_tracks")
        assert len(tracks) == len(spans)
        assert set(store.derived_from(tracks[0].id)) & {e.id for e in spans}
        assert len(tracks[0].attributes["f1_hz"]) == len(tracks[0].attributes["times_s"])

    def test_the_formant_track_is_measured_once_over_the_whole_stream(
        self,
        store: ProvStore,
        phonation_config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """One call, over the whole stream, however many spans come out of it.

        "Tracks are computed once on the stream and then sliced" is a property of PREPROCESS's own
        call graph (it measures the tracks, once, before TAXONOMY ever runs), so the call graph is
        what this asserts: a per-span re-fit produces identical stored attributes on a steady fixture
        and is invisible to every other test in this file. The duration assertion is the half that
        matters -- a re-fit would pass a span-length fragment, and a fragment renormalises to its own
        maximum, which is the failure the rule exists to prevent.
        """
        seen_durations: list[float] = []
        measure = preprocess_module.formant_track

        def counting(audio: Any, **kwargs: Any) -> Any:  # noqa: ANN401 — delegates to the real one
            """Record what the tracker was handed, then track it for real."""
            seen_durations.append(audio.waveform.shape[-1] / audio.sampling_rate)
            return measure(audio, **kwargs)

        monkeypatch.setattr(preprocess_module, "formant_track", counting)
        samples = np.concatenate([_steady_vowel(), np.zeros(int(0.2 * SR)), _steady_vowel()])
        _seed_admit(store, tmp_path, wav_writer, samples=samples)
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), phonation_config, run_dir=tmp_path)
        taxonomy(store, "plain", phonation_config, run_dir=tmp_path)
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert len(spans) >= 2, "the fixture must yield several spans or a call count of one proves nothing"
        assert len(seen_durations) == 1
        assert seen_durations[0] == pytest.approx(3.2, abs=0.05)

    def test_a_null_criterion_leaves_the_spans_absent(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        wav_writer: Callable[..., Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The packaged config fits nothing, so the pass is absent rather than run on invented floors."""
        _seed_admit(store, tmp_path, wav_writer, samples=_steady_vowel())
        _stub_models(monkeypatch)
        preprocess(store, _audio(tmp_path), config, run_dir=tmp_path)
        taxonomy(store, "plain", config, run_dir=tmp_path)
        assert not [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
