"""Cross-signal physical support as a measured weight (replaces the declared gate).

A signal's weight must come from quantities, not from a name in a config. The declared
`embedding_silhouette: derived` gate was calibrated on one recording and demonstrably wrong
on another: on a 4.9 s group introduction the two "independent" diarizers merged four named
speakers into one, while the down-weighted clusterer recovered all five in the right places.

What can be measured without ground truth is whether a signal's claims are *physically
supported*: a diarizer placing a speaker where independent, non-diarizer evidence reports
silence or non-speech background has made a claim the audio does not carry.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.support import (
    SUPPORT_FLOOR,
    signal_support,
)


def _bucket(start: float, speaks: dict[str, bool], evidence: dict[str, float]) -> dict:
    """One presence bucket: which signals claimed speech, plus independent evidence."""
    votes: dict[str, object] = {m: {"speaks": v, "native_confidence": None} for m, v in speaks.items()}
    votes.update({m: {"p_speech": p} for m, p in evidence.items()})
    return {"start": start, "end": start + 0.5, "votes": votes}


def test_a_signal_claiming_speech_where_evidence_agrees_keeps_full_weight() -> None:
    """The anchor: a corroborated claim must not be discounted."""
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.95}), _bucket(0.5, {"diar": True}, {"vad": 0.90})]
    assert signal_support(buckets, evidence_signals=["vad"])["diar"] == pytest.approx(1.0, abs=0.1)


def test_a_signal_counting_a_speaker_during_silence_is_down_weighted() -> None:
    """The user's criterion, stated directly: a speaker claimed where there is no speech."""
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.02}), _bucket(0.5, {"diar": True}, {"vad": 0.01})]
    assert signal_support(buckets, evidence_signals=["vad"])["diar"] < 0.2


def test_support_is_attenuation_not_exclusion() -> None:
    """Support attenuates rather than excludes.

    A wholly unsupported signal stays visible, as with the perturbation floor: it may be the
    only source that noticed something.
    """
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.0})]
    assert signal_support(buckets, evidence_signals=["vad"])["diar"] >= SUPPORT_FLOOR


def test_buckets_the_signal_made_no_claim_in_are_not_held_against_it() -> None:
    """Silence about a bucket is not a claim about it.

    Otherwise a correctly-conservative signal would be penalised for every region it
    declined to speak for, which is the opposite of what support should measure.
    """
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.95}), _bucket(0.5, {"diar": False}, {"vad": 0.01})]
    assert signal_support(buckets, evidence_signals=["vad"])["diar"] == pytest.approx(1.0, abs=0.1)


def test_a_signal_that_claims_nothing_anywhere_gets_no_support_measure() -> None:
    """Nothing was measured, so nothing is asserted — it keeps its default weight."""
    buckets = [_bucket(0.0, {"diar": False}, {"vad": 0.9})]
    assert "diar" not in signal_support(buckets, evidence_signals=["vad"])


def test_other_diarizers_do_not_count_as_supporting_evidence() -> None:
    """Agreement among diarizers is not physical support.

    A bad diarizer can always say one speaker, so three models claiming a speaker in silence
    must all be down-weighted rather than mutually validated.
    """
    buckets = [
        _bucket(0.0, {"a": True, "b": True, "c": True}, {"vad": 0.01}),
        _bucket(0.5, {"a": True, "b": True, "c": True}, {"vad": 0.02}),
    ]
    support = signal_support(buckets, evidence_signals=["vad"])
    assert set(support) == {"a", "b", "c"}
    assert all(v < 0.2 for v in support.values())


def test_evidence_signals_are_never_scored_against_themselves() -> None:
    """A signal cannot be its own corroboration."""
    buckets = [_bucket(0.0, {"vad": True}, {"vad": 0.9})]
    assert "vad" not in signal_support(buckets, evidence_signals=["vad"])


def test_with_no_independent_evidence_no_signal_is_penalised() -> None:
    """Absent evidence is not evidence of absence.

    A run where no independent presence signal was available must not silently down-weight
    every diarizer — that would make a missing model look like a wrong one.
    """
    buckets = [_bucket(0.0, {"diar": True}, {})]
    assert signal_support(buckets, evidence_signals=["vad"]) == {}


def test_signals_are_scored_independently_of_each_other() -> None:
    """A well-supported signal must not inherit a badly-supported one's discount."""
    buckets = [
        _bucket(0.0, {"good": True, "bad": False}, {"vad": 0.95}),
        _bucket(0.5, {"good": False, "bad": True}, {"vad": 0.01}),
    ]
    support = signal_support(buckets, evidence_signals=["vad"])
    assert support["good"] > 0.8
    assert support["bad"] < 0.2


def test_partial_support_lands_between() -> None:
    """Support is graded, not a threshold — half-corroborated claims read as half-supported."""
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.95}), _bucket(0.5, {"diar": True}, {"vad": 0.05})]
    assert 0.3 < signal_support(buckets, evidence_signals=["vad"])["diar"] < 0.7


def test_several_evidence_signals_are_pooled() -> None:
    """One evidence signal failing must not decide support on its own."""
    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.9, "audioset_speech": 0.8})]
    assert signal_support(buckets, evidence_signals=["vad", "audioset_speech"])["diar"] > 0.7


# ── the evidence set is derived structurally, not listed in config ─────


def test_frame_posteriors_and_scene_classifiers_are_recognised_as_evidence() -> None:
    """A config list would drift the moment a voter is renamed or added.

    Frame posteriors and scene classifiers observe speech presence directly; diarizers and
    ASR infer it from a decision that already presupposes a speaker.
    """
    from senselab.audio.workflows.audio_analysis.support import evidence_signal_names

    votes = {
        "frame_segmentation": {"speaks": True, "native_confidence": 1.0},
        "frame_brouhaha_vad": {"speaks": True, "native_confidence": 0.99},
        "ast": {"speaks": True, "native_confidence": 0.58, "coarse": True},
        "yamnet": {"speaks": True, "native_confidence": 0.98, "coarse": True},
        "pyannote/speaker-diarization-community-1": {"speaks": True, "native_confidence": None},
        "nyralabs/CrisperWhisper2.0_turbo": {"speaks": True, "native_confidence": None},
        "__sources__": {"classifiers": ["ast", "yamnet"]},
    }
    names = evidence_signal_names([{"start": 0.0, "end": 0.5, "votes": votes}])
    assert {"frame_segmentation", "frame_brouhaha_vad", "ast", "yamnet"} <= names
    assert "pyannote/speaker-diarization-community-1" not in names
    assert "nyralabs/CrisperWhisper2.0_turbo" not in names


def test_a_newly_added_classifier_is_picked_up_from_the_harvest() -> None:
    """No second place to update.

    The harvest already declares which voters are classifiers; reading that keeps the
    evidence set correct as voters come and go.
    """
    from senselab.audio.workflows.audio_analysis.support import evidence_signal_names

    votes = {
        "beats": {"speaks": True, "native_confidence": 0.7},
        "__sources__": {"classifiers": ["beats"]},
    }
    assert "beats" in evidence_signal_names([{"start": 0.0, "end": 0.5, "votes": votes}])


def test_bookkeeping_entries_are_not_evidence() -> None:
    """``__quality__`` and ``__sources__`` carry metadata, not presence observations."""
    from senselab.audio.workflows.audio_analysis.support import evidence_signal_names

    votes = {"__quality__": {"snr_estimates_db": [12.0]}, "__sources__": {"classifiers": []}}
    assert evidence_signal_names([{"start": 0.0, "end": 0.5, "votes": votes}]) == set()


# ── the weight a signal's doubt actually carries ───────────────────────


def test_the_weight_is_stability_times_support() -> None:
    """Both factors are measured; neither is declared.

    Stability asks whether a signal agrees with itself under a transform, support whether the
    audio carries what it claimed.
    """
    from senselab.audio.workflows.audio_analysis.reliability import measured_weights

    steady_supported = measured_weights({"a": 0.0}, {"a": 1.0}, ["a"])["a"]
    steady_unsupported = measured_weights({"a": 0.0}, {"a": 0.1}, ["a"])["a"]
    unstable_supported = measured_weights({"a": 0.9}, {"a": 1.0}, ["a"])["a"]
    assert steady_supported == pytest.approx(1.0)
    assert steady_unsupported < steady_supported
    assert unstable_supported < steady_supported


def test_both_factors_compound() -> None:
    """A signal that is both unstable and unsupported is discounted by each."""
    from senselab.audio.workflows.audio_analysis.reliability import measured_weights

    both = measured_weights({"a": 0.9}, {"a": 0.1}, ["a"])["a"]
    one = measured_weights({"a": 0.9}, {"a": 1.0}, ["a"])["a"]
    assert both < one


def test_an_unmeasured_signal_keeps_full_weight() -> None:
    """Neither factor may discount a signal on evidence that was never gathered."""
    from senselab.audio.workflows.audio_analysis.reliability import measured_weights

    assert measured_weights({}, {}, ["a"])["a"] == pytest.approx(1.0)


def test_an_identity_signal_inherits_its_claimants_support() -> None:
    """Support resolves through the claimant.

    Identity sub-signals are keyed ``<diar>::<embedding>``, but the claim about where a
    speaker is belongs to the diar model, so that is whose support applies.
    """
    from senselab.audio.workflows.audio_analysis.reliability import measured_weights

    w = measured_weights({}, {"pyannote/x": 0.1}, ["pyannote/x::speechbrain/y"])
    assert w["pyannote/x::speechbrain/y"] < 0.5


def test_no_signal_is_named_in_the_weighting_logic() -> None:
    """The regression guard for the whole change: weights must not depend on identity.

    A gate keyed on a model name encodes a judgement from one recording, and a judgement from
    one recording was wrong about this exact model on a second recording.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis import reliability

    source = inspect.getsource(reliability)
    body = "\n".join(line for line in source.splitlines() if not line.strip().startswith("#") and '"""' not in line)
    for name in ("embedding_silhouette", "pyannote", "sortformer", "speechbrain"):
        assert name not in body, f"{name!r} appears in weighting logic; weights must be measured"


# ── uninformative evidence is not evidence ────────────────────────────


def test_sparse_evidence_is_informative_even_though_it_is_mostly_low() -> None:
    """A percentile spread would call this flat, which is backwards.

    Speech evidence on a mostly-silent recording is near zero almost everywhere and high in
    the few buckets that matter. Judged by p90 - p10 that reads as no variation at all, and
    the one signal that actually locates speech gets discarded.
    """
    from senselab.audio.workflows.audio_analysis.support import informative_evidence

    buckets = [_bucket(0.0, {"diar": True}, {"vad": 0.95})]
    buckets += [_bucket(0.5 * i, {"diar": True}, {"vad": 0.01}) for i in range(1, 8)]
    assert "vad" in informative_evidence(buckets, ["vad"])


def test_a_near_constant_evidence_signal_is_not_used() -> None:
    """Measured on a real run: support came out 0.967-1.000 for every claimant, i.e. inert.

    The cause was pooling by max over an evidence set that included acoustic proxies
    reporting ~0.57 everywhere, silence included. A signal with no variation across the file
    cannot say *where* speech is, so admitting it guarantees every claim looks supported.
    Discrimination is a property of the signal and is measurable without any example.
    """
    from senselab.audio.workflows.audio_analysis.support import informative_evidence

    buckets = [_bucket(i * 0.5, {"diar": True}, {"flat": 0.57, "vad": 0.95 if i < 2 else 0.02}) for i in range(6)]
    informative = informative_evidence(buckets, ["flat", "vad"])
    assert "vad" in informative
    assert "flat" not in informative


def test_dropping_flat_evidence_restores_discrimination() -> None:
    """The end-to-end consequence: the same claims now separate instead of all scoring ~1."""
    from senselab.audio.workflows.audio_analysis.support import informative_evidence

    buckets = [_bucket(0.0, {"honest": True, "overclaimer": True}, {"flat": 0.57, "vad": 0.95})]
    buckets += [
        _bucket(0.5 * i, {"honest": False, "overclaimer": True}, {"flat": 0.57, "vad": 0.01}) for i in range(1, 6)
    ]
    naive = signal_support(buckets, evidence_signals=["flat", "vad"])
    refined = signal_support(buckets, evidence_signals=sorted(informative_evidence(buckets, ["flat", "vad"])))
    assert naive["overclaimer"] > 0.5, "max-pooling over flat evidence hides the overclaim"
    assert refined["overclaimer"] < 0.4
    assert refined["honest"] > 0.8


def test_all_evidence_flat_means_no_support_measure() -> None:
    """With nothing informative left, no signal is penalised — the absent-evidence rule."""
    from senselab.audio.workflows.audio_analysis.support import informative_evidence

    buckets = [_bucket(i * 0.5, {"diar": True}, {"flat": 0.5}) for i in range(4)]
    assert informative_evidence(buckets, ["flat"]) == set()
    assert signal_support(buckets, evidence_signals=[]) == {}


def test_a_signal_that_never_says_no_is_not_evidence() -> None:
    """Support runs entirely on negative evidence, so this is the criterion that matters.

    Measured over 697 buckets of a real recording, four of seven candidates never once fell
    below 0.20 — two acoustic proxies, a spectral-activity heuristic, and AST — and pooled
    alongside genuine VAD they held support at 0.996 for every claimant. A signal that cannot
    say "no speech here" cannot withhold support from anything.
    """
    from senselab.audio.workflows.audio_analysis.support import informative_evidence

    # Varies by 0.5 — passes a range test — but never reaches a negative verdict.
    buckets = [
        _bucket(0.5 * i, {"diar": True}, {"eager": 0.5 + 0.05 * (i % 10), "vad": 0.9 if i < 3 else 0.01})
        for i in range(20)
    ]
    informative = informative_evidence(buckets, ["eager", "vad"])
    assert "vad" in informative
    assert "eager" not in informative, "a signal that never says no cannot withhold support"
