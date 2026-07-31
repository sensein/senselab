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
