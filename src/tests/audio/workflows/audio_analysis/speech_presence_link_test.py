"""L1 evidence → L2 belief for the speech-presence axis (register items 1-5, 13, 15).

The harvester used to return ``{"speaks": bool, "native_confidence": float}`` per model. Both
fields are conclusions, and every threshold that produced them — segment coverage, word overlap,
Whisper's ``no_speech_prob`` gate, the frame-mean cut, the coarse-voter demotion — was applied
inside L1, where it could not be re-decided without re-running a model.

These tests pin the split: the harvester emits measurements in native units, and
``speech_presence_link`` turns them into beliefs under a named, replaceable policy. The test that
matters most is not that the defaults reproduce the old numbers (they do) but that a *different*
policy produces different beliefs from the *same* measurements — that is the property the
single-layer design could not have.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.speech_presence_link import (
    SpeechPresencePolicy,
    link_speech_presence,
)

# ── the L1 measurement: per-chunk scalars, not a bucket-level belief ─────────


def test_asr_chunk_evidence_keeps_each_chunk_rather_than_a_pooled_confidence() -> None:
    """L1 emits the per-chunk logprobs; pooling them is a choice with two defensible answers.

    ``mean(exp(x))`` and ``exp(mean(x))`` differ by Jensen's inequality, and the old harvester
    baked in the first. Keeping the list lets L2 pick, and lets a reader see that a pick was made.
    """
    from senselab.audio.workflows.audio_analysis.harvesters import asr_bucket_chunk_evidence

    result = [
        {
            "chunks": [
                {"start": 0.0, "end": 0.4, "avg_logprob": -0.1, "no_speech_prob": 0.02},
                {"start": 0.4, "end": 0.9, "avg_logprob": -2.0, "no_speech_prob": 0.30},
            ]
        }
    ]
    ev = asr_bucket_chunk_evidence(result, 0.0, 1.0)
    assert ev["avg_logprobs"] == pytest.approx([-0.1, -2.0])
    assert ev["no_speech_probs"] == pytest.approx([0.02, 0.30])
    assert ev["n_words"] == 2
    assert ev["word_overlap_s"] == pytest.approx(0.9)


def test_asr_chunk_evidence_unions_overlapping_word_spans() -> None:
    """Coverage is a union: two aligners' spans overlapping cannot exceed the bucket."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_bucket_chunk_evidence

    result = [{"chunks": [{"start": 0.0, "end": 0.8}, {"start": 0.2, "end": 1.0}]}]
    ev = asr_bucket_chunk_evidence(result, 0.0, 1.0)
    assert ev["word_overlap_s"] == pytest.approx(1.0)


def test_asr_chunk_evidence_clips_word_spans_to_the_bucket() -> None:
    """A word straddling the boundary contributes only the part inside this bucket."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_bucket_chunk_evidence

    result = [{"chunks": [{"start": -1.0, "end": 0.25}]}]
    ev = asr_bucket_chunk_evidence(result, 0.0, 1.0)
    assert ev["word_overlap_s"] == pytest.approx(0.25)


def test_asr_chunk_evidence_is_empty_without_a_result() -> None:
    """No transcript is not a transcript saying nothing happened."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_bucket_chunk_evidence

    ev = asr_bucket_chunk_evidence(None, 0.0, 1.0)
    assert ev["n_words"] == 0
    assert ev["avg_logprobs"] == []
    assert ev["word_overlap_s"] == pytest.approx(0.0)


# ── L2: the same measurements, different policies, different beliefs ─────────


def _bucket(evidence: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"start": 0.0, "end": 0.5, "evidence": evidence, "frame_dispersion": None}


def test_diar_coverage_threshold_is_a_policy_not_a_constant() -> None:
    """A segment grazing 5% of a bucket can be a speaker or noise; the policy decides.

    Under the default (any coverage counts) it reads as speech, which is what the old bool did.
    Raise the threshold and the same measurement reads as absence — the re-decision that was
    impossible while only the bool survived.
    """
    rows = [_bucket({"pyannote": {"covered_fraction": 0.05, "speaker_label": "SPEAKER_00"}})]

    lenient = link_speech_presence(rows)[0]["votes"]["pyannote"]
    strict = link_speech_presence(rows, policy=SpeechPresencePolicy(diar_coverage_threshold=0.5))[0]["votes"][
        "pyannote"
    ]
    assert lenient["speaks"] is True
    assert strict["speaks"] is False
    # Either way the measurement travels, so the parquet records what was seen, not only the verdict.
    assert lenient["covered_fraction"] == pytest.approx(0.05)
    assert lenient["speaker_label"] == "SPEAKER_00"


def test_diar_reports_no_native_confidence() -> None:
    """A segment boundary is a hard claim — there is no model-native confidence to report."""
    votes = link_speech_presence([_bucket({"pyannote": {"covered_fraction": 1.0, "speaker_label": "S"}})])[0]["votes"]
    assert votes["pyannote"]["native_confidence"] is None


def test_diar_with_no_segments_anywhere_votes_absence_rather_than_dropping_out() -> None:
    """``covered_fraction=None`` means the model ran and placed nothing here."""
    votes = link_speech_presence([_bucket({"pyannote": {"covered_fraction": None, "speaker_label": None}})])[0]["votes"]
    assert votes["pyannote"]["speaks"] is False


def test_asr_confidence_pooling_is_named_and_switchable() -> None:
    """Both poolings are available, and they differ — which is why L1 must not choose.

    ``mean(exp(x)) > exp(mean(x))`` strictly whenever the logprobs differ (Jensen), so the choice
    is not cosmetic: it changes how much a mixed-confidence bucket contributes.
    """
    rows = [
        _bucket({"whisper": {"avg_logprobs": [-0.1, -2.0], "no_speech_probs": [], "word_overlap_s": 0.5, "n_words": 2}})
    ]

    mean_exp = link_speech_presence(rows)[0]["votes"]["whisper"]["native_confidence"]
    exp_mean = link_speech_presence(rows, policy=SpeechPresencePolicy(asr_confidence_pooling="exp_of_mean"))[0][
        "votes"
    ]["whisper"]["native_confidence"]

    assert mean_exp == pytest.approx((math.exp(-0.1) + math.exp(-2.0)) / 2)
    assert exp_mean == pytest.approx(math.exp((-0.1 + -2.0) / 2))
    assert mean_exp > exp_mean, "Jensen's inequality — the two poolings are not interchangeable"


def test_asr_hallucination_gate_is_re_decidable_from_the_recorded_measurement() -> None:
    """Words over probable silence: the verdict is L2's, and the ``no_speech_prob`` is recorded."""
    rows = [
        _bucket(
            {
                "whisper": {
                    "avg_logprobs": [-0.2],
                    "no_speech_probs": [0.7],
                    "word_overlap_s": 0.5,
                    "n_words": 3,
                }
            }
        )
    ]
    default = link_speech_presence(rows)[0]["votes"]["whisper"]
    assert default["hallucinated"] is True
    assert default["speaks"] is False, "a transcript over 0.7 no-speech is suppressed by default"
    assert default["no_speech_prob"] == pytest.approx(0.7)

    trusting = link_speech_presence(rows, policy=SpeechPresencePolicy(no_speech_threshold=0.9))[0]["votes"]["whisper"]
    assert trusting["hallucinated"] is False
    assert trusting["speaks"] is True


def test_no_speech_prob_sibling_voter_inverts_at_L2() -> None:
    """``speaks = nsp < t`` and the confidence in that direction are interpretations, not measurements.

    The confidence must follow the direction. At ``nsp = 0.9`` the voter is 0.9-confident of
    *absence*; reporting ``1 - nsp`` regardless made ``_weighted_p_voice`` read that denial as
    ``p_voice = 0.9`` — a confident assertion of the opposite.
    """
    rows = [_bucket({"whisper::no_speech_prob": {"no_speech_prob": 0.2, "native_window_s": 30.0}})]
    vote = link_speech_presence(rows)[0]["votes"]["whisper::no_speech_prob"]
    assert vote["speaks"] is True
    assert vote["native_confidence"] == pytest.approx(0.8)
    assert vote["no_speech_prob"] == pytest.approx(0.2)

    denying = link_speech_presence([_bucket({"whisper::no_speech_prob": {"no_speech_prob": 0.9}})])[0]["votes"][
        "whisper::no_speech_prob"
    ]
    assert denying["speaks"] is False
    assert denying["native_confidence"] == pytest.approx(0.9)


def test_frame_posterior_threshold_and_channels_are_L2s() -> None:
    """Item 13: the bucket mean travels, the cut applied to it is policy, channels stay intact."""
    rows = [
        _bucket(
            {
                "frame_segmentation": {
                    "frame_mean": 0.4,
                    "frame_std": 0.1,
                    "n_frames": 30,
                    "channel_means": [0.4, 0.05],
                    "channel_labels": ["speaker#1", "speaker#2"],
                    "resolution_s": 0.0169,
                    "native_window_s": 5.0,
                }
            }
        )
    ]
    default = link_speech_presence(rows)[0]["votes"]["frame_segmentation"]
    assert default["speaks"] is False, "0.4 is below the default 0.5 cut"
    assert default["native_confidence"] == pytest.approx(0.6), "0.4 P(speech) is a 0.6-confident no"
    assert default["frame_mean"] == pytest.approx(0.4), "the measurement itself still travels"
    assert default["channel_means"] == pytest.approx([0.4, 0.05])
    assert default["channel_labels"] == ["speaker#1", "speaker#2"]

    sensitive = link_speech_presence(rows, policy=SpeechPresencePolicy(frame_speech_threshold=0.3))[0]["votes"][
        "frame_segmentation"
    ]
    assert sensitive["speaks"] is True
    assert sensitive["native_confidence"] == pytest.approx(0.4), "the direction and the confidence move together"


def test_frame_posterior_reaching_the_aggregator_is_not_inverted() -> None:
    """The defect this pins is not cosmetic: it inverted the presence axis where it matters most.

    ``_weighted_p_voice`` reads ``native_confidence`` as confidence in the voter's own ``speaks``
    direction. A frame posterior of 0.02 reported raw became a 0.02-confident *no*, which the
    aggregator maps to ``p_voice = 0.98`` — a near-certain claim of speech from a detector that had
    just reported near-certain silence. Every downstream gate keyed on ``p_voice`` (region seeding,
    diarization voicing, corroboration) then read silence as speech.
    """
    from senselab.audio.workflows.audio_analysis.aggregate import speech_presence_p_voice

    rows = [_bucket({"frame_brouhaha_vad": {"frame_mean": 0.02, "n_frames": 30}})]
    votes = link_speech_presence(rows)[0]["votes"]
    assert speech_presence_p_voice(votes) == pytest.approx(0.02)


def test_coarse_demotion_reads_the_declared_resolution_rather_than_a_hardcoded_flag() -> None:
    """Item 15: L1 declares ``native_window_s``; L2 compares it against the reporting grid.

    The old harvester set ``coarse: True`` by hand per voter and then applied a fixed 0.25 weight.
    Which voters are coarse is not a property of the voter — it is the relation between its window
    and the grid it is being reported on, so it can only be decided where both are known.
    """
    rows = [
        _bucket(
            {
                "ast": {"speech_label_mass": 0.8, "native_window_s": 10.24},
                "frame_brouhaha_vad": {"frame_mean": 0.9, "frame_std": 0.0, "n_frames": 12, "resolution_s": 0.0169},
            }
        )
    ]
    # Reported at the grid AST's window dwarfs: AST is demoted, the frame voter is not.
    fine = link_speech_presence(rows, reporting_win_s=0.1)[0]["votes"]
    assert fine["ast"]["weight"] == pytest.approx(SpeechPresencePolicy().coarse_voter_weight)
    assert "weight" not in fine["frame_brouhaha_vad"]

    # Reported at a grid as coarse as AST's own window: nothing is being stretched, no demotion.
    coarse = link_speech_presence(rows, reporting_win_s=10.24)[0]["votes"]
    assert "weight" not in coarse["ast"]


def test_scene_classifier_mass_maps_to_a_direction_and_a_confidence() -> None:
    """A mass of 0.38 is a 0.62-confident *no*; a mass of 0.8 is an 0.8-confident *yes*."""
    rows = [_bucket({"yamnet": {"speech_label_mass": 0.38, "native_window_s": 0.96}})]
    vote = link_speech_presence(rows, reporting_win_s=0.96)[0]["votes"]["yamnet"]
    assert vote["speaks"] is False
    assert vote["native_confidence"] == pytest.approx(0.62)
    assert vote["speech_label_mass"] == pytest.approx(0.38)


def test_level_above_floor_still_abstains_at_low_excess() -> None:
    """The asymmetry survives the move: a low excess is uninformative, not a denial.

    A source running through the whole recording is absorbed into its own floor estimate, so a low
    excess has two indistinguishable causes and the signal must not assert either.
    """
    rows = [
        _bucket({"acoustic_level_above_floor": {"excess_db": 0.0}}),
        {
            "start": 0.5,
            "end": 1.0,
            "evidence": {"acoustic_level_above_floor": {"excess_db": 0.5}},
            "frame_dispersion": None,
        },
        {
            "start": 1.0,
            "end": 1.5,
            "evidence": {"acoustic_level_above_floor": {"excess_db": 30.0}},
            "frame_dispersion": None,
        },
    ]
    linked = link_speech_presence(rows)
    at_floor = linked[0]["votes"]["acoustic_level_above_floor"]["native_confidence"]
    just_above = linked[1]["votes"]["acoustic_level_above_floor"]["native_confidence"]
    assert at_floor == pytest.approx(0.5), "nothing above the floor is uninformative, not absence"
    # The ramp only leaves abstention gradually, so a marginal excess stays near-uninformative
    # rather than crossing into a claim.
    assert 0.5 < just_above < 0.53
    assert linked[2]["votes"]["acoustic_level_above_floor"]["speaks"] is True


def test_hnr_abstains_low_and_asserts_high() -> None:
    """Whispered speech has low HNR, so a low value must not be read as silence."""
    quiet = link_speech_presence([_bucket({"acoustic_hnr": {"hnr_db": 0.0}})])[0]["votes"]["acoustic_hnr"]
    voiced = link_speech_presence([_bucket({"acoustic_hnr": {"hnr_db": 14.0}})])[0]["votes"]["acoustic_hnr"]
    assert quiet["native_confidence"] == pytest.approx(0.5)
    assert voiced["speaks"] is True
    assert voiced["native_confidence"] == pytest.approx(1.0)


def test_link_is_pure_and_leaves_the_evidence_untouched() -> None:
    """Re-linking under a second policy must see the original measurements, not the first pass's.

    Mutating the evidence in place would make the layer order silently matter, and the adaptive
    loop re-links the same harvest every round.
    """
    rows = [_bucket({"pyannote": {"covered_fraction": 0.05, "speaker_label": "S"}})]
    before = copy.deepcopy(rows)
    link_speech_presence(rows, policy=SpeechPresencePolicy(diar_coverage_threshold=0.5), reporting_win_s=0.1)
    assert rows == before, "linking must not write back into the measurements it read"
    assert "votes" not in rows[0]


def test_frame_dispersion_passes_through_unrescaled() -> None:
    """The link layer decides beliefs; the dispersion→doubt mapping stays in ``votes.py``."""
    rows = [{"start": 0.0, "end": 0.5, "evidence": {}, "frame_dispersion": 0.37}]
    assert link_speech_presence(rows)[0]["frame_dispersion"] == pytest.approx(0.37)


def test_policy_from_params_reads_the_run_configuration() -> None:
    """Thresholds reach L2 through run params, so a run records the policy it used."""
    from senselab.audio.workflows.audio_analysis.speech_presence_link import policy_from_params

    assert policy_from_params({}) == SpeechPresencePolicy()
    tuned = policy_from_params({"speech_presence_policy": {"no_speech_threshold": 0.8, "unknown_key": 1}})
    assert tuned.no_speech_threshold == pytest.approx(0.8)
    assert tuned.frame_speech_threshold == pytest.approx(SpeechPresencePolicy().frame_speech_threshold)


# ── register item 11: the PPG argmax discards the distribution ───────────────


def test_ppg_silence_posterior_keeps_what_the_argmax_threw_away() -> None:
    """The same defect as the AST/YAMNet top-1, on a different model.

    Counting frames whose argmax is not ``<silent>`` reduces each frame's whole distribution to a
    hard 0 or 1. A frame that is 60% silent votes exactly as confidently as one that is 100%
    silent, so a bucket the model was barely sure about is indistinguishable from one it was
    certain about. The posterior keeps the difference.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.harvesters import (
        ppg_argmax_per_frame,
        ppg_silence_posterior_per_frame,
    )

    labels = ["aa", "<silent>"]
    # Every frame leans silent, but only just: argmax says "silent everywhere".
    posteriors = np.tile(np.array([[0.4, 0.6]]), (10, 1)).T  # (phonemes, frames)
    per_frame, _ = ppg_argmax_per_frame([posteriors], labels, 1.0)
    assert set(per_frame) == {"<silent>"}, "the argmax is unanimous"

    silence, hop = ppg_silence_posterior_per_frame([posteriors], labels, 1.0)
    assert silence.shape == (10,)
    assert hop == pytest.approx(0.1)
    assert silence.mean() == pytest.approx(0.6), "the posterior records how sure the model was"


def test_ppg_vote_reads_the_posterior_not_a_frame_count() -> None:
    """L2 maps mean silence posterior to voice probability; the hard count is gone."""
    rows = [_bucket({"ppg_voice_fraction": {"mean_silence_posterior": 0.6, "n_frames": 10}})]
    vote = link_speech_presence(rows)[0]["votes"]["ppg_voice_fraction"]
    # 0.6 silent → 0.4 voice → a 0.6-confident *no*, not the flat "no voice" the count gave.
    assert vote["speaks"] is False
    assert vote["native_confidence"] == pytest.approx(0.6)
    assert vote["mean_silence_posterior"] == pytest.approx(0.6)


def test_ppg_vote_absent_without_a_posterior() -> None:
    """A bucket with no PPG frames drops the signal rather than voting from nothing."""
    rows = [_bucket({"ppg_voice_fraction": {"n_frames": 0}})]
    assert "ppg_voice_fraction" not in link_speech_presence(rows)[0]["votes"]


# ── register item 12: clustering is an L2 derivation over L1 vectors ─────────


def _embedding_windows(vectors: "list[list[float]]", width_s: float = 2.0, hop_s: float = 1.0) -> list[Any]:
    import numpy as np

    from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding

    return [
        WindowEmbedding(start_s=i * hop_s, end_s=i * hop_s + width_s, vector=np.asarray(v, dtype=np.float64))
        for i, v in enumerate(vectors)
    ]


def _two_speaker_windows() -> list[Any]:
    """Six windows in two well-separated directions — a clustering with real structure."""
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]
    return _embedding_windows([a, a, a, b, b, b])


def test_silhouette_is_derived_at_L2_not_measured_at_L1() -> None:
    """The harvester no longer accepts embeddings, and no evidence row carries a silhouette.

    Clustering needs every window of the pass at once and yields a conclusion about speaker
    structure, so it is a derived signal (D-7) rather than a measurement of any one bucket.
    """
    import inspect

    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence

    params = inspect.signature(harvest_speech_presence_evidence).parameters
    assert "per_window_embeddings" not in params, "L1 must not be given the vectors to cluster"


def test_link_derives_the_silhouette_vote_from_the_vectors() -> None:
    """Given L1 vectors, the vote appears — and carries the score it was derived from."""
    rows = [_bucket({}), {"start": 3.0, "end": 3.5, "evidence": {}, "frame_dispersion": None}]
    linked = link_speech_presence(rows, per_window_embeddings={"ecapa": _two_speaker_windows()})
    vote = linked[0]["votes"]["embedding_silhouette"]
    assert "silhouette" in vote and 0.0 <= vote["silhouette"] <= 1.0
    assert vote["embedding_model"] == "ecapa"
    assert vote["native_window_s"] == pytest.approx(2.0)


def test_silhouette_vote_carries_the_cluster_assignment() -> None:
    """The half of the computation that assigns labels must survive, not only the voicing score.

    A later stage repairs speaker labels; if it re-clustered, the two stages could disagree about
    the structure they are each reasoning over.
    """
    rows = [
        _bucket({}),
        {"start": 4.0, "end": 4.5, "evidence": {}, "frame_dispersion": None},
    ]
    linked = link_speech_presence(rows, per_window_embeddings={"ecapa": _two_speaker_windows()})
    ids = [b["votes"]["embedding_silhouette"].get("cluster_id") for b in linked]
    assert all(i is not None for i in ids)
    assert ids[0] != ids[1], "buckets 4 s apart sit in the two different speaker clusters"


def test_derive_window_clusters_exposes_the_whole_result() -> None:
    """Callers that need the clustering itself get it, rather than only the per-window score."""
    from senselab.audio.workflows.audio_analysis.speech_presence_link import derive_window_clusters

    derived = derive_window_clusters({"ecapa": _two_speaker_windows()})
    assert derived is not None
    assert derived["model"] == "ecapa"
    assert set(derived["clusters"]) >= {"n_speakers", "labels", "p_voice"}
    assert derived["clusters"]["n_speakers"] == 2


def test_no_embeddings_means_no_silhouette_vote() -> None:
    """Absent vectors drop the signal rather than contributing a neutral one."""
    linked = link_speech_presence([_bucket({})])
    assert "embedding_silhouette" not in linked[0]["votes"]
    assert "embedding_silhouette" not in link_speech_presence([_bucket({})], per_window_embeddings={})[0]["votes"]
