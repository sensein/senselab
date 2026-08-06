"""L2 link layer for the speech-presence axis: measurements → beliefs.

L1 (``speech_presence.py``) reports what each tool measured in that tool's own units — segment
coverage as a fraction, word spans in seconds, Whisper's per-chunk log-probabilities, dB above the
measured noise floor, frame posteriors and their per-speaker channels. Nothing there decides
whether a bucket contains speech.

This module decides. Every threshold, inversion, and pooling that used to sit inside the harvester
lives in :class:`SpeechPresencePolicy`, so each is named, replaceable, and recorded with the run
rather than compiled into the measurement.

Why the split is not cosmetic — three properties it buys that the single layer could not have:

**A verdict can be revisited without re-running a model.** A diarization segment grazing 5% of a
bucket and one covering all of it both set ``speaks=True``. Once the bool was the only survivor,
nothing downstream could tell them apart, and the difference matters most at segment boundaries —
exactly where speaker uncertainty peaks.

**A pooling choice becomes visible.** Whisper's bucket confidence was ``mean(exp(avg_logprob))``.
By Jensen's inequality that strictly exceeds ``exp(mean(avg_logprob))`` whenever the chunks
disagree, so the two are different statistics and one of them had been picked silently. L1 now
emits the per-chunk list and :attr:`SpeechPresencePolicy.asr_confidence_pooling` names the choice.

**"Coarse" stops being a property of a voter.** The old harvester hand-marked AST, YAMNet and the
Whisper segment voters ``coarse: True`` and applied a fixed 0.25 weight below a 0.5 s grid. But a
voter is only coarse *relative to the grid it is reported on*: AST's 10.24 s window is stretched
across 100 buckets at 0.1 s and across none at 10.24 s. That comparison needs both numbers, so it
can only be made here.

Two asymmetries are deliberate and are preserved verbatim from the harvester, because they encode
measured limits of the signals rather than tuning:

- **Level-above-floor abstains at low excess.** The floor is a percentile of this file's own
  frames, so a source that never stops *is* the floor. A low excess is therefore ambiguous between
  "nothing is happening" and "something is happening continuously", and voting absence there made
  the signal contradict correct models on any recording without pauses.
- **HNR abstains at low values.** Whispered and distorted voice both read low, so a low HNR cannot
  distinguish them from silence.

In both cases the signal maps its uninformative end to ``0.5`` rather than to a denial. LUFS keeps
the ability to claim absence, because −90 LUFS is unambiguous on an absolute scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

__all__ = [
    "SpeechPresencePolicy",
    "directed_presence_vote",
    "link_speech_presence",
    "policy_from_params",
    "votes_for_harvest",
    "derive_window_clusters",
]

AsrConfidencePooling = Literal["mean_of_exp", "exp_of_mean"]


@dataclass(frozen=True)
class SpeechPresencePolicy:
    """Every interpretation applied to a speech-presence measurement, in one place.

    Frozen so a policy cannot drift mid-run: the adaptive loop re-links the same harvest each
    round, and a mutated policy would make round *n*'s beliefs unreproducible from the log.

    Attributes:
        diar_coverage_threshold: Fraction of the bucket a diarization segment must cover to count
            as a speaker claim. The default ``0.0`` means "any overlap", which reproduces the
            boolean the harvester used to emit.
        word_overlap_threshold_s: Seconds of the bucket that transcript word spans must cover for
            the ASR voter to claim speech. Default ``0.0`` — any overlap.
        no_speech_threshold: Whisper ``no_speech_prob`` at or above which a transcript over this
            bucket is treated as hallucinated, and at or above which the sibling ``no_speech_prob``
            voter reports absence.
        asr_confidence_pooling: How per-chunk log-probabilities become one bucket confidence.
            ``"mean_of_exp"`` averages per-chunk confidences (each chunk is an independent "am I
            confident here?" vote); ``"exp_of_mean"`` exponentiates the mean log-probability (the
            unbiased aggregation of a log-domain quantity).
        label_mass_threshold: Share of a scene classifier's score mass on the speech label set
            above which the window reads as speech.
        frame_speech_threshold: Bucket-mean frame posterior above which a frame voter reports
            speech.
    """

    diar_coverage_threshold: float = 0.0
    word_overlap_threshold_s: float = 0.0
    no_speech_threshold: float = 0.5
    asr_confidence_pooling: AsrConfidencePooling = "mean_of_exp"
    label_mass_threshold: float = 0.5
    frame_speech_threshold: float = 0.5
    speech_excess_db: float = 12.0
    lufs_silence: float = -60.0
    lufs_speech: float = -30.0
    coarse_voter_weight: float = 0.25
    coarse_window_ratio: float = 2.0


DEFAULT_POLICY = SpeechPresencePolicy()
"""The policy whose numbers match the thresholds the harvester used to apply inline."""


def policy_from_params(params: Mapping[str, Any] | None) -> SpeechPresencePolicy:
    """Build a policy from run params, ignoring keys the policy does not define.

    Unknown keys are dropped rather than raising, and *silently* accepting them would be worse
    than either: a misspelt threshold would appear to have been applied while doing nothing. They
    are dropped because ``params`` is the whole CLI parameter block, not a policy document — most
    of its keys legitimately belong to other stages.
    """
    raw = (params or {}).get("speech_presence_policy")
    if not isinstance(raw, Mapping):
        return DEFAULT_POLICY
    known = {f.name for f in fields(SpeechPresencePolicy)}
    updates = {str(k): v for k, v in raw.items() if str(k) in known}
    return replace(DEFAULT_POLICY, **updates) if updates else DEFAULT_POLICY


def _finite(value: Any) -> float | None:  # noqa: ANN401 — evidence values are duck-typed
    """Coerce an evidence field to a finite float, or ``None`` if it is not a usable number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _ramp(value: float, low: float, high: float) -> float:
    """Linear ``[low, high] → [0, 1]``, clamped outside."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


def _abstaining_ramp(value: float, low: float, high: float) -> float | None:
    """Linear ramp into ``(0.5, 1]``, or ``None`` at or below ``low`` — where it cannot tell.

    Used where a low reading has two indistinguishable causes (see the module docstring on HNR and
    level-above-floor). Mapping that end to ``0.0`` would let the signal contradict correct models on
    inputs where it simply cannot tell.

    **An abstention is the absence of a claim, so it returns ``None``.** It used to return ``0.5``,
    which ``_directed`` then turned into ``speaks=True`` at confidence ``0.5`` — a half-confident
    *yes* cast in exactly the region where the signal has no opinion, and read by the fold as ``0.5``
    of doubt: the largest contribution a single voter can make. So the buckets where HNR knew least
    were the buckets where it pushed hardest.

    ``link_speech_presence`` has always had the correct rule for this, twenty lines below and stated
    in its own words: *"The signal reported nothing usable in this bucket. Dropping the vote is right:
    a fabricated 0.5 would be indistinguishable from a real abstention."* That is precisely what this
    function was manufacturing.

    Measured on a clean two-speaker conversation, ``acoustic_hnr`` contributed mean 0.2675 doubt
    (median 0.2574, max 0.5000) while all four diarizers, all three recognizers and the brouhaha VAD
    read exactly 0.0000. It is genuine voicing evidence where HNR is high — vowels are periodic —
    which is why the signal stays; what goes is its vote in the range where a low reading means
    "voiceless" and "absent" equally well.

    The graded region is untouched: only ``value <= low`` abstains, so a reading part-way up the ramp
    still votes at the strength the ramp gives it. This is deweighting by *scope* rather than by a
    hand-set discount — the kind ``fuse.cross_axis_inputs`` documents as inadmissible, since a factor
    never measured must not act as a discount.
    """
    ramped = _ramp(value, low, high)
    if ramped <= 0.0:
        return None
    return 0.5 + 0.5 * ramped


def _directed(p_voice: float) -> tuple[bool, float]:
    """``P(speech)`` → ``(speaks, confidence in that direction)``.

    The aggregator reads ``native_confidence`` as the voter's confidence in *its own* ``speaks``
    direction, so a 0.2 probability of speech is an 0.8-confident *no* rather than a weak *yes*.
    """
    speaks = p_voice >= 0.5
    return speaks, p_voice if speaks else 1.0 - p_voice


def directed_presence_vote(p_speech: float, *, threshold: float = 0.5) -> dict[str, Any]:
    """``P(speech)`` → the presence-vote payload, with the confidence read in the direction cast.

    The one place a raw ``P(speech)`` becomes a vote, so that a producer outside this module cannot
    reintroduce the inversion by hand-building the dict. Every hand-built payload found so far had
    the same shape — ``{"speaks": p >= 0.5, "native_confidence": p}`` — which
    :func:`~senselab.audio.workflows.audio_analysis.support.presence_probability` reads as
    ``1 − p`` whenever the vote is a *no*, turning a 0.02 posterior into 0.98.

    Args:
        p_speech: The voter's probability of speech in this bucket, in ``[0, 1]``.
        threshold: Cut separating a *yes* from a *no*. Passed through rather than fixed at 0.5 so a
            moved policy cut moves the direction and the confidence together.

    Returns:
        ``{"speaks": bool, "native_confidence": float}`` — the confidence is ``p_speech`` for a
        *yes* and its complement for a *no*, so ``presence_probability`` recovers ``p_speech``
        exactly either way.
    """
    p = max(0.0, min(1.0, float(p_speech)))
    speaks = p >= float(threshold)
    return {"speaks": speaks, "native_confidence": p if speaks else 1.0 - p}


def _pool_confidence(logprobs: Sequence[float], pooling: str) -> float | None:
    """Per-chunk log-probabilities → one bucket confidence under the named pooling."""
    values = [v for v in (_finite(x) for x in logprobs) if v is not None]
    if not values:
        return None
    try:
        if pooling == "exp_of_mean":
            return max(0.0, min(1.0, math.exp(sum(values) / len(values))))
        confidences = [max(0.0, min(1.0, math.exp(v))) for v in values]
    except (OverflowError, ValueError):
        return None
    return sum(confidences) / len(confidences)


def _mean(values: Iterable[Any]) -> float | None:
    """Mean of the finite entries, or ``None`` when none are usable."""
    kept = [v for v in (_finite(x) for x in values) if v is not None]
    return sum(kept) / len(kept) if kept else None


# ── per-signal link rules ────────────────────────────────────────────────────
#
# Each rule receives one signal's L1 measurements and returns the belief fields to merge over a
# copy of those measurements — so the vote always carries both what was seen and what was
# concluded, and a consumer can check the second against the first.


def _link_diar(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any]:
    """Coverage → claim. ``None`` coverage means the model ran and placed nothing here."""
    covered = _finite(ev.get("covered_fraction"))
    speaks = covered is not None and covered > policy.diar_coverage_threshold
    # No native confidence: a segment boundary is asserted, not scored. Reporting a number here
    # would invent a precision the model never expressed.
    return {"speaks": speaks, "native_confidence": None}


def _link_asr(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any]:
    """Word coverage → claim, per-chunk logprobs → confidence, ``no_speech_prob`` → gate.

    The two *no* branches take their confidence from something other than the token scores, because
    a pooled token confidence scores the **transcript** — it is confidence that these words were
    said, so it is evidence *for* speech and can never be this voter's confidence in silence.
    Reporting it undirected meant a hallucination flagged at token confidence 0.9 arrived as
    ``P(speech) = 0.1`` while the *same* flag at 0.2 arrived as 0.8: the surer the model was of its
    words, the more confidently this voter denied that anyone spoke.
    """
    overlap = _finite(ev.get("word_overlap_s")) or 0.0
    said_something = overlap > policy.word_overlap_threshold_s
    nsp = _mean(ev.get("no_speech_probs") or [])
    hallucinated = bool(said_something and nsp is not None and nsp >= policy.no_speech_threshold)
    logprobs = ev.get("avg_logprobs") or []
    speaks = said_something and not hallucinated
    if speaks:
        confidence = _pool_confidence(logprobs, policy.asr_confidence_pooling)
    elif hallucinated:
        # What this voter is confident about here is the silence head that overruled the words.
        confidence = nsp
    else:
        # It placed no words in this bucket. There is no scored quantity for that, and inventing
        # one from the words it placed elsewhere would be a magnitude nobody measured.
        confidence = None
    out: dict[str, Any] = {
        "speaks": speaks,
        "native_confidence": confidence,
        "hallucinated": hallucinated,
        "confidence_pooling": policy.asr_confidence_pooling,
    }
    if nsp is not None:
        out["no_speech_prob"] = nsp
    # The raw mean log-probability as well as the confidence: they are different scales, and only
    # the log-domain mean can be pooled further without bias.
    avg_lp = _mean(logprobs)
    if avg_lp is not None:
        out["avg_logprob"] = avg_lp
    return out


def _link_no_speech_prob(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Whisper's own silence head, inverted here rather than at L1."""
    nsp = _finite(ev.get("no_speech_prob"))
    if nsp is None:
        return None
    speaks = nsp < policy.no_speech_threshold
    # Directed, like every other rule here: at nsp = 0.9 the voter is 0.9-confident of *absence*,
    # not 0.1-confident of it. Reporting `1 - nsp` regardless of direction made the aggregator read
    # a confident denial as a confident assertion.
    return {"speaks": speaks, "native_confidence": (1.0 - nsp) if speaks else nsp}


def _link_label_mass(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Share of a scene classifier's score mass on the speech labels → direction + confidence."""
    mass = _finite(ev.get("speech_label_mass"))
    if mass is None:
        return None
    speaks = mass >= policy.label_mass_threshold
    return {"speaks": speaks, "native_confidence": mass if speaks else 1.0 - mass}


def _link_frame(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Bucket-mean frame posterior → claim. The mean itself was L1's; the cut is policy.

    Directed against the policy cut rather than against ``_directed``'s fixed 0.5, so a moved
    threshold moves the direction and the confidence together. Reporting the mean raw made a
    posterior of 0.02 read as a 0.02-confident *no*, which ``_weighted_p_voice`` maps to
    ``p = 0.98`` — the exact inverse of what the detector measured, on the voters the presence axis
    trusts most.
    """
    mean = _finite(ev.get("frame_mean"))
    if mean is None:
        return None
    speaks = mean >= policy.frame_speech_threshold
    return {"speaks": speaks, "native_confidence": mean if speaks else 1.0 - mean}


# ``_link_hnr`` lived here. **HNR is voicing evidence, but its ramp was never fitted, so on ordinary
# speech it read as a floor rather than as a measurement.**
#
# The ramp was a code literal — 2 dB to 10 dB — and the register called it "fixed" for that reason.
# Measured on `english_conversation_higgs_audio_v2`, median HNR is **8.12 dB**: *below* the anchor that
# means "confidently voiced", so ordinary conversational speech read as only partly voiced. 102 of its
# 145 votes fell in the graded region, and it ended up the **largest contributor** on the presence axis
# (mean doubt 0.1568, against `ast` 0.1094 and everything else at or below 0.05) while all four
# diarizers, all three recognizers and the brouhaha VAD read exactly 0.0000. Removing it takes presence
# doubt from 0.0250 to 0.0160, and buckets below 0.01 from 66 of 214 to 112.
#
# Making the abstention silent (see :func:`_abstaining_ramp`) was a real fix and not this one: it cut
# HNR's mean from 0.2675 to 0.1568 by removing the fabricated half-confident votes, but the remaining
# doubt is the graded region, which is the anchors' fault and not the abstention's.
#
# **The measurement is not lost.** ``harvest_speech_presence_evidence`` still emits ``hnr_db`` in
# decibels and ``votes._signal_rows_from_buckets`` records it from the *evidence*, not from the votes —
# so ``L1/signals/acoustic_hnr.parquet`` is unchanged and a consumer wanting voicing evidence reads the
# dB directly. What is gone is the unfitted dB→probability step in between.
#
# Reinstating it as a voter means fitting the anchors to measured HNR on known-voiced speech and
# writing them into ``data/`` with their derivation, the way ``detection_margin`` and the scene-quality
# profile are. Recorded as the open item in ``l1-post-processing-register.md`` item 10.


def _link_lufs(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Absolute loudness → presence. This one *may* claim absence: −90 LUFS is unambiguous."""
    lufs = _finite(ev.get("lufs"))
    if lufs is None:
        return None
    speaks, confidence = _directed(_ramp(lufs, policy.lufs_silence, policy.lufs_speech))
    return {"speaks": speaks, "native_confidence": confidence}


def _link_excess(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Excess in dB over the measured floor → activity, abstaining low (see module docstring)."""
    excess = _finite(ev.get("excess_db"))
    if excess is None:
        return None
    # Same construct, same correction: no excess over the measured floor is not evidence of speech at
    # half confidence, it is no evidence. Fixed together because it is one function and one argument —
    # leaving the sibling emitting a fabricated 0.5 would make the two disagree about what an
    # abstention is. Its contribution was smaller (mean 0.0858, median 0.0000) but the same in kind.
    p_voice = _abstaining_ramp(excess, 0.0, policy.speech_excess_db)
    if p_voice is None:
        return None
    speaks, confidence = _directed(p_voice)
    return {"speaks": speaks, "native_confidence": confidence}


def derive_window_clusters(
    per_window_embeddings: Mapping[str, Sequence[Any]] | None,
) -> dict[str, Any] | None:
    """Cluster one pass's windowed speaker embeddings — an L2 derivation over L1 vectors.

    L1 measures embedding *vectors*; whether they form coherent speaker clusters is a conclusion
    about the pass, needs every window at once, and is exactly the kind of judgement that belongs
    here (D-7). It used to run inside the harvester, which meant the only thing surviving to L2 was
    a thresholded ``speaks``.

    The full clustering result is returned rather than only the per-window voice score, because the
    same computation answers a second question: ``labels`` assigns each window to a cluster, which
    is what a later stage needs to assign and re-assign speaker labels. Recomputing it there would
    risk two stages disagreeing about the clustering they are each reasoning over.

    Args:
        per_window_embeddings: ``{embedding_model → [WindowEmbedding]}``.

    Returns:
        ``{"model", "windows", "clusters"}`` for the first embedding model (alphabetically) with
        usable windows, or ``None`` when none produced a clustering. Alphabetical rather than
        merged: running every model would be a vote of voters inside one signal class, which is the
        aggregator's job, not this function's.
    """
    if not per_window_embeddings:
        return None
    from senselab.audio.workflows.audio_analysis.embeddings import cluster_pass_speakers

    for model in sorted(per_window_embeddings):
        entries = list(per_window_embeddings.get(model) or [])
        if not entries:
            continue
        clusters = cluster_pass_speakers(entries)
        if clusters is not None:
            return {"model": str(model), "windows": entries, "clusters": clusters}
    return None


# ``_silhouette_votes_by_bucket`` lived here. **A cluster silhouette is not presence evidence.**
#
# It answered "does a coherent speaker sit here" by reading the silhouette coefficient as a
# confidence — ``_directed(score)`` with no ramp and no anchors, unlike every linker above it
# (``_link_hnr`` ramps between ``policy.hnr_low_db`` and ``hnr_high_db``, ``_link_level_above_floor``
# likewise). Three things were wrong with it, all measured on a clean two-speaker conversation:
#
# - **It answers a different question.** Silhouette measures cluster *geometry* — separation — over
#   every window including silent ones. Silence embeds consistently too, so well-separated silence
#   scores well. It cannot distinguish "a coherent speaker is here" from "this window is coherently
#   not speech".
# - **It carried almost no information.** 214 buckets, doubt spanning 0.4022-0.4996, stdev 0.0227 —
#   a constant to within ±0.05. An ordinary good silhouette of 0.58 became 0.42 of standing doubt.
# - **And the weighting rewarded it for that.** It held weight **1.0**, the highest of all fifteen
#   presence signals, while every informative voter sat at 0.78-0.91: ``reliability.signal_stability``
#   measures cross-pass ``|delta|``, and a near-constant is perfectly stable. The least informative
#   voter earned the most weight.
#
# Together those meant **no bucket could reach zero presence doubt** however unanimous the evidence:
# all four diarizers, all three recognizers and the brouhaha VAD read exactly 0.0000 while the axis
# reported 0.0682. Without this voter it reads 0.0385, and 47 of 214 buckets can reach zero.
#
# Nothing is lost. The clustering this scored already reaches the **speaker** axis as a first-class
# diarization source per D-20 — ``compute.harvest_pass`` injects a synthetic
# ``embedding_silhouette/<model>`` diarizer built from :func:`derive_window_clusters`, whose spans and
# cluster ids feed ``attribution.speaker_assignment_doubt`` directly. Asking the same clustering to
# also vote on presence counted one body of evidence twice, on the axis where it was least apt.
# The vote's ``cluster_id`` had no consumer: label reassignment reads the synthetic diarizer's spans.
#
# ``derive_window_clusters`` below stays — it is what ``compute.harvest_pass`` calls.
# Register: ``l1-post-processing-register.md`` item 12.


_SUFFIX_RULES = (("::no_speech_prob", _link_no_speech_prob),)

_EXACT_RULES = {
    "ast": _link_label_mass,
    "yamnet": _link_label_mass,
    "acoustic_lufs": _link_lufs,
    "acoustic_level_above_floor": _link_excess,
}

_FRAME_PREFIX = "frame_"


def _rule_for(name: str, ev: Mapping[str, Any]) -> Any:  # noqa: ANN401 — heterogeneous rule callables
    """Pick the link rule for a signal.

    Resolved by name for the fixed voters and **structurally** for the model-named ones: a
    diarizer is called ``pyannote/speaker-diarization-3.1`` and an ASR model
    ``openai/whisper-large-v3``, so no name pattern distinguishes them. What does distinguish them
    is which measurement they reported, which is exactly the right thing to key on — a signal is
    linked according to what it measured, not according to what it is called.
    """
    for suffix, rule in _SUFFIX_RULES:
        if name.endswith(suffix):
            return rule
    if name in _EXACT_RULES:
        return _EXACT_RULES[name]
    if name.startswith(_FRAME_PREFIX) or "frame_mean" in ev:
        return _link_frame
    if "covered_fraction" in ev:
        return _link_diar
    if "word_overlap_s" in ev:
        return _link_asr
    return None


def _is_coarse(ev: Mapping[str, Any], reporting_win_s: float | None, policy: SpeechPresencePolicy) -> bool:
    """Is this voter's native window wide enough that one value spans many reporting buckets?

    Read from the declared ``native_window_s`` rather than from a hand-set flag, so adding a voter
    cannot forget to mark itself and changing the grid cannot leave a stale marking behind.
    """
    if not reporting_win_s or reporting_win_s <= 0:
        return False
    native = _finite(ev.get("native_window_s"))
    if native is None or native <= 0:
        return False
    return native > policy.coarse_window_ratio * reporting_win_s


def link_speech_presence(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: SpeechPresencePolicy = DEFAULT_POLICY,
    reporting_win_s: float | None = None,
    per_window_embeddings: Mapping[str, Sequence[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Turn L1 speech-presence evidence rows into vote rows the aggregator can fold.

    Args:
        rows: Per-bucket ``{"start", "end", "evidence", "frame_dispersion"}`` dicts from
            :func:`~.speech_presence.harvest_speech_presence_evidence`.
        policy: The interpretations to apply. Defaults to the documented anchors.
        reporting_win_s: Width of the reporting bucket, needed to decide which voters are coarse
            relative to it. Omit to skip demotion entirely.
        per_window_embeddings: L1 embedding vectors per window. When given, the
            ``embedding_silhouette`` signal is *derived here* (see :func:`derive_window_clusters`)
            and appears only in the votes — it is a conclusion about the pass, never an L1
            measurement of a bucket. Omit and the signal is simply absent.

    Returns:
        Per-bucket ``{"start", "end", "votes", "frame_dispersion"}`` dicts, where each vote carries
        the belief fields **merged over** the measurements it was derived from. Nothing is dropped:
        a consumer can always recover what was measured, which is what makes a verdict auditable
        rather than merely recorded.

    Pure — the input rows are not mutated, so the adaptive loop can re-link the same harvest under
    a different policy each round and get the same answer every time.
    """
    out: list[dict[str, Any]] = []
    for row_idx, row in enumerate(rows or []):
        evidence = row.get("evidence") or {}
        votes: dict[str, dict[str, Any]] = {}
        for name, ev in evidence.items():
            if not isinstance(ev, Mapping):
                continue
            rule = _rule_for(str(name), ev)
            if rule is None:
                continue
            belief = rule(ev, policy)
            if belief is None:
                # The signal reported nothing usable in this bucket. Dropping the vote is right:
                # a fabricated 0.5 would be indistinguishable from a real abstention.
                continue
            vote = {**dict(ev), **belief}
            if _is_coarse(ev, reporting_win_s, policy):
                vote["weight"] = policy.coarse_voter_weight
            votes[str(name)] = vote
        out.append(
            {
                "start": row.get("start"),
                "end": row.get("end"),
                "votes": votes,
                # Dispersion stays in probability units here too. Mapping it into a doubt is a
                # separate modelling choice and lives with the aggregation in ``votes.py``.
                "frame_dispersion": row.get("frame_dispersion"),
            }
        )
    return out


def votes_for_harvest(
    harvest: Any,  # noqa: ANN401 — PassHarvest, duck-typed to keep this module import-free
    *,
    policy: SpeechPresencePolicy = DEFAULT_POLICY,
) -> list[dict[str, Any]]:
    """Link one pass's speech-presence evidence at the grid it was harvested on.

    The single entry point every consumer of speech-presence *beliefs* should use, so the reporting
    grid is always read from the harvest that produced the measurements rather than passed in
    alongside it. Getting those two out of step is how a coarse voter ends up demoted against the
    wrong bucket width.
    """
    grids = getattr(harvest, "grids", None) or {}
    return link_speech_presence(
        getattr(harvest, "speech_presence_evidence", None) or [],
        policy=policy,
        reporting_win_s=(grids.get("speech_presence") or {}).get("win_length"),
        per_window_embeddings=getattr(harvest, "per_window_embeddings", None),
    )
