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
    "link_speech_presence",
    "policy_from_params",
    "votes_for_harvest",
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
        hnr_low_db: HNR at or below which the voter abstains (``0.5``).
        hnr_high_db: HNR at or above which the voter is fully confident of voicing. Typical
            conversational HNR is 8–14 dB.
        speech_excess_db: dB above the measured noise floor at which a bucket reads as clearly
            active. Speech usually sits 12–20 dB above a room's floor.
        lufs_silence: LUFS at or below which the loudness voter is confident of absence.
        lufs_speech: LUFS at or above which it is confident of presence.
        silhouette_threshold: Silhouette coefficient above which an embedding window is taken to
            sit inside a coherent speaker cluster.
        coarse_voter_weight: Weight given to a voter whose native window is much wider than the
            reporting bucket, so one value repeated across many buckets cannot dominate the fold.
        coarse_window_ratio: How many times wider than the reporting bucket a voter's native
            window must be before it is demoted. At ``2.0`` a voter is only demoted once its window
            spans more than two reporting buckets.
    """

    diar_coverage_threshold: float = 0.0
    word_overlap_threshold_s: float = 0.0
    no_speech_threshold: float = 0.5
    asr_confidence_pooling: AsrConfidencePooling = "mean_of_exp"
    label_mass_threshold: float = 0.5
    frame_speech_threshold: float = 0.5
    hnr_low_db: float = 2.0
    hnr_high_db: float = 10.0
    speech_excess_db: float = 12.0
    lufs_silence: float = -60.0
    lufs_speech: float = -30.0
    silhouette_threshold: float = 0.5
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


def _abstaining_ramp(value: float, low: float, high: float) -> float:
    """Linear ramp into ``[0.5, 1]``: the low end is uninformative, not a denial.

    Used where a low reading has two indistinguishable causes (see the module docstring on HNR and
    level-above-floor). Mapping that end to ``0.0`` would let the signal contradict correct models
    on inputs where it simply cannot tell.
    """
    return 0.5 + 0.5 * _ramp(value, low, high)


def _directed(p_voice: float) -> tuple[bool, float]:
    """``P(speech)`` → ``(speaks, confidence in that direction)``.

    The aggregator reads ``native_confidence`` as the voter's confidence in *its own* ``speaks``
    direction, so a 0.2 probability of speech is an 0.8-confident *no* rather than a weak *yes*.
    """
    speaks = p_voice >= 0.5
    return speaks, p_voice if speaks else 1.0 - p_voice


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
    """Word coverage → claim, per-chunk logprobs → confidence, ``no_speech_prob`` → gate."""
    overlap = _finite(ev.get("word_overlap_s")) or 0.0
    said_something = overlap > policy.word_overlap_threshold_s
    nsp = _mean(ev.get("no_speech_probs") or [])
    hallucinated = bool(said_something and nsp is not None and nsp >= policy.no_speech_threshold)
    logprobs = ev.get("avg_logprobs") or []
    out: dict[str, Any] = {
        "speaks": said_something and not hallucinated,
        "native_confidence": _pool_confidence(logprobs, policy.asr_confidence_pooling),
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
    return {"speaks": nsp < policy.no_speech_threshold, "native_confidence": 1.0 - nsp}


def _link_label_mass(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Share of a scene classifier's score mass on the speech labels → direction + confidence."""
    mass = _finite(ev.get("speech_label_mass"))
    if mass is None:
        return None
    speaks = mass >= policy.label_mass_threshold
    return {"speaks": speaks, "native_confidence": mass if speaks else 1.0 - mass}


def _link_frame(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Bucket-mean frame posterior → claim. The mean itself was L1's; the cut is policy."""
    mean = _finite(ev.get("frame_mean"))
    if mean is None:
        return None
    return {"speaks": mean >= policy.frame_speech_threshold, "native_confidence": mean}


def _link_hnr(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Harmonics-to-noise ratio → voicing, abstaining at the low end (see module docstring)."""
    hnr = _finite(ev.get("hnr_db"))
    if hnr is None:
        return None
    speaks, confidence = _directed(_abstaining_ramp(hnr, policy.hnr_low_db, policy.hnr_high_db))
    return {"speaks": speaks, "native_confidence": confidence}


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
    speaks, confidence = _directed(_abstaining_ramp(excess, 0.0, policy.speech_excess_db))
    return {"speaks": speaks, "native_confidence": confidence}


def _link_ppg(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Mean PPG posterior on ``<silent>`` → voicing, by complement.

    ``1 − P(silent)`` rather than a count of non-silent argmax frames: the count reduces every
    frame to a hard verdict, so a bucket the model was 60% sure about becomes indistinguishable
    from one it was certain about (register item 11, the same defect as the scene-classifier
    top-1).
    """
    silence = _finite(ev.get("mean_silence_posterior"))
    if silence is None:
        return None
    speaks, confidence = _directed(1.0 - max(0.0, min(1.0, silence)))
    return {"speaks": speaks, "native_confidence": confidence}


def _link_silhouette(ev: Mapping[str, Any], policy: SpeechPresencePolicy) -> dict[str, Any] | None:
    """Cluster silhouette → does a coherent speaker sit here."""
    score = _finite(ev.get("silhouette"))
    if score is None:
        return None
    speaks, confidence = _directed(score)
    return {"speaks": speaks, "native_confidence": confidence}


_SUFFIX_RULES = (("::no_speech_prob", _link_no_speech_prob),)

_EXACT_RULES = {
    "ast": _link_label_mass,
    "yamnet": _link_label_mass,
    "acoustic_hnr": _link_hnr,
    "acoustic_lufs": _link_lufs,
    "acoustic_level_above_floor": _link_excess,
    "ppg_voice_fraction": _link_ppg,
    "embedding_silhouette": _link_silhouette,
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
) -> list[dict[str, Any]]:
    """Turn L1 speech-presence evidence rows into vote rows the aggregator can fold.

    Args:
        rows: Per-bucket ``{"start", "end", "evidence", "frame_dispersion"}`` dicts from
            :func:`~.speech_presence.harvest_speech_presence_evidence`.
        policy: The interpretations to apply. Defaults to the documented anchors.
        reporting_win_s: Width of the reporting bucket, needed to decide which voters are coarse
            relative to it. Omit to skip demotion entirely.

    Returns:
        Per-bucket ``{"start", "end", "votes", "frame_dispersion"}`` dicts, where each vote carries
        the belief fields **merged over** the measurements it was derived from. Nothing is dropped:
        a consumer can always recover what was measured, which is what makes a verdict auditable
        rather than merely recorded.

    Pure — the input rows are not mutated, so the adaptive loop can re-link the same harvest under
    a different policy each round and get the same answer every time.
    """
    out: list[dict[str, Any]] = []
    for row in rows or []:
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
    )
