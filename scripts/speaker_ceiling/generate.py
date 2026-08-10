"""Build a synthetic multi-speaker corpus with exact-by-construction ground truth.

**Supersedes the original brief.** The brief's Task 2 pointed at NeMo's
``MultiSpeakerSimulator``, but that class does not synthesize speech: its own docstring says
it "simulates multispeaker audio sessions using *single-speaker audio files and
corresponding word alignments*" — it composes real recordings sampled from
``data_simulator.manifest_filepath``, with ``_min_alignment_count = 2`` rejecting any
manifest without alignments (confirmed against the installed ``nemo-diarization`` venv,
job 20080436, and against the upstream source directly). Running it would have meant
sourcing an aligned single-speaker corpus (e.g. LibriSpeech-plus-alignments) that this
effort never costed, and deriving ground truth from someone else's forced alignment folds
their alignment error into a probe whose entire point is measuring *our* diarizers.

So this module generates the speech itself, with senselab's Qwen3-TTS backend
(:mod:`senselab.audio.tasks.text_to_speech.qwen_tts`), and composes sessions directly —
borrowing NeMo's *session model* (turn-taking, dominance, sentence-length, overlap) rather
than its simulator. See ``specs/20260809-112417-speaker-ceiling-probe/plan.md``'s "Task 2:
Corpus generation" section for the full reasoning; this module is that section's
implementation.

Why constructive ground truth, not merely cheaper
--------------------------------------------------
Placing a synthesized utterance at a chosen timeline offset makes the RTTM **exact by
construction**: there is no alignment step, no VAD, no energy threshold, no forced-alignment
call whose error could later be mistaken for a diarizer miscounting. Every ``.rttm`` line's
start and duration comes from the real synthesized waveform this module just placed — never
an estimate of where speech was detected to begin or end.

Voice pool
----------
Qwen3-TTS-12Hz-1.7B-CustomVoice exposes 9 named speaker identities reachable directly by
name, no reference-audio cloning required (verified via
:func:`~senselab.audio.tasks.text_to_speech.qwen_tts.supported_speakers` against the pinned
checkpoint's ``config.json``: aiden, dylan, eric, ono_anna, ryan, serena, sohee, uncle_fu,
vivian). That covers the probe's *k* = 1…8 sweep with one identity to spare. Each session
draws exactly *k* distinct voices from this pool (without replacement) — a *k* above the
pool size is refused outright rather than silently cloning a 10th identity from reference
audio, which this generator does not do.

The session model, borrowed from NeMo's ``MultiSpeakerSimulator``
--------------------------------------------------------------------
The parameters below are not this module's invention — they are NeMo's own documented
knobs for what makes a simulated session behave like a conversation rather than a
round-robin (turn-taking, unequal speaking time, variable utterance length, overlap). Their
*values* are copied from NeMo's own shipped default config,
``tools/speech_data_simulator/conf/data_simulator.yaml`` (read directly from the installed
``nemo-diarization`` venv, not from memory), because a round-robin at equal dominance makes
speaker counting far easier than real conversation and would inflate every ceiling this
probe produces:

- ``TURN_PROB`` (0.875): probability of switching speakers after each utterance. NeMo's
  ``_get_next_speaker`` reads this as "probability of switching", not "probability of
  staying" — kept identical here, including the comparison direction, since getting this
  backwards would silently produce a corpus of near-monologues.
- ``DOMINANCE_VAR`` (0.11) / ``MIN_DOMINANCE`` (0.05): each speaker's share of speaking time
  is drawn from a normal distribution centered on ``1/k``, floored at ``MIN_DOMINANCE``, and
  renormalized. NeMo's own ``_init_dominance`` passes ``dominance_var`` directly as the
  *standard deviation* argument to ``np.random.normal(scale=...)``, not as a variance
  despite the name — verified by reading that function, not assumed from the name — and
  this module reproduces that quirk rather than "fixing" it, to keep the borrowed constant
  meaning what NeMo's default was tuned against.
- ``SENTENCE_LENGTH_PARAMS`` (0.4, 0.05): ``(n, p)`` for
  ``rng.negative_binomial(n, p) + 1``, giving each utterance's target word count (mean
  ≈ 9.6 words). Copied verbatim from ``_build_sentence``'s call, including the ``+ 1`` floor
  that guarantees at least one word.
- ``MEAN_OVERLAP`` (0.10) / ``MEAN_SILENCE`` (0.15) / ``MEAN_SILENCE_VAR`` (0.01): NeMo's
  documented mean proportion of overlapping speech and of silence relative to speaking time.

What is *not* borrowed, because NeMo splices real audio at the sample level with windowed
crossfades and this module places whole synthesized utterances instead: NeMo's exact
per-segment silence/overlap sampling (``per_silence_var``, gamma-jittered, independently
clipped) is replaced with a simpler proportional model in :func:`_lay_out_session` — a
Bernoulli draw against ``MEAN_OVERLAP`` per turn boundary, then either an overlap fraction
of the previous utterance's real duration or a silence gap proportional to it. This is a
simplification, not a re-implementation, and is named as one rather than presented as
faithful to NeMo's algorithm.

``SESSION_LENGTH_SECONDS`` (45.0) and ``ASSUMED_WORDS_PER_SECOND`` (2.5) are **this
module's own judgement**, not measured and not NeMo's — NeMo's own default
(``session_length: 600``) targets a ten-minute recording, which at Qwen3-TTS's measured
RTF ~6.3 (H100, see the ``qwen_tts`` module docstring) would cost roughly an hour of GPU
time *per session*; 45 s keeps a 160-session sweep (8 counts × 20 sessions) tractable.
Critically, this budget only decides *how many turns get planned* — every turn's actual
placed duration always comes from the real synthesized waveform, never from this estimate,
so an inaccurate words-per-second assumption cannot corrupt the ground truth it under- or
over-shoots.

``enforce_num_speakers`` is not a tunable knob here: it is always true, by construction,
not by NeMo's probabilistic late-session enforcement window (``speaker_enforcement``,
triggered somewhere between 25% and 75% through the session). :func:`_plan_session` assigns
the first *k* turns to a shuffled permutation of every requested speaker *before* the
probabilistic turn model runs at all, so a session can never end up short a speaker for
:func:`generate_corpus` to discover after the fact — and :func:`generate_corpus` asserts
this against the RTTM it just wrote, not against its own in-memory plan, since the file on
disk is what the evaluation script will actually read.

The caveat that ships with every number this probe produces
---------------------------------------------------------------
A ceiling measured on this corpus is a ceiling on *clean, synthetically distinct voices*:
no room acoustics, no channel variation, and vocoder characteristics shared across every
speaker in a session. That plausibly makes counting easier than real speech (more
separable identities) and could make it harder (shared synthesis artifacts). Either way the
measured value is an upper bound on well-conditioned audio, not a guarantee about a real
recording. ``manifest.json`` records the generation method beside every session precisely
so a reader of the resulting profile can judge that for themselves.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.text_to_speech.qwen_tts import (
    _QWEN_TTS_DEFAULT_MODEL,
    supported_speakers,
    synthesize_texts_with_qwen,
)
from senselab.utils.data_structures import DeviceType, HFModel

# --------------------------------------------------------------------------------------
# Session model constants, borrowed from NeMo's MultiSpeakerSimulator defaults.
# See the module docstring's "session model" section for what each one means, where its
# value comes from, and which are ours rather than NeMo's.
# --------------------------------------------------------------------------------------
TURN_PROB = 0.875
DOMINANCE_VAR = 0.11
MIN_DOMINANCE = 0.05
SENTENCE_LENGTH_PARAMS: Tuple[float, float] = (0.4, 0.05)
MEAN_OVERLAP = 0.10
MEAN_SILENCE = 0.15
MEAN_SILENCE_VAR = 0.01

# Ours, not NeMo's -- judgements, recorded in the manifest rather than left as bare
# literals. See the module docstring for why 45 s and 2.5 words/s were chosen.
SESSION_LENGTH_SECONDS = 45.0
ASSUMED_WORDS_PER_SECOND = 2.5

# The fraction of the *previous* utterance's real duration that an overlapping turn
# starts inside, when the MEAN_OVERLAP draw fires. Ours -- NeMo instead samples an
# absolute overlap duration per segment from a separately-parameterized gamma
# distribution (`per_overlap_var`); a proportion of the real duration is simpler and
# cannot exceed the utterance it overlaps into.
OVERLAP_FRACTION_RANGE: Tuple[float, float] = (0.05, 0.4)

# Safety valve, not a session-model parameter: stops `_plan_session` from looping forever
# if a degenerate word-count draw kept the budget from ever being met. Never expected to
# bind at SESSION_LENGTH_SECONDS=45 with SENTENCE_LENGTH_PARAMS' ~9.6-word mean utterance.
_MAX_TURNS_PER_SESSION = 400

# A small bank of generic, punctuation-light filler sentences spanning a range of word
# counts, so `_pick_sentence` has something to select from at whatever target length
# `_next_utterance`'s negative-binomial draw produces. Content is deliberately neutral
# (no names, no claims) since only *identity* and *timing* are under test here, never
# transcript content.
_SENTENCE_BANK: Tuple[str, ...] = (
    "Okay.",
    "Sure, go ahead.",
    "I think so, yes.",
    "That makes sense to me.",
    "Can you say that again please.",
    "I was about to mention the same thing.",
    "Let us come back to that in a minute.",
    "I am not entirely sure that is right.",
    "We should probably check the schedule again before deciding.",
    "That is a fair point, but I still have some doubts about it.",
    "I would rather wait until we hear back from the rest of the team.",
    "Honestly, I think the plan we discussed last week still makes the most sense.",
    "It took a while, but I finally found the file we were looking for yesterday.",
    "I understand the concern, and I think there is a reasonable way to address it.",
    "Before we move on, I want to make sure everyone agrees with the current approach.",
    "There were a few details in the report that I think are worth revisiting together.",
    "I spent most of the morning going through the notes from our previous conversation.",
    "If it is alright with everyone, I would like to spend a bit more time on this topic.",
    "We looked at several options before settling on the one that seemed most practical overall.",
    "I know this has come up before, but I think it is worth raising again given what changed.",
)


@dataclass(frozen=True)
class _SessionTurn:
    """One placed utterance in a session's timeline.

    ``speaker_idx`` indexes into that session's own speaker-name list (0-based, local to
    the session) rather than a global speaker id -- turning it into the RTTM's speaker
    label happens once, in :func:`_write_rttm`.
    """

    speaker_idx: int
    text: str
    start: float
    end: float


def _pick_sentence(rng: np.random.Generator, target_words: int) -> str:
    """Return the bank sentence whose word count is closest to ``target_words``.

    Ties are broken by the caller's own rng draw rather than always taking the first
    match, so two equally-close sentences do not silently favor whichever was listed
    first in ``_SENTENCE_BANK``.
    """
    counts = [len(s.split()) for s in _SENTENCE_BANK]
    best_distance = min(abs(c - target_words) for c in counts)
    candidates = [s for s, c in zip(_SENTENCE_BANK, counts) if abs(c - target_words) == best_distance]
    return str(rng.choice(candidates))


def _next_utterance(rng: np.random.Generator) -> Tuple[str, int]:
    """Sample one utterance's text and word count from ``SENTENCE_LENGTH_PARAMS``.

    Mirrors NeMo's ``_build_sentence`` sentence-length draw exactly (negative binomial
    plus one, guaranteeing at least one word) but selects a bank sentence near that length
    instead of splicing real audio up to a sample budget, since this module has no audio
    to splice yet -- synthesis happens after every session's turns are planned.
    """
    n, p = SENTENCE_LENGTH_PARAMS
    target_words = int(rng.negative_binomial(n, p)) + 1
    text = _pick_sentence(rng, target_words)
    return text, len(text.split())


def _dominance_cdf(rng: np.random.Generator, num_speakers: int) -> np.ndarray:
    """Return a cumulative distribution over ``num_speakers`` speaking-time shares.

    Reproduces NeMo's ``_init_dominance``: draw raw shares from a normal distribution
    centered on ``1/num_speakers`` (using ``DOMINANCE_VAR`` directly as the standard
    deviation, matching NeMo's own call despite the "var" name -- see the module
    docstring), clip negatives to zero, then rescale the remainder so every speaker's
    final share is at least ``MIN_DOMINANCE``. The result is cumulative (not per-speaker)
    because :func:`_next_speaker` samples from it by inverse-CDF lookup, exactly as
    NeMo's ``_get_next_speaker`` does.
    """
    if num_speakers == 1:
        return np.array([1.0])

    mean = 1.0 / num_speakers
    raw = rng.normal(loc=mean, scale=DOMINANCE_VAR, size=num_speakers)
    raw = np.clip(raw, a_min=0.0, a_max=None)
    total = raw.sum()
    if total <= 0:
        # Every draw landed negative -- vanishingly unlikely at DOMINANCE_VAR=0.11, but
        # falling back to equal shares beats propagating a NaN from a zero-division.
        raw = np.full(num_speakers, mean)
        total = raw.sum()

    remaining_budget = 1.0 - MIN_DOMINANCE * num_speakers
    shares = raw / total * remaining_budget + MIN_DOMINANCE
    return np.cumsum(shares)


def _next_speaker(
    rng: np.random.Generator,
    num_speakers: int,
    prev_speaker: Optional[int],
    dominance_cdf: np.ndarray,
) -> int:
    """Pick the next speaker, following NeMo's ``_get_next_speaker`` turn-taking rule.

    With probability ``1 - TURN_PROB`` the previous speaker continues (a multi-utterance
    turn); otherwise a new speaker is drawn from ``dominance_cdf`` by inverse-CDF lookup,
    re-drawing until it differs from ``prev_speaker`` -- a switch that lands back on the
    same speaker is not a switch.
    """
    if num_speakers == 1:
        return 0
    if prev_speaker is not None and rng.random() > TURN_PROB:
        return prev_speaker

    while True:
        draw = rng.random()
        candidate = min(int(np.searchsorted(dominance_cdf, draw, side="right")), num_speakers - 1)
        if candidate != prev_speaker:
            return candidate


def _plan_session(rng: np.random.Generator, num_speakers: int) -> List[Tuple[int, str]]:
    """Plan a session's turns as ``(speaker_idx, text)`` pairs, before any synthesis.

    Guarantees every one of ``num_speakers`` appears at least once: the first
    ``num_speakers`` turns are a shuffled permutation covering every speaker exactly once,
    *before* the probabilistic turn-taking model in :func:`_next_speaker` runs at all. That
    ordering is what makes ``enforce_num_speakers`` true by construction rather than by a
    probabilistic near-certainty -- there is no draw here that could produce a session
    missing a speaker for :func:`generate_corpus` to discover after the fact.

    Continues past that guarantee with the probabilistic model until the planned word
    count reaches ``SESSION_LENGTH_SECONDS * ASSUMED_WORDS_PER_SECOND``. That budget only
    controls how many turns get planned; every placed duration later comes from the real
    synthesized waveform, so an inaccurate words-per-second assumption cannot corrupt the
    ground truth it targets too loosely.
    """
    dominance_cdf = _dominance_cdf(rng, num_speakers)
    budget_words = SESSION_LENGTH_SECONDS * ASSUMED_WORDS_PER_SECOND

    plan: List[Tuple[int, str]] = []
    total_words = 0.0

    order = rng.permutation(num_speakers)
    prev_speaker: Optional[int] = None
    for speaker_idx in order:
        text, words = _next_utterance(rng)
        plan.append((int(speaker_idx), text))
        total_words += words
        prev_speaker = int(speaker_idx)

    while total_words < budget_words and len(plan) < _MAX_TURNS_PER_SESSION:
        speaker_idx = _next_speaker(rng, num_speakers, prev_speaker, dominance_cdf)
        text, words = _next_utterance(rng)
        plan.append((speaker_idx, text))
        total_words += words
        prev_speaker = speaker_idx

    return plan


def _lay_out_session(
    plan: Sequence[Tuple[int, str]],
    durations: Sequence[float],
    rng: np.random.Generator,
) -> List[_SessionTurn]:
    """Place planned turns on a timeline using each turn's *real* synthesized duration.

    Simplified relative to NeMo's per-segment silence/overlap sampling (see the module
    docstring): each turn boundary is a Bernoulli draw against ``MEAN_OVERLAP``. On a hit,
    the new turn starts inside the tail of the previous one, by a fraction of the
    previous turn's real duration drawn from ``OVERLAP_FRACTION_RANGE`` -- so an overlap
    can never exceed the utterance it overlaps into. Otherwise a silence gap proportional
    to the previous duration is inserted, sized around ``MEAN_SILENCE``.
    """
    turns: List[_SessionTurn] = []
    cursor = 0.0
    prev_duration = 0.0

    for (speaker_idx, text), duration in zip(plan, durations):
        if not turns:
            start = 0.0
        elif rng.random() < MEAN_OVERLAP:
            overlap_fraction = rng.uniform(*OVERLAP_FRACTION_RANGE)
            start = max(0.0, cursor - overlap_fraction * prev_duration)
        else:
            silence = max(0.0, rng.normal(MEAN_SILENCE, math.sqrt(MEAN_SILENCE_VAR))) * prev_duration
            start = cursor + silence

        end = start + duration
        turns.append(_SessionTurn(speaker_idx=speaker_idx, text=text, start=start, end=end))
        cursor = max(cursor, end)
        prev_duration = duration

    return turns


def _compose_waveform(turns: Sequence[_SessionTurn], waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
    """Overlay-add every turn's real waveform onto one buffer at its placed offset.

    Addition (not replacement) is what makes an overlapping region actually contain both
    speakers' signal rather than whichever was placed last silently overwriting the other
    -- the RTTM already declares both speakers present there, so the audio must too.
    Peak-normalized afterward to avoid clipping from summed overlaps, not for loudness
    matching.
    """
    total_samples = int(math.ceil(max(t.end for t in turns) * sample_rate)) + 1
    buffer = np.zeros(total_samples, dtype=np.float64)

    for turn, wav in zip(turns, waveforms):
        start_sample = int(round(turn.start * sample_rate))
        end_sample = start_sample + len(wav)
        if end_sample > len(buffer):
            buffer = np.pad(buffer, (0, end_sample - len(buffer)))
        buffer[start_sample:end_sample] += wav

    peak = np.max(np.abs(buffer))
    if peak > 0:
        buffer = buffer / peak * 0.95
    return buffer.astype(np.float32)


def _write_rttm(path: Path, audio_id: str, turns: Sequence[_SessionTurn], speaker_names: Sequence[str]) -> None:
    """Write one RTTM line per turn, labeling each with its real TTS voice identity.

    The ground-truth speaker label is the voice name itself (e.g. ``"ryan"``), not an
    anonymized index -- the identity *is* known exactly, by construction, so there is no
    reason to hide it behind a placeholder the way a real annotation would.
    """
    lines = [
        f"SPEAKER {audio_id} 1 {turn.start:.3f} {turn.end - turn.start:.3f} <NA> <NA> "
        f"{speaker_names[turn.speaker_idx]} <NA> <NA>"
        for turn in turns
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def generate_corpus(
    out_dir: Path,
    counts: Sequence[int],
    sessions_per_count: int,
    seed: int,
    tts_model: Optional[HFModel] = None,
    device: Optional[DeviceType] = None,
) -> Path:
    """Generate a synthetic multi-speaker corpus with exact-by-construction RTTM ground truth.

    Writes ``out_dir/k=<k>/session_<i>.wav`` with a sibling ``session_<i>.rttm`` for every
    ``k`` in ``counts`` and every ``i`` in ``range(sessions_per_count)``, plus one
    ``out_dir/manifest.json`` recording the generation method, the resolved TTS model, the
    session-model parameters, and ``seed``.

    Reproducible from ``seed`` alone: each session's random draws come from
    ``numpy.random.default_rng(numpy.random.SeedSequence([seed, k, i]))``, so re-running
    with the same arguments and the same (mocked or real) synthesis backend reproduces
    identical plans, layouts, and RTTM files.

    Args:
        out_dir: Root directory for the corpus. Created if missing. Must not be under a
            package ``data/`` directory -- a blanket repo rule gitignores those silently.
        counts: Speaker counts (*k*) to generate sessions for. Every value must be at least
            1 and at most the TTS backend's named-voice pool size (9 for Qwen3-TTS
            CustomVoice) -- this generator assigns one distinct voice per speaker and does
            not clone additional identities from reference audio.
        sessions_per_count: Number of sessions to generate per count.
        seed: Top-level seed. Combined with ``k`` and the session index to seed each
            session independently (see above).
        tts_model: The TTS backend model. Defaults to Qwen3-TTS-12Hz-1.7B-CustomVoice.
        device: Device for TTS synthesis. Defaults to the backend's own auto-selection.

    Returns:
        ``out_dir``.

    Raises:
        ValueError: if any requested ``k`` is less than 1, or the largest requested ``k``
            exceeds the TTS backend's named-voice pool size.
        RuntimeError: if a session's written RTTM does not contain exactly ``k`` distinct
            speakers -- ``enforce_num_speakers`` is guaranteed by construction (see
            :func:`_plan_session`), so this signals a bug in this module, not a rare
            sampling outcome to retry past.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tts_model is None:
        tts_model = HFModel(path_or_uri=_QWEN_TTS_DEFAULT_MODEL)

    voice_pool = sorted(str(s) for s in supported_speakers(tts_model))

    if counts and max(counts) > len(voice_pool):
        raise ValueError(
            f"requested up to k={max(counts)} speakers, but TTS backend "
            f"{tts_model.path_or_uri!r} exposes only {len(voice_pool)} named voices "
            f"({voice_pool}). This generator assigns one distinct voice per speaker and "
            "does not clone additional identities from reference audio, so k cannot "
            "exceed the pool size."
        )
    for k in counts:
        if k < 1:
            raise ValueError(f"speaker count must be >= 1, got {k}")

    session_records: List[Dict[str, object]] = []

    for k in counts:
        k_dir = out_dir / f"k={k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        # Plan every session at this k before synthesizing anything: each session's rng
        # is kept alive across the synthesis call below so _lay_out_session can keep
        # drawing from the same stream afterward, and every session's utterances are
        # flattened into one synthesis call to amortize the subprocess venv's one-time
        # model load across `sessions_per_count` sessions instead of paying it per session.
        session_rngs: List[np.random.Generator] = []
        session_plans: List[List[Tuple[int, str]]] = []
        session_speaker_names: List[List[str]] = []

        for i in range(sessions_per_count):
            rng = np.random.default_rng(np.random.SeedSequence([seed, k, i]))
            speaker_names = [str(name) for name in rng.choice(voice_pool, size=k, replace=False)]
            plan = _plan_session(rng, k)
            session_rngs.append(rng)
            session_plans.append(plan)
            session_speaker_names.append(speaker_names)

        flat_texts: List[str] = []
        flat_speakers: List[str] = []
        session_spans: List[Tuple[int, int]] = []
        for plan, speaker_names in zip(session_plans, session_speaker_names):
            start = len(flat_texts)
            for speaker_idx, text in plan:
                flat_texts.append(text)
                flat_speakers.append(speaker_names[speaker_idx])
            session_spans.append((start, len(flat_texts)))

        audios = (
            synthesize_texts_with_qwen(texts=flat_texts, model=tts_model, speaker=flat_speakers, device=device)
            if flat_texts
            else []
        )
        sample_rate = audios[0].sampling_rate if audios else 24000
        flat_waveforms = [audio.waveform.squeeze(0).numpy() for audio in audios]

        for i in range(sessions_per_count):
            rng = session_rngs[i]
            plan = session_plans[i]
            speaker_names = session_speaker_names[i]
            start, end = session_spans[i]
            durations = [len(wav) / sample_rate for wav in flat_waveforms[start:end]]

            turns = _lay_out_session(plan, durations, rng)
            waveform = _compose_waveform(turns, flat_waveforms[start:end], sample_rate)

            audio_id = f"session_{i}"
            wav_path = k_dir / f"{audio_id}.wav"
            rttm_path = k_dir / f"{audio_id}.rttm"

            Audio(
                waveform=torch.from_numpy(waveform).unsqueeze(0),
                sampling_rate=sample_rate,
            ).save_to_file(wav_path)
            _write_rttm(rttm_path, audio_id, turns, speaker_names)

            # Verify against the file just written, not the in-memory turns -- the
            # guarantee that matters is what a later reader (the evaluation script) sees
            # on disk, and _plan_session's guarantee-by-construction is exactly the thing
            # under test here, not assumed.
            written_speakers = {line.split()[7] for line in rttm_path.read_text().splitlines() if line.strip()}
            if len(written_speakers) != k:
                raise RuntimeError(
                    f"k={k} session_{i}: enforce_num_speakers violated -- wrote "
                    f"{len(written_speakers)} distinct speakers ({sorted(written_speakers)}), "
                    f"requested {k}. _plan_session guarantees this by construction, so this "
                    "is a bug in this module, not a sampling outcome to retry past."
                )

            session_records.append(
                {
                    "k": k,
                    "session_index": i,
                    "wav": str(wav_path.relative_to(out_dir)),
                    "rttm": str(rttm_path.relative_to(out_dir)),
                    "speakers": speaker_names,
                    "num_turns": len(turns),
                    "duration_seconds": round(max(t.end for t in turns), 3),
                }
            )

    manifest = {
        "method": (
            "tts-composed sessions: single-speaker utterances are synthesized directly with "
            "named Qwen3-TTS voices and placed at chosen timeline offsets, so every RTTM line's "
            "start and duration comes from a real synthesized waveform -- no VAD, forced "
            "alignment, or energy threshold estimates ground truth. Session composition "
            "(turn-taking, dominance, sentence length, overlap) borrows NeMo "
            "MultiSpeakerSimulator's parameterization; see generate.py's module docstring for "
            "which values are NeMo's own defaults and which are this module's judgement."
        ),
        "tts_model": {
            "path_or_uri": str(tts_model.path_or_uri),
            "revision": tts_model.revision,
            "resolved_commit_sha": tts_model.commit_sha,
            "voice_pool": voice_pool,
        },
        "session_params": {
            "turn_prob": TURN_PROB,
            "dominance_var": DOMINANCE_VAR,
            "min_dominance": MIN_DOMINANCE,
            "sentence_length_params": list(SENTENCE_LENGTH_PARAMS),
            "mean_overlap": MEAN_OVERLAP,
            "mean_silence": MEAN_SILENCE,
            "mean_silence_var": MEAN_SILENCE_VAR,
            "overlap_fraction_range": list(OVERLAP_FRACTION_RANGE),
            "session_length_seconds": SESSION_LENGTH_SECONDS,
            "assumed_words_per_second": ASSUMED_WORDS_PER_SECOND,
            "enforce_num_speakers": True,
            "params_source": (
                "turn_prob, dominance_var, min_dominance, sentence_length_params, mean_overlap, "
                "mean_silence, mean_silence_var are NeMo's own shipped defaults "
                "(tools/speech_data_simulator/conf/data_simulator.yaml); "
                "overlap_fraction_range, session_length_seconds, assumed_words_per_second are "
                "this module's own judgement, not NeMo's and not measured -- see generate.py's "
                "module docstring."
            ),
        },
        "seed": seed,
        "counts": list(counts),
        "sessions_per_count": sessions_per_count,
        "sessions": session_records,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out_dir


def main() -> int:
    """CLI: ``uv run python scripts/speaker_ceiling/generate.py --out DIR --counts ... --sessions N --seed S``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path, help="output corpus root directory")
    parser.add_argument("--counts", required=True, type=int, nargs="+", help="speaker counts (k) to generate")
    parser.add_argument("--sessions", required=True, type=int, help="sessions per speaker count")
    parser.add_argument("--seed", required=True, type=int, help="top-level reproducibility seed")
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu", "mps"], help="device for TTS synthesis (default: auto)"
    )
    args = parser.parse_args()

    device = DeviceType(args.device) if args.device else None
    out_dir = generate_corpus(
        out_dir=args.out,
        counts=args.counts,
        sessions_per_count=args.sessions,
        seed=args.seed,
        device=device,
    )
    print(f"wrote corpus to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
