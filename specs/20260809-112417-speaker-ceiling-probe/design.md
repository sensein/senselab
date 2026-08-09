# Measuring each diarization backend's speaker ceiling

Status: design approved 2026-08-09, not yet implemented.

Follows [`../20260809-002926-diarization-capabilities/design.md`](../20260809-002926-diarization-capabilities/design.md),
which introduced `DiarizationCapabilities.max_speakers` and deliberately left it `None` — meaning
*unmeasured* — for four of six backends rather than guessing.

## The problem

`max_speakers` is currently declared for two backends and unknown for four:

| Backend | `max_speakers` | Where the number came from |
|---|---|---|
| USC-SAIL child-adult | 2 | structural: it distinguishes exactly two talkers |
| NeMo Sortformer | 4 | its own checkpoint name, `diar_sortformer_4spk` |
| Pyannote | `None` | unmeasured |
| VibeVoice-ASR-HF | `None` | unmeasured |
| MOSS-Transcribe-Diarize | `None` | unmeasured |
| DiariZen | `None` | unmeasured |

Two of those numbers are *claims*, not measurements. The other four are honest blanks. Neither is
good enough for a value that ships in a registry people read when choosing a model.

This is not hypothetical. On a 4.92-second clip, VibeVoice and MOSS each reported **one** speaker
while DiariZen reported **two**. Nothing in the codebase can adjudicate that, because nothing knows
what any of them is capable of.

## Method

NeMo's `MultiSpeakerSimulator` (`nemo.collections.asr.data.data_simulation`, driven by a
`data_simulator.yaml` session config) generates multi-speaker sessions with a controllable speaker
count and emits **RTTM ground truth** alongside the audio. Ground truth is the whole point: it turns
"how many speakers did it find?" from an argument into a measurement.

It runs inside the `nemo-diarization` subprocess venv this repository already builds
(`nemo_toolkit[asr]`), so no new dependency enters the host or a new venv.

- **Sweep:** true speaker count *k* = 1…8.
- **Sample size:** 20 sessions per *k*, so 160 sessions total. Chosen so an 80 % rule means 16/20
  rather than 8/10, where a single session would flip the verdict.
- **Metric: exact speaker-count accuracy.** For each session, does the backend report exactly *k*
  distinct speakers? Diarization error rate would say more — whether a backend that gets the count
  right also gets the boundaries right — but it needs alignment scoring and considerably more GPU
  time, and it does not answer *this* question. The generated corpus is kept so DER can be computed
  later without regenerating audio.
- **Backends:** all six, including the two with declared ceilings. Testing child-adult's 2 and
  Sortformer's 4 against reality is part of the point; a declared number that fails its own probe is
  a finding.

## Output

`src/senselab/audio/tasks/speaker_diarization/data/speaker_ceiling_profile.json`, carrying the
**full confusion** — for every (backend, true *k*), the distribution of predicted counts — not just
a verdict. Anyone who disagrees with the derivation rule can recompute from the same numbers without
re-running 160 sessions on a GPU.

## The derivation rule, and why it is written down rather than hidden

`max_speakers` has to be reduced from a curve by *some* rule. This one:

> the largest *k* at which the backend reports exactly *k* speakers in **≥ 80 %** of sessions.

The 80 % is a judgement, not a measurement, and is recorded as such in the profile beside the curve
it was applied to. That follows this repository's convention, and its counter-example: `CLAUDE.md`
records two defects that came from literals nobody ever fitted, and `run_config`'s `snr_floor_db`
carries an explicit "⚠ UNDERIVED" marker rather than pretending. A threshold that ships with the
distribution it was applied to can be argued with; one that ships alone cannot.

If the curve is non-monotonic — a backend that scores well at *k*=5 but poorly at *k*=4 — the rule
takes the largest *k* such that **every** count up to and including it clears the threshold. A
ceiling that a backend intermittently exceeds is not a ceiling.

## Refusal

`scripts/calibrate_detection_margin.py` is the precedent: it refuses to emit a profile from
insufficient measurement rather than emitting a weak one. This probe does the same. It **hard-errors
rather than warns** when:

1. Any (backend, *k*) cell has fewer than 20 completed sessions — a partial sweep silently biases
   the ceiling downward, because a backend that crashed on the hard cases looks like one that
   handled them badly.
2. A backend produced zero successful sessions at *k*=1. That means the probe measured the harness,
   not the backend.

A backend that legitimately refuses — child-adult raising on a clip under its 10-second window, or
on a CPU host — is recorded as `refused`, distinct from both a wrong answer and a crash.

## What this does not do

- It does **not** compute DER. The corpus is retained so that remains possible.
- It does **not** change any backend's behaviour, only the declared `max_speakers` values and the
  profile they derive from.
- It does **not** make `max_speakers` load-bearing anywhere. Nothing consumes it at runtime yet;
  this fills in a declaration.
- It does **not** run in CI. It is a GPU measurement job producing a checked-in artifact, like the
  other calibration scripts.

## Success criteria

Four `None`s become measured integers with a curve behind each. Child-adult's declared `2` and
Sortformer's declared `4` are either confirmed or contradicted — and a contradiction is a result
worth having, not a failure of the probe.
