# Design decisions, in the order they were taken

The first draft of `flowchart.md` was written by reading the existing implementation, so it
reproduced the current call graph with better labels instead of designing from the questions the
workflow owes a caller. These decisions correct that, one at a time. The diagrams and `ports.md`
are redrawn once the set is closed, not per decision.

## D1 — ADMIT discards a file only for no signal or a flat signal

ADMIT answers one question: is this file measurable at all. Decode it, reject a signal that is
absent or flat, and stop. No models, no speech test, no thresholds worth arguing about. Its only
verdict is "unusable file", with the reason.

Every other evaluation — including whether there is speech — belongs to TAXONOMY, where it is a
measurement carrying uncertainty rather than a gate returning a boolean.

**Why the earlier draft was wrong.** It gated admission on a speech threshold
(`cfg.triage.speech_threshold`, `cfg.triage.min_speech_s`). An off-target speaker is quiet and
incidental by construction, and a non-lexical vocalization carries no words at all, so a speech gate
at the front discards precisely the recordings this workflow exists to catch.

## D2 — Enhancement is a perturbation, not a route

`perturbations.py:49-66` already models this correctly: an open set of transforms, `identity` plus
`speech_enhancement`, with a registry at `L1/perturbations.json`. The first draft promoted
enhancement to a branch inside ADMIT, which is the implementation's control flow, not a design.

`variant` is a dimension the graph is mapped over. Each task declares which variants it runs on, as
a scope on the task rather than a wire in the graph. VOICE IDENTITY declares `variant = identity`,
so enhancement cannot reach it however many perturbations are added later.

Variants are probes, not repairs applied for our convenience: an answer that flips between raw and
enhanced is unstable, and that instability is evidence the review flag should carry. See
`perturbations.py:79-92`, which already argues a repair has no standing where nothing is broken.

## D3 — Speech detection is a taxonomy measurement, and separation is the candidate primitive

Follows from D1. Open question, deliberately not settled here: whether a speech separation or
extraction model should be the primary evidence for what is in the recording, in place of the
VAD + classifier + diarizer chain.

The argument for it is consolidation rather than accuracy. A separator returns streams, not a flag,
and one run answers three questions: a coherent speech stream means a voice is present; two distinct
streams mean more than one; the residual is the non-vocal content. Today those three answers come
from three mechanisms that disagree about one population — an infant's cry is simultaneously not
speech (the word gate), a background source (`people` in the AudioSet map), and vetoed by YAMNet.

The argument against taking it on faith: separators hallucinate streams from single-speaker input.
Any use must declare its own checks — whether the streams reconstruct the input, whether they are
distinct by embedding distance, whether the energy split is degenerate — and those checks are the
uncertainty of the answers built on them.

What exists today is thin. `speech_enhancement/` wires `speechbrain/sepformer-wham16k-enhancement`,
a denoiser rather than a speaker separator; `source_separation/` has an API and `unasdiff.py` with
no separation model configured in `default.yaml`. This is a new capability, not a rewiring, and
separation-first versus classifier-first is a measurement to run, not a claim to draw.

## D4 — TAXONOMY is a file-level question, and it is the workflow's real gate

The question is **does this recording contain these kinds of sound** — vocal, cough, breathing —
not where they are. Localisation is a later question, asked only of files that got past this one and
only by the consumers that need spans.

Three outcomes, not two:

| verdict | condition | what happens |
| --- | --- | --- |
| present | a target class is confidently present | the file proceeds |
| absent | every target class is confidently absent | the file is discarded, with the reason |
| uncertain | neither of the above | the file is flagged for a human, never discarded |

**Both edges need confidence, and they are not symmetric.** Confident absence is not a low presence
posterior: with weakly-supervised classifiers a low score can mean "not there" or "there but quiet or
masked". Discarding is the destructive action, so it requires positive agreement that nothing is
there — every family low, and the families agreeing with each other. Anything else is `uncertain`,
which flags. The default under doubt is to keep and flag.

The keep and discard thresholds are config parameters with written derivations, and they are
separate values: the cost of discarding a usable recording is not the cost of admitting an empty one.

**This is the first real consumer of `Estimate`** (`utils/data_structures/estimate.py:28`), which has
had none since Phase 1 built it. A per-class presence verdict carrying its evidence count and its
spread is exactly what the type is for.

**Aggregation over time is a decision, not arithmetic.** Clip-level posteriors come from window-level
scores, and a cough is ~0.3 s in a recording of minutes. A mean over windows dilutes a short event
into nothing; a max fires on one spurious window. The aggregator is therefore a named config
parameter per class, and short-duration classes need a high quantile or top-k mean rather than either
extreme. This is the same failure the four-axis grid had: a default that silently disabled what it
claimed to measure.

**SSL frame embeddings are dropped** from the evidence set. They need a trained probe, the repo has
no labelled vocal spans, and an unvalidated probe would be another unfitted decision.

**Evidence families for the fold**, chosen so their failure modes do not correlate:
AudioSet posteriors from two independent classifiers over the vocal label subset; periodicity, HNR
and jitter/shimmer aggregated over the file; and recognised words as corroboration only, never as a
gate. Disagreement between families is the uncertainty, and it is what separates `uncertain` from
the two confident verdicts.

**Blocking prerequisite.** `data/audioset_source_map.json` currently sends `Baby cry, infant cry`,
`Crying, sobbing`, `Laughter`, `Cough`, `Breathing`, `Whispering` and `Singing` to `people`, a
background source category, while `Babbling` goes to `speech`. The classifiers already produce these
labels; the map discards them. Whispered speech being filed as background is a target-speech failure,
not only the pediatric one the register filed as F-168.

## D5 — The task hint is optional, and it conditions the decision, not the measurement

`AudioHints` becomes an optional parameter port on the gate.

- **Without a hint**, the gate defaults to speech: a file with no confident speech presence is
  discarded, everything else proceeds down the speech branch.
- **With a hint**, the gate uses that task's target vocabulary and may discard a file that does not
  meet it, then branches to the breathing, coughing or speaking target branch. A hint may name more
  than one target.

**TAXONOMY measures the full vocabulary either way.** The hint never changes what is measured, only
what the verdict is compared against and which branch runs next. This is the repo's existing
L1-measures / L2-decides rule applied to the gate: a task-conditioned measurement cannot be reused
to answer a different task's question, and it was task-conditioning inside the measurement that made
the background mask unusable as evidence for attribution.

It also gives `AudioHints` its first reader. It has been declared in
`audio/data_structures/audio_hints.py` with zero consumers anywhere in the workflow.
