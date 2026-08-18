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
