# Enhancement and separation: what each model preserves, and where that matters

There is no best enhancement or separation model. There is a best model **per use**, because
different consumers want opposite things: a use tolerant of losing breath and cough differs from one
destroyed by exactly that. The table below is tool characterisation, not a branch routing table — per
D19, enhancement is not a variant, pathway or dimension in this design; it is an operation a node may
invoke, whose output that node alone consumes. Nothing routes enhanced audio anywhere, so there is no
port to assign a model to.

**Two independent criteria, both required.** Energy preservation and legibility are not the same test:

- `FRCRN_SE_16K` keeps breath energy at −2.0 dB and CrisperWhisper stops annotating the breaths at all.
- `MossFormerGAN_SE_16K` destroys energy across the board (−17 to −53 dB) and yields the richest token
  set of any stream.
- DriftSE v1 preserves per-window energy and hallucinates the content (`[laughter] [laughter]`,
  input-output correlation 0.204).

A model qualifies for a use only if it preserves both the energy of the elements that use cares about
and their legibility to a reader that can name them.

## What was measured

| model | preserves | destroys | notes |
| --- | --- | --- | --- |
| `MossFormerGAN_SE_16K` | speech (+0.4 dB) | everything else (−17 to −53 dB) | richest token set of any stream |
| `MossFormer2_SE_48K` | — | breaths | speech −0.2 dB; token set matches input |
| `sepformer-wham16k-enh` (repo default) | speech (+2.4 dB) | — | net-harmful above ~5 dB SNR; drops "There's" |
| **DriftSE v2** (`..._pesq_sisdr_ccmse_with_z`) | every element, both criteria | — | needs output normalisation first; the one measured safe general-purpose choice |
| `FRCRN_SE_16K` | breath (−2.0/−5.5 dB), cough (−1.0/+0.1 dB) | all non-speech tokens | legibility fails despite preserved energy |
| `MossFormer2_SS_16K` src1 | cough (−1.2/−0.1 dB) | everything else (−31 to −50 dB) | a cough-only channel; ASR reads `[cough]` alone |
| `sepformer-dns4-16k-enh` | breath (−1.7/+1.7 dB) | cough (−20.6/−17.0 dB) | a breath-dominant channel; 8.71% of samples clip on a PCM_16 write |
| DriftSE v1 (senselab's prior default) | — | content itself | hallucinates (`[laughter] [laughter]`, input-output correlation 0.204); output peaks at 51,741× |

**The two accidental element filters are the useful discovery.** `MossFormer2_SS_16K` src1 is a cough
channel and `sepformer-dns4-16k-enh` is a breath-dominant channel, and they are near-exact complements
of `MossFormer2_SE_48K`, which keeps cough and destroys breath. That gives a **candidate second
confirmation channel inside `span_reconfirm`** — reconfirm a proposed cough span against the cough-only
channel, a breath span against the breath-dominant channel — without a conditioned separator, which
matters because unasdiff can condition on `Cough` but has no `Breathing` class among its 41. This is a
candidate, not a decision: it rests on n=1, and using a separator as a class filter is not what any of
these models was built for.

## Where this actually applies

**AIRWAY uses no enhancer at all.** Its detection plan (settled in
[`branch-airway.md`](branch-airway.md)) is YAMNet/AST corroborate → DSP/CrisperWhisper refine the
spans → `span_reconfirm` confirms → grouping, attribution, per-source measurement. This table touches
AIRWAY in exactly two ways: the candidate second confirmation channel above, and the measured fact
that no node needing breath evidence should invoke an enhancer that destroys it — the repo default
takes breath to −26.4 dB, `MossFormer2_SE_48K` takes the two breaths to −37 and −40 dB.

**SPEECH's enhancement question is a genuine, still-open choice.** If enhancement is ever introduced
there for SNR reasons, DriftSE v2 is the only model measured to preserve every element on both
criteria — the airway plan does not want it, but SPEECH's choice is unresolved.

## What this rests on, and its limits

**One recording**, one speaker, close-miked, quiet, with background music. n=1 for every cell above.

**A reference-transcript discrepancy that is not yet explained.** Two runs of
`nyralabs/CrisperWhisper2.0_turbo` at the same pinned revision, on nominally the same audio, produced
different references:

| run | non-speech tokens on the input |
| --- | --- |
| senselab-resampled 16 kHz copy | `[breath] [breath] [cough] [UH] [breath]`, `[cough]` spanning 580 ms |
| harness "raw recording" | `[cough]` alone, spanning 160 ms |

So CrisperWhisper's non-speech annotation depends on the preprocessing path — resample route, load
path, or level. Until that is reconciled, token preservation cannot arbitrate between models, and the
earlier claim that CrisperWhisper reliably annotates breath is weaker than stated: it did on one path
and largely did not on the other.

**Energy figures for three models were affected by a PCM_16 write in an earlier harness** and are only
now correct: `dns4` lost 8.71% of samples, `libri2mix` src0 2.16%, `wsj02mix` src1 1.06%.
