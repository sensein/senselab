# Model selection per branch — draft, and the criterion it rests on

There is no best enhancement or separation model. There is a best model **per branch**, because the
branches want opposite things: the speech branch is content to lose breath and cough, and the airway
branch is destroyed by exactly that. The preserve/destroy matrix is therefore a selection table, not a
ranking.

**Two independent criteria, both required.** Energy preservation and legibility are not the same test:

- `FRCRN_SE_16K` keeps breath energy at −2.0 dB and CrisperWhisper stops annotating the breaths at all.
- `MossFormerGAN_SE_16K` destroys energy across the board (−17 to −53 dB) and yields the richest token
  set of any stream.
- DriftSE v1 preserves per-window energy and hallucinates the content (`[laughter] [laughter]`,
  input-output correlation 0.204).

So a model qualifies for a branch only if it preserves both the energy of that branch's elements and
their legibility to a reader that can name them.

## Draft assignment

| branch | candidate | why | disqualified |
| --- | --- | --- | --- |
| speech | `MossFormerGAN_SE_16K` | speech +0.4 dB, everything else −17 to −53 dB; richest token set | — |
| speech | `MossFormer2_SE_48K` | speech −0.2 dB, breaths destroyed; token set matches input | — |
| speech | `sepformer-wham16k-enh` (repo default) | speech +2.4 dB | net-harmful above ~5 dB SNR; drops "There's" |
| airway, both elements | **DriftSE v2** (`..._pesq_sisdr_ccmse_with_z`) | every window KEPT; token set richer than the input pass | needs output normalisation first |
| airway, both elements | `FRCRN_SE_16K` | breath −2.0/−5.5, cough −1.0/+0.1 | loses all non-speech tokens |
| **cough channel** | `MossFormer2_SS_16K` src1 | cough KEPT (−1.2, −0.1), all else −31 to −50; ASR yields `[cough]` alone | — |
| **breath channel** | `sepformer-dns4-16k-enh` | breaths −1.7/+1.7, coughs −20.6/−17.0 | 8.71% of samples clip on a PCM_16 write |
| any | DriftSE v1 (senselab's current default) | — | hallucinates content; output peaks at 51 741× |

**The two accidental element filters are the useful discovery.** `MossFormer2_SS_16K` src1 is a cough
channel and `sepformer-dns4-16k-enh` is a breath-dominant channel, and they are near-exact complements
of `MossFormer2_SE_48K`, which keeps cough and destroys breath. That gives the airway branch *named*
channels without a conditioned separator — which matters because unasdiff can condition on `Cough` but
has no `Breathing` class among its 41.

Using an enhancer as a class filter is not what any of these models was built for, so it needs
replication before it is load-bearing.

## What this draft rests on, and why it is not yet a decision

**One recording**, one speaker, close-miked, quiet, with background music. n=1 for every cell.

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
