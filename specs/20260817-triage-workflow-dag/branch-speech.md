# Branch — speech. What, whose, and how well

Draft, 2026-08-19. This file governs the speech branch; where `flowchart.md`, `design.md` or
`workflow.nf` disagree they are stale and are not a source of structure. Sibling branches own their
own files.

## What it decides

TAXONOMY has already decided that speech is present. This branch is not asked *whether*. It is asked
three things, and it must be able to decline each one separately:

- **what** was said — a transcript whose per-word confidence comes from agreement between
  recognizers, not from any single recognizer's own head;
- **whose** — spans carrying a speaker label, and target-versus-other only when a target sample was
  supplied;
- **how well** — a quality reading that gates the branch before anything else runs, because a
  recording too poor to measure must be dismissed rather than measured badly.

Elements, from D11 and D9: syllable repetition, word production, connected speech, singing.

## Signature

```
speech(audio, kinds, hint?, airway_events?) -> fail(reason) | flag(reason, partial) | pass(product)
```

| port | direction | type | meaning |
| --- | --- | --- | --- |
| `audio` | in | decoded audio | from ADMIT, the recording as supplied — unresampled, unmixed |
| `kinds` | in | one `Estimate` per kind | from TAXONOMY. Read for **one** thing: whether airway is present |
| `hint` | in | `AudioHints` or absent | `audio_hints.py`. Conditions decisions, never measurements (D5) |
| `airway_events` | in | spans, or absent | branch 1's product. Absent when airway is absent, or when branch 1 has not run |
| `fail` | out | reason | the instrument cannot measure this recording; nothing is claimed about its content |
| `flag` | out | reason, partial | a judgement that could have gone either way; the partial product travels with it |
| `pass` | out | product | transcript, spans, count, quality; target attribution only if a target was given |

`kinds` is an input rather than an implicit precondition because the cough defence below needs to know
whether airway was claimed *absent by unanimity* or merely not localised. Those are different
situations and the branch behaves differently in each.

`airway_events` makes this branch depend on branch 1 whenever airway is present. That is a real
ordering constraint between two branches that would otherwise run concurrently, and it is taken
deliberately: the alternative is a second cough detector inside this branch, which is a second
producer of a product branch 1 already produces, and `ports.md` rule 5 makes that a build error
rather than a merge. Where the input is absent the defence still runs on its other three legs.

## Nodes

| # | node | kind | in | out |
| --- | --- | --- | --- | --- |
| 0 | `resample_16k` | pure | `audio` | `audio_16k` |
| 1 | `dsp_tracks` | pure | `audio_16k` | `speech_energy_track`, `speech_periodicity_track` |
| 2 | `voice_frames` | model, cached | `audio_16k` | `brouhaha_track` — raw VAD posterior, SNR, C50 per frame |
| 3 | `coarse_regions` | pure | `brouhaha_track` | `coarse_speech_regions[]`, `coarse_other_regions[]` |
| 4 | `measurability` | **gate** | `coarse_*_regions`, `audio_16k` | `quality`, or `fail` |
| 5 | `speaker_segments` | model, cached | `audio_16k`, `cfg.count_diarizer` | `primary_diarization` — spans **and** raw per-speaker posteriors |
| 6 | `transcribe` | model, cached, ×N | `audio_16k`, `cfg.recognizers[]` | `hypotheses{model_id: [word, onset, offset]}` |
| 7 | `withdraw_nonlexical` | pure | `primary_diarization`, `brouhaha_track`, `hypotheses`, `dsp_tracks`, `airway_events?`, `kinds` | `speech_spans[]`, `withdrawn[]` |
| 8 | `count_speakers` | **gate** | `speech_spans`, `primary_diarization`, `hint.targeted_speaker_count?` | `speaker_count`, `count_moved_by_withdrawal` |
| 9 | `second_opinion` | model, cached | `audio_16k`, `cfg.second_diarizer`, `speaker_count` | `secondary_diarization` |
| 10 | `label_agreement` | pure | `primary_diarization`, `secondary_diarization` | `clusters{C*: per-model labels}`, `label_disagreement` |
| 11 | `fuse_words` | pure | `hypotheses`, `speech_spans`, `dsp_tracks`, `brouhaha_track` | `transcript[]`, `word_agreement`, `fabrication_candidates[]` |
| 12 | `embed_labels` | model, cached | `audio_16k`, `speech_spans`, `hint.target_speaker` | `label_embeddings{label: vector, dispersion}` |
| 13 | `attribute_target` | pure | `label_embeddings`, `hint.target_speaker`, `cfg.embedder` | `target_attribution{label: Estimate}` |
| 14 | `speech_verdict` | **gate** | everything above | `fail` \| `flag` \| `pass` |

Nodes 9, 12 and 13 have inputs that are frequently absent, and `ports.md` rule 7 governs: a task whose
input port has no product does not run and produces nothing. There is no skip flag. Node 9 does not run
when `cfg.second_diarizer` is unset or when `speaker_count` is 1; nodes 12-13 do not run when the hint
carries no target. Their outputs are then **absent from the product**, which is a different thing from
present-and-empty and is what §"The product" turns into a type.

Every consumed product in that table is produced by a node in it or is a declared branch input. That
check is the point of writing the table — `branch-1-airway.md` found four dangling ports in its own
first draft this way, which is F-187 caught before any code existed.

## 0-1. What the branch does to the audio, and why ADMIT did not

ADMIT deliberately hands over the recording as supplied: no resampling, no channel reduction, because
"whatever normalisation a consumer needs, that consumer performs." This branch is that consumer, and it
needs 16 kHz mono three times over: `torchaudio_squim.py:63-66` raises `ValueError` on anything else,
`speaker_embeddings/speechbrain.py:114` records 16 kHz from the ECAPA/ResNet/x-vector model cards, and
both recognizers are 16 kHz models. So the resample happens **once**, in a declared node, and
`audio_16k` is the product every later node names. A resample performed privately inside three nodes is
three chances to differ.

`dsp_tracks` is short-time RMS and normalised autocorrelation, the same pair TAXONOMY's residual gate
uses. It is computed here rather than consumed from TAXONOMY because TAXONOMY publishes only `kinds` —
"there is no separate evidence port for a consumer to join back" — and a second producer of the same
product name is a build error. The names are branch-local for that reason.

Its purpose is narrow and it is the only non-model member of any fold in this branch. The
verified-label work established two rules that hold here: *the fold must include at least one
non-classifier member*, and it is the only member with full recall — 6 of 6 events on the probe
recording, against 5 of 6 for every model instrument. `fuse_words` uses it to ask whether a word slot
sits over any energy and any periodicity at all.

## 2-4. Quality, as a gate

**Quality is the first gate and it is not the first measurement.** SQUIM is a family of *speech*-quality
estimators, and applied to something that is not speech its numbers are not interpretable: the
separation probe measured **subjective MOS 4.259 on a stream containing one isolated cough**, against
3.058 for the input that held the actual sentence. Scoring the whole recording — speech, coughs, room
tone and background music together — asks the estimator a question it cannot answer. So one measurement
comes first: Brouhaha's raw VAD posterior, which on the probe read **0.689** through the verified
utterance and **0.0049-0.0085** on the two breaths and two coughs. That contrast is what makes coarse
regions possible, and it is a measurement, not a decision.

Coarse regions are presence, never edges. The repo records frame-posterior VADs firing through a 0.4 s
inter-turn gap with onset/offset MAE ≈ 2.6 s (`SPEECH_DETECTION_SOTA_REVIEW_2026.md:56-61`), and on the
verified span Brouhaha's raw >0.5 boundary was −110 ms at onset. Edges come from §11.

### The objective head only

| head | reference-free | used here |
| --- | --- | --- |
| SQUIM objective — STOI, PESQ, SI-SDR | yes | **yes**, per coarse region and for the file |
| SQUIM subjective — MOS | **no**, needs a non-matching reference | **no** |

The subjective head is refused rather than deferred. A non-matching reference is a recording someone
chose, so the MOS it produces is a comparison against that choice, and the 4.259-on-a-cough measurement
is what an uninterpretable comparison looks like. If MOS is ever wanted, the reference becomes a
declared config artifact with a derivation, which is a decision nobody has taken.

### What "too poor to measure" can mean when nothing is fitted

Three conditions, and none of them is a margin on a quality number:

| condition | why it needs no fit |
| --- | --- |
| the objective head **refuses** — raises, or returns non-finite STOI/PESQ/SI-SDR | an instrument that will not run has not measured anything. This is ADMIT's decode-failure row one level up |
| the coarse speech regions do **not** score higher than the non-speech regions on STOI | a self-comparison inside the recording, so no literal is involved. The probe supports it existing: SQUIM STOI reads **0.18-0.44** across the cough region, where two independent Whisper models transcribe nothing, against the utterance region where both transcribe the sentence |
| a required model is **unavailable** — gated, 403, or the pinned revision cannot be fetched | measured precedent: `pyannote/segmentation` returns `GatedRepoError: 403` for this account on every file fetch, and `api.model_info()` *succeeds* on the same repo, so a metadata-based access check reports success wrongly (D17) |

The second row is the branch's dismissal case, and it is the honest form of "too poor to measure": the
estimator cannot tell the parts of the recording that hold speech from the parts that do not, so it has
no purchase on this file. It is unavailable when the recording is wall-to-wall speech and there are no
non-speech regions to compare against; the gate then reports *not applicable* and falls back to refusal
only. n=1: the 0.18-0.44 contrast is one file.

**What is deliberately not here: a floor on STOI, PESQ or SI-SDR.** Nothing has fitted one, this project
has been bitten twice by unfitted literals — a silhouette coefficient read as a probability, and a
2→10 dB HNR ramp under which ordinary voiced speech at a median 8.12 dB read as only partly voiced —
and a threshold invented here would decide the destructive outcome. The slot is declared and stays
empty. The consequence is that almost every recording clears this gate, exactly as almost every
recording clears ADMIT, and the graded quality reading instead travels as an `Estimate` per measure into
every downstream confidence and into the flag.

The graded dismissal is still reachable, by a route that needs no threshold: **unanimous refusal** at
node 14. When quality is low *and* neither recognizer produced a word *and* no diarizer span survived
withdrawal, every instrument has declined, and unanimity is a fail. That mirrors TAXONOMY's rule that
absence needs unanimity while presence needs only agreement — the destructive outcome is the hardest to
reach.

**The measurement that would fill the empty slot.** Score SQUIM objective against whether this branch's
own products subsequently agreed — per-word recognizer agreement, count stability under withdrawal,
per-label embedding dispersion — over a corpus spanning SNR and reverberation. That needs no human
labels, because the target variable is the branch's own downstream self-consistency, which is what
"measurable" means operationally. Fitting against human "is this good enough" verdicts is the
alternative and needs a labelled corpus nobody has.

## 5. Speaker count, with pyannote

`pyannote/speaker-diarization-community-1`, pinned to a resolved 40-hex commit, never to `main`.

Seed-17 speaker-ceiling probe, recorded in `model_registry.yaml:168-280` and designed in
`specs/20260809-112417-speaker-ceiling-probe/`. **Exact speaker-count accuracy** — does the backend
report exactly *k* distinct speakers:

| backend | k=1 | k=2 | k≥3 | structural ceiling |
| --- | --- | --- | --- | --- |
| **pyannote community-1** | **100%** | **85%** | **≤45%** | none observed; emits up to 8 |
| NVIDIA Sortformer | 0% (0/20) | 65-80% at k=2..4 | 0% at k≥5 | 4, confirmed 20/20 at k=8 |
| DiariZen | unrecorded | **75-90% at k=2-3** | degrades beyond | none observed; up to 8 |
| VibeVoice-ASR-HF | unrecorded | **95%** | 20% by k=8 | none observed; up to 16 |
| USC-SAIL child-adult | 50% | 70% | necessarily 0% | 2, confirmed 20/20 at k=8 |
| MOSS-Transcribe-Diarize | 0% | 25-65% | 25-65% | none observed; up to 12 |

**Two caveats travel with every number in that table, and must be restated wherever it is quoted.**

1. The corpus is **TTS-composed**, 160 sessions, k=1..8, **20 sessions per k**, with **no room acoustics
   and no channel variation**. Every figure is a measurement under those conditions and not a claim
   about a real recording. Reverberation, a far-field talker and a phone channel are the three things a
   diarizer meets in the field and none of them is in this corpus.
2. The raw artifact is **not in the repository.** The probe's design names
   `src/senselab/audio/tasks/speaker_diarization/data/speaker_ceiling_profile.json` as its output,
   carrying the full confusion; that directory **does not exist** in a checkout. The per-k figures
   survive only as summaries transcribed into `model_registry.yaml` and cannot be recomputed. Anyone who
   disagrees with the derivation rule has nothing to recompute from, which is precisely the failure the
   probe's own design set out to prevent.

### The count's codomain is {1, 2, ≥3}, not an integer

At k≥3 pyannote is exactly right at most 45% of the time, so a specific integer above 2 is wrong more
often than right, and publishing it invites a consumer to trust it. The branch therefore reports a
**bucket**, and `≥3` means "more than two voices, count unverified on this evidence".

`speaker_count` is an `Estimate` (`utils/data_structures/estimate.py:28`) and its fields are wired to
carry the caveats rather than to state a number:

| field | value | why |
| --- | --- | --- |
| `raw` | fraction of contributing diarizers reporting this bucket | the statistic available **on this recording** |
| `n_evidence` | how many diarizers contributed — 1, or 2 when node 9 ran | with one diarizer this is 1, and shrinkage keeps a single opinion from publishing as certainty |
| `prior` | the probe's measured accuracy for this bucket | so one diarizer agreeing with itself resolves toward the accuracy actually measured for it |
| `prior_key` | the config key holding that accuracy | its derivation is the seed-17 probe, **and records that the profile is not recomputable from a checkout** |
| `population` | `tts-composed-seed17-no-room-acoustics` | `Estimate` rejects a blank population precisely so a figure fitted on one population cannot silently reach a recording from another. This is that mechanism doing its job |

That `prior_key` points at a derivation citing a number nobody can recompute is a defect, not a design
feature. The fix is to regenerate the profile into `data/`, and it is cheap relative to re-running 160
GPU sessions, because the probe's design records that the generated corpus was kept.

Two further honesty items. **20/20 is not 100%.** A Clopper-Pearson interval on 20 successes in 20
trials has a lower bound near 83%, so a published 1.0 overstates it; the shrinkage in `Estimate` is what
stops that, and it is the same defect the class was written for — "a bucket backed by 4 unanimous
sources and one backed by 20 both published `P = 1.000`". And **`hint.targeted_speaker_count` never sets
the count.** It is protocol intent, and a measured count that disagrees with declared intent is a
finding that flags, on the same principle as the airway branch's breath-count prior.

## 6, 9, 10. "If not one speaker, a better diarizer" — the claim the evidence does not support

The instruction is to run a better diarizer when the count is not 1. **The premise is unmeasured, and
the recorded numbers point the other way in the regime where the node would run.**

- At **k=1**, pyannote is the only backend that counts reliably at all: 100%, against 0% for Sortformer
  and MOSS and 50% for child-adult. "Every alternative is worse" is true here, and here is exactly
  where the second diarizer does *not* run.
- At **k=2**, VibeVoice measured **95%** and DiariZen **75-90%** against pyannote's **85%**. At **k=3**
  DiariZen's band sits above pyannote's **≤45%**. So in the regime the node would actually run in, two
  alternatives measured at or above pyannote, and "better for multiple speakers" is not an unmeasured
  claim in the direction stated — it is a claim the recorded numbers weakly *support*, for two specific
  backends, under caveats that make the support thin.
- **n = 20 separates nothing.** 17/20, 18/20 and 19/20 are what 85%, 90% and 95% are; their binomial
  intervals overlap heavily. On this corpus no backend is distinguishable from pyannote at k=2.

So the design cannot name a default, and it does not:

```yaml
speech:
  count_diarizer: pyannote/speaker-diarization-community-1
  # derivation: seed-17 probe — 100% exact count at k=1, 85% at k=2; the only backend with a
  # usable k=1 figure, which is what a count gate needs first. Caveats in branch-speech.md.
  second_diarizer:            # unset
  # derivation:               # empty. Nothing measured settles this.
```

`second_diarizer` unset means node 9 does not run and node 10 has no input, so the count stands on
pyannote alone with the bucket rule, `n_evidence = 1`, and the shrinkage that implies. An operator who
sets it gets a second opinion whose contribution is visible in `n_evidence` and whose disagreement is
visible in `label_disagreement`. What is refused is a hidden default that quietly makes a k≥3 count look
better-supported than it is.

**What would settle it.** Regenerate the seed-17 corpus at k=2 and k=3 with room impulse responses and
channel variation applied, at n large enough to separate 85% from 95% — n=20 cannot; separating those at
p<0.05 needs on the order of hundreds of sessions per cell — and score not exact-count accuracy alone
but the **joint** behaviour: does adding backend B raise accuracy over pyannote alone, and does B
disagree with pyannote where pyannote is wrong rather than where it is right? A second opinion that errs
in the same direction adds `n_evidence` without adding information, and per-backend exact-count accuracy
cannot detect that. DER on the retained corpus is the secondary measurement — the probe's design records
that DER was deliberately deferred and the audio kept so it could be computed later.

When node 10 does run, its output labels are `C*`, the pass-wide cluster namespace, and the map back to
each model's own `SPEAKER_00`-style labels travels with them. Where only one diarizer contributed, spans
carry that diarizer's own labels unchanged. The fused `S*` namespace is not this branch's to allocate,
and identity repair's `R*` is not either.

## 7. The hazard — a cough reads as a speaker turn

The measured hazard, from D16 and the verified labels:

| region | Brouhaha VAD, raw mean | community-1, max over speakers, raw mean |
| --- | --- | --- |
| breath ~2.28 s | 0.0049 | 0.0000 |
| breath ~5.31 s | 0.0055 | 0.0000 |
| silence [0, 2] | 0.0068 | 0.0000 |
| **cough @ 7.924 s** | 0.0053 | **0.574** |
| **cough @ 9.609 s** | 0.0085 | **0.906** |
| real utterance [11.5, 13.3] | 0.689 | 0.790 |

**The louder cough scores higher than the real speech.** And community-1's *thresholded segment list*
reports both coughs as clean speech spans with no hesitancy whatsoever. A count taken from that segment
list is wrong in a way nothing downstream can see, because the graded response underneath has been
flattened into a binary claim. Brouhaha is not missing the coughs for want of energy — its SNR head
reads 9.37 and 13.31 dB on them against 2.50 dB for silence — its VAD head is *declining to call them
speech*, which is a semantic judgement and is what makes the contrast trustworthy.

`withdraw_nonlexical` is the defence, and it has four legs so that no single one carries it:

| leg | the measurement behind it | available when |
| --- | --- | --- |
| **1. airway subtraction** — remove diarizer spans overlapping a branch-1 airway event | branch 1's spans are the only instrument carrying an event *identity*: HeAR labelled 4 of 4 correctly, YAMNet `Cough` 1.000 | `airway_events` present |
| **2. the D16 discriminator** — a span whose Brouhaha VAD raw mean is at floor while community-1 responds is voiced non-lexical, not speech | 0.0053 / 0.0085 against 0.689: two orders of magnitude | always |
| **3. posterior hesitancy** — count from raw posteriors, never from the segment list | inside the cough region community-1's raw posterior is *partial*: mean 0.69, 20% of frames below 0.5, **3.4%** above 0.99, against **78.5%** above 0.99 in the real utterance | always |
| **4. word absence** — a span in which no recognizer produced a lexical token is not evidence of a speaker | on the coughs CrisperWhisper emitted `[cough]`, `[UH]`, `[breath]` — non-lexical tokens; on the verified speech its words covered 98.3% | always |

**Leg 4 is asymmetric and must stay that way.** Word *absence* withdraws a span. Word *presence* does
not confirm one, because a confidently fabricated transcript over non-lexical vocal sound is a known
failure mode of exactly this input: F-166, `speech_presence_link.py:249-287`, status verified-latent —
Whisper's `no_speech_prob` head is trusted to flag hallucination when it reads high, and it can emit
confident, low-`nsp` fabricated words over a cry or a babble. So **`no_speech_prob` is not read anywhere
in this branch.** A span with words still has to clear legs 2 and 3.

Two things this defence is careful not to overclaim. On the probe file the coughs did **not** invent a
speaker: community-1 assigned both to `SPEAKER_00`, the same label as the genuine utterance, and found
one speaker in total. So the measured failure here is a false *span*, not a false *count* — the count
error is a mechanism, not something observed on this file. And leg 1 cannot fire for breath or mouth
noise, which are invisible to both members of the discriminator: with breath, mouth noise and silence
all in the "neither responds" cell, that cell cannot be read as "no voice".

Withdrawal is reported, never silent. `withdrawn[]` carries every removed span with the leg that removed
it, and `count_moved_by_withdrawal` says whether the count would have been different without the
defence. A count that moved is a flag, because it means the diarizer and the defence disagree about how
many voices this recording holds.

## 11. Multi-ASR, and a confidence built from agreement

| recognizer | role | pin | measured |
| --- | --- | --- | --- |
| `nyralabs/CrisperWhisper2.0_turbo` | **edges** and one identity vote | resolved commit; this branch prefers the 2026-08-17 retrain `de0369c8` | on the verified speech span: onset **−13 ms**, offset **−27 ms**, coverage **98.3%** — best of six instruments |
| `Qwen3-ASR` (`qwen-asr==0.0.6`, subprocess venv, aligner `Qwen/Qwen3-ForcedAligner-0.6B`) | second identity vote | resolved commit | word-timing accuracy **unmeasured** on any verified span |

**The revision choice differs from branch 1's, and both are deliberate.** The CrisperWhisper retrain
changed only the non-speech annotations — `831f87e1` emits `[breath] [breath] [cough] [UH] [breath]`,
`de0369c8` emits `[cough]` — and **both revisions recover the speech identically**. Branch 1 needs the
older weights or it loses its only breath-edge source; this branch is indifferent, so it takes the
current model rather than deliberately running a superseded one. Either way the revision is pinned and
recorded: the earlier measurement session reached the old weights through a locally cached `refs/main`
one day after upstream pushed, which is CLAUDE.md's own recorded hazard occurring inside our own
measurements.

Whisper large-v3-turbo is available and is **not** a member of the fold for edges: on the same verified
span its onset was **+187 ms** against CrisperWhisper's −13 ms, and its coverage 87.5%. It remains a
candidate third identity vote.

### What is compared, exactly — three comparisons, three products

**Per-word slot, and this is the confidence.** Align the two normalised token sequences by edit distance
(`speech_to_text_evaluation/utils.py:50`, `normalize_transcript_for_wer`, already in the repo), then per
aligned slot:

| slot outcome | `raw` | `n_evidence` |
| --- | --- | --- |
| both recognizers produced the same token | 1.0 | 2 |
| both produced a token, different tokens | 0.0, with both readings retained | 2 |
| one produced a token, the other nothing in that slot | 1.0 | **1** |

`n_evidence` is what separates the last two rows from each other, and it is why a two-recognizer
agreement cannot publish as certainty: with two voters the maximum `n_evidence` is 2, and shrinkage
toward the prior is large. A slot filled by one recognizer alone is not 100% agreement; it is one
source, and the `Estimate` says so.

**File level, and it is not the confidence.** WER between two transcripts is *direction-dependent* —
`jiwer.wer` divides the edit distance by the reference's length and there is no reference here — so a
single "WER between them" is a number that changes when you swap the arguments. The branch reports the
edit distance normalised by the **mean** of the two token counts, and both directional WERs beside it.
One summary number in place of those three would be a choice of reference nobody declared.

**Per span, and it is what matters for attribution.** Did both recognizers place their words inside the
*same* speaker span? Two recognizers can agree on the words and disagree about which voice said them,
and that disagreement is invisible to both comparisons above while being decisive for the product's
spans.

### Agreement is not accuracy

Two recognizers can be wrong together, and on this branch's hardest input they systematically are: the
F-166 mechanism is a *speech prior* imposed on non-lexical vocal sound, and both members have one. The
same mechanism is legible in the one case measured end to end — CrisperWhisper mapped a cough's voiced
phase to `[UH]`, a filler vowel, and its aspirate tail to `[breath]`, which is "a speech prior imposed
on a non-speech event, not a random error."

So `fuse_words` carries a non-classifier check: a word slot whose region shows no energy and no
periodicity in `dsp_tracks`, or whose Brouhaha VAD raw mean is at floor, becomes a
`fabrication_candidate` — reported, and flagged, never silently deleted. This is the verified-label rule
applied literally: *the fold must include at least one non-classifier member*, and it was the envelope,
not classifier agreement, that made the correct labels correct.

And the standing rule from D12 survives: **word absence is not speech absence.** A surviving span with
no words does not become "no speech"; it stays a span, with `n_recognizers_with_words = 0` on it.

## 12-13. Target comparison, only with a target

`speechbrain/spkrec-ecapa-voxceleb` is the default embedder, `spkrec-resnet-voxceleb` the alternative;
both 16 kHz. Embeddings are computed **per surviving span, pooled per label**, on raw audio.

**With a target.** `hint.target_speaker` is a `TargetSpeakerEmbedding`: a unit-norm vector plus
`SpeakerEmbeddingProvenance` carrying `model_id` and a validated 40-hex `model_commit_sha`. The guard is
therefore available and is mandatory: **if the provenance's model and commit do not match the embedder
this branch ran, the comparison does not happen.** A cosine between vectors from two different embedders,
or two commits of one embedder, is a number with no meaning, and the provenance class exists precisely
so that "recording a SHA while loading through a ref" cannot hide it. Mismatch yields
`target: unresolved(reason)`, which reaches the flag.

Where the guard passes, `target_attribution` is one `Estimate` over {target, other} **per label**, not
per span. Labels are the unit because a per-span decision multiplies a weakly-supported judgement by the
number of spans; spans inherit their label's estimate, and D14's ordering is preserved — every label's
measurements already exist, and target assignment selects among results rather than gating them.

The separation test uses the recording's own dispersion, not a literal: a target call is made only when
the gap between the best and second-best label-to-target cosine exceeds the **within-label** cosine
dispersion of those labels. Two labels closer together than their own internal spread are not separated
by this instrument, and that is `unresolved`. The reason for that construction is that the shipped
alternatives are bare literals — `speaker.same_floor: 0.30`, `diff_floor: 0.70`,
`cluster_cosine_threshold: 0.5`, `centroid_min_similarity: 0.5` in `default.yaml:183-195`, none with a
derivation — and this branch will not add a fifth.

Two limits stated rather than assumed. The embedders are speech-trained, so identity is "strong for
connected speech, weak for sustained vowels, and close to unusable for cough and breath" (D13); this is
why attribution runs only on spans that survived §7. And the whole comparison reads **raw** audio:
`audio_enhanced` is deliberately not wired to speaker identity in the existing port design, an
off-target speaker is quiet and incidental by construction, and enhancement is what removes exactly
that. The configured default enhancer would also be actively harmful here —
`sepformer-wham16k-enhancement` is net-harmful above ≈5 dB SNR, and **all four quiet
`streaming-audio-*` captures meet the harmful condition.**

**Without a target.** Spans carry their label and nothing else. **No span may be called "the target",
and the product has no target field at all** — not an empty one, not a null one, not a "probably". The
tempting prior is available and is refused as a *product*: D13 offers "the dominant close-miked source",
and it is a prior that is sometimes exactly wrong, most wrong where the participant speaks least. The
branch reports each label's share of surviving speech and its proximity evidence — level, spectral tilt,
Brouhaha's C50 — as measurements a human can read, and it does not promote any of them to a role.

## Spans, and why one `(start, end)` pair is not enough

Onset and offset are not comparably measurable, and on this branch the asymmetry is directional as well
as unequal in size. Every instrument scored against the verified speech span placed the **offset early**:
−27 ms (CrisperWhisper words), −28 ms (Brouhaha raw), −149 ms (community-1 segment), −179 ms (community-1
raw), −7 ms (Whisper large-v3-turbo). None placed it late. Meanwhile onsets scattered in both directions:
−13, −110, −29, −60, **+187** ms. A symmetric ± on either edge would describe neither.

```
Edge      = { t_s, minus_s, plus_s, instrument, instrument_commit, method }
Boundary  = Edge | Unresolved(reason)

SpeechSpan = {
  onset:    Edge,                    # never Unresolved
  offset:   Boundary,                # legitimately Unresolved
  label:    str,                     # the primary diarizer's own label
  cluster:  str | absent,            # C*, present only when node 10 ran
  target:   Estimate | absent,       # over {target, other}; absent when no target was given
  words:    [WordRef],
  n_recognizers_with_words: int,
  quality:  { stoi, pesq, si_sdr },  # this span's region, each an Estimate
  withdrawn: reason | absent,
}
```

- `minus_s` and `plus_s` are the **spread across the instruments that actually fired on this edge**,
  which is a measurement of the recording rather than a fitted constant.
- **A single-instrument edge carries `unknown`, not zero.** One instrument agreeing with itself has no
  spread, and publishing 0 ms there is the same defect as a crashed diarizer publishing a confidence
  indistinguishable from one that ran and agreed.
- **The onset is never `Unresolved` and the offset legitimately is.** Onsets are the solved problem: two
  independent instruments agreed within ~30 ms on four verified events. Offsets are where the
  measurements come apart — a breath offset moved **2.03 s** under an envelope-threshold sweep, and five
  independent detectors all stop early on turbulent events, with coverage of **10-52%**.
- **For speech specifically, the asymmetry is representable but barely measured**: on the one verified
  speech span both edges were tight, −13 and −27 ms. The type carries the asymmetry because it must be
  *expressible* — a single `(start, end)` pair invites every consumer to trust both ends equally — and
  because D12 records that for a definitional offset a single-threshold rule reports a choice as a
  measurement. Calling speech offsets as bad as breath offsets would be overclaiming from n=1.

## The product on `pass`

```
SpeechProduct = {
  transcript:      [Word],              # the clean transcript
  nonlexical:      [Token],             # [cough], [breath], [UH] — evidence, not transcript
  spans:           [SpeechSpan],
  speaker_count:   Estimate,            # bucket in {1, 2, >=3}
  quality:         { file: {...}, per_region: [...] },
  agreement:       { per_word: [...], file_distance: {...}, per_span: [...] },
  withdrawn:       [ {span, leg} ],
  fabrication_candidates: [WordRef],
  target:          { label -> Estimate } | absent,
  clusters:        { C* -> {model_id: label} } | absent,
}

Word = { text, onset: Edge, offset: Boundary, confidence: Estimate,
         readings: {model_id: text}, span_label }
```

**"Clean" is defined, not a synonym for tidy.** The transcript holds lexical words only. Non-lexical
tokens are routed to `nonlexical` because those *labels* are unreliable while their *timings* are not —
`[UH]` on a voiced cough phase is the measured case — so they are evidence for §7's defence and for the
airway branch, and they are not words. And clean does not mean single-reading: where the recognizers
disagreed, both readings are retained on the word with `raw = 0.0` and `n_evidence = 2`. Silently
picking one would publish a resolution nobody made.

**What is absent when no target was given:** the `target` map. Spans then carry `label` and, when a
second diarizer ran, `cluster`. Nothing anywhere in the product asserts which label is the participant.

## fail, flag, pass

| outcome | condition |
| --- | --- |
| **fail** | the objective quality head refuses, or returns non-finite values |
| **fail** | the coarse speech regions do not score above the non-speech regions on STOI — *too poor to measure* |
| **fail** | a required model is unavailable: gated, 403, or the pinned commit cannot be fetched |
| **fail** | **unanimous refusal** — no recognizer produced a word, no diarizer span survived withdrawal, and quality is low |
| **flag** | TAXONOMY said speech and this branch's coarse regions find none |
| **flag** | `count_moved_by_withdrawal` — the defence and the diarizer disagree about how many voices are present |
| **flag** | `speaker_count` is `≥3`, always, on the ≤45% figure alone |
| **flag** | recognizer disagreement that changes the span set or the speaker attribution |
| **flag** | `fabrication_candidates` is non-empty |
| **flag** | a target was supplied and is `unresolved` — provenance mismatch, or labels not separated beyond their own dispersion |
| **flag** | a measured count that contradicts `hint.targeted_speaker_count` |
| **pass** | quality gate cleared, at least one span survived withdrawal, and no flag condition holds |

Two notes on the shape of that table. **The flag on recognizer disagreement is structural, not
thresholded** — it fires when the disagreement changes something a consumer reads, not when a distance
exceeds a literal, because no such literal is fitted. And **fail is always about the instrument.** "No
words in this recording" is not a fail: word absence is not speech absence, and a branch that failed on
it would delete the quiet incidental speech this workflow exists to catch, silently, since a rejected
file produces no evidence of what it contained.

## What this branch does not do

- No phonation, glide, laughter or crying measures. Those are the residual branch's, and F0 dispersion
  is its discriminator, not this one's.
- No airway event labels. It *consumes* them and it withdraws spans with them; it never emits one.
- No enhancement of the audio it counts or embeds. Enhancement may run as a probe whose answer-flip is
  evidence (D19), and a word appearing only in an enhanced pass carries that in its provenance — but the
  count, the embeddings and the withdrawal read `audio_16k` from the recording as supplied.
- No `no_speech_prob`, anywhere (F-166).
- No SQUIM subjective MOS.
- No `S*` speaker ids and no `R*` repair ids. This branch emits the diarizer's own labels, and `C*` when
  two diarizers were harmonised.
- No PII scan, no task-match verdict, no trim proposal.

## Parameters

| parameter | status |
| --- | --- |
| `speech.count_diarizer` | set, with a derivation: the seed-17 k=1 figure, caveats attached |
| `speech.second_diarizer` | **unset, derivation empty.** Nothing measured settles it |
| `speech.recognizers[]` + pinned commits | set; CrisperWhisper `de0369c8`, Qwen3-ASR pinned |
| `speech.embedder` + pinned commit | set: `speechbrain/spkrec-ecapa-voxceleb` |
| `speech.quality_floor_stoi` / `_pesq` / `_sisdr` | **declared, empty.** The gate does not read them until they are fitted |
| `speech.count_prior[bucket]` | the probe's per-bucket accuracy; derivation cites a profile **not recomputable from a checkout** |
| `speech.word_agreement_prior` | **declared, empty** |
| `speech.dsp_rms_floor`, `speech.dsp_periodicity_floor` | the one pair with an empirical basis, inherited from TAXONOMY's gate: periodicity **0.933-0.934** at RMS 0.016-0.019 in sustained voicing, against **0.22-0.44** in quiet stretches. One recording |

Every threshold this branch would otherwise need is replaced by a comparison against the recording's own
dispersion — speech regions against non-speech regions for quality, best-versus-second label gap against
within-label spread for the target, instrument spread for an edge's ±. That is a deliberate pattern and
TAXONOMY uses it too, for loud phonation measured "relative to the rest of the recording". It is not
free: a self-referential comparison is undefined when the recording has nothing to compare against, and
each use above names that case.

## Choices the evidence does not force

1. **Quality gates on degeneracy and a self-comparison, not on a quality floor.** The instruction was
   that too-poor-to-measure is a fail; a graded fail needs a fitted floor and none exists, so the graded
   dismissal was moved to unanimous refusal at the verdict. *Settled by*: fitting SQUIM against this
   branch's own downstream self-consistency over a corpus spanning SNR and reverberation.
2. **The second diarizer has no default.** The recorded k≥2 numbers weakly favour DiariZen or VibeVoice,
   and n=20 cannot separate them from pyannote. *Settled by*: the joint measurement in §6.
3. **`airway_events` is a cross-branch input rather than a private cough detector.** This orders the
   speech branch after the airway branch whenever airway is present. The alternative duplicates a
   producer. *Settled by*: measuring whether legs 2-4 alone recover the same withdrawals that leg 1
   does, on a file with verified coughs — the probe file is exactly that file, and the measurement is
   one run.
4. **The count's codomain is a three-way bucket.** The cut between 2 and ≥3 comes from where pyannote's
   accuracy collapses, 85% → ≤45%, on the TTS corpus. A different corpus would put it elsewhere.
5. **Per-label target attribution, not per-span.** Cheaper and more stable, and it loses the case where
   one diarizer label genuinely holds two voices. *Settled by*: per-label embedding dispersion against
   overlap-annotated ground truth.
6. **This branch takes CrisperWhisper's newer revision while branch 1 takes the older one.** Justified
   by both revisions recovering the speech identically — on one file.
7. **Whisper large-v3-turbo is not a third voter.** Excluded on its +187 ms onset at n=1, which is thin
   ground for excluding a whole family. *Settled by*: scoring all three recognizers over a corpus of
   verified spans.

## Contradictions in the source material

Recorded because two of them change what this branch should do.

1. **"Every alternative measured was worse at counting than pyannote" is true at k=1 and false at
   k≥2.** `model_registry.yaml:168-280` records VibeVoice at 95% and DiariZen at 75-90% for k=2, and
   DiariZen's band at k=3 sits above pyannote's ≤45%. The tension is therefore real but inverted: the
   second diarizer is not obviously worse in the regime it would run in, and it is also not measurably
   better, because n=20 separates none of them.
2. **D16 says the k-figures are "unrecorded for VibeVoice and DiariZen"; the registry records both.**
   Same probe, same seed. D16's summary is stale against the registry it cites.
3. **The seed-17 caveat is stronger than "the profile is not in the repository."** The directory the
   probe's design names as its output — `speaker_diarization/data/` — does not exist at all, so the full
   confusion the design promised, so that "anyone who disagrees with the derivation rule can recompute",
   was never landed anywhere a checkout can reach.
4. **"Quality first" and "quality is interpretable" pull against each other.** SQUIM cannot be the
   branch's first *measurement* without being asked about non-speech, which the MOS-4.259-on-a-cough
   figure shows it answers confidently and wrongly. One measurement — the coarse regions — has to
   precede the first gate.
5. **The cough hazard is a mechanism here, not an observed count error.** community-1 scored 0.574 and
   0.906 on the verified coughs, above its 0.790 on real speech, and its segment list reported both as
   clean speech spans — but it assigned them to `SPEAKER_00`, the same label as the genuine utterance,
   and reported one speaker in total. The false *span* is measured; the false *count* is inferred.
