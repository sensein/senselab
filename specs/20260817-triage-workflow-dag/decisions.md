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

## D6 — Four detection families, liberal thresholds, spans adjudicate

**Families.** AST and YAMNet count as two. Acoustic evidence — periodicity, HNR, jitter and shimmer
aggregated over the file — is the third. An **audio language model** is the fourth, run on Engaging
because it needs a GPU; it assesses content and may report classes outside our vocabulary. Recognised
words remain corroboration only and never gate.

The correlation risk between AST and YAMNet is accepted rather than resolved: both are trained on
AudioSet with the same label space, so they may be wrong together. Recorded because it bounds what
their agreement means — it is not two independent opinions, and the audio LM is the family most
likely to fail differently.

**The audio LM is open-vocabulary, and its output is split.** Classes inside our vocabulary
contribute presence evidence. Classes outside it are **proposals that reach the report and the flag,
never the gate** — otherwise a model inventing a category silently changes what gets discarded. It
also has no calibrated posterior, so it contributes an assertion, not a probability: one vote, with
its text preserved for the reader.

**Thresholds are liberal on content, deliberately.** No labelled verdicts exist, so no threshold can
be derived, and synthesising a benchmark is a separate task. Until then the gate is biased toward
keeping: it discards only a file where nothing fires in any family. The derivation slot stays in the
config and stays empty rather than being filled with a literal nobody measured.

**Span detection adjudicates.** File-level presence is a liberal pre-filter, not a verdict. Whether
the content is really there is settled downstream by localisation, which can withdraw what this stage
admitted. That is what makes liberal thresholds safe here.

## D7 — The AudioSet source map is deleted, and the real categories are brought forward

`data/audioset_source_map.json` collapses the classifiers' own labels into four source buckets, so
`Whispering`, `Cough`, `Breathing`, `Crying, sobbing`, `Baby cry, infant cry`, `Laughter` and
`Singing` all become `people`, a background category, while `Babbling` becomes `speech`. The models
already emit the distinctions the workflow needs; the map throws them away and then the workflow
tries to recover them with three other mechanisms.

The map goes. Actual categories are carried forward as themselves. Pre-alpha policy is rename and
replace outright, so nothing is kept for compatibility.

This subsumes F-168, which was filed as a pediatric mislabelling. It is wider than that: whispered
speech filed as background is a target-speech failure.

## D8 — Spans are a mixture problem, so detection is ordered and its confidence is conditional

Sounds mix: environmental with vocal, speaker with speaker, and breath with the speech it precedes or
rides inside. Classes are therefore not separable at the span level by running independent per-class
detectors over the same waveform — the mixture makes their errors correlate, and each one's
confidence depends on what else is sounding at that moment.

Three consequences the design has to carry.

**Local mixture complexity is a first-class output.** How many distinguishable sources are active in
a region conditions every other confidence in it. It is reported, not hidden inside another number,
because a low-confidence span in a busy region and a low-confidence span in quiet mean different
things and warrant different follow-ups.

**Detection is ordered, not parallel.** Speech is localised first, from speech extraction plus
SQUIM's intelligibility estimate (`features_extraction/torchaudio_squim.py` already wires STOI, PESQ,
SI-SDR and subjective MOS): an intelligibility estimate that holds up is direct evidence that what is
there is speech, not merely voice-like energy. Breath is then localised in what speech did not
explain, where the confounds are fewer — which is what makes a liberal downstream gate on breath
defensible in the absence of speech.

**Breath in quiet and breath against speech are two populations, kept apart.** A pre-speech intake
breath and a breathy phonation are the cases that matter and the cases where every acoustic variable
degrades: HNR is low for breath and also low for breathy voice, periodic energy change is swamped by
syllabic modulation, and a fixed-window embedding of a 0.5-1.5 s breath adjacent to speech captures
both. Merging the two populations into one breath rate or one span set averages a reliable
measurement together with an unreliable one and reports the mean as if it were either. They stay
separate products, each with its own confidence.

**Breath evidence families:** YAMNet's own breathing labels; HeAR embeddings if they can be extracted
on segments as short as a breath, which is unverified and is a measurement, not an assumption — HeAR
is not currently in the repo; low-frequency periodic energy change; and HNR. Every one of these
degrades under mixture, which is why the ordering above exists rather than a flat fold.

## D9 — The branch vocabulary comes from production kind, not from speech/breath/cough

The hint vocabulary is the b2ai task registry (not vendored into senselab; a hint arrives naming a
task). Its 45 tasks elicit eight kinds of vocal production, and the workflow's existing three task
types cover three of them:

| production kind | examples from the registry | what makes it different |
| --- | --- | --- |
| sustained phonation | `adult.prolonged-vowel`, `adult.maximum-phonation-time.v2`, `pediatric.long-sounds` | voice with no words at all; duration is itself the measurement |
| pitch glide | `adult.glides` | the F0 trajectory is the signal; steady-state assumptions are wrong |
| read speech, scripted | `adult.harvard-sentences`, `adult.caterpillar-passage`, `adult.cape-v-sentences.v2` | the expected text is known in advance |
| spontaneous speech | `adult.free-speech.v2`, `adult.story-recall.v2`, `pediatric.conversation-*` | no reference text; highest PII exposure |
| elicited speech | `adult.picture-description`, `adult.loudness.v2`, `pediatric.generative-naming-task` | short, prompted, often single words |
| diadochokinesis | `adult.diadochokinesis.v2` (`puhtuhkuh`), `pediatric.silly-sounds` | rate and rhythm are the measurement; words are meaningless |
| non-speech vocal maneuver | `adult.respiration-and-cough.v2`, `adult.voluntary-cough`, `pediatric.noisy-sounds` | countable events, not turns |
| singing | `pediatric.abcs-and-123s` | pitched and lexical, and it will read as music |

**Four tasks break the current pipeline's assumptions outright, and they are not edge cases:**

- **`pediatric.noisy-sounds`** asks a child to imitate animal and object sounds. AudioSet will label
  these as the animal, confidently and correctly, and the source map then files them as background.
  The child's own vocalization is discarded as environmental noise. This is F-168's mechanism with a
  different population, and the classifier is not even wrong.
- **`adult.loudness.v2`** asks for "hey" spoken normally and then loudly. Level, clipping and
  over-loudness are the measurement; a quality stage treating them as defects reports the task being
  performed correctly as a fault.
- **`adult.diadochokinesis.v2`** is rapid nonsense-syllable repetition. Any ASR returns garbage, and
  any word-conditioned gate reads it as absent speech.
- **Sustained phonation and glides** carry no words at all, so every word-gated path nulls them —
  the general form of the F-165 defect, arriving from four more directions.

**Scripted versus unscripted is the sharpest split for downstream work**, and it cuts across the
speech kinds: where the registry supplies the expected text, alignment has a reference, task match is
a text comparison, and ASR can be constrained. Where it does not, all three become open problems.

**Decisions taken here:** breath spans use temporal exclusion, not a signal-level residual. Per-class
event detection is the mechanism for the countable classes. HeAR is `google/hear` on Hugging Face and
would be new to the repo.

## D10 — Branch on physiological mechanism: four measurement stacks, not 45 tasks

The eight production kinds in D9 are protocol categories. Combining them by the mechanism each
probes collapses them to four stacks, and the stack — not the task — is what selects the measurement
machinery.

| stack | mechanism | primary tasks | what it measures |
| --- | --- | --- | --- |
| A respiratory / airway | subglottal drive, airflow, glottic closure | `respiration-and-cough`, `breath-sounds`, `voluntary-cough` | countable events, onset and offset, noise-band energy, cycle rate; no F0 |
| B laryngeal source | vocal-fold vibration rate, regularity, adduction | `prolonged-vowel`, `maximum-phonation-time`, `pediatric.long-sounds`, `glides`, `loudness` | F0 and its trajectory, HNR, jitter, shimmer, intensity, duration; no words |
| C vocal tract / articulation | tongue, lips, jaw, velum sequencing and precision | `diadochokinesis`, `pediatric.silly-sounds`, `repeating-words`, read passages segmentally | syllable onsets, rate, formant transitions, phone alignment; syllables required, words optional |
| D higher-order control | prosody, lexical retrieval, discourse, executive | `free-speech`, `story-recall`, `picture-description`, naming and fluency, `word-color-stroop`, `abcs-and-123s` | transcript, discourse timing, pauses, semantic content |

**A task is a loading vector over the four, not a member of one.** Maximum phonation time loads A and
B together, which is what makes it diagnostic. Singing loads B, C and D. Read speech loads C
segmentally and D prosodically. A task added to the registry tomorrow needs a loading vector, not a
new branch.

**Only stack D needs ASR.** A, B and C are word-free, which retires the word gate structurally rather
than patching it: F-165 was a D-stack assumption applied to A/B/C material.

**Resonance has no instrumentation.** Nasality and velopharyngeal function are a real mechanism,
probed implicitly by the CAPE-V sentences, and nothing in the pipeline measures it — no nasalance, no
nasal-formant analysis. It is either a fifth stack that is admittedly uninstrumented or it is out of
scope. It must not sit inside C implying coverage that does not exist.

**`pediatric.noisy-sounds` stops being anomalous.** Under mechanism it is B plus C material whose
acoustic target happens to be non-vocal. That framing is what stops a source classifier from
discarding the child for correctly identifying the animal being imitated.

## D11 — The taxonomy vocabulary, and the branches it implies

Eighteen elements, merged as agreed: isolated words absorbs rote sequence recitation, and connected
speech absorbs the read/spontaneous split, because that split is carried by whether the hint supplies
a reference text rather than by anything acoustic.

**The stacks are not a runtime construct.** A-D in D10 were how the branches were derived; they do
not survive as a separate layer. Stacks C and D share their material — syllables and words — and
differ only in what is computed from it, so they collapse into one branch.

| branch | elements | selected when |
| --- | --- | --- |
| airway | inhalation, exhalation, cough, throat clear | those elements are present, or the hint names a respiratory task |
| phonation | sustained vowel, pitch glide, loud phonation | present, or the hint names a phonation task |
| speech | syllable repetition, word production, connected speech, singing | present, or the hint names any speech task |
| imitation | vocal imitation of a non-vocal target | **only with a hint** — see the hazard below |

Elements that are always measured and never select a branch, because each changes a decision without
being anyone's target: other-speaker speech, laughter, crying, environmental sound, device and
handling noise, silence.

**Element 14 is not identifiable without a hint, and that is a discard hazard.** A child imitating a
dog and a dog are the same event to a source classifier, which will be correct and unhelpful. A
`pediatric.noisy-sounds` recording contains imitation and little else, so with no hint the classifier
family reports environmental sound, no vocal element fires, and a liberal gate can still discard a
perfectly good recording.

What protects it is that human imitation keeps vocal-tract structure — formants, F0, harmonic
structure — that the imitated source does not. That evidence comes from the acoustic family, not the
classifier family, which is the concrete reason those two families must stay independent rather than
being folded into a single "content" score.

**Other-speaker speech is expected, not anomalous.** Every pediatric and elicited task has an
examiner or parent speaking by design. Off-target detection therefore cannot be "is there a second
speaker"; it is "is there a voice this protocol does not account for". The anomaly is an unaccounted
voice, and that is a different question from the one the memory-recorded goal states.

## D12 — Onset, offset and span detection, per branch

The three branches need different detectors because their events have different shapes and, more
importantly, different timing tolerances. The tolerance is what drives the machinery, so it is stated
first and the detector chosen to meet it.

| branch | unit | timing tolerance needed | why that number |
| --- | --- | --- | --- |
| speech, syllable repetition | syllable onset | ~10 ms | rate is 5-7 syllables/s; a 50 ms error is a third of a syllable and corrupts the rate |
| airway | event onset and offset | ~50 ms | a cough's explosive phase is ~50 ms; counting needs separation, not precision |
| phonation | phonation onset and offset | ~50 ms, but the offset is definitional | maximum phonation time is a duration, so the offset is the measurement |
| speech, connected | turn and word spans | ~100 ms | word boundaries at conversational rate; alignment supplies these where text is known |

**Airway.** Cough is a transient: broadband energy rise with high-frequency emphasis, detectable by
onset detection on the energy envelope and spectral flux. Countable, so each event carries an onset
and an offset, and bouts need a second level — a bout is several coughs, and reporting a bout as one
event or as N independent events are different clinical claims. Inhalation and exhalation are the
opposite shape: low-amplitude turbulent noise, gradual edges, 0.3-1 s, so hysteresis thresholds on a
band-limited envelope suit them and hard onset detection does not. The protocol supplies a count
prior — `breath-sounds` asks for three deep breaths — and a detected count that disagrees with the
protocol is itself a finding.

**Phonation.** The event is one long steady span, and its offset is the hard part. Phonation at the
end of a breath degrades into creak and irregularity before it stops, so "when did phonation end"
has no single answer: voicing-based offset, amplitude-based offset and regularity-based offset
disagree by hundreds of milliseconds, and maximum phonation time is exactly that duration. The
offset criterion must therefore be named and its alternatives reported, not silently chosen. Glides
are spans whose interior is an F0 trajectory rather than sub-events; `loudness` is two short events
whose comparison, not whose timing, is the measurement.

**Speech.** Syllable repetition is the tightest requirement in the workflow: plosive bursts at
5-7/s, needing ~10 ms onset precision, which no window-level classifier can supply and which
forced alignment can if a reference is available. Word and turn spans come from alignment where the
hint supplies text and from ASR word times where it does not — with the standing caveat that word
absence is not evidence of speech absence.

**Imitation.** Worth testing rather than assuming: a speech extractor may pull vocalic imitation out
of a mixture precisely because it is vocal-tract produced, even though the imitated target is not
vocal. If it does, imitation gets spans from the extracted stream like any other vocal element, and
the classifier's confident "dog" becomes a label on the residual rather than on the child. This is a
measurement to run, and `test-examples.md` records the material to run it on.

## D13 — Inside a branch: detect, attribute, select target, then measure

Every branch runs the same four steps. Only the first and last are branch-specific; the two in the
middle are one shared sub-workflow, parameterised by the element, because the evidence for which
source is the participant does not depend on which element it produced.

1. **Detect element spans** — branch-specific, per D12. Produces candidate spans with element labels
   and their own uncertainty. Everyone's spans, not just the participant's.
2. **Attribute spans to sources** — shared.
3. **Select which source is the target** — shared.
4. **Measure the branch's quantities on target spans** — branch-specific, and the only step that
   produces the clinical numbers.

**Attribution rests on three legs, and their reliability differs by element:**

- *Channel and proximity* — level, spectral tilt and direct-to-reverberant ratio. The participant is
  close-miked and an examiner or bystander is not. Speaker-independent, element-independent, and the
  only leg that works for a cough, so it carries the airway branch almost alone.
- *Speaker identity* — embeddings and clustering. Strong for connected speech, weak for sustained
  vowels, and close to unusable for cough and breath, because the embedders are trained on speech.
  The branch that most needs attribution has the weakest version of this leg.
- *Protocol structure* — the prompt precedes the response, the expected count and duration are known
  from the task. Available only with a hint, and it is the leg that can catch the case the other two
  agree on and get wrong.

**Selection without an anchor is a prior, and must be reported as one.** With a target embedding
supplied, selection is direct. Without it, the target is the dominant close-miked source — a prior
that is usually right and sometimes exactly wrong, which is why it is an `Estimate` carrying the
possibility of being wrong rather than a label. On the airway and phonation branches the participant
may produce very little speech, so the loudest and most speech-like voice in the file can easily be
the examiner; dominance alone would invert the answer precisely where the branch is not speech.

**The common case is cheap and should stay cheap.** Most protocol recordings hold one participant and
possibly one examiner. When the channel evidence shows a single source, attribution and selection
collapse to a confirmation, and the expensive clustering runs only where the evidence says more than
one source is present. That gating is a measurement, not a configuration flag.

## D14 — Measure per source; assign the target later, or not at all

Amends D13, whose ordering was wrong. The four steps become:

1. **Detect element spans** — branch-specific.
2. **Attribute spans to sources** — shared. Sources are unnamed clusters, not roles.
3. **Measure per source** — the branch's quantities computed independently for every source.
4. **Assign the target** — a separate, later decision that *selects among* results already computed,
   and is allowed to return "unresolved".

**Why this ordering and not D13's.** Target selection without an anchor is a prior that is sometimes
exactly wrong, and it is most wrong on the airway and phonation branches where the participant speaks
little and the examiner talks throughout. Under D13 that error is unrecoverable: maximum phonation
time gets computed for the examiner and the participant's is never calculated. Measuring per source
first makes the target decision non-destructive — a wrong assignment can be corrected by re-selecting,
because every source's numbers already exist. It is also the repo's own L1-measures / L2-decides rule:
target assignment is a decision and must not sit upstream of the measurements it conditions.

**Unresolved is a valid outcome.** With no anchor, no hint and two plausible sources, the honest
output is per-source measurements plus the statement that the target could not be determined and why.
That reaches the review flag as a reason, rather than being resolved by a prior nobody can audit.

**It also makes the unaccounted-voice question answerable.** With per-source measurements in hand, a
voice this protocol does not account for is a source that no expected role explains — which is a
comparison over results, not a detection problem, and it needs no target to have been chosen.

**Cost.** Work multiplies by the number of sources, which is one or two in nearly every protocol
recording, and the single-source path is a confirmation rather than a clustering. The multiplication
falls exactly where the ambiguity is, which is where the compute belongs.

## D15 — This is a new graph; the current pipeline is evidence, not a template

Nothing in `audio_analysis` constrains this design. Where the existing code is cited in these
decisions it is as measured evidence of a failure mode — the word gate nulling non-lexical
vocalization, the source map discarding the classifier's own labels, an axis reading zero doubt from
diarizers that were never compared — not as a structure to be preserved or refactored into.

Pieces are built for the triage graph as it is designed here. Whether any current module is reused is
decided per node, on merit, after the node's ports are declared.

## D16 — The vocal-versus-lexical discriminator, from two raw posteriors

Supersedes an earlier draft that rested on `pyannote/segmentation-3.0`, which senselab dropped in
commit `5dd416f0` and which a probe reached only through a dead constant. Re-measured on currently
supported models.

**The count gate is pyannote alone, and only to two voices.** Seed-17 probe, recorded in
`model_registry.yaml:168-280`: exact speaker-count accuracy at k=1 is 100% for pyannote, 0% for
Sortformer (0/20) and MOSS, 50% for child-adult, unrecorded for VibeVoice and DiariZen. Pyannote holds
85% at k=2 and falls to ≤45% for k≥3. Two caveats travel with those numbers: the corpus is
TTS-composed with no room acoustics or channel variation, 20 sessions per k, and the raw
`profile.json` is **not in the repository** — the per-k figures exist only as summaries transcribed
from a cluster-side artifact and cannot be recomputed from a checkout.

**The discriminator is `speaker-diarization-community-1`'s raw posterior against Brouhaha's raw
posterior.** Measured per event on the probe recording:

| event | Brouhaha VAD (raw mean) | community-1 segmentation, max over speakers (raw mean) |
| --- | --- | --- |
| breath ~2.28 s | 0.0049 | **0.0000** |
| breath ~5.31 s | 0.0055 | **0.0000** |
| silence [0, 2] | 0.0068 | 0.0000 |
| cough @ 7.924 s | 0.0053 | 0.574 |
| cough @ 9.609 s | 0.0085 | 0.906 |
| utterance [11.5, 13.3] | 0.689 | 0.790 |

Three classes separate, not two:

| | community-1 responds | community-1 silent |
| --- | --- | --- |
| **Brouhaha responds** | lexical speech | unexpected — flag if seen |
| **Brouhaha silent** | **voiced non-lexical vocalization** | breath, or no voice |

Breath is invisible to **both**, so it cannot come from this pair at all — it needs the DSP envelope
and HeAR route, which is consistent with breath's own measurements.

Brouhaha is not missing the coughs for lack of energy: its SNR head reads 9.37 and 13.31 dB on them
against 2.50 dB for silence. Its VAD head is declining to call them speech, which is a semantic
decision, not a sensitivity limit. That is what makes the contrast trustworthy.

**Use raw posteriors, never thresholded segments.** community-1's segment list reports the coughs as
two clean speech spans with no hesitancy. The raw posterior underneath is *partial* — mean 0.69, 20%
of frames below 0.5, only 3.4% above 0.99 — against 78.5% above 0.99 in the real utterance. The
thresholded view flattens a graded response into a binary claim and destroys exactly the distinction
the discriminator depends on. It also corrects the earlier draft's magnitude: a saturated 1.0 across
the whole cough region overstated what a supported model gives.

Neither model is a boundary detector: the repo records frame-posterior VADs firing through a 0.4 s
inter-turn gap with onset/offset MAE ≈ 2.6 s (`SPEECH_DETECTION_SOTA_REVIEW_2026.md:56-61`). They
contribute presence; edges come from DSP.

**Incidental, and useful for attribution:** community-1 assigns both coughs to `SPEAKER_00`, the same
label as the genuine utterance, and finds one speaker in total. Non-lexical vocalizations therefore
arrive already attributed to the same source as that speaker's speech, which is what D14's
per-source measurement needs.

## D17 — The dedicated pyannote VAD path is broken, independently of gating

`pyannote/voice-activity-detection` cannot be used, for two unrelated reasons.

It is a **config-only repo** whose pipeline delegates to `pyannote/segmentation@Interspeech2021`, and
`pyannote/segmentation` still returns `GatedRepoError: 403` on every file fetch for this account. What
was ungated was the wrapper config, not the weights. On the same account `segmentation-3.0`,
`speaker-diarization-community-1` and `brouhaha` all fetch normally. Note that `api.model_info()`
succeeds on the gated repo, so an access check based on metadata reports success wrongly.

Independently, senselab's own path **cannot succeed even if the gate were lifted**:
`detect_human_voice_activity_in_audios` with that model raises
`ValueError: Revisions must be passed with 'revision' keyword argument` from
`pyannote/audio/core/model.py:573`, because pyannote-audio 4.x rejects the `repo@revision` syntax
outright and the hub config's inner `pyannote/segmentation@Interspeech2021` reference is never
rewritten. `pyannote_vad.py:79` pins the outer pipeline's revision and leaves the inner one as the
config supplies it. The path is untested against a real load.

## D18 — Only pyannote 4.x models are candidates; the pre-4.x ones are dropped, not repaired

pyannote.audio 4.0.4 is installed, and its pipeline classes are `MultiLabelSegmentation`,
`SpeakerDiarization`, `SpeechSeparation` and `VoiceActivityDetection`. Pre-4.x checkpoints are out of
consideration entirely — their READMEs bind them to earlier library versions, and repairing a path to
them is work spent on a model that is not a candidate.

**Kept** — both verified loading under 4.0.4 today:

- `pyannote/speaker-diarization-community-1` — the count gate (D16) and the voiced-vocalization half of
  the discriminator.
- `pyannote/brouhaha` — the speech half of the discriminator, plus SNR and C50. It is a
  `MultiLabelSegmentation` model, which is the shape the taxonomy wants: several labels active at once
  rather than a partition.

**Dropped:**

- `pyannote/voice-activity-detection` — a config-only repo delegating to `pyannote/segmentation`
  (Interspeech2021), which its README binds to pyannote.audio 2.x. **D17's resolution therefore changes
  from repair to removal**: the `@`-revision failure in senselab's dedicated-VAD path is not worth
  fixing, because the destination is not a model we will use. Delete the path.
- `pyannote/segmentation` (2021) and `pyannote/segmentation-3.0` — the latter was already dropped in
  `5dd416f0`; its dead constant and stale docstring in `frame_posteriors.py` are what let a probe
  resurrect it, so they go too.
- `pyannote/speaker-diarization-3.1` — superseded by community-1.

**Two 4.x pipelines worth evaluating that nothing currently uses:**

- `SpeechSeparation` — pyannote's own joint separation and diarization. Given that the configured
  SepFormer enhancer is net-harmful above ≈5 dB SNR, and that a 2-source SepFormer assigns cough
  bursts to whichever stream is free rather than isolating a class, a separation pipeline that is
  jointly trained with diarization is a better-founded candidate than either.
- `MultiLabelSegmentation` — the pipeline class behind Brouhaha, and the natural home for a taxonomy
  that must report overlapping classes. Whether a suitable multi-label checkpoint exists for our
  vocabulary is unmeasured; training one would need the labels we do not have.

**Cleanup this implies**, all pre-existing and now unambiguous:
`voice_activity_detection/pyannote_vad.py` (the whole dedicated path),
`voice_activity_detection/api.py:28,47,66,90` (`_PYANNOTE_VAD_PREFIXES` and its docs),
`voice_activity_detection/frame_posteriors.py:3,14,44,63,113` (dead `SEGMENTATION_MODEL_ID` and the
paragraphs describing a model the same docstring says is no longer loaded), and the registry's
`speaker-diarization-3.1` entry. `speaker_diarization/pyannote.py:140` and `scene_quality/brouhaha.py:34`
mention `segmentation-3.0` descriptively — as community-1's internal segmentation model and Brouhaha's
extractor respectively — which stays true and stays.

## D19 — Enhancement is an operation a node may invoke, not a variant, pathway or dimension

**Supersedes D2.** D2 framed enhancement as a perturbation and made `variant` a dimension the graph is
mapped over, with each task declaring which variants it runs on and variants serving as invariance
probes. That is `perturbations.py`'s model — correct for the existing refiner, and wrong here.

In this design there is no enhanced variant. Nothing carries a `variant` label, no task declares a
variant scope, and no product exists in two versions. Enhancement is an **operation** a specific node
may apply for a specific purpose, whose output is consumed by that node and reaches the rest of the
graph only through that node's declared output ports.

Three consequences:

**There is no prohibition to state.** Earlier notes recorded that enhanced audio "must not reach the
airway branch" as a structural no-port rule. With no variant flowing anywhere the rule is vacuous —
there is no pathway to close. The measured facts behind it remain useful as *tool* properties: an
enhancer that takes breath to −26 or −40 dB is unusable by any node that needs breath, which is a
statement about the tool, not about a route.

**The model comparison is a capability table for tools, not an assignment of models to branches.** If a
node needs a channel in which cough is the only surviving content, `MossFormer2_SS_16K` src1 supplies
one. That is a choice inside a node.

**Invariance stops being free.** D2 treated an answer flipping between raw and enhanced as evidence of
instability the review flag should carry. Without a variant dimension that evidence does not arise as a
by-product; a node that wants it must deliberately run the comparison and declare it as its own
measurement.

Recorded because the same error has now recurred three times — the first flowchart reproduced the
existing call graph, then enhancement appeared as a route inside ADMIT, then as a mapped dimension. The
existing implementation is evidence about failure modes and about what tools do. It is not a source of
structure for this design.
