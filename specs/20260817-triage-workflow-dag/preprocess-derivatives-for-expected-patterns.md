# PREPROCESS derivatives for the expected-pattern table

A review of [`expected-patterns.md`](expected-patterns.md) from the measurement side: for each
expected pattern in that table, what shared measurement would have to exist for a branch to match
it, and which of those measurements PREPROCESS does not produce today.

Read with [`preprocess.md`](preprocess.md), which is what PREPROCESS is *documented* to write.
Every claim below was checked against `src/senselab/audio/workflows/triage/nodes/preprocess.py` at
this branch's tip rather than against that document, and where the two disagree the divergences are
named in the last section — several of them change what an implementer would build.

**Scope.** PREPROCESS is the node that measures once on the waveform and shares its products with
every branch. A measurement that only one branch can use, and that needs no audio, is not a
PREPROCESS derivative and is not proposed here even where it is missing. Three things are proposed.
Everything else the table asks for is either already in the store, or is arithmetic over something
already in the store, or is a branch-local decision that no derivative will supply.

---

## 1. The inventory this was checked against

The block list is read off the code (`preprocess.py:2945`), not off `preprocess.md`:

`clip_spans`, `yamnet_scores`, `yamnet_windows`, `silence`, `ast_scores`, `ast_windows`,
`hear_scores`, `hear_windows`, `level`, `disruptions_file`, `asr_crisperwhisper`, `asr_qwen`,
`consensus_transcript`, `phonation_tracks`, `energy_envelope`, `normalized_envelope`,
`spectrogram_wideband`, `spectrogram_narrowband`, `continuity_trace`, `spans`, `residual`,
`enhanced_yamnet`, `enhanced_ast`, `enhanced_hear`, `residual_yamnet`, `residual_ast`,
`residual_hear`, `squim`, `span_hear`, `span_yamnet`, `gammatone`, `ppg_posteriorgram`,
`praat_features`, and one `<stream>_diarization` block per entry in `diarization.streams`.

The per-frame and per-sample arrays a branch can slice over any extent it later invents:

| array | grid | signal | written at |
| --- | --- | --- | --- |
| `energy_envelope` (+ its global floor) | per sample, dBFS, 40 Hz lowpass | pre-emphasised | `preprocess.py:1612` |
| `normalized_envelope` | per sample, dBFS | AGC'd pre-emphasised | `:1705` |
| `continuity_trace` | per sample, `[0, 1]` | narrowband spectrogram of pre-emphasised | `:2549` |
| `spectrogram_wideband` / `_narrowband` | 5 ms hop, 5 / 20 ms window | pre-emphasised | `:2507` |
| `gammatone` | 5 ms hop, 40 ERB channels 80–7800 Hz | pre-emphasised | `:2584` |
| `phonation_tracks` — `f0_hz`, `strength`, `f1..f4_hz`, `f1..f4_bw_hz` | 10 ms hop | F0 on pre-emphasised, formants on `plain` | `:1300` |
| `ppg_posteriorgram` — 40 ARPAbet | per frame | `enhanced` | `:1090` |
| `consensus` `word` entities — `text`, `bracketed`, `extent`, per-source `timings`, `agreement` | per word | `plain` | `:2460` |
| `span` entities — `measure ∈ {amplitude, continuity, asr, gap}` | per span | various | `:1875`–`:1911` |
| `<stream>_diarization` segments | per turn | `enhanced`, `residual` | `:2617` |

`residual.enabled` is **`true`** in the shipped config (`default.yaml:342`), so `enhanced`,
`residual`, their six classifier passes and both diarizations run by default. `preprocess.md` says
"off by default" for all of these; it is stale, and it matters, because `ppg_posteriorgram` and
`praat_features` both read `enhanced` and would be absent if it were not.

---

## 2. The derivatives that are missing

Three. Ranked by how many expected patterns they unblock.

### D1 — `stimulus_alignment`: the consensus word stream aligned against the declared utterance

**Built.** `src/senselab/audio/workflows/triage/stimulus.py` and the `stimulus_alignment`
block in `nodes/preprocess.py`; design, corpus measurements and the one config key in
[`stimulus-alignment.md`](stimulus-alignment.md). Two claims below were wrong when checked
against the 4.0-release adult tree and are corrected there: `prolonged-vowel` and `loudness`
carry an empty `stimulus_text` (their count-in and token are prescribed in `instructions`), and
CAPE-V carries **one** sentence per recording rather than six.

**What it measures.** One alignment between the `consensus_transcript`'s `word` stream and the
ordered tokens the recording was declared to expect (`AudioHints.expected_speech`). Per expected
token: which consensus column realised it, that column's extent, its `outcome` and its `agreement`,
or that nothing realised it. Per consensus word: whether it matched an expected token at all. Where
the expectation has internal structure — six CAPE-V sentences, fifteen Stroop colours, a Harvard
sentence — the alignment yields that structure's boundaries as spans.

**Which patterns require it.**

| pattern | families | n |
| --- | --- | --- |
| ordered, fully-specified read text | `harvard-sentences-list`, `cape-v-sentences`, `-v2`, `rainbow-passage`, `caterpillar-passage` | 18,793 |
| ordered answer sequence, not the displayed words | `word-color-stroop` | 472 |
| ordered short-token pattern | `prolonged-vowel` (the `one two three` count-in), `loudness`, `loudness-v2` | 3,206 |
| verbatim-echo anti-pattern, read inverted | `free-speech` v1, `story-recall`, `-v2` | 4,623 |
| single shared prompt | `open-response-questions` | 199 |
| per-sentence extents that must reach **VOICE**, not SPEECH | `cape-v-sentences`, `-v2` | 3,594 (counted above) |
| the complement — lexical material where none was expected | every AIRWAY family; every non-`buttercup` `SYLLABLE_REPETITION` family | ~19,400 |

**Why an existing derivative will not serve.** `consensus_transcript` is the transcript. Nothing in
the store relates it to what was declared; `expected_speech` reaches every branch as a hint and no
shipped code reads it. The decisive argument for putting the comparison in PREPROCESS rather than in
SPEECH is that its products are wanted by three different branches at once: SPEECH wants the diff,
**VOICE** wants CAPE-V's per-sentence boundaries as selectable spans (the table marks this "owed a
code change — they must reach VOICE as selectable spans"), and **AIRWAY** wants the complement —
the extents of lexical material in a recording where none was expected, which is its `off_task`
finding. Branches do not read each other. If SPEECH computes this, the other two do without it.

**Cost.** No model pass, no audio. Sequence alignment of two token lists, plus a read of the word
entities that already exist. This is the cheapest of the three by a wide margin.

**Already implemented in senselab?** The aligner, yes: `harmonize_transcripts`
(`src/senselab/audio/workflows/audio_analysis/harmonize.py:522`) and `align_sources`
(`src/senselab/audio/workflows/triage/consensus.py:276`) already align token streams under a cost
model, for the ASR-against-ASR case. Aligning consensus-against-stimulus is the same operation with
one side carrying no timings. `align_transcriptions`
(`src/senselab/audio/tasks/forced_alignment/forced_alignment.py:691`) is the *acoustic* variant and
is a different, heavier instrument; it is what the table's omission problem refers to, and it is not
what this derivative needs.

**Two preconditions that are not part of the derivative and must not be folded into it.**
`expected_speech` is unpopulated today — `runs/b2ai-v2/make_hints.py` parses the BIDS `task-` token
and never reads `stimulus_text`. And `stimulus_text` is empty on 36 of the 48 families, 33,430
recordings, so for those the expectation has to come from a human-read table that does not exist.
The derivative is worth building before either is fixed — the twelve families that do carry
`stimulus_text` are 29,117 recordings and include the largest family in the corpus — but it will be
silent on the rest, and that silence is an absent input, not an absent measurement.

---

### D2 — voice-quality tracks on the existing analysis hop: HNR, short-time RMS, per-frame CPPS

**What it measures.** Three more columns on the grid `phonation_tracks` already writes: harmonics-
to-noise ratio in dB, short-time RMS, and smoothed cepstral peak prominence, one value per 10 ms
frame, over `plain`. Not a new entity — three arrays added to
`derivatives/phonation_tracks.npz` (`preprocess.py:1300`), beside `f0_hz`, `strength` and the four
formant tracks it already carries.

**Which patterns require it.** Every acoustic voice row, and the effort half of two airway rows:

| pattern | families | n |
| --- | --- | --- |
| sustained phonation, measured over the vowel and not the file | `prolonged-vowel`, `maximum-phonation-time`, `-v2` | 5,113 |
| glide, measured over the sweep | `glides-low-to-high`, `-high-to-low`, `high-to-low` | 3,193 |
| voice quality **per sentence**, where pooling discards the instrument's design | `cape-v-sentences`, `-v2` | 3,594 |
| effort, in its level-invariant half | `loudness`, `loudness-v2`, `respiration-and-cough-v2-hardcough`, `voluntary-cough` | 2,627 |
| VOICE on families that declare no voice task | all | 22,277 routed |

**Why an existing derivative will not serve.** `praat_features` is 40 whole-file scalars
(`preprocess.py:1179`). That is the worked example's central defect and it cannot be repaired by
arithmetic: a mean taken over the whole file cannot be re-pooled over the vowel alone. On a
`prolonged-vowel` recording `mean_hnr_db` and `mean_cpp` average in the count-in's consonants and
the silences; on a CAPE-V recording one number stands for six sentences that were chosen to load six
*different* phonatory conditions. `phonation_tracks` carries F0 and formants but no noise and no
amplitude measure, so there is nothing in the store from which HNR, jitter, shimmer or CPPS over an
arbitrary extent can be recovered.

Tracks rather than per-span scalars, deliberately. A `span_praat` block shaped like `span_hear`
(`preprocess.py:2186`) would lock the scalars to PREPROCESS's own span boundaries, which are
amplitude/continuity/ASR/gap and are not the task extent for any family here. A track is poolable
over whatever extent a branch later proposes or refines — which is the entire point of the
decomposition the table is arguing for.

**Cost.** One additional Praat harmonicity pass over the whole file, plus an array that is already
computed and thrown away. No model, CPU only. **The config keys already exist and are already
non-null**: `phonation.hop_s`, `phonation.silence_threshold`, `phonation.periods_per_window`
(`default.yaml:141-144`), whose own comment calls them "Praat's harmonicity and pitch settings for
the whole-stream F0/formant tracks". Nothing new is owed a derivation.

**Already implemented in senselab?** All three, and the first two are *already being computed on the
whole file inside a branch*:

- `hnr_track` (`src/senselab/audio/tasks/phonation/api.py:108`) — one `to_harmonicity_cc` call over
  the whole stream. **VOICE calls it at `voice.py:267`**, over the whole `plain` stream, and
  `_rms_track` at `:274` likewise; both are then sliced per phonation span and the slices
  concatenated into `derivatives/voice_tracks.npz` (`:366`). Since the phonation-span detector was
  retired, VOICE takes the no-span early return at `voice.py:235` **before** either call, on every
  recording measured. So the graph contains a whole-file HNR pass that is written for no recording
  and readable by no other branch. Moving it to PREPROCESS is a relocation, and it is the relocation
  the node's own contract already implies: a measurement that answers a whole-file question belongs
  here and is not to be re-run by a later node.
- Per-frame CPPS — `extract_cpp_descriptors`
  (`src/senselab/audio/tasks/features_extraction/praat_parselmouth.py:936`) computes `prominence`
  as a per-frame array at `:1030` and immediately reduces it to mean, std and a frame count at
  `:1032`. The smoothed power cepstrogram behind it (`:899`) is the expensive part and is already
  paid for. Returning the series is the whole change.

**Two signal-path constraints that have to travel with this derivative.**

1. **It must be computed on `plain`.** `_smoothed_power_cepstrogram` pre-emphasises internally
   (`praat_parselmouth.py:912`), so handing it the `preemphasised` stream pre-emphasises twice and
   moves the low-quefrency trend line the prominence is measured against. `hnr_track`'s
   autocorrelation is likewise a periodicity measure that should not be taken on a first-difference
   filtered signal.
2. **Today's forty scalars are measured on the denoiser's output.** `praat_features` resolves the
   `enhanced` stream (`preprocess.py:1206`) — the lag-aligned FRCRN output. Jitter, shimmer, HNR and
   CPPS are all measures of the harmonic-to-noise structure of the signal, and FRCRN's entire
   function is to alter that structure. Whatever those scalars are measuring, part of it is the
   enhancement model. This is not an argument for deleting them; it is an argument that the tracks
   proposed here must be taken on `plain`, and that no consumer should read a `praat_features`
   scalar and a `phonation_tracks` value as two views of one signal. They are not: F0 is on
   `preemphasised`, formants on `plain`, and the scalars on `enhanced` — three signals for one
   voice.

---

### D3 — `band_profile`: the recording's own spectral band, measured before the resample

**What it measures.** On the `recording` stream as supplied, at its own rate: the frequency below
which 95% of the energy sits, and a coarse long-term average spectrum. One number and one short
vector.

**Which patterns require it.**

| pattern | families | n |
| --- | --- | --- |
| nasal vs oral route (A7) | `respiration-and-cough-fivebreaths` (both routes by index), `-threequickbreaths`, `-v2-breath`, `-v2-threebreaths`, `-v2-threebreathsnose`, `-v2-threebreathsmouth`, `breath-sounds` | 8,416, of which 4,974 sit in a declared route contrast |
| the recording-hygiene clause — an occluded microphone is a level and spectral-tilt finding | `respiration-and-cough-v2-hardcough` | 698 |
| any AIRWAY conclusion whose validity depends on the input's band | all airway families | ~13,000 |

**Why an existing derivative will not serve.** Everything spectral in the store is computed after
the resample to 16 kHz, so nothing can see above 8 kHz and nothing records whether the file had
content there. `disruptions_file` is the only derivative taken on the un-resampled `recording`
stream and it is clipping runs and zero-crossing rate. ADMIT records the container's declared
`sampling_rate` on the recording stream (`admit.py:96-103`), which is not the content's band: an
8 kHz-sourced file stored at 48 kHz declares 48 kHz. `preprocess.md` already asserts that "a
narrowband input with a 4 kHz ceiling restricts what the airway branch can conclude" — and nothing
in the graph measures whether the input is one.

**Cost.** One STFT on the original file, once. No model.

**Already implemented in senselab?** The core was: `_rolloff_hz`
(`src/senselab/audio/workflows/audio_analysis/quality.py:156`) is a cumulative-energy quantile over
a `torch.stft`, about forty lines. It was a private function in a sibling workflow, not a `tasks/`
capability.

**BUILT.** Promoted to `senselab/audio/tasks/band_profile/` rather than lifted, and
`audio_analysis` now calls it too, so the two workflows report one statistic; the rewire is
bit-identical over 72 cases, so no cache bump is owed. PREPROCESS writes a `band_profile`
measurement on the `recording` stream with a `band_profile.npz` sidecar carrying the LTAS. The
decisions and what was verified before building are in [band-profile-d3.md](band-profile-d3.md).

**Ranked third, and honestly.** This does not by itself make the route measurable — see §4.1. What
it buys is that a negative A7 result becomes *attributable*: a route contrast that fails on files
whose content stops at 4 kHz has failed for a reason that can be written down, rather than being
recorded as "route is not measurable" for the whole corpus.

---

## 3. The patterns that are already fully served

This is where an implementer should start. In every row below, the measurement exists in the store
today; what is owed is branch code, and in some rows an operating point — the same thing owed for
every row of the table. **No new derivative.**

| pattern | what serves it | families | n |
| --- | --- | --- | --- |
| **lexical presence and extent** | consensus `word` entities + `span` `measure: "asr"` | `picture-description` ×3, `open-response-questions`, `cinderella-story`, `free-speech-v2`, `productive-vocabulary` | 7,078 |
| **lexical material where none was expected** | the same, as a complement | every AIRWAY family; every non-`buttercup` `SYLLABLE_REPETITION` family | ~19,400 |
| **the count-in / vowel split** | consensus word extents locate `one two three`; the vowel is the voiced production under no lexical word, bounded by `spans` | `prolonged-vowel` | 1,604 |
| **sustained vs moving F0 — V1's "stationarity"** | `phonation_tracks.f0_hz` + `strength` at 10 ms, and `continuity_trace` per sample | `prolonged-vowel`, `maximum-phonation-time`, `-v2`, and VOICE's 22,277 routings | 5,113 + 22,277 |
| **glide direction — V3** | the sign of the dominant monotone segment of `phonation_tracks.f0_hz` over the span | `glides-low-to-high`, `-high-to-low`, `high-to-low` | 3,193 |
| **within-recording loudness contrast — V6** | `level` + `energy_envelope`, differenced across the two word extents | `loudness-v2` | 705 |
| **repeat counting** | a counter over normalised consensus tokens | `diadochokinesis-buttercup`, `-v2-buttercup`, `animal-fluency`, `random-item-generation` ×2 | 2,265 |
| **DDK rate — D1** | the modulation spectrum of `energy_envelope` over the train | all ten DDK families | 7,989 |
| **DDK inter-onset intervals — D3, and airway event onsets — A5/A6** | envelope-derivative peaks inside a span, over `energy_envelope` | all DDK + every airway family declaring a count | ~17,800 |
| **cough presence** | `span_hear` `Cough` / `span_yamnet` cough subtree, joined on `span_id` | `respiration-and-cough-cough`, `-v2-hardcough`, `voluntary-cough` | 2,813 |
| **breath presence** | `hear_windows` / `span_hear` `Breathe` | `respiration-and-cough-breath`, `-v2-breath`, `breath-sounds` | 2,813 |
| **vocal tremor — V4a** | the 2–12 Hz band of the spectrum of `f0_hz`, and of `energy_envelope`; the 40 Hz envelope lowpass passes the whole band | every sustained family | 5,113 |
| **connected-speech pause and breath-group structure — S4** | inter-word gaps from consensus `timings`; breath positions from `[BREATH]` bracketed `word` entities and HeAR `Breathe` | passages, picture description, free speech, story recall, fluency | ~13,900 |
| **more than one voice in the recording** | `enhanced_diarization` / `residual_diarization` — shipped, one measurement per stream | all | 62,547 |
| **material matching nothing** | `span` `measure: "gap"` | all | 62,547 |
| **declared against measured duration** | the sidecar's `recording_duration` against the stream extent | all | 62,547 |

Three of these deserve a sentence, because the table marks them owed.

**"F0 stationarity, formant stationarity, spectral flux" is called "the single most load-bearing gap
in this document". It is not a gap.** `spectral_continuity`
(`src/senselab/audio/tasks/spectral_continuity/api.py:10`) is the cosine similarity between
consecutive log-magnitude spectra — a spectral-stationarity trace by construction, high through a
held vowel and through a glide's slowly-moving harmonic structure, dipping at every plosive and
every onset. It is computed over the narrowband spectrogram and written per sample as
`continuity_trace` (`preprocess.py:2549`). F0 and formant stationarity are statistics of `f0_hz` and
`f1..f4_hz`, which are in the store at a 10 ms hop. What is owed is a *named statistic and a
window* — a decision, in `data/` with a derivation — not an estimator. Nothing here needs building.

**A5/A6 event boundaries need no new derivative, and the mechanism named in the table is not the
one that will bite.** `spans.min_separation_ms` is 30 ms (`default.yaml:39`); volitional coughs are
seconds apart and are not merged by it. The real merging case is a cough series produced on a single
exhalation, where the envelope never falls back within `k_db` of the floor between bursts and the
whole series is one span. That is visible as multiple maxima in `energy_envelope` inside one span,
and separating them is arithmetic over an array the store already holds. The counting defect the
table names — `by_label` incrementing once per (span, label) pair at `airway.py:280` — is real and
is branch code.

**Repetition counting does not need `transcript_repeat` moved into the store.** It is a counter over
normalised consensus tokens; `routing_analysis/features.py` having its own copy is a convenience,
not the source of the capability.

---

## 4. Detection approaches in the table that will not work as written

### 4.1 Nasal vs oral route from `gammatone`

> "Route is A7 and **may not be measurable**; the `gammatone` 40-channel energy is the only spectral
> instrument that could carry it."

Two objections, one small and one that decides the row.

**Gammatone is not privileged, and calling it "the only instrument" misdirects the work.** The
filterbank is an ERB rebinning of the same short-time spectral information already in
`spectrogram_narrowband` and `spectrogram_wideband`; it carries no frequency content they lack. If
route is unrecoverable from a 40-channel ERB profile it is unrecoverable from the spectrograms too,
and vice versa. The constraint is the band and the geometry, not the filterbank.

**The band is cut where the discrimination lives, and the geometry confound is not controllable by
this design.** Oral breath noise is turbulence generated at the lips, teeth and tongue tip and is
broad and high; nasal breath noise is generated in a narrower, more damped passage and is quieter
and lower. Much of what separates them sits above the 8 kHz ceiling the 16 kHz working rate imposes,
and what remains below it is a spectral-tilt difference — which is confounded, one for one, with
mouth-to-microphone distance and angle. And the confound is not incidental here: **a participant
instructed to breathe through the mouth points the mouth at the phone, and one instructed to breathe
through the nose with the mouth closed does not.** The route and the geometry change together by
construction. The 1,778 / 1,778 within-session `fivebreaths` split is therefore not the clean
contrast the table takes it for — it is perfectly balanced on route and perfectly confounded on
source-to-microphone transfer. A classifier fitted on it will separate the two conditions, and what
it will have learned is unidentifiable from the recordings alone.

Recommendation: build D3 first so a null result is attributable to a measured band limit, report the
route question as *not separable by this design* rather than as *not yet fitted*, and do not treat
`fivebreaths` as validation-grade for A7.

### 4.2 D6 sequence conformance from the PPG posteriorgram

The table already notes the posteriorgram is out of domain on rapid nonsense repetition, and marks
the per-span query "owed a code change". Both are right, and the conclusion should be stronger: the
right instrument is already in the store and is cheaper.

`/p/`, `/t/` and `/k/` differ in burst spectrum in the textbook way — `/t/` high-frequency dominant,
`/k/` a compact mid-frequency peak, `/p/` diffuse and falling — and all three of those are
comfortably inside the 8 kHz band. `spectrogram_wideband` is a 5 ms window at a 5 ms hop, which is
the classical resolution for exactly this measurement, and `gammatone` at a 5 ms hop gives the same
thing pre-pooled. A per-span PPG query would spend a model pass to recover less: a phonetic
posteriorgram is trained on connected speech, and on a rapid nonsense CV train with no lexical
context its acoustic-model prior is actively working against the discrimination, smearing adjacent
frames toward whatever phoneme sequence the training distribution favours. Prefer the array that is
already written.

### 4.3 Syllable-nucleus rate (D2) via Praat's speech rate

`extract_speech_rate` (`praat_parselmouth.py:160`) is de Jong & Wempe's nucleus counter and it is
*already running* inside `praat_features` on every recording. It will not carry DDK, for two reasons
both internal to the method:

- A candidate intensity peak is only counted if the dip to the next peak exceeds `min_dip` (2 or
  4 dB, chosen by a whole-file HNR test). In a fast `/pʌ/` train with weak bilabial closure the
  inter-syllable intensity dip is frequently smaller than that, so the fastest trains are the ones
  most likely to be under-counted.
- A peak must be voiced *and* inside a "sounding" interval, where the silence tier is built with
  `min_pause = 0.3 s`. That is longer than an entire DDK syllable cycle.

Both failures are correlated with the quantity DDK exists to measure — rate — so the bias is not
noise, it is signal-dependent. D1's envelope modulation spectrum over `energy_envelope` has neither
failure mode, needs no new derivative, and answers the same question. Use it.

### 4.4 Absolute effort from `level`

Rows for `respiration-and-cough-v2-hardcough`, `voluntary-cough` and `loudness` v1 want to know
whether a production was maximal. `level` is peak dBFS, RMS dBFS and LUFS of an uncalibrated
consumer recording with unknown microphone sensitivity, unknown source-to-microphone distance and,
on many handsets, automatic gain control inside the capture path. There is no SPL reference anywhere
in the graph and none can be recovered after the fact. Any absolute effort statement from `level` is
a statement about the recording chain.

What survives without calibration is the level-invariant half: vocal and expiratory effort change
the *spectral balance* of the source largely independently of gain, and that is what D2's per-frame
tracks plus the existing spectrograms measure. The table is right that `loudness-v2` and
`voluntary-cough` are fine — both carry a within-recording contrast. For `loudness` v1 and
`v2-hardcough` the honest output is a measurement with its covariates, never a `hard` / `not hard`
verdict, and the row should say so rather than marking effort "owed a measurement" as though a
future estimator would settle it.

### 4.5 `residual` `energy_fraction` as the breath-presence reading

The `respiration-and-cough-breath` / `-v2-breath` row (2,487 recordings) is marked "file-level
presence is **implementable today**: the `airway.breath` gate reads `residual` `energy_fraction`".
The gate exists — `[residual, energy_fraction] >= 0.10` (`default.yaml:285-288`) — but it does not
measure what the row needs it to.

`residual = plain − g·FRCRN(plain)`, and FRCRN is a *speech* enhancer. PREPROCESS records
`speech_present` on the residual measurement precisely because the subtraction only means
"background" where there was speech to separate background from — and on a comfortable-breathing
recording there is none, which is the whole point of the family. What `energy_fraction` is high for
is "FRCRN removed most of this signal", i.e. *this is not speech*. A cough-only file, a glide, room
noise and a near-silent file all satisfy that. As a routing gate, "not speech" may be adequate; as
the row's claimed presence measurement it is not, because it does not distinguish breathing from any
other non-speech content.

The half of the row that stands is HeAR's `Breathe` label, which is a positive detection of the
thing being asked about. Drop the residual half rather than marking the row implementable on it.

### 4.6 A smaller one: `declared_duration_s` against measured

Several rows call this free, and it is. It is worth keeping straight that it is a *sidecar
consistency* check, not a task-completion measurement — the document itself reports 2,020 sidecars
declaring under a second. A file that runs 13 s against a declared 73 s (the `breath-sounds` relax
period) tells you the declaration and the recording disagree; which of the two is wrong is a
separate question.

---

## 5. What is missing but should **not** become a PREPROCESS derivative

Named so the scope stays closed.

| what | why not here |
| --- | --- |
| category membership of a produced word (`animal-fluency`, `random-item-generation`) | no waveform, one consumer (SPEECH), 667 recordings. A lexicon or a text embedding, branch-local. `text/tasks/embeddings_extraction` exists and is not wired into triage |
| semantic coverage for `story-recall` | n-gram overlap against the source story answers the question the instruction actually poses — *recalled or read* — and verbatim echo is an n-gram phenomenon. Given D1 it is a bag-of-tokens count. A semantic-coverage measure would need a cut nobody has fitted, to answer a question nobody asked |
| per-span PPG | a code change on an existing derivative (`extract_ppg_segments`, `tasks/features_extraction/ppg.py:349`, called from `routing_analysis` and never from a node), and §4.2 argues the per-span query is the wrong instrument for the row that wants it |
| Praat scalars over an extent | superseded by D2. Tracks are re-poolable over any extent; scalars over PREPROCESS's spans are not |
| `transcript_repeat` into the store | arithmetic over word entities already in the store |
| a dedicated VAD | there is no dedicated-VAD backend in senselab — `detect_human_voice_activity_in_audios` relabels diarization segments (`tasks/voice_activity_detection/api.py:27`, `:36-40`), and the whole-file diarization the relabelling would read is now a PREPROCESS derivative. Nothing is owed |

---

## 6. Where `preprocess.md` and the code disagree

Found while checking, and each one would mislead an implementer building from the document.

| `preprocess.md` says | the code does | consequence |
| --- | --- | --- |
| `spans`: "`floor(t)` = rolling 10th percentile of `energy_envelope`, 3 s window"; "`propose` = peaks where envelope − floor ≥ K, minimum separation 150 ms"; "`onset` = walk back from the peak to peak − 15 dB"; "`offset` = walk forward to peak − 0.7·(peak − floor)" | one **global** floor, the 5th percentile of the whole envelope (`envelope/api.py:187`, `default.yaml:33`); candidates are **threshold-crossing runs**, not peaks later expanded (`spans/api.py:42`); onset and offset walk by the **identical** `k_db` rule in opposite directions; `min_separation_ms` is **30**, not 150 | anyone designing A5/A6 event counting against the document is designing against a peak-anchored detector with a local floor that the code deliberately replaced, and the replacement has a written derivation |
| `residual` and the six `{enhanced,residual}` classifier blocks are "off by default" | `residual.enabled: true` (`default.yaml:342`) | `enhanced`, `residual`, six classifier passes and both diarizations run on every recording; `praat_features` and `ppg_posteriorgram` depend on `enhanced` existing |
| the `diarization` row is written as owed / settled-in-docs; `dag.md:526-527` says "`diariz` and `pyannote` appear nowhere in `nodes/preprocess.py`" | `_diarization` is a shipped block, one per entry in `diarization.streams` (`preprocess.py:2617`, `:2945`), and `diarization_streams` defaults to `[enhanced, residual]` | the derivative exists; `dag.md` is behind the code |
| the derivative table does not list `normalized_envelope` or `continuity_trace` as derivatives in their own right | both are written with provenance and their own npz (`preprocess.py:1705`, `:2549`) | two per-sample arrays a branch may read are invisible to a reader of the table |

---

## 7. Which one to build first

**D1, `stimulus_alignment`.** It unblocks more expected patterns than the other two together
(roughly 27,000 recordings positively, and the lexical complement on another 19,000), it costs no
model pass and no audio, and the aligner it needs already exists in the tree for the adjacent case.
It is also the only one of the three whose absence forces *duplicated* work: SPEECH, VOICE and
AIRWAY each need a different projection of the same alignment, and branches cannot read each other,
so leaving it out means either three implementations or two branches doing without.

D2 second: it is nearly free — the whole-file HNR pass is already being computed inside VOICE and
discarded on every recording, and the per-frame CPPS array is already computed and reduced away —
and it repairs the defect at the centre of the worked example, that every voice-quality number in
the store is taken over the whole file including the count-in and the silence.

D3 third, and with its limits stated rather than as a route detector.
