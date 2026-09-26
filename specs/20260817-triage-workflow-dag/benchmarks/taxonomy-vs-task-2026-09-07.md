# Taxonomy against task — ten b2ai v3.1 recordings, run of 2026-09-07

What was read, and from where. The run is `~/Downloads/triage_10subj_20260907/` (ADMIT → PREPROCESS →
TAXONOMY → figure, config hash `0a6c6915e764adf1`, every ADMIT `PASS`). Each recording's
`run/store.jsonl` was read with `ProvStore.read_jsonl` from this branch; the whole-file summaries
(`yamnet_label_summary`, `ast_label_summary`, `hear_label_summary`), `consensus_taxonomy`,
`consensus_transcript`, the `word` entities, the `span` entities and their `span_yamnet` /
`span_hear` measurements come from the store; per-window scores come from
`run/derivatives/{yamnet,ast,hear}_scores.json`, which the summaries were derived from. The
`<stem>_recording-metadata.json` sidecars were read from the ORCD tree
(`/orcd/data/satra/002/datasets/b2aivoice/post_3.0/v3.1/adult/bids_07_01_v3/`); only three of the
ten are in the local `~/Downloads/b2ai_v31_bids_07_01_v3/` copy. Nothing was run and nothing was
listened to: every "cannot tell" below is a statement about what the store holds.

Two task families and one odd one. Nine sidecars say `story-recall` (grandfather passage, 136
words) or `story-recall-(v2)` (frog story, 271 words); both are recall from memory after reading.
`sub-014813db` is `picture-description-option2`, whose sidecar `stimulus_text` is empty (the
stimulus is a picture; the instruction is "Describe everything that is happening in the picture").
All ten were recorded through the same "USB-C to 3.5mm Headphone Jack Adapter", input gain
0.525–0.798.

Where a number in the brief disagreed with the store, the store is what is written here. The brief
said `sub-01a1f0fd` showed `Engine`/`Rodents` on gap spans: `Rodents, rats, mice` 0.742 is from a
0.239 s *gap* span, but `Engine` 0.569 is from a 0.095 s *amplitude* span. Everything else in the
brief checked out.

## Per-recording table

Subjects are shortened to their first block. `dur` is the recording stream's extent. Peak and median
are over the whole-file windows (YAMNet 0.96 s / 0.48 s hop; AST 10.24 s; HeAR 2 s). `tax` is
`consensus_taxonomy.n_labels` at `consolidation_floor` 0.2. Words are lexical consensus words
(bracketed tokens excluded); wps is words / recording duration.

| subject | task | dur s | LUFS / peak dBFS | YAMNet top-3 (peak, median) | AST top-3 (peak, median) | HeAR top-2 (peak, median) | tax | words / wps | matches stimulus |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 17578482 | story-recall | 25.5 | −31.0 / −13.7 | Speech .99/.78; Insect .90/.00; Wild animals .83/.00 | Speech .54/.52; Hum .21/.18; Mains hum .16/.14 | Speech .77/.20; Snore .25/.01 | 23 | 44 / 1.72 | yes |
| 17cee767 | story-recall | 47.8 | −23.1 / 0.0 | Speech 1.0/.99; Music .20/.00; Mantra .09/.00 | Speech .75/.61; Stomach rumble .28/.01; Busy signal .17/.00 | Snore .69/.01; Speech .68/.31 | 59 | 99 / 2.07 | yes |
| 004d42e9 | story-recall | 88.1 | −15.5 / 0.0 | Speech 1.0/.68; Fart .85/.00; Snoring .81/.00 | Speech .71/.53; Sigh .45/.05; Whoosh .19/.00 | Snore .99/.04; Throat Clear .97/.08 | 102 | 106 / 1.20 | yes (punchline forgotten) |
| 016023f6 | story-recall | 60.3 | −32.5 / −6.2 | Silence 1.0/.00; Speech 1.0/.99; Narration .24/.01 | Speech .82/.79; Female speech .41/.12; Narration .35/.10 | Laugh .97/.30; Throat Clear .93/.03 | 29 | 98 / 1.63 | yes |
| 0032892c | story-recall-(v2) | 76.5 | −26.6 / −5.6 | Silence 1.0/.00; Speech 1.0/.99; Inside, small room .61/.00 | Speech .83/.76; Clicking .79/.19; Sigh .71/.08 | Snore 1.0/.04; Speech .83/.32 | 30 | 173 / 2.26 | yes |
| 00e88aac | story-recall-(v2) | 44.8 | −39.4 / −17.1 | Silence 1.0/.03; Speech 1.0/.63; Inside, small room .21/.00 | Speech .85/.70; Clicking .76/.08; Whispering .36/.08 | Snore .91/.08; Breathe .51/.05 | 21 | 61 / 1.36 | yes (abridged) |
| 00f74092 | story-recall-(v2) | 101.2 | −33.3 / −8.3 | Silence 1.0/.00; Speech 1.0/1.0; Speech synthesizer .30/.00 | Clicking .93/.70; Speech .87/.74; Speech synthesizer .16/.04 | Speech .76/.25; Laugh .55/.06 | 27 | 202 / 2.00 | yes |
| 01a1f0fd | story-recall-(v2) | 114.4 | −20.7 / 0.0 | Speech 1.0/.99; Breathing .50/.00; Inside, small room .40/.00 | Speech .92/.74; Scissors .65/.00; Clicking .47/.03 | Snore .92/.01; Speech .89/.53 | 94 | 203 / 1.78 | yes |
| 1f4ea26f | story-recall-(v2) | 79.5 | −38.1 / −19.1 | Silence 1.0/.00; Speech 1.0/1.0; Conversation .08/.00 | Speech .87/.83; Female speech .51/.26; Clicking .44/.06 | Speech .85/.29; Laugh .66/.15 | 32 | 223 / 2.81 | yes |
| 014813db | picture-description-option2 | 60.9 | −24.5 / 0.0 | Silence 1.0/.00; Speech 1.0/.97; Chewing .82/.00 | Speech .85/.76; Clicking .40/.10; Writing .30/.02 | Throat Clear .93/.06; Speech .90/.19 | 27 | 54 / 0.89 | consistent with a kitchen picture; picture not stored |

Three files reach 0.0 dBFS peak (`17cee767`, `004d42e9`, `01a1f0fd`; `014813db` is −0.01) with
`clipped_runs` 0 in every case — consistent with the tree's known one-isolated-full-scale-sample
per file (`~/Downloads/b2ai_v31_bids_07_01_v3/PROVENANCE.md`), not with clipping.

## 1. Do the top labels match the task?

For seven of ten, the whole-file YAMNet labels with peak ≥ 0.2 are drawn entirely from {`Speech`,
`Silence`, `Inside, small room`, `Narration, monologue`, `Speech synthesizer`, `Breathing`,
`Snoring`, `Snort`} — two to five labels each — and `Speech` has median 0.63–1.0. That is a person
talking in a room with pauses. The three exceptions are 17578482 (16 such labels), 004d42e9 (23)
and 014813db (6, two of them `Chewing`/`Writing` from the last two seconds); all are discussed
below. The expected `Narration, monologue` is weak everywhere: YAMNet peak 0.05–0.24, AST peak
0.03–0.35; YAMNet's `Speech` absorbs it. `Silence` at peak 1.0 with median 0.00 is the pauses, not
the file.

The labels in a top-6 that a person talking in a room does not explain, per recording, with what the
per-window scores say they are:

- **17578482** — YAMNet `Insect` 0.90, `Wild animals` 0.83, `Buzz` 0.81, `Animal` 0.75, `Mains hum`
  0.64 (medians ≤ 0.001); AST `Hum` 0.21 median 0.18 and `Mains hum` 0.16 median 0.14 across all
  three 10.24 s windows. The animal trio all come from one window (10.56–11.52 s: `Insect` 0.90,
  `Wild animals` 0.83, `Buzz` 0.81, plus `Mosquito`, `Cricket`, `Fly, housefly`). `Buzz`,
  `Mains hum`, `Hum` or `Electric shaver` is the top label in every window that is not speech:
  0.48–0.96, 4.80, 8.64, 11.04, 14.88–15.84, 20.16, 22.56–24.96 s. (b) the recording carries a
  continuous electrical buzz that surfaces whenever the speaker pauses and that AST hears under the
  speech; (a) `Insect`/`Wild animals`/`Animal` are the classifier's names for that buzz, not
  animals.
- **004d42e9** — YAMNet `Fart` 0.85, `Snoring` 0.81, `Breathing` 0.80, `Explosion` 0.64,
  `Gunshot, gunfire` 0.58 (top-6 in the figure), plus `Dog` 0.47, `Clang` 0.45 further down.
  `Snoring`/`Breathing` are windows 0.48 s and 40.32 s (0.81/0.80 and 0.33/0.42) — audible
  in-breaths at the hottest microphone of the ten (LUFS −15.5, peak 0.0 dBFS). `Fart` is 2.88 s
  (0.85) and 84.48 s (0.25). `Explosion`/`Gunshot` (83.04 s), `Dog`/`Bow-wow` (84.0 s), `Clang`/`Ding`
  (84.96 s) fall between the consensus words "sentence." (81.44 s) and "I" (83.36 s) … "don't
  remember" (84.39–84.54 s) … "it." (86.76 s), where `Speech` drops to 0.07–0.09. The file also has
  39 `discontinuities`, the most of the ten. (b) there are impulsive transients near the end of the
  recording and breaths at the start; (a) the names are impulsive-sound confusions, not a gun or a
  dog; (c) what the transients are — desk, handling, a door — is not stored.
- **014813db** — YAMNet `Chewing, mastication` 0.82, `Writing` 0.65, `Biting` 0.20; AST `Writing`
  0.30. All from two windows at 56.16–58.56 s, after the last consensus word (57.31 s). (c) a
  short non-speech noise at the end; pen, paper or mouth noise are all plausible and nothing stored
  separates them.
- **0032892c** — AST `Clicking` 0.79 (median 0.19), `Sigh` 0.71, `Gasp` 0.28, `Stomach rumble`
  0.23. `Clicking` is 0.56–0.79 in four of eight windows. (c) see the AST `Clicking` note below.
- **00f74092** — AST `Clicking` 0.93 with median 0.70: ≥ 0.70 in six of ten windows, ≥ 0.10 in all
  ten. YAMNet's whole-file `Clicking` never reaches 0.2. Something click-like runs through the
  whole file at the 10 s scale and only AST reports it. (c) mouth clicks, an operator keyboard and
  a clock are all consistent with what is stored.
- **01a1f0fd** — AST `Scissors` 0.65 (one window, 81.92 s), `Shuffling cards` 0.23 (61.44 s),
  `Clicking` 0.47; YAMNet `Breathing` 0.50 / `Snoring` 0.38 / `Snort` 0.23 in the last window
  (112.8 s, after the final word at 112.87 s and a `[laughter]` token). The transcript at 80.6 s
  reads "I'm distracted by these people outside of this booth": people were audible to the
  participant. In 74.9–86.4 s the YAMNet windows are `Speech` 0.32–1.0 with `Conversation` ≤ 0.02
  and nothing else ≥ 0.05, so (c) whatever the participant heard is not separable in what is
  stored; the paper-like AST labels are the only candidate trace.
- **17cee767** — YAMNet `Music` 0.20 in one window (10.56 s); AST `Stomach rumble` 0.28,
  `Busy signal` 0.17. Nothing to explain; the file is clean at the whole-file level.
- **016023f6, 1f4ea26f, 00e88aac** — nothing outside the talking-in-a-room set. `00e88aac` is the
  quietest (LUFS −39.4) with 44% silence windows; AST `Whispering` 0.36 there is low-level speech.

Two labels recur across files and need naming once. HeAR `Snore` peaks 0.69–1.0 in six of ten
(median 0.01–0.08); its peak spans are 0.12–3.7 s and several are transcribed speech spans
(0032892c: `Snore` 0.978 from a 3.716 s span that carries consensus words). HeAR is an eight-label
health model; on voiced low-frequency speech it fires `Snore`, and on ten files of speech that peak
says nothing. AST `Clicking` ≥ 0.4 appears in six of ten and is the one label only AST reports; it
is discussed under (c) above and is unresolved.

## 2. Which recordings carry something the task does not predict?

The anchor holds: **17578482** is the only file with a continuous non-speech signature. It has 16
YAMNet whole-file labels ≥ 0.2 against 2–6 for eight others; AST shows `Hum`/`Mains hum` in all
three windows; the buzz is the top label in every pause. Nothing comparable elsewhere:

- Engine / vehicle: no file has `Engine`, `Vehicle` or `Motorcycle` ≥ 0.2 in any whole-file YAMNet
  window, and none in AST. In **17cee767** the consensus taxonomy carries `Engine` 0.688,
  `Vehicle` 0.627, `Idling` 0.534, `Motorcycle` 0.486, `Medium engine` 0.463, `Motor vehicle
  (road)` 0.378 — every one from a single 0.083 s gap span that was frame-filled. In **01a1f0fd**
  `Engine` 0.569 is a 0.095 s amplitude span and `Rodents` 0.742 / `Patter` 0.674 / `Mouse` 0.404 a
  0.239 s gap span, both filled. Confined to sub-frame spans; they never reach the whole-file
  summary.
- Animal: 17578482's `Insect`/`Animal` is the buzz; 004d42e9's `Dog` 0.47 is one transient window.
- Music: whole-file `Music` ≥ 0.2 only in 17cee767 (0.201, one window) and 17578482 (0.26, the
  final window after speech ends). No music in any file.
- Other speakers: YAMNet `Conversation` peaks ≤ 0.075 and AST `Conversation` ≤ 0.166 (016023f6)
  across the ten; `Child speech` ≤ 0.06. The only evidence of a second voice is 01a1f0fd's own
  sentence about people outside the booth, and the classifiers do not show it.
- Transients: **004d42e9** is the second outlier (23 whole-file labels ≥ 0.2), all impulsive
  labels from 81.6–85.9 s plus two breath windows; a property of the recording, not the task.

## 3. Does the transcript corroborate the labels?

| subject | words | dur s | wps | speaking wps (first→last word) | silence windows | YAMNet Speech median / windows ≥ 0.5 | HeAR Speech median | ASR word counts (CW / Qwen), agreements |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1f4ea26f | 223 | 79.5 | 2.81 | 2.99 (0.7→75.4) | 10% | 1.00 / 147 of 165 | 0.29 | 225 / 213, 211 |
| 0032892c | 173 | 76.5 | 2.26 | 2.34 (0.3→74.2) | 20% | 0.99 / 119 of 159 | 0.32 | 174 / 168, 165 |
| 17cee767 | 99 | 47.8 | 2.07 | 2.18 (0.8→46.2) | 0% | 0.99 / 94 of 99 | 0.31 | 108 / 98, 97 |
| 00f74092 | 202 | 101.2 | 2.00 | 2.06 (1.3→99.3) | 15% | 1.00 / 171 of 210 | 0.25 | 203 / 200, 197 |
| 01a1f0fd | 203 | 114.4 | 1.78 | 1.82 (1.5→112.9) | 0% | 0.99 / 220 of 238 | 0.53 | 212 / 201, 194 |
| 17578482 | 44 | 25.5 | 1.72 | 2.15 (2.3→22.8) | 0% | 0.78 / 32 of 53 | 0.20 | 44 / 42, 39 |
| 016023f6 | 98 | 60.3 | 1.63 | 1.66 (0.3→59.2) | 1% | 0.99 / 121 of 125 | 0.48 | 105 / 99, 99 |
| 00e88aac | 61 | 44.8 | 1.36 | 1.49 (2.3→43.3) | 44% | 0.63 / 47 of 93 | 0.11 | 61 / 61, 58 |
| 004d42e9 | 106 | 88.1 | 1.20 | 1.22 (0.2→87.0) | 0% | 0.68 / 112 of 183 | 0.04 | 115 / 105, 101 |
| 014813db | 54 | 60.9 | 0.89 | 0.94 (0.0→57.3) | 23% | 0.97 / 92 of 126 | 0.19 | 38 / 50, 30 |

Six files sit at 1.6–2.8 wps, the ordinary range for narrative speech, with the two ASRs agreeing on
≥ 92% of words. The four low ones separate:

- **17578482 (44 words, 25.5 s)** — not a failed decode. The two ASRs return 44 and 42 words and
  agree on 39; speaking rate between first and last word is 2.15 wps; the transcript is a coherent
  four-sentence summary that hits five of the passage's elements. The sidecar's
  `recording_duration` is 25.5 s, so the recording was stopped there. A brief participant, not a
  quiet one.
- **004d42e9 (106 words, 88 s)** — slow and disfluent, not failed. Nine `[UM]`/`[UH]` tokens,
  meta-commentary ("Now what happens next", "there was a puncher for the last sentence. I don't
  remember it"), ASRs agree on 101 of ~110 words. YAMNet `Speech` median 0.68 and HeAR `Speech`
  median 0.04 are the lowest of the story-recalls: the speech is intermittent and, at this gain,
  the between-word windows are full of breath and transient labels rather than silence.
- **00e88aac (61 words, 44.8 s)** — a quiet participant giving an abridged recall: 44% silence
  windows, LUFS −39.4, both ASRs return exactly 61 words. Consistent across all three signals.
- **014813db (54 words, 60.9 s)** — the one case where the labels and the transcript disagree.
  YAMNet says speech in 92 of 126 windows (median 0.97) — roughly 70% of the file is speech — yet
  the consensus holds 54 words, under one per second. The sources disagree: CrisperWhisper 38
  words, Qwen 50, only 30 agreements and 20 insertions, and the CrisperWhisper text repeats the
  phrase "looking over the dishes that they're foaming over the s-" twice, which is a decoder
  loop, not something the participant necessarily said twice. So the labels are right that speech
  is present most of the time; the decode is unreliable; and whether the participant is halting
  (the `clea- cleaning`, `s- this` fragments and "to complete the sentence / The what" suggest word
  finding trouble) or the decoders failed cannot be separated from the store. Both, on this
  evidence. The task label is not wrong (see §4).

## 4. Does the transcript match the stimulus?

Every file is a response to its named task. The check is qualitative — which stimulus elements the
transcript contains — because no overlap statistic is computed anywhere in the run.

- **Grandfather passage** (`story-recall`): 17578482 — long flowing beard, walks except in winter,
  respect, cut back on smoking, "banana oil". 17cee767 — ninety-three, black coat missing buttons,
  long beard, respect, plays the organ, walks except winter, told to stop smoking, "banana oil".
  004d42e9 — ninety-three, dresses himself, buttons loose on his coat, walks except ice or snow,
  respect; says the last sentence was a punchline and does not recall it. 016023f6 — long beard,
  respect, black coat missing buttons, plays the organ, walks in winter (inverted), walk more
  smoke less, "banana something", apologises for reading it only twice.
- **Frog story** (`story-recall-(v2)`): 0032892c, 00f74092, 01a1f0fd and 1f4ea26f each recount the
  whole arc — frog in a jar escapes through the window, boy and dog search, dog's head in the jar
  (00f74092), calling for the frog, branches that are deer antlers, the fall off the cliff into
  water, the log, the frog with its mother and babies. 00e88aac recounts an abridged arc (escape,
  search, fall, log, mother frog and babies) and skips the deer.
- **Picture description** (014813db): "The man is cleaning or drying dishes … the other one is
  watching him … dishes that they're foaming over". A kitchen scene with one figure at a sink that
  is overflowing and another watching is what a picture-description transcript should look like;
  the picture itself is not in the sidecar, so the match to *this* option cannot be confirmed from
  what is stored, but the transcript is not a story recall and the task label is not wrong.

No transcript is unrelated to its prompt, so no file is mislabelled or misattached. One small
sidecar-versus-store discrepancy: `recording_duration` differs from the stream extent by 0.0–2.2 s
(1f4ea26f: 77.3 s in the sidecar, 79.5 s in the store); the others are within 0.9 s.

## 5. Is `consensus_taxonomy` doing its job?

It is not, and the reason is structural rather than a threshold.

**What it reads.** `_write_consensus_taxonomy` in `nodes/taxonomy.py` consolidates the *per-span*
`span_yamnet` and `span_hear` `raw_scores` — the max over spans per label, dropped when that max is
under 0.2 — and never reads the whole-file windows or AST. A short span is prepared by
`span_yamnet_input` (`tasks/classification/yamnet.py`): anything under 0.96 s is centred in a
0.96 s buffer and the remainder is the span's own samples *repeated periodically* on both sides,
recorded as `frame_filled`. Survivors per recording: 21, 23, 27, 27, 29, 30, 32, 59, 94, 102.

**Where the survivors come from.** Tracing each YAMNet row back to the span that produced its peak:

| subject | YAMNet rows | from a span < 0.96 s (filled) | from a span < 0.3 s | from a gap span | whole-file YAMNet labels ≥ 0.2 |
| --- | --- | --- | --- | --- | --- |
| 0032892c | 24 | 23 | 22 | 1 | 3 |
| 004d42e9 | 95 | 88 | 69 | 51 | 23 |
| 00e88aac | 16 | 16 | 15 | 4 | 3 |
| 00f74092 | 21 | 20 | 20 | 4 | 4 |
| 014813db | 20 | 19 | 15 | 2 | 6 |
| 016023f6 | 23 | 22 | 21 | 1 | 3 |
| 01a1f0fd | 88 | 84 | 63 | 42 | 5 |
| 17578482 | 17 | 12 | 10 | 6 | 16 |
| 17cee767 | 52 | 49 | 42 | 31 | 2 |
| 1f4ea26f | 28 | 27 | 16 | 1 | 2 |
| **total** | **384** | **360 (94%)** | **293 (76%)** | **143** | |

Tiling a 50–250 ms fragment produces a signal periodic at 4–20 Hz with a harmonic comb, and the
labels that survive are the labels for periodic machinery and tones: `Synthesizer` in nine of ten
taxonomies (0.28–0.89), `Keyboard (musical)` in seven, `Buzzer`, `Sine wave`, `Tick-tock`,
`Ratchet, pawl`, `Mechanisms`, `Sewing machine`, `Effects unit`, `Engine`/`Idling`/`Motorcycle`.
The most confident rows in the two "busiest" files are all of this kind: 17cee767 `Toothbrush`
0.965 and `Electric toothbrush` 0.943 from one 0.207 s span; 014813db `Stomach rumble` 0.971 from a
0.126 s span; 004d42e9 `Animal` 0.916 from a 0.207 s gap span and `Wind` 0.896 from 0.388 s;
01a1f0fd `Spray` 0.930 from 0.349 s. `hear-yamnet.md` in this directory already ruled "YAMNet must
not be fed a padded span" after measuring a cough padded to 0.96 s read as `Laughter`; periodic
filling is a different fill with the same defect — the model classifies an input that never
occurred — and the consensus is built almost entirely from such inputs. The count of survivors
(21–102) tracks how many short spans happened to tile into something periodic, not what is in the
file: 004d42e9 and 01a1f0fd have 102 and 94 rows because 51 and 42 of them come from gap spans.

**The one real thing is there, but buried.** In 17578482 the buzz survives — `Mains hum` 0.945,
`Hum` 0.903, `Buzz` 0.682, `Electric shaver` 0.416, `Insect` 0.351 — and the `Buzz`/`Electric
shaver`/`Insect` peaks come from a 2.285 s *unfilled* span, so those three are honest. But `Mains
hum` 0.945 and `Hum` 0.903 come from 0.097 s and 0.146 s filled spans (whole-file peaks 0.639 and
0.612), and above `Buzz` in the ranking sit `Synthesizer` 0.870, `Keyboard (musical)` 0.823,
`Noise` 0.807, `Music` 0.768 and `Pulse` 0.721, none of which the whole-file windows or AST
support. A reader of the taxonomy alone would name this a music file with hum.

**Obviously-present labels that are missing.** `Inside, small room` is in the whole-file YAMNet
top-6 of seven files and survives in five taxonomies. `Narration, monologue` never survives.
Everything AST alone reports — `Clicking` 0.93 in 00f74092, `Female speech, woman speaking` 0.51
in 1f4ea26f and 0.41 in 016023f6, `Narration, monologue` 0.30–0.35 — is absent by construction
because `classifiers` is `["hear", "yamnet"]`.

**Obviously-absent labels that survive.** `Frog` (00e88aac 0.288, 004d42e9 0.309, 17cee767 0.230 —
the story is about a frog, the label is from 0.06–0.21 s spans), `Toothbrush`, `Camera`,
`Single-lens reflex camera`, `Owl`, `Crow`, `Horse`, `Chicken, rooster`, `Livestock`, `Siren`,
`Police car (siren)`, `Telephone`, `Printer`, `Rain`, `Lawn mower`, `Explosion`, `Gunshot`.

**Two further properties of the consolidation.** First, `n_classifiers` is 2 only for `Speech` (ten
of ten) and once for `Sneeze` (016023f6: HeAR 0.411, YAMNet 0.251). The two vocabularies do not
share names — HeAR `Snore` / YAMNet `Snoring`, `Breathe` / `Breathing`, `Laugh` / `Laughter`,
`Throat Clear` / `Throat clearing` — so 0032892c, 004d42e9 and 01a1f0fd each list `Snore` and
`Snoring` as separate rows, and cross-classifier agreement is structurally impossible for anything
but `Speech`, `Cough` and `Sneeze`. Second, the taxonomy includes spans that carry consensus words
(0032892c `Snore` 0.978 from a 3.716 s transcribed span; 004d42e9 `Throat Clear` 0.934 from a
1.007 s transcribed span; 17cee767 `Toothbrush` from a transcribed span), while TAXONOMY's own
`airway` lines exclude every transcribed span on the stated ground that ASR outranks both
classifiers. The same node applies the rule to one product and not the other.

## What this says about the screening rules

Routing decides on content plus hint, leniently. This is ten files that all pass, so it measures
what the labels can say about a positive, not what they would say about a negative.

- The whole-file summaries are legible content evidence. In seven of ten, the labels ≥ 0.2 name a
  person talking in a room and nothing else; of the three exceptions, two are the files a human
  would also flag — a continuous buzz and a run of transients — and the third is two seconds of
  noise after the last word. The count of whole-file YAMNet labels ≥ 0.2 (2–6 against 16 and 23)
  separated the first two here; two positives is not a threshold, but it is
  the right kind of number, and it is not stored anywhere today.
- The consensus taxonomy, as built, cannot carry any screening weight: 94% of its rows are
  artefacts of periodic filling, its row count tracks span count, its peaks are inflated relative
  to the whole-file windows, and it omits the one classifier (AST) that reported the two persistent
  labels nobody else saw. Any rule reading it would route 17cee767 (`Toothbrush` 0.965, `Engine`
  0.688) before it routed the buzz file.
- The content check that works today is transcript against `stimulus_text`: ten of ten consistent,
  including the one file (014813db) whose decode is poor. It is not run by the pipeline — the
  sidecar is not read — and it is the cheapest check available for a task whose prompt is a fixed
  text.
- Leniency is right for all ten. The buzz file yields 44 words agreed by two ASRs; the transient
  file 106; the poor-decode file still says what its picture shows. Nothing here should have been
  discarded at screening, and nothing in the stored labels argues otherwise once the taxonomy is
  set aside.
- HeAR at the whole-file level adds nothing on speech: `Snore` peaks 0.69–1.0 in six files of
  people talking. Its value, if any, is on non-transcribed spans, which is what the airway line
  already restricts it to.

## Still unmeasured

- Nothing was listened to. Every (c) above — AST `Clicking` in 00f74092 and 0032892c, `Scissors`
  in 01a1f0fd and whether the people outside the booth are audible, `Chewing`/`Writing` at the tail
  of 014813db, the identity of 004d42e9's transients, and whether 014813db is halting speech or a
  failed decode — is one listen away.
- There is no negative. No file without speech, with a second speaker, with music, with a
  television, or of the wrong task. The whole-file label set has a measured sensitivity of ten out
  of ten and no measured specificity.
- The tiling effect is inferred from the label pattern, not measured directly: the same short spans
  classified in their natural 0.96 s context versus tiled would settle it in one pass.
  `hear-yamnet.md` measured zero-padding, not periodic filling.
- No alternative consolidation was computed in the store. The whole-file-at-0.2 column above is the
  only comparison, and it is not the same thing as consolidating over unfilled spans.
- Transcript-to-stimulus agreement is a reading, not a number. Word overlap or an alignment score
  against `stimulus_text` would make §4 reproducible.
- The `picture-description-option2` picture is not in any sidecar; the transcript's match to it is
  plausibility, not verification.
