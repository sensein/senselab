# Hints and routing on real b2ai audio — run of 2026-09-15

What a declaration changes today, what the ruleset routed, and what the gates read while doing it.
Thirteen recordings, each run twice. This file holds the numbers; the design files that own each rule
carry a one-paragraph reading of them and point back here.

## Method

Thirteen recordings from `b2ai_v31_bids_07_01_v3`, three subjects, 16 kHz, spanning airway, voice,
speech and DDK material. Each was run through `run_triage` **twice** — once `hint=None`, once with an
`AudioHints` built from the BIDS sidecars — for 26 runs.

**Gate *values* are not in the store.** `route_attributes` (`live_evidence.py:172-190`) records
`state`, `routed`, `gate_outcomes`, `unavailable`, `flags`, `sources` and `stem`; `gate_outcomes` is
`{name: outcome}` and the number each gate compared is nowhere. Every value below was recovered by
re-reducing each finished `store.jsonl` through `extract_features`
(`routing_analysis/features.py:1026`) and `gate_value` / `evaluate_gate`
(`routing_analysis/ruleset.py:379`, `:394`) offline. The counterfactual in finding A was computed the
same way, by replaying `routing._map_tags` and the `will_run` rule (`routing.py:218-220`) over the
stored evaluations.

### Two caveats that govern every number here

**Thirteen recordings across three subjects is not a basis for any threshold.** Every figure below is
an observation on real material. None of it is a fitted value, none of it licenses moving a cut, and
the per-gate separations are reported because they are what a real run produced, not because thirteen
files can locate a boundary. The standing limit in [`README.md`](README.md) applies unchanged.

**The hint extractor used for the hinted arm carried a bug, disclosed by the measuring agent.** A
substring test matched the task-family prefix, so **two breath recordings recorded
`may_contain: [cough, airway]` where the correct tags are `[breath, airway]`**. It changed nothing
downstream — which is itself finding A — but the recorded tag strings for those two rows are wrong and
must not be read as what the sidecars declare.

### The recordings

| id | task | dur s |
| --- | --- | --- |
| A1 | Cough-1 | 6.0 |
| A2 | HardCough | 2.4 |
| A3 | FiveBreaths-1 | 20.5 |
| A4 | ThreeQuickBreaths-1 | 6.9 |
| V1 | Prolonged-vowel | 12.1 |
| V2 | MPT-1 | 20.1 |
| V3 | Glides-Low-to-High | 13.5 |
| S1 | Rainbow-Passage | 28.2 |
| S2 | Harvard-49-1 | 4.0 |
| S3 | Story-recall | 47.7 |
| S4 | Picture-description | 29.1 |
| D1 | DDK-buttercup | 6.0 |
| D2 | DDK-puhtuhkuh | 5.1 |

No full gate × recording matrix is recorded, because only the values named in the per-gate sections
below were recovered. A matrix with empty cells would read as a measurement that declined rather than
one that was never taken.

---

## A — A hint changes nothing today, and would change no route even if the map were populated

Across all 13 pairs, in the structured result **and** a full entity-by-entity diff of the two stores
(ids, `mtime_ns` and `config_hash` normalised away), **the only difference anywhere** is
`branch_decision.unmapped_tags` going from `[]` to the declared tag list on each of four decisions.
Derivative checksums byte-identical. `forced_by_hint` false on **52/52** decisions. No gate value, no
`ruleset_routing` attribute, no TAXONOMY output, no branch verdict, no file verdict and no triage
outcome differs.

Where a hint reaches, each checked against the tree:

| consumer | what it does with the hint |
| --- | --- |
| `nodes/taxonomy.py` | **never reads it** — `taxonomy.py:336` says so in the signature's own docstring. Confirmed empirically: `consensus_taxonomy`, all three label summaries and the node verdict are identical in all 13 pairs |
| `nodes/routing.py` | the only real consumer. `_declared_tags` collects `may_contain` and `metadata["speech_type"]` (`routing.py:79-82`); `_map_tags` resolves them against `routing.hint_branch_map`, which is `null` (`default.yaml:127`), so every tag is unmapped, `forced_by_hint` is always False (`routing.py:219`) and the tags land in `unmapped_tags` (`routing.py:233`) |
| `nodes/voice.py` | reads `metadata["population"]` (`voice.py:66`) and `metadata["task"]` (`voice.py:146`), but `voice.f0_range_by_population` (`default.yaml:183`) and `voice.task_duration_ranges` (`:185`) are both null, so both paths return before the declaration is used |
| `nodes/speech.py` | `speech.py:536` appends the flag *"this branch identifies the target by enrollment, not by `hint.target_speaker`, which was supplied and is not read"*. Never fired in this run — no arm supplied a `target_speaker` |
| `admit.py`, `preprocess.py`, `airway.py`, `quality.py` | accept and discard, each saying so in its own docstring (`admit.py:56`, `preprocess.py:1103`, `airway.py:179`, `quality.py:225`; QUALITY `del hint` at `:238`) |

`AudioHints.targeted_speaker_count`, `environment` and `expected_speech`
(`audio_hints.py:150`, `:151`, `:152`) are **read by nothing in `src/senselab`** — the only occurrence
of `targeted_speaker_count` outside the data structure is a docstring at `speech.py:492` saying it is
not read as evidence. Two runs carried verbatim Rainbow and Harvard prompts; SPEECH transcribed them
correctly and nothing compared the transcript against the prompt.

### The counterfactual — a populated map would have forced nothing on these 13

Replaying `_map_tags` and the `will_run` rule (`routing.py:218-220`) offline over the stored
evaluations with the map populated as
`cough→AIRWAY, sustained-vowel→VOICE, read-speech→SPEECH, ddk→DDK, read→SPEECH`, **zero branches
would be forced on any of the 13**: the content ruleset had already routed every declared branch. The
only unmapped tag left would be `non-lexical`, which names no branch.

**This is evidence *for* content-first routing, not an argument that the map is unnecessary.** Thirteen
recordings say nothing about the recordings where content and declaration disagree, and that
population is precisely what a forcing map exists for. What the run does settle is that the map's
absence is not what is costing these files a route.

### The metadata contract and the readers disagree

`voice.py` reads `hint.metadata["task"]` and `hint.metadata["population"]`.
[`../../20260913-branch-contract-and-hints/design.md:355-362`](../../20260913-branch-contract-and-hints/design.md)
specifies the `metadata` contract as `task_name`, `acoustic_task_name`, `speech_type`,
`declared_duration_s`, `microphone`, `channels`, `sample_rate`. **Neither key `voice.py` reads is in
that contract**, so a hint written to spec reaches neither consumer. `speech_type` — ROUTING's key
(`routing.py:42`, read at `:80`) — is the one read that the contract does cover.

**Owed a code change**, and the contract is the side to change: `task_name` and `acoustic_task_name`
are the sidecar's own field names and are per-recording and per-family respectively, which is the
distinction `voice.py`'s single `task` cannot express, and `population` is not a declaration the
sidecars carry at all. Deciding which key `voice.py:146` should read is deciding which grain a
duration range is keyed at — that is the contract's question, not the branch's. Recorded as owed
against the contract in that spec; the reading of V7 against the contract's frozen entry keys is
already in [`../branch-voice.md`](../branch-voice.md) *Unresolved*.

### The sidecar shape, which constrains any future hint builder

The `*_acoustictask-metadata.json` is **per task family, not per recording** — one file serves all 11
breath/cough wavs — so it alone cannot build a per-recording hint. Measured across all **112**
recordings of the local corpus copy:

- `speech_type` and `language` **never disagree** between the acoustictask and recording sidecars.
- `stimulus_text` **disagrees on 44**: empty in the acoustictask JSON for Harvard, Cape-V,
  Productive-Vocabulary and Stroop, family-generic for Free-speech, while the recording JSON carries
  the actual prompt.
- corpus `speech_type`: `non-lexical` 59, `read` 30, `elicited` 19, `recall` 4 — the same four values
  [`open.md`](open.md) § *Hints* already enumerates, now with counts.
- `recording_profile_name`: `Speech` 87, `Breathe` 20, `Cough` 5.

So a builder that reads only the family sidecar drops the prompt on 44 of 112 recordings, and the
field `AudioHints.expected_speech` exists to hold is exactly the field that disagrees between grains.

---

## B — `FileVerdict.hints` states a falsehood when the map is null, and it renders

With `routing.hint_branch_map: null`, no decision carries `hint_tags`, so `_hint_claims` returns `{}`
rather than `None` (`nodes/verdict.py:182-184`). `fold_file_verdict` then builds the `hints` table
(`vocabulary.py:350-357`) through `_hint_reading(claimed=False, …)` (`vocabulary.py:279-291`), which
writes `found_unclaimed` or `no_claim` for **every** branch.

On the four runs handed `may_contain: [cough, airway]`, the file verdict therefore says
**`AIRWAY: found_unclaimed`** — an assertion that nobody claimed AIRWAY, on a run where the
declaration claimed it. It propagates to `summary.json`'s `recording.declared_hints`
(`report.py:1097`) and onto the PDF header (`report.py:1426-1427`, rendered at `:1467`), verified in
a released summary.

`UNREAD_DECLARATION` (`vocabulary.py:385-386`) exists for exactly this situation and does not fire:
`_hint_claims` returns `None` only when `hint is not None and not decisions` (`nodes/verdict.py:182`),
i.e. when ROUTING left no decision at all.

**This is a defect, not an owed derivation**, and it is recorded against the verdict in
[`../verdict.md`](../verdict.md). **Owed a code change**, blocked on a vocabulary decision the owner
holds: `verdict.md`'s product pins four `hints` tokens, and what the field should say when no map is
configured — a fifth token such as `unresolvable`, or `hints: {}` plus the existing flag — is not a
measurement. The upstream causes are registered at
[`../../20260913-branch-contract-and-hints/design.md:288-311`](../../20260913-branch-contract-and-hints/design.md);
this downstream consequence on the verdict table and the rendered report was registered nowhere before
this run.

---

## C — Routing recall was 13/13; precision is what generates flags

Every declared branch was reached: **AIRWAY 4/4, VOICE 3/3, SPEECH 4/4, DDK 2/2**. Every whole-file
`RouteState` was `ROUTED` — no recording read `empty` and none read `unexplained`.

Over-routing, on the same 13: **VOICE on 3 speech tasks, AIRWAY on 4 non-airway tasks, DDK on 3 pure
speech tasks.**

File outcomes: **5 `pass`, 8 `flag`.** No `discard`.

Recall is not the axis under strain here, which is the opposite of what the corpus sweep's
sensitivity figures foreground ([`../family-taxonomy-ruleset.md`](../family-taxonomy-ruleset.md)
scored 0.97 / 0.95 / 0.96 / 0.94). On this material every flag traces to a branch that ran on
material it has no subject in (D), a branch with no node (F), or a route taken on a label that is
wrong (G).

### Every gate's value on every recording

Recorded because nothing in the store keeps it: `route_attributes` stores `gate_outcomes` but not
the values (`../../../src/senselab/audio/workflows/triage/live_evidence.py:172-190`), so these were
recovered by re-reducing each finished store through `extract_features` + `gate_value`. They are the
evidence under findings C through G, and without them each of those findings rests on a number
nobody can re-read.

`*` = fired. `na` = **unavailable**, meaning the feature key was absent and the gate was never
judged — which is not the same as a gate that read a value and declined, and must never be folded
into one. Thresholds are in the header row.

| rec | sp.lexical ≥2 | sp.agree ≥3 | vo.sustained ≥3.0 | vo.glide ≥.05 | vo.chant ≥.02 | ai.breath ≥.10 | ai.cough ≥50 | ai.bracket ≥1 | ai.ppg_silent ≥.757 | ddk.lex ≥3 | ddk.ppg ≥10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A1 | 0 | 0 | 0.63 | .013 | .011 | .010 | 43.10 | 0 | **0.949\*** | 0 | 0.49 |
| A2 | 0 | 0 | 0.80 | .0002 | .00005 | .0008 | na | **1\*** | 0.699 | 1 | 4.47 |
| A3 | 0 | 0 | 1.80 | .006 | .003 | **0.938\*** | na | 0 | 0.750 | 0 | 3.55 |
| A4 | 0 | 0 | 1.03 | .0009 | .0007 | **0.927\*** | na | 0 | **0.984\*** | 0 | 0.72 |
| V1 | **3\*** | **3\*** | **4.39\*** | **.364\*** | **.161\*** | **0.249\*** | na | 0 | 0.456 | 2 | 4.46 |
| V2 | 1 | 0 | **15.89\*** | **.909\*** | **.909\*** | **0.238\*** | na | 0 | 0.131 | 1 | 3.78 |
| V3 | 1 | 0 | **12.27\*** | **.659\*** | **.659\*** | .004 | na | 0 | 0.733 | 1 | 0.74 |
| S1 | **68\*** | **67\*** | 2.57 | .030 | **.0221\*** | .007 | na | 0 | 0.180 | **5\*** | 8.89 |
| S2 | **10\*** | **10\*** | 2.81 | 0 | 0 | .00001 | na | 0 | 0.238 | 2 | 7.10 |
| S3 | **98\*** | **97\*** | 2.71 | **.088\*** | **.086\*** | **0.252\*** | na | 0 | 0.190 | **4\*** | 8.64 |
| S4 | **53\*** | **51\*** | **4.68\*** | **.112\*** | **.088\*** | .013 | na | **1\*** | 0.400 | **8\*** | 7.44 |
| D1 | **11\*** | **10\*** | 1.51 | .001 | .001 | .005 | na | 0 | 0.145 | **11\*** | **12.33\*** |
| D2 | **20\*** | 0 | 1.28 | .00001 | 0 | 0 | na | 0 | 0.102 | **8\*** | **14.70\*** |

**Every cell is identical in the hinted run**, which is finding A restated as data.

Three readings this table supports and the prose above does not:

- **`voice.sustained` has a value on all 13** — 0.63 to 15.89 s — so VOICE is routed by a
  measurement that exists on every recording, and then fails for want of a `phonation` span that
  exists on none. It is the clearest statement of the inconsistency finding D describes.
- **`ai.cough` is `na` on 12 of 13.** The one value, 43.10 dB against a 50.0 cut, is finding E.
- **`ddk.ppg` separates cleanly** — 12.33 and 14.70 on the two real DDK recordings against a
  maximum of 8.89 across all four speech recordings, a margin of 1.11 /s below the cut on the
  nearest speech recording. `ddk.lex` does not separate at all, which is finding F.

**13 recordings across 3 subjects fits no threshold.** Every number here is an observation on real
material; none is a fitted value, and the corpus sweep is the measurement that would be.

---

## D — VOICE fails on every recording it runs, including true voice material

All **6** VOICE-routed recordings returned `Outcome.FAIL` with
*"no phonation span in the store; the detector that proposed them was retired on 2026-09-04 and VOICE
is pending a rework onto `consensus_taxonomy`"* (`voice.py:235-239`). That includes:

| id | what it holds | `voice.sustained` value (cut 3.0 s) |
| --- | --- | --- |
| V2 | MPT — held phonation | **15.89 s** |
| V3 | glides low to high | **12.27 s** |

VERDICT converts each to `mismatch: routing routed VOICE, it found no subject`
(`vocabulary.py:393-397`). **This is the largest single flag source in the run, 6 of 13**, and it is
the sole reason V3 flags at all.

`default.yaml:181` still ships `voice.hint_tags` marked *"unread as of v2"*, so the branch also
carries a second, dead copy of the hint vocabulary while having no subject to apply it to.

**Owed a code change**, already registered in [`../branch-voice.md`](../branch-voice.md) *The state of
this branch* and [`../dag.md`](../dag.md) § *5c. VOICE*. On this evidence it is the highest-value next
piece of work in the graph: the fix removes 6 of the run's 8 flags' leading reason and is the only one
of these findings that is blocked on nothing but implementation.

---

## E — `airway.cough` never fired, on either cough recording

The gate reads `span_label_set_stats["yamnet.cough_labels.peak_over_floor_db_max"]`
(`default.yaml:269`), which exists only when a live amplitude span carries a YAMNet cough-set label
under the `windows.yamnet` membership rule — top 4 and at or above 0.2 (`default.yaml:93`, `:95`).

- **UNAVAILABLE on 12 of 13.**
- On the one recording where the key existed — **A1, five deliberate coughs** — it read **43.10 dB**
  against the **50.0 dB** cut (`default.yaml:271`) and stayed silent.

[`../family-taxonomy-ruleset.md:141-144`](../family-taxonomy-ruleset.md) already records that cut as
*"carried over onto the conditioned population **unswept**"*. 43.10 dB on five deliberate coughs is a
data point that it is too high for this population. It is one recording and licenses no new value.

**Consequence for the branch.** AIRWAY's route state reads `unavailable` rather than `declined` on
**4 of 13**, so on nearly a third of the sample AIRWAY is effectively ungated — and `unavailable` is
not a negative ([`../routing.md`](../routing.md), [`../branch-airway.md`](../branch-airway.md)
*The number that governs this branch*).

The two cough recordings reached AIRWAY by unrelated gates:

| id | gate that routed it | value | what the gate measures |
| --- | --- | --- | --- |
| A1 | `airway.ppg_silent_fraction` (cut 0.757) | **0.949** | a *silence* measure. The file is 6 s of mostly silence with five bursts |
| A2 | `airway.bracketed_event` (cut 1) | **1** | the ASR emitting `[cough]`; the transcript is literally `'[cough]'` |

Neither is the cough detector. A1 routes AIRWAY because it is mostly quiet and A2 because a
recogniser wrote the word down.

---

## F — DDK: the PPG gate discriminates perfectly; the unmeasured lexical gate causes every false positive

| gate | fired on | value | cut |
| --- | --- | --- | --- |
| `ddk.ppg_segment_rate_per_s` | D1, D2 — and nothing else | **12.33, 14.70** /s | 10 |
| `ddk.ppg_segment_rate_per_s` | the four speech recordings, silent | **8.89, 8.64, 7.44, 7.10** /s | 10 |
| `ddk.lexical_repetition` | S1, S3, S4 | **5, 4, 8** | 3 |
| `ddk.lexical_repetition` | S2 (4.0 s Harvard sentence), silent | **2** | 3 |

The acoustic gate separated the two real DDK recordings from every speech recording perfectly on this
sample: the nearest speech value sits 1.11 /s below the cut, the nearest DDK value 2.33 /s above it,
and the two populations are 3.44 /s apart with the cut inside that gap. Thirteen files do not locate
that boundary, and [`../family-taxonomy-ruleset.md`](../family-taxonomy-ruleset.md) already records
the gate as fitted elsewhere; this is the first time it has been watched on material of both kinds in
one run.

`ddk.lexical_repetition` — threshold 3, marked **UNMEASURED** at `default.yaml:283` — fired on
ordinary function-word repetition in connected speech and is the **sole** gate routing all three false
DDK positives. The only speech recording it missed is the 4.0 s Harvard sentence, and it missed that
one by a single repeat.

### The missing node adds a flag *reason* to 38% of the sample and a flagged *file* to none

DDK routes **5 of 13**. Each is recorded `SKIPPED` with `note: "no node implements this branch"`
(`run.py:302-305`, `NO_NODE` at `:46`), and `fold_file_verdict` raises each to
`"DDK was asked to run and never ran"` (`vocabulary.py:398-401`).

**But it changes no file's triage outcome: all five already carry another flag.** That distinction is
the difference between an artifact and a defect, and it cuts both ways — the flag reason is real and
over-fires on speech (above), and building the node would have changed no file's outcome in this run.
Recorded against the scope claim in [`../dag.md`](../dag.md) § *5d. DDK*.

---

## G — TAXONOMY passes 13/13, and the label evidence under it is noisy on real b2ai audio

TAXONOMY returned `pass` on all 13, and `consensus_taxonomy` was byte-identical across the hinted and
unhinted arms of every pair. What the labels under it say:

**Breath reads as music on one subject and as breath on another.**

| id | task | top consensus labels |
| --- | --- | --- |
| A3 | five deep breaths, 20.5 s | `Music` **0.629**, `Synthesizer` 0.473, `Keyboard (musical)` 0.465, `Wild animals` 0.430; **`Breathing` 12th, at 0.288** |
| A4 | three quick breaths, 6.9 s, another subject | `Breathing` **0.991** |

**A held vowel reads as a musical instrument.** V2, a 20 s held "ah": `Chant` 0.937, `Music` 0.930,
`Mantra` 0.899, `Brass instrument` 0.661 — all outranking anything voice-specific. VOICE routed
correctly on this file, via a musical-instrument confusion.

**HeAR fires indiscriminately on speech.** `Snore` ≥ 0.2 on **8 of 13**, including **0.977** on
continuous narration and **0.991** on V1; `Cough` ≥ 0.2 on **8 of 13**, including six recordings with
no cough in them. With `airway.labels_of_interest: [Cough, Breathe]` (`default.yaml:136`), that means
HeAR hits most speech in the label space AIRWAY acts on — which is how S3 reached AIRWAY and then
flagged `lexical_contamination` (`airway.py:368`, `:373`). This is the same behaviour
[`hear-yamnet.md`](hear-yamnet.md) § *Presence gates are not locators* and
[`taxonomy.md`](taxonomy.md) § *HeAR is barred from speech* measured on one recording, at
corpus-shaped rates.

**`voice.glide` and `voice.chant` return the identical number.** On V2 both read **0.90926** and on V3
both read **0.65875**: the singing-subtree union's peak *is* the bare `Chant` peak on this material.
`../family-taxonomy-ruleset.md` § *`voice.chant`, added* justifies the gate precisely as *"the bare
`Chant` label … **not the singing union that contains it**"*, and on the two recordings where VOICE
matters most the distinction does not exist — two gates of the eleven, one detector.
**Owed a code change**, recorded there.

**AST is computed and summarised but never reaches `consensus_taxonomy`.** `SUMMARISED_CLASSIFIERS`
holds three (`taxonomy.py:42`) and `PER_SPAN_CLASSIFIERS` holds two — `{"yamnet": "span_yamnet",
"hear": "span_hear"}` (`taxonomy.py:99`) — so `ast_label_summary` is written on every run and the
consolidation reads none of it. Whether that is deliberate is **not asserted here**: the exclusion is
consistent with [`taxonomy.md`](taxonomy.md)'s *"AST disagrees usefully, and shares a corpus"* and
with [`open.md`](open.md)'s note that `classify_audios` softmaxes AST scores, and it is recorded
nowhere as a decision. Owed a decision, not a measurement.

---

## H — The label-summary ordering guarantee is not observable from the store

`taxonomy.py:94` sorts `<classifier>_label_summary.labels` by descending peak and `:68` documents that
ordering in the function's `Returns:`. `ProvStore.write_jsonl` serialises every entity with
`json.dumps(..., sort_keys=True)` (`prov_store.py:536`), so the persisted mapping is **alphabetical**.

Harmless today: the only consumer re-sorts by `(peak, median)` before printing (`figure.py:646-650`),
and `consensus_taxonomy.labels` is a list and keeps its rank. Recorded because a reader checking the
docstring's guarantee against a store will find it violated, and because the next consumer that trusts
the order will be wrong on read-back rather than on write.
