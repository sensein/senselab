# Removing the preprocessing-related nulls from the packaged config — a proposal for review

Opened 2026-09-04. **Nothing here is applied.** `default.yaml` is untouched; this document exists to
be reviewed, amended and then implemented.

The owner's ruling: the packaged config should carry **no preprocessing-related nulls**, each value
set from prior work. A second ruling frames it: **no pipeline parameter may be set for a
visualisation's benefit** — a scratch figure tool had been setting three of these keys as
"visualization-only overrides", which is re-processing under another name.

All 47 nulls are attributed below. 24 are read by PREPROCESS or TAXONOMY and are what the ruling
covers; 18 are branch-only; 1 is ROUTING's; **4 are read by nothing at all.**

## How a null actually behaves — the distinction that governs everything below

`config.require` raises on a null exactly as on a missing key; `config.get` returns `None`
(`config.py:92`, `config.py:79`). What happens next differs by node:

- **PREPROCESS absorbs it.** Every derivative runs inside a `try` that catches `ValueError` and
  `LookupError` and records the message in `absent[name]` as "a cascading absence, not a new
  failure" (`preprocess.py:1377-1385`). PREPROCESS then returns `PASS` with
  `why="conditioning complete; absent derivatives are listed"`. **A null silently deletes a
  derivative and the node still passes.**
- **The branches raise.** SPEECH and VOICE `require` the same keys and have no such guard, which is
  why the stage-0 collection recorded `SPEECH was asked to run and errored without a verdict` on all
  112 recordings.

So the 24 keys split again, by whether the null disables work or merely leaves a decision open:

| | keys | effect of the null today |
| --- | --- | --- |
| **Blocking** | `windows.*.default_threshold` (3), `windows.*.label_thresholds` (3), `voice.f0_range_hz`, `phonation_spans.*` (8) | 15 keys. The block does not run. Four PREPROCESS derivatives absent (`yamnet_windows`, `ast_windows`, `hear_windows`, `phonation_tracks`) plus TAXONOMY's phonation pass |
| **Degrading** | `speech.word_gap_ms` | 1 key. `config.get` guarded by `if word_gap_ms is not None` (`preprocess.py:541-548`), so the **ASR span source is silently skipped**. Corroborated by stage-0: of 1,951 spans, 1,858 are `amplitude`, 72 `continuity`, 21 carry no `measure` (signal `recording`) — **not one `asr` span** |
| **Undecided** | `taxonomy.presence_floor.*` (4), `taxonomy.voice_{min,uncertain}_duration_s` (2), `taxonomy.speech_labels`, `words.onomatopoeic_tokens` | 8 keys. All read via `config.get`; the node runs and reports `unavailable`. These are the honest nulls — they cost a decision, not a measurement |

One cascade worth seeing whole: **`voice.f0_range_hz` alone disables the entire voice kind.**
`preprocess.py:1224` requires it for `phonation_tracks`; TAXONOMY proposes phonation spans over those
tracks; the voice line reads those spans. Null → tracks absent → line `unavailable` → voice
`uncertain`. Stage-0 shows exactly that: `voice: phonation=0.0 seconds (floor=—, unavailable)`.

## (A) Read by PREPROCESS or TAXONOMY — 24 keys, the ruling's scope

| key | reader | citation |
| --- | --- | --- |
| `windows.yamnet.default_threshold` | PREPROCESS | `preprocess.py:724` (generic), `:1033` (span_yamnet) |
| `windows.ast.default_threshold` | PREPROCESS | `preprocess.py:724` |
| `windows.hear.default_threshold` | PREPROCESS | `preprocess.py:724`, `:969` (span_hear) |
| `windows.{yamnet,ast,hear}.label_thresholds` | PREPROCESS | `preprocess.py:727` (require), `:971`, `:1035` (get) |
| `speech.word_gap_ms` | **PREPROCESS** + SPEECH | `preprocess.py:541` (get, guarded); `speech.py:117` (require) |
| `voice.f0_range_hz` | **PREPROCESS** + VOICE | `preprocess.py:1224` (require); `voice.py:68` (require) |
| `words.onomatopoeic_tokens` | PREPROCESS | `preprocess.py:1146` (get, `or []`) |
| `phonation_spans.f0_stability_cents` | TAXONOMY | `taxonomy.py:282` |
| `phonation_spans.formant_stability_hz` | TAXONOMY | `taxonomy.py:283` |
| `phonation_spans.glide_min_excursion_cents` | TAXONOMY | `taxonomy.py:284` |
| `phonation_spans.hangover_ms` | TAXONOMY | `taxonomy.py:285` |
| `phonation_spans.voicing_strength_floor` | TAXONOMY | `taxonomy.py:286` |
| `phonation_spans.mixed_voiced_fraction` | TAXONOMY | `taxonomy.py:287` |
| `phonation_spans.unvoiced_max_formant_bandwidth_hz` | TAXONOMY | `taxonomy.py:288` |
| `phonation_spans.word_aligned_min_evidence_fraction` | TAXONOMY | `taxonomy.py:290` |
| `taxonomy.presence_floor.speech.{acoustic,lexical}` | TAXONOMY | `taxonomy.py:516-517` |
| `taxonomy.presence_floor.airway.{health_acoustic,acoustic}` | TAXONOMY | `taxonomy.py:518-519` |
| `taxonomy.voice_min_duration_s` | TAXONOMY | `taxonomy.py:552` |
| `taxonomy.voice_uncertain_duration_s` | TAXONOMY | `taxonomy.py:553` |
| `taxonomy.speech_labels` | TAXONOMY + SPEECH | `taxonomy.py:512`; `speech.py:530` |

Note `phonation_spans.*` is read by TAXONOMY, not PREPROCESS, despite the name — the phonation pass
moved this session. `phonation_spans.{hop_s, max_formants, formant_max_hz, formant_window_s,
formant_preemphasis_hz}` ship real defaults and are read by PREPROCESS (`preprocess.py:1227-1231`);
they are not null and are out of scope.

## (B) Branch-only — 18 keys, no proposal offered

| key(s) | reader | citation |
| --- | --- | --- |
| `airway.contest_labels` | AIRWAY | `airway.py:105` |
| `phonation.hnr_floor_interval_db`, `phonation.rms_floor_interval` | VOICE | `voice.py:188-189` |
| `redaction.padding_ms`, `redaction.fill` | REDACT | `redact.py:51,53,375` |
| `speech.second_diarizer` | SPEECH | `speech.py:626` |
| `speech.target_match_cosine` | SPEECH | `speech.py:130` |
| `speech.enrollment_model` | SPEECH | `speech.py:127` |
| `speech.speech_test_stoi_floor`, `speech.speech_test_si_sdr_floor` | SPEECH | `speech.py:546-547` |
| `speech.separation_backend`, `speech.separation_sound_class` | SPEECH | `speech.py:647-648` |
| `speech.nontarget.{level_db,tilt_db_per_octave,d_to_r_db}` | SPEECH | `speech.py:997` |
| `voice.f0_range_by_population` | VOICE | `voice.py:65` |
| `voice.f0_range_ratio_max` | VOICE | `voice.py:70` |
| `voice.task_duration_ranges` | VOICE | `voice.py:138`, `:251` |

## (C) ROUTING — 1 key

`routing.hint_kind_map`, read at `routing.py:160`. Not preprocessing, but the only null standing
between the corpus's existing hint extractor and a working hinted run: `runs/b2ai-v2/override.yaml`
already carries a complete 12-entry map, and `make_hints.py` parses that file and refuses to emit a
tag absent from it. **Recommend promoting the override's map into the packaged config verbatim** —
it is a reading of the b2ai task registry's vocabulary, not a fit, and `non-lexical` is deliberately
absent because forcing only ever adds a branch and no single kind is right for it.

## (D) Read by nothing — 4 keys, recommend deletion

`quality.stoi_floor`, `quality.pesq_floor`, `quality.disruption_clipped_s_max`,
`quality.disruption_dropout_s_max`.

Not an inference from the key prefix — `dag.md:461` and `dag.md:469-470` already state it: *"all four
declared, all four null, all four read by nothing in the codebase… Verified by search."* I re-ran the
search and confirm it. `src/tests/audio/workflows/triage/config_test.py:82` references
`quality.stoi_floor`, but as a declared-key assertion, not a read.

Pre-alpha says rename and replace outright, no deprecation shims, so these should be **deleted with
the derivations that describe them**. The capability they were declared for — SPEECH step 8's quality
gate, goal 1 — is unimplemented (`dag.md:588`), and re-adding a key when the gate is built costs
nothing. Deleting them also removes four of the 47 without deciding anything.

Counter-argument the owner may prefer: they document an intended gate, and `open.md:118` treats their
nullity as a live debt ("Clipping is measured and never adjudicated"). Keeping them as declared-null
is defensible **provided** they are recorded as not-yet-read rather than as unmeasured thresholds.
The distinction matters because only one of the two states is a measurement debt.

## The three window thresholds — measured on this corpus, not carried forward

`runs/b2ai-v2/override.yaml` sets all three to `0.5`, and is explicit that this is "the score-gap
heuristic, and EXPLICITLY UNVALIDATED as a per-window cut", carried only "because a run needs the
window folds to execute at all". `benchmarks/hear-yamnet.md:47-64` retracts the reasoning behind it:
over b2ai-28, 170 spans produced 14 labels, the winners spanned 0.501–0.821, two sat within 0.005 of
the floor, and **both dedicated cough files got zero labels** — "on top of the data, not in a gap".

Stage-0 supplies the first corpus-wide view. The raw `<classifier>_scores` derivative is written by
its own earlier step with **no threshold applied**, so 113 runs of raw per-window scores are on disk
independent of the null. Measured over them (3,362 YAMNet / 215 AST / 885 HeAR windows):

| | YAMNet | AST | HeAR |
| --- | --- | --- | --- |
| top-1 score, p25 | 0.377 | 0.370 | 0.135 |
| top-1 score, **median** | **0.885** | **0.537** | **0.288** |
| top-1 score, p75 | 0.995 | 0.695 | 0.520 |
| max observed | 1.000 | **0.874** | 1.000 |
| shape | bimodal — **44.4%** of windows top-1 ≥ 0.95 | unimodal, mode **0.50–0.55 (12.1%)** | monotone decay, mode 0.00–0.05 (13.1%) |
| windows retaining ≥1 label at 0.5 | 68.2% | 59.5% | **26.3%** |

Three conclusions, each load-bearing:

1. **The scales are not comparable.** Median top-1 is 0.885 / 0.537 / 0.288. A single scalar shared
   across the three encodes three different strictnesses. The override says as much ("one curve per
   classifier because the three score distributions are not comparable"); these are the numbers.
2. **There is no empty interval in any of the three.** The retraction established this for HeAR; it
   generalises. YAMNet is flat at 2–3% per 0.05 bin from 0.45 to 0.90; AST's densest bin *is*
   0.50–0.55; HeAR decays monotonically. No gap-based argument is available for any of them.
3. **0.5 costs HeAR most of its airway material.** Against the 25 stage-0 recordings the protocol
   *declares* are cough or breath — a check against a declaration, not a fit, used only to select a
   subset whose content is not in doubt — the peak target-label score clears:

   | threshold | declared-airway recordings firing |
   | --- | --- |
   | 0.50 | **11 / 25 (44%)** |
   | 0.40 | 15 / 25 (60%) |
   | 0.30 | 17 / 25 (68%) |
   | 0.20 | 20 / 25 (80%) |
   | 0.15 | 21 / 25 (84%) |

   b2ai-28's finding reproduces at 4× the corpus size. **At 0.5, 56% of declared airway material
   produces no HeAR label.**

   A separate observation the owner should see, because no threshold fixes it: the *same task token
   on different subjects* spans nearly the whole range — `Cough-1` scores **0.011** on one subject
   and **0.954** on another; `Cough-2`, **0.007** and **0.442**. The variance is in the recordings,
   not in the cut.

### Proposal — per-classifier, and explicitly a declared default rather than a fit

The three play one role: produce a per-window label set whose windows are counted as a line's
evidence. A window retaining no label contributes nothing. Since the scores are not comparable but
the role is identical, the least arbitrary policy available without labels is to make **retention**
comparable rather than the number, and to place each cut **off the mode of its own distribution** so
the value is not maximally sensitive to itself.

| key | proposed | basis | confidence | home |
| --- | --- | --- | --- | --- |
| `windows.yamnet.default_threshold` | **0.50** | unchanged from the campaign. Its distribution is bimodal with 44.4% of windows ≥0.95, and retention barely moves across the plausible range (73.4% at 0.40, 68.2% at 0.50, 63.7% at 0.60) — the value is not load-bearing here | medium-high | `data/` profile |
| `windows.ast.default_threshold` | **0.40** | 0.50 sits in AST's single densest bin (0.50–0.55, 12.1% of windows), the worst available placement; 0.40 sits below the mode and retains 72.6%, matching YAMNet's retention at the same value. Also note **nothing exceeds 0.874**, so any cut ≥0.9 silences AST entirely | medium | `data/` profile |
| `windows.hear.default_threshold` | **0.20** | 0.50 discards 73.7% of windows and 56% of declared airway material, on a distribution with no gap and a documented retraction. 0.20 sits below the p50 of 0.288, retains 64.6% of windows and 80% of declared airway recordings | **low-medium** | `data/` profile |
| `windows.{yamnet,ast,hear}.label_thresholds` | **`{}`** | not null. `preprocess.py:727` reads it through `require`, which raises on null identically to a missing key. `{}` declares "no per-label override"; null declares "nobody decided" and refuses to run. It is a `config.DATA_MAP_PATHS` mapping, so entries can be added by override without editing the package | high | `default.yaml` |

**What upgrades all four to fitted:** a per-classifier ROC over windows labelled by a human, one
curve per classifier, plus per-label cuts for labels whose curves separate from their classifier's
default. Nothing short of that makes these measurements.

**The risk to weigh, and it is the reason to review these together:** a permissive cut interacts
with `taxonomy.presence_floor` = 1, where one window is presence. Measured over the 113 stage-0 runs:

| | HeAR → `airway.health_acoustic` | YAMNet → `speech.acoustic` |
| --- | --- | --- |
| runs reading present at thr 0.50 | 34.5% | 77.9% |
| at 0.30 | 49.6% | 81.4% |
| at 0.15 | 58.4% | 89.4% |

The speech line is insensitive to the threshold; the airway line is not. If a permissive HeAR cut is
adopted, `presence_floor.airway.health_acoustic` = 1 is where the weakness lands, and raising that
floor is the cheaper correction than raising the threshold — because the floor costs a decision while
the threshold costs the measurement itself.

## The remaining (A) keys

Every value below is sourced from `runs/b2ai-v2/override.yaml`, which ran the b2ai-v2 campaign and
carries a written note per key naming what the number is, why that number, and what would replace it.
**Its own header says every value is "an ADMITTED GUESS, not a measurement."** Promoting a guess from
a campaign override into the packaged default changes its status, so each row below states confidence
honestly rather than inheriting the override's authority.

| key | proposed | basis | confidence | home |
| --- | --- | --- | --- | --- |
| `voice.f0_range_hz` | `[75.0, 500.0]` | Praat's default adult analysis range. Unblocks `phonation_tracks` and with it the whole voice kind | medium | `data/` profile |
| `speech.word_gap_ms` | `500` | ported unchanged from v1; a pause over half a second splits word runs. Round conversational figure, nothing fitted. Unblocks the ASR span source in PREPROCESS | low-medium | `default.yaml` |
| `phonation_spans.f0_stability_cents` | `100.0` | one semitone of F0 movement per 10 ms hop still continues a sustain; same order as ordinary sustained-vowel jitter | low | `data/` profile |
| `phonation_spans.formant_stability_hz` | `400.0` | the disjunction's other limb, deliberately wide so a sustain with a broken F0 track still continues — the disordered-voice case the detector exists for | low | `data/` profile |
| `phonation_spans.glide_min_excursion_cents` | `400.0` | a third of an octave of monotone movement separates a glide from drift; the b2ai glide task instructs far more, a held vowel far less | low | `data/` profile |
| `phonation_spans.hangover_ms` | `60.0` | 6 hops of continuous criterion failure closes the span. The fitted `spans.hangover_ms` 120 does **not** transfer — it was fitted for the energy-envelope detector, and a phonation span closes on continuity, not level | low | `data/` profile |
| `phonation_spans.voicing_strength_floor` | `0.3` | Praat pitch strength above which a frame counts voiced. Does **not** gate the span — only sorts frames for the mixed-fraction test, because a periodicity floor would exclude the voices most in need of measuring | low | `data/` profile |
| `phonation_spans.mixed_voiced_fraction` | `0.5` | a bare majority, chosen for symmetry. The override calls this "the value with the least claim behind it in this file" and it probably wants two cutoffs, not one | **lowest** | `data/` profile |
| `phonation_spans.unvoiced_max_formant_bandwidth_hz` | `250.0` | widest F1/F2 pole admitted as resonant evidence when periodic F0 is absent; the conservative screening value the task-level tests exercise | low | `data/` profile |
| `phonation_spans.word_aligned_min_evidence_fraction` | `0.8` | four fifths of a timed consensus word must show periodic or narrow-resonant evidence. An evaluation choice, not a lexical rule | low | `data/` profile |
| `taxonomy.presence_floor.*` (4) | `1` each | one window or one word is presence — screening-permissive, so a run surfaces candidates rather than missing them. Absence still needs **both** of a kind's lines absent, so a floor of 1 does not make absence easy. See the interaction table above before accepting for `airway.health_acoustic` | medium for speech, **low for airway** | `data/` profile |
| `taxonomy.voice_min_duration_s` | `1.0` | every phonation task in this protocol instructs ≥3 s, so 1.0 s is a third of the shortest intended production | low-medium | `data/` profile |
| `taxonomy.voice_uncertain_duration_s` | `0.3` | roughly one long syllable nucleus — enough to say something was voiced, not enough to call it a sustain | low-medium | `data/` profile |
| `taxonomy.speech_labels` | `[Speech, "Child speech, kid speaking", Conversation, "Narration, monologue"]` | exact display strings verified against the shipped YAMNet class map (indices 0–3) and AST's `id2label`. **Commas, not slashes.** The family stops at the four labels *both* detectors express, because the acoustic line sums the two grids' counts and an AST-only member would let one grid contribute evidence the other cannot | **high** | `default.yaml` |

Two dependency notes for implementation, both from the override's own text:

- **`voice.f0_range_hz` and `voice.f0_range_ratio_max` must be decided together.** A run is flagged
  ambiguous when `f0*2 <= f0_max OR f0/2 >= f0_min`, so the unflagged band is
  `(f0_max/2, 2*f0_min)`, non-empty only when `f0_max < 4*f0_min` — narrower than two octaves. At
  `[75, 500]` the ratio is 6.67× and the unflagged band is empty, so **100% of clean phonation
  flags and `ambiguous_runs_n` carries no information.** `f0_range_ratio_max` exists to refuse such
  a configuration at load and is null, so nothing is refused. It is category (B), read only by
  VOICE, but adopting `[75, 500]` without it ships a vacuous flag.
- **`taxonomy.speech_labels` is read by SPEECH too** (`speech.py:530`), so it is not a
  TAXONOMY-private vocabulary.

## Where I decline to propose a value

- **`words.onomatopoeic_tokens`** — the override leaves it null as "owed the corpus it would be
  drawn from", and I have no corpus for it. It is read as `config.get(...) or []`
  (`preprocess.py:1146`), so **null already behaves as empty** and costs nothing at runtime. If the
  ruling requires literal non-nullity, `[]` is the correct value and carries no claim; inventing a
  token list would. Cheapest path to a real value: the consensus transcripts of the 112 stage-0
  recordings already on disk, read for non-lexical vocalisations a rater would call onomatopoeic.
- **`windows.hear.default_threshold`** — I propose 0.20 above because the ruling asks for a value,
  but I want the low confidence on record. HeAR is the one classifier where the cut is genuinely
  load-bearing (34.5% → 58.4% of runs change verdict across 0.50 → 0.15) and where the inter-subject
  spread on identical declared material (0.011 vs 0.954) suggests no scalar serves all three
  subjects. Of everything in this document, this is the value most worth measuring properly and
  least worth trusting.

## What this document does not do

It proposes no value for any (B) or (C) key beyond recommending ROUTING's map be promoted, and it
applies nothing. Every threshold marked `data/` needs a profile with a written derivation before it
lands, per CLAUDE.md — the values above are the derivations' inputs, not a substitute for them.
