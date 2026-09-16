# SPEECH's two modes — what porting the design decided, and what did not survive it

[`branch-speech.md`](branch-speech.md) is the branch's design (S1–S11);
[`expected-patterns.md`](expected-patterns.md) is the executable one, whose `align_speech` (`:2006`)
and `detect_speech` (`:2293`) bodies this ports;
[`branch-foundation.md`](branch-foundation.md) is the shared foundation the port sits on. This
document is the rest: the five places the design left a choice to code, the two owner-decided
migrations, and what did not survive contact with the real objects.

The implementation is `src/senselab/audio/workflows/triage/nodes/speech.py`; its tests are
`src/tests/audio/workflows/triage/nodes/speech_test.py` (the node) and `speech_modes_test.py` (the
two modes, over hand-built stores).

## D-S1. An unmeasured operating point degrades one answer, not the whole node

Every numeric key in the `branch` config section ships `null`, and `BranchParams` raises on reading
one (foundation D-F4). Ported verbatim, that makes `detect_speech` — which needs
`branch.run_gap_max_s` — raise on every recording routed to SPEECH out of family, which is 8,330 of
the 41,565 SPEECH routes, and takes the transcript, the PII scan and the quality measurements down
with it. **`run.py:311` gates REDACT on SPEECH having found PII**, so a branch that raises before
its step 7 silently withdraws redaction from the run.

So the mode bodies read operating points through `_Points`, which returns `None` for a key that is
null, records which keys it was asked for, and emits them as one
`unmeasured_operating_points` measurement. A capability whose cut is missing is skipped; a `done`
that depends on a missing cut is `UNDETERMINED`. Nothing is defaulted: a typo still raises, because
`_Points` refuses a name outside `PARAM_KEYS`.

**Rejected: catching `ValueError` around `config.require`.** `require` raises the same type for an
unmeasured key and for a misspelled one, so a catch would turn `p_peek_prominence_db` into a silent
absence — the failure mode foundation D-F4 chose properties over `__getattr__` to avoid.

`detect_speech` therefore proposes **nothing** under the packaged configuration. That is the honest
state and not a regression to work around: falling back to adjacency (`merge`, or
`group_extents_into_runs`, which needs no gap) is what the design's own execution caught, because
ordinary speech has a gap between every pair of words and adjacency yields one span per word. What
the node still proposes with no cut at all is its `speech_run_*` spans (below), which are grouped by
adjacency *by design* — `group_extents_into_runs`' docstring is the argument — and carry the SQUIM,
disruption and proximity measurements.

## D-S2. The node's word runs stay, and become proposals

S2's spans — one per run of touching lexical words — carry the corroboration votes, the per-span
SQUIM, the disruption counts, the proximity legs, `attributed_to` and `nontarget`. The design's
`align_*` bodies propose a different set (`task_extent`, `structure_*`, `breath_group_*`) and
`detect_speech` a third (`lexical_run_*`). All three coexist: the design already has `task_extent`
overlapping `structure_*`, and an aggregate over the same ground is a distinct object, not a
duplicate.

What changed is that the run spans are now minted through `PROPOSERS["SPEECH"]` and written by
`propose_spans`, so each names its evidence — the consensus measurement, its member words, and the
PREPROCESS spans it overlaps. Their role is `speech_run_<n>`.

**A run the consensus places at one instant is dropped with a flag.** `propose_span` refuses a span
of no positive duration, and it is right to: a zero-width extent names no region. The old code
wrote one (`(0.72, 0.72)` on the Glides shape, which `speech_test` pins). Dropping it costs the
per-span measurements for that run and nothing else.

## D-S3. `align_speech` rebuilds the stimulus alignment rather than reading its sidecar

`stimulus_alignment` is a measurement whose three tables live in
`derivatives/stimulus_alignment.npz`, and reading a sidecar needs the run directory. **A mode is
handed no run directory** — the contract is
`align_speech(task_family, store, hint, params)` — and widening it would change the signature all
four branches are written against.

So `_stimulus` reads the measurement for its *presence* (which is what says PREPROCESS produced
one) and its *id* (which is what a proposal names), and rebuilds the projections by calling
`stimulus.align_stimulus` over the same three inputs PREPROCESS gave it: the declaration's
`expected_speech`, the lexical consensus words, and `stimulus.sentence_terminators`. The routine is
deterministic and carries no model, so this is not a second measurement — but it is a second
*computation*, so `_stimulus_agrees` checks the rebuilt counts against the five the measurement
recorded (`n_expected`, `n_realised`, `n_substituted`, `n_absent`, `n_unexpected`) and emits
`stimulus_alignment_rebuild_agrees: false` when they diverge. A silent divergence would be the worst
outcome; a visible one is a finding.

**Rejected: run directory through `BranchParams`.** It is the operating-point record, shared by four
branches, and a path is not an operating point.

**Rejected: reading the alignment in the node and passing it in.** Same signature problem, and it
would give SPEECH a private channel into its own modes that the other three branches do not have.

## D-S4. `repeat_reading` needed an instrument the design names but does not define

`_speech_ordered` calls `alignment.covers_sequence_twice(params.p_repeat_overlap_min)`.
`StimulusAlignment` has no such method and nothing in the graph computes one. The measure
implemented is: **the fraction of expected token keys the production realised more than once**,
emitted as `expected_sequence_repeat_fraction` whatever the configuration says, with the
`repeat_reading` deviation fired only when `branch.repeat_overlap_min` has a value. The measurement
is separable from the decision, so the fraction accumulates on real runs before anyone fits the cut.

It is a *proxy*: a speaker who says one Harvard word twice in a disfluency raises it, and the
alignment's own path cannot distinguish that from a second pass over the sentence. The cut is what
would separate them and nobody has fitted it.

## D-S5. An omission is anchored where it should have been

The two designs disagree. `expected-patterns.md` emits `deviation("omission", None, None, …)`;
`branch-speech.md` says *"an omission is recorded as a zero-width extent at the alignment point"*.
Both are implemented as one: the anchor is the end of the last extent-carrying expected token before
the omitted one, so the deviation is zero-width at that point, and extent-free only when nothing
before it was realised. The stronger claim loses nothing and the weaker one is its fallback.

`expected_index` is carried on the deviation because the store's entity ids are content digests:
two omissions of the same token text at the same anchor would otherwise collapse into one entity.

## D-S6. Breath groups read `span_hear`, not `hear_scores`

`breath_group_extents` reads `store.hear_scores` — the whole-file HeAR windows, which live in
`derivatives/hear_scores.json` and need the run directory D-S3 explains a mode does not have. The
per-span `span_hear` measurements *are* store entities carrying `raw_scores` and an extent on the
recording's own timeline, which is what `airway.py:241` already reads for the same reason. So the
breath evidence is `span_hear`'s `Breathe` score against `branch.score_min`, with the label set
taken from `branch.label_sets`, which is the one key in the section that ships a value.

All three `*_windows` derivatives are absent under the packaged configuration, so reading `labels`
rather than `raw_scores` would have made this permanently inert.

## D-S7. `off_task_extent` — the withdrawal was of a definition, not of the helper

`branch-speech.md` S4 says *"`off_task_extent` is withdrawn from this branch"*, and
`expected-patterns.md` calls `off_task(...)` in three of the four SPEECH bodies. Not a
contradiction: what S4 withdrew is the *old definition* — "a region carrying no lexical speech where
the task asked for reading" — which would have made every inter-phrase pause in a passage reading a
deviation. The shared `off_task` helper reports PREPROCESS's own `measure: "gap"` spans that no
proposed span covers, which is a different object, and `_speech_no_lexical` emits `off_task_extent`
over a *lexical word on a syllable train*, which is a third. Both are kept; the withdrawn definition
is implemented nowhere.

## Migration 1 — `attribute` becomes a `propose`

`speech.py:793` wrote `verb: "attribute"`, one assertion per word, which was the highest-volume
assertion in the store and was never one of the contract's five verbs. Owner decision 2026-09-15
gives it two destinations, `refine` or `propose`. **One site existed and it is a `propose`.**

* **The per-word assertion → `propose`.** One `speaker_turn_<n>` span per contiguous run of words
  the diarization gives to the same speaker, minted in `family: "speech"`, deriving from the words
  and the segments it aggregates, editing neither. Resolving across speakers *creates* an object
  that did not exist, which is what makes it a propose rather than a refinement of one that did.
  A run whose words the diarization could not resolve keeps its `straddles` or `unassigned` note on
  the aggregate, so the case stays legible with no per-word assertion.
* **The span-level `attributed_to` and `nontarget` (`speech.py:877-888`) needed no migration.** They
  were never assertions: they are attributes stamped on the run span at mint time, which under
  propose-only is where they belong. Nothing was converted to `refine`.

## Migration 2 — SPEECH reads the shared diarization, and runs no diarizer

`speech.py:642-646` ran pyannote over `(min word start, max word end)`. Owner decision 2026-09-15:
*"speech should not have to rerun pyannote to do this, just take the output and use it."*

**Verified complete.** Everything the branch read from its own run is in the derivative:

| what SPEECH needs | where it now comes from |
| --- | --- |
| per-segment `(start, end, speaker)` | the `.npz`'s four parallel columns (`starts`, `ends`, `speakers`, `streams`) |
| the speaker count | recomputed from the segments, cross-checked against the measurement's `n_speakers` and flagged on disagreement |
| which model produced it | the measurement's `model` attribute |
| the segments' clock | already the recording's; a whole-file derivative needs no offset, and the crop's offset arithmetic is gone |
| the audio the enrollment probe embeds | the `plain` stream, sliced by segment (below) |
| the audio separation reads | the whole `plain` stream: the reading that gates separation is whole-file, so a crop of the lexical hull would be the wrong extent |

Reading the sidecar needs the run directory, which the **node** has; this is why the diarization read
lives in the node's step 4 and not inside either mode.

The three predicted consequences, each confirmed:

1. **The stream changes `plain` → `enhanced`.** `diarization.streams` is `[enhanced, residual]`, so
   SPEECH reads the first with a live measurement. Recorded in the verdict rather than left implicit
   — the count is now a property of the enhanced signal, and a consumer comparing it against a
   `plain`-signal measurement must be able to see that.
2. **`exclusive: false` lets a word straddle.** SPEECH's own pass took `diarize_audios`' default,
   `exclusive=True`, so its segments were a partition and a word could overlap at most one. The
   shared derivative keeps pyannote's overlapping view, so the `straddles` note — previously
   reachable only where segments abutted — now fires on genuine concurrency.
3. **The per-speaker enrollment slices are affected, and are repaired.** With overlapping segments a
   speaker's concatenated audio carries the other speaker's voice wherever they spoke at once, which
   the old exclusive partition made impossible. `_exclusive_slices` takes only what a speaker holds
   alone. This is arithmetic over the segments and asks for no threshold; where the exclusive view is
   taken it is a no-op. A speaker holding nothing alone gets no probe and is flagged.

**Not acted on, as directed.** `speech.second_diarizer` is null (`data/config/default.yaml:212` —
`branch-speech.md` cites `:189`, which is stale; `:189` is now the line of the `diarization`
section's own comment saying the read-swap "is not yet made", which this work makes and which is
corrected with it), so
`second_record` stays `"not_consulted"`. Under the migration the question becomes whether the
*shared derivative* should carry a second diarizer, which is a PREPROCESS decision. What the code
does with the key it already has: when a second diarizer is named, it runs over the signal the
derivative was measured on, because two counts over two different signals corroborate nothing.
`residual_diarization` — the second stream PREPROCESS writes — is deliberately **not** wired in as
that corroborator: it answers a different question ("was a voice taken out"), not the same question
twice.

## What did not survive contact with the code

**The store facade, again.** As foundation D-F5 records. `store.words` is `consensus_words(store)`,
`store.spans` is `live_entities(store, "span")`, `store.id_of("consensus_transcript")` is
`find_measurement(store, "consensus_transcript").id`, and `w.agreement` is
`word.attributes["agreement"]`. `store.stimulus_alignment` and `store.hear_scores` do not exist at
all, which is D-S3 and D-S6.

**`covers_sequence_twice` does not exist.** D-S4.

**Two normalisations, one docstring.** `BranchParams.p_normalise` resolves to
`consensus.vocabulary_key` (casefold, strip edge punctuation) and its docstring says this is "the
normalisation PREPROCESS's own consensus and stimulus alignment already declare". The stimulus
alignment's is `stimulus.normalise_token` — *"casefold; keep alphanumerics and apostrophe"* — which
is not the same function. `vocabulary_key` strips only *edge* characters and only from
`.,;:!?"'()`; `normalise_token` drops every non-alphanumeric character wherever it sits. So
`well-known` keys as `well-known` under one and `wellknown` under the other, and `[um]` keys as
`[um]` under one and `um` under the other.
The bodies that go through the alignment therefore compare under `normalise_token` and the bodies
that go through `ordered_run` under `vocabulary_key`. In practice they agree on this corpus's
tokens; the docstring's claim of identity is wrong and is a foundation matter, not a branch one.

**`speech.second_diarizer` is at `default.yaml:212`, not `:189`.** `branch-speech.md`'s own table
notes an earlier revision cited `:166` and corrected it to `:189`; the section has moved again.

**No new config key was needed.** Every operating point these bodies read is already declared:
`run_gap_max_s`, `breath_group_min_gap_s`, `score_min`, `pause_min_s`, `response_min_s`,
`echo_ngram_n`, `echo_overlap_max`, `verbatim_overlap_max`, `coverage_min`, `expected_lexical_max`,
`repeat_overlap_min`, `omission_score_max`, `gap_off_task_min_s` and `label_sets`.

**One name is spelled in two modules.** `speech.diarization_measurement` repeats
`preprocess.diarization_measurement`'s `f"{stream}_diarization"` rather than importing it, because
importing PREPROCESS into a branch puts PREPROCESS's whole model import path on a branch's. A test
pins the two against each other, so the coupling fails loudly rather than drifting.

## The families the design warned about, and what the code does with each

| family | the trap | what is implemented |
| --- | --- | --- |
| `harvard-sentences-list`, `cape-v-sentences`, `rainbow-passage`, `caterpillar-passage` | nothing compared a transcript to its stimulus | `ORDERED_TOKENS` over the rebuilt alignment; one `structure_<n>` span per realised declared sentence, which is where CAPE-V's six conditions and the Rainbow's sentence boundaries come from |
| `word-color-stroop` | `stimulus_text` is the expected **answers**, not the displayed words; matching the display scores it backwards | the alignment is against the declaration, whatever it carries; two tests pin that declaring the answers scores a correct response clean and declaring the display produces departures |
| `free-speech` v1 | the instruction says not to read the prompt, so the prompt in the transcript is the deviation | `anti_pattern: verbatim_prompt`, fired off the n-gram echo fraction. v2 carries no anti-pattern, so the same transcript is clean — same family name, opposite expectation |
| `picture-description`, `cinderella-story`, `productive-vocabulary` | `stimulus_text` is empty; the prescription is in `instructions` | `FREE_RESPONSE` with no token source: nothing lexical is expected, so the whole transcript is the response and no word is a departure. `cinderella-story` additionally records `source_overlap` as unviable |
| the ten `SYLLABLE_REPETITION` families | in family for SPEECH *and* for DDK, with opposite expectations | `NO_LEXICAL`: no span is proposed, a lexical word is one `off_task_extent` deviation, and the count is reported against a declared zero |
| `loudness`, `loudness-v2` | in `LEXICAL_SPEECH` but a single non-lexical shout | `ORDERED_TOKENS` with literal tokens from the instruction, so no stimulus derivative is needed and the count is what the row declares |
| `animal-fluency`, `random-item-generation` | two of ten categories permit repetition, and the category lives only in `instructions` | `ITEM_LIST`; an unreadable category returns `UNDETERMINED` with a `repetition_rule` unviable rather than applying the majority rule to the minority |

## Mutation results

Ten mutations were applied to `speech.py`, one at a time, with the two SPEECH test modules and the
PII interlock run against each. Three survived the first pass and were killed by strengthening an
existing assertion rather than by adding a test: counting a substituted token as realised (the
substitution test did not read `task_extent`'s counts), proposing a span on a syllable train (the
assertion was on the empty-transcript case, where no hull exists), and giving the enrollment probe
its overlapped audio back (nothing read what the embedder was handed). All ten are caught.
