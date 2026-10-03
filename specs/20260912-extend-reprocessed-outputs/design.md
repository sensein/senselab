# Extending a finished run with the outputs three config changes redefined

Three changes have landed that alter what PREPROCESS and TAXONOMY write. The finished corpus —
62,550 recordings — predates all three, and a full PREPROCESS re-run to obtain them costs roughly a
day on a 128-task array: enhancement, YAMNet, AST, HeAR and every ASR recomputed.

1. `words.onomatopoeic_tokens` is populated with the cough lexicon, so `consensus.bracketed_form`
   turns a matching token into a bracketed word. Stored runs carry `cough` as an ordinary lexical
   word.
2. `_write_consensus_taxonomy` merges its classifiers by AudioSet node identity rather than by exact
   label string, so `Throat Clear`/`Throat clearing`, `Baby Cough`/`Cough` and `Snore`/`Snoring`
   become single rows named by the node and carrying `labels_by_classifier`. Stored runs hold the
   string-matched consolidation.
3. `voice.f0_search_range_hz` is `[50.0, 600.0]`, redefined as the wide search each recording's own
   floor and ceiling is derived from. It was null when the corpus ran, and `config.require` on a null
   raises, so the block never executed: **the corpus has no phonation tracks at all.**

**All three are in `scripts/extend_reprocessed_outputs.py`.** Re-bracketing was left out of the
first version of the driver on the grounds that it moves span extents and the whole alignment
lattice. Measured and read against the code, neither is true, and §3 replaces that argument with
what the vocabulary does change.

## The question each change has to answer

`specs/20260912-quality-clip-consistency/design.md` settled the shape of an extend pass: read
`store.jsonl` under the run's own id, derive what is derivable from what the store and the run tree
already hold, write the store back atomically, re-export `prov/`, touch nothing outside the
recording's own run. Both existing drivers are **appends** — a new entity beside untouched ones —
and both converge by skipping a store that already carries their output.

None of these three is that case unexamined, so each has to answer four questions:

* **Is it derivable?** Can the new record be computed from stored state alone, with no model run?
* **Is it containable?** If the store already holds the record, retiring it is an invalidation edge.
  What else in the store was computed *from* the old record, and does that have to be retired too?
* **Does a second pass change anything?** "Already appended" is a presence check. "Already
  rewritten" is not, and needs an equivalent that does not depend on probing attributes.
* **What may the activity claim?** The clip driver deliberately writes
  `PREPROCESS`/`clip_amplitude` rather than the fresh path's `PREPROCESS`/`clip_spans`, because an
  activity carrying `near_threshold`, `leniency_samples`, `minimum_extreme` and `merge_gap_ms` it
  never ran would assert that the detector executed there. The same judgement applies here.

## 1. Phonation tracks — an append

Derivable. `f0_track` runs over the pre-emphasised stream, `formant_track` over `plain`, and both
streams are files in the run's own `run/streams/`. Neither is a model: they are Praat through
parselmouth. `derive_f0_range` narrows the recording's own `[floor, ceiling]` off `plain` from the
configured wide search, and that range is recorded in the measurement, so a reader never has to
guess which search a track was made under.

Containable trivially, because there is nothing to contain. The measurement is absent from every
corpus store, so it supersedes nothing and contradicts nothing. Its only readers are VOICE
(`nodes/voice.py:277`, which raises `LookupError` when it is absent) and REPORT's phonation figure.
VOICE cannot run on these recordings either way — its own subject, the phonation spans, was retired
on 2026-09-04 (`_retired_voice_line`) — so what this append buys today is the figure and the
measurement itself, not a branch that starts working.

### Both paths read the streams out of the store

The fresh block held `sharp` and `plain` as local arrays from the conditioning pass and handed those
to the trackers. An extend pass has no conditioning pass, so it must read the FLACs back — and if
the fresh path keeps using its in-memory arrays, the two paths hand the trackers different samples
and mint different entities for the same recording.

So `_phonation_tracks` is now a thin call into module-level
`preprocess.phonation_tracks(store, config, run_dir=...)`, which resolves both streams out of the
store, exactly as `ppg_posteriorgram` and `praat_features` already do. The fresh path and the extend
path are then the same function over the same bytes. The pre-emphasised stream is resolved by name
and `plain` is the fallback when pre-emphasis was disabled, which is the same choice the
conditioning block makes for `sharp_signal`.

The activity is `PREPROCESS`/`phonation_tracks` on both paths, with the same parameters, because the
extend really does run that computation on those inputs. There is nothing here for a different name
to protect against.

### Convergence

The measurement's attributes are five scalars and two signal names. **No `path_attributes`.** This
matters: the PPG extend found that `path_attributes` includes `mtime_ns`, so re-running a block that
writes a sidecar produces a byte-identical file at a new mtime, a different attribute set and
therefore a *differently identified entity*. The npz here is written to `derivatives/` and named
nowhere in the entity, so two passes mint the same activity id and the same entity id and the second
is a genuine set-union no-op. The driver skips a store that already carries the measurement anyway,
so it saves the decode as well.

## 2. The consensus taxonomy — a rewrite

Derivable. `_write_consensus_taxonomy` reads `span_yamnet` and `span_hear`'s `raw_scores` — the
models' own outputs, written whatever the labelling threshold said — and consolidates them. Every
input is in the store. No model runs, no audio is decoded, no sidecar is read.

Containable. Nothing in the graph reads `consensus_taxonomy`: ROUTING reads `kind` entities, VOICE
is only *pending* a rework onto it, and the one reader in the tree is
`routing_analysis/features.py`, which is offline analysis rather than a node. Retiring the old row
set therefore invalidates nothing downstream, and the rewrite's blast radius is one entity.

### Why it must be retired and not merely joined

The store is append-only. Writing the new consolidation without retiring the old leaves two live
`consensus_taxonomy` measurements over one recording, disagreeing about how many classifiers reached
a node. `find_measurement` takes the last live one, so most readers would get the right answer by
accident; `find_measurements` and any reader that counts would not. An invalidation edge is the
store's own answer to this and the only one available, since nothing is ever deleted.

### Deciding whether a store is already current

Not by probing for `labels_by_classifier`. An attribute probe asserts that the writer's output shape
is a version marker, which is true today and silently wrong the next time a field is added.

The store already has an exact test. An entity id is `sha256([run_id, prov_type, extent,
attributes])`, so the id of a recomputed consolidation is equal to the live one's **iff the store
already holds exactly this reading**. `rewrite_consensus_taxonomy` recomputes, compares ids, and
retires the old measurement only when they differ. A fresh run recomputes to the id it already has
and nothing is retired; a corpus run recomputes to a different id and one edge is added. The test is
the record's own identity, not a guess about its shape.

### What the activity claims

Two activities, which is one more than the fresh path writes.

The measurement is generated by the canonical `TAXONOMY`/`consensus_taxonomy` activity, whose
parameters are `consolidation_floor`, `classifiers` and `ontology_profile`. That is a true claim:
the consolidation really ran with those values, on this path as on the fresh one.

But those parameters are also *unchanged* by this whole change — what changed is the merge rule,
which the activity never named — so the recomputed activity's id equals the one that generated the
superseded measurement. Pointing `wasInvalidatedBy` at it would say that the activity which
generated a record also retired it, which is incoherent. The retirement is therefore its own
activity, `TAXONOMY`/`consensus_taxonomy_superseded`, carrying the superseded entity's id and the
reason. A reader arriving at a retired row follows one edge and is told why it is not current, which
is the fact the canonical activity is structurally unable to carry.

The generic half of that — create the activity, record `used`, record `wasInvalidatedBy` — is
`extend.supersede`, in `extend.py` beside the layout helpers, because every future rewrite driver
needs the identical three lines and a second copy of them is a second chance to omit the `used` edge.

### A store carrying no consolidation is left alone

`before is None` means TAXONOMY did not run on that recording (a store whose per-span classifiers
produced nothing writes no consolidation and still writes none here). Synthesising one would give
the run a TAXONOMY output beside none of TAXONOMY's others, which reads as the node having run. The
driver reports `absent` and writes nothing.

## 3. Re-bracketing — a rewrite

Derivable, containable, and in the driver. An earlier version of this document argued it was none of
those, on two claims about the code. Both are wrong, and what replaces them is below.

### The vocabulary decides surfaces, not columns

The first claim was that `bracketed_form` is applied before alignment, so the vocabulary "changes the
lattice itself: slot count, column membership, each word's `index`, the fitted extents".

`harmonize_transcripts` normalises before it aligns:

```python
tokens = {m: [normalise_token(t) for _, _, t in words[m]] for m in models}
```

and `normalise_token` keeps only alphanumerics and the apostrophe — its own docstring says
"Brackets and punctuation drop, so `[UM]`, `um` and `Um.` share one key". `bracketed_form` builds
its display from the token's own `vocabulary_key`, which differs from the raw token only by
casefolding and edge punctuation, and `normalise_token` drops both. So
`normalise_token(bracketed_form(t, v) or t) == normalise_token(t)` for every token and every
vocabulary: **the key a member groups on is invariant under bracketing.**

Everything the alignment decides is therefore fixed — the reference model, the slot keys, column
membership, each word's `index`, its per-source `readings` and `timings`, its fitted `extent`, the
spreads, `outcome`, `agreement`, and the provenance counts `n_words`, `outcomes`,
`n_words_time_shifted`, `max_time_shift_s`. What moves is each member's *display*, and through it
the column's surface `text`, its `bracketed` flag, its variants' surfaces and `bracket_overrides_n`.
`consensus_test.py::TestBracketingDoesNotMoveTheAlignment` pins it: one pair of hypotheses aligned
under the null vocabulary and under the lexicon gives columns equal in every other field.

Bracketing manufactures no agreement either. `bracketed_form` returns `f"[{key.upper()}]"` from the
token's *own* key, so `cough` → `[COUGH]` and `咳` → `[咳]`, which normalise to `cough` and `咳` and
stay two readings. A variant column of two onomatopoeic spellings is still a variant column.

The words are therefore re-read, not re-aligned: `bracketed_form` is evaluated again over the
`readings` each stored word already carries, and `_column_word` — the same function the fresh path
calls — returns the column's new surface. No model runs, no audio is decoded, nothing is re-timed.

### The measured effect

At the ≥1-token threshold the reader actually applies:

```
group              n   fires at >=1   drops below the SPEECH gate
cough           2813          0.704       849
breath         10204          0.003         3
DDK             7989          0.007        26
voice           8306          0.001         0
lexical speech 33235          0.003         0
```

849 cough recordings route to SPEECH on transcribed coughs and would stop. No lexical-speech
recording loses its SPEECH routing.

### What is retired with the words

A word entity's id is a digest over its attributes, so a re-flagged word is a new entity and the old
one is retired — but only the ones that moved. A column no vocabulary entry touches recomputes to
the id it already has, stays live, and keeps every edge into it.

The `consensus_transcript` lists the words by id and renders them into `text`, so it is retired and
rewritten: the new id list, the new text, the new `bracket_overrides_n`. Every other field is the
alignment's and is carried through verbatim, which is what makes the rewritten measurement **the
entity a fresh run under the lexicon would have minted** — asserted in
`extend_reprocessed_outputs_test.py::TestRebracketingTheWords::test_the_re_flagged_run_is_the_run_the_lexicon_would_have_made`.

Nothing else is retired, and `supersede`'s own contract is the reason: it retires an entity *in
favour of a replacement already written beside it*. What is downstream of the words — TAXONOMY's
`kind` lines, ROUTING's `branch_decision`s, the branch verdicts, VERDICT's fold — are conclusions
reached under the reading the run then held. This pass writes no competing conclusion, and retiring
one with nothing to put in its place would turn "the run concluded X under the words it had", which
is true and recoverable, into "the run concluded nothing", which is false. They stay live, they
still name the retired words in their own `element_ids` and `word_ids`, and a reader that resolves
those ids is told they are retired. Re-deriving them is a re-run of the graph from TAXONOMY onward:
TAXONOMY and ROUTING are store arithmetic, but a changed route selects branches that never ran, and
SPEECH's diarization, embeddings and PII are models. That is a different program from this one, and
point 4 of the consumer notes below says so in the store's own terms.

### The spans the ASR proposer contributed

The second claim was that re-flagging moves span extents, and that a moved span needs HeAR, YAMNet
and SQUIM re-run over it. The mechanism is real — this is the one thing downstream of the flag that
is not a decision — but the size is not what the claim assumed, and the conclusion drawn from it is
not the only one available.

`_spans` builds its ASR candidates from the extents of the **non-bracketed** words, so re-flagging
does change that proposer's input. The set it changes is small:

```
cough families: 150 stores, 1 carrying an asr-measure span   (0.01 per recording)
harvard:        150 stores, 2                                 (0.01 per recording)
free-speech:    150 stores, 27                                (0.21 per recording)
```

`_novel` rejects nearly every ASR candidate, because the 4–30 amplitude spans a recording already
carries cover the same audio.

They are kept, not recomputed. A span's extent is what `span_hear`, `span_yamnet` and `squim` were
measured over at PREPROCESS time; a recomputed span carries none of those and no arithmetic over the
store produces one, so recomputing would retire measurements of real audio and put in their place
spans every per-span consumer reads as unclassified. The store says which reading proposed them in
two places: each such span keeps its `wasDerivedFrom` edge to the transcript that proposed it, which
is now the retired one, and the `rebracket` measurement names the span ids beside
`asr_proposed_spans_recomputed: false`.

### Deciding whether a store is already re-bracketed

By recomputation, as with the taxonomy, and never by probing for a bracketed token. Each word is
re-read and its attributes compared with the stored ones; a store where no column moves is not
touched and the driver reports `current`. The same test is the convergence proof: a second pass
re-reads the words the first pass wrote, finds every one equal, writes nothing, and leaves the
fingerprint where it was.

A store carrying no `consensus_transcript` is reported `absent` and given nothing, for the reason a
store carrying no consolidation is given none: PREPROCESS wrote none there.

### What the activity claims

`PREPROCESS`/`rebracket`, carrying `words.onomatopoeic_tokens` — not `PREPROCESS`/`consensus`, whose
parameters are the routine, the source order and the sources it aligned. This pass aligned nothing,
and an activity carrying those would assert that it had. The two retirements are their own
activities, `PREPROCESS`/`word_superseded` and `PREPROCESS`/`consensus_transcript_superseded`, by
the same argument as the taxonomy's.

### The config hash, and what the store now adds to it

`config.py` hashes the whole merged mapping and `words.onomatopoeic_tokens` is inside it, so a run
made under the null vocabulary and one made under the cough lexicon were already distinguishable by
identity rather than silently conflated. After an extend pass the store says it directly: it carries
a `rebracket` measurement naming what moved, and the reading it replaced is still in the store,
retired.

## One pass, and the order within it

One driver, one pass, because the three derivations are independent of each other and of everything
else, and two passes would read and rewrite every store twice.

* Phonation tracks read `plain` and the pre-emphasised stream. The consensus taxonomy reads
  `span_yamnet` and `span_hear`. Re-bracketing reads the `consensus_transcript` and the words it
  names. None reads another's output, in either direction.
* `_write_consensus_taxonomy` reads `find_measurements(store, "span_yamnet")` and
  `find_measurements(store, "span_hear")` and nothing else. It has never read a word, so
  re-bracketing would have to precede it only if it changed the per-span classifier set — which is
  exactly what recomputing the ASR-proposed spans would have done, and one more cost that decision
  would have carried. As the three stand the order is free.
* The phonation block stopped depending on the consensus transcript when span *detection* moved out
  of it into TAXONOMY.

So the order inside `extend_one` — re-bracketing, phonation, taxonomy — is fixed for determinism and
nothing else.

## Convergence, for the pass as a whole

The driver does not decide to skip per derivation. It takes `store.fingerprint()` before applying
the three, applies them, and writes the store, re-exports `prov/` and records the environment **only
if the fingerprint moved**.

`fingerprint()` is a content hash over the sorted entity, activity, agent, environment and relation
keys, so it ignores insertion order and is exactly "does this store hold different records than it
did". A derivation that produces records the store already holds is a set-union no-op and leaves it
where it was. This is stronger than a per-derivation presence check and it is what makes the
rewrite convergent: recomputing a current consolidation mints the id already present, adds no
relation, and the pass writes nothing at all.

Taking the fingerprint before `capture_environments` is deliberate: the host environment entity is
itself a record, and capturing it unconditionally would make every pass look like a change.

Phonation is still skipped by presence rather than by recomputation, because the recomputation is a
decode and two Praat passes and the presence check is a dictionary lookup.

`write_store` writes to a `.partial` sibling and `replace`s it, so a task killed mid-write leaves no
truncated store for the next pass to read, and a task killed mid-slice restarts and redoes only what
never landed.

## A derivation that cannot apply is an absence, not a failure

The corpus pass over 60,202 recordings reported **613 errors**, and every array task exited nonzero.
All 613 were one thing:

```
phonation_tracks: ValueError: no F0 range could be derived from this recording over [50.0, 600.0] Hz
```

on 0.33 s fragments, coughs and breaths — 271 harvard fragments and 243 airway recordings. Unvoiced
audio has no derivable pitch. `derive_f0_range` is right to refuse: the alternative is a guessed
range, and the function's own contract is "an absence, never a guessed range". What was wrong is the
classification. The work on those recordings was not lost — each store carries its phonation record,
its merged consensus and its re-flagged words — and the row still read `error`, the slice still
counted it, and the task still exited 1.

### Where the line is drawn

This codebase already separates the two, with a typed `ValueError` subclass that a caller catches as
a cascading absence: `SpanTooShortForYAMNet`, `AudioTooShortForAST`,
`CrisperWhisperDecoderPositionsExceeded`, `PpgsPosteriorgramUnavailable`. `derive_f0_range` now
raises `F0RangeUnavailable` in that idiom, which is a `ValueError` subclass, so every existing
caller — PREPROCESS's block loop, VOICE, the tests — is unchanged in behaviour.

What was missing is the other half: the **driver** has to know that such a raise is an answer. It is
recorded once, at the driver level, rather than per case:

* `extend.UNAVAILABLE` — the five typed absences, in one tuple. Whether a raise is an absence is a
  property of the exception, not of which derivation raised it.
* `extend.attempt_derivation(call)` — runs one derivation and returns
  `DerivationOutcome(detail, failed)`. A `UNAVAILABLE` raise is `absent: <Class>: <message>`, not
  failed; any other `OSError`/`ValueError`/`LookupError` is the class and message, failed.
* The row's `status` is `error` iff some derivation failed, and `ok`/`skipped` otherwise by the
  fingerprint. The three drivers no longer each carry their own list of words to treat as
  determinate — that list was the per-case form of this rule, and it is what let one derivation's
  absence read as the whole row's failure.

The outcome words themselves (`ok`, `error`, `skipped`, `present`, `current`, `rewritten`, `absent`)
moved into `extend.py` with the rule, so the three drivers spell them identically.

**`absent` covers two things and says which.** The bare word is "the store holds nothing of this to
work from" — no consensus transcript, no consolidation. `absent: <reason>` is "this recording has no
such thing to derive". Both are absences of the derivation and neither is a failure; only the second
needs a reason, and it carries one.

### What this does not soften

A store that will not open, a run tree with no conditioned stream, a `LookupError` from a
prerequisite the store should hold — all still fail, still make the row `error` and still exit the
task nonzero. The distinction is between *this recording has no such thing* and *this pass could not
do its job*, and only the exception's own type decides which.

## Two guards on the extract path

The same defect in a different place, and worth stating here because it is the same reading error:
silence taken for success.

`build_manifest` in `scripts/analyze_routing_evidence.py` reused an existing manifest
unconditionally, printing one line. Pointed at a stale output directory it reused a manifest built
before the corpus was relocated: every row intact, every `store` path resolving to nothing. Each row
became `missing`, the extract wrote **0 features out of 62,550 rows**, and the job exited **0**.
`--expect` checks the manifest's row count, which was right; what was wrong was what the rows
pointed at. The only signal was a downstream scorer reporting zero recordings, one job later.

1. **A reused manifest must still resolve the tree it names.** `check_manifest_resolves` stats up to
   `MANIFEST_PROBE` = 64 rows, evenly spread through the file, and refuses when **not one** of them
   exists. The rule is threshold-free on purpose: "no evidence this manifest resolves at all" needs
   no fitted fraction, and a manifest that has lost individual recordings — deleted since it was
   built — still resolves and is still reused, because those are the extract's `missing.jsonl` rows
   and always have been. The probe is bounded rather than exhaustive so that reuse stays a
   constant-cost check on a 62,550-row manifest.
2. **A zero-feature extract fails.** After `load_features`, a manifest with rows in it and no
   features out of it raises `SystemExit` naming `missing.jsonl`, rather than scoring an empty
   record set and returning 0. This one catches whatever the first misses — a manifest that resolves
   a handful of stores and no more, a shard directory that never landed.

## What a consumer of an extended store must know

1. **`phonation_tracks` was measured after the run finished, from the streams the run left behind.**
   Its activity is the one a fresh run writes, and its entity is the one a fresh run would mint, so
   the store does not distinguish the two paths. What separates them is the `config_hash` on the
   run's own log: the corpus ran under a null `voice.f0_search_range_hz` and would have written no
   tracks at all.
2. **A `consensus_taxonomy` measurement may not be the only one in the store.** The retired reading
   is still there, still readable, marked `wasInvalidatedBy`. Any reader that selects by name must
   filter on `is_invalidated` — `find_measurement`, `find_measurements` and `live_entities` all do —
   and a reader that walks `entities("measurement")` raw will see two.

   **The one reader in the tree that did not filter now does.** `routing_analysis/features.py`'s
   `extract_features` collected the invalidated ids while streaming the JSONL and applied them to
   `word` and `span` records only; `_absorb_measurement` ran for every measurement record whatever
   its state, and for `consensus_taxonomy` it *assigns* each peak rather than taking a maximum, so
   the record written later in the file won. It was right by insertion order alone. Since an extend
   pass leaves every extended store carrying two consensus readings and two transcripts, that is now
   load-bearing, so the invalidated ids are collected in their own pass over the file
   (`invalidated_ids`, which decodes nothing but the retirement edges) and every entity record —
   measurements, kinds and verdicts included — is filtered against them before it is absorbed.
3. **The generating activity does not separate the two readings.** Its parameters are unchanged by
   the merge-rule change, so both measurements hang off the same activity id. The retirement
   activity, `TAXONOMY/consensus_taxonomy_superseded`, is the only record that says which is which,
   and the only edge into it comes from the retired measurement.
4. **The words are re-flagged; the conclusions drawn from them are not.** An extended store spells
   transcribed coughs as bracketed words and its `consensus_transcript` is the one the lexicon would
   have produced. Its TAXONOMY `kind` lines, its ROUTING decisions and its branch verdicts are still
   the ones the run reached under the old reading, and they name retired word ids in their own
   `element_ids` and `word_ids` — which is how a reader tells. Its ASR-proposed spans are likewise
   the ones the old reading proposed, still carrying their `span_hear`, `span_yamnet` and `squim`
   measurements, and still derived from the retired transcript. The extension does not claim
   otherwise, and re-deriving any of it is a re-run of the graph from TAXONOMY onward.
5. **The `prov/` tree was re-exported from the merged store.** BEP028 carries entity attributes
   through verbatim, including the invalidation, so the exported graph agrees with `store.jsonl`.

## Verification

`src/tests/scripts/extend_reprocessed_outputs_test.py`, over synthetic finished runs with real FLAC
streams, real per-span scores and a real two-recognizer consensus. All three derivations land and
are correct; the superseded consolidation and the superseded words are no longer live and each
retirement names why; a word no vocabulary entry touches keeps its id; the re-flagged words and the
rewritten transcript are the entities a fresh run under the lexicon mints; the ASR-proposed span is
kept, still derived from the retired transcript, and the `rebracket` measurement records that it was
not recomputed; a run already carrying all three forms is skipped with a byte-identical store and an
unchanged fingerprint and gains no retirement edge; three passes retire exactly what the first pass
retired; the extend path's phonation measurement equals a fresh pass's; `prov/` is re-exported
carrying them; a store with no conditioned stream records the failure and still has its taxonomy
made current; a store with no consolidation is given none, and one with no transcript is given none.

`src/tests/audio/workflows/triage/consensus_test.py` pins what re-bracketing rests on: one pair of
hypotheses aligned under either vocabulary gives the same lattice and the same columns, differing
only in surfaces and the `bracketed` flag, and two onomatopoeic spellings stay two readings.

`src/tests/audio/workflows/triage/routing_analysis_test.py` pins the reader: a store holding a live
and a retired `consensus_taxonomy` reports the live one's peaks whichever order they were written
in.

The absence rule is pinned in `extend_reprocessed_outputs_test.py::TestADerivationThatCannotApply`,
over a run whose conditioned streams are silence: the phonation row reads
`absent: F0RangeUnavailable`, the run's status is `ok`, the store gains no phonation measurement, the
other two derivations still land on that recording, and a run whose streams the tree no longer holds
is still an `error`.

The two guards are pinned in `src/tests/scripts/analyze_routing_evidence_test.py`: a manifest none of
whose stores exist is refused, one that still resolves is reused, a manifest missing a few recordings
is not refused, an empty one is not refused by the probe, and an extract that writes no features from
a non-empty manifest exits nonzero instead of scoring nothing.
