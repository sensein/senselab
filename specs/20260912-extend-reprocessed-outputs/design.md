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

**Two of the three are in `scripts/extend_reprocessed_outputs.py`. Re-bracketing is not, and the
rest of this document is mostly about why.**

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

## 3. Re-bracketing — not derivable, and left out

The measured case for it is strong. At the ≥1-token threshold the reader actually applies:

```
group              n   >=1 fires   drops below the SPEECH gate
cough           2813       0.704       849
breath         10204       0.003         3
DDK             7989       0.007        26
voice           8306       0.001         0
lexical speech 33235       0.003         0
```

849 cough recordings route to SPEECH on transcribed coughs and would stop. No lexical-speech
recording loses its SPEECH routing. That is exactly the correction the change was made for.

It is still not an extend, and the reason is mechanical rather than a matter of taste.

### The words themselves are derivable

`align_sources` is a pure function of the stored `asr_hypothesis` measurements — each carries its
recognizer's verbatim `text`, `start` and `end` per word — and of the vocabulary. Recomputing the
consensus needs no ASR.

It is also a larger recomputation than "one boolean flips". `bracketed_form` is applied to each
token **before** alignment, and the bracketed display is what
`harmonize_transcripts` aligns on (`consensus.py:282`, `:293`). The vocabulary therefore changes the
lattice itself: slot count, column membership, each word's `index`, the fitted extents, `outcome`,
`agreement`, `variants`, and the provenance counts `n_words`, `bracket_overrides_n`,
`n_words_time_shifted`. Every `word` entity in the store is replaced, not amended.

### What is not derivable is everything the words fed

**Words propose spans.** `_spans` takes the extents of the *non-bracketed* words, groups them with
`group_extents_into_runs`, and keeps those that overlap nothing already proposed
(`preprocess.py:1345-1357`). Re-bracketing removes words from that set, so a run either disappears,
shortens, or splits — and three further things move with it:

* candidates that *overlapped* an existing span are not discarded but recorded as `corroborated_by`
  entries on it, so amplitude spans' own attributes change even where no span is added or removed;
* `measure="gap"` spans are cut against the covered set, so losing an ASR span widens or creates a
  gap span;
* if ASR was the only source that proposed anything, the block falls into its `spans_no_contrast`
  branch and `span_hear`, `span_yamnet` and `squim` all become absent.

**Changed spans need models.** `span_hear` runs HeAR per span; `span_yamnet` runs YAMNet per span
(only the sub-window case is re-derivable, from the stored whole-file windows); `squim` runs SQUIM
per span. A span whose extent moved has no stored measurement that belongs to it, and no arithmetic
over the store produces one.

That is the end of the argument. A driver that rewrote the words would have to retire every per-span
measurement over a moved span and could not replace it — the store would come out of the pass
holding fewer measurements than it went in with, which is not "updated" in any sense a consumer
would accept.

### And the rest of the graph reads them too

For completeness, since it decides what a re-bracketing pass would have to be even if the spans were
free: TAXONOMY's `_lexical_line` and `_transcribed_span_ids` read `lexical_words`, so the `kind`
entities change and ROUTING's branch selection with them; AIRWAY skips transcribed spans and flags
lexical contamination; SPEECH derives its diarization interval from the first to the last lexical
word and its PII haystack from the consensus text, so its `speaker` entities, embeddings, enrollment
match and PII findings all rest on it. Diarization, embeddings and PII are models.

Re-bracketing the corpus is therefore a **re-run of the graph from the consensus step onward**, not
an extend. It is cheaper than a full PREPROCESS — the ASR hypotheses, the enhancement, the whole-file
classifier sidecars, the envelopes and the spectrograms all survive — but it is a different program
from this one, and it is not written here. A driver that quietly did the derivable half would leave a
store whose words and whose spans disagreed about the same recording, which is worse than a store
that is honestly out of date.

### One consolation

`config.py` hashes the whole merged mapping, and `words.onomatopoeic_tokens` is inside it. A corpus
store is stamped with the old `config_hash`, so a run made under the null vocabulary and one made
under the cough lexicon are distinguishable by identity rather than silently conflated.

## One pass, and the order within it

One driver, one pass, because both remaining derivations are independent of each other and of
everything else, and two passes would read and rewrite every store twice.

* Phonation tracks read `plain` and the pre-emphasised stream. The consensus taxonomy reads
  `span_yamnet` and `span_hear`. Neither reads the other's output, in either direction.
* Neither depends on the words, which is what lets the driver ship while re-bracketing does not. The
  consensus taxonomy is a consolidation of per-span classifier scores and has never read a word; the
  phonation block stopped depending on the consensus transcript when span *detection* moved out of
  it into TAXONOMY.

So the order inside `extend_one` is fixed for determinism and nothing else. Had re-bracketing been
included it would have had to run **first** — it changes the spans the consensus taxonomy
consolidates — which is a further reason the three do not compose into one pass as they stand.

## Convergence, for the pass as a whole

The driver does not decide to skip per derivation. It takes `store.fingerprint()` before applying
both, applies them, and writes the store, re-exports `prov/` and records the environment **only if
the fingerprint moved**.

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

   **One reader in the tree does not filter.** `routing_analysis/features.py`'s `extract_features`
   collects the invalidated ids while streaming the JSONL and applies them to `word` and `span`
   records, but `_absorb_measurement` is called for every measurement record whatever its state. It
   would therefore absorb both readings, and for `consensus_taxonomy` it *assigns* each peak rather
   than taking a maximum, so the record written later in the file wins. `write_jsonl` emits entities
   in insertion order and the replacement is inserted after the retirement, so the current reading
   does win — by write order, which is not a property anyone designed. Nothing in the corpus is
   affected today: `extract_features` reaches a live store only through TAXONOMY's `ruleset_routing`
   measurement, and no corpus store carries one. A store that is both extended and re-routed is the
   case to fix first, in that reader, by filtering measurements the way it already filters words.
3. **The generating activity does not separate the two readings.** Its parameters are unchanged by
   the merge-rule change, so both measurements hang off the same activity id. The retirement
   activity, `TAXONOMY/consensus_taxonomy_superseded`, is the only record that says which is which,
   and the only edge into it comes from the retired measurement.
4. **The words are the corpus's own.** An extended store still spells transcribed coughs as lexical
   words, its spans still include the ones those words proposed, and its routing still reflects
   them. The extension did not touch that and does not claim to.
5. **The `prov/` tree was re-exported from the merged store.** BEP028 carries entity attributes
   through verbatim, including the invalidation, so the exported graph agrees with `store.jsonl`.

## Verification

`src/tests/scripts/extend_reprocessed_outputs_test.py`, over synthetic finished runs with real FLAC
streams and real per-span scores. Both derivations land and are correct; the superseded
consolidation is no longer live and the retirement names why; a run already carrying both forms is
skipped with a byte-identical store and an unchanged fingerprint and gains no retirement edge; three
passes produce exactly one retirement; the extend path's phonation measurement equals a fresh
pass's; `prov/` is re-exported carrying both; a store with no conditioned stream records the failure
and still has its taxonomy made current; a store with no consolidation is given none.
