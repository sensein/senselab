# Cross-classifier corroboration by ontology identity — 2026-09-10

AIRWAY corroborated a HeAR label with a YAMNet label by exact string membership in a hand-written
map that covered two of HeAR's eight labels. The map is replaced by a generated profile that
resolves every HeAR label onto AudioSet ontology nodes and corroborates from the node's subtree.

## The defect

`airway.py` read `airway.confirmation_map` and tested `yamnet_label in confirmation_map.get(label,
set())`. The shipped map was two entries:

```yaml
confirmation_map:
  Cough: [Cough]
  Breathe: [Breathing, Sigh, Gasp]
```

HeAR emits eight labels (`hear.py`, `HEAR_EVENT_LABELS`, in the detector's graph order): `Cough`,
`Snore`, `Baby Cough`, `Breathe`, `Sneeze`, `Throat Clear`, `Laugh`, `Speech`. Six had no entry, so
`.get(label, set())` returned empty and no YAMNet label could ever confirm them. The branch's
`labels_of_interest` defaulted to `[Cough, Breathe]`, which hid the consequence: widening it by one
label — the config's documented way for a campaign to screen for sneezes — produced a label that
could only ever abstain.

Three of the six failed on spelling alone:

| HeAR label | AudioSet class | why the string test missed |
| --- | --- | --- |
| `Throat Clear` | `Throat clearing` | capitalisation and word form |
| `Snore` | `Snoring` | word form |
| `Baby Cough` | `Cough` | AudioSet has no infant cough class |

`Sneeze` and `Speech` matched by string but had no entry at all. `Laugh` needs `Laughter`.

A second defect ran the other way. `Breathe`'s entry named `Sigh`, which AudioSet does **not** place
under `Breathing`: `Sigh` (`/m/07plz5l`) is a child of `Human voice`. It confirmed breaths for no
reason the ontology supports.

## What replaced it

One generated profile, `data/classifier_ontology/2026-09-10.json`, built by
`scripts/build_classifier_ontology_map.py` and never hand-edited. Re-running the script against the
same pins reproduces it byte for byte.

The profile carries the whole AudioSet node table (632 classes: name, `child_ids`, `restrictions`,
and whether the class is in the released 527 and in YAMNet's 521), the eight HeAR labels in graph
order, and one mapping entry per label: the AudioSet ids it denotes, the subtree closure of those
ids, the closure's display names, which of them are unreleased, which YAMNet cannot report, and a
free-text note where the mapping is not one-to-one.

Corroboration is subtree membership, not equality: a HeAR label is corroborated by its mapped node
**or any descendant of it**. `Throat clearing` is AudioSet's only child of `Cough`, so a throat
clear corroborates a cough without either name being written down twice.

`senselab.audio.workflows.triage.classifier_ontology` loads and validates the profile.
`routing_analysis/labels.py`'s `AUDIOSET_COUGH`, `AUDIOSET_BREATH`, `HEAR_COUGH`, `HEAR_BREATH` and
`HEAR_AIRWAY` are now derived from it rather than listed; they were the same duplication in a second
place.

### The one authored input

HeAR publishes no ontology, so the crosswalk from its eight labels to AudioSet class names is
authored — it is `CROSSWALK` in the generator, eight rows, checked against `HEAR_EVENT_LABELS` at
build time. Everything else in the profile is derived from a pinned source.

## What was fetched, and how it is pinned

Every artifact is addressed by commit, never by branch, and its SHA-256 is recorded and re-verified
on each build; a mismatch aborts rather than writing a profile whose provenance is confidently
wrong.

| artifact | repository | commit | sha256 | retrieved | licence |
| --- | --- | --- | --- | --- | --- |
| `ontology.json` | `audioset/ontology` | `d417d32bf59c711abb5910fd2f76a0eb44697991` | `9c685f44…d1f2d15d` | 2026-09-10 | CC BY 4.0 |
| `egs/audioset/data/class_labels_indices.csv` | `YuanGongND/ast` | `9fc5b67075f6c59a84c7931c4b5a0bf60c1416c6` | `cdd10498…99708429` | 2026-09-10 | CC BY 4.0 |
| `research/audioset/yamnet/yamnet_class_map.csv` | `tensorflow/models` | `dfffd623b6be8d1d9744b8e261fbac370d17c46d` | `cdf24d19…787c6df2` | 2026-09-10 | Apache 2.0 |

Attribution, as the licences require: **AudioSet ontology and released class index, Google LLC,
licensed CC BY 4.0**; **YAMNet class map, TensorFlow Model Garden, licensed Apache 2.0**. Both lines
are in the profile beside the artifact they cover.

`audioset/ontology` has had no commit since 2017-03-08 and `d417d32` is its HEAD; the two CSVs are
pinned to the last commit that touched each file, so the pin guarantees the bytes.

The AST list was taken from the AST repository rather than from
`storage.googleapis.com/us_audioset/…`, which serves no version identifier at all. Its 527 rows are
AudioSet's released class list.

## The eight resolutions

| HeAR label | group | AudioSet root | corroboration set (root first, then subtree) |
| --- | --- | --- | --- |
| `Cough` | cough | `Cough` | Cough, Throat clearing |
| `Snore` | breath | `Snoring` | Snoring |
| `Baby Cough` | cough | `Cough` | Cough, Throat clearing |
| `Breathe` | breath | `Breathing` | Breathing, Wheeze, Snoring, Gasp, Pant, Snort |
| `Sneeze` | cough | `Sneeze` | Sneeze |
| `Throat Clear` | cough | `Throat clearing` | Throat clearing |
| `Laugh` | laugh | `Laughter` | Laughter, Baby laughter, Giggle, Snicker, Belly laugh, Chuckle chortle |
| `Speech` | speech | `Speech` | Speech, Male speech man speaking, Female speech woman speaking, Child speech kid speaking, Conversation, Narration monologue, Babbling, Speech synthesizer |

Every one of the 632 ontology nodes has a unique `name`, so the name↔id correspondence the profile
relies on is a bijection; the generator refuses to build if that ever stops being true.

Three expectations in the brief did not survive contact with the ontology and are recorded here
rather than encoded:

- **`Snort` is not a descendant of `Snoring`.** Both `Snoring` and `Snort` are children of
  `Breathing`; they are siblings. `Snore`'s set is therefore `{Snoring}` alone.
- **`Sigh` is not under `Breathing`.** It is a child of `Human voice` and corroborates nothing.
- **`Sniff` is not under `Breathing`.** It is a sibling of `Breathing` under `Respiratory sounds`
  and corroborates nothing.

`Sigh` and `Sniff` were both in the hand-listed `AUDIOSET_BREATH`; the derived tuple drops them and
gains `Pant` and `Snort`. `AUDIOSET_COUGH` is unchanged in content — `(Cough, Sneeze, Throat
clearing)` — because the hand list happened to be the correct closure.

## The overlaps, recorded rather than deduplicated

`Snoring` is a descendant of `Breathing`, so a YAMNet `Snoring` window corroborates **both** HeAR
`Snore` and HeAR `Breathe`. Nothing trims either set to make them disjoint: the ontology says a
snore is a kind of breathing, and a reader of two confirmations on one window must be able to see
that they came from one AudioSet class rather than two independent ones. The profile's `overlaps`
block enumerates every such pair.

| labels | shared |
| --- | --- |
| `Cough`, `Baby Cough` | Cough, Throat clearing |
| `Cough`, `Throat Clear` | Throat clearing |
| `Baby Cough`, `Throat Clear` | Throat clearing |
| `Snore`, `Breathe` | Snoring |

## YAMNet's 521 against AudioSet's 527

YAMNet's `yamnet_class_map.csv` carries 521 rows, not 527, and the six it omits are not a subset
anyone had written down here. Compared by machine id, YAMNet is a strict subset of the AST list —
there is no class YAMNet has that AST lacks — and the six missing are:

`Battle cry`, `Female singing`, `Female speech, woman speaking`, `Funny music`, `Male singing`,
`Male speech, man speaking`.

Two of those, `Male speech, man speaking` and `Female speech, woman speaking`, are in HeAR
`Speech`'s corroboration set. **YAMNet can never emit either**, so those two entries are inert when
the corroborating classifier is YAMNet; AST can emit them. The profile records them per label under
`not_emittable_by_yamnet` rather than dropping them, because the same sets are read for AST.

No cough- or breath-group label depends on a class YAMNet cannot report.

## Configuration

`airway.confirmation_map` is gone. Pre-alpha: renamed and replaced, no alias.

```yaml
airway:
  labels_of_interest: [Cough, Breathe]
  corroboration_profile: null      # path to a profile; null takes the packaged latest
  corroboration_overrides: null    # HeAR label -> AudioSet class names, REPLACING that label's derived set
  contest_labels: null
```

`corroboration_overrides` replaces `airway.confirmation_map` in `config.DATA_MAP_PATHS` and keeps
the extensibility that path existed for: a campaign may state its own corroboration for a label
without editing the installed package. It ships null, and an override naming a HeAR label the
profile does not map is refused as a typo rather than accepted as an extension.

`labels_of_interest` **was not widened**. It stays `[Cough, Breathe]`, which is a separate decision
with no measurement behind it yet; what changed is that the other six labels are no longer silently
unconfirmable if someone does widen it.

`contest_labels` checks disjointness against the AudioSet airway evidence set, which is now derived
rather than listed — see the next section.

## The airway kind, from the same ontology — 2026-09-11

The section above closed the corroboration duplication and left the airway *kind* hand-listed in two
places, as its own Open item. This closes that one the same way, and the two definitions that had
already drifted are what it cost.

### The second defect

`routing_analysis/labels.py`'s `AUDIOSET_AIRWAY` was a nine-name tuple and
`taxonomy.audioset_airway_labels` restated the same nine names in `data/config/default.yaml`. Two
definitions of one thing, read by different nodes, with nothing holding them together.

Measured against the shipped profile's `Respiratory sounds` subtree, both were wrong in the same
three ways:

| finding | why |
| --- | --- |
| `Sigh` was in the set and is not respiratory | AudioSet places `Sigh` (`/m/07plz5l`) under `Human voice` |
| `Pant` and `Snort` were missing | both are direct children of `Breathing`, both `in_yamnet_521` |
| `Sniff` is in the subtree but in neither HeAR-mapped closure | it is a sibling of `Breathing` under the root, not a descendant of `Cough` or `Breathing` |

The third finding is why `AUDIOSET_COUGH ∪ AUDIOSET_BREATH` is **not** the airway set even though it
is derived and nearly the same shape. That union answers a corroboration question about HeAR's own
labels; the airway kind is a question about the ontology, and asking it of HeAR's mapped roots loses
every respiratory class HeAR has no label for.

### What replaced it

One configuration key, naming ontology **roots** rather than labels:

```yaml
taxonomy:
  airway_ontology_roots: [Respiratory sounds]
```

Both evidence vocabularies are read off that one closure, so they cannot disagree:

* the AudioSet set is the subtree closure of the roots, minus what no classifier can emit;
* the HeAR set is every HeAR label whose mapped AudioSet node falls inside the same closure.

`taxonomy.audioset_airway_labels` and `taxonomy.hear_airway_labels` are both gone. Pre-alpha: one key
replaces two, no alias. `classifier_ontology.airway_audioset_labels` and `airway_hear_labels` are the
readers, and `nodes/taxonomy.py`, `nodes/airway.py` and `routing_analysis/labels.py` all go through
them. A root that is not an AudioSet display name in the profile is refused, because a misspelled
root would silently shrink the kind rather than fail.

The HeAR set is unchanged in content — the same six labels — but it is now derived from the roots
rather than from the two crosswalk groups, so widening the roots widens both sides at once.

### Emittability, as a rule rather than a name

`Respiratory sounds` is `in_audioset_527: false` and `in_yamnet_521: false`: it is an abstract
ontology node no classifier has an output for. A closure that included it would put a name in a set
the detectors match against that can never match.

The closure therefore drops every node in neither the released 527 nor YAMNet's 521, checked on the
node's own flags. Naming `Respiratory sounds` as the exception would have been a literal that the
next abstract root reintroduces the defect around.

### Which labels moved

| set | before | after |
| --- | --- | --- |
| AudioSet airway | Cough, Throat clearing, Sneeze, Sniff, Breathing, Wheeze, Snoring, Gasp, Sigh | Breathing, Cough, Gasp, **Pant**, Sneeze, Sniff, Snoring, **Snort**, Throat clearing, Wheeze |
| HeAR airway | Cough, Snore, Baby Cough, Breathe, Sneeze, Throat Clear | unchanged |

### The `Sigh` trade-off

Dropping `Sigh` is a behaviour change, not a tidy-up, and it is the one part of this that the
ontology decides against clinical intuition. A sigh is an airway event to a clinician; AudioSet
files it under `Human voice`, arguably because its taxonomy is organised by production rather than
by physiology. `Sigh` is `in_yamnet_521`, so this is a label YAMNet actually emits — the change is
observable, not theoretical.

It is dropped anyway, because the whole point of the section above is that a set with one derivation
beats a set that is nearly right for reasons nobody wrote down. A departure from the ontology needs a
measurement behind it.

Three readers see the difference:

* **TAXONOMY's airway `acoustic` line.** A span whose only YAMNet label is `Sigh` no longer counts as
  airway acoustic evidence, and one labelled `Pant`, `Snort` or `Sniff` now does. The airway state
  itself folds from the `health_acoustic` line, so this moves the reported evidence count and the
  line's state, not the kind's verdict.
* **AIRWAY's `contest_labels` disjointness check.** `Sigh` may now be declared a contest label;
  `Pant`, `Snort` and `Sniff` may not. `Sigh` corroborated nothing already — the section above
  removed it from `Breathe`'s set — so contesting is now the only role it can hold.
* **The routing analysis.** `AUDIOSET_AIRWAY` is the `airway` family in `FAMILIES`, so the
  `airway.<classifier>_peak.<stream>` detectors take their maximum over the new set, and
  `TRACKED_LABELS` keeps `Pant`, `Snort` and `Sniff` peaks and stops keeping `Sigh`'s.

Adding it back is one line, naming it as a second root:

```yaml
taxonomy:
  airway_ontology_roots: [Respiratory sounds, Sigh]
```

What would justify that line: a firing spread on the triage corpus showing that YAMNet `Sigh`
windows land on spans a HeAR airway label also covers — that is, that `Sigh` behaves like the rest of
the subtree rather than like the voice classes it is filed with. Absent that, adding it widens the
kind on an intuition, which is what the nine-name tuple was.

## The consensus taxonomy, from the same ontology — 2026-09-12

The two sections above fixed corroboration in AIRWAY and the airway kind in TAXONOMY. Both were
described at the time as closing the string-matching defect. They closed half of it: the same defect
survived in `_write_consensus_taxonomy`, which is a second place where two classifiers are merged
and was still merged on the exact label string.

### The third defect

`consensus_taxonomy` is one row per label with `peak`, `peak_by_classifier`, `classifiers` and
`n_classifiers`, folded from `PER_SPAN_CLASSIFIERS` = `{yamnet: span_yamnet, hear: span_hear}`. The
fold keyed its rows on the label string each classifier emitted, so HeAR's spelling and AudioSet's
spelling of one event were two rows, each reading `n_classifiers: 1`:

| HeAR row | AudioSet row | the one node both denote |
| --- | --- | --- |
| `Baby Cough` | `Cough` | `Cough` (`/m/01b_21`) |
| `Snore` | `Snoring` | `Snoring` (`/m/01d3sd`) |
| `Breathe` | `Breathing` | `Breathing` (`/m/0lyf6`) |
| `Throat Clear` | `Throat clearing` | `Throat clearing` (`/m/0dl9sf8`) |
| `Laugh` | `Laughter` | `Laughter` (`/m/01j3sz`) |

Five of HeAR's eight labels could therefore never reach `n_classifiers: 2` however loudly YAMNet
agreed with them. `Cough`, `Sneeze` and `Speech` merged, and only because HeAR happens to spell
those three the way AudioSet does — a coincidence of the authored crosswalk, not a property the fold
relied on.

### Identity is the mapped node, and never its subtree

Corroboration and identity are two different questions asked of the same profile, and the fix is not
to reuse `corroboration_sets` here. Corroboration is directional and closed over descendants: does
this AudioSet name fall inside that HeAR label's subtree. Identity is which single node a spelling
denotes, and it is the mapped root alone.

Merging a table on subtree membership would be wrong twice over. `Throat clearing` is in `Cough`'s
closure, so it would have to fold into the `Cough` row — and YAMNet emits both names, which are two
different AudioSet classes that the released 527 keeps apart. And `Snoring` is in both `Snore`'s
closure and `Breathe`'s, so one YAMNet window would enter two rows and be counted twice.

`classifier_ontology.canonical_names` is the identity read: every AudioSet display name maps to
itself, and every HeAR label maps to the display name of the single node its crosswalk entry denotes.
A HeAR label denoting several nodes at once is absent from the map, because no one node names it; it
keeps its own spelling and its own row. The profile's names are a bijection over its 632 classes, so
one map covers both vocabularies in one namespace.

Three HeAR labels — `Cough`, `Sneeze`, `Speech` — are also AudioSet display names. All three denote
exactly the class they are spelled as, so the namespace is consistent. A profile where that stopped
being true would silently merge two different events into one row, so the loader now refuses one: a
mapped label whose own spelling is an AudioSet class name must denote that class.

### The merged row is named by the ontology

A merged row needs one name, and it is the AudioSet display name: `Throat clearing`, not
`Throat Clear`.

The ontology is the authority everywhere else in this document — the airway kind is a closure over
its nodes, corroboration is subtree membership in it, and the crosswalk exists only to get HeAR's
labels *into* it. The crosswalk is also the one authored input the profile carries, so its spellings
are the least derived strings in the system; naming a derived row after one of them would make a
hand-typed string the identity of a measured thing. Naming the row by the node also means a reader
who has the profile can resolve the row, and one who does not still gets the name the released
527-class list uses.

Each classifier's own spellings are not discarded: the row carries `labels_by_classifier`,
`{classifier: the spellings of its own that reached this node}`. That is what a consumer matching on
a classifier's native vocabulary reads, and it is also where the `Cough`/`Baby Cough` merge is
visible rather than inferred.

### Overlaps are counted once, by construction

The `overlaps` block records two shapes, and identity-merging handles both without a deduplication
pass:

* **One node, two HeAR roots.** `Snoring` is `Snore`'s mapped node and also a descendant of
  `Breathe`'s. Under identity it is one row, `Snoring`, which HeAR reaches through `Snore`; `Breathe`
  is its own row, `Breathing`. A YAMNet `Snoring` window lifts `n_classifiers` on one row, once. It
  is subtree merging that would have double-counted here, which is the argument for identity stated
  from the other end.
* **Two HeAR labels, one node.** `Cough` and `Baby Cough` both denote `Cough`. They fold to one row,
  and because `n_classifiers` counts *classifiers* rather than labels, HeAR contributes one entry to
  `peak_by_classifier` — the higher of its two scores — not two. Both spellings appear under
  `labels_by_classifier`.

Resolution happens per span, before consolidation. Folding after consolidation would have left two
medians and two span counts per classifier per row with no defined way to combine them; folding
before means `_consolidate` computes `peak`, `median` and `n_spans` over the node directly.

### Labels outside the mapping

Most of what YAMNet emits has no HeAR counterpart, and nothing may drop it. The identity map is total
over the profile's AudioSet names, so every such label maps to itself and keeps its row with
`n_classifiers: 1` — the fold's output is the same size it was for them. A spelling in neither
vocabulary — a classifier reporting a name the profile does not hold — is carried through unchanged
rather than dropped, so an unrecognised label is visible as a row instead of vanishing.

### AST is not widened

`PER_SPAN_CLASSIFIERS` stays `{yamnet, hear}`. AST runs whole-file in this pipeline and writes no
`span_ast` measurement, which is why the three `*.ast_peak.consensus` detectors read a constant 0 and
were removed. Adding AST here is a PREPROCESS change — a per-span AST pass that does not exist —
and belongs to whoever measures whether it is worth its cost.

### Configuration

One profile per run, named once. `airway.corroboration_profile` is gone;
`taxonomy.classifier_ontology_profile` replaces it. Pre-alpha: renamed and replaced, no alias.

```yaml
taxonomy:
  classifier_ontology_profile: null   # path to a profile; null takes the packaged latest
  airway_ontology_roots: [Respiratory sounds]
```

Every ontology read in a run now goes through that one key: AIRWAY's corroboration, the airway
closure, and the consensus taxonomy's label identity. It sits beside `airway_ontology_roots` because
both are ontology configuration and neither belongs to one branch; a profile named under `airway:`
and read by TAXONOMY was a key whose section no longer said who read it. The resolved profile's
filename is recorded in the `consensus_taxonomy` activity's parameters, so a run says which ontology
decided its merges.

### Which readers see the difference

`consensus_taxonomy` has one consumer in the tree, `routing_analysis/features.py`, which reads
`peak_by_classifier` and keeps a peak only when the row's `label` is in `TRACKED_LABELS[classifier]`.
`TRACKED_LABELS["hear"]` is built from HeAR's own spellings, so the rename moves four keys out from
under that gate:

| feature key | before | after |
| --- | --- | --- |
| `consensus\|hear\|Baby Cough` | emitted | folded into `consensus\|hear\|Cough`, which may rise |
| `consensus\|hear\|Snore` | emitted | absent — the row is now `Snoring` |
| `consensus\|hear\|Breathe` | emitted | absent — the row is now `Breathing` |
| `consensus\|hear\|Throat Clear` | emitted | absent — the row is now `Throat clearing` |
| `consensus\|hear\|Cough`, `\|Sneeze`, `\|Speech` | emitted | unchanged |
| `consensus\|yamnet\|*` | emitted | unchanged; YAMNet's labels are already AudioSet names |

One detector reads those keys: `airway.hear_peak.consensus`, whose maximum is taken over
`FAMILIES["airway"]["hear"]` — HeAR's six airway spellings. Four of the six now find nothing and
contribute 0.0, so the detector reads lower than it did. `speech.hear_peak.consensus` is unaffected
(`Speech` is spelled the same), `voice.hear_peak.consensus` is not built (HeAR has no voice label),
and every `*.yamnet_peak.consensus` is unchanged. No `peak_set` or `peak_label` detector reads the
consensus stream.

That gate is the routing analysis's to close, not this change's: `labels_by_classifier` is on the row
for exactly that, and matching it rather than `label` restores all six keys under HeAR's spellings
while the row stays named by the ontology. Until it is closed, `airway.hear_peak.consensus` is a
detector reading a narrowed set, and its sweep numbers are not comparable across this commit.

`nodes/figure.py` and `nodes/report.py` read `consensus_taxonomy` not at all, and no TAXONOMY line
folds from it — `voice`'s rework onto it is still pending — so no kind's state and no branch's route
moves. `n_labels` falls by the number of merges a recording produces.

## Open

- Nothing has been measured about whether widening `labels_of_interest` beyond `Cough` and `Breathe`
  improves anything. The map no longer blocks it; no evidence yet says to do it.
- The `Sigh` firing spread above has not been run, so the root list stays at one entry.
