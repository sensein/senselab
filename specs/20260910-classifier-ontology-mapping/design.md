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

## Open

- Nothing has been measured about whether widening `labels_of_interest` beyond `Cough` and `Breathe`
  improves anything. The map no longer blocks it; no evidence yet says to do it.
- The `Sigh` firing spread above has not been run, so the root list stays at one entry.
