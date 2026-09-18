# DDK — the syllable template, and why one global vowel set was wrong

What replaced `branch.ddk_vowel_phonemes`, what it fixed, and what it deliberately does not do.
Landed 2026-09-17, on the owner's decision of the same day.

The code is `src/senselab/audio/workflows/triage/nodes/ddk.py` and the expectation rows are in
`nodes/branches.py`; the operating points are `branch.ddk_nucleus_classes` and
`branch.ddk_stop_places` in `data/config/default.yaml`, with their derivations in
[`config-derivations.md`](config-derivations.md). The instrument this sits inside is
[`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md).

## The defect

`Expectation.sequence` held places — `("labial", "alveolar", "velar")` — and the CV walk admitted a
nucleus from one global list, `branch.ddk_vowel_phonemes: [aa, ah, ao, ow, uh, uw]`. **The nucleus is
stimulus-specific, so no global set fits all ten families.**

Measured on the corpus, 150 recordings per family, stop→next-run pairs off the PPG argmax raster:

- `/pa/`, `/ta/`, `/ka/`, `/pataka/` — nuclei are `ah` and `aa` throughout.
- **`buttercup` = /bʌ-tər-kʌp/**, whose three most common pairs are exactly its three syllables:
  `/k/→ah` 19.7%, **`/t/→er` 14.2%**, `/b/→ah` 10.1%. The middle nucleus is rhotic `er`, which no
  low-vowel set holds — so a global set found **2 of buttercup's 3 syllables**, on every recording,
  silently.
- Also visible and correct: `/p/→<silent>` 6.5% and `/p/→b` 4.0% in buttercup are the **coda** `p` of
  "cup". A consonant with no following nucleus yields no unit, which is the behaviour wanted;
  **coda modelling is deliberately absent.**

## What the template is

One entry per syllable position, carrying **both** the onset place and the nucleus class
(`branches.Syllable`, a `NamedTuple`):

| family | template |
| --- | --- |
| `/pa/`, `-v2-puh` | `(("labial","low"),)` |
| `/ta/`, `-v2-tuh` | `(("alveolar","low"),)` |
| `/ka/`, `-v2-kuh` | `(("velar","low"),)` |
| `/pataka/`, `-v2-puhtuhkuh` | `(("labial","low"), ("alveolar","low"), ("velar","low"))` |
| `buttercup`, `-v2-buttercup` | `(("labial","low"), ("alveolar","rhotic"), ("velar","low"))` |

`branch.ddk_nucleus_classes` maps class → phonemes, in the shape `branch.ddk_stop_places` already
uses for place → stops, so the two share one idiom and one accessor type.

## Extraction permissive, conformance positional

**Extraction** admits any nucleus in the **union** of the classes the declared template names
(`admitted_nuclei`), so all three buttercup syllables are found. **Conformance** is then checked
**per position** against the template — place and nucleus class both, as
`ddk_expected_place_fraction` and `ddk_expected_nucleus_fraction`, each with its own `by_position`.

`syllable_detail` reports the new one as `ppg_expected_nucleus_fraction`. **`report.py`'s
`BRANCH_MEASURES["SPEECH"]` does not name it yet**, so the summary does not print it; both readers
select with `if key in detail`, so nothing breaks and nothing is fabricated. Adding it, and removing
the stale `lexical_repetitions_n` entry beside it, is one line each in a file this change
deliberately did not touch.

This preserves the property the flat list had: `/pa/`, `/pah/` and `/paw/` all satisfy `low`, so
phonemic variation inside a class costs nothing. A substituted place or nucleus is a **finding**, not
a rejection — "buttercap" keeps all three units and scores 2/3 on the nucleus, which is the
clinically meaningful reading.

## `buttercup` stops being a lexical special case

With a three-position template, both `buttercup` rows are `SYLLABLE_SEQUENCE` and structurally
identical to `pataka`: `expected_event_count: 30` for the counted v1 row (10 cycles × 3 syllables)
and `declared_duration_s: 5.0` for the timed v2 row, exactly as `-pataka` and `-v2-puhtuhkuh` carry.

Deleted with it: `_repeated_word` and its `align_ddk` fork, `DdkReads.transcript_id` and the
`TRANSCRIPT` constant that fed it, the `production="lexical_repetition"` marker,
`ddk.lexical_repetitions_n`, and `"repetition"` from `TRAIN_ROLES`.

`ddk.lexical_repetitions_n` counted `role == "lexical_repetition"` while the only writer minted
`role="task_extent"` with `lexical_repetition` in **`production`**, so it had always been 0. That is
item 4 of [`branch-figure.md`](branch-figure.md), settled by deletion rather than by correction.
`"repetition"` was minted nowhere.

### Does anything still notice a participant who said nothing at all?

Asked before deleting the token match, because it was the only lexical test of "was the task
performed". **It is not the only thing that notices, and the syllable path answers the same
question on the same recording.** On silence the envelope carrier finds no amplitude span clearing
`train_min_s`, so `ddk_carrier` returns `(None, None)` and `done` is `False`; the CV walk finds no
stop run, so `PpgReading.train` is None and `_with_ppg` leaves that `False`. Pinned by
`ddk_test.py::TestButtercupIsAThreeSyllableTemplate::test_a_participant_who_said_nothing_at_all_is_still_noticed`.

What the deletion does change is a participant who said *something else*: the token match answered a
flat `False`, and the template answers with a per-position conformance finding instead. That is the
owner's stated intent — a substitution is a finding, not a rejection.

## `Pattern.ORDERED_TOKENS`

Checked, not assumed: after this change **no syllable family uses it**. The member stays, because
SPEECH's own lexical families use it heavily — `harvard-sentences-list`, `cape-v-sentences`,
`cape-v-sentences-v2`, `rainbow-passage`, `caterpillar-passage`, `word-color-stroop`, `loudness`,
`loudness-v2` — eight rows. `align_ddk` no longer branches on a pattern at all.

## What was measured and not implemented

Two readings on 1,800 recordings at `tol=1.5`:

- Widening the global set from `[aa ah ao ow uh uw]` to the low vowels `[aa ae ah ao aw ay]` moved
  sensitivity 0.633 → 0.693 at unchanged specificity (0.950 → 0.949). `ow`, `uh` and `uw` never
  appear in the top 14 post-stop nuclei — they were dead weight.
- Requiring the nucleus run to be **strictly adjacent** to the stop run costs sensitivity
  (0.693 → 0.605) and shortens median trains 6 → 5, buying specificity 0.949 → 0.995. **Not
  implemented.** Since the DDK dissolution there is no DDK routing, so the instrument only ever runs
  in-family on the ten declared families and specificity is close to irrelevant. The intervening-run
  gap is admitting genuine units.

`low` is the six a-initial ARPAbet vowels — the low and open-mid nuclei. **The category is
principled: it is read off the phonetic category, not fitted.** The corpus numbers above are
confirmation that it is the right category, not the derivation of its membership.
