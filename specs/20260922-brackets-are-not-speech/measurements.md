# What the bracket artefact costs, measured

Corpus: the replayed run at `/orcd/scratch/bcs/002/satra/triage_replay_20260922/out/`, read-only.
Each recording's `run/store.jsonl` was read with `ProvStore.read_jsonl` and reduced with the
graph's own liveness rule — `live_entities`, which is `wasInvalidatedBy`-aware — so every count
below is over the **live** generation only. Job `23526646` on `pi_satra`, 32 workers, no read
errors on any store.

Reproduce with `measure-corpus.py <out.jsonl> <workers>` (one row per recording, run under the
replay venv with `PYTHONPATH` at a checkout of this commit), then `summarise.py <out.jsonl>` and
`summarise-bounds.py <out.jsonl>`, both beside this file.

**Population.** 17,980 recordings carry a `REDACT` activity. 17,925 of them carry at least one live
`pii` entity, holding 115,753 findings between them. The brief's figure of 17,924 matches the
second of these to within one, not the first; shares below are given against 17,980, and against
17,924 they differ in the second decimal.

No transcript text and no detected string appears here. `[UH]` and its siblings are transcription
conventions, not anyone's data.

## 1. Findings that overlap a bracketed consensus word

**3,852 recordings — 21.42% of the 17,980 where REDACT ran, 21.49% of the 17,925 that have any
finding.** 9,457 of the 115,753 findings (8.17%) overlap at least one bracketed word; 4,228 (3.65%)
overlap **nothing but** bracketed words and so cannot have been raised by speech at all.

By declared family, the recordings hit over the recordings where REDACT ran:

| family | hit | of | share |
| --- | ---: | ---: | ---: |
| open-response-questions | 83 | 127 | 65.4% |
| story-recall | 510 | 804 | 63.4% |
| respiration-and-cough-cough | 11 | 20 | 55.0% |
| free-speech | 1121 | 2195 | 51.1% |
| random-item-generation | 96 | 196 | 49.0% |
| respiration-and-cough-v2-hardcough | 7 | 15 | 46.7% |
| random-item-generation-v2 | 62 | 137 | 45.3% |
| maximum-phonation-time | 38 | 88 | 43.2% |
| free-speech-v2 | 706 | 1700 | 41.5% |
| respiration-and-cough-fivebreaths | 10 | 28 | 35.7% |
| picture-description-option2 | 34 | 97 | 35.1% |
| prolonged-vowel | 147 | 430 | 34.2% |
| maximum-phonation-time-v2 | 10 | 30 | 33.3% |
| respiration-and-cough-threequickbreaths | 8 | 24 | 33.3% |
| picture-description | 105 | 335 | 31.3% |
| productive-vocabulary | 404 | 1375 | 29.4% |
| cinderella-story | 65 | 230 | 28.3% |
| glides-high-to-low | 12 | 62 | 19.4% |
| glides-low-to-high | 17 | 89 | 19.1% |
| rainbow-passage | 9 | 52 | 17.3% |
| animal-fluency | 15 | 104 | 14.4% |
| word-color-stroop | 35 | 264 | 13.3% |
| story-recall-v2 | 63 | 487 | 12.9% |
| diadochokinesis-v2-kuh | 49 | 412 | 11.9% |
| harvard-sentences-list | 49 | 1030 | 4.8% |
| diadochokinesis-ka | 22 | 533 | 4.1% |
| loudness | 4 | 47 | 8.5% |
| the remaining 20 families | 143 | 8,196 | under 3% each |

Only `high-to-low` is untouched. The gradient is the one the mechanism predicts: the families that
invite free speech carry the most filler, and the sustained-phonation and respiration families
carry almost nothing **but** filler, which is why a third to a half of them are hit despite having
few words at all.

## 2. Finding durations, against the words they cover

Seconds. "hull" is the union of the `extent`s of the consensus words the finding overlaps.

| population | n | p25 | median | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| every finding | 115,753 | 0.49 | 0.80 | 1.36 | 3.20 | 4.76 | 27.46 | 301.30 |
| no bracketed word overlapped | 106,296 | 0.52 | 0.80 | 1.34 | 2.87 | 4.34 | 9.04 | 120.76 |
| only bracketed words overlapped | 4,228 | 0.16 | 0.28 | 0.46 | 0.88 | 2.72 | 7.52 | 34.64 |
| covered-word hull | 111,516 | 0.61 | 1.05 | 1.67 | 2.84 | 4.08 | 26.54 | 301.29 |

Ratio of finding duration to covered hull:

| population | n | p25 | median | p75 | p90 | p95 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| every finding | 111,516 | 0.68 | 0.98 | 1.09 | 1.35 | 1.67 | 3.35 | 431.6 |
| no bracketed word | 102,062 | 0.67 | 0.95 | 1.09 | 1.34 | 1.65 | 3.28 | 431.6 |
| any bracketed word | 9,454 | 0.91 | 1.00 | 1.02 | 1.41 | 1.81 | 5.59 | 234.0 |

**The expectation going in was wrong, and this is the finding that matters most.** A finding's
duration tracks the words it covers: the median ratio is 0.98, and at p90 it is 1.35. The long tail
in the ratio is *heavier in the bracket-free population* than in the bracketed one. And the
bracket-only findings are **shorter** than average — median 0.28 s against 0.80 s overall — because
a filler token is one short word and `_timings_hull` gives it only its own sources' timings.

The long absolute tail is a third mechanism, not this one. Of the 2,153 findings over 10 s, **1,288
cover every consensus word in the recording**: that is the deliberate *"a finding nothing in the
transcript places covers the whole of it"* widening in `speech.py`, whose ratio is 1.00 by
construction. 8,306 findings (7.18%) take that path at any length, median 4.35 s. No length or
ratio rule separates them from a genuine long finding; the discriminator is whether `_locate`
placed the finding, which the code already knows and does not record on the entity. That is a
separate item.

Words covered per finding: 2 (32,060), 3 (25,990), 1 (25,497), 4 (10,430), 5 (4,390), 0 (4,182),
6 (2,372). The 4,182 with no overlapping word are findings whose per-source timing hull falls
outside every fitted consensus `extent` — 3.6%, and also not this defect.

## 3. Findings reaching the recording's own task extent

16,989 recordings have a `task_extent` span at all. **16,573 — 92.17% of the 17,980 — have a live
finding whose extent overlaps one.** That number is too blunt to be the population the brief wanted:
in a free-speech recording the task extent *is* the speech, so a genuine name inside it overlaps by
definition, and redaction reaching it is redaction working.

The subset the defect owns:

- 3,373 recordings where a task-extent-reaching finding overlaps a bracketed word.
- **1,829 recordings where a finding raised on nothing but bracketed words reaches the task extent.**
- **469 recordings where *every* task-extent-reaching finding is bracket-only** — that is, the whole
  of the destruction of task content in that recording is this artefact and nothing else. By family:
  free-speech 158, prolonged-vowel 83, productive-vocabulary 72, free-speech-v2 59,
  picture-description 20, open-response-questions 18, maximum-phonation-time 15, story-recall 9,
  and 12 more in single digits.

2,732 seconds of audio sit inside bracket-only findings in total.

## 4. Categories, and where the artefact concentrates

| category | findings | overlap a bracket | bracket-only | bracket-only share |
| --- | ---: | ---: | ---: | ---: |
| PERSON | 60,229 | 7,161 | 4,166 | 6.92% |
| NAME | 32,192 | 520 | 33 | 0.10% |
| DATE_TIME | 10,321 | 704 | 1 | 0.01% |
| LOCATION | 6,607 | 402 | 22 | 0.33% |
| MISC | 1,759 | 68 | 1 | 0.06% |
| UNIQUE_IDENTIFIER | 1,345 | 19 | 0 | — |
| NRP | 1,214 | 31 | 3 | 0.25% |
| AGE | 858 | 354 | 0 | — |
| LOC | 830 | 157 | 1 | 0.12% |
| VEHICLE_IDENTIFIER | 231 | 3 | 1 | 0.43% |
| the remaining 11 categories | 167 | 38 | 0 | — |

**The artefact is one category and one detector.** 4,166 of the 4,228 bracket-only findings (98.5%)
are `PERSON`, and 4,134 (97.8%) come from `gliner/name`. The next detectors contribute 58
(`presidio`) and 35 (the `rules/*` family) between them. `NAME` — the same concept from a different
detector — produces 33, three orders of magnitude fewer, which is itself evidence that the
behaviour is one model's response to a short all-caps token rather than anything about the text.

By haystack, over findings overlapping a bracket: consensus 7,250, `asr_qwen` 1,938,
`asr_crisperwhisper` 269. All three are affected, which is why the fix filters every haystack and
not only the consensus one. CrisperWhisper emits bracketed filler natively; the Qwen positions are
ones the consensus vocabulary bracketed from an unbracketed raw surface.

## Would a duration bound have caught this? No.

Every bound, scored against the 4,228 bracket-only findings it should catch and the 106,296
bracket-free findings it would refuse:

| rule | bracket-only caught | share | bracket-free refused |
| --- | ---: | ---: | ---: |
| duration > 2 s | 253 | 6.0% | 15,006 |
| duration > 3 s | 198 | 4.7% | 10,105 |
| duration > 5 s | 111 | 2.6% | 3,490 |
| duration > 6 s | 85 | 2.0% | 2,245 |
| duration > 8 s | 31 | 0.7% | 1,326 |
| duration > 10 s | 17 | 0.4% | 911 |
| duration / hull > 1.5 | 489 | 11.6% | 6,881 |
| duration / hull > 2.0 | 140 | 3.3% | 2,450 |
| duration / hull > 3.0 | 93 | 2.2% | 1,147 |
| duration / hull > 5.0 | 69 | 1.6% | 649 |

No row is defensible. The best absolute bound refuses 31 bracket-free findings for every artefact
it catches, and the best ratio bound 12; both refusals are in the unsafe direction, because a
refused finding is a disclosure that is not redacted. The *reason* no row works is the second
table in §2: bracket-only findings are shorter than ordinary ones, so length is not the signature
of this defect at all. Even the recording in the brief — the one that motivated the bound — is a
6.32 s finding over a 3.38 s hull, a ratio of 1.87, which sits around p95 of the bracket-free ratio
distribution and is therefore not separable from it.

**(b) is not shipped.** No threshold, no config key, no `data/` profile. A mechanism that catches
2.6% of what it is for, at a cost of 3,490 refusals with no measured verdict behind any of them, is
worse than no mechanism: it would be an unmeasured decision with a public interface, and it would
make the next reader believe the long-finding problem had been dealt with when the actual long-tail
mechanism — the whole-transcript widening of an unlocatable finding, 8,306 findings and 1,288 of the
2,153 over ten seconds — is untouched by it.

## What could not be measured

- **The counterfactual for the mixed findings.** 5,229 findings overlap some bracketed and some
  lexical words. After the fix, those are scanned on a text with the bracketed tokens removed; what
  each detector then returns cannot be derived from the recorded runs and would need a re-run. Only
  the 4,228 bracket-only findings are certain to disappear.
- **Whether a bracket-only finding was ever a true positive.** Nothing in the corpus adjudicates a
  finding. The argument that they are all false is the mechanism — the token is a transcription
  convention, not something the participant said — and not a labelled comparison.
- **Whether REDACT's verification re-scan would now fail on the released brackets.** It is the same
  detector on the same token, so it follows; it was not re-run over the corpus, and that is why the
  re-scan change carries its own unit tests rather than a corpus number.
