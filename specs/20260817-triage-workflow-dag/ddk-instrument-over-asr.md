# DDK — the CV instrument is the authority, and the recogniser text is not

What this records, what it deliberately does not change, and why the safety constraint rules out
the one consumer the change was expected to reach. Landed 2026-09-19.

The code is `src/senselab/audio/workflows/triage/nodes/ddk.py`
(`cv_covered_extent`, `contradicted_words`, `instrument_authority`) and the one report key it adds
in `nodes/common.py` (`BRANCH_MEASURES["SPEECH"]`). No operating point is introduced; see
[the no-threshold ruling](#why-any-overlap-and-no-threshold).

## The problem

On a declared diadochokinesis recording the recognisers produce text, and the text is not a reading
of what was produced. Three consensus transcripts from the corpus:

```
Pap- papapapapapapapapapapapapapa...
Patica pataca patka potica patica pataca
Pay-paypay-pay-pay-pay-pay-pay-pay-pay.
```

Each is a recogniser forcing a lexical vocabulary onto nonsense syllables. Meanwhile the CV
instrument ([`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md)) has read the same
region off the phonetic posteriorgram: the CV units, their onsets, their places of articulation,
and how many complete cycles of the declared template they form. Over that region the CV
instrument knows what was produced and the recogniser does not.

Nothing in the store said so. The two readings sat side by side with no statement of which is
authoritative for this task, and a downstream reader taking the transcript at face value on a DDK
recording was taking nonsense syllables as words.

## The safety constraint, which is the frame and not a caveat

> A participant may say anything during a task, including something identifying, and a DDK task is
> no exception. Suppressing or replacing recogniser text *before* the PII scan would create a
> disclosure path.

The PII scan in `nodes/speech.py` step 7 is handed exactly two kinds of text: the consensus
transcript entity's own `text`, and each recogniser's own `transcript`. Its findings are located
back onto `word` entities and become `label`/`pii` assertions, which REDACT reads to plan the
released `consensus.json`, `transcript.txt` and `audio.wav`.

So the whole of this change is **additive**. It writes new entities. It invalidates no `word`, it
does not touch `consensus_transcript.text`, it does not touch any `asr_hypothesis.transcript`, and
it runs in a node — SPEECH — that is itself unconditionally routed for a declared DDK recording
(below). The set of strings reaching `scan_for_pii` is byte-for-byte what it was.

It also carries **no word text of its own**. AIRWAY's `off_task_extent` already established the
convention — "the word's text never enters the store here" — and for the same reason: a second
copy of a possibly-identifying string in an assertion payload is a second thing redaction would
have to cover, and REDACT plans off `word` entities only.

## The shape

Two records, both taken inside `ppg_evidence`, both travelling as ordinary `Finding`s through the
`write_findings` path SPEECH already calls.

**1. The instrument's reading, as its own record.** One `measure` finding,
`ddk_cv_instrument_reading`, over the extent the CV units span. Its value is the realised place
series — one entry per CV unit, which is the granularity the instrument measures at. Its
covariates carry the nuclei, the onsets, the unit count, the completed cycles, and
`authority="cv_instrument"` / `supersedes="lexical_transcript"`, which is the statement itself. It
is derived from the posteriorgram entity and from every consensus word it contradicts, so the join
runs in both directions.

**2. One `contest` per contradicted word.** Claim `lexical_transcript`, reason
`instrument_contradicted`, over the word's hull, derived from the word entity. The word is
untouched and stays live.

`contest` is the existing vocabulary for exactly this and was chosen over inventing one. QUALITY
already writes `verb="contest"` against a clip span it "answers but does not withdraw"
(`nodes/quality.py`), and `write_findings` turns a contest into an assertion beside the subject
rather than an edit to it. `deviation_names` ignores contests, so the branch report's `deviations`
tuple — which is what VERDICT folds — is unchanged by construction.

### The extent is the CV instrument's own, not the task extent

`align_ddk` proposes one `task_extent`, minted by whichever instrument found a carrier: the
envelope instrument when it found a train, `cv_task_extent` off the CV units when it did not. The
authority record is scoped to neither. It is scoped to the hull of the CV units — literally "the
extent the instrument covers" — through the shared `cv_covered_extent`, which `cv_task_extent` now
also uses.

Two consequences, both wanted. Where the envelope's train is wider than the CV reading, the excess
is not claimed: the CV instrument said nothing there, so it contradicts nothing there. And the
record appears whether or not the CV instrument's own `task_extent` survived the drop in
`align_ddk`, because the reading is a reading regardless of which span was kept.

`cv_covered_extent` reuses `cv_task_extent`'s own gate unchanged: units exist, **and** either a
complete cycle or a train was found. Units with neither is no evidence the task was performed, and
an instrument that read nothing is the authority on nothing.

### Why any overlap, and no threshold

A word is contradicted when its **hull** — the union of every recogniser's placement of it and the
fitted extent, which is `word_hull`'s "must not miss the word" read — shares any interval with the
covered extent.

No minimum-overlap fraction is introduced. A fraction would be a new operating point with no
measurement behind it, which is the failure CLAUDE.md names ("adding a flag back is adding an
unmeasured decision with a public interface"). There is nothing to sweep it against: the
contradiction changes no decision (below), so a threshold on it would be a number tuning the
contents of an annotation. Any-overlap on the hull is the wider of the two available reads, and
because the record is purely additive, wider is the conservative direction.

## Every consumer, and which of them changes

Traced across the whole of `src/senselab/audio/workflows/triage/`.

| Consumer | Reads | On a declared DDK recording |
| --- | --- | --- |
| `speech.py` step 7 PII scan | `consensus_transcript.text`, each `asr_hypothesis.transcript` | **unchanged** — the safety pin |
| `redact.py` `_render` / `_verify` / `_write_artifacts` | live `word` entities, `pii` entities | **unchanged** — no word invalidated, no `pii` written |
| `speech.lexical` routing gate | `features.words["lexical"]` | **unchanged, and deliberately** — see below |
| `speech.transcript_agreement` flag | `features.words["agreement"]` | unchanged; annotates, never routes |
| `voice.py` `lexical_content` | `lexical(evidence.words)` under `forbid_lexical` | **unreachable** — `forbid_lexical` is set on `maximum-phonation-time{,-v2}` only, and those are reached only from VOICE's in-family mode, which a DDK declaration does not select |
| `airway.py` `off_task_extent` (`reading="lexical_intrusion"`) | `lexical_words` | **unreachable** — in-family AIRWAY only |
| `airway.py` `overlaps_transcript` | `lexical_words` | unchanged; a covariate, not a filter |
| `speech.py` `detect_speech` (`count("lexical_words", …)`, `lexical_run_*` spans) | `lexical_words` | **unreachable** — a DDK declaration selects `align`, not `detect` |
| `speech.py` `_speech_ordered` / `_speech_item_list` / `_stimulus` / `_phrase_runs` | `lexical_words` | **unreachable** — `align_speech` routes `SYLLABLE_REPETITION` to `align_ddk` before any of them |
| `speech.py` step 2 `speech_run_*` spans, `detail["words_n"]`, `detail["speech_s"]` | `lexical` | unchanged — the words are still there and still spanned |
| `stimulus.py` `align_stimulus` / `n_lexical_words` / `unexpected` | `LexicalWord` from `lexical_words` | unchanged; no DDK row declares prompts |
| `extend.py` `rebracket_words` (`n_lexical_before/after`) | stored `word` attributes | unchanged; a post-run maintenance pass |
| `report.py` `_consensus_transcript` / `_marked_transcript` / `_redacted_transcript` / `_token_record` | `consensus_words`, `label`/`pii` assertions | unchanged |
| `figure.py` `_words` / `_asr_lane_panel` / `_consensus_word_stats` | `consensus_words` | unchanged |
| `routing_analysis/features.py` `words` counts | live `word` records | unchanged |
| SPEECH branch report `detail` | `syllable_detail(result)` | **gains** `ppg_contradicted_words_n` |
| VERDICT | branch reports' `deviations`, route measurements | unchanged — a contest is not a deviation |

**No consumer changes behaviour. Every change is information.** That is the finding, not a
shortfall against the brief, and the reason is in the next two sections.

## Why `speech.lexical` is not discounted

The owner's brief named this as the half that would change behaviour. It cannot be changed, for a
DAG reason, and it must not be, for a safety reason.

**It cannot.** `GRAPH_ORDER` is `ADMIT, PREPROCESS, TAXONOMY, routing, AIRWAY, SPEECH, …`. The gate
is evaluated in `routing`, two nodes before SPEECH. The CV walk — `ppg_reading`, `cv_units`,
`syllable_trains` — runs inside `align_ddk`, inside SPEECH. At the moment the gate reads
`features.words["lexical"]` the instrument's reading does not exist in the store, and `routing`
reduces the live store through the same `extract_features` the offline path uses, so there is no
second definition to amend either. Moving the CV walk earlier would move a body that needs the
declared syllable template — a SPEECH concern — into PREPROCESS, which is a different change from
this one.

**It must not.** For a *declared* DDK recording the gate is already irrelevant: the ruleset's
`reference_family_set` maps `SPEECH: speech`, `speech = lexical_speech | syllable_repetition`, and
a recording whose declared family is a positive there routes to that branch whatever its gates
read. The packaged config says so in as many words, and says why:

> SPEECH's set is wide on purpose: it carries the diadochokinesis families, whose syllable trains
> carry no lexical word and would never clear `speech.lexical`, and SPEECH is where the PII scan
> runs.

So discounting there buys nothing on the declared case. On the *undeclared* case it does the one
thing the constraint forbids. `speech.lexical` is the only entry in `branch_gates.SPEECH`. A
recording with no declaration, whose recogniser text is the only thing routing SPEECH, would lose
that route the moment an incidentally-found train discounted its words — and losing the SPEECH
route means the PII scan never runs, on a recording whose text nobody has looked at. That is a
disclosure path created by a cleanliness change, which is precisely the trade the constraint
rules out.

## Declared-only, not any-train

Declared-only, and the ruling predates the question. From
[`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md), quoting the owner on 2026-09-16:

> no participant would have done a DDK under a different instruction. it's a very specific task. so
> all tasks labeled ddk will go to the ddk branch, and tasks that don't have ddk will not go to
> that branch.

**The declared task label is ground truth for "is this DDK", in both directions.** The instrument
is not a DDK detector and was deliberately demoted from being one; a train found on connected
speech is a rhythm the CV walk happened to segment, not an instruction the participant followed,
and it carries no claim that the words around it were not words. Authority comes from the
declaration, not from the segmentation.

The implementation inherits this without a test of its own: `instrument_authority` is called from
`ppg_evidence`, which is called only from `align_ddk`, which `align_speech` reaches only for
`task_family in SYLLABLE_REPETITION`. An undeclared recording takes `detect_speech` and the CV
walk never runs.

The safety argument above is the second, independent reason, and it bites only on the undeclared
case: any-train would be the arm that could de-route SPEECH.

## Verification

Twenty-one regression tests, each proven to fail against code that does not carry the behaviour it
pins. The full revert — the implementation restored to its parent commit with the tests kept — is
an `ImportError` on `CONTRADICTED` / `CV_AUTHORITY` at collection, which is what an unfixed tree
gives and says nothing per test, so the record below is by targeted mutation. Each row is one
mutation applied to product code alone; the clean tree runs all 116 tests in these three files
green.

| Mutation | What it changes | Tests it kills |
| --- | --- | --- |
| M1 un-wired | drop the `instrument_authority` call from `ppg_evidence` | the reading, its extent, the contests, the outside-word control, the deviation-kind check, the report key, the gate-count pin, the interlock's control — 9, including the pre-existing `BRANCH_MEASURES` contract test |
| M2 bracketed | `contradicted_words` reads `consensus_words`, not `lexical_words` | the contest list, the bracketed-token control, the outside-word control, the report key, the gate-count pin — 5 |
| M3 no gate | `cv_covered_extent` drops the cycle-or-train half of its gate | the no-cycle-no-train claim, plus the pre-existing `cv_task_extent` gate test — 2 |
| M4 text leak | the contest carries `text=` instead of `index=` | the contest payload, the no-text pin — 2 |
| M5 narrow scan | SPEECH withholds contested words from `scan_for_pii` | **all five safety pins**: the scanned list, the sentinel, the declared-vs-undeclared equality, the PII marking, the released transcript |
| M6 invalidate | `instrument_authority` retires the words it contradicts | the survive-verbatim pin, the deviation-kind check, the gate-count pin, the released transcript |
| M7 any train | `align_speech` sends every declared family to `align_ddk` | the declared-lexical control |
| M8 undeclared is DDK | `declared_task_family` falls back to a DDK family | the undeclared control, the gate-count pin |
| M9 deviation | the contest is written as a `lexical_content` deviation | the contest list, the deviation-kind check, the gate-count pin, the interlock's control |
| M10 discount gate | `features.py` subtracts contested words from `words["lexical"]` | both routing-gate pins in `live_evidence_test.py` |

M5 and M10 are the two forbidden designs written out and shown to fail. M6 is the third — deleting
rather than marking — and it fails on the released transcript, which is the outcome that matters.

No operating point was added, so `data/config/default.yaml` and
[`config-derivations.md`](config-derivations.md) are unchanged.

## What was deliberately not done

- **No lexical consumer discounts anything.** Every candidate is either unreachable on a declared
  DDK recording or is the routing gate, and the routing gate is argued above.
- **No text in the new records.** Word ids and indices only.
- **No operating point.** Any-overlap on the hull; nothing to sweep.
- **`report.py` and `figure.py` untouched.** The reading is in the store and in the branch report's
  `detail`; rendering it is a separate change with a separate owner.
- **No new deviation type.** A contest is not a deviation and does not reach VERDICT's fold. The
  instrument contradicting the recogniser is not a departure by the participant.
