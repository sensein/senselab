# `stimulus_alignment` — D1

The PREPROCESS derivative that relates the consensus word stream to what the recording was
*declared* to expect. Ranked first in
[`preprocess-derivatives-for-expected-patterns.md`](preprocess-derivatives-for-expected-patterns.md);
every consumer is marked `†stimulus_alignment` in
[`expected-patterns.md`](expected-patterns.md).

Code: `src/senselab/audio/workflows/triage/stimulus.py` (the alignment, store-free),
`nodes/preprocess.py` (`stimulus_input`, `write_stimulus_alignment`, `stimulus_alignment`, and the
`stimulus_alignment` block). Config: one key, `stimulus.sentence_terminators`, derived in
[`config-derivations.md`](config-derivations.md#stimulus).

---

## 1. Why PREPROCESS and not SPEECH

Three branches want different projections of one alignment and no branch reads another's output:

- **SPEECH** wants the diff — substitutions, insertions, omissions, per token, with the consensus
  `agreement` attached.
- **VOICE** wants the boundaries of the declared structure as selectable extents, so a per-sentence
  voice-quality number is possible at all.
- **AIRWAY** wants the complement — the extents of lexical material where none was expected, which
  is its `off_task` finding.

If SPEECH computes it, the other two do without it. The cost is one sequence alignment over two
token lists: no model pass, no audio.

## 2. What it is measured from

One input that already exists in every store, and one that reaches every node and was read by
nothing until this change.

| input | where it comes from |
| --- | --- |
| the lexical consensus words | `lexical_words(store)` — the non-bracketed `word` entities `consensus_transcript` names, in `index` order, with `text`, `extent` and `agreement` |
| the declared utterance | `AudioHints.expected_speech`, an ordered list of `ExpectedSpeech` |

`preprocess.py`'s node docstring said of its `hint` parameter: *"Accepted for the shared node shape;
not read."* `run_triage` has always handed the hint to PREPROCESS and to every other node
(`run.py`, `_run_graph`); nothing in `src/senselab` read `expected_speech`. That is now one reader.

**Bracketed words are set aside before the alignment.** `[BREATH]`, `[UM]` and the onomatopoeic
renderings PREPROCESS brackets are not lexical material and are not candidates for realising a
declared word; the complement this derivative emits is therefore a *lexical* complement, which is
what the AIRWAY row asks for. The bracketed words remain in the store, where the rows that want
them (`S4`'s breath-group structure, `filler`) read them directly.

## 3. Where the expectation comes from, and that it is often absent

Measured on `bids_adult_2026_09_04` (4.0-release, adult), the recording-grain sidecar
`*_recording-metadata.json` carries the fields `stimulus_text`, `instructions`, `speech_type`,
`language` and `task_name`. The recording grain is the source: the acoustictask grain is
one-to-many and disagrees on 39.9% of recordings.

`stimulus_text` is present and fully specifies the utterance for the read families:

| family | recording-grain `stimulus_text` |
| --- | --- |
| `harvard-sentences-list` | one sentence per recording, e.g. `"The pencils have all been used."` |
| `cape-v-sentences`, `-v2` | **one** sentence per recording, e.g. `"He helped her hurry home."` |
| `rainbow-passage` | the whole four-sentence passage in one string |
| `caterpillar-passage` | the whole passage, more than ten sentences |
| `word-color-stroop` | the fifteen-colour answer sequence, unpunctuated |
| `free-speech` | the question, to be read as an anti-pattern |

and it is empty by design elsewhere: `cinderella-story` (a physical storybook),
`picture-description` (an image), and — see §7 — `prolonged-vowel` and `loudness`.

**An absent expectation is a typed outcome, not a failure.** The block raises
`StimulusExpectationUnavailable`, a `ValueError`, which the PREPROCESS block runner files as a soft
absence exactly as `PpgsPosteriorgramUnavailable` and `SpeakerDiarizationUnavailable` are filed. The
node still returns a pass verdict and every other block still runs.

**A declared-but-empty expectation is different, and is a real alignment.** A hint carrying
`ExpectedSpeech(text="")` declares *no particular words*: the alignment has zero expected tokens and
every lexical word falls into the complement. That is the AIRWAY projection, and it is why the two
cases are kept apart rather than collapsed into one absence. Which of the two a corpus populator
emits for an empty `stimulus_text` is the populator's decision, and both are handled.

**Populating the hint is not part of this derivative.** `runs/b2ai-v2/make_hints.py` parses the BIDS
`task-` token and never reads `stimulus_text`. That is a precondition, named in
`preprocess-derivatives-for-expected-patterns.md` §2 D1 and deliberately not folded in here.

## 4. The alignment

`align_pair` (`audio_analysis/harmonize.py`) — the same weighted Levenshtein path
`harmonize_transcripts` uses for the ASR-against-ASR case, and through it the same path
`consensus.align_sources` already runs on every recording. It was private; it is now public and is
called from two places. Its sclite edit costs (match 0, substitution 4, indel 3) are derived in
[`transcript-alignment.md`](transcript-alignment.md); they were designed for the
reference-against-hypothesis case, which is what this derivative is and a weaker assumption than the
symmetric case they already serve. Timings are passed on neither side: the expected tokens have
none, and `align_pair`'s time tie-break requires both.

Tokens are compared on `normalise_token` — casefold, keep alphanumerics and the apostrophe — the
same key the consensus itself is aligned on. An expected token normalising to the empty key (pure
punctuation) is dropped, as the consensus drops one.

Each expected token therefore lands in exactly one of three states, and none of them is a cut over
a score:

| `realisation` | the aligner's path | what it carries |
| --- | --- | --- |
| `realised` | paired with a word of the same key | that word's index, surface, extent, agreement |
| `substituted` | paired with a word of a different key | the same, plus the read surface |
| `absent` | no word paired with it | nothing; this is the tenth deviation type SPEECH names `omission` |

A word paired with no expected token is an `UnexpectedWord`, carrying the expected-token index it
follows (`-1` before the first), so a run of them sorts into place.

## 5. The declared structure

A `StructureUnit` is one prompt, or one sentence inside one. `AudioHints.expected_speech` is already
a list and its docstring says why — a caller declaring six sentences as six entries gets six units
with no splitting. The terminator split exists so a caller declaring one multi-sentence passage as
one string gets the same units; without it `rainbow-passage` is one unit and the per-sentence
boundaries VOICE wants do not exist.

A unit's `extent` is the hull of its extent-carrying tokens. **No gap rule is applied inside it, on
purpose.** A participant who restarts mid-sentence widens the hull, and that is a true statement
about where the unit's realised tokens lie; suppressing it would need a maximum-gap cut, which
nobody has measured. The per-token extents are all in the sidecar, so a branch that wants to split
on a gap has everything it needs and owns the cut.

## 6. No threshold was fitted, and none was invented

One config key was added, `stimulus.sentence_terminators`, and it is orthographic rather than
fitted — a declaration about writing systems. The alignment needs no operating point: its three
outcomes are the aligner's own path.

Every number the consumers name — `p_omission_score_max`, `p_repeat_overlap_min`,
`p_echo_overlap_max`, `p_verbatim_overlap_max`, `p_count_in_tokens` — is a branch decision over this
output and belongs to the branch that makes it. A PREPROCESS cut here would have been an unmeasured
decision with a public interface, which is the thing the seventy retired grid flags were.

The derivative does emit the quantities those cuts will be taken over — `realised_fraction`,
`n_substituted`, `n_absent`, `n_unexpected` — as measurements with no verdict attached.

## 7. Two things in the brief for this work that were wrong when checked

1. **`prolonged-vowel`'s `stimulus_text` is empty.** The count-in *is* prescribed — `instructions`
   reads *"…repeating the sentence "1, 2, 3 aah" in your normal voice. Please hold the sound "aah"
   until the timer runs out."* — but it is prescribed in `instructions`, not in `stimulus_text`,
   which is `''` on the recordings measured. So `prolonged-vowel` is a designed-empty
   `stimulus_text` family in this corpus, alongside `picture-description` and `cinderella-story`.
   `loudness` is the same: `instructions` says *"shout "hey" as loud as possible 3 times"*,
   `stimulus_text` is `''`. A populator that wants the count-in as a declared expectation must read
   `instructions`, and that is a populator decision with its own derivation, not this derivative's.

   Note also that the prescribed count-in is written `"1, 2, 3"` — digits. Against a transcript
   reading `one two three`, all three tokens are `substituted`, not `realised`. Normalising numerals
   to their spelled forms is a decision nobody has fitted and is not taken here; the substitution is
   reported as what it is.

2. **CAPE-V is one sentence per recording, not six.** `cape-v-sentences-v2-2` carries
   `"He helped her hurry home."` alone. The six sentences are six recordings, so
   `measure_cape_v_per_sentence`'s "six boundaries" are met by one unit per recording under this
   derivative, and the multi-sentence case that actually exercises the split is
   `rainbow-passage`/`caterpillar-passage`.

## 8. Output

`derivatives/stimulus_alignment.npz` — three parallel-column tables in one file, following the
diarization precedent (what grows with the input goes to the sidecar, and the columns are
self-describing so two files concatenate):

- expected: `expected_index`, `expected_unit`, `expected_text`, `expected_key`, `realisation`,
  `expected_word_index` (`-1` when absent), `expected_read`, `expected_start`, `expected_end`,
  `expected_agreement` (NaN when absent)
- units: `unit_index`, `unit_prompt`, `unit_text`, `unit_n_expected`, `unit_n_realised`,
  `unit_n_substituted`, `unit_start`, `unit_end`
- complement: `unexpected_word_index`, `unexpected_text`, `unexpected_after`, `unexpected_start`,
  `unexpected_end`, `unexpected_agreement`

The measurement entity carries the path, its SHA-256 and size, the counts
(`n_prompts`, `n_units`, `n_expected`, `n_lexical_words`, `n_realised`, `n_substituted`, `n_absent`,
`n_unexpected`, `realised_fraction`), the algorithm/routine/normalisation, and
`signal: "plain"`. Its `extent` is the hull of every realised expected token — the read-text task
extent — or none when nothing was realised. It is `derived_from` the `consensus_transcript`
measurement.

The in-memory `StimulusAlignment` carries the projections the consumer bodies are written against:
`substitutions`, `omissions`, `unexpected`, `structure_spans()`, and `run_for(keys)` for an ordered
sub-run (`detect_count_in`, `detect_loudness_token`).
