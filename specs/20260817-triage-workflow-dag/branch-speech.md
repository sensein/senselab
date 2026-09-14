# SPEECH branch

What the branch answers: **what was said, who said it, and did the speaker do what the task asked?**

The frame is [`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md).

## The state of this branch

SPEECH is the largest node in the graph — 1,104 lines, nine steps — and most of it runs. Its problem
is the opposite of VOICE's: it **measures a great deal and decides almost nothing**, because five of
its decision points are gated behind null config.

| step | state |
| --- | --- |
| 1 transcript | live; the only source of `FAIL` |
| 2 spans | live |
| 3 corroboration | measures SQUIM and YAMNet; **both votes inert** |
| 4 diarization | live, but scoped to the lexical hull |
| 5 second diarizer | **never consulted** (`speech.second_diarizer` null) |
| 6 separation | **never selected** (`speech.separation_backend` null) |
| 7 identification | word→speaker attribution live; enrollment path **gated** |
| 8 PII | live |
| 9 quality / non-target | SQUIM and proximity measured; non-target **null** |

The branch's shape is right. What it needs is a declaration to measure against — which is what the
contract supplies.

## The tasks this branch serves

Declared families, not ground truth.

| family | n | shape |
| --- | --- | --- |
| `harvard-sentences-list` | 13,705 | read, known stimulus |
| `free-speech` | 3,074 | spontaneous |
| `productive-vocabulary` | 2,910 | elicited, semi-constrained |
| `cape-v-sentences` | 2,370 | read, known stimulus, voice-quality protocol |
| `free-speech-v2` | 2,120 | spontaneous |
| `cape-v-sentences-v2` | 1,224 | read, known stimulus |
| `rainbow-passage` | 897 | read, known stimulus |
| `picture-description` | 889 (+ 373 option1, 329 option2) | spontaneous, prompted |
| `story-recall` | 889 | spontaneous, prompted |
| `caterpillar-passage` | 597 | read, known stimulus |
| `story-recall-v2` | 660 | spontaneous, prompted |
| `word-color-stroop` | 472 | read, known stimulus, timed |

The division that matters is **read versus spontaneous**, and the declaration carries it directly as
`speech_type` (`read` / `non-lexical`, and spontaneous variants). A read task has a `stimulus_text`
and can be scored word-against-word; a spontaneous task has none and cannot. Roughly 19,265 of these
declarations are read tasks with a known stimulus — the single largest measurable population in the
corpus, and today nothing compares a transcript to it.

**CAPE-V is a voice protocol wearing a sentence task.** Its six sentences are designed to elicit
specific phonatory conditions, and the clinically meaningful measurements on it are VOICE's (V4),
not SPEECH's. SPEECH's job on CAPE-V is to confirm the right sentences were read; the voice quality
belongs to the branch that measures voice quality.

### A recording routed here whose declared task is not speech

SPEECH routed 41,565 against 33,235 declaring a speech family — and `speech.lexical` is a
high-recall gate, so any recording with agreed lexical words reaches it. On a breath task that
contains an aside, SPEECH transcribes it, diarizes it, scans it for PII and records what it found.
That content is a finding about the recording, not a fault of it.

## Capabilities

### S1 — Consensus transcript (**built**)

**Question.** What words were said, and where?

**Reads.** The `consensus_transcript` measurement and its `word_ids` (`speech.py:517-534`); raises
`LookupError` when absent. Splits lexical from bracketed.

**Emits.** `FAIL` with `kind="speech"` when no lexical word survives (`speech.py:544-565`) — the
branch's only fail path.

**Serves.** All families.

### S2 — Speech spans (**built**)

**Reads.** Lexical word extents, grouped where they touch by `group_extents_into_runs`
(`speech.py:573`, imported from `tasks/spans/api.py:34`). **Never the energy envelope** — SPEECH
proposes its own spans from the transcript, which is why it was already the contract's `propose`
precedent before the contract existed.

**Emits.** `span` entities, `family: "speech"` (`speech.py:879-883`).

### S3 — Stimulus conformance (**not built; the branch's largest gap**)

**Question.** Did the speaker read the sentence they were given?

**Reads.** `stimulus_text` from the declaration, and the lexical consensus words from S1.

**Computes.** A word-level alignment of the lexical transcript against the stimulus. **Alignment
machinery already exists** — `align_sources` (`consensus.py:276`) aligns recognizer hypotheses as
sequences and emits one word per aligned column, over `harmonize_transcripts` which
`consensus.py:15` imports. Aligning one hypothesis against a reference string is the same operation
with a different second argument.

**It is not a drop-in.** `align_sources` raises `LookupError` below two hypotheses
(`consensus.py:288-290`) and returns a `Consensus` over sources, so a stimulus comparison needs
either a second entry point beside it or a reference hypothesis wrapped as a `SourceHypothesis`.
Which of the two is right is a design decision this document does not make.

**Emits.** One `stimulus_mismatch` deviation per lexical word that is not the word the stimulus
expected, each carrying that word's extent. Substitutions, omissions and insertions are distinct
rows, not one score.

**Fillers are already separated and must stay separate.** The consensus brackets disfluency and
non-speech tokens — `[uh]`, `[um]`, `[breath]` — so they are not lexical and never enter the
alignment. They are reported as `filler` deviations in their own channel. *"Read the wrong
sentence"* and *"read it with six fillers"* are different findings.

**Serves.** `harvard-sentences-list` (13,705), `cape-v-sentences` (2,370 + 1,224),
`rainbow-passage` (897), `caterpillar-passage` (597), `word-color-stroop` (472).

**Parameter-free?** The output is — an enumeration of differences, not a score against a cut.
Nothing here is fitted, and nothing should be: whether six fillers disqualify a reading is precisely
the judgement the contract hands downstream. The aligner's own edit weights are internal to
`harmonize_transcripts` and are not a decision this branch makes.

### S4 — Speaker count (**built, but scoped wrong**)

**Question.** How many people are audible?

**Reads today.** pyannote community-1 over `[first word start, last word end]` only
(`speech.py:642-646`).

**That scope is the defect.** A second speaker outside the lexical hull — before the participant
starts, after they stop, or during a pause — is invisible. The branch-contract spec moves
diarization to PREPROCESS as a whole-file shared derivative, which makes the count correct and makes
it available to every branch rather than this one.

**Emits.** A `speaker` entity per segment, and — under the contract — a `counts` entry
`speaker_count` carrying `found` and the `declared` half from the declaration's
`targeted_speaker_count`. **It asserts no discrepancy**: the count is an observation, and whether a
second voice invalidates the recording is not SPEECH's call.

**Serves.** Every family. Most of these protocols are single-target, which is what makes the count
informative.

**The second diarizer never runs.** `speech.second_diarizer` is null (`default.yaml:166`), so
`second_record` is always `"not_consulted"` (`speech.py:685-706`). Under the contract this becomes a
question about the shared derivative rather than about SPEECH's own pass.

### S5 — Word→speaker attribution (**built**)

Word-level attribution assertions always run (`speech.py:779-869`), `verb: "attribute"`, one per
word (`:788-793`). Note for [`branch-quality.md`](branch-quality.md) and for REPORT: this is the
highest-volume assertion verb in the store and is *not* one of the contract's five.

**The enrollment path is gated.** `speech.enrollment_model` and `speech.target_match_cosine` are
both null (`default.yaml:167, 171`), so supplying an enrollment without an override lands in
`_flag_before_measuring` (`speech.py:502-506`). A cosine threshold for speaker identity is **owed
ground truth** and cannot be fitted against declared families — a declaration says which task was
performed, not who performed it.

### S6 — PII (**built**)

One `scan_for_pii` over the consensus and over each recognizer's own transcript
(`speech.py:871-957`), producing `pii` entities and per-word `verb: "label", label: "pii"` marks at
`speech.py:961`. `redact.py:237` selects on exactly that verb/label pair — which is why the
contract's earlier proposal to rename `label` to `mark` was withdrawn: it would have stopped
redaction silently.

`pii.required_detectors` is populated. This is the one decision point in the branch that is neither
null nor inert.

### S7 — Speech quality (**measured, inert**)

Per-span SQUIM (`stoi`, `pesq`, `si_sdr`) and `disruptions` (`speech.py:959-1010`), plus a per-span
`proximity` measurement (`:1012-1054`).

**Both votes are inert.** `yamnet_vote` is `"unavailable"` because `taxonomy.speech_labels` is null
(`default.yaml:181`, read at `speech.py:589`); `squim_vote` is `"not_evaluated"` because
`speech.speech_test_stoi_floor` and `speech.speech_test_si_sdr_floor` are both null
(`default.yaml:168-169`, read at `speech.py:604-605`). So step 3 measures SQUIM and decides nothing.

**Neither floor can be fitted.** A STOI floor fitted against declared families would encode which
recordings the protocol labelled, not which are intelligible.
[`branch-quality.md`](branch-quality.md) takes up what should read these numbers instead.

### S8 — Non-target speech (**gated behind null config**)

`nontarget_speech_s` is always `None` because all three `speech.nontarget` legs are null
(`default.yaml:174`). Three thresholds, all owed ground truth, none fittable here.

### S9 — Separation (**never selected**)

`speech.separation_backend` is null (`default.yaml:172`), so `separation_state` is
`"not_selected"` whenever two or more speakers are found (`speech.py:708-777`). Both the
ClearerVoice and unasdiff paths are unreached.

## Deviations

| type | evidence |
| --- | --- |
| `stimulus_mismatch` | a lexical word that is not the word the stimulus expected (S3) |
| `filler` | a bracketed disfluency or non-speech token where the task expected lexical content (S3) |
| `off_task_extent` | a region carrying no lexical speech where the task asked for reading |

`speaker_count` is a `counts` entry, not a deviation (S4).

## What exists today

| capability | status |
| --- | --- |
| S1 transcript | built |
| S2 spans | built |
| S3 stimulus conformance | **not built** — the aligner exists, nothing calls it with a stimulus |
| S4 speaker count | built, scoped to the lexical hull; moves to PREPROCESS under the contract |
| S5 attribution | built; enrollment gated |
| S6 PII | built |
| S7 quality | measured, both votes inert |
| S8 non-target | gated behind null config |
| S9 separation | gated behind null config |

## What the branch emits

```
spans        family: "speech", one per run of touching lexical words
assertions   attribute (one per word), label/pii, flag
entities     speaker (one per diarized segment), pii
measurements squim, disruptions, proximity (per span)
counts       speaker_count {found, declared}
deviations   stimulus_mismatch, filler, off_task_extent
verdict      { lexical_n, bracketed_n, speakers, second_diarizer,
               separation_state, nontarget_speech_s, flags, ... }
```

**The verdict's basis, exactly** (`speech.py:1078-1090`): `FAIL` only from the no-lexical-words row
at `:544-565`; `FLAG` when any flag accumulated; `PASS` otherwise.

## Out of scope

Any normative judgement on a deviation — whether a mismatch or a filler count disqualifies a reading
belongs to whoever reads the deviations. Voice quality on CAPE-V, which is VOICE's (V4). Any refit of
a STOI, SI-SDR, cosine or non-target threshold against declared families.
