# REDACT

The last step of the [SPEECH branch](branch-speech.md), not a node beside it. Produces a **releasable
derivative** of the recording and its transcript.

## When it runs

**Only when SPEECH's PII scan over the consensus transcript found something.** Three states, and
only the first reaches this step:

| state | REDACT | release axis |
| --- | --- | --- |
| SPEECH ran and found PII | **runs** | `releasable` on a pass, `withheld` otherwise |
| SPEECH ran and found no PII | does not run | `nothing_to_redact` |
| SPEECH did not run, or failed for want of words | does not run | `nothing_to_redact` |

A wordless recording has no PII scan, no REDACT verdict, and no withheld release. There is no
incomplete-scan row here, because there is no scan to be incomplete: a file with nothing to redact is
not a file whose redaction failed.

**Not running is not an unknown.** The last two rows are the *ordinary* case — 71% of the b2ai adult
corpus — and until 2026-09-22 [`verdict.md`](verdict.md) read them off REDACT's silence as
`not_assessed`. They are determinations, and VERDICT now makes them from the evidence: SPEECH's
lexical count, its scan record and the live findings. Which of the four determined grounds a row
lands on is in that file's release fold; nothing about *this* node's gate changed.

## Signature

```
redact(store, policy) -> fail(reason) | flag(reason) | pass(artifacts, verdict)
```

Reads the [element store](store.md). Writes new artifacts and new elements. **It changes nothing that
is already there** — the store is append-only, so redaction cannot unmake the PII it contains.

| element read | author | used for |
| --- | --- | --- |
| `pii` findings | SPEECH step 7 | what to redact |
| `consensus_transcript` and its `word` elements | PREPROCESS | the text redaction is planned and verified on |
| `recording`, `plain` streams | PREPROCESS | the audio the fill is written into |

## What it redacts

**Every finding the scan produced, regardless of speaker.** SPEECH flags only target-speaker PII
because flagging is about which recordings need attention; redaction is about whether an artifact is
safe to release, and a non-target speaker naming the participant is exactly as unsafe.

| | SPEECH step 7 | REDACT |
| --- | --- | --- |
| scope | target speaker's spans | every finding |
| purpose | does this recording need a human | is this artifact releasable |

## What the declared stimulus accounts for

A PII-shaped token the *stimulus text asked the participant to read* is not a disclosure. The
rainbow passage carries `rainbow`, which reads as a LOCATION to more than one detector; the Harvard
sentences carry proper nouns; a picture-description prompt names places. Redacting them costs the
recording the very content the task was designed to elicit, and discloses nothing, because the
prompt is public and was chosen before the participant spoke.

So REDACT reads `hint.expected_speech`. **With no hint, or a hint declaring no utterance, nothing is
exempted and the node is exactly what it was** — that is the control, and it is tested as one.

A candidate is exempt only when all of the following hold:

1. its own (unpadded) extent reaches at least one consensus word, by the hull of that word's
   per-source timings — the same hull the mask decides on;
2. it does not reach *every* consensus word. That is SPEECH's signature for a finding whose text its
   locator could not place anywhere (`pii_unlocated`), whose extent is then the whole transcript.
   A whole-passage prompt would "account for" such a finding without anything having been matched,
   so the whole-stream case is refused outright rather than thresholded;
3. every word it reaches normalises to a non-empty key under `BranchParams.p_normalise` — the
   branches' own lexical normaliser (`consensus.vocabulary_key`: casefold, strip edge punctuation),
   not a second spelling invented here. A punctuation-only word would otherwise match trivially;
4. those keys occur **in order and contiguously** inside one declared structure unit. A run, not a
   subsequence: two words the prompt happens to contain in different sentences do not account for
   them said together.

### The verifier still sees it, and that is handled explicitly

An exempt word stands verbatim in the released text, so the verification re-scan finds the candidate
again. Left alone, the re-planning pass would widen onto it and the second scan would call it
`unremediable` — the exemption would never take effect, and the node would fail every recording it
applied to. Two changes make it coherent:

- the re-plan never widens onto a word an exemption covers;
- a surviving category is attributed to the exemption when there is at least one **exempt** word
  carrying it that no planned extent covers *and* no non-exempt word carrying it in the same
  position. With no exemptions the first condition can never hold, so every survivor is a failure
  exactly as before.

The attribution is per word, not per category. A recording where `rainbow` is exempt and `alice`
carries the same category still redacts `alice`, still re-plans, and still fails if `alice` survives.

### Every suppression is recorded

A redaction not made is a decision, and an unrecorded decision is indistinguishable from a bug. Each
exemption writes a live `assertion` entity — `verb: exempt`, `label: expected_speech` — carrying the
category, the matched tokens, which prompt and which structure unit accounted for it, that unit
verbatim, and the number of words covered; `wasDerivedFrom` the `pii` finding and every word it
covers. A `redaction_exemptions` measurement carries the counts, and the verdict carries
`expected_exempt_n`, `expected_exempt_by_category`, `expected_survivors` and
`expected_speech_declared`. The matched tokens in the assertion are taken from the **prompt**, not
from the transcript, so a fault in the matcher cannot put participant text into the audit record.

## Redaction is conservative at the edges

Word edges are the consensus ASR word extents and carry their own temporal confidence and timing
source count. A boundary off by 100 ms either leaves a fragment of a name audible or clips the
neighbouring word, and only one of those two failures is recoverable. So every redacted extent is
**padded outward** by `redaction.padding_ms`, which must exceed the worst consensus-word edge error
rather than the median. It ships at 250 ms as a convention; config-derivations.md states the
arithmetic that bounds it and says plainly that it is not a fit. Two words whose padded extents overlap are merged into one redaction.

## The fill is configurable

| `redaction.fill` | what is written into the extent |
| --- | --- |
| `silence` | digital silence |
| `noise` | speech-shaped noise at the extent's own level |
| `bleep` | a tone at the extent's own level |

The key ships with no default and its derivation is **deferred**: which fill is least damaging to the
measurements taken downstream of a released artifact has not been measured. A run declares the fill it
used and the verdict records it, so two artifacts made under different fills are never compared as
one.

## Verification does not re-transcribe

**REDACT runs no recognizer.** Re-transcription would draw a second sample from the recognizers,
which is a different measurement of a different signal, not a check on this one.

**Verification is a re-scan of the redacted consensus text.** The planned redactions are applied to
the consensus transcript, the same PII detectors are re-run over the redacted text, and:

- a finding that survives is a **`fail`** — the artifact is not released and the surviving category is
  reported;
- a re-scan that skipped a required detector is a **`flag`**, judged by the same completeness rule as
  the planning scan: `required ⊆ scanned_by` and `failed` empty, with `required` the config key
  `pii.required_detectors`.

**The audio claim is explicitly bounded.** Verification establishes that the redacted *text* no longer
carries the finding. It establishes nothing about the audio: the fill was written over an extent
derived from consensus ASR timing, and whether intelligible speech survives outside that extent is not something
a text re-scan can answer. `audio_check` is the constant `"bounded"` on every path, and no consumer
may read a `pass` as a claim about the recording.

**A finding that the planner placed and the verifier still sees is remediable exactly once**: the
verifier's extent is fed back for a single re-planning pass. A finding that survives that pass is
`unremediable`, named as such in the verdict, so an operator can distinguish it from an ordinary
withhold.

## Two exfiltration paths that are not the artifact

**An error message is a disclosure path.** `plan_redactions` refuses an invalid extent by raising with
that extent's bounds and category in the message, so the guarantee that a category never carries
matched text is what keeps the exception out of the logs.

**`+` is a reserved character in a category label.** Merged extents join their categories with it and
re-planning splits on it, so a label containing `+` would be silently decomposed. None may contain
one.

## The store cannot be made releasable

The store holds the unredacted consensus transcript with provenance, by design. Redaction produces a
**derivative** alongside it and cannot retroactively clean it. Therefore:

- the store is a **sensitive artifact** and is not released;
- the redacted artifacts carry no back-reference that would let a reader recover a redacted span;
- **element ids are not shared** between the store and a released artifact, because an id that indexes
  both is a join key back to the PII. The [report](report.md) carries element ids and is not a
  released artifact.

## The source is not destroyed

This node writes; it does not delete. Removing the original recording or the store is an **operator
decision** with its own authorisation.

## Product

```
artifacts: { audio?, transcript?, consensus? }   # each redacted, released together on a pass
streams:   { redacted }                         # the masked audio, in the store, on every path
verdict:   { redactions_n, by_category{}, padding_ms, fill, verified: bool, survived[],
             outstanding[], unremediable[], replanned_n, scan_failed[], scan_missing[],
             required_detectors[], unplaced_words_n, audio_check, artifacts_withheld: bool,
             expected_exempt_n, expected_exempt_by_category{}, expected_survivors[],
             expected_speech_declared }
```

**Only a pass produces a released pair; a flag withholds exactly like a fail.** On anything but a pass
`artifacts` is empty, nothing is written under the release directory, and `artifacts_withheld` is
`true`.

`survived` is non-empty only on `fail` and names **categories**, never matched text. `unremediable`
names the categories that survived the one re-planning pass. `scan_failed` names detectors, never
their messages; `scan_missing` names required detectors nothing attempted; `required_detectors` names
the set both scans were judged against. `unplaced_words_n` counts words released as `[UNPLACED]`
because their extent is unknown — text of unknown location is never released verbatim. `padding_ms`
must be a non-negative whole number of milliseconds; zero is accepted and contradicts the margin's
purpose.

`survived` is the re-scan's raw answer; `outstanding` is what remains of it after the declared
stimulus is accounted for, and `outstanding` is what decides the outcome. `expected_survivors` names
the difference. `consensus` is the redacted consensus stream as JSON: one record per surviving word
with its extent, per-source timings and readings, variants and agreement share, and one placeholder
record per planned extent carrying its category, padded bounds and word count — and no surface,
readings or variants for anything masked. The flat transcript is the records joined, so the two
cannot disagree about what was redacted.

The `redacted` stream is written under the **run** directory and registered in the store on every
path, pass or not. The run directory is the store side of the disjointness check and already holds
the unredacted audio; the release directory is the only thing gated on a pass.

## Out of scope

Deciding *whether* to release, running any recognizer, scanning anything but the consensus
transcript, speaker anonymisation, and voice conversion: a redacted recording still carries a voice,
and this step does not claim otherwise.

Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `redaction.fill` | which of `silence`, `noise`, `bleep` is least damaging to downstream measurement is still unmeasured. `silence` now **ships** as an owner-directed declared choice, not as the answer to that question |
| `redaction.padding_ms` | a positive floor exceeding the worst measured consensus-word edge error. 250 ms now **ships** as a stated convention; the fit — the edge-error distribution at its maximum — is still owed (benchmarks/open.md) |

## Moved from `nodes/redact.py` (2026-09-20)

Two derivations that lived only in the module's docstrings when the prose was cut back to what the
code is and how to call it.

### The exemption's fifth condition: the covered words must span the whole extent

`_expected_exemptions` refuses any candidate whose identified words do not reach the whole of the
finding's own extent (within a 1e-9 s float slack). The condition is what makes *"every word its
extent reaches"* trustworthy: SPEECH builds a located finding's extent as the hull of the words it
covers, so the covered words' hulls reconstruct it exactly — unless one was missed, and a missed
word is one the exemption would be accounting for without having looked at it. This sits alongside
the four conditions listed under *What the declared stimulus accounts for*.

### `verify_failed` / `verify_missing` are separate verdict keys from `scan_failed` / `scan_missing`

`_Verification` keeps *attempted and raised* apart from *never attempted* for the reason the planning
scan keeps them apart: "it broke" and "nobody ran it" are different findings, and the second is the
silent one. Both pairs reach the verdict under their own keys — `verify_failed` and `verify_missing`
beside `scan_failed` and `scan_missing` — because a store whose planning scan was complete and whose
verification was not is a different state from the reverse, and an operator reading one pair of keys
for both could not tell which half failed. The *Product* block above lists only the planning pair;
the node writes all four.
