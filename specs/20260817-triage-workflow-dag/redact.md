# REDACT

The last step of the [SPEECH branch](branch-speech.md), not a node beside it. Produces a **releasable
derivative** of the recording and its transcript.

## When it runs

**Only when SPEECH's PII scan over the consensus transcript found something.** Three states, and
only the first reaches this step:

| state | REDACT | release axis |
| --- | --- | --- |
| SPEECH ran and found PII | **runs** | `releasable` on a pass, `withheld` otherwise |
| SPEECH ran and found no PII | does not run | `not_assessed` |
| SPEECH did not run, or failed for want of words | does not run | `not_assessed` |

A wordless recording has no PII scan, no REDACT verdict, and no withheld release. There is no
incomplete-scan row here, because there is no scan to be incomplete: a file with nothing to redact is
not a file whose redaction failed.

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
| `alignment` | PREPROCESS | the extents to silence |
| `recording`, `plain` streams | PREPROCESS | the audio the fill is written into |

## What it redacts

**Every finding the scan produced, regardless of speaker.** SPEECH flags only target-speaker PII
because flagging is about which recordings need attention; redaction is about whether an artifact is
safe to release, and a non-target speaker naming the participant is exactly as unsafe.

| | SPEECH step 7 | REDACT |
| --- | --- | --- |
| scope | target speaker's spans | every finding |
| purpose | does this recording need a human | is this artifact releasable |

## Redaction is conservative at the edges

Word edges come from `alignment` and carry error. A boundary off by 100 ms either leaves a fragment of
a name audible or clips the neighbouring word, and only one of those two failures is recoverable. So
every redacted extent is **padded outward** by `policy.padding_ms`, chosen to exceed the worst
measured edge error rather than the median. Two words whose padded extents overlap are merged into
one redaction.

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
derived from alignment, and whether intelligible speech survives outside that extent is not something
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
artifacts: { audio?, transcript? }   # each redacted, each independently optional
verdict:   { redactions_n, by_category{}, padding_ms, fill, verified: bool, survived[],
             unremediable[], replanned_n, scan_failed[], scan_missing[], required_detectors[],
             unplaced_words_n, audio_check, artifacts_withheld: bool }
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

## Out of scope

Deciding *whether* to release, running any recognizer, scanning anything but the consensus
transcript, speaker anonymisation, and voice conversion: a redacted recording still carries a voice,
and this step does not claim otherwise.

Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `redaction.fill` | which of `silence`, `noise`, `bleep` is least damaging to downstream measurement; **deferred**, no default |
| `redaction.padding_ms` | a positive floor exceeding the worst measured alignment edge error; **null** |
