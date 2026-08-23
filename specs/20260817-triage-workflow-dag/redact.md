# REDACT

Runs after SPEECH. Produces a **releasable derivative** of the recording and its transcript.

## Signature

```
redact(store, policy) -> fail(reason) | pass(artifacts, verdict)
```

Reads the [element store](store.md). Writes new artifacts and new elements. **It changes nothing that
is already there** — the store is append-only, so redaction cannot unmake the PII it contains.

## What it redacts, and why that differs from what SPEECH flags

**Everything the PII scan found, regardless of speaker.** SPEECH flags only target-speaker PII because
flagging is about which recordings need attention. Redaction is about whether an artifact is safe to
release, and a non-target speaker naming the participant is exactly as unsafe. So the two rules differ
deliberately, and the difference is the point rather than an inconsistency.

| | SPEECH step 7 | REDACT |
| --- | --- | --- |
| scope | target speaker's spans | every finding |
| purpose | does this recording need a human | is this artifact releasable |

## Redaction is conservative at the edges

Word edges come from `alignment` and carry error. A boundary off by 100 ms either leaves a fragment of a
name audible or clips the neighbouring word, and **only one of those two failures is recoverable**. So
every redacted extent is **padded outward** by a margin from `policy`, and the margin is chosen to
exceed the worst measured edge error rather than the median.

Two words whose padded extents overlap are merged into one redaction, because redacting them separately
leaves an audible sliver between them.

## Verification is part of the node, not a downstream concern

A redaction that is asserted rather than checked is not a redaction. So the node **re-runs ASR and the
PII scan on its own output**, and:

- a finding that survives is a `fail` — the artifact is not released, and the surviving category is
  reported;
- a finding that disappears from the text is evidence about the text only.

**The audio check is the weaker one, and the spec says so.** ASR on redacted audio may simply fail to
transcribe a region that still contains intelligible speech, so a clean re-scan is consistent with an
incomplete redaction. Verification bounds the failure; it does not prove the negative.

**Refusing a release needs no recognizer.** The node used to resolve its re-runnable recognizers
before reading the scan evidence, and that resolution raises when nothing can re-verify — so a
recording with no speech, whose SPEECH run wrote a scan saying nobody scanned, raised instead of
withholding. That made the empty scan inert on exactly the recordings it exists for. Scan evidence is
now read first: an incomplete scan (a failed detector, a required detector that was never attempted,
or none that ran) concludes `fail` and withholds without naming any recognizer, and the verdict's
`expected_source` records `not_required`.

**A required detector that was never ATTEMPTED must not read as complete.** `failed` only records a
detector that ran and raised, so a detector nothing asked for left no trace: locally the scan ran
`[presidio, rules]` with `failed={}` and REDACT released, while on the cluster the same recording
attempted gliner, recorded its failure and withheld. Completeness is `required ⊆ scanned_by` **and**
`failed` empty, where `required` is the config key `pii.required_detectors` (derivation in
`data/config/default.yaml`); a detector in `required` but neither scanned nor failed lands in the
verdict's `scan_missing`. The node's own re-scan over its output is judged by the same rule, since a
re-scan that skipped a required detector reports no findings for the same reason a complete one
would. The zero-recognizer raise stays for a store whose scan *claims* completeness while
carrying no recognizer agent — that store is incoherent, and is a different thing from a complete
store with nothing to scan.

**The recognizer set verification is measured against comes from PREPROCESS's declaration, not from
the words in the store.** Deriving it from words failed open: a recognizer whose ASR raised inside
PREPROCESS — which is a `pass` there, with the step named in the verdict's `absent` map — wrote no
word, so it left the expected set entirely, `unverifiable` came out empty, and a check that ran on
one of two recognizers reported itself undegraded and released the pair. The expected set is instead
every PREPROCESS activity naming a model in its parameters and running under that model's agent,
which exists whether or not the recognizer went on to transcribe. A store carrying no such
declaration falls back to the word-derived set, and the verdict's `expected_source` says which of
the two was read so a `pass` on the weaker basis is legible rather than silent.

## Two exfiltration paths that are not the artifact

**An error message is a disclosure path.** `plan_redactions` refuses an invalid extent by raising with
that extent's bounds and category in the message, so the guarantee that a category never carries matched
text is what keeps the exception out of the logs. This module trusts the construction boundary for that,
which makes the membership check at the node building extents from findings more than a nicety: it is
what secures the error message as well as the artifact.

**`+` is a reserved character in a category label.** Merged extents join their categories with it and
re-planning splits on it, so a label containing `+` would be silently decomposed. No label in
[`taxonomy.md`](taxonomy.md) contains one, and none may.

## The store cannot be made releasable

The store holds the unredacted transcript with provenance, by design. Redaction produces a **derivative**
alongside it and cannot retroactively clean it. Therefore:

- the store is a **sensitive artifact** and is not released;
- the redacted artifacts carry no back-reference that would let a reader recover a redacted span;
- **element ids are not shared** between the store and a released artifact, because an id that indexes
  both is a join key back to the PII.

That last point is the one easiest to get wrong: the figure and the view both carry element ids for
traceability, which is correct inside the store and disqualifying in a released artifact.

## The source is not destroyed

This node writes; it does not delete. Removing the original recording or the store is an **operator
decision** with its own authorisation, and nothing here performs it as a side effect of producing a
redacted copy.

## Product

```
artifacts: { audio?, transcript?, figure? }   # each redacted, each independently optional
verdict:   { redactions_n, by_category{}, padding_ms, verified: bool, survived[], verify_systems[], scan_failed[], scan_missing[], required_detectors[], unplaced_words_n, audio_check, artifacts_withheld: bool }
```

**Only a pass produces a released pair; a flag withholds exactly like a fail.** The node used to
write the pair on `flag` too, so the file-level fold — which maps `flag` to withheld — reported a
recording as unreleased while `released/audio.wav` sat on disk and `run.json` named it. On anything
but a pass `artifacts` is empty, nothing is written under the release directory, and
`artifacts_withheld` is `true` with the `why` already carrying the reason.

`survived` is non-empty only on `fail`, and names **categories**, never matched text.

`verify_systems` names the recognizers verification actually re-ran; fewer than both of
PREPROCESS's is a `flag`, never a silent degrade. `scan_failed` names detectors (never their
messages); `scan_missing` names the required detectors nothing attempted, and `required_detectors`
names the set both were judged against, so a run made under a narrowed set says so. `unplaced_words_n` counts words released as `[UNPLACED]` because their extent is
unknown — text of unknown location is never released verbatim. `audio_check` is the constant
`"bounded"` on every path: verification bounds the audio failure, it never proves the negative —
on the scan-incomplete path where no ASR re-ran, the discriminating facts are `verify_systems: []`
and `verified: false`, not this key. `padding_ms` must be a
non-negative whole number of milliseconds; zero is accepted as valid but contradicts the
margin's purpose, and a positive floor is unmeasured — benchmarks/open.md.

## Out of scope

Deciding *whether* to release — this node makes an artifact releasable and does not release it. Speaker
anonymisation, voice conversion, and any transformation aimed at the speaker's identity rather than at
the PII they uttered: a redacted recording still carries a voice, and this node does not claim otherwise.

Derivations live in [`benchmarks/`](benchmarks/).
