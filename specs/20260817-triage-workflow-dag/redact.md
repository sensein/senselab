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
verdict:   { redactions_n, by_category{}, padding_ms, verified: bool, survived[] }
```

`survived` is non-empty only on `fail`, and names **categories**, never matched text.

## Out of scope

Deciding *whether* to release — this node makes an artifact releasable and does not release it. Speaker
anonymisation, voice conversion, and any transformation aimed at the speaker's identity rather than at
the PII they uttered: a redacted recording still carries a voice, and this node does not claim otherwise.

Derivations live in [`benchmarks/`](benchmarks/).
