# The reviewer resets REDACT's masks

2026-09-26. Owner: *"released with redaction -> only if the reviewer considers resetting the redaction
to the original prose is not fine, otherwise release without redaction or partial redaction."*

## The rule

For a recording the fold would release **with redaction** — a REDACT `pass`, or a REDACT re-scan
`fail` the reviewer cleared (`specs/20260926-redact-rescan-survival/design.md`) — the reviewer's
`release` entries name masks to reset to the original words.

| masks the reading resets | release | ground |
|---|---|---|
| none | `release_with_redaction`, REDACT's copy | as before |
| some | `release_with_redaction`, the source re-masked with the kept masks | `REVIEWER_RESET_SOME_MASKS` |
| every one | `release_without_redaction`, the original | `REVIEWER_RESET_EVERY_MASK` |

A reading that proposes any `redact` resets nothing (and withholds, under `llm_redaction_withholds`).
A reading of the original as `carries_pii` that would reset every mask resets none: releasing the
whole original contradicts that judgment. No reading, or a non-reading (`absent`, `disabled`,
`nothing_to_read`), resets nothing. A reset never moves `withheld` or `not_assessed`.

## Why the axis keeps four values

The release axis answers *which artefact may be handed on*: the original, the redacted copy, neither,
or unknown. A partial redaction is still the redacted copy — the one file set under `released/` — with
a smaller mask set, and the ground says which. A fifth value would split one artefact into two states
a consumer must join back. The parquet's `release_ground` carries the partial case as a category, so
an evaluation counts it without a schema change.

## Mapping a quote to a mask

`reviewer_reset` in `nodes/redact.py`. The reviewer's `release` entries carry `text` (a quote), a
category and a reason — no span, no word id. What was measured on the r5 readings (all 8,514 recordings
REDACT ran on, 10,092 `release` quotes):

- Quotes are the **original** words the reviewer read ("the past couple of weeks", "Casino Royale",
  "incubated for a week"), sometimes wider than a mask ("Tardy? What's tardy? I don't know what's
  tardy." over two single-word masks), sometimes a placeholder quoted back ("[PERSON+NAME]").
- A quote is matched as a run of whole tokens — lower-cased, curly apostrophes straightened,
  punctuation stripped from both ends — against the residue words the reviewer read, **at every
  place it occurs**. The judgment is about the words: "Cinderella" is not identifying wherever it
  is said. Substring matching was rejected because it would let a fragment ("brook") reset a mask
  over "brooklyn".
- A mask hides the residue words its padded extent overlaps (the renderer's own rule). With
  `redaction.padding_ms: 250`, a mask routinely folds in a neighbouring word the quote does not name.
  Requiring every hidden word to be named kept **5,259** masks for that reason alone. So:
  - where the reviewer read the original as `clean`, a mask is reset once a matched quote names **any**
    residue word it hides — the neighbouring words are part of an original the reviewer already
    judged; this leaves **33** partly covered masks kept;
  - otherwise every residue word the mask hides must be named.
- A quote matching no run resets nothing (**42** of 10,092: placeholders quoted back, paraphrases, a
  truncated quote). A mask hiding no residue word is kept.

## What it moves, estimated on the r5 readings

The r6 re-review produces new readings, so these are estimates. Of the 8,514 recordings REDACT ran on,
7,406 would be released with redaction before the reset (2,585 of them cleared re-scan fails); the
other 1,108 are withheld on a `redact` proposal or an uncleared fail.

| of 7,406 masked releases | count |
|---|---|
| every mask reset → `release_without_redaction` | 6,115 |
| none reset → `release_with_redaction` | 1,130 |
| some reset → partial redaction | 145 |
| every mask reset, original read as `carries_pii` → kept | 16 |

11,842 masks planned over the 7,406; the reviewer's quotes reset 4,899 under the all-named rule and
most of the rest under the clean-original rule.

## The release directory

`settle_release` makes `released/` hold exactly what the fold released, from the store: emptied for
every release but `release_with_redaction`, including of a copy REDACT itself wrote on a pass; REDACT's
own copy kept where it carries every planned mask; otherwise written from the `redacted` stream, or
for a partial redaction from the source stream the `redacted` stream was masked from, re-masked with
the kept masks under REDACT's recorded fill. A later fold that changes which masks stand replaces the
copy; the consensus artifact's `n_redactions` is how a full copy is told from a thinned one.
