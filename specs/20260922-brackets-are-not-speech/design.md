# Brackets are not speech

A transcription convention cannot disclose anything, and must not be handed to the PII scan.

## The defect, traced end to end

One recording, `sub-c403a6c7…_task-prolonged-vowel`. Its live consensus stream is five words: three
lexical (a spoken count-in) and two bracketed filler tokens. Its live `pii` findings are two
`PERSON` extents, `[3.12, 4.96]` and `[5.44, 11.76]`, each overlapping one of the two bracketed
words and nothing else. The second is 6.32 s long and contains the recording's whole
`task_extent` (`production: sustained`) at `[5.49, 11.70]`.

The chain, one link at a time:

1. The participant counts in, so there **are** lexical words outside the declared stimulus.
   `words_outside_stimulus` is non-empty and the conditional scan added in
   `specs/20260919-pii-against-the-stimulus/` correctly fires. That gate is about **whether** to
   scan.
2. The scan is then handed the **whole** transcript. `haystacks[0]` was
   `consensus.attributes["text"]`, which `render_transcript` builds by joining **every** word's
   text, bracketed ones included; each source haystack was that recogniser's own
   `asr_hypothesis.transcript`, and CrisperWhisper emits bracketed filler tokens natively.
3. A detector returns `PERSON` on the bracketed token.
4. `_locate` places it. `_norm_token` strips `[]`, so the bracket token matches its own word
   position exactly, and `_timings_hull` widens the finding to the union of every source's timing
   of that word.
5. Nothing verifies it: `redaction.llm_check.enabled` is `false` by default.
6. REDACT pads, plans, masks, re-scans the redacted text, finds it clean, and reports *"every
   finding redacted; the redacted transcript re-scans clean"*. The file is marked `releasable`
   with 6.3 s of the task destroyed.

Every layer did what it was told. The gate reasons about whether to scan, not about what to scan.

## Corpus measurement

Over the replayed corpus at `triage_replay_20260922/out/`, read through `ProvStore.read_jsonl` and
the graph's own liveness rule (`wasInvalidatedBy`), restricted to the recordings where REDACT ran.
Numbers are in `measurements.md`.

## The discriminator

Two candidates, and the code already treats one as authoritative.

`ConsensusWord.bracketed` is set once, in `consensus.py`, as `is_bracketed(text)` over the
consensus **display**, which `bracketed_form` produced from the raw surface and the
`words.onomatopoeic_tokens` vocabulary. Every reader in the tree selects on that attribute
(`lexical_index` in `speech.py`, `lexical_words` in `branches.py`, the word-extent reads in
`preprocess.py`), so the flag is the authority for a **position**.

It is not sufficient on its own for a per-source haystack. `readings[source]` holds `m.raw` — the
recogniser's own surface, before bracketing — so the two disagree in both directions:

- raw `uh`, display `[UH]`: the position is flagged; the raw surface is not bracketed.
- raw `[UH]` outvoted by a plain twin in the same column (`rebracket`): the position is **not**
  flagged; that one source's surface still is.

So the rule is both, applied at the level each governs: drop the **position** when the consensus
word is `bracketed`, and drop an individual **token** when the surface a source contributed is
itself `is_bracketed`. Both use `is_bracketed`, the same predicate the flag was set with; no new
convention is introduced.

## The change

`_scan_tokens(words, positions, haystack)` returns the positions kept and their tokens, filtered by
the rule above. Each haystack's scanned text is then `" ".join(tokens)` and its position list is
the kept positions, so the text handed to the detector and the tokens `_locate` searches are the
same list — they were not before, which is a second, latent mismatch this closes.

For the consensus haystack with no bracketed word, `" ".join(word texts)` is byte-for-byte
`render_transcript(words, strong=("", ""))`, so nothing changes on a recording that has none. For a
source haystack it is that recogniser's raw surfaces in stream order, single-spaced;
capitalisation and punctuation are preserved, which matters because the NER detectors read them.

The scan gate is untouched: `words_outside_stimulus` already read `lexical` only, and
`invites_disclosure` reads the declared family. A recording carrying a real disclosure is still
scanned and still redacted.

## What was rejected

**String surgery on the stored transcripts** — deleting `\[[^]]*\]` from
`consensus.attributes["text"]` and each `asr_hypothesis.transcript`. It fails on the raw-`uh`
case, where the vocabulary and not the punctuation is what says the token is non-lexical, and it
leaves the text and the located tokens derived by two different rules.

**Filtering the findings instead of the input** — letting the scan see the brackets and discarding
a finding whose covered words are all bracketed. It leaves a detector call on content that cannot
disclose, it spends LLM review budget on it when `llm_check` is on, and it cannot express the
per-source case, where the same position is legitimate in one haystack and a bracket in another.
