# A free response is the words' hull

`_speech_free_response` reported `conformance: False` — "what the instruction asked for did not
happen" — on recordings carrying 20 to 109 spoken words. A paired probe put it at 16 of 18
recordings with no usable `response`; of a paired 12, eight had no ASR span at all, three had a
surviving fragment and one a workable hull.

## The composition path

```python
runs = asr_spans(live_entities(store, "span"))          # spans whose measure is "asr"
response = hull([span.extent for span in runs ...])
done = response is not None and duration(response) >= response_min_s   # 0.5
```

PREPROCESS writes those spans through `_novel`:

```python
asr = _novel(asr_candidates, primary + supplement + continuity, measure="asr", signal="consensus")
```

and `_novel` keeps **only a candidate with zero overlap** against the amplitude and continuity
spans. A candidate that overlaps one is attached to it as a `corroborated_by` record instead.

So on any recording where the amplitude envelope already segments the speech — which is every
recording where someone talks — every ASR candidate overlaps an amplitude span and none survives
as an `asr`-measure span. **By construction, an `asr` span exists only where ASR disagreed with the
envelope.** Reading the response off that set measured the deduplication, not the participant.

The two failure shapes follow directly:

- **no candidate escaped** → `response is None` → `False` on a recording with 109 words.
- **one candidate fell in a gap** → `response` is that fragment alone. Measured fragments: `0.00 s`,
  `0.05 s` and `0.34 s` on recordings of 21, 141 and 103 words, all `False`; and
  `free-speech-v2-3`, one span, `0.53 s` hull, 63 words → **`True`**. A 0.53 s "response" passing on
  a 63-word recording is the same artefact wearing the other sign: both values are decided by
  whether the surviving fragment happened to clear `response_min_s`.

This is not the same defect as
[`../20260919-diarization-turns-past-the-decode/design.md`](../20260919-diarization-turns-past-the-decode/design.md).
That one is a model reading reaching past the file; this one is a reader asking the store for a set
the writer's deduplication rule guarantees is empty. They share only the branch they land in.

## What `response` is now, and why

```python
words = lexical_words(store)
response = hull([word_extent(word) for word in words])
```

`asr_candidates` in PREPROCESS is `group_extents_into_runs` over exactly the non-bracketed consensus
word extents. Grouping into runs never changes the union's hull, so **the hull over the words is
the hull the ASR candidates would have had** — the quantity this body always meant, and the one
quantity `_novel` cannot take away, because it decides which candidates become entities and not
what they span. It is also already the pattern next door: `_speech_item_list` takes
`hull([word_extent(word) for word in items])` and never touches the span table.

The two alternatives were rejected:

- **Amplitude spans carrying ASR corroboration.** Recovers the same region indirectly, by reading a
  `corroborated_by` attribute list whose shape is PREPROCESS's private storage decision. It would
  put a second reader on the very coupling that caused this, and it answers a worse question:
  where the envelope found energy that ASR agreed with, not where lexical content is.
- **Remapping `response is None` to UNDETERMINED.** Fixes the thirteen that read `False` on nothing
  and leaves every fragment case reporting an arbitrary `True` or `False`. If `asr` spans cannot
  survive `_novel` wherever someone talks, the term is not a response measurement on this corpus at
  all, and a `None` guard would preserve the artefact under a tidier name.

`_novel` is unchanged. Collapsing agreeing sources into corroboration is the right storage design
and altering it would move every span population in the graph.

`task_extent` is minted from the same `response` and therefore relocates with it: it is now the
words' hull, derived from the consensus measurement and every lexical word, and it stays a boundary
rather than a score — the same shape the DDK work settled on.

`asr_spans` is deleted. `spans_by_measure(spans, "asr")` still answers the question for a caller who
genuinely wants the candidates that disagreed with the envelope; a named accessor implying "the ASR
spans" is a trap, and this body was its only caller.

## Two absences, told apart

An absent instrument must read `UNDETERMINED`, never `False`. Two were conflated:

1. **No recognizer reached the consensus.** `_transcribed(store)` is false when PREPROCESS wrote no
   consensus, or wrote one naming no source. Previously `response` was then `None` and the branch
   reported `False` — an extractor absence delivered as a participant failure. It now writes an
   `unviable("response", ...)` reading and reports `UNDETERMINED`.
2. **The recognizers ran and found no lexical word.** That is a reading with a value, and it still
   reports `False`.

## The stimulus no longer erases a reading it does not underwrite

An anti-pattern row reads PREPROCESS's `stimulus_alignment`. When it was absent the body set
`done = UNDETERMINED` for every anti-pattern, which masked 3,074 `free-speech` recordings whose
response had in fact been measured.

The two rows are not alike:

- `verbatim_source` (`story-recall`, `story-recall-v2`) **reassigns** `done` from
  `source_content_coverage`. No source, no term: `UNDETERMINED` stays.
- `verbatim_prompt` (`free-speech`) uses the stimulus for one deviation, `stimulus_mismatch`, and
  never for `done`. Its absence is recorded as `unviable` and the response reading stands.

## Corpus sizing

11,701 recordings sit on `FREE_RESPONSE` rows. 7,078 read the broken term today in the shipped
configuration with no hints and nothing masking them (`free-speech-v2` 2,120,
`productive-vocabulary` 2,910, the three `picture-description` rows 1,591, `open-response-questions`
199, `cinderella-story` 258). 3,074 (`free-speech`) were masked as `UNDETERMINED` by the absent
stimulus. 1,549 (`story-recall`) are insulated because `verbatim_source` reassigns `done`.

Measured before/after rates are in [`measurement.md`](measurement.md).
