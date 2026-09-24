# A recall task conforms on having been responded to, not on how much was recalled

Owner, 2026-09-24:

> that particular task should only check whether a sentence was at least produced. that's not a
> decision about how much the story was recalled.

This is a scope correction, not a threshold change. What follows is what was withdrawn, what
conformance for these two families now tests, what happened to the bound, and what the corpus does
under the change.

## 1. What was there

`nodes/gates.py` carried a second conformance table beside `CONFORMANCE_GATES`:

```python
RECALL_CONFORMANCE_GATES: tuple[str, ...] = ("coverage_min",)
VERBATIM_SOURCE = "verbatim_source"

def conformance_gate_names(pattern, *, anti_pattern=None):
    if pattern is Pattern.FREE_RESPONSE and anti_pattern == VERBATIM_SOURCE:
        return RECALL_CONFORMANCE_GATES
    return CONFORMANCE_GATES[pattern]
```

`coverage_min` **replaced** `response_min_s` — it did not join it. So for the two rows that declare
`anti_pattern="verbatim_source"`, conformance was `source_content_coverage >= 0.5` and nothing
else; how long the participant spoke and how many words they said entered no term.

Those rows are `story-recall` and `story-recall-v2`, and only those. `family_gates.py` enumerates
the live expectation tables and reports `48 families, 2 change: ['story-recall',
'story-recall-v2']`; that is the whole reach of the withdrawn rule, established from the code
rather than by inspection.

`source_content_coverage` is `content_coverage(source, produced)` in `nodes/branches.py`: the
fraction of the source's **distinct normalised tokens** that appear anywhere in the production,
unordered. It is a content-scoring statistic. The branch's own spec already disclaimed it —
`specs/20260817-triage-workflow-dag/branch-speech.md`: *"Discourse-content scoring keys are not in
the inventory. Story recall and picture description have established content-scoring instruments;
this document does not invent them."* — and the move that created the gate said the same:
`specs/20260921-gates-in-verdict/implementation.md` records the reassignment as *"reproducing the
old code's behaviour"* with *"re-deriving is out of scope"*.

## 2. What conformance for these families now tests

`conformance_gate_names` no longer takes an anti-pattern. The group is the whole rule, so
`FREE_RESPONSE` resolves to `("response_min_s",)` for every row in it, recall included:
`response_duration_s >= 0.5`, the hull of the consensus's lexical words against half a second.

That is the existing production term, derived in `config-derivations.md` as *"half a second holds
at most one short word, so it is the shortest extent that can be a response to an instruction
asking for one."* It reads whether a response was produced. It does not read whether that response
is a **sentence**.

**The owner's words name a test that does not exist.** A gate called `response_is_a_sentence`
appears once in this repository, as one line of illustrative YAML in
`specs/20260921-gates-in-verdict/design.md`, and the same spec's implementation note says why it
never shipped: *"`ordered_match_min: 0.75` and `response_is_a_sentence: true` appear in the
design's illustrative YAML; both are refits, both are out of scope by the design's own last
section, and neither is shipped."* There is no design for it, no algorithm, and no validation. A
pickaxe over every ref for `finite verb`, `pos_tag`, `sentencehood`, `is_sentence`, `sentence_min`
and `spoken by the participant` returns nothing, in tracked files, untracked files, every worktree
and 246 dangling objects. Any recollection that an nltk POS approach was *validated as
discriminating correctly* is not supported by anything in the tree.

So the answer to "is the right term `response_min_s` as it stands, or the sentence test" is:
**`response_min_s` as it stands, for now, because it is the only one of the two that has been
built and measured.** It is a weaker test than the owner asked for — it admits a response of one
long word — but it is a production test, which is the axis the correction moves conformance onto,
and it decides nothing about content.

### What shipping the sentence test would take

It is small, and it is not free:

- **No new dependency.** `nltk>=3.9` is already declared in the `nlp` extra of `pyproject.toml`
  and resolved at 3.9.4 in `uv.lock`; `averaged_perceptron_tagger_eng` is already downloaded by
  `audio_analysis/harvesters.py`. No triage module imports nltk today, so this would be the first,
  and the download has to become a declared, offline-safe asset rather than a lazy fetch —
  a cluster node with no egress must not fail a branch on it.
- **A reading, then a gate.** SPEECH would write one new `measure` finding over the consensus
  words — the branch measures, VERDICT decides — and `verdict.gates` would gain a boolean gate
  reading it. A boolean gate is new: every gate in `GATE_SPECS` today is a numeric bound with
  `at_least`/`at_most`.
- **A derivation, which is the actual work.** "Finite verb plus a nominal" over an ASR transcript
  with no punctuation and no casing is a claim about the tagger's behaviour on this corpus, not a
  definition. It needs the same shape of evidence `branch.breath_coverage_min`'s deletion has: the
  statistic's distribution over the families it would gate, and a paired contrast that shows it
  separates recordings a reader would call a response from ones they would not. Without that it
  is another unfitted prior, which is what was just removed.

That is a separate task with its own measurement. Nothing here builds it.

## 3. What became of `coverage_min`

**Deleted** — from `GATE_SPECS`, from `verdict.gates.by_group.FREE_RESPONSE`, from
`recording_vectors.GATE_NAMES` and from the viewer's gate axes. The parquet schema is bumped 3 → 4
and its three `gate_coverage_min*` columns are gone.

The three candidate dispositions and why this one:

| disposition | why not |
| --- | --- |
| retained as a **flag ground** (`FLAG_GATES`) | `FLAG_GATES`' own contract is *"it says something about the recording's circumstances, not about whether the participant performed the instruction."* Coverage is squarely the second. `verdict.deviation_flags` is false corpus-wide and `conformance_flags` is true, so a flag ground is a genuinely different mechanism from a conformance term — but the thing it would flag is still how much of the story was recalled, on 93.6% of the recordings, which is the decision the owner withdrew wearing a different label. |
| retained as a **bound recorded beside the reading**, acting on nothing | A number in `verdict.gates` that no rule applies is read by the next person as a rule. This tree has ruled on exactly that before: `branch.breath_coverage_min` was *"**deleted**, not left null … a key nothing reads is worse than a missing one"*, for a bound with the same provenance — a majority prior, never fitted, decided against a statistic whose distribution has no structure at the bound. |
| **deleted** | Chosen. |

**The reading survives, which is the point.** `source_content_coverage` is still computed in
`_speech_free_response` for every `verbatim_source` row, still written as a `measure` finding, and
still a column in `recording_vectors` (`SCALAR_MEASUREMENTS`) and a viewer measurement axis. A
content-scoring instrument is what would consume it, and it will find it where it was.

The bound was also above its own distribution, which is why it fired almost everywhere; §5 has the
figures.

## 4. `verbatim_overlap_max` and `source_content_coverage` are different statistics

Checked, not changed — it is an anti-pattern check and the owner did not ask about it.

The config reads as if the two bounds sandwich a narrow band:

```
coverage_min: 0.5          # source content a recall must realise
verbatim_overlap_max: 0.5  # source content overlap above which a recall is verbatim
```

and `config-derivations.md` said of `verbatim_overlap_max` *"the same convention on the same
measure, for source content rather than prompt."* **They are not the same measure.** Both are
computed in `_speech_free_response` from the same two normalised token streams — the stimulus
alignment's expected tokens and the consensus's lexical words — and there they part:

| finding | function | statistic |
| --- | --- | --- |
| `source_content_coverage` | `content_coverage(source, produced)` | `|set(source) ∩ set(produced)| / |set(source)|` — **unigram types**, unordered |
| `verbatim_overlap_fraction` | `ngram_echo_fraction(source, produced, n)` | fraction of the source's distinct **n-grams** reproduced, `n = branch.echo_ngram_n = 3` |

Reproducing a trigram requires reproducing three tokens *in order*; a unigram type needs the word
to appear anywhere. So `verbatim_overlap_fraction <= source_content_coverage` in ordinary use and
the two bounds do not close a band: a retelling in the participant's own words that happened to
reuse most of the source's content words could read coverage 0.8 and trigram overlap 0.05, and a
verbatim read reads both high. The band was never as narrow as the two comment lines suggest, and
with `coverage_min` gone there is no band at all — `verbatim_overlap_max` stands alone as what it
always was, a deviation cut on a trigram statistic.

The corpus bears the separation out; §5 has the paired figures.

## 5. The measurement

[`measurement.md`](measurement.md) — the corpus effect on the finished r3 tree, both halves:
the fall on the two recall families, and that nothing else moves.

The two scripts that produced it, read-only over the run tree:

- [`family_gates.py`](family_gates.py) — the per-family conformance gate names on both sides of
  the change, built from the live expectation rows so the A/B is one table.
- [`corpus_scan.py`](corpus_scan.py) — one row per recording: the decision the run recorded, the
  readings from its store, and the conformance under both rules. It re-derives the recorded
  decision as a self-check before reporting any difference.

## 6. What changed in the code

| file | change |
| --- | --- |
| `nodes/gates.py` | `RECALL_CONFORMANCE_GATES`, `VERBATIM_SOURCE` and the reassignment removed; `conformance_gate_names` loses its `anti_pattern` parameter; `coverage_min` removed from `GATE_SPECS` |
| `nodes/verdict.py` | the one call site |
| `data/config/default.yaml` | `verdict.gates.by_group.FREE_RESPONSE.coverage_min` removed |
| `recording_vectors.py` | `coverage_min` out of `GATE_NAMES`; `SCHEMA_VERSION` 3 → 4 |
| `viewer/decode.js`, `viewer/axes.js`, `viewer/recording_vectors_viewer.html` | the decoder's schema version and the gate axis |
| `specs/20260922-compact-recording-vectors/schema.md` | the version, the bump table, and the gate list |

`anti_pattern` itself is untouched: it still selects which overlap `_speech_free_response`
measures and which cut raises `stimulus_mismatch`. What it no longer does is re-gate the task.
