# TAXONOMY

Classifies which kinds are in the recording. [`routing.md`](routing.md) reads the classification and
decides which branches run.

## Signature

```
taxonomy(store) -> fail(reason) | flag(reason, kinds) | pass(kinds)
```

Reads and writes the [element store](store.md). Writes one `kind` element per kind with its state and
the evidence behind it.

**TAXONOMY runs no models.** Every piece of evidence it reads was written by
[`PREPROCESS`](preprocess.md). It is a fold over stored classifications, not a detector committee.

**It infers; it classifies. It does not predict.** The word matters here because a branch may
subsequently refute the classification for its own kind ([`verdict.md`](verdict.md)), and a
classification that called itself a prediction would invite being scored as one.

**It localises nothing.** Which cough, which vocal task, and where, belong to the branches.

## What it reads

| element | author | used for |
| --- | --- | --- |
| `yamnet_windows` | PREPROCESS | speech-family and airway-family labels in the window sets |
| `ast_windows` | PREPROCESS | the same families, on AST's 10.24 s grid |
| `hear_windows` | PREPROCESS | cough and breath windows |
| `consensus_transcript`, `word` elements | PREPROCESS | lexical evidence for the speech kind |
| `phonation_spans` | PREPROCESS | sustained-phonation and glide spans with their `duration_s` |

**Hints are not an input.** TAXONOMY classifies from acoustics alone. A hint may force a branch to run
([`routing.md`](routing.md)) and may be compared against the branches' conclusions
([`verdict.md`](verdict.md)), but it never enters the classification, because a classification that
reads the declaration cannot disagree with it.

## The three kinds and their rules

| kind | what it is | classified from |
| --- | --- | --- |
| **speech** | lexical content | YAMNet/AST speech-family labels present in the window sets, **and** lexical words |
| **airway** | non-voice, non-speech vocal-tract sound: cough, breath, throat clear | HeAR cough/breath windows, **and** YAMNet/AST airway-family labels |
| **voice** | phonation that is neither: sustained vowels, humming, glides | `phonation_spans` of long duration |

### speech

Two evidence lines, both read from the store:

| line | evidence |
| --- | --- |
| acoustic | a window whose set contains a member of `taxonomy.speech_labels`, from `yamnet_windows` or `ast_windows` |
| lexical | `word` entities from the consensus transcript. **Bracketed and onomatopoeic events are not words** and carry no lexical evidence — see [`preprocess.md`](preprocess.md) |

Present when both lines carry evidence at or above their configured floors; absent when neither does;
uncertain otherwise.

### airway

| line | evidence |
| --- | --- |
| health-acoustic | a `hear_windows` window whose set contains a member of `taxonomy.hear_airway_labels` |
| acoustic | a window whose set contains a member of `taxonomy.audioset_airway_labels`, from `yamnet_windows` or `ast_windows` |

Present when both lines carry evidence at or above their configured floors; absent when neither does;
uncertain otherwise.

### voice

Classified from `phonation_spans` alone: **present when a sustained-phonation or glide span's
`duration_s` reaches `taxonomy.voice_min_duration_s`.** The span's production mode — voiced, unvoiced
or mixed — does not condition this; a disordered voice sustaining without periodicity is phonation.

Below the duration threshold and above a configured shorter floor the kind is uncertain; with no
phonation span at all it is absent.

## States

| state | meaning |
| --- | --- |
| **present** | the kind's rule is met |
| **absent** | every line the rule names carries evidence below its floor |
| **uncertain** | the lines disagree, or evidence sits between the floors, or a line's evidence is missing from the store |

**A missing derivative is not absence evidence.** A classifier that wrote no windows, or a Praat pass
that produced no spans because it did not run, leaves the line `unavailable`, and a kind whose only
line is unavailable is `uncertain`, never `absent`.

## Every threshold is configurable

| key | governs |
| --- | --- |
| `taxonomy.speech_labels`, `taxonomy.audioset_airway_labels`, `taxonomy.hear_airway_labels` | the label families, per kind. Vocabulary, not thresholds |
| `taxonomy.presence_floor.<kind>.<line>` | how much of a line's evidence a kind needs |
| `taxonomy.voice_min_duration_s` | the long-duration cutoff for the voice kind |
| `taxonomy.voice_uncertain_duration_s` | the shorter floor below which a phonation span is not even uncertain |

Each key carries its derivation in `data/config/default.yaml`. A key with no derivation ships
**null** and the run fails at load rather than substituting a number.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | every kind is absent |
| `flag` | any kind is uncertain |
| `pass` | every kind is present or absent, and at least one is present |

A `fail` here is not a file verdict: [`verdict.md`](verdict.md) decides what an all-absent
classification means, and [`routing.md`](routing.md) decides what still runs.

## Product

```
outcome:  fail(reason) | flag(reason, kinds) | pass
verdict:  { kinds: { airway: state, speech: state, voice: state } }
view:     the kind element ids
```

Each `kind` element carries, per evidence line, what that line said, the elements it read, and the
score or duration behind it, so a reader can see why a kind is uncertain rather than only that it is.

**All three kinds are screened.** There is no `not_screened` state and no residual kind: `voice` has
its own rule and its own evidence, and nothing in this graph is a kind by virtue of what the other
kinds did not claim.

## Out of scope

Running any model, localising anything, naming which airway event or which vocal task, reading
hints, and deciding which branches run.

Derivations live in [`benchmarks/taxonomy.md`](benchmarks/taxonomy.md).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `taxonomy.presence_floor.<kind>.<line>` | how much window or lexical evidence each line needs before it says present; **null** per line until fitted |
| `taxonomy.voice_min_duration_s` | the long-duration cutoff at which a phonation or glide span makes the voice kind present; **null** until fitted, across voiced, unvoiced and mixed production |
| `taxonomy.voice_uncertain_duration_s` | the shorter floor separating `uncertain` from `absent` for the voice kind; **null** until fitted |
| `taxonomy.speech_labels` | the AudioSet speech family, beyond the single `Speech` label the earlier list carried |
