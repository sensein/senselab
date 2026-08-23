# TAXONOMY

Decides which kinds are in the recording, so the graph knows which branches to run.

## Signature

```
taxonomy(store) -> fail(reason) | flag(reason, kinds) | pass(kinds)
```

Reads and writes the [element store](store.md). Writes one `kind` element per kind with its presence
state and the evidence behind it. **It localises nothing** — every detector answers presence on its own
terms, and no grid is shared.

## The three kinds

| kind | what it is | screened here? |
| --- | --- | --- |
| **airway** | non-voice, non-speech vocal-tract sound: cough, breath, throat clear, mouth sounds | yes |
| **speech** | lexical content | yes |
| **voice, no words** | vocalic activity that is neither — phonation, humming, imitation | **no** |

The third kind is **not screened**: it is whatever the other two do not claim. It has no roll-up label in
any detector's vocabulary, and its determination needs a phonation gate that belongs to
[`VOICE`](branch-voice.md) step 2. So TAXONOMY screens two kinds and the residual passes through by
construction. Earlier designs had TAXONOMY emit `residual_windows`; with a store the residual is a fold
over what the other branches assert, so nothing needs to produce it here.

## The screening set

| detector | contributes | barred from |
| --- | --- | --- |
| **YAMNet** | 521 AudioSet labels, the broadest vocabulary, the only explicit `Silence` | — |
| **AST** | a second AudioSet opinion | — |
| **CrisperWhisper** | the only source of **words**, so the speech kind rests on it | — |
| **HeAR** | strongest on breath | **speech** |

HeAR is barred from the speech kind: on verified speech it reports `Snore` 0.88 and `Speech` 0.01 across
six measurements. Not a weak vote, a wrong one.

## Windows — each detector on its own default

| detector | window | hop |
| --- | --- | --- |
| YAMNet | 0.96 s | 0.48 s |
| HeAR | 2 s fixed | 0.25 s |
| AST | 10.24 s | file-level |
| CrisperWhisper | tokens with timings | no grid |

Aggregate detection mode: each detector answers one presence question on its own grid, and the verdicts
combine. A detector whose window spans the file answers presence directly rather than by counting.

## Eligibility comes before any threshold

A detector is **eligible** for a kind only if its label space can express that kind. An ineligible
detector does not vote, and is not counted as a vote for absence.

**An empty window list is not a vote either.** A grid-reading member (YAMNet from the store's native
windows, HeAR from its own) that receives `[]` has no measurement, which is the same state as a
missing one, so it reads `unavailable` rather than `absent`. Folding `[]` to `absent` produced
absence evidence out of a detector that never ran — and absence needs unanimity, so one such member
could carry a kind to `absent` on nothing at all. A list whose scores are all below the floor is the
real absence evidence and still votes `absent`.

## Independence — count families, not detectors

| family | members |
| --- | --- |
| A — AudioSet | YAMNet, AST |
| B — lexical | CrisperWhisper |
| C — health-acoustic | HeAR |

**Airway has three eligible families; speech has two.** So `min_families` is per kind and cannot be one
global number: "two families agree" is a modest bar for airway and near-unanimity for speech.

## What defines a kind as present

| state | condition |
| --- | --- |
| **present** | at least `min_families[kind]` eligible families say present |
| **absent** | **every** eligible family says absent |
| **undecided** | families disagree, or any is unsure |

**Presence needs agreement; absence needs unanimity.** A low score means either "not there" or "there but
quiet or masked", and masked is the case this workflow exists to catch, so no single family may retire a
kind alone.

## Outcome

| outcome | when |
| --- | --- |
| `fail` | every kind is absent — nothing is predicted present |
| `flag` | any kind is undecided |
| `pass` | every kind is present or absent, and at least one is present |

**TAXONOMY is advisory, not a gate. Every branch runs regardless of what it says.** Its verdict is a
prediction, and [`verdict.md`](verdict.md) scores the branches against it.

This is deliberate and it costs compute. The alternative — gating, so a kind called absent skips its
branch — makes TAXONOMY's own errors invisible: a masked event is exactly what this workflow exists to
catch, and a screen that can retire a kind before any branch looks can never be shown to have been wrong.
It also makes `verdict.md`'s "absent, and the branch passed anyway" row unreachable by construction,
which is the row that detects precisely that failure.

Gating is available as an opt-in for cost-constrained runs, and a gated run is recorded as such, because
under gating an absent kind's `fail` carries no evidence.

## Product

```
outcome:  fail(reason) | flag(reason, kinds) | pass
verdict:  { kinds: { airway: state, speech: state, voice_no_words: "not_screened" } }
view:     the kind element ids
```

Each `kind` element carries, per eligible family, what that family said and the score behind it, so a
reader can see why a kind is undecided rather than only that it is.

## Out of scope

Localising anything, naming which airway event or which vocal task, and screening the residual.

Derivations live in [`benchmarks/taxonomy.md`](benchmarks/taxonomy.md).
