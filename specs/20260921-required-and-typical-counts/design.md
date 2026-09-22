# A count the instruction gave, and a count nobody gave

`expected_event_count` held both. It is replaced — no alias, no parallel field, no shim — by two
fields whose types say which kind they carry, what that number counts, and whether anything may be
judged against it.

## The defect, and why a docstring could not hold it

The field's docstring said *"how many events the instruction asks for, when it counts them"*. That
is true of eight AIRWAY rows and two SPEECH rows and false of the five syllable-repetition rows:

| family | count | where the number came from |
| --- | ---: | --- |
| `respiration-and-cough-fivebreaths` | 5 | the instruction — it is in the task's own name |
| `respiration-and-cough-threequickbreaths`, `-v2-threebreaths*` | 3 | the instruction |
| `voluntary-cough`, `breath-sounds` | 3 | the instruction |
| `respiration-and-cough-cough` | 5 | the instruction |
| `loudness` / `-v2` | 3 / 2 | the instruction — its `tokens` enumerate them |
| `diadochokinesis-pa`, `-ta`, `-ka` | 10 | **nobody** |
| `diadochokinesis-pataka`, `-buttercup` | 30 | **nobody** |

A participant told to take five breaths who gives four departed from the instruction. A participant
producing eight `/pa/` instead of ten did the task correctly: the instruction is *repeat as fast as
you can*, no number is spoken, and individuals vary. `measure-distributions.md` measures how much
they vary — `/pa/` runs 4 at the 5th percentile to 24 at the 95th, a factor of six — which is why
this kind of number can never be a bound.

**The second defect is a unit error, and it is the reason a unit must be explicit.** `pataka` and
`buttercup` declared 30 where `pa` declared 10, and the two are the same quantity: 30 syllables is
10 repetitions of a three-syllable carrier. One field, two units, nothing recording which. The
divergence was visible in the code — `decode_evidence` divided the declaration by the template's
vowel positions to recover a repetition count — but nowhere in the data.

## The shape

Three questions the single field conflated, and where each now lives:

| question | where the answer lives |
| --- | --- |
| **who gave the number** | the field's *type*: `RequiredCount` (an instruction) or `TypicalCount` (a measurement over the corpus) |
| **what it counts** | `unit: CountUnit`, mandatory on both, one of `events`, `tokens`, `repetitions` |
| **whether anything may be judged against it** | the type again: only `RequiredCount` is gateable, and `gates.UNGATEABLE_READINGS` refuses the other by name |

```python
class CountUnit(Enum):
    EVENTS = "events"
    TOKENS = "tokens"
    REPETITIONS = "repetitions"


@dataclass(frozen=True)
class RequiredCount:
    value: int
    unit: CountUnit


@dataclass(frozen=True)
class TypicalCount:
    median: int
    unit: CountUnit
    derivation: str
```

and on `Expectation`:

```python
required_count: RequiredCount | None = None
typical_count: TypicalCount | None = None
```

### Why two fields rather than one discriminated field

A single `count: RequiredCount | TypicalCount | None` also forces every reader to narrow, and was
the first shape considered. Two fields were chosen because the *read site* is then unambiguous
without narrowing at all: a consumer that wants a number it may compare against reads
`required_count` and cannot reach a median by any spelling. A row may in principle carry both — a
required count and a measured median of what people actually did — and the two-field shape can say
that; the union shape would have made the median displace the instruction. No row carries both
today, and a test says so.

An enumerated `source: INSTRUCTION | CORPUS` field on one class was rejected for the reason the
original field failed: it puts the distinction somewhere a caller has to remember to consult.

### What each kind refuses at construction

- `RequiredCount(0, …)` raises. Zero is not a number an instruction speaks.
- `TypicalCount(10, …, "")` raises. **A measured number that cites no measurement is exactly the
  defect this change removes**, so the citation is not optional: the `derivation` is the spec path
  the median was measured in. Replacing an underived 10 with a measured 10 is the point.

### Where the ungateable rule lives

`gates.py` declares both finding names and the set that may never be bound:

```python
REQUIRED_COUNT = "required_count"
TYPICAL_COUNT = "typical_count"
UNGATEABLE_READINGS = frozenset({TYPICAL_COUNT})

GATE_SPECS: dict[str, GateSpec] = _gate_specs({ ... })
```

`_gate_specs` raises `ValueError` if any gate in the table it is handed reads a name in
`UNGATEABLE_READINGS`, so a future gate bound to a median fails at **import**, not in review. The
declaration site is `gates.py` rather than `branches.py` because `branches` already imports
`gates` for `Pattern`; putting the names there also keeps "which readings a gate may bind" in the
one module that answers that question. A test parses `gates.py` and asserts the shipped table is
built *through* the factory, so the refusal cannot be made unreachable by assigning around it.

### What the store records

Two count findings, named apart, with the number under a key that says what it is:

| finding | evidence |
| --- | --- |
| `required_count` | `{found, required, unit}` |
| `typical_count` | `{found, typical, unit, derivation}` |

The typical finding deliberately carries **no** `declared`, `required` or `expected` key — nothing
a comparator would reach for by habit. A row that declares no count of a kind writes no finding of
that kind: a `required_count` entry whose `required` is `None`, which is what `-v2-hardcough` used
to write, is the same conflation in a new shape. The found value stays available either way, as the
`airway_events_found`, `ddk_repetitions_found` and `expected_tokens_matched` measurements.

## What each family declares

**Every required count keeps its current value.** Those came from the instruction and were not this
change's to alter; only the unit is newly recorded.

| branch | family | declares |
| --- | --- | --- |
| AIRWAY | `respiration-and-cough-cough` | required 5 events |
| AIRWAY | `voluntary-cough` | required 3 events |
| AIRWAY | `respiration-and-cough-fivebreaths` | required 5 events |
| AIRWAY | `respiration-and-cough-v2-threebreathsnose` | required 3 events |
| AIRWAY | `respiration-and-cough-v2-threebreathsmouth` | required 3 events |
| AIRWAY | `respiration-and-cough-threequickbreaths` | required 3 events |
| AIRWAY | `respiration-and-cough-v2-threebreaths` | required 3 events |
| AIRWAY | `breath-sounds` | required 3 events |
| AIRWAY | `-v2-hardcough`, `-breath`, `-v2-breath` | neither |
| SPEECH | `loudness` | required 3 tokens |
| SPEECH | `loudness-v2` | required 2 tokens |
| SPEECH | `diadochokinesis-pa` | **typical 11 repetitions** |
| SPEECH | `diadochokinesis-ta` | **typical 11 repetitions** |
| SPEECH | `diadochokinesis-ka` | **typical 10 repetitions** |
| SPEECH | `diadochokinesis-pataka` | **typical 10 repetitions** (was 30 syllables) |
| SPEECH | `diadochokinesis-buttercup` | **typical 10 repetitions** (was 30 syllables) |
| SPEECH | the six `-v2` syllable rows | neither: they are timed, not counted |
| VOICE | `loudness`, `loudness-v2` (pending declaration) | required 3 / 2 events |

Every median is the p50 column of
[`measure-distributions.md`](../20260817-triage-workflow-dag/measure-distributions.md)'s
`ddk_repetition_count_from_ppg_decode` scan over 62,273 recordings, and every row cites it.

The unit differs between VOICE's `loudness` row (events — the row's pattern is `EFFORT` and the
count is of productions) and SPEECH's (tokens — its matcher counts matched lexical tokens). They
are separate rows serving separate consumers, and the unit is the consumer's.

The five multi-syllable medians are all 10 repetitions, which is what the unit fix buys: the number
is now comparable across the five families, where 10-against-30 was not.

## What did not change

- **No gate reads either kind**, which `verdict.gates` also did not before. Deriving a tolerance for
  the required kind is the work this change enables and is not done here.
- **No conformance moves.** See [`corpus-replay.md`](corpus-replay.md).
- `decode_evidence` no longer divides a declared count by the template's vowel positions to recover
  repetitions, because the declaration is already in repetitions. Its `declared_event_count` and
  `declared_repetitions` covariates are one `typical_repetitions` covariate, and the report detail
  key `ppg_declared_event_count` is `ppg_typical_repetitions`.
- `_voice_sustained`'s `attempt_count` passed the row's count as its `declared` half. Every
  `SUSTAINED` row declares none and `align_voice` serves only `SUSTAINED` and `GLIDE`, so that half
  was always `None`; it is now written as `None` rather than read off a field that cannot hold one.

## What this supersedes

`ddk-template-decode.md` (9) ruled that `expected_event_count` **is kept, not renamed**, and named
the per-task count-kind property as owed and sequenced separately. This is that work, and the
rename it declined follows from it: a field that carries a kind cannot keep a name that asserts one
kind.

## Mutations

`mutations.py` in this directory. 14 mutations, each one line of the implementation rewritten into
a plausible wrong version of itself.

**14 of 14 caught.**

| mutation | caught by |
| --- | --- |
| M1 `-pataka` declares 30 again, as the retired field did | `branches_test`, `ddk_test` (4 failures) |
| M2 `/pa/` carries the underived ten rather than its measured eleven | `branches_test`, `speech_modes_test` (2) |
| M3 a measured median is declared without citing the measurement | collection errors: the dataclass refuses it |
| M4 a syllable-repetition row declares its median as a required count | `branches_test`, `ddk_test` (4) |
| M5 five breaths becomes a heuristic rather than the instruction's number | `branches_test`, `airway_test` (2) |
| M6 the median is written under `declared`, the key a comparator reaches for | `branches_test`, `ddk_test` (4) |
| M7 the required count is written without its unit | `branches_test`, `airway_test`, `speech_modes_test` (5) |
| M8 `UNGATEABLE_READINGS` is emptied, so a median becomes gateable | `gates_test`, `branches_test` (3) |
| M9 the gate table is assigned around its own refusal (`dict(` for `_gate_specs(`) | `gates_test`'s AST sweep (1) |
| M10 a family whose instruction counts writes no count against it | `airway_test` (3) |
| M11 the decoded count is compared in syllables against a median in repetitions | `ddk_test` (1) |
| M12 the repetition measurement stops carrying the median beside it | `ddk_test` (2) |
| M13 a required count of zero is admitted | `branches_test` (1) |
| M14 the speech token count is written against the wrong half of the row | `speech_modes_test` (1) |

M9 is the one that decided a test: the factory's refusal is only structural if the shipped table
actually goes through it, and nothing else would have noticed an assignment that bypassed it. The
AST sweep over `gates.py` was added for it.

## Found and not fixed

- **No tolerance is derived for the required kind.** This change makes one expressible — a gate may
  now be bound to `required_count` without the refusal firing — and derives none. `events_min` and
  `repetitions_min` still ask, at a bound of one, whether the asked-for sound happened at all.
- **The `p5`/`p95` spread is not carried.** `TypicalCount` holds a median only. A reader who wants
  to know that `/pa/` runs 4 to 24 has to open the derivation. Carrying an interval would be the
  honest shape for a distribution, and nothing consumes one yet.
- **Only the five measured families carry a median.** The other in-family rows that declare no
  required count — `-v2-hardcough`, the two `SOUND_COVERAGE` breath families, the six timed `-v2`
  syllable rows, every free-response family — carry neither kind, because no median has been
  measured for what they produce. `-v2-tuh`'s p50 of 14 is in the scan and is not declared here:
  the timed families are bounded by their own 5 s timer, and what a median over a fixed window
  means has not been reasoned.
- **`measure-distributions.md`'s own `n` differs slightly from the replay's.** The scan read 62,273
  recordings and the replay 62,1xx; both denominators are stated where they are used.
