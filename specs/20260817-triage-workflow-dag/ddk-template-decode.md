# DDK — the instrument becomes a probabilistic decode of the token's phoneme sequence

What replaces the per-syllable CV walk, the mechanism that replaces it, every parameter it
introduces and where each comes from, what it deletes, what it must not break, and what would have
to be measured before it lands. **Design only. Nothing here is implemented.**

The code this describes replacing is `src/senselab/audio/workflows/triage/nodes/ddk.py`; the
documents it supersedes are [`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md),
[`ddk-syllable-template.md`](ddk-syllable-template.md) and
[`ddk-cycle-counting.md`](ddk-cycle-counting.md). It preserves
[`ddk-task-extent-precedence.md`](ddk-task-extent-precedence.md) and
[`ddk-instrument-over-asr.md`](ddk-instrument-over-asr.md) intact.

## The owner's design, in the owner's words

> "why does it have to be per syllable given ppg extracts phoneme sequences. also it should be
> template against probablistic matches"

> "cycles are individual specific, so numbers are approximate there. also the template should be
> matched with place and vowel height approximations rather than exactness. it's the specific
> sequence and gaps between repeats that matters."

> "many extractors, including ppg are imperfect, and counts are inexact. so expected count is not a
> target it's a heuristic."

Read together those are one instruction and it is not "exact phoneme matching": the template is the
**phoneme sequence**, the match is **probabilistic** (posterior mass, not a decision) and
**approximate** (an equivalence class, not a phoneme), and the output is the **sequence and its
repetition timing**, not a count scored against a declaration.

## The three defects in the shipped shape

### 1. The per-syllable abstraction is inherited from a poorer instrument

The place vocabulary `branch.ddk_stop_places` was built for the **burst-spectrum** instrument
(`ddk_places`, `ddk.py:274`), where a spectral centroid over a 20 ms window after an onset recovers
place and can recover nothing else. The posteriorgram path inherited that vocabulary although the
derivative carries all 40 ARPAbet phonemes with probabilities per 10 ms frame
(`preprocess.py:790 write_ppg_posteriorgram`). The whole pipeline —
`phoneme_runs` → `cv_units` → `unit_places` → `cycle_scan` — exists to reduce 40 probabilities to
one of three place labels per syllable.

### 2. `argmax` discards the evidence that would settle the ambiguous cases

`phoneme_runs` (`ddk.py:525`) takes `np.argmax` per frame and everything downstream reads the
winner only. Two measured consequences:

- **Stop voicing.** The argmax raster's stop voicing agrees with the stimulus on 81.7 %–96.5 % of
  units inside a complete cycle, measured over all 7,994 declared-DDK recordings (Slurm job
  23105390, `figwork/voicing.py`): `-pa` 81.7 %, `-ta` 85.7 %, `-v2-puh` 88.2 %,
  `-buttercup` 89.8 %, `-pataka` 90.6 %, `-v2-buttercup` 90.5 %, `-ka` 91.2 %, `-v2-tuh` 91.6 %,
  `-v2-puhtuhkuh` 93.9 %, `-v2-kuh` 96.5 %. Labial is worst and velar is best. The shipped design
  already answers this by putting the voiced partner in the place class (`labial: [p, b]`), which is
  the right answer arrived at the wrong way round: it is a *hard* class over a *hard* decision, where
  what the derivative offers is the sum of two probabilities.
- **The coda.** See defect 3.

The brief this design answers also asserts that buttercup's coda `/p/` reads as `/t/` in 8 of 10
repetitions. **That was not reproduced here, and it cannot be read off the shipped instrument**, because
the shipped instrument emits no unit for a coda at all (below), so it never assigns the coda any
place. Whatever probe produced that number read the argmax raster directly. It is not load-bearing
here: the measurement below establishes the coda's recoverability without it.

### 3. The template cannot express a coda

`Expectation.sequence` is `tuple[Syllable, ...]` and `Syllable` is `(place, nucleus)`
(`branches.py:491`). `buttercup` is /bʌ.tər.kʌp/ — **CVCVCVC** — and its template has three
positions, so the final consonant has no position to occupy. `cv_units` (`ddk.py:607`) says so in
its own docstring: *"A consonant with no following nucleus — a coda, such as the `p` of
`buttercup` — closes no unit and is therefore counted as none."* `cycle_scan` then scans a
three-entry place sequence, so nothing anywhere in the instrument can read the coda.

**Measured.** Decoding the full seven-phoneme sequence `[b, ah, t, er, k, ah, p]` over all 7,994
declared-DDK recordings (Slurm job 23107265, below), the coda position is reached on **78.4 %** of
`-buttercup` and **68.9 %** of `-v2-buttercup` recordings, holds a median **150 ms** of frames, and
the labial class holds a median **0.74** (p10 0.49, p90 0.88) of the posterior mass over those
frames. The shipped instrument reads it on **0 %** by construction. The coda is there, it is
strongly evidenced, and only the template's shape was hiding it.

---

## The design

### The template is the token's phoneme sequence

One position per phoneme, read off the stimulus, with no syllable layer:

| family | template |
|---|---|
| `-pa` | `p aa` |
| `-ta` | `t aa` |
| `-ka` | `k aa` |
| `-v2-puh` / `-v2-tuh` / `-v2-kuh` | `p ah` / `t ah` / `k ah` |
| `-pataka` | `p aa t aa k aa` |
| `-v2-puhtuhkuh` | `p ah t ah k ah` |
| `-buttercup`, `-v2-buttercup` | `b ah t er k ah p` |

It is data with no fitted content: the derivation of every entry is the stimulus text. `Syllable`
and the `(place, nucleus)` pair are deleted; `Expectation.sequence` becomes a phoneme tuple.

### Matching is probabilistic and approximate: class-summed posterior mass

A template position does not demand its phoneme. It scores the **sum of posterior mass over the
equivalence class its phoneme belongs to**. A frame carrying 0.5 `/p/` + 0.3 `/b/` scores **0.8**
for a labial position. This is the soft form of the hard class collapse the shipped instrument
already performs, and it keeps the tolerance that makes the instrument work while spending the
probabilities instead of discarding them.

**Consonant classes: place.** Unchanged membership, new role —

```
labial    p b        alveolar  t d        velar  k g
```

**Vowel classes: height, with rhoticity carried separately.** The shipped config declares only
`low: [aa ae ah ao aw ay]` and `rhotic: [er]` and leaves the other eight ARPAbet vowels — `eh ey ih iy ow oy uh uw` — in no
class at all, which is admissible only because no DDK template names them. A template that is a phoneme
sequence must classify **every** vowel, so the inventory is partitioned. Diphthongs are classified
by their first target, which is the standard convention:

| class | members | reading |
|---|---|---|
| `close` | `iy ih uw uh` | close and near-close |
| `mid` | `ey ow` | close-mid |
| `open` | `aa ae ah ao aw ay eh oy` | open and open-mid |
| `rhotic` | `er` | the one r-coloured vowel |

**Read off the phonetic category, not fitted.** The `open` class contains the six a-initial vowels
the shipped `low` class already holds, so **for every phoneme any DDK template names, the membership
is byte-identical to what ships** — the reclassification changes nothing measurable on these ten
families, and that is deliberate: the extension exists so the inventory is total, not so the DDK
reading moves.

Two entries are choices rather than consequences, and both are flagged:

- **`eh` and `oy` in `open`.** /ɛ/ and the /ɔ/ onset of /ɔɪ/ are open-mid by the same reading that
  puts `ah` /ʌ/ and `ao` /ɔ/ there. Strict height is consistent; the shipped `low` class simply
  never had to decide. Unobservable on the ten DDK families. **Owner's call if the instrument is
  ever pointed at other stimuli.**
- **`er` kept out of `open`.** A pure three-way height reading puts `er` /ɝ/ with `ah` and destroys
  the one thing buttercup's middle nucleus does: distinguish position 3 from positions 1 and 5. The
  owner asked for "vowel height approximations", and rhoticity is a second dimension, not a height.
  Keeping it separate is what preserves the discrimination and it is what ships today. The measured
  consequence: the rhotic position is reached on 79.0 % of `-buttercup` recordings with a median
  realised mass of 0.67 (p10 0.17). Merging it into `open` would make positions 1, 3 and 5 of
  buttercup the same class and the template would carry three interchangeable vowel slots.

**The template's `/t/` is a flap.** In "butter" the /t/ is an intervocalic flap /ɾ/ in American
English. ppgs's ARPAbet-40 inventory has no flap symbol, so it must surface as `t`, `d` or `r`.
`t` and `d` are both alveolar, so the class absorbs two of the three. **Whether the alveolar class
should also admit `r` is measurable and is not measured here** — see the open questions.

---

## The mechanism: a cyclic left-to-right decode

### The topology

One tier of states per template position, plus one filler state between consecutive positions:

```
              ┌──────────────────────── wrap ────────────────────────┐
              │                                                      │
  ──►  pos 0 ─┼─► F0 ─► pos 1 ─► F1 ─► … ─► pos N-1 ─► F(N-1) ───────┘
        ▲ │   │    ▲ │      ▲ │     ▲ │        ▲ │        ▲ │
        └─┘   │    └─┘      └─┘     └─┘        └─┘        └─┘     (self-loops)
              └──────────── pos N-1 ─► pos 0 directly ─────────────┘

  pos i  =  a chain of D sub-states, only the last self-looping   (minimum phone duration)
  F i    =  one self-looping state between position i and position i+1
```

- **`pos i`** is a chain of `D` sub-states with no self-loop until the last, which self-loops. The
  chain is how a minimum phone duration is enforced structurally rather than by a penalty.
- **`F i`** sits between position `i` and position `i+1`, self-loops, and holds the frames the
  template does not account for — silence, transitions, intruding speech. `F(N-1)` sits on the wrap
  arc and therefore doubles as lead-in and lead-out: there is no separate garbage tier and no
  separate silence tier.
- The wrap arc `pos N-1 → pos 0` (directly, or through `F(N-1)`) is the repetition boundary. Every
  traversal of it closes one repetition.
- The path may start in any state and end in any state. A recording that begins mid-performance is
  decodable; a trailing partial repetition is simply not a completed one.

`N(D+1)` states — 21 for buttercup at `D = 2`. Viterbi is `O(T · arcs)` with `T ≈ 1,000` for a 10 s
recording, which is nothing.

### The emissions: one partition of the phoneme inventory

The emission model went through three forms before one worked, and **which one it is was decided by
measurement, not by argument** — the section after next records all three and a control, because the
two that failed are the evidence that this is the load-bearing choice.

The shipped form partitions the 40 ARPAbet phonemes into **the distinct equivalence classes the
template names, plus one `other` part holding everything else.** For `-pataka` the named classes are
`{labial, alveolar, velar, open}` and `other` is the remaining 26 phonemes including `<silent>`; for
`-buttercup` the named classes are those four plus `rhotic`.

| state | emission |
|---|---|
| `pos i` | `log m_c(t)` where `c` is the class of template phoneme `i` |
| `F i` | `log m_other(t)` where `m_other = 1 − Σ over the named classes` |

Every state's emission is then **the likelihood of the same observation under a different part of one
partition**, which is the property that makes log-scores comparable across states. It is a full
probability model of each frame with no free weight in it: the parts sum to one because the
posteriorgram does.

The reading is direct — *this frame's mass sits in the class this position expects*, against *this
frame's mass sits outside everything the template asks for*. Silence and intruding speech fall in
`other` for free, so no silence state, no garbage state and no lead-in or lead-out tier is needed:
`F(N-1)` on the wrap arc is all three.

### Every arc costs zero

There are no transition probabilities. The topology is a **constraint** on which paths are legal,
not a **prior** over which are likely. Two reasons, and the second is the stronger:

1. A duration prior would need measured per-phoneme durations on rapid DDK material, which do not
   exist. Published conversational phone durations are not those durations.
2. The obvious "neutral" alternative — proper uniform, `log(1 / out-degree)` — is not neutral. It
   penalises a state in proportion to how many arcs leave it, which is an artefact of how the
   topology was drawn and has no physical meaning. A state that can be left three ways is not
   thereby less likely to be occupied.

With zero-cost arcs and a chain-enforced minimum duration, **this recursion is dynamic time warping**
— `max{stay, advance}` over a local cost is DTW's recursion with the sign flipped. What the Viterbi
framing adds over calling it DTW is the cyclic topology, a local cost that is a likelihood rather
than a distance, and an explicit repetition-boundary event. See the comparison below.

### What the decode yields

| output | read from |
|---|---|
| repetition count | traversals of the wrap arc that completed |
| repetition start times, and the period between them | the frames those traversals occurred at |
| per-position realised mass | the mean of `m_i` over the frames charged to `pos i` |
| per-position occupancy | how many frames each position held — a phone-duration reading |
| the extent | first frame of the first completed repetition to last frame of the last |
| path score per frame | `total / T`, a per-frame log-likelihood |

**The per-position realised mass is the primary sequence-realisation measurement, not the repetition
count.** It is defined at every position on every recording where the position was reached, it
degrades continuously, and it is honestly ambiguous between "produced differently" and "read poorly"
— which is the correct epistemic state. The repetition count is a coarser summary of the same path.

---

## Alternatives considered

### DTW against a repeated template

Build an expected class-probability matrix for one repetition and align it to the posteriorgram.

DTW and the decode above are **the same recursion**, so the comparison is only about what surrounds
it. DTW's handling of the four cases:

- **Variable phone duration** — handled, identically: the warping path is the self-loop.
- **A missed phone** — handled, but silently. The standard `(1,1),(1,0),(0,1)` step pattern permits a
  template row to be crossed with zero width, which deletes the position leaving no record. The
  chain-enforced minimum duration above is what turns a deletion into a *reported low-mass position*.
- **An inserted syllable** — the same limitation as the decode: monotone warping must charge it to
  some template row. Neither approach localises an insertion without a filler that costs something.
- **Rate drift** — handled if the alignment restarts per repetition; **not** handled by a single
  whole-recording alignment against `k` concatenated repetitions, because that requires declaring
  `k` in advance, which is exactly what the owner's "counts are individual-specific" forbids. The
  cyclic topology is what removes the need to declare it.
- **Silence between repetitions** — must be written into the template as an expected silence of
  unknown length. That is a template entry with no stimulus behind it. The filler above absorbs
  silence without one, because `<silent>` falls in the `other` part of the partition.

DTW's one genuine advantage is the global band constraint (Sakoe–Chiba), which bounds how far the
path may deviate from the diagonal. That bound **is a rate assumption**, so it is unavailable here.

**Not recommended**, and the reason is not that it is worse at alignment — it is the same at
alignment. It is that it needs `k` declared, an expected-silence entry, and a distance function
chosen by hand, and it gives back an uncalibrated distance instead of a log-likelihood.

### Sliding-window correlation / matched filter

Form a one-dimensional class-mass series per class, build an idealised single-repetition kernel, and
cross-correlate; or take the autocorrelation of the class-mass series and peak-pick.

- **Variable phone duration** — not handled. A fixed-length kernel smears when phones lengthen.
- **A missed or inserted phone** — lowers the correlation and cannot be localised to a position.
  There is no per-position accounting at all, which is the owner's "specific sequence".
- **Rate drift** — breaks. Drift detunes a fixed kernel; the only remedy is a kernel bank, which is
  a rate grid, which is an unfitted parameter with a public interface.
- **Silence between repetitions** — must be built into the kernel's shape, another unfitted choice.
- **Parameters** — kernel length (a rate), kernel shape, and a peak-picking criterion. Three, none
  derivable.

**Not recommended as the instrument.** It is worth naming for one narrow job it is good at: it needs
no alignment at all, so it is the natural *second, independent* estimate of repetition rate — the
posterior-domain counterpart of the envelope modulation spectrum. See the envelope section.

### Recommendation

**The cyclic decode.** It is the only one of the three that yields repetition boundaries (which the
gap statistics need), per-position accounting (which the sequence realisation needs) and a calibrated
score, and it is the only one whose parameter count can be brought to two — both derived.

It is recommended on measurement, not on argument: the emission-model table below shows it running
over the corpus and recovering both positions the shipped instrument is structurally blind to. The
residual risk it carries is **over-counting**, which A2 bounds on the control families and A5 would
settle; the risk it removes — silently converting extractor error into apparent participant error —
is the one the owner's ruling on counts names directly.

---

## Every parameter, and where it comes from

### Shipped, with a derivation

**`min_phone_frames` (`D`) — the chain length that enforces a minimum phone duration.**
Not a config key: computed at read time from the posteriorgram's own frame period and the burst
window the config already declares. `D = ceil(burst_window_ms / (1000 · seconds_per_frame))`.
`branch.burst_window_ms: 20.0` carries its derivation already — the stop burst and its aspiration
occupy the first 10–25 ms after release (Blumstein & Stevens, 1979) — and `seconds_per_frame` is on
the measurement entity. **Measured across all 7,994 declared-DDK posteriorgrams: median 10.00 ms,
min 10.00 ms, max 10.26 ms.** So `D = 2` on this corpus.

It is a **floor against degenerate traversal, not a duration model**, and it must stay as small as
the frame rate allows so that it never binds against real production. The boundary is stateable: the
fastest clinically reported `/pataka/` rate is about 8 cycles/s, which is 48 phones/s, which is
21 ms per phone. `D = 2` (20 ms) sits just under that. **`D = 3` would make the fastest reported
pataka rate structurally undecodable.** That is the argument against ever tuning it upward.

**`emission_floor` — the value `log` is taken of when a class holds no mass.**
Not a config key either: derived at read time from the sidecar's own recorded `dtype`.
`write_ppg_posteriorgram` stores the array as **float16** (`preprocess.py:822`) and the entity
carries `"dtype": "float16"` (`preprocess.py:842`), so a posterior below float16's smallest positive
subnormal — 5.960 × 10⁻⁸ — is stored as exactly zero and its log is −∞. The floor is
`np.finfo(np.dtype(entity.attributes["dtype"])).smallest_subnormal`: **the smallest value the
recorded data can distinguish from zero**, which is a storage fact and not a tuning knob, and which
moves on its own if the sidecar's dtype ever moves.

### Shipped as data, not as numbers

**The class memberships** (`labial: [p b]`, …; `open: [aa ae ah ao aw ay eh oy]`, …) and **the
templates** (one phoneme sequence per family). Read off the phonetic category and the stimulus text
respectively. These are the only points that can be null, and a null one leaves the dependent
conformance `UNDETERMINED` with the key named in `params.missing`, exactly as today.

### Not shipped, and why

**A skip arc and its penalty.** A deleted phone does not need one: the position is traversed at its
`D`-frame minimum on near-floor mass and is *reported* as a low-mass position, which is more
informative than a structural deletion. A skip arc would buy 20 ms of frames at the price of one
unfitted number.

**A transition prior of any kind.** See "every arc costs zero" above. The named upgrade is an
explicit-duration model (an HSMM with a per-phone duration distribution), and what it needs is
measured per-phoneme durations on DDK material at DDK rates, per family. Those do not exist.

**A plausibility band on the decoded rate.** Still a verdict-level decision awaiting a derivation,
unchanged from the ruling in [`config-derivations.md`](config-derivations.md). A branch does not
decide.

**A threshold on per-position realised mass.** This is what a `syllable_sequence_mismatch` deviation
would need, and it has no derivation. See "what it deletes".

### The emission model is the load-bearing choice, and it was chosen by measurement

The section above states the answer. This records how it was reached, because two earlier forms —
both defensible on paper, one of them the textbook default — fail on the corpus, and that is what
establishes that the filler's emission is not a detail.

All four rows are the **same topology, the same `D = 2`, the same floor, the same class
memberships**. Only the filler's emission differs. Medians of decoded repetitions per family, against
the shipped `cycle_scan` on the same recordings:

| filler emission | `-pa` | `-ta` | `-ka` | `-pataka` | `-buttercup` | `-v2-puhtuhkuh` | `-v2-buttercup` | n |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **(A)** `log max_j p_j`, lead-in/lead-out only | 9 | 10 | 7 | 4 | **1** | **1** | **1** | 7,994 |
| **(B)** no filler at all — every frame on a position | 12 | 12 | 11 | 10 | 10 | 7 | 7 | 150/fam |
| **(D)** `log(1 − m_{i+1})`, per position | 8 | 9 | 7 | 3 | **1** | 2 | **0** | 400/fam |
| **(E)** `log m_other`, the partition above | 11 | 11 | 10 | 10 | 10 | 7 | 6 | 400/fam |
| shipped `cycle_scan` | 10 | 10 | 8 | 6 | 6 | 2 | 4 | — |
| the row's declared cycles | 10 | 10 | 10 | 10 | 10 | — | — | — |

Jobs 23107265 (A, all 7,994), 23107979 (B, and a control at `log` of the top-two mean), 23108420 (D),
23109184 (E), all on `mit_preemptable`. Scripts and per-recording rows in
`/orcd/scratch/bcs/002/satra/checks_20260916/figwork/`.

**(A) is the textbook parameter-free background model and it absorbs the recording.** The argument
for it is sound as far as it goes: a class sum that contains the frame's argmax phoneme necessarily
exceeds `max_j p_j`, so genuine template material always beats garbage. What the argument misses is
the *transition* frames, which are most of them: under (A) a median of **41 %** of the recording lay
inside the decoded extent and only **32 %** of frames were charged to any template position. On the
three-position families it decodes one repetition and abandons.

**(B) removes the filler entirely and the path is then forced to keep wrapping**, which over-counts —
12 against a declared 10 on `-pa`, 11 on `-ka`. Weakening (A)'s garbage emission from `max_j p_j` to
the mean of the top two posteriors reproduces (B) almost exactly, which is the diagnostic: the
outcome is controlled by *how strong the filler is*, and "how strong" is a weight.

**(D) fails for a reason worth writing down, because it is the reason (E) is right.** (D) made the
filler between positions `i` and `i+1` emit `log(1 − m_{i+1})` — "the class we are waiting for is not
here" — on the argument that `log m` and `log(1 − m)` are complementary and therefore carry no free
weight. They are complementary, and they are complementary **for different events**, which is the
error: the filler's event ("the next class is absent") is trivially satisfied almost everywhere, so
its emission is `log 1 = 0` and it is free, while every position pays a real `log m < 0`. A path that
never advances scores zero and a path that decodes the task scores less. Inspecting decoded paths
shows exactly that — positions collapsing to their two-frame minimum while the filler holds the rest
of each phone, and on recordings where one position reads poorly, a single filler state holding 821
consecutive frames to the end of the recording.

**The lesson generalises past this instrument.** When states in a decode emit likelihoods of
different events, their log-scores are not comparable and the decode optimises an incoherent
objective — silently, and with plausible-looking output. The partition in (E) is the fix and the only
thing it does is make the parts mutually exclusive and exhaustive.

### What (E) reads on the corpus

400 recordings per family, `D = 2`:

| family | repetitions | period s | syll/s | extent / CV hull | extent / recording | filler frames | per-position realised mass |
|---|---:|---:|---:|---:|---:|---:|---|
| `-pa` | 11 | 0.270 | 3.70 | 1.00 | 0.72 | 0.46 | p 0.71, aa 0.80 |
| `-ta` | 11 | 0.250 | 4.00 | 1.00 | 0.73 | 0.41 | t 0.73, aa 0.75 |
| `-ka` | 10 | 0.290 | 3.45 | 1.00 | 0.71 | 0.48 | k 0.66, aa 0.77 |
| `-v2-puh` | 14 | 0.240 | 4.17 | 1.00 | 0.76 | 0.42 | p 0.74, ah 0.70 |
| `-v2-tuh` | 14 | 0.250 | 4.00 | 1.01 | 0.76 | 0.41 | t 0.74, ah 0.67 |
| `-v2-kuh` | 12 | 0.261 | 3.83 | 1.00 | 0.74 | 0.48 | k 0.71, ah 0.65 |
| `-pataka` | 10 | 0.561 | 5.35 | 1.00 | 0.81 | 0.31 | p 0.83, aa 0.76, t 0.79, aa 0.68, k 0.81, aa 0.80 |
| `-v2-puhtuhkuh` | 7 | 0.501 | 5.99 | 1.00 | 0.77 | 0.35 | p 0.88, ah 0.67, t 0.83, ah 0.53, k 0.86, ah 0.75 |
| `-buttercup` | 10 | 0.560 | 5.36 | 0.96 | 0.79 | 0.32 | b 0.78, ah 0.78, t 0.69, **er 0.57**, k 0.81, ah 0.83, **p 0.77** |
| `-v2-buttercup` | 6 | 0.540 | 5.56 | 0.93 | 0.74 | 0.33 | b 0.79, ah 0.79, t 0.68, **er 0.57**, k 0.85, ah 0.83, **p 0.78** |

**These are descriptive. Nothing in the design compares against them and no threshold is derived
from them.** Four things they establish:

1. **Every template position carries real mass, including the two the shipped instrument cannot
   reach.** buttercup's coda `p` realises 0.77 and its rhotic `er` 0.57, against 0.78–0.83 for the
   positions the shipped instrument does read. The coda is not marginal evidence.
2. **The three-position families stop sitting at roughly half the single-position families' count.**
   That ratio — shipped `-pataka` 6 and `-buttercup` 6 against `-pa` 10 and `-ta` 10 — is the
   signature of the losses defects 1–3 describe, and it is gone.
3. **Every decoded period lands inside `branch.modulation_band_hz: [1.0, 10.0]`**, the band already
   derived from published DDK rates, with syllable rates of 3.45–5.99/s.
4. **The decoded extent is the CV hull.** `extent / CV hull` is 0.93–1.01 across all ten families, so
   the extent the decode proposes for the `task_extent` union agrees with what the shipped
   instrument's hull already asserted — while covering 71–81 % of the recording rather than (A)'s
   41 %.

**And what they do not establish.** `-pa` at 11 and `-v2-puh` at 14 may be over-wrapping; nothing here
separates a real eleventh repetition from a spurious one, and only listening would. The per-recording
agreement with the shipped scan is loose — exact agreement on 7–24 % of recordings and within one on
13–46 % — which is what a systematic recovery of missed repetitions looks like and is also what an
over-count would look like. Distinguishing them is the acceptance test's job.

## Counts are heuristics, not targets

The owner, 2026-09-19: *"many extractors, including ppg are imperfect, and counts are inexact. so
expected count is not a target it's a heuristic."*

This is the governing constraint on everything the decode reports, and the shipped graph already
honours it, so the requirement is **not to regress it**:

- `expected_event_count` reaches the store only through `count(...)` findings, which write `found`
  beside `declared` and whose docstring says *"Asserts no discrepancy"* (`branches.py:280`).
- `write_findings` folds those into a `counts` measurement; `deviation_names` ignores counts, so a
  count can never become a deviation; and `verdict.deviation_flags` ships `false` in any case.

**Therefore no term of this design compares a decoded count against a declared one.** Not in
conformance, not in a score, not in a gate. The decoded repetition count and the declared count are
written beside each other and nothing folds them. The `ppg_cycles` / `ppg_declared_cycles` pairing the
cycle work already established is the right shape and is **descriptive only**; this document states
explicitly that nothing may fold that pair into a judgement.

### What each count is evidence of

The distinction the corpus forces is between *the participant did fewer* and *the extractor found
fewer*, and on the evidence the second is usually the true cause:

- The envelope peak walk resolved a median of **4** onsets against a declared 10 or 30, and **1 or 0**
  onsets on **24.8 %** of the recordings where it found a carrier at all (below).
- The posteriorgram CV walk finds no train on a known set of recordings whose ASR transcripts
  confirm the task was performed — 105 diagnosed in
  [`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md), of which 74 had CV units present
  whose *timing* failed the regularity test.

So each reported count carries what it is a count *of*, and none of them is a count of syllables the
participant produced:

| reported | what it is evidence of |
|---|---|
| repetitions decoded | how many times the decode found the template's classes realised in order. A floor on what was produced, never a measurement of it |
| per-position realised mass | how much of the posterior the expected class held where the decode placed that position. Ambiguous between production and extraction, and must be reported as such |
| per-position occupancy | how long the decode held that position. A duration reading, not a correctness one |
| period between repetition starts | the quantity the owner named. Measured over the decode's own boundaries and independent of any declared count |

**This is the strongest argument for the probabilistic match and the design should say so plainly.**
If every extractor is imperfect, then any mechanism that commits early to a hard decision — `argmax`,
a hard class membership, a count compared against a declaration — silently converts extractor error
into apparent participant error. Carrying the posterior mass all the way to the score leaves the
uncertainty where it belongs: a low realised mass is honestly ambiguous between *produced differently*
and *read poorly*, and the instrument says so rather than choosing.

### What `expected_event_count` should mean

It stays on the row and it stays in the store, and it stops being a target in the one place it still
reads like one: `count("expected_event_count", len(onsets), expectation.expected_event_count, …)`
(`ddk.py:1310`) puts a found count beside a declared one **whose units do not match** — the
declaration is 30 syllables for `-pataka` and the found value is envelope onsets. The pairing is
legible only because a reader knows both conventions.

**Proposal.** Rename the declaration to `declared_event_count` and carry it as a **covariate on the
measurement it qualifies**, not as the second half of a `count`. Then there is no place in the store
where a found number sits in a slot labelled "expected", and the heuristic reading is the only one
available. The number itself does not change and neither does any row.

---

## What it deletes, confirmed against the code

Every entry below was checked by reading the definition and every call site. The right-hand column
is what is orphaned and therefore has to change in the same commit.

### Code

| deleted | `ddk.py` | what it was | orphans |
|---|---|---|---|
| `phoneme_runs`, `PhonemeRun` | 525, 407 | the argmax raster collapsed to runs | — |
| `phoneme_class`, `CONSONANT`/`VOWEL`/`OTHER` | 591, 386–388 | the three-way frame class | — |
| `stop_phonemes`, `nucleus_classes`, `admitted_nuclei` | 546, 559, 576 | class→phoneme inversions and the template-gated nucleus admission | `ddk_test.py:1405` |
| `cv_units`, `CvUnit` | 607, 422 | the CV syllable | — |
| `syllable_trains`, `SyllableTrain` | 640, 444 | the regularity segmentation | — |
| `ppg_reading`, `PpgReading` | 682, 468 | the walk's entry point | `ddk_test.py:67, 970` |
| `cycle_scan`, `CycleScan`, `NO_CYCLES` | 782, 743, 778 | the greedy place scan | `ddk_test.py:64, 1341–1383` |
| `cycle_gaps`, `cycle_rate`, `cycle_nucleus_fraction` | 826, 840, 859 | the scan's statistics | — |
| `unit_places`, `unit_nuclei` | 715, 729 | per-unit label lookups | — |
| `place_agreement`, `cycle_evidence` | 884, 905 | the two-instrument agreement, and the scan's findings | — |
| `ddk_places` | 274 | **the burst-spectrum place instrument** | `branch.burst_window_ms`, `branch.place_centroid_bands_hz`, `branch.place_margin_db` — `ddk_places` is their **only** consumer (verified: no other call site in `src/`) |
| `dispersion_by_position` | 364 | one dispersion per cycle position | `ddk_test.py:66, 923` |
| `events_in_span` **as used by DDK** | `ddk.py:1298` | the envelope peak walk over the carrier | none — `airway.py:359` is the other caller and is untouched |

`cv_covered_extent`, `cv_task_extent`, `merged_task_extent`, `contradicted_words`,
`instrument_authority` and `intervals_of`/`dispersion`/`trend` **survive**, with `cv_covered_extent`'s
input changing from "the hull of every CV unit" to "the hull of the completed repetitions" and its
gate from "a complete cycle or a train" to "a complete repetition".

### Config

| key | fate |
|---|---|
| `branch.ddk_stop_places` | membership unchanged, role changed from *argmax label map* to *emission class*. **Rename it** — pre-alpha rules out an alias, and `ddk_` no longer describes it. `branch.phoneme_place_classes` |
| `branch.ddk_nucleus_classes` | replaced by the total four-class partition above. `branch.phoneme_vowel_classes` |
| `branch.ddk_interval_tolerance` | **deleted.** It segmented by regularity; the decode segments by the template. `config-derivations.md` already records that it "separates nothing" |
| `branch.ddk_min_repetitions` | **deleted.** There is no train to have a minimum length |
| `branch.burst_window_ms` | **kept**, and its consumer changes: it stops feeding `ddk_places` and starts deriving `D` |
| `branch.place_centroid_bands_hz`, `branch.place_margin_db` | **deleted** with `ddk_places` |
| `branch.train_min_s`, `branch.modulation_band_hz`, `branch.rate_prominence_min` | **kept** — the envelope modulation channel survives (below) |
| `branch.smoothing_window_s`, `peak_prominence_db`, `trough_return_db`, `event_min_s` | **kept** — AIRWAY reads them |

**A defect found in passing.** `branch.ddk_stop_places` and `branch.ddk_nucleus_classes` are
described as campaign-overridable data mappings in both `default.yaml:301-313` and
`ddk-syllable-template.md:42` — *"a campaign may add a place without editing the installed package"* —
but neither is in `DATA_MAP_PATHS` (`config.py:45-58`), which lists only
`branch.place_centroid_bands_hz` among the branch place keys. So an override adding a place is
**rejected as a schema violation today**, contrary to two documents. Their replacements should be
added to `DATA_MAP_PATHS`, and the same commit should note that the claim was untrue for the
originals.

### Measurements and report keys

| deleted | why |
|---|---|
| `ddk_cv_unit_count` | there is no CV unit. (It also has **no report reader** — `syllable_detail` takes `ppg_cv_units_n` off `PPG_RATE`'s covariate instead, so this count is absent from the report whenever the train is absent. A live defect, settled by deletion) |
| `ddk_cycle_rate_from_ppg_places_hz` and its six detail keys (`ppg_cycles`, `ppg_declared_cycles`, `ppg_cycle_rate_hz`, `ppg_cycle_gap_cv`, `ppg_cycle_consumed`, `ppg_cycle_insertions_n`) | the scan they summarise is gone. Replaced by the repetition count, the period statistics and the per-position mass |
| `ddk_cycle_nucleus_fraction` | subsumed. Per-position realised mass is defined at **every** position, consonants included, and does not carry the by-construction 1.00 ceiling `ddk-cycle-counting.md` documents for eight of ten families |
| `ddk_place_agreement_ppg_vs_burst` | deleted with the burst instrument. Its purpose was to measure whether the PPG place could be trusted as the place decision; under the decode there is no place decision to make, only mass to sum |
| `ddk_ppg_interval_dispersion` and its `by_position` | the by-position split existed because envelope onsets land differently per place. The decode measures one period per repetition, start to start, so there is nothing to split. Per-position **occupancy** replaces it with a quantity that is actually about the phones |
| `syllable_place` | the "wideband absent" placeholder, deleted with the burst instrument |
| `sequence_collapse_fraction` | subsumed, and improved: a `/pa-pa-pa/` produced for `/pa-ta-ka/` is now a per-position mass vector with position 0 high and positions 2 and 4 near floor, which says *which* positions collapsed. `dominant_place` needed an argmax over resolved places and is gone with it |
| `realised_cycles` | replaced by the repetition count |
| `ppg_train` (span role) and `PPG_RATE`'s five detail keys | **see the open question.** The train was a segmentation by regularity, which `ddk-cycle-counting.md` already argued was the wrong unit to score over. `ppg_rate_hz` — syllables per second — survives with a new derivation: `(vowel positions per template) / (median period)`, needing no train at all |

**`ddk_cv_instrument_reading` survives and must.** `pii_interlock_test.py:459` filters store
entities by that name to assert the CV instrument leaks no transcript text, and
`ddk-instrument-over-asr.md` is the contract it enforces. Its **value** changes from the realised
place series per CV unit to the realised class series per repetition; it still carries no word text.

### What it deletes that is not code

- **The phase problem.** `ddk-cycle-counting.md`'s whole subject. The decode establishes phase as
  part of finding the path; there is no index to take modulo anything.
- **The coda gap.** Defect 3 above.
- **The voicing abstraction that was about to be added.** Voicing never needs its own channel: the
  place class already holds both partners, and summing their mass is what the 81.7 %–96.5 % argmax
  agreement rate was going to be worked around by.
- **`syllable_sequence_mismatch`.** This loses its only writer and **cannot be re-raised without a
  threshold on per-position mass, which has no derivation**. See the open questions; the vocabulary
  entry is in `branches.py:123` and `write_findings` refuses any deviation not in it.

### Readers that must change in the same commit

`common.py:616-643` `BRANCH_MEASURES["SPEECH"]` — of its 27 entries, four (`speaker_count`,
`words_n`, `speech_s`, `nontarget_speech_s`) are not this instrument's and three (`trains_n`,
`train_s`, `train_fraction`) keep their meanings; **every one of the other twenty is replaced or
deleted** by the table above. `report.py:1816, 1972` and `figure.py:2394, 2833` read that table generically, so they need no
code change but their output changes. Tests: `ddk_test.py` throughout,
`report_test.py:145-166` (`_SYLLABLE_DETAIL`, 22 keys) and `1483-1549`, `branch_lanes_test.py:228-247`,
`report_test.py:2452-2488` (renders `speech span: ppg_train/syllable_train_from_ppg`),
`speech_modes_test.py:619, 659`, `pii_interlock_test.py:459`, `branch_contract_test.py:532-539`
(`train_min_s` null override).

---

## What it must preserve

**The branch contract.** A branch returns spans, a conformance, and typed located deviations, and it
writes no `Outcome` and no `why` (`verdict.md` § *A branch reports; this fold decides*). The decode
changes what is measured, nothing about who decides. In particular there is **no plausibility band on
the decoded rate and no acceptance gate on the path score** — a branch does not decide.

**Conformance narrows and does not widen.** `Done` stays `true | false | UNDETERMINED` and the only
honest `false` remains what it is today: the instrument ran, and found no evidence the task was
performed at all — zero completed repetitions *and* no envelope carrier. A collapsed sequence, a
weak position, or fewer repetitions than declared is **never** `false`. This is exactly `_with_ppg`'s
present logic with the train replaced by the repetition; the point of restating it is that the
decode makes a tempting new `false` available (a low path score) and it must not be taken.

**`propose_span` refusing a span that names no evidence.** `branches.py:348` raises if
`proposal.derived_from` is empty. Every span the decode proposes names the posteriorgram entity, as
`cv_task_extent` does today.

**The `task_extent` union rule merged in `fe9167db`.** Unchanged in form and in every one of its
three cases. What changes is the *input on each side*:

| | today | under this design |
|---|---|---|
| CV side | hull of every CV unit, gated on a cycle or a train | hull of the completed repetitions, gated on one completed repetition |
| envelope side | `hull(onsets)` from the peak walk | **the carrier span's own extent** (see the envelope section) |
| merge | `min(starts)` … `max(ends)`, `production` `syllable_task_from_both` | unchanged |

The monotonicity argument in `ddk-task-extent-precedence.md` — the merged extent contains what each
instrument alone would have proposed, so no recording loses coverage — is preserved, and the
envelope-side change strictly improves it: on the recording that motivated that document the carrier
was 2.237–4.114 (1.88 s) while `hull(onsets)` was 0.0986 s.

**"The instrument did not run" versus "it ran and found nothing".** Three states stay distinct and
each keeps its own record:

| state | record |
|---|---|
| the posteriorgram derivative is absent | a measurement with no value carrying `unavailable="ppg_posteriorgram"`; conformance unaffected by this instrument |
| a class mapping is null | no findings at all from the decode; the key named in `params.missing`; conformance `UNDETERMINED` |
| the decode ran and completed zero repetitions | a measurement whose value is **0**, with the per-position mass vector beside it. This is a reading, not an absence, and it is the only one of the three that can support a conformance `false` |

The new instrument has **no numeric operating point that can be null** — `D` and the floor are both
derived from the store at read time — so only the class mappings can reach `params.missing`. That is
a real narrowing of the "unmeasured" surface and should be stated rather than discovered.

**The PII interlock.** `ddk-instrument-over-asr.md`'s frame is unchanged: additive records only, no
word text in any finding, nothing removed from the set of strings reaching `scan_for_pii`.

---

## The acceptance test

The brief this design answers originally proposed *"must reproduce a median of exactly 10 cycles
against a declared 10"*. **That is the wrong target** and the owner's correction is why: counts are
individual-specific and every extractor is imperfect, so a decoded count that matches a declaration
is as likely to be two errors cancelling as it is to be right, and a decoded count that does not
match is not evidence about the participant. The decode does in fact land on 10 or 11 on all five counted
families, where the shipped scan gives 10, 10, 8, 6, 6 — and that is offered below as *context*, in a
row that no criterion reads.

What the corpus comparison should demonstrate, in four parts. All four run over all 7,994 declared-DDK
recordings, paired per recording against the shipped instrument replayed on the same stored
derivative — the shape jobs 23105513, 23096468 and 23107265 already use.

### A1 — sequence realisation where the shipped instrument was structurally blind

The one part of this that is a capability claim rather than an agreement claim. On both `buttercup`
families, the decode must place frames at position 6 (the coda `/p/`) and position 3 (the rhotic
`/er/`) on a majority of recordings, and the realised mass at each must be materially above the
emission floor. Report the full distribution per position; the criterion is **"non-degenerate where
the shipped instrument is structurally zero"**, not a pass mark.

*Present reading, job 23109184:* coda `p` 0.77 and rhotic `er` 0.57 median realised mass, against
0.78–0.83 for the positions the shipped instrument does read. Job 23107265 over all 7,994 gives the
coda reached on 78.4 % / 68.9 % of the two families with a median 150 ms of frames.

### A2 — no regression where the shipped instrument is known-good

The six single-position families are the control, and `ddk-cycle-counting.md` establishes why:
`i % 1 == 0` always, so the phase defect cannot bite there and the shipped scan agrees with itself.
Requirement: the per-recording difference `decoded − shipped` is centred on zero on `-pa`, `-ta` and
`-v2-tuh` — the families whose shipped `consumed` is 1.00, meaning every detected unit was
consumed by a complete cycle. **A systematic offset is a defect to explain, not a tolerance to
widen.** Present reading, as median decoded minus median shipped rather than the median per-recording
difference the criterion asks for: +1, +1, +4. The `-v2-tuh` offset is unexplained and is the single
clearest thing to resolve before landing.

### A3 — inter-repetition timing

The quantity the owner named — *"the gaps between repeats"* — and the corpus answers a question
about it that changes what should be reported. **Measured: DDK has essentially no inter-repetition
silence.** Under the topology with an explicit silence state (job 23107265), the gap from the end of
one repetition to the start of the next exceeded one frame on only 3.7 %–37.6 % of repetition pairs
and its median was **0.000 s** in every one of the ten families. The train is continuous; the
"gap between repeats" that exists to be measured is the **period**, start of one repetition to start
of the next.

Requirement: the decode reports a period distribution per family whose median falls **inside
`branch.modulation_band_hz: [1.0, 10.0]`** — the one external check available that is not a corpus
fit, since that band's derivation is published DDK rates and already in `config-derivations.md`. A
family whose median period lands outside it is a defect in the decode, not a finding about the
corpus. Present reading: 1.78–4.17 cycles/s and 3.45–5.99 syllables/s, all inside.

Report beside it, as a covariate and not as a criterion: the coefficient of variation of the periods,
and the least-squares trend in seconds per repetition, which are `dispersion` and `trend`
unchanged (`ddk.py:326, 345`) applied to the new interval series.

### A4 — coverage is monotone

The shipped instrument reports nothing on a known set of recordings where the task was performed —
105 diagnosed in `branch-ddk-ppg-instrument.md`, of which 74 had CV units whose *timing* failed the
regularity test. Requirement: the set of recordings on which the decode reports a repetition count
**contains** the set on which the shipped instrument reports cycles, and the added recordings are the
ones the shipped instrument left silent rather than a different set. Like the `task_extent` union in
`ddk-task-extent-precedence.md`, this can only add coverage, so a regression can only ever be an
over-report — which is the failure the design can tolerate and the one A5 exists to bound.

### A5 — the listening check that does not exist, named so that it is not pretended away

**Nothing above distinguishes a recovered repetition from a spurious one.** A2's control bounds it on
the families where the shipped instrument is trusted; A3's band bounds it physiologically; neither
is ground truth. What would settle it is hand-counted repetitions on a modest sample — 30–50
recordings stratified across the decoded-minus-shipped difference, weighted toward the tails — and
that sample does not exist. Until it does, **A1–A4 are the whole of the acceptance test and the
design should not claim more from them than agreement-where-trusted plus capability-where-blind.**

### Explicitly not in the acceptance test

- Any comparison of a decoded count against `expected_event_count`, in any direction.
- Any sensitivity or specificity figure. The instrument classifies nothing
  (`branch-ddk-ppg-instrument.md`: the declared task label is ground truth for "is this DDK").
- Any threshold fitted on these ten families and then applied to them.

---

## The envelope instrument's future

### The defect, and its mechanism

On the owner's `-pataka` recording the envelope path reported `expected_event_count` **1** against a
declared 30, and an `onset_rate_hz` of **10.1/s** computed over a **0.0986 s** hull — which is one
event's own width, since `hull` of a single event is that event. The recording holds roughly 20
syllables.

The mechanism is not the carrier. `ddk_carrier` found a carrier at 2.237–4.114, 1.88 s, comfortably
past `train_min_s`. It is `events_in_extent` (`branches.py:1973`, the two prominence tests at 2008 and 2018), which admits a maximum only if it
stands `peak_prominence_db` (6.0 dB) **both** over the global floor **and** over
`max(left trough, right trough)`. At DDK rates the inter-syllable trough does not fall 6 dB below the
adjacent peaks, so the walk rejects nearly every syllable and keeps whichever one happens to sit
beside a real pause. **The walk is specified for events separated by silence — coughs, breaths — and
a DDK train is not that.**

### How often, across the corpus

Read off the per-recording rows of the `task_extent` sweep (job 23105513, all 7,994 declared-DDK
recordings), `onsets_n` being what `events_in_span` returned inside the carrier:

| | value |
|---|---|
| declared-DDK recordings | 7,994 |
| no carrier at all | 3,325 (**41.6 %**) |
| carrier found | 4,669 |
| median onsets inside the carrier | **4** (p10 0, p90 11) |
| ≤ 1 onset | **24.8 %** of carrier recordings |
| ≤ 2 onsets | **36.1 %** |
| fewer onsets than the CV walk found units | **84.8 %** of the 4,618 paired recordings |
| median CV units on those same recordings | 13 |

Per family, median onsets against the declared count: `-pa` 4/10, `-ta` 4/10, `-ka` 5/10,
`-pataka` 3/30, `-buttercup` 3/30, `-v2-puhtuhkuh` 2 (≤2 on 63.8 %).

### The verdict: split the instrument, retire one channel, repair the other

The envelope instrument is two independent channels sharing a carrier, and they do not share a fate.

**Retired for the syllable families — the peak-walk onset channel.** `events_in_span` over the
carrier, and everything read off its onsets: `expected_event_count`'s found value,
`syllable_onset_s`, `inter_onset_interval_s`, `interval_dispersion` and its `by_position`,
`onset_rate_hz`, `hull(onsets)` as the envelope's `task_extent`, and `ddk_places` over those onsets
with the whole burst-spectrum place path behind it. The evidence is not marginal: a median of 4
onsets against 10 or 30, one or none on a quarter of the recordings it ran on, and fewer than the
posteriorgram found on 85 %. **It is not repairable by re-deriving `peak_prominence_db`** — a
prominence low enough to resolve DDK syllables is a prominence that resolves amplitude ripple on
sustained phonation, and the key is shared with AIRWAY, where it is doing its job. The function stays
for AIRWAY (`airway.py:359`) untouched; only DDK stops calling it.

**Survives — the modulation-spectrum rate channel.** `train_rate_hz` (`branches.py:2076`) takes the
FFT of the envelope over the carrier and reports the peak inside `modulation_band_hz` when it stands
`rate_prominence_min` over its own band mean. It reads periodicity without segmenting anything, so
the trough-return failure cannot reach it, and on the owner's recording it returned a plausible rate
while the peak walk was returning one onset. It is a **genuinely independent** estimate of repetition
rate — amplitude, not phonetic identity — and two instruments on one quantity is worth keeping
precisely because the posteriorgram is imperfect. It keeps `ddk_syllable_rate_from_envelope_modulation_hz`,
`branch.modulation_band_hz` and `branch.rate_prominence_min`.

One covariate on it must change with the onset channel: `onset_rate_hz` and `support_syllables`
(`ddk.py:1314-1315`) are read off the onsets and go with them.

**Repaired — the envelope's contribution to `task_extent`.** `align_ddk` currently mints the
envelope-side extent as `hull(onsets) or train.extent` (`ddk.py:1299`). With the onset channel gone
it becomes **the carrier span's own extent**, which is what `ddk_carrier` actually found and what the
length guard actually qualified. On the owner's recording that is 2.237–4.114 (1.88 s) instead of
0.0986 s — a 19× improvement in the input to the union rule, achieved by deleting a step rather than
adding one.

**Ambiguity resolved as a side effect.** `CYCLES_OR_SYLLABLES_PER_S` exists because "a sequential
train modulates at the cycle rate as well as the syllable rate; the peak is one of the two and the
harmonic-equality tolerance that would separate them is unmeasured" (`ddk.py:107`). Under this design
the decode measures the period directly, so the envelope's modulation peak can be reported **against**
the decoded cycle rate and the decoded syllable rate and the reader can see which it matched. The
unit stays ambiguous — nothing new is measured — but the ambiguity becomes checkable per recording
instead of permanent.

### The alternative that was considered and is not recommended now

The correlation instrument rejected above as *the* mechanism is the natural third channel: the
autocorrelation of the class-mass series would give a rate with no alignment at all, in the posterior
domain rather than the amplitude domain. It is not recommended **yet**, because the envelope
modulation channel already occupies that role and adding a third rate estimate before the second one
has been read against the first is instrumentation for its own sake.

---

## Migration

### A staged landing that does not ship a parallel instrument

`CLAUDE.md` forbids parallel fields and deprecation shims outright, so the staging cannot be "land
both instruments and choose later". The stages are **measurement campaign, then one replacement**,
which is the shape every prior DDK change already took — the prototype sweeps in
`checks_20260916/figwork/` preceded each of `ddk-syllable-template.md`, `ddk-cycle-counting.md` and
`ddk-task-extent-precedence.md`.

| stage | what happens | where |
|---|---|---|
| 1 | the acceptance test A1–A4 as a throwaway sweep against stored derivatives; the A5 listening sample collected or explicitly declined | ORCD, not the package |
| 2 | the owner reads the per-family table, the buttercup per-position distribution, and the `-v2-tuh` offset, and settles the open questions below | — |
| 3 | one commit: the decode replaces the CV walk, the train finder and the cycle scan; `BRANCH_MEASURES["SPEECH"]` is rewritten; `Expectation.sequence` becomes a phoneme tuple; the burst place path and the envelope onset channel are deleted; the envelope's `task_extent` becomes the carrier extent | package |
| 4 | corpus reprocess. **Every stored DDK reading changes**, so the report/figure fixtures and any cached artefact keyed on the old names are invalidated deliberately, not incidentally | — |

Stage 3 is one commit because splitting it would leave the tree in a state where `cycle_scan` exists
with no caller or `BRANCH_MEASURES` names a key with no writer — and `ddk_test.py:1418-1440` already
pins "every `ppg_` key the table names has a writer", which is the test that would catch it.

### What changes for a reader

| surface | change |
|---|---|
| SPEECH branch report `detail` | twenty of its 27 keys replaced or deleted. `trains_n`, `train_s`, `train_fraction` keep their meanings, and the four non-DDK keys are untouched |
| the span axis | `ppg_train` disappears (see the open question). `task_extent` stays, one per recording, wider on most and never narrower |
| the report's rate rows | two rates instead of five: the decoded syllable rate and the envelope modulation rate. `ppg_cycle_rate_hz`, `ppg_period_s`, `ppg_jitter_over_median`, `onset_rate_hz` go |
| the new rows | one realised-mass number per template position, printed in template order — which is the first time the report says anything about *which part of the sequence* was realised |
| VERDICT | `deviations` loses `syllable_sequence_mismatch` on this family; `unmeasured` loses four branch keys and gains none |
| conformance | same three values, same grounds, narrower `false` |

A reader who currently reads `ppg_cycles` against `ppg_declared_cycles` loses that pair and gains
a repetition count beside a declared count carried as a covariate. **The pair was descriptive and
the replacement is descriptive; neither is folded into any judgement, and this document is where that
is written down.**

---

## Open questions for the owner

1. **The vowel-height partition.** `close`/`mid`/`open`/`rhotic` as above, with `eh` and `oy` in
   `open`? Only `open` and `rhotic` are observable on the ten DDK families, so the rest is a
   declaration made now to avoid making it later under pressure.
2. **`er` as its own class.** Keeping rhoticity out of the height dimension is what preserves
   buttercup's middle-syllable discrimination. Merging it into `open` would give buttercup three
   interchangeable vowel slots. Confirm.
3. **Buttercup's flap.** The `/t/` of "butter" is an intervocalic flap with no ARPAbet-40 symbol; it
   surfaces as `t`, `d` or `r`, and the alveolar class holds the first two. **Should it hold `r`?**
   Measurable on the corpus and not measured here. Note the present reading already gives that
   position 0.68–0.69 realised mass, the lowest consonant position in either buttercup row.
4. **`syllable_sequence_mismatch`.** It loses its only writer and cannot be re-raised without a
   threshold on per-position mass that has no derivation. Delete the vocabulary entry
   (`branches.py:123`) outright per the pre-alpha rule, or keep it with no writer?
5. **`ppg_train` and the regularity segmentation.** The train was a maximal stretch of regular
   onsets, and `ddk-cycle-counting.md` already argued that scoring over it discards the repeats the
   instrument exists to recover. Delete the span role and the five `PPG_RATE` detail keys, or keep a
   regularity segmentation as its own reported span over the decoded periods?
6. **The burst place instrument.** `ddk_places` existed to be the place *decision* the posteriorgram
   was not allowed to be. Under the decode there is no place decision, only mass. Confirm that
   `ddk_places`, `branch.place_centroid_bands_hz` and `branch.place_margin_db` go — they have no
   other consumer in `src/`.
7. **`expected_event_count` → `declared_event_count`, as a covariate rather than the second half of
   a `count`.** The proposal above. It removes the last place in the store where a found number sits
   in a slot labelled "expected"; it changes no row's value.
8. **`DATA_MAP_PATHS`.** Two shipped documents say a campaign may add a place or a nucleus class
   without editing the package, and the config rejects it. Add the replacements to `DATA_MAP_PATHS`?
9. **Which stream.** The posteriorgram is computed on `enhanced` (`preprocess.py:849, 865`). Under
   this design the posteriorgram is the *only* phonetic instrument, so enhancement's effect on it
   matters more than it did. Should a decode over the `plain` stream be measured before landing?

---

## What could not be established here

- **The brief's coda claim.** *"buttercup's coda `/p/` reads as `/t/` in 8 of 10 repetitions"* was not
  reproduced, and cannot come from the shipped instrument, which emits no unit for a coda at all. The
  coda's recoverability is established above without it.
- **The brief's per-recording drop count.** *"~1 stop dropped per repetition, 10 across 29 CV units"*
  was not reproduced; the corpus-level statement it corresponds to — shipped `consumed` at 0.69–0.75
  on the sequence families — is in `ddk-cycle-counting.md` and is consistent with it.
- **Whether the decode's added repetitions are real.** A4 is monotone by construction and A2 bounds
  the single-position families; nothing here bounds the three-position families, where the decode
  adds the most. Only A5 would.
- **`-v2-tuh` +4 and `-v2-puh` +4 against the shipped scan.** Unexplained. The v2 rows are timed
  (`declared_duration_s: 5.0`) rather than counted, so there is no declaration to read it against.
- **Whether `D` should be derived per recording or per corpus.** It is computed from each
  recording's own `seconds_per_frame`, which is 10.00 ms at the median and 10.26 ms at the maximum
  across 7,994 recordings — so `ceil(20 / spf)` is 2 everywhere on this corpus and the question has
  not yet had to be answered.
- **Cost.** The decode is `O(T · arcs)` with at most 21 states, so it is far cheaper than the
  posteriorgram it reads; nothing here measured it in situ inside SPEECH.
