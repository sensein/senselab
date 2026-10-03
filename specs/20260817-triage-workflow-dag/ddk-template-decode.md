# DDK — the instrument becomes a probabilistic decode of the token's phoneme sequence

What replaces the per-syllable CV walk, the mechanism that replaces it, every parameter it
introduces and where each comes from, what it deletes, what it must not break, and what would have
to be measured before it lands.

**Status.** The design's nine open questions were settled by the owner on 2026-09-19 and their
answers are in [*The owner's decisions*](#the-owners-decisions-2026-09-19) below, which is
authoritative wherever it and the body above it disagree. The body has been brought into line with
it. Implementation follows this document.

The code this describes replacing is `src/senselab/audio/workflows/triage/nodes/ddk.py`; the
documents it supersedes are [`branch-ddk-ppg-instrument.md`](branch-ddk-ppg-instrument.md),
[`ddk-syllable-template.md`](ddk-syllable-template.md) and
[`ddk-cycle-counting.md`](ddk-cycle-counting.md). It supersedes the union rule of
[`ddk-task-extent-precedence.md`](ddk-task-extent-precedence.md) (decision 6) and preserves
[`ddk-instrument-over-asr.md`](ddk-instrument-over-asr.md) intact.

**One decision here has since been reversed.** The envelope fallback for `task_extent` — decision 6
below, and the second row of both tables in this document — is removed by
[`ddk-envelope-mints-no-extent.md`](ddk-envelope-mints-no-extent.md). The envelope mints no
`task_extent`; where the decode reads no repetition, none is proposed. Everything else here stands,
the modulation-rate channel included. The paragraphs arguing for the fallback are left as written so
the reversal is legible.

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

**The template's `/t/` is a flap, and the alveolar class admits `r`.** In "butter" the /t/ is an
intervocalic flap /ɾ/ in American English. ppgs's ARPAbet-40 inventory has no flap symbol, so the
flap must surface as `t`, `d` or `r`. `t` and `d` were already in the class; `r` is the third
spelling of the same segment and the owner has admitted it. The principle and every admission it
licenses are in *The owner's decisions* (3) below: an admission is legitimate when a **stimulus**
phoneme has more than one ARPAbet spelling in the realisation the stimulus prescribes, and it is
not a licence to widen a class toward whatever the posteriorgram happens to read.

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

**And what they do not establish.** `-pa` at 11 and `-v2-puh` at 14 may be over-wrapping; nothing
here separates a real eleventh repetition from a spurious one, and only listening would. The
per-recording agreement with the shipped scan is loose, which is what a systematic recovery of
missed repetitions looks like and is also what an over-count would look like. The acceptance test
below narrows it — the extra repetitions lie inside the shipped instrument's own hull at a rate both
instruments agree on — without closing it.

**These figures are the prototype's, on 400 recordings per family.** The landed instrument's own
readings over all 7,989 are in the acceptance section below and supersede this table wherever the
two differ.

## Counts are heuristics, not targets

The owner, 2026-09-19: *"many extractors, including ppg are imperfect, and counts are inexact. so
expected count is not a target it's a heuristic."*

This is the governing constraint on everything the decode reports, and the shipped graph already
honours it, so the requirement is **not to regress it**:

- A row's declared count reaches the store only through a count finding — since 2026-09-21,
  `typical_count` on these families, which writes `found` beside `typical` and asserts no
  discrepancy.
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

### What `expected_event_count` means — the owner's ruling

> **Superseded 2026-09-21** by `specs/20260921-required-and-typical-counts/design.md`, which builds
> the per-task count-kind property this section calls owed. The field is gone: a row now declares
> `required_count` or `typical_count`, and the debt stated at the end of this section — *a reader
> of a DDK row has no field telling them the declared count is a guide* — is paid.

The design proposed renaming it to `declared_event_count` and carrying it as a covariate rather than
the second half of a `count`. **The owner kept the name**, and gave the reason:

> "there are certain places where this is true: for example 2 or 3 heys, 3/5 breaths etc. keep
> expected but use it based on underlying task."

So the field is one name over **two kinds of count**. For a discrete enumerable event the
instruction names a number for — `loudness`'s 3 "hey"s, `respiration-and-cough-fivebreaths`'s 5
breaths, `threequickbreaths`'s 3 — the declared number is meaningful and "expected" is the honest
word. For DDK's rapid repetition it is a rough guide and the rate is the point. Which kind it is
follows from the underlying task, and **there is no field that says so yet**.

The per-task count-kind property is **owed**. It touches AIRWAY, SPEECH and VOICE and is sequenced
as its own change after this one. What this work does instead is narrower and is the part that must
not be got wrong: the declared count stays on the row, the decoded repetition count is reported
beside it, and **nothing in the decode compares them** — no conformance term, no score, no gate.

The place that still read like a target was
`count("expected_event_count", len(onsets), expectation.expected_event_count, …)` (`ddk.py:1310`),
which put a found count beside a declared one **whose units do not match**: 30 syllables declared
for `-pataka` against a found value that was envelope onsets. That specific pairing goes, not
because the field was renamed but because the envelope onset channel it counted is retired — the
declared count now sits beside the decoded repetition count, which is at least a count of the same
kind of thing, and the `found`/`declared` pair remains descriptive and unfolded.

The debt this accepts, stated plainly: until the count-kind property exists, a reader of a DDK row
has no field telling them the declared count is a guide rather than a target. The graph does not
fold it, but the row does not say so.

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
| `ddk_places` | 274 | **the burst-spectrum place instrument**, removed entirely per decision 7 — both call sites, the agreement covariate at 1202 and the envelope place reading at 1351 | `branch.place_centroid_bands_hz`, `branch.place_margin_db` (deleted; `ddk_places` is their **only** consumer — verified: no other call site in `src/`), `ddk_place_agreement_ppg_vs_burst`, `syllable_place`. `branch.burst_window_ms` survives, re-pointed at `D` |
| `working_rate`, `DdkReads.wideband`/`wideband_id`, `WIDEBAND` | 165, 159 | the wideband spectrogram DDK read only to run `ddk_places` | none outside DDK — `read_spectrogram_block`, `SpectrogramBlock` and `band_power` survive for `spectral_balance_db` and AIRWAY |
| `dispersion_by_position` | 364 | one dispersion per cycle position | `ddk_test.py:66, 923` |
| `events_in_span` **as used by DDK** | `ddk.py:1298` | the envelope peak walk over the carrier | none — `airway.py:359` is the other caller and is untouched |

`cv_covered_extent`, `cv_task_extent`, `merged_task_extent`, `contradicted_words`,
`instrument_authority` and `intervals_of`/`dispersion`/`trend` **survive**, with `cv_covered_extent`'s
input changing from "the hull of every CV unit" to "the hull of the completed repetitions" and its
gate from "a complete cycle or a train" to "a complete repetition".

### Config

| key | fate |
|---|---|
| `branch.ddk_stop_places` | role changed from *argmax label map* to *emission class*, and `alveolar` gains `r` for buttercup's flap (decision 3). **Renamed** — pre-alpha rules out an alias, and `ddk_` no longer describes it. `branch.phoneme_place_classes`, and in `DATA_MAP_PATHS` |
| `branch.ddk_nucleus_classes` | replaced by the total four-class partition above. `branch.phoneme_vowel_classes`, and in `DATA_MAP_PATHS` |
| `branch.ddk_interval_tolerance` | **deleted.** It segmented by regularity; the decode segments by the template. `config-derivations.md` already records that it "separates nothing" |
| `branch.ddk_min_repetitions` | **deleted.** There is no train to have a minimum length |
| `branch.burst_window_ms` | **kept**, and its consumer changes: it stops feeding `ddk_places` and starts deriving `D` |
| `branch.place_centroid_bands_hz`, `branch.place_margin_db` | **deleted** with `ddk_places` (decision 7). `place_centroid_bands_hz` leaves `DATA_MAP_PATHS` with the key |
| `branch.train_min_s`, `branch.modulation_band_hz`, `branch.rate_prominence_min` | **kept** — the envelope modulation channel survives (below) |
| `branch.smoothing_window_s`, `peak_prominence_db`, `trough_return_db`, `event_min_s` | **kept** — AIRWAY reads them |

**A defect found in passing, confirmed and fixed here.** `branch.ddk_stop_places` and
`branch.ddk_nucleus_classes` are described as campaign-overridable data mappings in both
`default.yaml:301-313` and `ddk-syllable-template.md:42` — *"a campaign may add a place without
editing the installed package"* — but neither is in `DATA_MAP_PATHS` (`config.py:45-58`), which
lists only `branch.place_centroid_bands_hz` among the branch place keys. So an override adding a
place was **rejected as a schema violation**, contrary to two documents. Owner's decision 8:
`branch.phoneme_place_classes` and `branch.phoneme_vowel_classes` are in `DATA_MAP_PATHS`, and
`branch.place_centroid_bands_hz` leaves it with the key itself. The claim was untrue for the
originals from the day both documents were written.

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
| `ppg_train` (span role) and `PPG_RATE`'s five detail keys | **deleted** (owner's decision 5). The train was a segmentation by regularity, which `ddk-cycle-counting.md` already argued was the wrong unit to score over, and the decode's own repetition boundaries are the segmentation. `ppg_rate_hz` — syllables per second — survives with a new derivation: `(vowel positions per template) / (median period)`, needing no train at all |

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
  threshold on per-position mass, which has no derivation**, so the vocabulary entry
  (`branches.py:123`) goes with it — owner's decision 4. `write_findings` refuses any deviation not
  in that vocabulary, and the closure guard in `branches_test.py` refuses any entry in it that
  nothing writes, so a writerless declaration is not an available state.

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

**The `task_extent` is a boundary, not a score, and exactly one survives.** The union rule merged
in `fe9167db` is superseded — owner's decision 6 — because the decode produces the extent directly.
What the union protected, and what must not be lost with it, is the *boundary* reading: the extent
runs from the first completed repetition's start to the last one's end **including every intervening
filler frame**, and is never a mask over the frames charged to matched positions only. A DDK train
is continuous (A3 measures the inter-repetition gap at a median 0.000 s in all ten families), so in
practice the filler inside the span is transition frames rather than silence — but the rule is about
what the span *means*, not about what it happens to contain.

The two cases that replace the three:

| | `task_extent` | `production` |
|---|---|---|
| the decode completed at least one repetition | first repetition's start … last repetition's end | `syllable_task_from_decode` |
| it did not, and the envelope found a carrier | **the carrier span's own extent** (see the envelope section) | `syllable_train` / `syllable_sequence` |

The envelope is a fallback rather than a deletion because the decode has no reading at all when the
posteriorgram derivative is absent, and deleting the envelope side outright would take the
`task_extent` away from those recordings — a coverage regression, which is the one failure mode
`ddk-task-extent-precedence.md` rules out. On the recording that motivated that document the
fallback is the carrier at 2.237–4.114 (1.88 s) rather than `hull(onsets)`'s 0.0986 s, so where the
fallback fires at all it fires 19× wider than what shipped before `fe9167db`.

> **Reversed.** The second row is gone;
> [`ddk-envelope-mints-no-extent.md`](ddk-envelope-mints-no-extent.md) holds the ruling and the
> measurement. 49 of 7,994 declared-DDK recordings have no posteriorgram (0.61%) and all 49 carry
> zero `word` entities, so what the fallback protected was not coverage of produced speech.

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

### The run that was scored, and how to reproduce it

Two sweeps over **all 7,994 declared-DDK recordings**, joined per recording by stem, on
`mit_preemptable`: Slurm **23119278** runs the landed decode from this commit's own source tree, and
Slurm **23119279** replays the **shipped** CV walk and cycle scan from the frozen pre-change
snapshot at `checks_20260916/code`, both off the same stored `ppg_posteriorgram` sidecars. Both
printed the resolved `ddk.py` path as their first line, so neither can have read the other's code.
7,989 recordings carry a readable posteriorgram; 49 have no sidecar and are rows with a `status`
rather than silent omissions. Scripts, sbatch files and per-recording rows:
`/orcd/scratch/bcs/002/satra/ddkdecode_20260919/`.

### A1 — sequence realisation where the shipped instrument was structurally blind — **met**

The capability claim. On both buttercup families the decode must place frames at position 6 (the
coda `/p/`) and position 3 (the rhotic `/er/`) on a majority of recordings, with realised mass
materially above the emission floor. The shipped instrument reads both on **0 %** by construction.

`-buttercup`, n = 895; `-v2-buttercup`, n = 701. Every position is reached on **99.2 %** and
**98.4 %** of recordings respectively — the decode completes at least one repetition on all but
those — and the realised mass per position is:

| position | 0 `b` | 1 `ah` | 2 `t` | 3 `er` | 4 `k` | 5 `ah` | 6 `p` |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-buttercup` median | 0.78 | 0.78 | 0.69 | **0.53** | 0.81 | 0.82 | **0.76** |
| `-buttercup` p10 | 0.60 | 0.54 | 0.41 | **0.07** | 0.66 | 0.69 | **0.56** |
| `-v2-buttercup` median | 0.79 | 0.79 | 0.71 | **0.56** | 0.85 | 0.84 | **0.77** |
| `-v2-buttercup` p10 | 0.57 | 0.55 | 0.46 | **0.04** | 0.69 | 0.67 | **0.56** |

The coda is not marginal evidence: at 0.76/0.77 it is the fourth-strongest position of the seven,
above the flap at 0.69/0.71. The rhotic is the weakest position in both families and its p10 of
0.07/0.04 is the one place the distribution reaches the floor — which is the honest reading, since
`er` is the position most often produced as a plain `ah`, and the instrument says so by degrading
continuously rather than by dropping the syllable.

Median occupancy at the coda is **0.780 s** (`-buttercup`) and **0.611 s** (`-v2-buttercup`) summed
across the completed repetitions. **The criterion is non-degeneracy where the shipped instrument is
structurally zero, not a pass mark, and it is met at every position.**

### A2 — no regression where the shipped instrument is known-good — **not met as written, and explained**

The criterion: the per-recording difference `decoded − shipped` is centred on zero on `-pa`, `-ta`
and `-v2-tuh`, the three families whose shipped `consumed` is 1.00. **It is not.**

| control family | n | median | mean | p10 | p25 | p75 | p90 | within one |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `-pa` | 884 | **+2** | +2.62 | 0 | 0 | +4 | +7 | 44.1 % |
| `-ta` | 888 | **+2** | +2.30 | 0 | 0 | +4 | +6 | 45.3 % |
| `-v2-tuh` | 701 | **+3** | +3.91 | 0 | +2 | +6 | +8 | 24.7 % |

Across all ten families the per-recording median offset is +2 (`-pa`, `-ta`, `-ka`,
`-v2-buttercup`), +3 (`-v2-puh`, `-v2-tuh`, `-v2-kuh`) or +4 (`-pataka`, `-v2-puhtuhkuh`,
`-buttercup`). The offset is systematic and positive everywhere, and the design says a systematic
offset is a defect to explain rather than a tolerance to widen. What follows is the explanation.

**Where the extra repetitions are.** They are inside the span the shipped instrument itself
asserts. Taking the **shipped instrument's own CV hull** and dividing by the decoded period gives
how many repetitions fit inside it — a quantity that reads nothing off the decode's extent:

| family | period s | shipped hull s | fit in it | decoded | shipped | decoded ÷ fit | shipped ÷ fit |
|---|---:|---:|---:|---:|---:|---:|---:|
| `-pa` | 0.290 | 3.70 | 11.4 | 11 | 10 | **0.96** | 0.79 |
| `-ta` | 0.275 | 3.41 | 11.2 | 11 | 10 | **0.97** | 0.83 |
| `-ka` | 0.300 | 3.54 | 10.7 | 10 | 8 | **0.92** | 0.70 |
| `-v2-puh` | 0.248 | 3.74 | 15.2 | 14 | 10 | **0.95** | 0.69 |
| `-v2-tuh` | 0.251 | 3.64 | 14.3 | 14 | 10 | **0.99** | 0.75 |
| `-v2-kuh` | 0.270 | 3.72 | 13.5 | 12 | 8 | **0.94** | 0.62 |
| `-pataka` | 0.550 | 6.03 | 10.2 | 10 | 6 | **0.96** | 0.54 |
| `-v2-puhtuhkuh` | 0.516 | 3.95 | 7.4 | 7 | 3 | **0.97** | 0.41 |
| `-buttercup` | 0.540 | 5.79 | 10.1 | 10 | 6 | **0.93** | 0.56 |
| `-v2-buttercup` | 0.546 | 4.00 | 7.1 | 7 | 4 | **0.92** | 0.62 |

The decode reports 0.92–0.99 of what fits inside the shipped instrument's own hull. The shipped scan
reports 0.41–0.83 of it. **The decode is not finding repetitions somewhere the shipped instrument
did not look; the shipped instrument is reporting fewer repetitions than fit inside the span it
itself asserts.**

**And the two instruments agree about the rate.** The per-recording ratio of the decoded cycle rate
to the shipped instrument's own cycle rate has p25 = 1.00 on every family and a median of 1.02–1.06
on nine of the ten (`-v2-puhtuhkuh` 1.50). Neither count enters that quantity. Given a shared hull
and a shared rate, the count is arithmetic — and the shipped instrument's own count is only
0.79–0.91 of its own rate over its own hull, against the decode's 0.92–0.99.

**`consumed = 1.00` does not mean what the criterion assumed.** For a one-position template
`consumed` is `cycles / units`, so it measures how many of the units the walk *found* it consumed,
never how many were there. All three control families read 1.00 while missing 17–25 % of what fits
inside their own hull, which is why they looked like a control and are not one.

**What this does and does not establish.** It locates the disagreement in the shipped instrument
rather than in the decode, and it does so without reading the decode's extent. It does **not** prove
the decode's extra repetitions are real: `fit` is computed at the decoded period, so a systematically
short period would inflate `fit` and the decoded count together. The period is checked independently
only by A3's physiological band, which is far too wide to separate an 11 from a 14. **A5 is what
would settle it and A5 does not exist.**

#### The `-v2-tuh` offset, resolved

The design flagged `-v2-tuh` at +4 as unexplained and the clearest thing to resolve before landing.
Measured over all 7,989 recordings rather than the 400/family subset, the per-recording median
offset is **+3**, and the resolution is that **it is not a `-v2-tuh` anomaly at all**:

- `-v2-tuh` and `-ta` are the same template shape over hulls of nearly the same length (3.64 s
  against 3.41 s) inside recordings of the same length (5.09 s and 5.09 s).
- `-v2-tuh` is produced **faster**: period 0.251 s against `-ta`'s 0.275 s, 3.99 syllables/s against
  3.57. So **14.3** repetitions fit inside its hull where 11.2 fit inside `-ta`'s.
- The decode reports 14 and 11, tracking that density. The shipped scan reports **10 on both**,
  because its argmax CV pairing needs a vowel run to win the argmax between two stops and loses more
  of them as the rate rises. Its `shipped ÷ fit` is 0.75 on `-v2-tuh` against 0.83 on `-ta`, which
  is exactly that ordering.
- The same ordering holds for the two v2 single-syllable families the design did not flag:
  `-v2-puh` 0.69 and `-v2-kuh` 0.62, both denser than their v1 counterparts, both at +3.

So the offset ranks with syllable density across all ten families; the shipped instrument is the
side whose count does not track its own hull at its own rate; and **`-v2-tuh` is the densest of the
three nominal control families and therefore shows the largest control offset.** Nothing about
`-v2-tuh` needs a separate explanation, and nothing in the decode is specific to it.

### A3 — inter-repetition timing — **met**

Requirement: the decode reports a period distribution per family whose median falls inside
`branch.modulation_band_hz: [1.0, 10.0]` — the one external check available that is not a corpus
fit, because that band's derivation is published DDK rates.

| family | cycles/s | syllables/s | in band | period CV | trend s per repetition |
|---|---:|---:|:--:|---:|---:|
| `-pa` | 3.45 | 3.45 | yes | 0.325 | +0.001 |
| `-ta` | 3.57 | 3.57 | yes | 0.307 | +0.001 |
| `-ka` | 3.27 | 3.27 | yes | 0.376 | +0.001 |
| `-v2-puh` | 4.00 | 4.00 | yes | 0.316 | +0.001 |
| `-v2-tuh` | 3.99 | 3.99 | yes | 0.285 | +0.001 |
| `-v2-kuh` | 3.70 | 3.70 | yes | 0.399 | +0.002 |
| `-pataka` | 1.82 | 5.45 | yes | 0.104 | +0.002 |
| `-v2-puhtuhkuh` | 1.92 | 5.76 | yes | 0.083 | +0.008 |
| `-buttercup` | 1.85 | 5.55 | yes | 0.101 | +0.006 |
| `-v2-buttercup` | 1.82 | 5.45 | yes | 0.071 | +0.006 |

All ten medians are in band, at 1.82–4.00 cycles/s and 3.27–5.76 syllables/s. The CV and the trend
are reported beside it as covariates and no criterion reads them; the three-position families' CV of
0.07–0.10 against the single-position families' 0.29–0.40 is what a longer repetition unit does to a
coefficient of variation, not a finding about motor control.

### A4 — coverage is monotone — **met, with three exceptions named**

Requirement: the set of recordings on which the decode reports a repetition contains the set on
which the shipped instrument reports cycles.

| | recordings |
|---|---:|
| joined, readable posteriorgram | 7,989 |
| decode reports at least one repetition | **7,859** |
| shipped reports at least one cycle | 7,470 |
| shipped reports a cycle **or** a train | 7,763 |
| shipped cycles **not** in the decode's set | **3** |
| shipped cycle-or-train **not** in the decode's set | 5 |
| the decode adds over shipped cycles | **392** |

Containment holds on 7,467 of 7,470 — 99.96 %. The three exceptions are named rather than rounded
away: all three are shipped readings of **exactly one cycle** off 1 or 3 CV units — `-buttercup`
3 units over a 3.27 s hull, `-v2-tuh` 1 unit over a 0.26 s hull, `-v2-buttercup` 3 units over a
0.46 s hull inside a 1.3 s recording. The decode charges 69–95 % of those recordings to filler and
completes no repetition. These are the weakest reading the shipped instrument can produce, and one
cycle off one unit is the case `ddk-cycle-counting.md` already describes as structurally
unfalsifiable on a one-position template.

The zero-reading rate is where the coverage gain shows:

| family | decode reads 0 | shipped reads 0 |
|---|---:|---:|
| `-pa` | 0.7 % | 3.2 % |
| `-ta` | 0.5 % | 1.4 % |
| `-ka` | 0.7 % | 4.2 % |
| `-v2-puh` | 1.7 % | 3.6 % |
| `-v2-tuh` | 1.4 % | 2.1 % |
| `-v2-kuh` | 1.7 % | 4.6 % |
| `-pataka` | 0.4 % | 5.0 % |
| `-v2-puhtuhkuh` | 2.0 % | 15.9 % |
| `-buttercup` | 0.8 % | 8.6 % |
| `-v2-buttercup` | 1.6 % | 13.3 % |

### A5 — the listening check that does not exist, named so that it is not pretended away

**Nothing in A1–A4 distinguishes a recovered repetition from a spurious one.** A1 is a capability
claim about positions the shipped instrument cannot reach and says nothing about any count. A2
locates the count disagreement in the shipped instrument, but computes what fits from the decode's
own period, so a systematically short period would inflate both sides together. A3 bounds the rate
physiologically against a published band far too wide to separate an 11 from a 14. A4 is monotone
by construction and can only ever fail as an over-report.

What would settle it is hand-counted repetitions on a modest sample — 30–50 recordings stratified
across the decoded-minus-shipped difference, weighted toward the tails — and **that sample does not
exist and was not collected.** A1–A4 are the whole of the acceptance test; they establish
capability-where-blind, agreement-about-rate, timing-in-band and coverage-that-only-grows, and they
must not be presented as standing in for A5.

### Explicitly not in the acceptance test

- Any comparison of a decoded count against the row's declared count, in any direction.
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
| 2 | the owner reads the per-family table, the buttercup per-position distribution, and the `-v2-tuh` offset, and settles the nine questions — **done, 2026-09-19; see *The owner's decisions*** | — |
| 3 | one commit: the decode replaces the CV walk, the train finder and the cycle scan; `BRANCH_MEASURES["SPEECH"]` is rewritten; `Expectation.sequence` becomes a phoneme tuple; the burst place path and the envelope onset channel are deleted; the `task_extent` becomes the decoded repetition span, and the carrier extent stops being able to be one (`ddk-envelope-mints-no-extent.md`; the fallback this row originally described was removed before the reprocess) | package |
| 4 | corpus reprocess. **Every stored DDK reading changes**, so the report/figure fixtures and any cached artefact keyed on the old names are invalidated deliberately, not incidentally | — |

Stage 3 is one commit because splitting it would leave the tree in a state where `cycle_scan` exists
with no caller or `BRANCH_MEASURES` names a key with no writer — and `ddk_test.py:1418-1440` already
pins "every `ppg_` key the table names has a writer", which is the test that would catch it.

### What changes for a reader

| surface | change |
|---|---|
| SPEECH branch report `detail` | twenty of its 27 keys replaced or deleted. `trains_n`, `train_s`, `train_fraction` keep their meanings, and the four non-DDK keys are untouched |
| the span axis | `ppg_train` disappears. `task_extent` stays, one per recording, and is now the decoded repetition span — so the DDK span axis carries exactly one span |
| the report's rate rows | two rates instead of five: the decoded syllable rate and the envelope modulation rate. `ppg_cycle_rate_hz`, `ppg_period_s`, `ppg_jitter_over_median`, `onset_rate_hz` go |
| the new rows | one realised-mass number per template position, printed in template order — which is the first time the report says anything about *which part of the sequence* was realised |
| VERDICT | `deviations` loses `syllable_sequence_mismatch` on this family; `unmeasured` loses four branch keys and gains none |
| conformance | same three values, same grounds, narrower `false` |

A reader who currently reads `ppg_cycles` against `ppg_declared_cycles` loses that pair and gains
a repetition count beside a declared count carried as a covariate. **The pair was descriptive and
the replacement is descriptive; neither is folded into any judgement, and this document is where that
is written down.**

---

## The owner's decisions, 2026-09-19

The nine questions below the design posed are settled. Each is recorded with what it decides and
what the decision costs, because several of them are choices rather than consequences and the cost
is what a later reader will want.

1. **The vowel-height partition stands as designed** — `close`/`mid`/`open`/`rhotic`, with `eh` and
   `oy` in `open`. No further investigation. The partition is total, so no phoneme falls outside it,
   and on the ten DDK families the membership is byte-identical to the shipped `low`/`rhotic` pair.

2. **`er` stays its own class.** Confirmed as designed: rhoticity is a second dimension and merging
   it into `open` would give buttercup three interchangeable vowel slots.

3. **Allophones are accepted, as a principle and not as one exception.** Where the stimulus's own
   phonology says a template phoneme is realised as a segment ppgs spells differently, the class
   admits that spelling. The admissions made, each with the stimulus fact behind it:

   | class | admitted | the stimulus fact |
   |---|---|---|
   | `alveolar` | `r` | "butter"'s `/t/` is an intervocalic flap /ɾ/. ARPAbet-40 has no flap symbol, so the flap surfaces as `t`, `d` or `r`; `t` and `d` were already in the class and `r` is the third spelling of the same segment |
   | `labial` | — | `p` and `b` are the two spellings of the labial stop and both are already in |
   | `velar` | — | `k` and `g` likewise |

   The principle: an admission is legitimate when a *stimulus* phoneme has more than one ARPAbet
   spelling in the realisation the stimulus prescribes. It is not a licence to widen a class toward
   whatever the posteriorgram happens to read. Nothing but buttercup's flap qualifies on these ten
   families: every other template phoneme is a canonical stop or a vowel with one spelling.

   The cost is stated rather than hidden. `r` in `alveolar` means a genuine `/r/` — which the ten
   DDK stimuli never ask for — would score as an alveolar position. On this family set that cannot
   fire; on any other stimulus it could, and the admission would have to be re-read against it.

4. **`syllable_sequence_mismatch` is deleted outright**, vocabulary entry and all, not kept
   writerless. Its only writer is the envelope place reading that goes with the burst instrument.
   The deviation guard in `branches_test.py`
   (`TestTheDeviationVocabularyIsClosed::test_every_declared_deviation_is_emitted`) sweeps the
   triage tree by AST for `deviation(...)` names and asserts every declared type has an emitter, so
   an orphaned declaration fails the suite — the guard forces the deletion rather than merely
   permitting it.

5. **`ppg_train` is replaced by a span over the decoded repetitions**, and the regularity
   segmentation goes with it. There is no second segmentation to report: the decode's own
   repetition boundaries are the segmentation.

6. **That repetition span is the DDK `task_extent`, and it is the only one.** This supersedes the
   union rule merged in `fe9167db` — which was the right fix for the instrument as it stood and
   becomes unnecessary once the decode produces the extent directly. Together with (5) the DDK span
   axis reduces to **exactly one span per recording**.

   The property the union protected is preserved and is the load-bearing part: **the extent is a
   boundary, not a score.** It runs from the first completed repetition's start to the last one's
   end **including every intervening filler frame**, and is never a mask over the frames charged to
   matched positions only. `ddk-cycle-counting.md`'s reading of the span — *"between these times,
   the subject produced consonant-vowel syllables in response to a syllable-repetition
   instruction"*, asserting nothing about whether every instant inside it holds speech — is
   unchanged.

   The envelope side is not deleted, it is **demoted to a fallback**, and the reason is coverage:
   the decode has no reading at all when the posteriorgram derivative is absent, and dropping the
   envelope outright would remove the `task_extent` from those recordings. So the rule is
   precedence, in two cases:

   | | `task_extent` | `production` |
   |---|---|---|
   | the decode completed at least one repetition | first repetition's start … last repetition's end | `syllable_task_from_decode` |
   | it did not, and the envelope found a carrier | the carrier span's own extent | `syllable_train` / `syllable_sequence` |

   > **Reversed by the owner, same day.** *"if there is no posteriorgram, there is likely no speech.
   > i don't think any DDKs have no posteriorgram."* The second row is removed and the first is the
   > whole rule; see [`ddk-envelope-mints-no-extent.md`](ddk-envelope-mints-no-extent.md). The
   > modulation-rate channel this decision kept is unaffected and still reported.

   What is given up against the merged rule: on a recording where both read the task and the
   envelope carrier reaches past the decoded repetitions, the extent no longer stretches to include
   that reach. That is the owner's call — the decode reads the phonetic sequence and the carrier
   reads amplitude, and where the decode has a reading it is the one that knows where the task was.

7. **The burst place instrument is removed entirely.** `ddk_places`, its two operating points
   `branch.place_centroid_bands_hz` and `branch.place_margin_db`, the
   `ddk_place_agreement_ppg_vs_burst` covariate and the `syllable_place` placeholder all go, along
   with both call sites — the agreement covariate and the envelope place reading. The owner's
   argument: *"if it's only ddk, then the counting/matching is sufficient. remove the burst
   instrument."* The measurement argument beside it is that the burst spectrum degrades in noisier
   recordings, so an agreement covariate built on it is least interpretable exactly where a check
   would matter. `branch.burst_window_ms` is **kept**, with its consumer changed from `ddk_places`
   to the derivation of `D`.

   DDK stops reading the wideband spectrogram altogether as a consequence. `band_power`,
   `read_spectrogram_block` and `SpectrogramBlock` all survive for `spectral_balance_db` and for
   AIRWAY.

8. **The replacement class mappings are added to `DATA_MAP_PATHS`.** `branch.phoneme_place_classes`
   and `branch.phoneme_vowel_classes` join it; `branch.place_centroid_bands_hz` leaves it with the
   key itself. The defect the design found in passing is real and confirmed: two shipped documents
   describe these mappings as campaign-overridable and the schema rejects such an override today.

9. **`expected_event_count` is KEPT, not renamed**, and the owner's refinement changes the design
   rather than merely declining the proposal. *(Superseded 2026-09-21: the count-kind property this
   item sequences separately is built in `specs/20260921-required-and-typical-counts/design.md`,
   and the rename follows from it — a field that carries a kind cannot keep a name asserting one.)*

   > "there are certain places where this is true: for example 2 or 3 heys, 3/5 breaths etc. keep
   > expected but use it based on underlying task."

   For a discrete enumerable event the instruction names a number for — `loudness`'s 3 "hey"s,
   `respiration-and-cough-fivebreaths`'s 5 breaths, `threequickbreaths`'s 3 — the count is
   meaningful and "expected" is the honest word. For DDK's rapid repetition the number is a rough
   guide and the rate is the point. So the field is one name over two kinds of count, and which
   kind it is follows from the task.

   **The per-task count-kind property is owed and is not built here.** It touches AIRWAY, SPEECH and
   VOICE and is sequenced as its own change. What this work does instead:

   - keeps the field and its name;
   - reports the decoded repetition count beside it;
   - and makes sure **nothing in the decode compares the two** — no conformance term, no score, no
     gate — which is the same constraint the *Counts are heuristics, not targets* section above
     already states, now also a scoping boundary.

   Until the count-kind property exists, a reader of a DDK row has no field telling them the
   declared count is a guide rather than a target. That is the debt this decision knowingly
   accepts.

### What was asked and declined

- **A decode over the `plain` stream.** Question 9 of the design asked whether enhancement's effect
  on the posteriorgram should be measured before landing, now that the posteriorgram is the only
  phonetic instrument. **No.** The posteriorgram stays on `enhanced` and no `plain`-stream decode is
  measured or shipped.
- **Renaming `expected_event_count`.** See (9).
- **Any further investigation of `eh` and `oy`.** See (1).

## What could not be established here

- **The brief's coda claim.** *"buttercup's coda `/p/` reads as `/t/` in 8 of 10 repetitions"* was not
  reproduced, and cannot come from the shipped instrument, which emits no unit for a coda at all. The
  coda's recoverability is established above without it.
- **The brief's per-recording drop count.** *"~1 stop dropped per repetition, 10 across 29 CV units"*
  was not reproduced; the corpus-level statement it corresponds to — shipped `consumed` at 0.69–0.75
  on the sequence families — is in `ddk-cycle-counting.md` and is consistent with it.
- **Whether the decode's added repetitions are real.** Still the open question, and the one A5
  would settle. A2's measurement over all 7,989 recordings narrows it usefully — the extra
  repetitions lie inside the shipped instrument's own hull, at a rate both instruments agree on,
  and the shipped count is the one that does not track its own hull — but `fit` is computed at the
  decoded period, so a systematically short period would inflate both sides together. Nothing here
  rules that out.
- **`-v2-tuh` against the shipped scan.** No longer unexplained: see *A2 → The `-v2-tuh` offset,
  resolved*. The per-recording median offset over the full corpus is +3, it is not specific to
  `-v2-tuh`, and it ranks with syllable density across all ten families. What remains open is the
  previous bullet, which is about every family and not about this one.
- **Whether `D` should be derived per recording or per corpus.** It is computed from each
  recording's own `seconds_per_frame`, which is 10.00 ms at the median and 10.26 ms at the maximum
  across 7,994 recordings — so `ceil(20 / spf)` is 2 everywhere on this corpus and the question has
  not yet had to be answered.
- **Cost.** The decode is `O(T · arcs)` with at most 21 states, so it is far cheaper than the
  posteriorgram it reads; nothing here measured it in situ inside SPEECH.
