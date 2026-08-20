# Still unmeasured

What the current design owes. A parameter listed here is in the spec with its interval or its shape
stated and its constant unjustified — not silently guessed.

## Blocking an outcome that the design promises

| item | what it blocks | what would settle it |
| --- | --- | --- |
| **SQUIM thresholds over speech spans** | SPEECH's quality `fail` is unreachable, so "dismiss because of quality" has no path | labelled quality verdicts on speech spans, from more than one recording |
| **The phonation gate's floors** | VOICE's gate carries an interval — periodicity `(0.44, 0.933)`, RMS `(0.0007, 0.0161)` — and no value | labelled voiced/unvoiced verdicts; a wide gap on one file cannot locate the boundary on another |
| **The redaction padding margin** | REDACT pads outward by a margin that must exceed the *worst* alignment edge error, which is unquantified | edge error distribution for `alignment` over many words, not its median |

## Decisions with no measurement behind them

| item | current state |
| --- | --- |
| **The word-gap threshold** grouping words into speech spans | unspecified. Any value is a claim about what makes one utterance |
| **What consumes fabrication candidates** | SPEECH detects them and nothing acts on them |
| **`min_families` per kind** | TAXONOMY states the asymmetry (airway 3 families, speech 2) but not the values |

## Measurements named in the specs and not run

| item | why it matters |
| --- | --- |
| **ASR plain vs pre-emphasised** — WER and token-edge displacement | PREPROCESS sends ASR the plain signal by analogy from SQUIM's incoherent shift, not from measurement |
| **`MossFormer2_SS_16K` across SNR** | never run; the two *enhancement* MossFormer variants diverge sharply, so its behaviour cannot be assumed from theirs |
| **Separation's benefit on overlapping speech** | on the reference recording speech and airway never overlap, so the case separation exists for is untested |
| **Span rules inside proposed rather than labelled regions** | every scored peak was located inside a labelled span; unsupervised operation over a whole envelope is the harder problem |
| **A recording with gaps longer than the widest reader window** | invented events cannot be measured on the reference file — its longest verified-empty stretch is 1.80 s against HeAR's 2 s window, so `verified_empty` measures window bleed |
| **Labelled data for the non-lexical kind** | VOICE has none, so nothing there can be scored |

## Retired with the round-based workflow

`design.md` §8 carried M1–M10. M1 (`nontarget_active` semantics), M2 (frame-posterior dip in a short
gap), M3 (frame-level speech/babble separation), M5 (dominant-cluster anchor under a talkative
intruder), M6 (the embedding window), M7 (the shrinkage pseudo-count) and M10 (region-narrowed cache
misses) all concern machinery the current DAG does not contain. They are not outstanding; they are moot.
M4 and M8 survive above as the non-lexical labelled-data gap.
