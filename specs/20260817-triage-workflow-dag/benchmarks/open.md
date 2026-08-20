# Still unmeasured

What the current design owes. A parameter listed here is in the spec with its interval or its shape
stated and its constant unjustified — not silently guessed.

## Blocking an outcome that the design promises

| item | what it blocks | what would settle it |
| --- | --- | --- |
| **SQUIM thresholds over speech spans** | SPEECH's quality `fail` is unreachable, so "dismiss because of quality" has no path | labelled quality verdicts on speech spans, from more than one recording |
| **The phonation gate's floors** | VOICE's gate carries an interval — periodicity `(0.44, 0.933)`, RMS `(0.0007, 0.0161)` — and no value | labelled voiced/unvoiced verdicts; a wide gap on one file cannot locate the boundary on another |
| **The redaction padding margin** | REDACT pads outward by a margin that must exceed the *worst* alignment edge error, which is unquantified | edge error distribution for `alignment` over many words, not its median |

## Blocking implementation, found by the capability audit

| item | what it blocks | why |
| --- | --- | --- |
| **`PiiSpan` carries no offsets and no time extent**, and its `asr_model` field receives the batch index as a string | **SPEECH step 7 and all of REDACT** | the speaker-scoped rule needs to know *where* a finding is to test it against target spans, and REDACT needs a time extent to redact. Neither is derivable from what the span currently carries. The highest-leverage single change in the codebase for this design |
| **HeAR's module refuses the padded input AIRWAY specifies** | AIRWAY step 1 | the whole-span-in-a-2 s-buffer path works only because a buffer of exactly 32000 samples passes the length check. It needs a named function rather than each caller rediscovering the coincidence |
| **`MossFormer2_SS_16K` RMS-normalises its input to −25 dBFS** | SPEECH steps 5, 7, 8 | absolute level is destroyed on separated streams, so `level` and any dBFS-referenced measurement taken on a separated stream is not comparable with one taken on the recording. Every measurement already records its stream; the consequence needs stating in the spec |
| **`classify_audios` defaults to `top_k=5` for YAMNet** | PREPROCESS `silence`, TAXONOMY | `Silence` can fall outside the top 5 and simply be absent, which reads as a zero score |
| **`classify_audios` applies softmax for AST** | TAXONOMY | AudioSet is multi-label; softmax makes the scores a distribution they are not |
| **`resample_audios` designs its anti-alias filter at the target rate and applies it at the source rate** | PREPROCESS's 16 kHz resample | for 48→16 kHz the explicit pre-filter may be close to inert. **Unverified** — SpeechBrain's sinc interpolation may cover it, and a sweep test would settle it. Not asserted as a defect |

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
