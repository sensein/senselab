# Still unmeasured

What the current design owes. A parameter listed here is in the spec with its interval or its shape
stated and its constant unjustified — not silently guessed.

## Blocking an outcome that the design promises

| item | what it blocks | what would settle it |
| --- | --- | --- |
| **SQUIM thresholds over speech spans** | SPEECH's quality `fail` is unreachable, so "dismiss because of quality" has no path | labelled quality verdicts on speech spans, from more than one recording |
| **How much disruption is too much** | the same `fail`. Disruption *counts* are exact and need no threshold; what has no value is the tolerance — how many clipped runs, or what clipped duration, makes a span unusable | labelled verdicts pairing disruption counts with a human judgement of usability |
| **The phonation gate's floors** | VOICE's gate carries an interval — periodicity `(0.44, 0.933)`, RMS `(0.0007, 0.0161)` — and no value | labelled voiced/unvoiced verdicts; a wide gap on one file cannot locate the boundary on another |
| **The redaction padding margin** | REDACT pads outward by a margin that must exceed the *worst* alignment edge error, which is unquantified | edge error distribution for `alignment` over many words, not its median |

## Blocking implementation, found by the capability audit

| item | what it blocks | why |
| --- | --- | --- |
| **A PII finding cannot say where it is** | **SPEECH step 7 and all of REDACT** | the speaker-scoped rule needs to know where a finding sits to test it against target spans, and REDACT needs an extent to redact. The fix is to carry the `ScriptLine` scanned rather than to add offset fields: `ScriptLine` already holds `start`, `end`, `speaker` and `timestamp_model`, and a bespoke `start_s` float would both duplicate `start` and discard which producer timed it. Planned as Task 1 of `plan-foundation.md` |
| **HeAR's module refuses the padded input AIRWAY specifies** | AIRWAY step 1 | the whole-span-in-a-2 s-buffer path works only because a buffer of exactly 32000 samples passes the length check. It needs a named function rather than each caller rediscovering the coincidence |
| **`MossFormer2_SS_16K` RMS-normalises its input to −25 dBFS** | SPEECH steps 5, 7, 8 | absolute level is destroyed on separated streams, so `level` and any dBFS-referenced measurement taken on a separated stream is not comparable with one taken on the recording. Every measurement already records its stream; the consequence needs stating in the spec |
| **`classify_audios` defaults to `top_k=5` for YAMNet** | PREPROCESS `silence`, TAXONOMY | `Silence` can fall outside the top 5 and simply be absent, which reads as a zero score |
| **`classify_audios` applies softmax for AST** | TAXONOMY | AudioSet is multi-label; softmax makes the scores a distribution they are not |
| **`resample_audios` designs its anti-alias filter at the target rate and applies it at the source rate** | PREPROCESS's 16 kHz resample | for 48→16 kHz the explicit pre-filter may be close to inert. **Unverified** — SpeechBrain's sinc interpolation may cover it, and a sweep test would settle it. Not asserted as a defect |

## Decisions with no measurement behind them

| item | current state |
| --- | --- |
| **The word-gap threshold** grouping words into speech spans | unspecified. Any value is a claim about what makes one utterance |
| **The F0 search range** — `phonation.f0_min_hz` and `f0_max_hz` | no single range serves both populations: wide enough for a low adult male fundamental admits period-doubling artefacts, narrow enough to exclude them cuts off infant and high-F0 voices. The caller must state which population it is measuring, and a run whose F0 sits where the range is ambiguous is flagged rather than resolved |
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
| **Separated-stream quality routing** | branch-speech.md §8 wants a known target's quality measured on that speaker's separated stream; no source-to-speaker assignment mechanism is specified, so SPEECH measures every span on `plain` and records `stream: plain` honestly. Needs a specified assignment (e.g. embedding match per stream) before the routing can exist |
| **Duplicate PII text within one span** | the word-labelling locator maps a repeated finding text to its first occurrence only; the second occurrence's words carry no `pii` label — an under-redaction hazard for REDACT. The token-subsequence scheme (N11) does not define the repeat case |
| **N7's record-similarities-without-deciding reading** | the shipped node refuses at entry when a provenanced target arrives while `speech.target_match_cosine` is null, so no similarities are recorded; N7's prose implies they would be. One of the two must move |
| **PREPROCESS's recognizer set is declared only per-block** | REDACT derives the expected verification set from the asr activities, which are written just before each transcribe call; a recognizer dying earlier (HFModel construction, offline commit resolution) leaves no activity and silently drops out of the expected set. The durable fix is PREPROCESS declaring its recognizer set unconditionally in the condition activity before any block runs |
| **Fusion coupling under the post-fusion bound** | out-of-bounds hallucinated words participate in `fuse_word_streams` before being dropped; if fusion couples one word's presence to another's confidence, survivors' confidences differ from a never-there baseline. Unmeasured; bounding the input streams pre-fusion would remove the question at the cost of a per-stream count |
| **`_can_align_segment` guards `t1` and never `t2`** (upstream, `forced_alignment.py:138-144`) | an over-long transcript end passes the guard and raises `ValueError` from `extract_segments` three frames deeper instead of the `False` the guard exists to return — every caller of forced alignment inherits this, not just triage |
| **The PII planner and its verifier do not scan the same thing** | SPEECH's planner scans per-span `ScriptLine`s built from CONSENSUS words, while REDACT's verifier re-transcribes the redacted audio whole-file per recognizer and scans fresh hypotheses. A name no recognizer pair agreed on is invisible to the planner and visible to the verifier, which is a permanently unreleasable state with no path to release — observed on DDK-KA: 0 planned redactions, verifier found NAME/PERSON. The verifier being the stricter of the two is defence in depth and should stay; what is missing is a designed path out, e.g. the planner falling back to single-recognizer hypotheses when consensus is empty or sparse |
| **PII findings on non-lexical audio are not reproducible** | re-transcription of the same wordless or DDK audio hallucinated different text across runs and across hosts, so a PII finding on such a file is not stable run-to-run — seen cross-host, and it is the mechanism behind the DDK-KA planner/verifier split above. Any release policy for wordless recordings has to be stated in terms of hallucination instability, not of a single scan's findings |

## Retired with the round-based workflow

`design.md` §8 carried M1–M10. M1 (`nontarget_active` semantics), M2 (frame-posterior dip in a short
gap), M3 (frame-level speech/babble separation), M5 (dominant-cluster anchor under a talkative
intruder), M6 (the embedding window), M7 (the shrinkage pseudo-count) and M10 (region-narrowed cache
misses) all concern machinery the current DAG does not contain. They are not outstanding; they are moot.
M4 and M8 survive above as the non-lexical labelled-data gap.
