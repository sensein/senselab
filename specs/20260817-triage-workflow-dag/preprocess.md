# PREPROCESS

## Signature

```
preprocess(audio, preemphasis=True) -> store
```

Every derivative below is **written to the [element store](store.md)** with its provenance. Later nodes
refine what they find there rather than receiving it as an argument.

No `fail`, no `flag`. A derivative that cannot be computed is simply absent from the store, and a
consumer that needs it does not run.

### What an absence records

The verdict's `absent` map is `{derivative: "Class: first line"}`, written by
`common.describe_exception`. The class is what REPORT reads to say *which kind* of failure it was —
`ValueError` for a null config key, `LookupError` for a missing input, anything else a genuine
crash — and the message is what says *which* key or *which* input. Recording the class alone was
attribution a reader could not act on: eleven blocks read null keys and all eleven read `ValueError`.

The message is bounded rather than allowlisted. Only the first line is kept, and it is truncated at
200 characters with an ellipsis. **Residual risk:** an exception raised from inside an ASR or PII
block could in principle interpolate transcript-derived text into its message, and 200 characters of
it would then be recorded and rendered on the page. The alternative considered was an allowlist of
exception types that cannot carry audio-derived text; it was rejected because the set is not
knowable across the third-party stack (any dependency may raise any type), and a wrong allowlist
gives a false assurance where a stated cap gives a known bound. The page and the store already carry
element ids and inherit the store's sensitivity, so neither is a release artifact; the bound is what
keeps an absence record from becoming a second transcript.

PREPROCESS runs every model that answers a whole-file question. No later node re-runs YAMNet or AST;
[`TAXONOMY`](taxonomy.md) reads their window classifications from here. AIRWAY does **not** use these
whole-file HeAR windows to label a candidate: it re-evaluates each eligible span in its own isolated
HeAR input. See [`routing.md`](routing.md) for the pass this node opens.

## Conditioning

```
                                     +--> plain ------> squim, level, ASR x2, consensus, silence,
                                     |                  yamnet/ast/hear windows (taxonomy)
recording (as supplied) --> resample-+
     |                     16 kHz    +--> pre-emphasis --> envelope --> spans
     |                                    a = 0.97          spectrograms, gammatone
     |                                    (switchable)      phonation/glide spans, formants
     |
     +--> clipping, zero-crossing rate                     # original rate, original level
```

- **Resample to 16 kHz.** Integer decimation from 48 kHz. Guard against overshoot past full scale.
- **Pre-emphasis** `y[n] = x[n] - 0.97·x[n-1]`, switchable, on by default.
- **`recording` is retained unconditioned.** Instruments whose evidence is destroyed by peak
  normalisation and resampling read it and no other stream.
- All three signals are retained. Every derivative below names which one it reads.

## Derivatives

| derivative | definition | signal | consumed by |
| --- | --- | --- | --- |
| `energy_envelope` | `\|x + jH{x}\|`, zero-phase 40 Hz Butterworth lowpass, **dBFS** | pre-emph | `spans`; voice branch level and modulation rate |
| `silence` | YAMNet `Silence` per window, threshold from `windows.yamnet` | plain | the local floor; airway negative evidence |
| `yamnet_windows` | per-window **set** of confident AudioSet labels | plain | TAXONOMY; AIRWAY confirm/contest; SPEECH corroboration |
| `ast_windows` | per-window **set** of confident AudioSet labels | plain | TAXONOMY |
| `hear_windows` | per-window **set** of confident health-acoustic labels | plain | TAXONOMY; AIRWAY re-evaluates candidates instead |
| `spans` | see below | pre-emph | AIRWAY classifies. **SPEECH derives its own spans from word timings and does not read this** |
| `phonation_spans` | sustained-phonation and glide spans, see below | pre-emph | TAXONOMY's voice kind; VOICE measures on them |
| `formant_tracks` | F1–F4 by Burg over each `phonation_spans` extent | pre-emph | TAXONOMY's voice kind; VOICE |
| `level` | peak dBFS, RMS dBFS, LUFS | plain | voice branch reference level. **File-level only** |
| `disruptions_file` | clipped runs, zero-crossing rate | **recording** | SPEECH step 8; VERDICT |
| `squim` | STOI, PESQ, SI-SDR — objective head only | plain | speech branch, **per span, not per file**; reported, not gated |
| `asr_crisperwhisper` | an `asr_hypothesis`: transcript and word timings | plain | one source of the consensus; SPEECH's PII scan reads its transcript |
| `asr_qwen` | an `asr_hypothesis`: transcript and word timings | plain | one source of the consensus; SPEECH's PII scan reads its transcript |
| `consensus_transcript` | the consensus text stream: one `word` per aligned column, each source's reading and timing verbatim, a monotone derived extent | plain | SPEECH's PII scan and spans; REDACT; TAXONOMY's lexical evidence; the figure and the report |
| `spectrogram_wb` | 5 ms window, 5 ms hop | pre-emph | onsets, transients, glottal pulses |
| `spectrogram_nb` | 20 ms window, 5 ms hop | pre-emph | harmonics, F0 by spacing, rendering |
| `gammatone` | 40 ERB channels, 80–7800 Hz, 5 ms hop | pre-emph | short-transient detection |

A derivative is admitted when it is written to the store with provenance. It does not need a
declared consumer — see [`store.md`](store.md).

### Consensus timing authority

`consensus_transcript` is the only triage transcript and word-timing authority. Its `word` elements
are the positions of one linear stream: each carries `text`, `bracketed`, `outcome`
(`agreement` / `variant` / `insertion`), `sources`, every source's own `readings` and `timings`
verbatim, `variants`, `agreement`, the spreads and `temporal_uncertainty_s`, and `index` — the only
order a reader may use. The extent is the derived onset and offset from the isotonic median fit
over the stream ([`consensus-asr-redesign.md`](consensus-asr-redesign.md) §3.6). PREPROCESS does
not run forced alignment after the consensus, so no competing word or phone edge product exists.
The per-recognizer `asr_hypothesis` measurements remain the evidence the stream was aligned from,
and SPEECH scans their transcripts beside the consensus text.

## Window classifications — sets, not accumulators

Three classifiers run over the whole file, each on its own grid:

| classifier | window | hop | config |
| --- | --- | --- | --- |
| YAMNet | 0.96 s (its native frame) | 0.48 s | `windows.yamnet` |
| AST | `windows.ast.win_length_s`, default 10.24 s (the 10 s directive; the model's 1024-frame input) | `windows.ast.hop_s` | `windows.ast` |
| HeAR | 2 s (fixed by the graph) | `windows.hear.hop_s` | `windows.hear` |

**A window's product is a set of labels, not a winner.** A label is a member of the window's set iff
its score clears **its own threshold**; a window may therefore carry zero, one or many labels. Each
classifier's thresholds are the config map `windows.<classifier>.label_thresholds`, falling back to
`windows.<classifier>.default_threshold` for a label with no entry of its own.

**Pooling across windows is set-union per label, and the windows are retained.** The file-level
product of a classifier is the union of its window sets, plus, per label, the list of windows whose
set contains it. Nothing counts windows into a score, ranks labels by how often they won, or takes an
argmax over a vocabulary. A consumer that needs a temporal extent reads the retained windows; a
consumer that needs presence reads the union.

Each window is written as an element carrying its extent, its label set, and the score behind each
member.

## Per-span re-evaluation — `span_hear`, `span_yamnet`

Beside the whole-file grids above, HeAR and YAMNet are re-run per span (`_span_hear`,
`_span_yamnet`), raw scores only — no labelling decision is taken here (`labels`/`scores` appear
only when `windows.<classifier>.default_threshold` is configured; `raw_scores` is always written).
`_span_hear` centres a short span in HeAR's fixed 2 s buffer; HeAR accepts nothing else.

**`_span_yamnet` does not do the equivalent for a short span.** A span at least YAMNet's own
0.96 s native frame is classified directly, letting YAMNet place its own windows over it
(`attribution: "native"`). A span shorter than that is never classified directly — it is attributed
from the whole-file `yamnet_scores` windows that overlap it, unpadded audio already computed above,
via the overlap-weighted mean of their raw scores (`attribution: "covering_windows"`,
`covering_windows_n`, `covering_seconds`). A short span with nothing covering it (only possible when
the whole-file YAMNet pass itself did not run) is recorded unmeasured rather than scored. An earlier
version filled a short span to the frame from its own samples, tiled periodically; a controlled
experiment (`benchmarks/span-fill-recovery-2026-09-08.md`) showed that fill manufacturing a false
label far more often than it recovered the true one, and it was removed on that evidence.

## `spans`

```
floor(t)  = rolling 10th percentile of energy_envelope, 3 s window, dBFS
propose   = peaks where envelope(t) - floor(t) >= K, minimum separation 150 ms
onset     = walk back from the peak to peak - 15 dB
offset    = walk forward to peak - 0.7·(peak - floor), closing after `hangover` ms continuously below
discard   = spans shorter than 50 ms
merge     = overlapping spans
```

| parameter | value | scope |
| --- | --- | --- |
| `K` | `spans.k_db` | per reader; AIRWAY reads at its own setting and may adjust it — [`branch-airway.md`](branch-airway.md) |
| `hangover` | 120 ms | per consumer; must be shorter than the shortest event to be bounded |

Spans are written to the store as elements of kind `span`, carrying `peak_over_floor_db` and **no
label**. Any node may read them; SPEECH proposes its own spans from word timings and `refine`s these
where they overlap. If no peak anywhere reaches `K` above the local floor, the node reports
**`no_contrast`** rather than an empty list.

## `phonation_spans` — sustained phonation and glides

Praat, over the pre-emphasised stream, in one pass over the whole file. Tracks are computed **once on
the stream and then sliced**; no criterion is ever renormalised to a fragment's own maximum.

A span is a maximal interval satisfying a continuity criterion over the tracks, closed by the same
hangover discipline as `spans`. Two productions qualify:

| member | what marks it |
| --- | --- |
| **sustained phonation** | a stable F0 / stable formant interval |
| **glide** | a monotone F0 or formant trajectory over the interval |

**Production may be voiced, unvoiced or mixed.** Disordered voices sustain with little or no
periodicity, so the detector may not require a periodicity floor to open a span; each span records
which of `voiced`, `unvoiced`, `mixed` its tracks support, and an unvoiced sustained production is a
span like any other. For the non-periodic formant limb, however, stable F1/F2 estimates also need
bandwidths no wider than `phonation_spans.unvoiced_max_formant_bandwidth_hz`: a stable LPC fit alone
is possible on broadband noise and is not phonation evidence. The key is null until fitted and is an
acoustic screening condition, not a diagnosis or a YAMNet label proxy.

**Duration in seconds is the primary feature.** Each span carries `duration_s` beside its track
statistics, and `duration_s` is what [`TAXONOMY`](taxonomy.md) reads to classify the voice kind and
what [`routing.md`](routing.md) gates the VOICE branch on.

`formant_tracks` are written per span: F1–F4 with their bandwidths on the analysis hop, and, for a
glide, the trajectory's direction and extent.

Timed consensus words add a complementary positive path: their extents bound a segment assessed from
the same periodic-or-resonant tracks, and a segment reaching
`phonation_spans.word_aligned_min_evidence_fraction` is written as `member: word_aligned`. Word text
is not read as phonation evidence, and no missing or rejected word suppresses a sustained/glide span.
VOICE consumes this ordinary phonation-span representation; a word span contained by an existing
sustained/glide span is omitted as redundant.

Parameters live under `phonation_spans` in the config.

## `consensus_transcript`

The consensus text stream over every `asr_hypothesis` measurement in the store, produced by
**`senselab.audio.workflows.triage.consensus.align_sources`**: the hypotheses are aligned as
sequences by `harmonize_transcripts` (cost model in
[`transcript-alignment.md`](transcript-alignment.md)), one `word` is emitted per aligned column in
column order, and time is read once, after alignment, to set each word's derived onset and offset.
Nothing re-sorts the stream by time; fewer than two hypotheses is a `LookupError`, recorded as the
absence of `consensus_transcript`. The design, the owner's rulings and the measurements are in
[`consensus-asr-redesign.md`](consensus-asr-redesign.md) and
[`consensus-asr-rulings.md`](consensus-asr-rulings.md).

**The consensus transcript is the text every downstream text consumer reads.** SPEECH's PII scan
reads it and each recognizer's own transcript; [`REDACT`](redact.md) reads it and nothing else.

## Words are bracket-aware

`word` entities are written here, and only here, one per aligned column:

- **A bracketed token is a bracketed word.** CrisperWhisper emits non-lexical markers in brackets
  (`[COUGH]`, `[BREATH]`, `[UH]`); these are `word` entities with `bracketed: true`, in the stream
  at their position, and no lexical product counts them — TAXONOMY's lexical line, AIRWAY's
  transcribed check, SPEECH's word count and speech spans all read `lexical_words`. A bracketed
  reading that shares a key with a plain one (`[COUGH]` against `cough`) is one column whose surface
  is the bracketed form; a bracketed token no other source produced is an insertion by one source.
- **An onomatopoeic cough- or breath-like token is normalised into a bracketed word** — a recognizer
  rendering a cough as `khh`, `uh-huh-huh` or `ahem` yields `[KHH]`, with the raw token kept in
  `readings`. The normalisation vocabulary is the config key `words.onomatopoeic_tokens`.
- **Word entities remain bounded to the decode.** A word exists where a recognizer decoded one;
  nothing here invents, extends or merges words across recognizers beyond what the alignment
  produces.

## Working rate

16 kHz. Every downstream model is 16 kHz native. A narrowband input with a 4 kHz ceiling restricts
what the airway branch can conclude. `disruptions_file` is the exception and is measured before any
rate or level change.

## Extensibility

A new derivative arrives with the consumer that reads it, and its parameters are named values.
Derivations live in [`benchmarks/`](benchmarks/).

## Open derivations (v2)

| key | what is owed |
| --- | --- |
| `windows.yamnet.default_threshold`, `.label_thresholds` | the score at which a YAMNet label is confident enough to enter a window's set; **null** until fitted |
| `windows.ast.default_threshold`, `.label_thresholds` | the same for AST; **null** until fitted. `.win_length_s` and `.hop_s` are **not** open: `.win_length_s` ships 10.24 s by the owner's 10 s directive (analyze_audio's window notes do not govern triage), and `.hop_s` ships 10.24 s, non-overlapping, because a null hop stopped AST running at all. A recording shorter than the window yields one zero-padded window covering the whole file |
| `windows.hear.default_threshold`, `.label_thresholds` | the same for HeAR; **null** until fitted on spans HeAR's 2 s input does not have to be padded to fill. `.hop_s` is **not** open: it ships 2.0 s, non-overlapping, by the same ruling as AST's, and a fit on unpadded spans lands as an override |
| `phonation_spans.*` continuity criterion and hangover | what opens and closes a sustained-phonation or glide span for voiced, unvoiced and mixed production; `unvoiced_max_formant_bandwidth_hz` is the required resonant-evidence guard for the non-periodic formant limb; **null** until fitted |
| `words.onomatopoeic_tokens` | the token set normalised into bracketed words; a vocabulary, owed a corpus it was drawn from |
