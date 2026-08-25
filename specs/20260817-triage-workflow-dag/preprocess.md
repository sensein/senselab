# PREPROCESS

## Signature

```
preprocess(audio, preemphasis=True) -> store
```

Every derivative below is **written to the [element store](store.md)** with its provenance. Later nodes
refine what they find there rather than receiving it as an argument.

No `fail`, no `flag`. A derivative that cannot be computed is simply absent from the store, and a
consumer that needs it does not run.

PREPROCESS runs **every model that answers a whole-file question**. No later node re-runs YAMNet, AST
or HeAR over the file: [`TAXONOMY`](taxonomy.md) and the branches read the window classifications from
here. See [`routing.md`](routing.md) for the pass this node opens.

## Conditioning

```
                                     +--> plain ------> squim, level, ASR x2, alignment, silence,
                                     |                  yamnet/ast/hear windows
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
| `hear_windows` | per-window **set** of confident health-acoustic labels | plain | TAXONOMY; **AIRWAY only** among the branches |
| `spans` | see below | pre-emph | AIRWAY classifies. **SPEECH derives its own spans from word timings and does not read this** |
| `phonation_spans` | sustained-phonation and glide spans, see below | pre-emph | TAXONOMY's voice kind; VOICE measures on them |
| `formant_tracks` | F1–F4 by Burg over each `phonation_spans` extent | pre-emph | TAXONOMY's voice kind; VOICE |
| `level` | peak dBFS, RMS dBFS, LUFS | plain | voice branch reference level. **File-level only** |
| `disruptions_file` | clipped runs, zero-crossing rate | **recording** | SPEECH step 8; VERDICT |
| `squim` | STOI, PESQ, SI-SDR — objective head only | plain | speech branch, **per span, not per file**; reported, not gated |
| `asr_crisperwhisper` | transcript, word and token edges | plain | word entities; airway lexical check; voice lexical exclusion |
| `asr_qwen` | transcript, word timings | plain | word entities; agreement confidence |
| `consensus_transcript` | see below | plain | SPEECH's PII scan; REDACT; TAXONOMY's lexical evidence |
| `alignment` | forced alignment of the consensus transcript | plain | word and phone edges |
| `spectrogram_wb` | 5 ms window, 5 ms hop | pre-emph | onsets, transients, glottal pulses |
| `spectrogram_nb` | 20 ms window, 5 ms hop | pre-emph | harmonics, F0 by spacing, rendering |
| `gammatone` | 40 ERB channels, 80–7800 Hz, 5 ms hop | pre-emph | short-transient detection |

A derivative is admitted when it is written to the store with provenance. It does not need a
declared consumer — see [`store.md`](store.md).

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
span like any other.

**Duration in seconds is the primary feature.** Each span carries `duration_s` beside its track
statistics, and `duration_s` is what [`TAXONOMY`](taxonomy.md) reads to classify the voice kind and
what [`routing.md`](routing.md) gates the VOICE branch on.

`formant_tracks` are written per span: F1–F4 with their bandwidths on the analysis hop, and, for a
glide, the trajectory's direction and extent.

Parameters live under `phonation_spans` in the config.

## `consensus_transcript`

The consensus over the recognizers' word streams, produced by **`fuse_consensus_words` in
`senselab.audio.workflows.audio_analysis.asr`** — the same routine the audio-analysis workflow uses,
called here rather than reimplemented. It carries per-word agreement between the recognizers.

**The consensus transcript is the text every downstream text consumer reads.** SPEECH's PII scan and
[`REDACT`](redact.md) read it and nothing else; the per-recognizer transcripts remain in the store as
the evidence it was fused from.

## Words are bracket-aware

`word` entities are written here, and only here. Before any is written:

- **A bracketed token is not a word.** CrisperWhisper emits non-lexical events in brackets
  (`[COUGH]`, `[BREATH]`, `[UH]`); these are written as `event` entities carrying their extent, never
  as `word` entities, and no word-derived product counts them.
- **An onomatopoeic cough- or breath-like token is normalised into a bracketed non-word** before word
  entities are written — a recognizer rendering a cough as `khh`, `uh-huh-huh` or `ahem` produces an
  `event`, not a `word`. The normalisation vocabulary is the config key
  `words.onomatopoeic_tokens`; the raw token travels with the event so the normalisation is legible.
- **Word entities remain bounded to the decode.** A word exists where a recognizer decoded one;
  nothing here invents, extends or merges words across recognizers beyond what the consensus routine
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
| `windows.ast.default_threshold`, `.label_thresholds`, `.hop_s` | the same for AST, and the hop between its frames; **null** until fitted. `.win_length_s` is **not** open: it ships 10.24 s by the owner's 10 s directive (analyze_audio's window notes do not govern triage). A recording shorter than the window yields one zero-padded window covering the whole file |
| `windows.hear.default_threshold`, `.label_thresholds`, `.hop_s` | the same for HeAR; **null** until fitted on spans HeAR's 2 s input does not have to be padded to fill |
| `phonation_spans.*` continuity criterion and hangover | what opens and closes a sustained-phonation or glide span for voiced, unvoiced and mixed production; **null** until fitted |
| `words.onomatopoeic_tokens` | the token set normalised into bracketed non-words; a vocabulary, owed a corpus it was drawn from |
