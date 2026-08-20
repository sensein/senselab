# PREPROCESS

## Signature

```
preprocess(audio, preemphasis=True) -> store
```

Every derivative below is **written to the [element store](store.md)** with its provenance. Later nodes
refine what they find there rather than receiving it as an argument.

No `fail`, no `flag`. A derivative that cannot be computed is simply absent from the store, and a
consumer that needs it does not run.

## Conditioning

```
                            +--> plain -------------> squim, level, ASR x2, alignment, silence
audio --> resample 16 kHz --+
                            +--> pre-emphasis ------> envelope --> spans
                                 a = 0.97               spectrograms, gammatone
                                 (switchable)
```

- **Resample to 16 kHz.** Integer decimation from 48 kHz. Guard against overshoot past full scale.
- **Pre-emphasis** `y[n] = x[n] - 0.97·x[n-1]`, switchable, on by default.
- Both signals are retained. Every derivative below names which one it reads.

## Derivatives

| derivative | definition | signal | consumed by |
| --- | --- | --- | --- |
| `energy_envelope` | `\|x + jH{x}\|`, zero-phase 40 Hz Butterworth lowpass, **dBFS** | pre-emph | `spans`; voice branch level and modulation rate |
| `silence` | YAMNet `Silence` per 0.96 s window, 0.48 s hop, threshold 0.5 | plain | the local floor; airway negative evidence |
| `spans` | see below | pre-emph | AIRWAY classifies. **SPEECH derives its own spans from word timings and does not read this** |
| `level` | peak dBFS, RMS dBFS, LUFS | plain | voice branch reference level; clipping |
| `squim` | STOI, PESQ, SI-SDR — objective head only | plain | speech branch, **per span, not per file**; reported, not gated |
| `asr_crisperwhisper` | transcript, word and token edges | plain | speech transcript and spans; airway lexical check; voice lexical exclusion |
| `asr_qwen` | transcript, word timings | plain | speech agreement confidence |
| `alignment` | forced alignment of the agreed transcript | plain | word and phone spans |
| `spectrogram_wb` | 5 ms window, 5 ms hop | pre-emph | onsets, transients, glottal pulses |
| `spectrogram_nb` | 20 ms window, 5 ms hop | pre-emph | harmonics, F0 by spacing, rendering |
| `gammatone` | 40 ERB channels, 80–7800 Hz, 5 ms hop | pre-emph | short-transient detection |

A derivative is admitted when it is written to the store with provenance. It does not need a
declared consumer — see [`store.md`](store.md).

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
| `K` | **18 dB** | per reader; AIRWAY reads at this setting |
| `hangover` | 120 ms | per consumer; must be shorter than the shortest event to be bounded |

Spans are written to the store as elements of kind `span`, carrying `peak_over_floor_db` and no label.
Any node may read them; SPEECH proposes its own spans from word timings and `refine`s these where they
overlap.

Spans carry `peak_over_floor_db` and **no label**. If no peak anywhere reaches `K` above the local
floor, the node reports **`no_contrast`** rather than an empty list.

## Working rate

16 kHz. Every downstream model is 16 kHz native. A narrowband input with a 4 kHz ceiling restricts
what the airway branch can conclude.

## Extensibility

A new derivative arrives with the consumer that reads it, and its parameters are named values.
Derivations live in [`benchmarks/`](benchmarks/).
