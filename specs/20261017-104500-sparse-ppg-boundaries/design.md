# Sparse PPGs and probabilistic phone boundaries

Goal: phonetic posteriorgrams (PPGs) that are cheap enough to run over ~27 h of conversational speech
on an M1 Pro and small enough to keep, plus phone onsets and offsets that carry their own uncertainty.
These support measuring how spoken segments are coordinated: phone rate, segment durations, gaps at
turn transitions, and checking fillers.

Code: `src/senselab/audio/tasks/features_extraction/{ppg.py, sparse_ppg.py, ppg_events.py}`.
Benchmark: `scripts/benchmark_ppgs.py` (`--matrix` re-runs the whole throughput table serially,
one process per row, threads capped, load average recorded).

## 1. Efficiency audit of `extract_ppgs_from_audios`

### What the old worker did

It called `ppgs.from_audio(audio, 16000, gpu=…)` once on the whole recording (ppgs 0.0.9, `mel`
representation, `mel-800k.pt` checkpoint).

* **The whole-file STFT is held in memory.** `ppgs.preprocess.spectrogram` builds a complex STFT
  (1024-point FFT, 160-sample hop) of the whole file before converting it to mel, so peak memory grows
  linearly with length (table below). That is about 0.085 GB per minute, or ~8–10 GB for a 90-minute
  file before senselab's own copy of the waveform is counted.
* **The model chunks, but one chunk at a time.** `ppgs.model.Transformer.forward` cuts the features
  into 500-frame (5 s) chunks with 50 frames of overlap on each side (a 400-frame stride) and runs
  them one after another with batch size 1.
* **Inference ran under `torch.autocast('cpu')`, i.e. bfloat16.** `torchutil.inference.context`
  turns on autocast on whatever device the model is on. On the M1 CPU, bfloat16 matmuls are about
  **10× slower** than float32 and also less accurate (table below).
* **MPS was never used.** The worker only passed `gpu` for CUDA, and `compatible_devices` excluded MPS.

### What the new worker does

* Loads the model once (`ppgs.load.model(representation="mel")`) and runs it **in float32 with no
  autocast**, on CPU, MPS or CUDA.
* Computes features with ppgs' own `ppgs.preprocess.from_audio`, so the features are bit-identical
  to what ppgs feeds its model.
* **Windowing aligned to ppgs' chunk grid.** The worker reads the WAV window by window through
  `portable_audio_io.read_audio(start, stop)`, by default 60 s windows, rounded up to a multiple of
  the 4 s stride. Each window gets exactly one stride (4 s) of context on each side, and the context
  is cropped afterwards. Because every window starts on the global chunk grid, its internal chunks
  are the same chunks the whole-file run would make. Two things differ, and the cropped context
  absorbs both: the replicate-padded first chunk, and the 3-frame reflect pad of the mel edge.
  **The windowed posteriorgram therefore equals the one-call posteriorgram**: max |Δ| = 8.3e-5 and
  100 % argmax agreement on 30 min.
* **Chunks are batched.** The worker re-implements `Transformer.forward`'s chunk split and stacks the
  full-length chunks into one forward pass (`CHUNK_BATCH = 16`). The short last chunk runs on its own
  so that its edge convolution sees zero padding, as it does in ppgs. Logits are **bit-identical**
  to `model(features, lengths)`, and this gives about 1.5× on CPU and 2× on MPS for the model step.

The chunk grid alignment matters because the ppgs output depends on where its chunks fall. On the
21.5 s conversation file:

| comparison (CPU fp32) | argmax agreement | max abs Δ |
|---|---|---|
| ppgs chunked vs `legacy_mode` (unchunked, 2148 positions) | 0.630 | 0.999 |
| windows 8 s / ctx 2 s, off-grid, vs chunked | 0.933 | 0.901 |
| windows 10 s / ctx 4 s, off-grid, vs chunked | 0.958 | 0.923 |
| windows 10 s / ctx 2 s, on-grid by accident, vs chunked | 0.991 | 0.622 |
| windows on-grid with 4 s context (shipped) vs chunked | 1.000 | 8e-5 |

Unchunked inference is out of distribution, since the model was not trained on positions past
500 frames. The chunked output is therefore the reference. Naive overlap-and-crop windowing would
have changed 4–7 % of argmax frames.

### Numerical precision (5 min, reference = CPU float32)

| device / precision | model time | max abs Δ | mean abs Δ | argmax agreement |
|---|---|---|---|---|
| CPU fp32 | 0.91 s | 0 | 0 | 1.000 |
| CPU bf16 autocast (old worker) | 8.99 s | 0.342 | 3.2e-4 | 0.9938 |
| MPS fp32 | 0.61 s | 7.8e-5 | 8.6e-8 | 1.000 |
| MPS fp16 autocast | 0.47 s | 0.057 | 6.2e-5 | 0.9988 |

MPS fp32 is shipped. Through the API on 30 min, MPS vs CPU gives max |Δ| 4.1e-4 and argmax
agreement 1.000.

### Throughput and memory (M1 Pro, 10 cores, 32 GB)

**These timings are upper bounds.** They were measured while other agents were benchmarking on the
same host, with a 1-minute load average of 14–20 on 10 cores. Threads were capped at 3
(`OMP_NUM_THREADS`, `torch.set_num_threads`). Wall time is the whole `extract_ppgs_from_audios`
call: WAV write, worker start-up (~2 s import), model load, inference and `.npy` read-back. The input
is `tutorial_audio_files/english_conversation_higgs_audio_v2.wav` resampled to 16 kHz and tiled to
the given length.

| audio | device | window | wall s | RTF | worker peak RSS | load 1-min |
|---|---|---|---|---|---|---|
| 5 min | CPU | 60 s | 11.9 | 0.040 | 1.02 GB | 15.5–18.8 |
| 5 min | MPS | 60 s | 14.9 | 0.050 | 0.48 GB | 18.6–19.3 |
| 30 min | CPU | 60 s | 30.7 | 0.017 | 1.08 GB | 19.2–16.3 |
| 30 min | MPS | 60 s | 16.8 | 0.0094 | 0.84 GB | 15.6–15.0 |
| 30 min | CPU | whole file | 33.5 | 0.019 | 3.49 GB | 14.3–14.6 |
| 90 min | CPU | 60 s | 83.3 | 0.015 | 1.14 GB | 14.7–20.4 |
| 90 min | MPS | 60 s | 35.7 | 0.0066 | 0.72 GB | 18.7–18.4 |
| *old worker* 5 min | CPU bf16 | whole | 20.1 | 0.067 | 0.91 GB | ~9–16, no cap |
| *old worker* 30 min | CPU bf16 | whole | 119.6 | 0.066 | 3.04 GB | ~9–16, no cap |

Phase split, 30 min, windowed, under load: import 2 s, model load 0.2 s, features (STFT and mel)
7–8 s, model 17–18 s before chunk batching. On CPU, features now take a comparable share to the
model. ppgs computes mel under CPU autocast, and that is left as is so the features stay ppgs'
own.

**Recommendation for 27 h on an M1 Pro:** `device=DeviceType.MPS`, `window_s=60` (the default),
`mel` representation. Under contention the RTF is 0.0066–0.0094, i.e. **≈ 11–15 min of compute for
27 h**, with the worker under 1 GB. CPU float32 is ≈ 25–28 min. The old worker would have needed
≈ 1.8 h and 8–10 GB per 90-minute file. A serial re-run on an idle host
(`scripts/benchmark_ppgs.py <wav> --matrix`) replaces these figures.

### Representation

ppgs 0.0.9 supports `bottleneck`, `w2v2fb`, `w2v2fc`, `mel` and `encodec`. It ships checkpoints for
`mel` (`mel-800k.pt`, 82 MB, the package's `BEST_REPRESENTATION` and default) and `w2v2fb`
(`w2v2fb-425k.pt`). `w2v2fb` additionally runs a wav2vec 2.0 front end, which is a further model download
and a far larger forward pass per frame than the 5-layer, 256-channel transformer over mel. ppgs
itself designates `mel` as `BEST_REPRESENTATION`; Churchwell et al. (2024) compare the
representations' accuracy. `w2v2fb` was **not measured here**
because of the disk budget for this change. The worker hard-codes `mel`; adding `w2v2fb` means
passing `representation` through and measuring it.

## 2. Sparse PPG (`sparse_ppg.py`)

`sparsify_ppg(ppg, top_k=3, min_prob=0.05, dtype=float16) -> SparsePPG` keeps, per frame, the
`top_k` most probable phones with posterior ≥ `min_prob`. The argmax is always kept. The result is
stored as COO arrays `frame` (uint32), `phone` (uint8) and `prob` (float16), with `frame_rate`,
`phones` and `num_frames`. `.to_dense(renormalize=False)`, `.save(npz)` (compressed) and `.load()`
are provided. `sparse_ppg_fidelity` reports retained mass, argmax agreement, abs errors and
compression.

Measured on 30 min of real ppgs output (180 000 frames; dense float32 28.8 MB; dense float16
compressed npz 9.3 MB):

| top_k / min_prob | entries/frame | mass kept mean (min) | argmax agree | max abs err | npz size |
|---|---|---|---|---|---|
| 1 / 0 | 1.00 | 0.888 (0.117) | 1.0 | 0.499 | 0.48 MB |
| 2 / 0.05 | 1.34 | 0.957 (0.225) | 0.99996 | 0.322 | 0.65 MB |
| **3 / 0.05 (default)** | **1.46** | **0.970 (0.327)** | **0.99996** | **0.222** | **0.71 MB** |
| 3 / 0.01 | 1.82 | 0.979 (0.327) | 0.99996 | 0.222 | 0.87 MB |
| 5 / 0.01 | 2.08 | 0.988 (0.515) | 0.99996 | 0.115 | 1.00 MB |

The default is **~40× smaller than dense float32 on disk** (13× smaller than compressed float16)
and keeps 97 % of the posterior mass. The 0.004 % argmax disagreement is float16 rounding tying the
top two entries. For 27 h this comes to ~38 MB, against ~1.6 GB dense float32. On the 21.5 s
recording, phone events computed from the sparse form match the dense ones in count and in every
boundary to within 2.4 ms (width within 10 ms).

## 3. Probabilistic phone events (`ppg_events.py`)

**Clock.** Frame *i* covers samples `[160i − 432, 160i + 592)`: a 1024-sample window and ppgs'
432-sample reflect pad. It is centred at **t_i = (i + ½)/100 s**. This is half a frame later than the
frame-edge times of `extract_ppg_segments`.

**Events.** An event is a maximal run of frames sharing an argmax phone. Silence (`<silent>`) is an
event like any other and gets class `silence`.

**Boundary time.** Between run A (phone *a*, last frame *i*) and run B (phone *b*), let
d(t) = p_b(t) − p_a(t). d ≤ 0 at frame *i* and d ≥ 0 at frame *i+1*. The boundary is the linearly
interpolated zero crossing, t* = t_i + (−d_i / (d_{i+1} − d_i))·Δt. When only *a* and *b* carry
mass this is the 0.5 crossing. When other phones compete it is the crossing between the two phones
that actually exchange the argmax. That makes it robust to mass leaking to a third phone, which
would bias a 0.5-crossing rule. On a linear transition the recovery is exact (tested to float32
precision, 1 µs).

**Transition width (uncertainty).** With level L = 0.8, the width runs from when p_a last falls
below L inside A to when p_b first rises above L inside B, both linearly interpolated. A run whose
phone never reaches L is measured at its own peak posterior instead, so a weak phone yields a wide
transition and not a missing one. **σ** converts the width to the standard deviation of the
equivalent logistic. For p(t) = 1/(1+e^{−(t−t0)/s}) the L and 1−L crossings are 2·s·ln(L/(1−L))
apart and its std is s·π/√3, so σ = width·π / (2√3·ln(L/(1−L))) (0.654·width at L = 0.8). Tests
confirm that width and σ grow monotonically with logistic blur and match 2·s·ln 4 and s·π/√3
within 3 % once s ≥ 1 frame. A one-frame step gives a width of 0.6 frames (σ ≈ 4 ms), which is the
quantisation floor.

**Sharpness** is (d_{i+1} − d_i)/2 ∈ [0, 1], where 1 means a complete swap in one frame.

**Per event:** peak and mean posterior of the event's phone, and the mean frame entropy in bits.
Recording-edge boundaries are NaN because no transition is observed there.

**Boundary probability.** b_t = ½·Σ_k |p_k(t) − p_k(t−1)| (total variation, in [0, 1]), located at
t/100 s. It is a continuous segmentation signal, and it integrates to roughly the number of
transitions: on the 21.5 s file Σb = 217.6 against 223 events.

Observed on the 21.5 s conversation: 223 events (214 non-silent, ≈ 10 phones/s). Median interior
duration is 72 ms. Median onset width is 14.6 ms (p90 40 ms) and median σ is 9.5 ms. Median sharpness
is 0.50, and 4.9 % of runs last a single frame. Runs are not merged; filtering short runs is the
caller's decision. 30 min of events (18 803) take 0.3 s to compute.

## 4. Coordination helpers

* `phone_rate_by_speaker(events, [(speaker, start, end), …])` returns, per speaker, the merged span
  time, the silence time inside the spans, the non-silent phone count (assigned by event midpoint),
  `phone_rate` = n/speech_s and `articulation_rate` = n/(speech_s − silence_s).
* `phone_class_duration_stats(events, speaker_spans=None)` returns count, mean, median, std, p10 and
  p90 per manner class (vowel, diphthong, stop, affricate, fricative, nasal, liquid, glide, silence),
  over interior events only.
* `turn_transition_timing(events, [(prev_end, next_start), …], search_s=0.5)` uses the non-silent
  offset nearest `prev_end` and the non-silent onset nearest `next_start`, each within `search_s`.
  It returns gap = onset − offset (negative means overlap) and gap σ = √(σ_off² + σ_on²), assuming
  the two boundaries are independent. Note that a single-channel PPG is speaker-agnostic, so in
  overlapped speech the nearest phone may belong to the other speaker.
* `filler_support(ppg, start, end)` reports the mean per-frame mass on vowel, diphthong and nasal
  phones (`uh` = ah, `um` = ah + m), the vowel, nasal and silence masses separately, stability
  = 1 − mean boundary probability over the span, the dominant phone with its mean posterior, and
  mean entropy. It is meant for verifying ASR-detected fillers. No decision threshold is shipped,
  because a threshold must be fitted from labelled fillers and recorded in `data/` with its derivation.

## Defaults and where they come from

* `top_k = 3`, `min_prob = 0.05` were specified by the requester. The table above measures what they
  keep.
* `level = 0.8` was specified by the requester. σ is normalised by the level, so it is comparable
  across levels.
* `DEFAULT_WINDOW_S = 60` was chosen because worker RSS stays about 1 GB at every length while the
  per-window overhead stays negligible (15 windows per 15 min).
* `CHUNK_BATCH = 16`: on CPU, batch 8–32 were indistinguishable (0.21–0.24 s per minute), and on MPS
  batches of 8 or more ran at 0.06 s per minute.

## Behaviour changes for existing callers

* Posteriors are float32 instead of bfloat16-autocast. Values move by up to 0.34 and about 0.6 % of
  argmax frames change. Cached PPG measurements computed by the old worker are not comparable
  bit-for-bit.
* With `device=None`, Apple Silicon now selects MPS (CUDA → MPS → CPU), whereas before CPU was the
  only non-CUDA option. MPS and CPU agree to 4e-4.

## Open items

* `ppgs.load.model` downloads `CameronChurchwell/ppgs` with `hf_hub_download` and no revision, so the
  checkpoint is ref-addressed, not commit-pinned. This predates this change and should be pinned the
  way CLAUDE.md requires for every model load.
* `w2v2fb` speed and quality are unmeasured (see Representation).
* Absolute throughput needs an idle-host re-run (`--matrix`).

## References

* C. Churchwell, M. Morrison, B. Pardo. *High-Fidelity Neural Phonetic Posteriorgrams.* ICASSP 2024
  Workshop on Self-supervision in Audio, Speech and Beyond (arXiv:2402.17735). Code:
  https://github.com/interactiveaudiolab/ppgs (v0.0.9 used here).
