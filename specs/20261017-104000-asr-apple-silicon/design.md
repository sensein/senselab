# CrisperWhisper + Qwen3-ASR on Apple silicon, and a two-system accuracy estimate

Host: Apple M1 Pro, 10 cores, 32 GB unified memory, macOS (Darwin 25.6), MPS. Other agents were
benchmarking on the same machine throughout, so **every RTF below is a load-contaminated upper
bound**; the 1-minute load average is recorded beside each figure. Relative comparisons (precision,
device, chunking) were run back to back, one model process at a time, 3 torch/OMP threads unless
noted. An idle-machine re-run is one command:
`bash specs/20261017-104000-asr-apple-silicon/bench/run_bench.sh [WORK_DIR]`.

Audio: public AMI meeting corpus, test split, `ihm` (headset) utterance clips of meetings
EN2002a/EN2002b placed back on their timeline to form a headset mix (`bench/build_mix.py`). This is
multi-party (3-4 speakers) conversational English with overlaps; references are AMI's orthographic
transcripts, fillers included. `bench600` = EN2002a 300-900 s; `bench180` = EN2002a 300-480 s.

## 1. How the two backends route today (alpha before this PR)

| | CrisperWhisper (`nyralabs/CrisperWhisper2.0*`) | Qwen3-ASR (`Qwen/Qwen3-ASR*`) |
|---|---|---|
| venv | `crisperwhisper-<torch tag>` | `qwen-asr-<torch tag>` |
| engine | CTranslate2 fork on Linux x86_64 only; **transformers backend everywhere else** (the fork `ctranslate2-crisperwhisper` ships manylinux x86_64 wheels only, and upstream `ctranslate2` lacks the fork APIs, so there is no CT2 and no int8 path on macOS) | `qwen-asr==0.0.6` wrapper over transformers 4.57; companion `Qwen3-ForcedAligner-0.6B` for word times |
| device | CUDA or CPU (MPS not offered) | CUDA or CPU (MPS not offered); weights loaded in the wrapper's default dtype (float32) |
| long audio | library's `continuation` strategy: 30 s windows, 26 s stride, each prompted with the last 12 confirmed words; constant memory per window | wrapper splits at a low-energy point near every 180 s with timestamps (1200 s without) and decodes **all chunks of a file as one batch** (`max_inference_batch_size=32`), `max_new_tokens=512` per chunk |
| word times | cross-attention + Viterbi per window (`[UM]`, `[UH]` get times) | forced aligner per chunk |
| word confidence | **none**: the library's `WordTimestamp` has no probability field, so every `score` was `None` | none |
| fillers | emitted as bracketed words `[UM]`, `[UH]` (also `[laughter]`, `[breath]`); repetitions kept (`I I`); senselab passes them through unchanged. Only `normalize_transcript_for_wer` strips bracketed markers, and only for WER comparisons | writes some fillers as plain words (`um`, `uh`) on clean speech, omits most in conversation (Section 4) |
| multi-file | long-lived worker keeps the model across calls, but the whole list was **one request with one 1800 s timeout** | same |

## 2. Measurements

### 2.1 CrisperWhisper (transformers backend)

| config | audio | RTF | words | peak footprint (GB) | load1 |
|---|---|---|---|---|---|
| CPU fp32, 3 thr | bench180 | 0.84 | 466 | 15.8 (`time -l`), RSS 15.7 | 10.4 |
| MPS fp32, 8 thr | bench600 | 1.06 | 1520 | 19.3 | 12.3 |
| **MPS fp16**, 8 thr | bench600 | **0.29** | 1520 | 10.6 | 9.9 |
| MPS fp16, 3 thr | bench180 | 0.38 | 466 | 10.5 (steady 10.0) | 8.3 |
| MPS fp16 + `empty_cache` per window | bench180 | 0.47 | 466 | 10.7 | 7.2 |
| MPS fp16, senselab path, EN2002a 35.7 min | full file | **0.27** | 4554 | 10-11 throughout (sampled every 10 s) | 7-9 |
| MPS fp16, senselab path, EN2002b 29.8 min | full file | 0.27 | 3632 | 10-11 | 8.8 |
| MPS fp16, senselab path, EN2002a + pink noise 5 dB | full file | 0.35 | 1647 | 10-11 | 8.5 |

- fp16 and fp32 on MPS give **identical transcripts** on bench600 (1520 words, same WER 0.383).
- On MPS, transformers 5.18's deferred stop check (`DeferredStopCheck.finish`) raises
  `'EncoderDecoderCache' object has no attribute 'layers'` for Whisper; the worker disables it on MPS
  (`is_supported -> False`), which restores the ordinary per-step stop check.
- Footprint is flat over a 36-minute file (window-by-window decoding), so a 98-minute interview
  stays at ~10-11 GB. Most of it is the MPS allocator's cache plus ~3 GB of CPU-side arrays; emptying
  the MPS cache per window did not lower the plateau and cost speed, so it is not done.

### 2.2 Qwen3-ASR + forced aligner

| config | audio | RTF | words | WER (fillers excluded) | peak footprint (GB) | load1 |
|---|---|---|---|---|---|---|
| CPU fp32, 180 s chunks (alpha default), 3 thr | bench180 | 2.02 | 410 | 0.517 | 17.2 | 12.6 |
| MPS fp32, 180 s chunks | bench180 | 0.87 | 410 | 0.517 | 39.8 | 41.8 |
| MPS fp16, 180 s chunks, batch 32 | bench600 | 0.52 | 314 | garbage (`!!`) | 20.0 | 14.4 |
| MPS bf16, 180 s chunks, batch 32 | bench600 | 0.74 | 319 | garbage | 20.8 | 14.7 |
| MPS bf16, 180 s chunks, batch 1 | bench600 | 0.16 | 264 | 0.905 | 13.9 | 18.2 |
| MPS fp16, 90 s chunks | bench180 | 0.67 | 1 | 0.999 | 11.4 | 7.9 |
| MPS fp16, 60 s chunks | bench180 | 0.42 | 625 | 0.232 | 11.0 | 8.1 |
| **MPS fp16, 30 s chunks, batch 1** | bench180 | **0.31** | 627 | **0.220** | 13.9 | 9.6 |
| MPS bf16, 30 s chunks, batch 1 | bench180 | 0.36 | 627 | 0.220 | 13.1 | 10.6 |
| MPS fp16, 30 s chunks, batch 1 | bench600 | 0.27 | 1831 | 0.276 | 15.9 | 9.5 |
| MPS fp16, 30 s chunks, **batch 4** | bench600 | 0.90 | 519 | 0.794 | 15.8 | 13.5 |
| MPS fp16, 30 s, `device_map=mps`, `empty_cache`, watermarks 0.4/0.3 | bench180 | 0.36 | 627 | 0.220 | 12.3 (series 8.5-11) | 7.9 |
| same, senselab path, EN2002a 35.7 min, one worker | full file | 0.34 | 5840 | 0.318 | min 9 -> 14, max 16 over 27 min of decoding (rising) | 5-13 |
| same, 20 min of EN2002a in one worker (qwen_bench) | 1200 s | 0.26 | 3353 | 0.308 | per sixth: min 8.6 -> 10.0, max 11 -> 13 (rising) | 2.7 |
| same + each chunk zero-padded to 35 s | 1200 s | 0.27 | 3355 | 0.310 | min 8.4 -> 9.5, max 10 -> 11 (rising more slowly) | 2.7 |
| same, ASR only (no aligner), padded | 1200 s | 0.24 | 3355 | | min 7 -> 8, max 9 -> 10 | 2.5 |
| **shipped: 20-min pieces, fresh worker per piece**, senselab path, EN2002a 35.7 min | full file | **0.25** | 5828 | 0.318 | piece 1: 8.6-12, piece 2: 8.7-12 | 2.6-3.3 |

Findings:

1. **The 180 s default chunk truncates on dense conversation on every device.** 180 s of AMI holds
   ~700 reference words; the wrapper's `max_new_tokens=512` stops after ~410 words. CPU fp32 and MPS
   fp32 agree word for word and both lose 40 % of the words (WER 0.517 vs 0.220 with 30 s chunks).
   Two-person interviews at 150-180 words/min put 450-540 words in a 180 s chunk, so they hit the
   same cap. This PR fixes it on MPS (30 s chunks) and exposes `max_chunk_sec` on every device, but
   **does not change CUDA/CPU defaults** (no CUDA host to measure on); that is an open item.
2. Half precision on MPS decodes long inputs to garbage (`!` runs, 1-word outputs) from 90 s chunks
   up; 30 s and 60 s chunks are correct in both fp16 and bf16, and match each other word for word.
3. Batching chunks on MPS (batch 4, batch 32) also yields garbage; batch 1 is correct and faster.
4. 30 s chunks are both the fastest and the most accurate setting measured.
5. Memory: the MPS allocator grows within each chunk's generate and drops when emptied; loading with
   `device_map="mps"` avoids a float32 copy on CPU (RSS 9.1 -> 5.6 GB), `torch.mps.empty_cache()` after
   each generate/align call and allocator watermarks (high 0.4, low 0.3) cap the per-chunk peak at
   11-12 GB on bench180. **Over a long file the floor still rises** (about 1 GB per 20 min of audio in
   one worker; 9 -> 14 GB over 27 min across three files), most of it in the ASR decode (it remains
   without the aligner) and only partly removed by padding chunks to one length, which points at
   per-shape state that `empty_cache` does not release (for example cached Metal graphs). Starting a
   fresh worker bounds it: audio longer than 20 min is cut at the quietest 100 ms near each 20-min mark
   and each piece goes to a new worker (reload ~5-10 s per piece, under 1 % of a piece's decode). A
   98-minute interview is five pieces, each expected to stay within the measured 8.6-12 GB.

### 2.3 Decisions

- CrisperWhisper: `compatible_devices = [CUDA, MPS, CPU]`; on MPS the transformers backend in float16.
  CUDA (float16, CT2 on Linux x86_64) and CPU (float32) unchanged.
- Qwen3-ASR: `compatible_devices = [CUDA, MPS, CPU]`; on MPS float16 loaded with `device_map="mps"`,
  30 s chunks, batch 1, `empty_cache` after each generate/align, allocator watermarks 0.4/0.3 in the
  worker env, audio over 20 min decoded in pieces by fresh workers. On CUDA and CPU the worker's init
  payload and chunking are unchanged from alpha (only the per-file request split below applies).
- Both: one worker request per audio, timeout `max(1800 s, 3 x duration)`; the worker keeps the model
  across the list, so a batch of interviews loads each model once.
- CrisperWhisper word confidence: in the worker, after each window's final decode, the window's
  prompt + generated tokens are scored teacher-forced against the encoder output the decode already
  produced (a forward hook keeps it), so the encoder is not re-run; word confidence = mean token
  probability over the word's tokens (Whisper's convention), mapped onto the final words by
  recording each window's word list and the continuation strategy's keep index. CT2 backend: `None`
  as before. A count mismatch leaves the scores `None` rather than misassigning them; on the three
  calibration files every word was scored.
- CT2 / int8 on macOS: not possible with released wheels (Section 1), so not measured.
- `CACHE_SCHEMA_VERSION` 24 -> 25 (MPS under `device: auto`, word scores now populated).

## 3. Recommended configuration for ~27 h of two-person Zoom interviews on this Mac

Run the two models **sequentially, not concurrently**: each sits at 10-12 GB footprint, they share
one GPU, and together they would leave little of the 32 GB for anything else. Pass the whole list of
files in one `transcribe_audios` call per model, so each model loads once (Qwen3-ASR reloads once per
20-minute piece by design). Defaults already select MPS; set `OMP_NUM_THREADS=3` if other work shares
the machine.

| model | device / settings | RTF, dense speech (load1) | RTF, 2-speaker mix with gaps (load1) | 27 h of audio |
|---|---|---|---|---|
| CrisperWhisper 2.0 turbo | MPS fp16 (default) | 0.27 (7-9) | 0.17-0.26 (3-4) | ~7.3 h |
| Qwen3-ASR-1.7B + aligner | MPS fp16, 30 s chunks, 20 min pieces (default) | 0.25 (2.6-3.3), 0.30-0.34 (5-8) | 0.17-0.19 (4) | ~6.8-9.2 h |
| both, sequential | | | | **~14-17 h** |

"Dense speech" is the all-speaker AMI mix, closer to an interview's density than the two-speaker
mix, which keeps the other speakers' time as silence. All figures are load-contaminated; the
coordinator's idle-machine pass (`bench/run_bench.sh`) should replace them.

## 4. Fillers

On the synthetic `say` clip ("So, um, I think that, uh, we should, uh, go ...") CrisperWhisper
returns `So [UM] I think that [UH] we should [UH] go to the store [UM] yeah, I mean, I I don't know.`;
every filler is a word chunk with start/end and a confidence, and the repetition `I I` is kept.
Qwen3-ASR wrote `um`/`uh` on the same clip.

On AMI (filler = reference `UM`/`UH`, matched inside the same reference utterance, `filler_recall.py`):

| recording | reference UM/UH | CrisperWhisper: emitted / matched / recall / precision | Qwen3-ASR: emitted / matched / recall / precision |
|---|---|---|---|
| EN2002a, all speakers | 113 | 66 / 51 / 0.45 / 0.77 | 27 / 26 / 0.23 / 0.96 |
| EN2002b, all speakers | 143 | 87 / 69 / 0.48 / 0.79 | 35 / 35 / 0.24 / 1.00 |
| EN2002a, two speakers | 63 | 60 / 44 / 0.70 / 0.73 | 17 / 16 / 0.25 / 0.94 |
| EN2002b, two speakers | 110 | 79 / 61 / 0.55 / 0.77 | 35 / 34 / 0.31 / 0.97 |
| EN2002b, two speakers, pink 10 dB | 110 | 54 / 42 / 0.38 / 0.78 | 9 / 8 / 0.07 / 0.89 |
| EN2002a, all speakers, pink 5 dB | 113 | 48 / 19 / 0.17 / 0.40 | 8 / 2 / 0.02 / 0.25 |

Nothing in senselab drops CrisperWhisper's fillers: they reach `ScriptLine.text` and `chunks` with
times and scores (test `test_fillers_survive_with_timestamps_and_confidences`, which synthesises the
clip with macOS `say` and runs the real model). The losses above are the model's: CrisperWhisper
finds 0.45-0.70 of reference fillers in clean two- and four-speaker audio, about twice Qwen3-ASR's
rate, and both collapse in noise. Fillers in overlapped speech are the ones most often missed. The
accuracy estimate therefore removes fillers from both sides (`exclude_fillers=True`), and its WERs
say nothing about filler accuracy; filler recall needs its own reference-based check on the target
data.

## 5. Two-system accuracy estimate

### 5.1 API

`senselab.audio.tasks.speech_to_text_evaluation.estimate_transcript_accuracy(primary, secondary, *,
segments=None, exclude_fillers=True, calibration_path=None) -> TranscriptAccuracyEstimate` with
`.overall` and `.segments[i]` (`AccuracyEstimate`): word counts, `inter_system_wer` (primary scored
against the secondary as reference), `inter_system_disagreement` (word edits / longer word count),
`mean_confidence`, `min_confidence`, `low_confidence_fraction`, `estimated_wer` and
`estimated_wer_low/high` (nominal 80 %). Both texts are lowercased, punctuation- and bracket-stripped
(`normalize_transcript_for_wer`) and, by default, stripped of `uh um uhm umm er erm ah eh ehm hm hmm
mm mhm`. Words go to spans by midpoint. `speech_to_text_ensemble` (ROVER-style voting) has no WER or
pairwise-agreement primitive to reuse, so the edit distance is computed here.

### 5.2 Calibration data

Public AMI test set, meetings EN2002a/EN2002b (`ihm` headset utterances placed on their timeline):
all speakers (4), the two most talkative speakers only (a stand-in for a two-person conversation),
all speakers + pink noise at 5 dB SNR, two speakers + pink noise at 10 dB: six recordings, 3.5 h.
Each recording transcribed by both models through senselab on MPS (`run_pair.py`), cut into windows
of ~30 s (and, as a check, ~60 s) at points outside every reference utterance; windows with < 20
reference words dropped. True WER = primary vs AMI reference, both normalised as above with fillers
removed. 285 windows (30 s), 18,082 primary words with confidences.

System accuracy on this data (pooled, fillers excluded): CrisperWhisper 0.48 (all speakers), 0.32-0.35
(two speakers), 0.53 / 0.88 (noise); Qwen3-ASR 0.30-0.32, 0.19, 0.49 / 0.86. CrisperWhisper's excess
is deletions: on EN2002a (whole-file alignment) substitutions / deletions / insertions per reference
word are 0.054 / 0.416 / 0.007 for CrisperWhisper vs 0.072 / 0.232 / 0.014 for Qwen3-ASR with all
speakers, and 0.044 / 0.267 / 0.011 vs 0.054 / 0.110 / 0.019 with two. Which words it drops (short
turns, backchannels, overlap) was not broken down; it matters for a verbatim pipeline and is worth
checking on the interviews.

### 5.3 Features and model choice

Word-weighted least squares, judged by **leave-one-recording-out** (LOO) prediction:

| feature(s), 30 s windows | Pearson r (fit) | LOO r | LOO MAE |
|---|---|---|---|
| WER of secondary vs primary-as-reference (first draft) | 0.40 | 0.19 | 0.228 |
| same + (1 - mean confidence) | 0.71 | 0.66 | 0.148 |
| WER of primary vs secondary-as-reference | 0.21 | 0.15 | 0.434 (blows up when the secondary has few words) |
| same, capped at 1 | 0.93 | 0.93 | 0.075 |
| **disagreement = edits / max(n_primary, n_secondary)** | 0.95 | 0.948 | 0.067 |
| **disagreement + (1 - mean confidence)** | 0.95 | 0.949 | 0.067 |
| (1 - mean confidence) alone | 0.71 | | |

At 60 s windows the disagreement models give LOO r 0.944-0.946, MAE 0.060-0.062. The shipped profile
(`data/accuracy_calibration.yaml`, 30 s fit) is

- with confidence: `wer = 0.0776 + 0.8791 * d + 0.1335 * (1 - mean_conf)`
- without: `wer = 0.0833 + 0.9267 * d`

Mean word confidence adds almost nothing once the disagreement is known (LOO MAE 0.067 either way);
it is kept because it is free and is the only signal when the two systems agree on a wrong word.

### 5.4 Word confidence and the low-confidence threshold

Primary word error rate by confidence bin (alignment against the reference, 18,082 words):

| confidence | < 0.3 | 0.3-0.5 | 0.5-0.7 | 0.7-0.8 | 0.8-0.9 | 0.9-0.95 | 0.95-0.99 | >= 0.99 |
|---|---|---|---|---|---|---|---|---|
| words | 718 | 1292 | 2234 | 1674 | 1967 | 1560 | 3273 | 5364 |
| error rate | 0.64 | 0.38 | 0.29 | 0.17 | 0.11 | 0.09 | 0.05 | 0.02 |

The confidence is monotone in error. `low_confidence_threshold = 0.45` is the highest value (0.05 grid)
below which words are wrong at least half the time (cumulative).

### 5.5 Interval and fit quality, stated plainly

- Interval: `estimate +- q / sqrt(n_primary_words)`, `q` = 80th percentile of |LOO residual| x
  sqrt(n) (0.831 with confidence). LOO coverage measured **0.789** against the nominal 0.80.
- Whole-recording LOO estimates (30 s windows, word-weighted): true -> estimated
  0.478 -> 0.446, 0.878 -> 0.833, 0.318 -> 0.357, 0.477 -> 0.452, 0.351 -> 0.388, 0.524 -> 0.596.
  Errors -0.07 to +0.07; the clean two-speaker recordings are over-estimated by ~0.04.
- Limits: two meetings, one corpus, four speaking conditions; leave-one-recording-out holds out a
  condition, not a corpus, and two of the six recordings share content with another. AMI headset
  mixes are not Zoom: codec, single far microphone and two-speaker turn-taking differ, and the true
  WER range here (median 0.47) is higher than interviews will likely show. Treat the estimate as a
  ranking of turns and recordings by likely error and as a rough level, and validate it on a few
  hand-transcribed minutes of the target interviews before reporting absolute WERs.


## 6. Open items

- Qwen3-ASR's 180 s chunk truncates dense speech at `max_new_tokens=512` on CUDA and CPU too
  (Section 2.2, finding 1). Not changed there for lack of a CUDA host to measure on; callers can pass
  `max_chunk_sec=30` today. Fix candidates: 30 s chunks everywhere, or a token budget scaled to chunk
  length.
- Which words CrisperWhisper deletes in conversation (Section 5.2).
- Calibration on the target domain (a few hand-transcribed interview minutes).
- MPS footprint growth inside one Qwen3-ASR worker is bounded by recycling, not explained.

## 7. Files

- `bench/run_bench.sh` — one-command serial re-run of the device/precision/chunking benchmark
  (`bench/crisper_bench.py`, `bench/qwen_bench.py` run inside the subprocess venvs; `bench/fpsampler.py`
  samples `footprint`; `bench/build_mix.py` builds the AMI mixes; `bench/wer_check.py` scores them).
  `bench/results.json` holds every run reported above, with load averages and footprint series.
- `calibration/reproduce.sh` — rebuilds the six calibration recordings, transcribes them through
  senselab (`run_pair.py`), refits (`fit_calibration.py`) and recounts fillers (`filler_recall.py`).
  `calibration/fit_window30.json` (shipped profile) and `fit_window60.json` hold every window's
  measurements; `timings*.jsonl` and `worker_footprint_*.log` hold the long-file RTF and memory runs.
