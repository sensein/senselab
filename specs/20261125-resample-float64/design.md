# `resample_audios` computes in float64; results record their environment

Issue: [#650](https://github.com/sensein/senselab/issues/650).

## What was wrong

`resample_audios` ran `torchaudio.functional.resample` (Kaiser sinc, `RESAMPLE_KAISER_PARAMS`) in
float32. The same input, senselab commit and Python environment gave different output on Intel
and AMD CPUs. Single thread (`torch.set_num_threads(1)`), 160 recordings, 48 kHz → 16 kHz:

| Step | Identical across nodes |
|---|---|
| `Audio(filepath=...)` load | 160/160 |
| `downmix_audios_to_mono` | 160/160 |
| `resample_audios`, float32 | 0/160 |
| after a 16-bit PCM write | 40/160 |

- The nodes split by CPU vendor, not by instruction width: AMD EPYC 7513, 7643, 7713, 9474F and
  9654 (AVX2 and AVX512) agree with each other; Intel Xeon Platinum 8462Y+ (AVX512) differs from
  all of them.
- Differences are at most ±1 step at 16-bit (300 differing files checked).
- Forcing `ATEN_CPU_CAPABILITY=avx2 ONEDNN_MAX_CPU_ISA=AVX2 MKL_CBWR=AVX2` changes the Intel
  output but does not make it match AMD.
- Two full builds of a 62,550-file corpus, one per vendor: 1,447 files byte-identical.

Nothing in the output recorded which CPU produced it.

## The change

1. The filter runs on `waveform.double()` and the result is cast to float32. The output dtype and
   the API are unchanged; `precision="float32"` keeps the old computation.
2. `senselab.utils.provenance` records the process a result was computed in (library versions,
   CPU vendor and model, torch CPU capability, threads, determinism flags, CUDA device when in
   use). `record_step` appends a step to `Audio.metadata["processing_steps"]` naming the
   environment by fingerprint, with the full record once per fingerprint under
   `metadata["environments"]`. `resample_audios` is the first caller and records its precision.
3. `CACHE_SCHEMA_VERSION` 27 → 28: resampled inputs change, so cached outputs keyed on the
   pre-resample input are stale.

## Measurements

Same 160 recordings, one Intel (Xeon Platinum 8462Y+) and three AMD nodes (EPYC 7713, 7513,
9474F):

| Step | Identical across Intel + AMD |
|---|---|
| resample, float32 | 0/160 |
| final 16-bit audio, float32 | 40/160 |
| resample, float64, before the cast | 0/160 |
| resample, float64, cast to float32 | 160/160 |
| final 16-bit audio, float64 | 160/160 |

- One AMD node (9474F) compared with the other three, 29.2M samples each (87.6M comparisons):
  float32 had 11,829 differing samples, all against the Intel node, at most one 16-bit step;
  float64 had 0.
- The raw float64 outputs still differ between vendors, far below float32 resolution, so the cast
  removes them. A value landing exactly on a float32 rounding boundary could still flip; none did.
- Against the float32 output, float64 changes 146/160 files by at most one 16-bit step (about 0.23%
  of samples), identically on every node.

Speed, resample only, single thread, best of 3, 30.4 min of audio:

| CPU | float32 | float64 | Ratio |
|---|---|---|---|
| Intel Xeon 8462Y+ | 1.29 s | 2.86 s | 2.2× |
| AMD EPYC 7713 | 1.90 s | 4.87 s | 2.6× |

In float64, 30.4 minutes of audio resample in 2.9–4.9 s, so a recording takes about 1/370 to 1/640
of its own duration.

## Choices

- **Default, not opt-in.** An opt-in flag leaves every caller that does not set it with
  CPU-dependent output. The opt-out exists for speed.
- **Only the resample changes.** Load and downmix were already identical across CPUs.
- **No timestamp or hostname in the environment record.** Both differ between two runs that give
  the same bytes, and a hostname identifies a machine. The rule matches `SpeakerEmbeddingProvenance`
  and `AudioHints.created_at`.
- **Versions and CPU are read once per process; threads, determinism and the CUDA device on every
  call**, because a caller can change those at run time.
- **CUDA is never initialized by the record.** The GPU name is recorded only when CUDA is already
  in use; the driver version stays with `utils.reproduce.host.host_record`, which runs
  `nvidia-smi`.
- **One environment per fingerprint, not per step.** A pipeline of several steps in one process
  stores the environment once.
