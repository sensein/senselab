# unasdiff: the GPU nobody could select, the ceiling that discarded the run, and PCM_16 again

Three defects in `src/senselab/audio/tasks/source_separation/unasdiff.py`, one of which reaches
`speech_enhancement/driftse.py` as well. Each was verified against the code and against upstream at
the pinned commit `5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa` before being fixed.

## D-1. A caller could not select a GPU, for two compounding reasons

### What was wrong

**Ours.** `separate_with_unasdiff` accepted `device`, handed it to `_select_device_and_dtype` for
validation and threw the return value away. Nothing about the caller's choice reached the worker
payload; the worker picked for itself with

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The docstring documented this ("Accepted for signature parity ... The worker selects CUDA when
available and CPU otherwise"), so it was a known limitation rather than a silent bug — but it made
the parameter a decoration, and the bare `"cuda"` took whatever index torch defaulted to.

**Upstream's.** `models/atten_unet.py` line 6 executes

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

at module scope, *before* its own `import torch` on line 7. `models/__init__.py` is
`from .atten_unet import *`, so a bare `import models` fires it. In our worker the sequence was:

1. `import torch` — CUDA is not initialised by an import,
2. `import models` — the pin overwrites whatever the launcher set,
3. `torch.cuda.is_available()` — the first CUDA API call, which initialises CUDA and enumerates
   against the *overwritten* value.

CUDA initialises lazily, which is exactly why the pin worked: nothing had read the variable before
step 2. Verified on a four-GPU node — four workers launched with distinct `CUDA_VISIBLE_DEVICES`
all ran on physical GPU 0, confirmed through `/proc/<pid>/environ`.

`diffusion/gaussian_diffusion.py` line 34 assigns the same variable, but inside `load_spk_model`
under `if device is None:` — a default-gathering branch of a function this worker never calls. It
was checked and deliberately left alone.

### What was done

The worker saves `CUDA_VISIBLE_DEVICES` immediately before `import models`, and restores it
(`os.environ.pop` when it was unset, plain assignment otherwise) immediately after `import
diffusion` — still ahead of any CUDA API call. The launcher's mask is therefore what CUDA
enumerates against, and the upstream pin has no observable effect.

The host now sends the caller's device in the payload:

- `device=None` sends `None`, and the worker takes `cuda:<current index>` when CUDA is available
  and CPU otherwise. `None` is *not* resolved on the host, because the host interpreter and the
  venv have separate torch builds and only the venv's `torch.cuda.is_available()` governs where the
  worker can actually run. Resolving here would let a CPU-only host build silently pin a
  CUDA-capable worker to CPU.
- An explicit `DeviceType` goes through `_select_device_and_dtype` (validation, unchanged) and then
  `device_run_opt`, which is the existing helper for exactly this: it returns
  `f"cuda:{torch.cuda.current_device()}"` rather than a hardcoded `0`, so under a Slurm-style
  `CUDA_VISIBLE_DEVICES` mask it names the allocated card.

The worker never builds a bare `torch.device("cuda")`; an index is always chosen. A CUDA request
that the venv cannot honour raises inside the worker naming the requested device and the mask, and
that error is propagated as-is by `parse_subprocess_result`.

### Rejected

*Resolving `device=None` on the host as well*, for a uniform payload. Rejected for the split-torch
reason above: it converts a worker-local fact (does this venv see a GPU) into a host-local guess.

*Passing an index through `DeviceType`.* The enum has no index and adding one is a repo-wide
change; the mask is the existing mechanism for choosing a card, and it now works.

## D-2. A timeout discarded the entire run

### What was wrong

`subprocess.run(..., timeout=3600)`, hardcoded, with no `except`. On `TimeoutExpired` the exception
propagated out of the `TemporaryDirectory` context manager, which deleted every per-window file the
worker had already written. This killed a 200-step run outright.

3600 s is not merely arbitrary — it is too small for realistic input. Measured on an exclusive A100
node: 560.71 s for 14.027 s of audio at 200 steps. That is RTF ≈ 40×, against the module docstring's
claim of 22-26× on an H100. 14.027 s at 16 kHz is 224 432 samples, which `_window_starts` covers
with 7 windows (six on the regular 2 s hop, plus one flush against the end), so

```
560.71 s / (7 windows × 200 steps) = 0.400 s per window-step
```

At that rate the shipped 3600 s ceiling covers about 45 windows, i.e. roughly 92 s of audio, on an
A100 — and far less on CPU, where the figure is unmeasured and the fallback softmax attention
materialises a `[b, h, t, t]` matrix per layer.

### What was done

`timeout_s: Optional[float] = None` on `separate_with_unasdiff` and on `api.separate_audios`.
`None` derives the ceiling from the work:

```
max(_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM × _SECONDS_PER_WINDOW_STEP × n_windows × diffusion_steps)
```

| constant | value | where it comes from |
| --- | --- | --- |
| `_SECONDS_PER_WINDOW_STEP` | 0.4 | the A100 measurement above, 560.71 s / (7 × 200) |
| `_TIMEOUT_HEADROOM` | 4.0 | see below |
| `_TIMEOUT_FLOOR_S` | 1800.0 | see below |

**Why a headroom factor rather than the measured number alone.** The measurement is one card, one
clip, an exclusive node and no flash-attn. A shared card, an older card, or a CPU host are all
slower by an amount this repository has not measured, and this is a *ceiling*, not a budget: the
cost of setting it too high is that a genuinely hung worker takes longer to fail, while the cost of
setting it too low is losing every window of a multi-hour run. The asymmetry is what picks 4×. It
is not a fitted threshold and nothing downstream reads it as one — it gates no verdict, only how
long a subprocess is allowed to live — which is why it is a module constant rather than a `data/`
profile.

**Why a floor at all.** A short input is not a short run on first use: the worker clones upstream
at the pinned commit, and both priors are loaded from checkpoint before the first window is
sampled. Those costs do not scale with the number of windows, so a work-proportional term alone
would put a one-window call under a ceiling smaller than its own startup.

**The failure is now actionable.** `TimeoutExpired` is caught and re-raised as `RuntimeError`
naming the ceiling that fired, how many of the N windows had been written when it fired, how many
inputs and how many seconds of audio were being separated, the mode, the diffusion step count, the
device, and the `timeout_s` parameter that raises the ceiling.

### Salvage: counted, not returned

The per-window files that *had* been written are counted (each window's full source set must be
present to count) and reported. They are not returned and not preserved. Two reasons:

1. **A partial result cannot be returned.** The contract is `List[List[Audio]]`, one full-length
   `Audio` per source per input. Windows 0..k of an N-window recording overlap-add into a signal
   that is silent past window k. Returning that is a silently truncated separation — the failure
   mode this repository has repeatedly paid for, and worse than raising.
2. **Preserving them only helps if a rerun can consume them**, which means a content-addressed
   per-window cache keyed on (upstream commit, checkpoint revision, mode, labels, diffusion steps,
   seed, window samples) plus resume logic in the worker. That is the `cached_inference` machinery,
   a design of its own with its own invalidation story — not something to graft onto a timeout
   handler. The worker also loads both priors once for the whole batch, so a resume would repay the
   model load on every attempt; the saving is real but it is not the trivial one it looks like.

Filed as follow-up rather than done here, and the docstring says outright that exceeding the
ceiling discards every window.

## D-3. PCM_16 intermediates, for the fourth time

### What was wrong

Both subprocess backends round-trip audio through WAV files, and every write took soundfile's
default WAV subtype, which is `PCM_16`:

- `unasdiff.py` worker — `sf.write(out_path, ...)`, the separated sources;
- `unasdiff.py` host — `Audio.save_to_file(in_path)`, which writes PCM_16 for a `.wav` (measured);
- `driftse.py` worker — `sf.write(out_path, ...)`, the enhanced signal;
- `driftse.py` host — `Audio.save_to_file(in_path)`.

Anything beyond ±1 is clipped. On the probe recording nothing clipped (largest sample 0.9949 across
126 files) because unasdiff's worker peak-normalises each window before sampling and inverts the
normalisation on the way out — but that is a property of one code path, not of the format, and the
host's *input* windows are not normalised at all: a recording peaking above full scale was clipped
before the worker ever saw it.

This default has silently corrupted a measurement three separate times in this project — two
harnesses and one GPU re-run, in the last of which three SepFormer streams lost up to 8.9% of their
samples and disagreed with the CPU run by 9.5 dB.

### What was done

All four sites write `subtype="FLOAT"`. The two host sites stop using `Audio.save_to_file` for the
hand-off and call `soundfile.write` directly, because `save_to_file` has no subtype parameter.
Each module carries a `_WAV_SUBTYPE = "FLOAT"` constant and the workers receive it in their payload
rather than inlining a literal, so there is one place per backend to change.

A grep of `source_separation/` and `speech_enhancement/` found no other `sf.write` /
`soundfile.write` call. Repository-wide, the remaining default-subtype writes are outside both
packages and outside this change: `features_extraction/sparc.py` (two), `text_to_speech/qwen_tts.py`
(one), and `video/tasks/input_output.py` (one, which passes an explicit `format`). The
`voice_cloning` and `text_to_speech` Coqui paths write FLAC and are unaffected.
`classification/yamnet.py` already had this fixed, as `LOSSLESS_WAV_SUBTYPE`; three modules now
carry the same literal under two names, and giving them one shared home is a follow-up rather than
part of this fix (it would cross task package boundaries).

## Tests

`src/tests/audio/tasks/source_separation_test.py`, fourteen added:

| test | defect | fails pre-fix |
| --- | --- | --- |
| `test_the_worker_restores_cuda_visible_devices_before_it_touches_cuda` | D-1 | yes |
| `test_the_worker_never_requests_a_bare_cuda_device` | D-1 | yes |
| `test_the_callers_device_reaches_the_worker_payload` | D-1 | yes |
| `test_no_device_leaves_the_choice_to_the_worker` | D-1 | yes |
| `test_an_incompatible_device_is_rejected_before_the_venv` | D-1 | no — held before, kept as a guard |
| `test_the_default_timeout_scales_with_windows_and_steps` | D-2 | yes |
| `test_the_derived_ceiling_reaches_subprocess_run` | D-2 | yes |
| `test_an_explicit_timeout_overrides_the_derived_one` | D-2 | yes |
| `test_a_non_positive_timeout_raises` | D-2 | yes |
| `test_a_timeout_names_the_ceiling_the_input_and_the_windows_written` | D-2 | yes |
| `test_separate_audios_forwards_timeout_s` | D-2 | yes |
| `test_separate_audios_forwards_device` | D-1 | yes |
| `test_input_windows_are_written_as_float_not_pcm16` | D-3 | yes |
| `test_every_worker_wav_write_names_an_explicit_subtype` | D-3 | yes |

`src/tests/audio/tasks/speech_enhancement_test.py`, two added:
`test_every_driftse_worker_wav_write_names_an_explicit_subtype` and
`test_driftse_input_wavs_are_written_as_float_not_pcm16`, both failing pre-fix.

**The D-1 ordering test is static, and that is a limitation worth naming.** The pin's effect is
only observable on a multi-GPU host, and this repository's CI has none. The test parses the worker
script with `ast` and asserts four line-number relations: the save precedes the pinning import, the
restore follows every upstream import, and the restore precedes the first `torch.cuda` reference.
That is the property the fix rests on, and it catches the realistic regressions (someone deletes
the restore, or moves a `torch.cuda` call above it). It cannot catch an upstream commit that starts
pinning from a module we do not import — nothing short of a GPU node can.

The `in_peak` assertion in `test_input_windows_are_written_as_float_not_pcm16` checks `> 1.5`
rather than an exact value, because `resample_audios` runs its filter even when the rate is
unchanged: a 1.75 impulse comes back as 1.7276. The property under test is only that the sample
survived a format whose ceiling is 1.0.

The existing end-to-end test stays gated on the venv being present *and* CUDA being available.
Nothing added here needs either.
