# DriftSE: the device that never left the host, and a ceiling that was a round number

Two defects in `src/senselab/audio/tasks/speech_enhancement/driftse.py`, the same pair fixed for
the unasdiff separation backend in `specs/20260818-071500-unasdiff-device-timeout-pcm16/design.md`.
That document is the reference for the approach; this one records where DriftSE agrees with it,
where it does not, and what was measured here rather than carried over.

Written 2026-08-18. The PCM_16 defect (D-3 there, which also names `driftse.py`) is fixed on the
unasdiff branch and is deliberately untouched here, so the two changes do not collide.

## D-1. A caller could not select a device

### What was wrong

`enhance_audios_with_driftse` accepted `device`, handed it to `_select_device_and_dtype` for
validation, and discarded the return value. Nothing about the caller's choice reached the worker
payload; the worker picked for itself with

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

The docstring said so outright — "Accepted for signature parity with the other enhancers. The
worker selects CUDA when available and CPU otherwise" — so this was a known limitation rather than
a silent bug. It still made the parameter a decoration on the one enhancer where the choice is a
real decision (DriftSE is 1 NFE, so CPU is genuinely usable and a caller may want to keep the GPU
for something else), and the bare `"cuda"` took whatever index torch defaulted to.

### The upstream `CUDA_VISIBLE_DEVICES` pin does **not** reproduce here

unasdiff needed a save/restore around its upstream imports because `models/atten_unet.py:6`
assigns `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` at module scope, before its own `import torch`.
This was checked for DriftSE rather than assumed, against the pinned clone at
`0a489dadfa2778e86e4b4b0af03f6255d2de8c69`:

- A `grep -rn "CUDA_VISIBLE_DEVICES"` over the whole upstream tree returns **three hits, all in
  shell scripts** — `train.sh:14,15` and `test.sh:10`, each `export CUDA_VISIBLE_DEVICES=$GPU_ID`
  in a launcher the worker never runs. No Python module assigns it.
- The worker's import chain is `backbones.ncsnpp_v2`, `backbones.ncsnpp_v2_drift` and `util.other`
  (plus `backbones/__init__.py`, which imports all six backbones). None of them touches the
  variable.

So no save/restore was added: there is nothing to work around. The only ordering fact worth
recording is that `util.other.set_torch_cuda_arch_list()` — called immediately before device
selection — is the worker's first CUDA API call (`torch.cuda.is_available()`, then
`get_device_capability` per card), and it *reads* the launcher's mask rather than writing one.

### What was done

The host sends the caller's device in the payload, exactly as unasdiff now does:

- `device=None` sends `None`, and the worker takes `cuda:<current index>` when CUDA is available
  and CPU otherwise. `None` is **not** resolved on the host: the host interpreter and the venv have
  separate torch builds, and only the venv's `torch.cuda.is_available()` governs where the worker
  can actually run. Resolving here would let a CPU-only host build silently pin a CUDA-capable
  worker to CPU.
- An explicit `DeviceType` goes through `_select_device_and_dtype` (validation, unchanged — MPS is
  still rejected) and then `device_run_opt`, which returns `f"cuda:{torch.cuda.current_device()}"`
  rather than a hardcoded `0`, so under a Slurm-style mask it names the allocated card.

The worker never builds a bare `torch.device("cuda")`; an index is always chosen. A CUDA request
the venv cannot honour raises inside the worker, naming the requested device and the mask, and
`parse_subprocess_result` propagates it as-is.

`api.enhance_audios` already forwarded `device` to this function, so nothing changed there.

## D-2. A hardcoded 1800 s ceiling

### What was wrong

`subprocess.run(..., timeout=1800)`, hardcoded, with no `except`. On `TimeoutExpired` the exception
propagated out of the `TemporaryDirectory` context manager, which deleted every output the worker
had already written, and the traceback said only that a subprocess had run too long.

### What one window costs, measured on this host

Measured on this machine (Apple silicon, CPU — DriftSE's realistic worst case, and the reason the
backend exists), default variant `distillhubert_three_layers_with_z`, σ 0.01, seed 0, `chunk_s=20`,
`overlap_s=2`, warm venv and warm checkpoint cache. Inputs were tiles of
`tutorial_audio_files/english_conversation_higgs_audio_v2.wav` at 16 kHz mono:

| input | windows | wall time |
|---|---|---|
| 20.0 s | 1 | 24.76 s, then 23.47 s on a repeat |
| 58.0 s | 4 | 87.92 s |

Differencing removes the per-call fixed cost (interpreter start, torch import, checkpoint load):

```
(87.92 - 23.47) / 3 windows = 21.48 s per 20 s window  ->  1.074 s per window-second
(87.92 - 24.76) / 3 windows = 21.05 s per 20 s window  ->  1.053 s per window-second
```

The fixed cost falls out as roughly 2 s on a warm cache. `_SECONDS_PER_WINDOW_SECOND = 1.1` rounds
the larger figure up.

This is a **measurement, not an estimate**, with three caveats: it is one host, one CPU, one
checkpoint, and another agent may have been running on the machine at the time — contention makes
it pessimistic, which is the safe direction for a ceiling. No GPU figure was taken; a GPU is
faster, so the CPU number is the conservative basis for a ceiling that must cover both.

**The old ceiling in those terms.** At 1.07 s per window-second, 1800 s of worker time covers about
84 windows, i.e. roughly 25 minutes of audio, on this host. Any longer recording lost everything.

### The formula, and why it is not unasdiff's

unasdiff's cost is dominated by 200 reverse-diffusion steps per window, so its constant is seconds
per (window x diffusion step). DriftSE is **one network evaluation per window** — that contrast is
in its own module docstring — so there is no step count to multiply by, and copying unasdiff's
formula would multiply this backend's cost by a number that does not exist here. The work term is
instead the audio the worker actually pushes through the network:

```
max(_TIMEOUT_FLOOR_S, _TIMEOUT_HEADROOM x _SECONDS_PER_WINDOW_SECOND x n_windows x chunk_s)
```

| constant | value | where it comes from |
| --- | --- | --- |
| `_SECONDS_PER_WINDOW_SECOND` | 1.1 | the CPU measurement above, 21.48 s / 20 s window |
| `_TIMEOUT_HEADROOM` | 4.0 | see below |
| `_TIMEOUT_FLOOR_S` | 1800.0 | see below |

`n_windows` is counted host-side by `_window_count`, which mirrors the worker's own chunking —
fixed-length windows on a `chunk_s - overlap_s` hop, plus a final window anchored at the end of the
signal when the regular ones do not reach it — summed over every input. Windows, not audio
seconds, because overlap makes the worker evaluate more audio than the caller supplied: 58 s of
input at 20/2 is four windows, i.e. 80 window-seconds.

**Why the work term is per window-second rather than per window.** `chunk_s` is a caller parameter,
and a 40 s window costs more than a 20 s one. Scaling by `chunk_s` keeps the ceiling proportionate
when a caller changes it. The scaling is linear while the true cost is slightly superlinear — the
NCSN++ backbone carries attention layers, which is the reason chunking exists at all — so for
`chunk_s` well above 20 the work term underestimates. The headroom factor is what absorbs that, and
`chunk_s` is not a knob any senselab caller currently changes.

**Why a headroom factor rather than the measured number alone.** One host, one clip, warm caches,
and a *ceiling* rather than a budget: setting it too high means a genuinely hung worker takes
longer to fail, setting it too low means losing a multi-hour run. That asymmetry picks 4x. It is
not a fitted threshold and nothing downstream reads it as one — it gates no verdict, only how long
a subprocess may live — which is why it is a module constant and not a `data/` profile.

**Why a floor, and why 1800.** A short input is not a short run on first use: the worker clones
upstream at the pinned commit into its venv, `ensure_venv` may be building that venv (torch,
torchaudio, librosa), and a 1.14 GB checkpoint is loaded before the first window is evaluated. None
of that scales with window count. 1800 s is the value the backend already shipped, so the floor is
also a compatibility statement: **this change never gives any call a smaller ceiling than it had.**
The work term overtakes the floor at about 21 windows of 20 s, i.e. about 6.4 minutes of audio.

**The failure is now actionable.** `TimeoutExpired` is caught and re-raised as `RuntimeError`
naming the ceiling that fired, how many of the N outputs had been written, how many inputs and how
many seconds of audio were being enhanced, the window count and window length, the variant, the
device, and the `timeout_s` parameter that raises the ceiling.

### Partial output: counted, not returned — for a different reason than unasdiff's

The unasdiff branch declined to salvage partial output because its per-window files overlap-add
into a signal that is silent past the last completed window, so returning it would be a silently
truncated result presented as complete. **That argument does not transfer.** DriftSE's worker
writes one file per *input*, with a single `sf.write` after every window of that input has been
overlap-added, so a file that exists is complete and correct — apart, at most, from the one file
being written at the instant the ceiling fired. There is no truncated-signal hazard: a DriftSE
output is either whole or absent.

Salvage is still refused, on two other grounds:

1. **The contract is positional.** `List[Audio]`, one per input, in order. Returning the two that
   finished out of three would silently re-associate the caller's inputs with the wrong outputs.
   Anything safer — padding with `None`, returning a sparse mapping — is a return-type change to
   serve a failure path, and the caller who wants per-input isolation can call per input.
2. **Preserving the files only helps if a rerun can consume them**, which means a content-addressed
   per-input cache keyed on (upstream commit, checkpoint revision, variant, sigma, seed, chunking)
   plus resume logic. That is the `cached_inference` machinery with its own invalidation story, not
   something to graft onto a timeout handler.

So the completed outputs are counted and reported — that count is the useful diagnostic, since it
says whether the ceiling was slightly or wildly too low — and discarded with the temporary
directory. The count is `Path(p).is_file()` per expected output, which is a progress indicator and
not a salvage manifest: it can include a file the worker was midway through writing. That costs
nothing, because nothing is returned. The docstring says outright that every output is discarded.

### `timeout_s` is not plumbed into `api.enhance_audios`

unasdiff's `api.separate_audios` is a single-backend entry point, so `timeout_s` belongs on it.
`speech_enhancement.api.enhance_audios` dispatches to SpeechBrain as well, which runs in-process and
has no subprocess to bound; a `timeout_s` there would be a parameter that silently does nothing for
the default backend. A caller who needs the knob calls `enhance_audios_with_driftse` directly, which
is already the documented route for `sigma`, `variant` and the chunking. The derived default is what
serves callers going through `enhance_audios`, and it is the reason the default has to be derived
rather than a constant.

## Tests

`src/tests/audio/tasks/speech_enhancement_test.py`, nine added. All but the last fail against the
pre-fix module:

| test | defect | fails pre-fix | pre-fix failure |
| --- | --- | --- | --- |
| `test_the_callers_device_reaches_the_driftse_worker_payload` | D-1 | yes | `KeyError: 'device'` |
| `test_no_device_leaves_the_choice_to_the_driftse_worker` | D-1 | yes | `KeyError: 'device'` |
| `test_the_driftse_worker_never_requests_a_bare_cuda_device` | D-1 | yes | `assert 'torch.device("cuda" if torch.cuda.is_available() else "cpu")' not in <worker script>` |
| `test_the_driftse_default_timeout_scales_with_windows_and_window_length` | D-2 | yes | `AttributeError: module ... has no attribute '_default_timeout_s'` |
| `test_the_window_count_mirrors_the_workers_own_chunking` | D-2 | yes | `AttributeError: module ... has no attribute '_window_count'` |
| `test_the_derived_driftse_ceiling_reaches_subprocess_run` | D-2 | yes | `AttributeError: ... has no attribute '_TIMEOUT_FLOOR_S'` |
| `test_an_explicit_driftse_timeout_overrides_the_derived_one` | D-2 | yes | `TypeError: enhance_audios_with_driftse() got an unexpected keyword argument 'timeout_s'` |
| `test_a_non_positive_driftse_timeout_raises` | D-2 | yes | same `TypeError` |
| `test_a_driftse_timeout_names_the_ceiling_the_input_and_the_progress` | D-2 | yes | same `TypeError` |
| `test_an_incompatible_device_is_rejected_before_the_venv` | D-1 | **no** | held before; kept as a guard on the validation the plumbing reuses |

None needs a GPU, the venv or the real checkpoint: the venv, `venv_python` and `subprocess.run` are
stubbed, and `SENSELAB_DRIFTSE_CHECKPOINT` is pointed at an empty `tmp_path` so the Hub is never
reached. The two device-payload tests therefore assert what the host *sends*, not what a worker does
with it; the worker's own `resolve_device` is exercised by the skip-gated end-to-end tests, and its
CUDA branch is unreachable on any host in this repository's CI.

`test_the_derived_driftse_ceiling_reaches_subprocess_run` monkeypatches `_TIMEOUT_FLOOR_S` down to
1 s. Without that, proving the ceiling is derived rather than constant would need enough audio to
push the work term past 1800 s — about six and a half minutes of synthetic input — for a property
that is about arithmetic.

`test_the_window_count_mirrors_the_workers_own_chunking` additionally asserts that the worker
script still contains the two chunking lines `_window_count` mirrors. It is a string check, and it
catches the realistic regression: someone changes the worker's windowing and the host's ceiling
silently starts sizing a different amount of work.
