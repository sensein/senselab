# unasdiff: MPS, and a ceiling that killed work it had granted

## What was reported

An agent running `speech_speech` on a 25.54 s recording reported the backend "stalled
reproducibly": two attempts, 23 min and 10 min, `uv` sleeping at 0% CPU with zero output and no
Python child, killed both times.

It was not stalled. `uv` and the host process sit at 0% because they are blocked on the subprocess,
which is what a healthy subprocess-venv call looks like; the venv's own python was at ~790% CPU
throughout. What the agent saw at the top of the tree said nothing about the bottom of it.

## Measured

One 4 s window, `diffusion_steps=200`, this model, timed inside the real 200-step schedule:

| device | per diffusion step | per 4 s window |
| --- | --- | --- |
| cpu | **16.7 s** | ~56 min |
| mps | **90-193 s** (108.6, 192.9, 90.2 measured consecutively) | ~5-10 h |
| cuda (A100, prior measurement) | 0.4 s | ~80 s |

So the "stall" was a 25.54 s input -- about 7 windows -- needing roughly **6.5 hours** on CPU.

The CPU figure also validates `_CPU_TIMEOUT_MULTIPLIER = 45`: it predicts 18 s/step against 16.7
measured.

## MPS is slower than CPU, so it is not selected by default

This is the finding that matters, and it inverts the obvious expectation. Two things had to be
fixed before MPS would run at all:

1. **float64.** `_extract_into_tensor` does `torch.from_numpy(arr).to(device=...)[timesteps]
   .float()` -- it moves the schedule to the device and only then casts. MPS has no float64, so the
   move raises before the cast that would have made it moot. `PYTORCH_ENABLE_MPS_FALLBACK` does not
   help: fallback covers unimplemented *ops*, not unsupported *dtypes*.
2. **Index device.** With the cast hoisted, indexing a CPU tensor with an MPS index tensor raises.
   The patch casts, then moves, then indexes -- upstream's own order, with the unconditional
   `.float()` moved ahead of the move. CUDA and CPU results are unchanged.

With both fixed MPS runs, on the GPU (25% device utilisation observed), with no op falling back --
and is **5-11x slower than CPU**. There is no warmup curve to amortise: consecutive steps measured
108.6, 192.9 and 90.2 s.

So `resolve_device` honours `"mps"` when it is named, and never selects it for an unresolved
device. Auto-selecting the fastest *available* accelerator would be a 10x pessimisation here.

**There are two gates, not one.** The worker's `resolve_device` is reached only after a host-side
allowlist, `_select_device_and_dtype(..., compatible_devices=...)`, which listed CUDA and CPU
alone -- so patching the worker without the host leaves the MPS branch dead code. Both were
changed, and `_COMPATIBLE_DEVICES` now names the list once so the two cannot drift apart.

A misattribution worth recording, because it nearly led to reverting a working fix: the float64
failure was blamed on `gaussian_diffusion.py:930` and `:937`
(`torch.from_numpy(sde.alphas).to(device)`), with line 24 believed safe because it ends in
`.float()`. It is the other way round. Line 24 is the failure -- its `.float()` comes *after* the
`.to(device)* -- and `alphas` and `sqrt_one_minus_alphas_cumprod` are **float32** arrays
(verified), so 930 and 937 convert to MPS without complaint. `p_sample_loop` constructs that very
`CorrectorVPConditional` at line 448 and completed on MPS with only the line-24 fix in place.

**Not investigated**: why MPS is this slow. 30 GB of allocated system memory was observed, which
suggests thrashing rather than compute, but that was not chased down.

## The ceiling was sized for a device the worker was not using

`_seconds_per_window_step(None)` returned the CUDA figure. `None` does not mean CUDA -- it means
*the caller left the choice to the worker*, precisely because the host cannot resolve it: the host
interpreter and the venv are different torch builds, and only the venv's
`torch.cuda.is_available()` governs where the worker can run.

The effect: one window at 200 steps was granted **1800 s** while the same work sized for CPU needs
**14400 s**. A run progressing normally was killed at the 8x-too-small ceiling and its completed
windows discarded.

Two changes, because they fix different halves:

- **The host sizes an unresolved device for the slow path.** An over-large ceiling costs nothing;
  an under-large one destroys finished work.
- **The worker refuses up front.** Only the worker knows the device it resolved to, so only the
  worker can compare the real cost against the ceiling. It is now told `deadline_s` and raises
  immediately, naming windows, steps, device, estimate and ceiling. A 60 s ceiling on 2 windows now
  fails in seconds with "about 6800s of work, over the 60s ceiling" instead of after 85 minutes.

## Rejected

**Monkeypatching MPS support in and defaulting to it.** The measurement says it would be slower.

**`PYTORCH_ENABLE_MPS_FALLBACK=1` as the mechanism.** It would mask exactly what we wanted to know
-- whether ops run on the GPU -- by silently returning them to CPU, reintroducing the original cost
with no signal. The strict setting was used for every measurement here.

**Estimating on the host.** It cannot: wrong torch build. That is the whole reason the worker
resolves its own device.
