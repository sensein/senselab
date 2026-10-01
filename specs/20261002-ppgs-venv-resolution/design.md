# The ppgs venv stopped resolving to a buildable set

## What broke

CI on `design/triage-workflow-dag` (PR #567, first at `98bc4eb8`, 2026-10-01) failed
`test_extract_ppgs_from_audios` and `test_extract_features_from_audios` on both the CPU and the GPU runner.
Both create the `ppgs` subprocess venv, and its Stage-2 install failed:

    error: Failed to build `tokenizers==0.10.3`
      cause: Call to `setuptools.build_meta.build_wheel` failed

CI had passed at `71016f39` (2026-09-30). Nothing in `ppg.py` or `subprocess_venv.py` changed in between.

## Cause

`huggingface-hub` 2.x was released after 2026-09-29. `uv pip compile` of the venv's Stage-2 requirements
(`ppgs>=0.0.9,<0.0.10`, `numpy`, `soundfile`, `safetensors`, with the torch 2.8 constraints), Python 3.11,
manylinux:

| | `--exclude-newer 2026-09-29` | today |
|---|---|---|
| huggingface-hub | 1.33.0 | 2.1.1 |
| transformers | 5.17.0 | **4.12.2** |
| tokenizers | 0.23.2 | **0.10.3** |

`ppgs` requires `transformers` and `huggingface-hub` unbounded. `transformers` 5.17 requires
`huggingface-hub<2.0,>=1.5.0`. Given hub 2.x, the resolver keeps the newest hub and backtracks
`transformers` to 4.12.2, the newest release with no hub upper bound. 4.12.2 pins `tokenizers==0.10.3`,
which has no cp311 wheel and does not build from source.

## Fix

`_PPGS_REQUIREMENTS` gains `huggingface-hub<2`. The resolution is again hub 1.33.0, transformers 5.18.0,
tokenizers 0.23.2. The bound can be lifted once a `transformers` release accepts hub 2.

## Consequences

- The venv's marker records its requirements list, and a changed list makes `ensure_venv` rebuild the venv
  in the same directory on first use. Every host's existing `ppgs` venv is rebuilt once by code at or after
  this commit.
- An older commit run against the same `SENSELAB_VENV_CACHE` afterwards sees its own (unbounded) list,
  rebuilds again, and that rebuild now fails the same way. The triage replay and re-fold do not call PPG
  (PREPROCESS is not replayed), so the pinned cluster runs at older commits are unaffected. A fresh PREPROCESS
  at an older commit would hit it.

## The gap this exposes

Subprocess-venv dependencies other than torch are unpinned: `ensure_venv` passes constraints for torch and
torchaudio only, so any upstream release can change what a venv resolves to, and the cache keys
(`specs/20260928-triage-content-cache/`) do not see it. A per-venv lock or constraints file, checked in and
passed to Stage 2, would make each venv reproducible. Not done here.
