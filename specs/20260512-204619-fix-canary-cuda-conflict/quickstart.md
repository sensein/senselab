# Quickstart — Validating the fix on a CUDA 12.9 host

## Pre-requisites

- Linux x86_64 host with NVIDIA driver supporting CUDA 12.9.
- `nvidia-smi` available on `PATH` and reports a driver version.
- Project cloned and main env synced:
  ```bash
  uv sync --extra text --extra video --extra senselab-ai --extra nlp --extra pii --group dev
  ```
- HuggingFace credentials configured if you don't already have the Canary model cached:
  ```bash
  export HF_TOKEN=<your token>
  ```

## Recovery — clean any pre-fix broken venv

If you hit the original bug before installing the fix, the Canary venv at
`~/.cache/senselab/venvs/nemo-canary-qwen/` may be partially built. After installing
the fix, the v1-marker mismatch will trigger an automatic rebuild on next run, so
you do **not** need to delete it manually. But if you want to force a clean state:

```bash
rm -rf ~/.cache/senselab/venvs/nemo-canary-qwen ~/.cache/senselab/venvs/nemo-canary-qwen.lock
```

## Happy path — Canary transcript on CUDA 12.9

```bash
uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/canary-qwen-2.5b \
  --no-enhancement \
  --skip comparisons
```

Expected:

1. `nvidia-smi` is probed once. Stderr shows a single info-level log line:
   `Detected host CUDA: 12.9 (source: nvidia-smi). Picked torch index: cu128.`
2. `~/.cache/senselab/venvs/nemo-canary-qwen/` is rebuilt cleanly.
3. Canary transcript appears at `artifacts/analyze_audio/<run-dir>/raw_16k/asr/nvidia_canary_qwen_2_5b.json`.
4. The venv's marker carries the resolved index:
   ```bash
   cat ~/.cache/senselab/venvs/nemo-canary-qwen/.senselab-installed | python -m json.tool
   ```
   Look for:
   ```json
   "torch_index": {
     "tag": "cu128",
     "url": "https://download.pytorch.org/whl/cu128",
     "source": "static-map"
   }
   ```
5. Verify both libraries have the same CUDA suffix inside the venv:
   ```bash
   ~/.cache/senselab/venvs/nemo-canary-qwen/bin/python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
   ```
   Both should end in `+cu128`. Critically, neither should raise `ImportError` or report a symbol mismatch.

## CPU-only fallback path

On a host with no GPU (or to force CPU for testing):

```bash
SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/canary-qwen-2.5b \
  --no-enhancement \
  --skip comparisons
```

Expected:

- Probe runs but the override is honored: stderr shows
  `Override SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu in effect.`
- Marker records `"torch_index": {"tag": "override", "source": "env-override", ...}`.
- `torch.__version__` and `torchaudio.__version__` both end in `+cpu`.
- Canary runs (slower) on CPU and produces a transcript.

## Negative path — unsupported wheel set

Simulate an unsupported CUDA by pointing at a non-existent index:

```bash
SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/nonexistent \
  uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/canary-qwen-2.5b \
  --no-enhancement \
  --skip comparisons
```

Expected:

- Single, named, actionable error:
  ```text
  SenselabCudaCompatibilityError: No torch+torchaudio wheel available for this host.
  Host CUDA: 12.9 (source: nvidia-smi)
  Attempted index: https://download.pytorch.org/whl/nonexistent
  Failing packages: ['torch>=2.8,<2.9', 'torchaudio>=2.8,<2.9']
  Action: downgrade CUDA, set SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu for CPU fallback, or wait for upstream wheels.
  ```
- Non-zero exit code.
- `~/.cache/senselab/venvs/nemo-canary-qwen/` does **not** exist (cleanup happened before the error was raised).
- The next run (with a corrected env var, or unset) starts from clean.

## Regression check — host with system CUDA ≤ default-wheels CUDA

On a host where today's behavior already works (e.g. CUDA 12.4):

```bash
uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/canary-qwen-2.5b \
  --no-enhancement \
  --skip comparisons
```

Expected:

- Probe reports `12.4`, picks `cu124` index.
- Install completes; runtime is within 5% of pre-fix baseline.
- Canary transcript appears as in the happy-path case.
- No `SenselabCudaCompatibilityError` raised.

## Smoke check for the other subprocess venvs

The same fix applies to the `nemo` and `qwen-asr` venvs because they share `ensure_venv`. Sanity-check on the CUDA 12.9 host:

```bash
uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/parakeet-tdt-0.6b-v3 Qwen/Qwen3-ASR-1.7B \
  --no-enhancement \
  --skip comparisons
```

Expected:

- Both `~/.cache/senselab/venvs/nemo/` and `~/.cache/senselab/venvs/qwen-asr/` rebuild cleanly and carry `cu128` markers.
- Both backends produce transcripts.

## Recovery path — broken from before the fix

If a teammate gives you a machine that was used pre-fix and has a half-built Canary venv:

```bash
# Just run the workflow normally — the marker mismatch (no torch_index field) triggers automatic rebuild.
uv run python scripts/analyze_audio.py path/to/short_speech.wav \
  --asr-models nvidia/canary-qwen-2.5b \
  --no-enhancement \
  --skip comparisons
```

Expected:

- First run rebuilds the venv (you'll see the install progress).
- Transcript produced.
- No manual `rm -rf` required.

This validates SC-004 ("100% of users who hit the original conflict can recover with one re-run").
