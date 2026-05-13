# Contract: `SENSELAB_TORCH_INDEX_URL`

## Purpose

Operator escape hatch for hosts where the static CUDA → PyTorch wheel index map is wrong (rare CUDA versions, internal PyPI mirrors, air-gapped builds, ARM CUDA hosts, ROCm).

## Semantics

| State | Meaning |
|---|---|
| Unset (or empty string) | `cuda_probe.pick_torch_index()` consults the static map, picks the highest `cuXX ≤ host CUDA`, falls back to `cpu`. |
| Set to a non-empty URL | That URL is used verbatim as the `--index-url` argument to `uv pip install`. The static map and host CUDA probe are ignored for the index choice (but the probe still runs and is recorded in the marker for diagnostic purposes). |

The override URL is passed to `uv pip install` exactly as-is. Senselab does not validate the URL is reachable or that it serves PyTorch wheels — if the operator misconfigures it, the install will fail and surface a `SenselabCudaCompatibilityError` with the override URL named in the diagnostic.

## Examples

```bash
# Force CPU wheels (e.g. a CUDA host where you want to verify CPU behavior)
SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu uv run python scripts/analyze_audio.py ...

# Internal PyPI mirror that proxies PyTorch wheels
SENSELAB_TORCH_INDEX_URL=https://pypi.internal.example.com/pytorch/cu128 uv run python ...

# ARM CUDA host (NVIDIA's PyPI mirror)
SENSELAB_TORCH_INDEX_URL=https://pypi.nvidia.com uv run python ...

# Force a specific PyTorch CUDA version even when host CUDA is higher
SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 uv run python ...
```

## Interaction with marker file

The marker file records the resolved index URL (whether from static map or override). If the user changes `SENSELAB_TORCH_INDEX_URL` between runs, the marker mismatch triggers a venv rebuild — the new URL is honored on the next install.

## Interaction with the rest of `uv pip install`

When the override is in effect, the install argv is:

```text
uv pip install --index-url $SENSELAB_TORCH_INDEX_URL --extra-index-url https://pypi.org/simple --python <venv-python> <requirements> safetensors numpy torchaudio
```

The `--extra-index-url` for PyPI is still present so non-torch packages (nemo_toolkit, qwen-asr, soundfile, etc.) continue to resolve from PyPI.

## Persistence

Senselab does not write the override anywhere outside the venv marker. Restart of the shell session resets the variable to unset. Users wanting a persistent override should add it to their shell profile or to the env file their workflow loads.

## Documentation pointers

- Listed in repo `CLAUDE.md` env-var index.
- Listed in repo `README.md` "Troubleshooting / CUDA hosts" section (to be added in the docs task).
- Mentioned in the `SenselabCudaCompatibilityError` message as a recommended user action.
