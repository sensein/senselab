# Contract: `ensure_venv` behavior change

## Signature (unchanged)

```python
def ensure_venv(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
) -> Path: ...
```

## Behavior change

### Before this fix

Inside the lock:
1. Compare requested `requirements` against the marker's stored list.
2. If matches → return cached venv.
3. Otherwise: `uv venv ...` → `uv pip install <requirements> safetensors numpy torchaudio`.
4. Write marker, return venv dir.

### After this fix

Inside the lock:
1. Resolve the torch index first:
   - `env_override = os.getenv("SENSELAB_TORCH_INDEX_URL") or None` (empty string treated as unset).
   - If `env_override` is set → skip `detect_host_cuda()` (it would be wasted work; the picker short-circuits on override).
   - Else → `host_cuda = cuda_probe.detect_host_cuda()`.
   - `torch_index = cuda_probe.pick_torch_index(host_cuda, env_override=env_override)`.
2. Compare requested `requirements` AND `stored["torch_index"]["url"] == torch_index.url` against the marker's stored data.
3. If matches → return cached venv.
4. Otherwise:
   a. `uv venv ...` (recreate fresh).
   b. `uv pip install --index-url <torch_index.url> --extra-index-url https://pypi.org/simple --python <venv-python> <requirements> safetensors numpy torchaudio`.
   c. On `subprocess.CalledProcessError`, parse stderr for "no matching distribution" / "could not find a version" patterns. If matched, raise `SenselabCudaCompatibilityError(host_cuda=..., attempted_index=..., failing_packages=[...])`. Otherwise, re-raise the original.
   d. Write the marker including `torch_index`.
5. Return venv dir.

## New failure mode

`SenselabCudaCompatibilityError` raised when:
- The chosen torch index has no wheel matching the venv's python version + platform.
- The override URL provided via `SENSELAB_TORCH_INDEX_URL` doesn't serve a compatible wheel.

Error message format (single line, actionable):

```text
SenselabCudaCompatibilityError: No torch+torchaudio wheel available for this host.
Host CUDA: 12.9 (source: nvidia-smi)
Attempted index: https://download.pytorch.org/whl/cu128
Failing packages: ['torch==2.8.1+cu128', 'torchaudio==2.8.1+cu128']
Action: downgrade CUDA, set SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu for CPU fallback, or wait for upstream wheels.
```

## Cleanup on failure

If `uv pip install` fails, the venv directory is removed before raising. This guarantees no partially-built venv survives, which is what FR-005 and SC-004 require.

## Lock semantics (unchanged)

Per-name file lock at `~/.cache/senselab/venvs/<name>.lock` with 600-second timeout. Multiple processes calling `ensure_venv` with the same `name` serialize through this lock. Behavior under concurrency is unchanged by this fix.

## Marker format

See `data-model.md` "Marker schema" section. Summary: adds a `torch_index` field. Markers written by the pre-fix code (no `torch_index` key) trigger a one-time rebuild on first run after upgrade — handled by the same mismatch-clause that handles every other change to the marker payload.

## Test surface

Unit tests required in `src/tests/utils/subprocess_venv_test.py`:

- Pre-fix-shape marker (no `torch_index` key) → mismatch → rebuild path.
- Marker with matching `torch_index.url` → cache hit.
- Marker with non-matching `torch_index.url` → mismatch → rebuild.
- `uv pip install` failure with "no matching distribution" stderr → `SenselabCudaCompatibilityError` raised, venv dir removed.
- `uv pip install` failure with unrelated error → original `CalledProcessError` re-raised, venv dir removed.
- Env override set → marker records `torch_index.source == "env-override"`, install argv uses override URL.
