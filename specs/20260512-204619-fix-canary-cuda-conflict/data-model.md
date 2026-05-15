# Phase 1 — Data Model

## Value objects

### `HostCuda`

Result of probing the host's runtime CUDA capability. Produced by `cuda_probe.detect_host_cuda()`.

| Field | Type | Description |
|---|---|---|
| `version` | `tuple[int, int] \| None` | Major + minor, e.g. `(12, 9)`. `None` when no CUDA detected. |
| `source` | `Literal["nvidia-smi", "nvcc", "none"]` | Which probe succeeded. `"none"` only when both failed. |
| `raw` | `str` | Raw stdout the probe parsed (or `""` for `"none"`). Kept for the diagnostic in FR-004. |

**Validation rules**:
- `version is None` ⟺ `source == "none"`.
- `version` has both elements `≥ 0` when present.
- Immutable; produced fresh on every probe call (no class-level cache; caching is the caller's responsibility — `ensure_venv` records it in the marker).

### `TorchIndex`

The chosen index URL for the subsequent `uv pip install`. Produced by `cuda_probe.pick_torch_index(host_cuda, env_override=None)`.

| Field | Type | Description |
|---|---|---|
| `url` | `str` | Full HTTPS URL of the chosen index. |
| `tag` | `str` | Short identifier — `"cu128"`, `"cu126"`, `"cu124"`, `"cu121"`, `"cpu"`, or `"override"` when the env var supplied an arbitrary URL. |
| `cuda_version` | `tuple[int, int] \| None` | The index's CUDA version, not the host's. `None` when tag is `"cpu"` or `"override"`. |
| `source` | `Literal["static-map", "env-override"]` | How this index was selected. |

**Validation rules**:
- `tag == "override"` ⟺ `source == "env-override"`.
- `tag == "cpu"` ⟺ `cuda_version is None and source == "static-map"`.
- `url` starts with `https://`.

## Marker schema (`.senselab-installed`)

Stored at `~/.cache/senselab/venvs/<name>/.senselab-installed` after a successful venv build.

**Current (pre-fix)**:
```json
{
  "requirements": ["nemo_toolkit[asr,tts] @ git+...", "torch>=2.8,<2.9", ...],
  "python_version": "3.12"
}
```

**After this fix**:
```json
{
  "requirements": ["nemo_toolkit[asr,tts] @ git+...", "torch>=2.8,<2.9", ...],
  "python_version": "3.12",
  "torch_index": {
    "tag": "cu128",
    "url": "https://download.pytorch.org/whl/cu128",
    "source": "static-map"
  }
}
```

**Mismatch rules** (any one triggers a rebuild):
- `stored.get("requirements") != sorted(requirements)` — existing behavior, unchanged.
- `stored.get("torch_index", {}).get("url")` differs from the currently-resolved `TorchIndex.url` — new clause. Absence of `torch_index` in the existing marker (old shape) counts as a mismatch, which is the intended one-time rebuild for users upgrading through this fix.

No `schema_version` field — the marker is internal state, not a published contract. Adding fields by field-presence is sufficient; explicit versioning would be ceremony with no payoff.

## State transitions

`ensure_venv` state diagram:

```text
                ┌─────────────────┐
   call ─────►  │ Acquire lock    │
                └────────┬────────┘
                         ▼
                ┌─────────────────────────┐
                │ Read marker (if exists) │
                └────────┬────────────────┘
                         ▼
              ┌───────── match? ──────────┐
              │                           │
        match │                           │ mismatch / missing
              ▼                           ▼
       Return venv_dir            Probe host CUDA
                                  Pick torch index
                                  Apply env override
                                          │
                                          ▼
                                   uv venv (recreate)
                                   uv pip install
                                   with --index-url
                                          │
                              ┌───────────┴───────────┐
                              │                       │
                          success                   failure
                              │                       │
                              ▼                       ▼
                  Write marker w/ torch_index   Wrap into
                  Return venv_dir              SenselabCudaCompatibilityError
                                                re-raise
```

## Public API surface added

```python
# src/senselab/utils/cuda_probe.py

@dataclass(frozen=True)
class HostCuda:
    version: tuple[int, int] | None
    source: Literal["nvidia-smi", "nvcc", "none"]
    raw: str

@dataclass(frozen=True)
class TorchIndex:
    url: str
    tag: str
    cuda_version: tuple[int, int] | None
    source: Literal["static-map", "env-override"]

class SenselabCudaCompatibilityError(RuntimeError):
    """Raised when no torch/torchaudio binary pair is installable on this host."""

def detect_host_cuda() -> HostCuda: ...

def pick_torch_index(
    host_cuda: HostCuda,
    env_override: str | None = None,
) -> TorchIndex: ...
```

The two existing public functions in `subprocess_venv.py` (`ensure_venv`, `venv_python`) keep their signatures. The behavior of `ensure_venv` changes: it now probes CUDA and routes the install through the chosen index; on unsupported-CUDA hosts it raises `SenselabCudaCompatibilityError` instead of letting `uv pip install` fail with an opaque error.

## Backwards compatibility

- Existing marker files without a `torch_index` key: trigger one rebuild on first run after upgrade (intentional, see Mismatch rules).
- Existing callers of `ensure_venv`: signature unchanged; no migration needed.
- Existing subprocess backends (`canary_qwen.py`, `nemo.py`, `qwen.py`): no code change required. They benefit automatically.
- Users without GPUs: behavior unchanged today — install used `cpu` wheels via PyPI's default torch resolution, post-fix they use `cpu` wheels via the explicit CPU index. Same wheel, different URL.
