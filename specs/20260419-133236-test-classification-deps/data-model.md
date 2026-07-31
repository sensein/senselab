# Data Model: Test Classification, Dependency Updates, and Modular Architecture

## Entities

### Compatibility Entry

| Field | Type | Description |
|-------|------|-------------|
| function_name | str | Fully qualified function name (e.g., `senselab.audio.tasks.voice_cloning.clone_voices`) |
| required_deps | list[str] | Pip package names required (e.g., `["coqui-tts>=0.27"]`) |
| python_versions | str | Python version constraint (e.g., `">=3.11,<3.13"`) |
| torch_versions | str | Torch version constraint (e.g., `">=2.8"`) |
| isolated | bool | Whether this function runs in a subprocess venv |
| venv_name | str or None | Name of the subprocess venv (e.g., `"coqui"`, `"ppgs"`) |
| gpu_required | bool | Whether CUDA is required |

### Subprocess Venv

| Field | Type | Description |
|-------|------|-------------|
| name | str | Unique venv identifier (e.g., `"coqui"`, `"ppgs"`) |
| requirements | list[str] | Pip install specs (e.g., `["coqui-tts~=0.27", "torch~=2.8"]`) |
| python_version | str | Python version for this venv (e.g., `"3.11"`) |
| cache_path | Path | `~/.cache/senselab/venvs/{name}/` |
| lock_path | Path | `{cache_path}.lock` |

### Test Classification

| Classification | Condition | Runner |
|---------------|-----------|--------|
| CPU-safe | No `torch.cuda.is_available()` skipif | GitHub Actions (macOS/ubuntu) |
| GPU-required | Has `torch.cuda.is_available()` skipif | EC2 only |
| Dual-mode | Has CUDA skipif but CPU path also valid | Both (skips GPU on GHA, runs GPU on EC2) |

### Dependency Tier

| Tier | In Core? | Upgrade Strategy |
|------|----------|-----------------|
| Core | Yes | `uv lock --upgrade`, keep in pyproject.toml |
| Isolated | No | Removed from pyproject.toml, specified in compatibility matrix venv specs |
