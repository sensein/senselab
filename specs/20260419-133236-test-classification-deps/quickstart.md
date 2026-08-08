# Quickstart: Test Classification, Dependency Updates, and Modular Architecture

## For Developers

### Running CPU tests locally
```bash
uv run pytest src/tests  # GPU tests auto-skip if no CUDA
```

### Running GPU tests on EC2
Add the `ec2-gpu-test` label to your PR.

### Checking if a function's dependencies are available
```python
from senselab.utils.compatibility import check_compatibility

# Returns True if deps installed, raises clear error if not
check_compatibility("clone_voices")
```

### Using an isolated backend function
```python
from senselab.audio.tasks.voice_cloning import clone_voices

# Works transparently — auto-provisions subprocess venv if needed
result = clone_voices(source_audios, target_audios)
```

## For Maintainers

### Upgrading dependencies
```bash
# On alpha branch:
uv lock --upgrade
uv run pytest src/tests  # CPU tests
# Label a test PR with ec2-gpu-test for GPU tests
```

### Adding a new isolated backend
1. Add entry to compatibility matrix in `src/senselab/utils/compatibility.py`
2. Implement the senselab wrapper to use `run_in_venv()`
3. Add test with `@pytest.mark.skipif` for the backend's availability

### Viewing the compatibility matrix
See `docs/compatibility-matrix.md` or:
```python
from senselab.utils.compatibility import get_matrix
print(get_matrix())
```
