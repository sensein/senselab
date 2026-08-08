# Research: Test Classification, Dependency Updates, and Modular Architecture

## Decision 1: Subprocess venv for isolated backends
**Decision**: `subprocess_venv.py` utility using uv to create/manage isolated venvs
**Rationale**: uv can create venvs with specific Python versions and install packages in <10s. JSON IPC over subprocess is simple and debuggable. Venvs are cached for reuse.
**Alternatives**: Plugin architecture (too complex), Docker containers (too heavy for function calls), importlib tricks (doesn't solve version conflicts)

## Decision 2: Test classification approach
**Decision**: No new markers. Rely on existing `torch.cuda.is_available()` skipif.
**Rationale**: macOS GHA runners don't have CUDA → GPU tests auto-skip. This already works. Adding custom markers would require modifying every test file.
**Alternatives**: Custom pytest marks (`@pytest.mark.gpu`), conftest-based filtering, separate test directories

## Decision 3: Dependency upgrade strategy
**Decision**: Phased upgrade on alpha — actions first, then core, then extras, then isolate conflicts
**Rationale**: Each phase is independently testable and revertable. Catches breakage early.
**Alternatives**: Big-bang upgrade (risky), automated dependabot merge (no conflict resolution)

## Decision 4: Compatibility matrix
**Decision**: Python code artifact + generated markdown
**Rationale**: Runtime-checkable enables graceful errors. Single source of truth avoids doc drift.
**Alternatives**: YAML config (not runtime-checkable), docstrings only (scattered, no aggregation)

## Decision 5: cv2/av conflict resolution
**Decision**: Keep both, use headless opencv, prefer imageio-ffmpeg for ffmpeg binary
**Rationale**: Both are needed (opencv for image processing, av for video I/O). Headless opencv reduces binary conflicts. imageio-ffmpeg provides a clean ffmpeg binary.
**Alternatives**: Drop opencv (breaks video tasks), use av only (lacks image processing), isolate video tasks

## Current Package Version Analysis

| Package | Current Pin | Latest | Conflict? | Action |
|---------|------------|--------|-----------|--------|
| torch | ~=2.8.0 | 2.10.0 | No | Upgrade |
| torchaudio | ~=2.8.0 | 2.10.0 | No | Upgrade |
| torchvision | ~=0.23.0 | 0.25.0 | No | Upgrade |
| transformers | ~=4.53.3 | 4.53+ | No | Upgrade |
| speechbrain | ~=1.0 | 1.0+ | No | Upgrade |
| pyannote-audio | ~=4.0 | 4.0+ | No | Upgrade |
| sentence-transformers | >=5.1,<5.4 | 5.4+ | torchcodec | Unpin, fix import |
| coqui-tts | ~=0.27 | 0.27 | torch pin | Isolate |
| ppgs | >=0.0.9,<0.0.10 | 0.0.9 | espnet → old deps | Isolate |
| snorkel | >=0.10.0,<0.11.0 | 0.10.0 | ppgs dep | Isolate with ppgs |
| lightning | ~=2.4.0 | 2.6+ | ppgs dep | Isolate with ppgs |
| opencv-python-headless | ~=4.11 | 4.11+ | av (libavdevice) | Upgrade, headless |
| ultralytics | ~=8.3 | 8.4+ | No | Upgrade |

## Test Classification Data

| Category | File Count | Examples |
|----------|-----------|---------|
| CPU-only | 26 | quality_control/*, utils/tasks/*, video/*, data structures |
| GPU-required | 21 | speech_to_text, speaker_diarization, voice_cloning, classification |
| Mixed (CPU+GPU) | 7 | audio_test, preprocessing, features_extraction, input_output |
