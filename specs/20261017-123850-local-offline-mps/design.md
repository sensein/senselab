# Local, offline, MPS: pyannote / SpeechBrain / SER on Apple Silicon, emotion2vec+, network guard

Requested by a downstream pipeline that analyses two-person Zoom interviews on an M1 Pro, fully
offline for privacy. Five changes:

1. `DeviceType.MPS` added to `compatible_devices` for pyannote diarization, SpeechBrain speaker
   embeddings and the in-process SER paths (custom-head wav2vec2/HuBERT/WavLM, SpeechBrain SER), plus
   the continuous-SER venv and the new emotion2vec+ backend.
2. `diarize_audios(..., exclusive=...)` — already on alpha (c81fc9bb); this change adds the test that
   the flag reaches pyannote and that the default stays `True`.
3. `classify_emotions_from_speech` routes from the local cache with `HF_HUB_OFFLINE=1`.
4. emotion2vec+ (`emotion2vec/emotion2vec_plus_{seed,base,large}`) through FunASR in a subprocess venv.
5. `SENSELAB_BLOCK_NETWORK=1`: subprocess-venv workers start behind a socket guard with every
   offline/telemetry switch set.

## Measurements

Host: Apple M1 Pro (10 cores, 32 GB), macOS 26, torch 2.8 (main env), `OMP_NUM_THREADS=3` and
`torch.set_num_threads(3)`, one model resident at a time, `HF_HUB_OFFLINE=1`. Other agents were
running: 1-minute load average 5.9-9.9 during every run (recorded per run by the script), so every
time below is a **load-contaminated upper bound**; ratios between devices measured minutes apart are
the useful number, not the absolute times.

Recording: `src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav` (21.48 s, two TTS
speakers), downmixed and resampled to 16 kHz and tiled 8× (171.84 s) for diarization and embeddings,
4× (85.92 s) for SER. Embeddings use 57 consecutive 3 s clips; SER 17 consecutive 5 s clips.
"× real time" is audio seconds / inference seconds (model load excluded; a warm-up call on one clip
precedes the timed call).

Script: `scripts/benchmark_device_parity.py <backend> <device> --repeat N --out x.json`, then
`--compare cpu.json mps.json`.

### Agreement and speed, CPU vs MPS

| Backend (commit) | Agreement MPS vs CPU | CPU × RT | MPS × RT | Load s CPU / MPS |
|---|---|---|---|---|
| pyannote `speaker-diarization-community-1` (3533c8cf) | 40/40 segments identical (speaker, start, end), both exclusive and overlapping views; max boundary difference 0.0 s | 1.04 (165.9 s) | 12.1 (14.2 s) | 10.9 / 3.4 |
| SpeechBrain `spkrec-ecapa-voxceleb` (0f99f2d0) | cosine min 0.99999999998 over 57 clips | 1.98 (86.9 s) | 47.6 (3.6 s) | 0.2 / 2.4 |
| audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` (6eba34a2) | max abs difference 9.2e-7 (A/V/D) | 24.5 | 44.0 | 0.6 / 2.7 |
| ehcalabres `wav2vec2-lg-xlsr-en-speech-emotion-recognition` (b520c9c4) | max abs prob difference 3.6e-6; argmax 17/17 | 16.5 | 51.6 | 0.8 / 3.6 |
| SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP` (117a9c3d) | max abs prob difference 1.9e-5; argmax 17/17 | 39.1 | 35.6 | 2.0 / 8.9 |
| emotion2vec+ large (6c303ba9), worker inference | max abs prob difference 2.8e-3; argmax 17/17 | 24.0 | 77.1 | 2.3 / 3.0 (in worker) |
| emotion2vec+ base (b318240b), worker inference | max abs prob difference 1.7e-3; argmax 17/17 | 32.8 | 55.6 | 0.7 / 0.9 (in worker) |

emotion2vec+ runs in a fresh worker per call: wall time including interpreter start, FunASR import
and checkpoint load was 9.97 s (CPU) / 8.45 s (MPS) for large and 7.02 s / 6.04 s for base on the
85.92 s input, i.e. 8.6× / 10.2× and 12.2× / 14.2× real time end to end. The worker reports its own
load and inference split (`emotion2vec timing:` at INFO).

The downstream measurement (pyannote community-1 on 178 s: MPS 20 s vs CPU 234 s, identical segments,
obtained by moving the cached pipeline to MPS by hand) agrees with the 11.7× ratio here.

### Decisions from the table

* pyannote, ECAPA, the custom-head SER path and emotion2vec+ are faster on MPS and agree with CPU, so
  `device=None` now selects MPS on Apple Silicon for them (selection order is CUDA → MPS → CPU).
* SpeechBrain SER agrees on MPS but is not faster (35.6× vs 39.1×, within load noise) and loads 7 s
  slower. MPS is allowed because it works; on Apple Silicon `device=None` therefore now picks MPS for
  it too. A caller who wants CPU passes `device=DeviceType.CPU`.
* SpeechBrain cannot construct a `Pretrained` on `mps`: `Pretrained.__init__` sets `device_type` only
  for `cpu`/`cuda*` and raises `AttributeError` otherwise. `speechbrain_load_device` constructs on CPU
  and `speechbrain_move_to` moves `mods` and `device` to MPS. Autocast is unaffected: SpeechBrain
  bypasses it for float32.
* emotion2vec+ differences (≤ 2.8e-3) are larger than the transformer SER models' (≤ 2e-5) but leave
  every argmax unchanged; FunASR's feature pipeline runs in float32 on the device.

## Offline routing for SER

`classify_emotions_from_speech` called `HFModel.get_model_info()` (Hub `model_info`) unconditionally to
read the tags, which raises `OfflineModeIsEnabled` under `HF_HUB_OFFLINE=1` even with the model fully
cached. Now:

1. emotion2vec repositories route first (they have no `config.json`).
2. The SER type comes from the cached `config.json`, loaded at `model.commit_sha` (it used the ref).
3. Tags are consulted only when the labels do not identify an emotion model, and come from the cached
   `README.md` front matter at the commit; the Hub is asked only when the card has no tags and the
   Hub is online.
4. The custom-head peek (`_peek_head_final_layer`) lists files from the local `snapshots/<sha>/` and
   reads safetensors keys from the local file when offline, instead of `HfApi.list_repo_files`.
   Returning `None` offline would have sent a custom-head checkpoint to the standard pipeline, whose
   head-load check then raises; reading locally keeps the right path.

Test: `ser_offline_test.py` — routing with `HF_HUB_OFFLINE` set and every Hub metadata call
replaced by an assertion, plus a fresh interpreter with `HF_HUB_OFFLINE=1` classifying with the
cached audeering model (skipped where it is not cached).

## audeering dimensional model

`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` is **CC BY-NC-SA 4.0** (non-commercial,
share-alike). It declares `Wav2Vec2ForSpeechClassification`, so it routes to the in-process
custom-head path, not to the `continuous-ser` venv (which serves other continuous checkpoints whose
configs the strict `huggingface_hub>=1` validators reject). That path now runs offline and on MPS
(table above). The `continuous-ser` venv also accepts MPS; its worker already ran with
`HF_HUB_OFFLINE=1` via `hf_subprocess_env`.

## emotion2vec+

* Licence: model card `license: other`, pointing to the FunASR Model Open Source License Agreement
  v1.1 — free use, copying, modification and sharing; attribution and the model name must be
  retained; no non-commercial clause. FunASR code is MIT.
* `emotion2vec/emotion2vec_base` is the representation model without a classifier head (no
  `tokens.txt`); it is refused with a `ValueError` naming the `_plus_` checkpoints.
* Venv `emotion2vec`: Python 3.11, `funasr>=1.2.6,<1.3` (locked 1.2.9), torch 2.8. Lock compiled with
  `scripts/lock_subprocess_venvs.py emotion2vec`.
* Pinning: the parent calls `resolve_model(repo, commit_sha)` and passes the local
  `snapshots/<sha>/` directory; the worker calls `AutoModel(model=<dir>, disable_update=True)`
  (FunASR otherwise checks PyPI for a newer version on every load). The payload carries no repo id or
  revision, so the file is listed in `LOADER_CANNOT_PIN_SUBPROCESS_FILES` (pinned by path, like
  `hear.py`).
* Labels: FunASR returns bilingual labels (`生气/angry`) and `<unk>`; they are reported as the English
  part and `unknown`.

## Network guard (`SENSELAB_BLOCK_NETWORK`)

An in-process guard does not reach subprocess-venv workers: each is a new interpreter. With
`SENSELAB_BLOCK_NETWORK` set to 1/true/yes/on, `_clean_subprocess_env` and `hf_subprocess_env` — the
two builders every worker launch goes through — call `apply_network_guard`, which

* sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `HF_HUB_DISABLE_TELEMETRY`,
  `HF_HUB_DISABLE_IMPLICIT_TOKEN`, `DO_NOT_TRACK`, `OTEL_SDK_DISABLED=true`,
  `OTEL_{TRACES,METRICS,LOGS}_EXPORTER=none`, `PYANNOTE_METRICS_ENABLED=false`, `WANDB_MODE=disabled`,
  `GRADIO_ANALYTICS_ENABLED=False`, `MODELSCOPE_OFFLINE=1`;
* prefixes `PYTHONPATH` with `senselab/utils/data/network_guard/`, whose `sitecustomize.py` the
  interpreter's `site` module imports before `-c` code or any module. It replaces
  `socket.socket.connect`/`connect_ex`/`sendto`, `socket.getaddrinfo`, `gethostbyname(_ex)` with
  versions that raise (`PermissionError` / `socket.gaierror`) for non-loopback addresses and names
  before any syscall; `AF_UNIX` and loopback pass. It then chains to any later `sitecustomize`.

`sitecustomize` was chosen over editing each worker string because there are dozens of launch sites and
three launch styles (`python -c`, `call_in_venv` shims, `serve_in_venv`); all of them take their
environment from the two builders. Limits: native code that opens sockets without going through
Python's `socket` module (a C library with its own resolver) is not intercepted, and a worker started
with `python -I`/`-E` ignores `PYTHONPATH` (no senselab launcher does).

Test: `network_guard_test.py` — a child started with `_clean_subprocess_env()` gets
`PermissionError` connecting to 1.1.1.1:443 and `gaierror` resolving `huggingface.co`, while a
loopback TCP connection and an `AF_UNIX` socketpair work; verified by hand in the `continuous-ser`
venv's Python 3.12 as well. The emotion2vec measurements above ran with the flag set.

`install_network_guard()` installs the same guard in the calling process.

## Open items

* SpeechBrain `emotion-recognition-wav2vec2-IEMOCAP` does not load with `HF_HUB_OFFLINE=1`: its
  `HuggingFaceWav2Vec2` lobe calls the Hub API for `facebook/wav2vec2-base` from inside SpeechBrain.
  Its first load also logs a `FileNotFoundError` on a `.incomplete` blob under the CWD-relative
  `wav2vec2_checkpoints/` before succeeding. Both predate this change.
* Absolute times need an idle-host rerun; the device ratios are what this spec relies on.
