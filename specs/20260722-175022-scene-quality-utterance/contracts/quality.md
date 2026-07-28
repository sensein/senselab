# Contract: scene-quality model + per-bucket quality harvester

## A. Brouhaha loader — `scene_quality/brouhaha.py` (new)

```python
@dataclass
class BrouhahaFrames:
    vad: np.ndarray       # (num_frames,) P(speech) in [0,1]
    snr_db: np.ndarray    # (num_frames,) estimated SNR in dB
    c50_db: np.ndarray    # (num_frames,) estimated C50 in dB
    frame_hop_s: float

def extract_brouhaha_frames(
    audios: list[Audio],
    model_id: str = "pyannote/brouhaha",
    revision: str = "<pinned>",
    device: DeviceType | None = None,
) -> list[BrouhahaFrames | None]:
    ...
```

- Loads via `Model.from_pretrained` + `Inference` (same pattern/caching as `frame_posteriors.py`), through `ensure_hf_model` + `local_files_only`, gated `HF_TOKEN`. No new pip dependency, no subprocess venv (FR-025).
- One forward pass on the whole 16 kHz mono audio → three per-frame arrays.
- Returns `None` per audio when the model can't load (FR-023).

## B. Quality harvester — `workflows/audio_analysis/quality.py` (new)

```python
QUALITY_ANALYSIS_WIN_S = 0.5
QUALITY_ANALYSIS_HOP_S = 0.25

def harvest_quality_scores(
    *,
    audio: Audio,                       # raw 16 kHz mono
    brouhaha: BrouhahaFrames | None,
    grid: BucketGrid,                   # the presence reporting grid
    calibration: CalibrationProfile | None = None,
) -> list[dict[str, Any]]:
    """One dict per presence bucket: {'start','end', quality_snr, quality_clip,
    quality_reverb, quality_bandwidth, quality_uncertainty, '_raw': {...}}."""
```

### Signal computation (all → `[0,1]` degradation, 0 = clean)

| Column | Source | Normalization |
|---|---|---|
| `quality_snr` | Brouhaha `snr_db` mean-in-analysis-window (primary); cross-check vs `spectral_gating_snr_metric`, `peak_snr_from_spectral_metric` on the 0.5 s slice | `clip((clean_db − snr_db)/(clean_db − floor_db), 0, 1)` per calibration |
| `quality_clip` | `proportion_clipped_metric` on the bucket slice (cheap, slice-safe) | already `[0,1]`; optional severity curve |
| `quality_reverb` | Brouhaha `c50_db` mean-in-window | `clip((clean_db − c50_db)/(clean_db − floor_db), 0, 1)` |
| `quality_bandwidth` | `librosa.feature.spectral_rolloff(y, sr, roll_percent=0.85)` mean over analysis window | `clip(1 − rolloff_hz / nyquist_ref_hz, 0, 1)` |
| `quality_uncertainty` | normalized stddev across the ≥2 independent SNR estimators (Brouhaha, DSP-a, DSP-b) mapped to `[0,1]` | high when estimators disagree |

### Rules

- **Analysis resolution ≠ reporting grid**: SNR/reverb/bandwidth computed on the fixed 0.5 s / 0.25 s analysis window; each presence bucket takes the value of its containing/nearest analysis window (broadcast). `quality_clip` may be computed on the bucket slice directly. The 0.5 s analysis grid is recorded in provenance.
- **Slicing** follows `embeddings.py:_slice_audio` + tail-anchored `_window_starts` so short trailing buckets get a valid-length slice; STFT metrics need ≥~128 ms.
- **`voice_signal_to_noise_power_ratio_metric` is NOT used per bucket** (internal VAD).
- **SQUIM** (`extract_objective_quality_features_from_audios`) is an optional secondary cross-check for `quality_snr`, batched over analysis windows, gated on `torchaudio_available()`; omitted from the P1 minimum.
- Any missing source → that column is `None` for the affected buckets (FR-023), rest still emitted.
- Raw dB/estimator values retained under `_raw` → serialized into `model_votes` JSON, not columns.

## C. librosa dependency

`librosa` promoted from transitive to an explicit `pyproject.toml` dependency (D8).
