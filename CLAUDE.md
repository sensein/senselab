# senselab Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-06-04

## Active Technologies
- Python ≥3.11,<3.15 (per `pyproject.toml`) + scientific stack already in the repo — `numpy`, `scikit-learn` (≥1.7; `LogisticRegression` for pairwise/learning-to-rank-lite recalibration, plus its transitive `scipy.stats` for `spearmanr` / `kendalltau`), `pyarrow` (parquet rankings + signal tables). No new third-party dependency. Optional integration adapter reads `audio_analysis` per-axis parquets / `disagreements.json`. No model inference in this feature — it operates on already-extracted signals. (20260604-173646-iterative-metric-ranking)
- Filesystem only — a per-corpus **ranking store** directory holding: metric-version JSON files, ranking parquet files (one per version), an annotations JSON store, movement-report JSON files, and a `manifest.json` index. Atomic write-then-`os.replace`; explicit `schema_version` on every artifact (mirrors `speaker_profile/io.py`). (20260604-173646-iterative-metric-ranking)

- Python ≥3.11,<3.15 (per `pyproject.toml`) + senselab audio stack — `Audio`, `extract_speaker_embeddings_from_audios` (SpeechBrain: ECAPA / ResNet-TDNN / X-Vector), `diarize_audios` (pyannote / Sortformer), scene classification (AST / YAMNet), openSMILE features; scientific stack: `numpy`, `scikit-learn` (SpectralClustering/KMeans, silhouette), `torch`, `pyarrow` (parquet) (20260527-151905-speaker-profile-embedding)

## Project Structure

```text
src/
tests/
```

## Commands

cd src && pytest && ruff check .

## Code Style

Python ≥3.11,<3.15 (per `pyproject.toml`): Follow standard conventions

## Recent Changes
- 20260604-173646-iterative-metric-ranking: Added Python ≥3.11,<3.15 (per `pyproject.toml`) + scientific stack already in the repo — `numpy`, `scikit-learn` (≥1.7; `LogisticRegression` for pairwise/learning-to-rank-lite recalibration, plus its transitive `scipy.stats` for `spearmanr` / `kendalltau`), `pyarrow` (parquet rankings + signal tables). No new third-party dependency. Optional integration adapter reads `audio_analysis` per-axis parquets / `disagreements.json`. No model inference in this feature — it operates on already-extracted signals.

- 20260527-151905-speaker-profile-embedding: Added Python ≥3.11,<3.15 (per `pyproject.toml`) + senselab audio stack — `Audio`, `extract_speaker_embeddings_from_audios` (SpeechBrain: ECAPA / ResNet-TDNN / X-Vector), `diarize_audios` (pyannote / Sortformer), scene classification (AST / YAMNet), openSMILE features; scientific stack: `numpy`, `scikit-learn` (SpectralClustering/KMeans, silhouette), `torch`, `pyarrow` (parquet)

<!-- MANUAL ADDITIONS START -->
<!-- MANUAL ADDITIONS END -->
