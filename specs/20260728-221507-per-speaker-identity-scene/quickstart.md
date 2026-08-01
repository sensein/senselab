# Quickstart

**Feature**: `20260728-221507-per-speaker-identity-scene` | **Date**: 2026-07-29

How to set up, run, and validate each phase. All Python goes through `uv` (constitution I).

## Setup

```bash
# Full dev environment
uv sync --extra text --extra video --extra senselab-ai --extra nlp --extra pii --group dev --group docs
uv run pre-commit install

# This feature adds two dependency changes (research D14):
#   - librosa promoted transitive -> explicit  (pcen, A_weighting)
#   - pyloudnorm added                          (BS.1770 LUFS)
# After editing pyproject.toml:
uv sync
```

Verify the signal-processing surface is present:

```bash
uv run python -c "
import librosa, pyloudnorm
print('librosa', librosa.__version__, 'pcen', hasattr(librosa,'pcen'), 'A_weighting', hasattr(librosa,'A_weighting'))
print('pyloudnorm', pyloudnorm.__version__)
"
```

## Fast development loop

`conftest.py` eagerly imports ~22 heavy modules, so use `--noconftest` for unit-only
iteration on pure functions:

```bash
uv run pytest --noconftest src/tests/audio/workflows/audio_analysis/noise_floor_test.py -q
```

Run the full suite (with conftest) before pushing, plus the lint gates:

```bash
uv run pytest -n auto
uv run ruff format && uv run ruff check && uv run mypy . && uv run codespell
```

---

## Phase A — pin level sensitivity (US2)

**Runs first: unblocked, cheap, and its output feeds the margin derivation.**

```bash
# Probe the installed classifiers. Cached models only — never downloads.
uv run python scripts/probe_classifier_levels.py \
    --input src/tests/data_for_testing/english_conversation_higgs_audio_v2.wav \
    --gains-db -40 -20 -10 0 10 \
    --out artifacts/level_probe/
```

**Expected** (`artifacts/level_probe/level-verdicts.json`): both classifiers report
`"verdict": "level_sensitive"`. This is the measured finding, not a hypothesis — a
`self_normalizing` verdict means the probe is wrong, not that the model changed.

Validate:

```bash
# SC-005: gain range spans >= 30 dB, verdict per classifier with window length
uv run python -c "
import json; d=json.load(open('artifacts/level_probe/level-verdicts.json'))
for v in d['verdicts']:
    lo,hi=v['gain_range_db']; assert hi-lo>=30, v['classifier']
    assert v['window_length_s'] and v['mechanism_source']
    print(v['classifier'], v['verdict'], v['low_level_floor_dbfs'], 'dBFS')
"
# FR-017b: the regression guard must pass offline
uv run pytest src/tests/audio/tasks/classification/level_probe_test.py -q
```

**Score-comparability fix (FR-017c).** Verify a secondary source is no longer crushed:

```bash
uv run pytest src/tests/audio/workflows/audio_analysis/sound_sources_test.py -q -k comparab
```

---

## Phase B — background mask (US4)

```bash
# Speech task (default target set)
uv run python scripts/analyze_audio.py --input <clip>.wav --task-type speech

# Breathing task — the decisive case (SC-024)
uv run python scripts/analyze_audio.py --input <breath_clip>.wav \
    --task-type breath --mask-introspect
```

**Expected**: `<run_dir>/<pass>/background_mask.json` with
`"metadata_provenance": "recognized"`, three-state regions in
`background_mask.parquet`, and — for the breathing task — **zero** target breaths in
`background_sources.parquet`.

Validate:

```bash
uv run python - <<'PY'
import json, pandas as pd, glob
mj = glob.glob('artifacts/analyze_audio/*/raw/background_mask.json')[-1]
m = json.load(open(mj))
assert m['metadata_provenance'] in ('recognized','fallback')          # SC-025
assert 'total_masked_s' in m and 'masked_fraction' in m               # SC-021
df = pd.read_parquet(mj.replace('.json','.parquet'))
assert set(df.state) <= {'target_free','target_active','indeterminate'}  # SC-019
assert df.uncertainty.between(0,1).all()
print('masked', m['total_masked_s'], 's', round(m['masked_fraction']*100,1), '%')
PY
```

**SC-022** (empty mask): run on a clip of continuous target activity and confirm
`"is_empty": true` with the limitation stated, not the field omitted.

---

## Phase C — noise floor, margins, suppression (US3)

Derive the margin profile first — it must not be hand-written:

```bash
uv run python scripts/calibrate_detection_margin.py \
    --level-verdicts artifacts/level_probe/level-verdicts.json \
    --out src/senselab/audio/workflows/audio_analysis/data/detection_margin/2026-07-29.json
```

Then run with suppression enabled (opt-in per FR-030):

```bash
uv run python scripts/analyze_audio.py --input <clip>.wav \
    --task-type speech \
    --foreground-suppression \
    --detection-margin-profile detection-margin/2026-07-29
```

Validate the guards — these are the tests that matter most, because each one blocks a
fabricated finding:

```bash
uv run python - <<'PY'
import pandas as pd, glob, json
run = sorted(glob.glob('artifacts/analyze_audio/*'))[-1]
src = pd.read_parquet(f'{run}/raw/background_sources.parquet')
# SC-014: every finding carries its margin
assert src.above_floor_db.notna().all()
# SC-008: human-sound categories from the suppressed variant carry leakage
hs = src[(src.category.isin(['speech','people'])) & (src.variant=='foreground_suppressed')]
assert hs.leakage_margin_db.notna().all()
# SC-031 / SC-033: excised results carry padding and are never conflated with grid
ex = src[src.computed_on=='excised']
assert ex.padding_fraction.notna().all()
assert set(src.computed_on) <= {'grid','excised'}
# FR-021d: the floor is bias-corrected
nf = pd.read_parquet(f'{run}/raw/noise_floor.parquet')
assert (nf.bias_correction_db > 0).all()
# SC-016: suppression depth always reported
s = json.load(open(f'{run}/raw/suppression.json'))
assert not s['requested'] or s['achieved_depth_db'] is not None
print('findings', len(src), '| bands', len(nf), '| depth', s.get('achieved_depth_db'))
PY
```

**SC-018 — the false-positive test.** Amplified pure noise floor must yield zero findings:

```bash
uv run python -c "
import numpy as np, soundfile as sf
rng = np.random.default_rng(0)
sf.write('/tmp/noise_floor.wav', (rng.standard_normal(16000*10)*1e-5).astype('float32'), 16000)
"
uv run python scripts/analyze_audio.py --input /tmp/noise_floor.wav --foreground-suppression
# expect background_sources.parquet with 0 rows
```

**SC-015 — the decisive test.** Two clips identical except one contains a faint
background source. The reported categories must differ. A 30 dB-suppression baseline
fails this by construction (research D6), so a *pass* here is the evidence that the
pipeline detects content rather than reporting residual foreground.

---

## Phase D — influence guards (infrastructure, must precede Phase E)

```bash
uv run pytest --noconftest src/tests/audio/workflows/audio_analysis/adaptive/influence_test.py -q
```

Must cover:

- **SC-027** — a value revised by influence records `resolution_kind: "revision"`, and its
  uncertainty drop is not reported as improved confidence.
- **SC-028** — a constructed oscillation terminates with
  `termination_reason: "oscillation"` and `converged: false`.
- **SC-030** — a `derived` signal alone cannot drive a revision an `independent` signal
  contradicts.
- **SC-029** — two identical runs produce byte-identical outputs.

---

## Phase E — per-speaker identity (US1) — **blocked on #537**

Do not start until #537 lands. It edits `identity.py`, `clustering.py`, `stages.py`, and
`stage_context.py`, and adds four diarizers.

```bash
git fetch origin && git log --oneline origin/alpha -1   # confirm #537 merged
git rebase origin/alpha
```

Then:

```bash
uv run python scripts/analyze_audio.py --input <multi_speaker_clip>.wav --per-speaker-identity
```

Validate:

```bash
uv run python - <<'PY'
import json, glob, pandas as pd
run = sorted(glob.glob('artifacts/analyze_audio/*'))[-1]
sp = json.load(open(f'{run}/final/speakers.json'))
cp = sp['count_posterior']
assert abs(sum(cp['probabilities'].values()) - 1.0) < 1e-9
assert set(cp['support']) <= set(cp['probabilities'])                    # FR-006
for s in sp['speakers']:
    assert set(s['source_kinds'].values()) <= {'independent','derived'}  # FR-007
    for r in s['revisions']:
        assert r['resolution_kind'] in {'new_evidence','revision','unresolved'}
presence = pd.read_parquet(f'{run}/final/per_speaker_presence.parquet')
for sid in {s['speaker_id'] for s in sp['speakers']}:
    assert (presence.speaker_id == sid).any()                                # SC-003
print('counts', cp['probabilities'], 'multimodal', cp['is_multimodal'])
PY
```

**The motivating case.** On `audio_48khz_mono_16bits` two diarizers reported one speaker
while embedding clustering reported five, on a clip that plausibly contains four talkers
("This is Peter / This is Johnny / Kenny and / and Joe"). Expected outcome:
`is_multimodal: true` with `support` naming which source backed each count — **not** a
collapse to either answer. The spec deliberately does not require resolving it in a
particular direction, so this validates representation, not accuracy.

---

## Reproducibility

```bash
# SC-004 / SC-029 — byte-identical across reruns
uv run python scripts/analyze_audio.py --input <clip>.wav --output-dir /tmp/run_a
uv run python scripts/analyze_audio.py --input <clip>.wav --output-dir /tmp/run_b
diff <(cd /tmp/run_a && find . -type f | sort) <(cd /tmp/run_b && find . -type f | sort)
for f in final/speakers.json final/convergence.json; do
  cmp /tmp/run_a/$f /tmp/run_b/$f && echo "OK $f"
done
```

`generated_at`-style timestamps are the only permitted difference; if any other file
differs, iteration order is non-deterministic (FR-011f).

## Cache

`CACHE_SCHEMA_VERSION` is bumped in this feature, so the first run after these changes
re-computes rather than reusing stale entries. That is intended — cache invalidation is
free, and two of the changes (score aggregation, amplify-before-serialize) alter outputs.

## Known-fragile validation points

- The **breathing/cough task test (SC-024)** needs a clip whose target is a non-speech
  vocal event. If none exists in `src/tests/data_for_testing/`, one must be synthesized —
  and note that a synthesized breath may not exercise the real failure mode.
- **Derived floor statistics** (bias correction, per-bin sigma) are unvalidated synthesis;
  validate on synthetic noise before trusting a threshold that depends on them
  (research.md open risk 2).
- **`FR-021h` activity-conditioned floors** have no published precedent. Compare against
  an unconditioned floor on the same recording and confirm the conditioned version does
  not systematically over-gate quiet stretches.
