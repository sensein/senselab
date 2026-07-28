# Quickstart: Uncertainty-driven adaptive analysis workflow

## Run it

```bash
# Default adaptive run (triage → baseline → ≤1 intervention round → fusion)
uv run python scripts/analyze_audio.py path/to/audio.wav

# Legacy single-shot (golden-compat mode)
uv run python scripts/analyze_audio.py audio.wav --max-rounds 1 --enhancement always
```

## What to look at, in order

1. `final/convergence.json` — `run_state`, per-axis converged/irreducible counts, budget, and
   `next_actions` (what more budget would buy).
2. `final/transcript.json` — the fused answer: words with speaker, confidence, alternates.
3. `final/iterations.json` — every decision the loop made (fired, deferred, blocked, failed) with
   trigger values and post-hoc uncertainty deltas.
4. `rounds/<k>/` — per-round belief parquets and region proposals when debugging a decision.
5. Existing artifacts (9 uncertainty parquets, LS bundle, `disagreements.json`, `timeline.png`) —
   unchanged locations; the LS bundle gains `final__*` consensus tracks and
   `final/disagreements_resolved.json` shows which round-1 disagreements the loop resolved.

## Validation recipes

```bash
# 1. Determinism (SC-004): identical runs → identical decision logs
uv run python scripts/analyze_audio.py audio.wav --output-dir /tmp/a
uv run python scripts/analyze_audio.py audio.wav --output-dir /tmp/b
diff <(jq -S . /tmp/a/*/final/iterations.json) <(jq -S . /tmp/b/*/final/iterations.json)  # empty

# 2. Golden compat (SC-005): legacy mode vs pre-feature golden run
uv run python scripts/analyze_audio.py audio.wav --max-rounds 1 --enhancement always \
    --no-adaptive-outputs --output-dir /tmp/compat
# then compare per-task JSONs + 9 parquets against the checked-in golden manifest

# 3. Targeted improvement (SC-001): inject 3 s of noise, watch the loop attack it
uv run python - <<'PY'
import torch, torchaudio
w, sr = torchaudio.load("clean.wav"); s, e = int(5.0*sr), int(8.0*sr)
w[:, s:e] += 0.3*torch.randn_like(w[:, s:e]); torchaudio.save("degraded.wav", w, sr)
PY
uv run python scripts/analyze_audio.py degraded.wav --enhancement auto --max-rounds 3
jq '[.entries[] | select(.status=="fired")] | map({rule, region_id, delta})' \
    <run_dir>/final/iterations.json   # expect U1/U2/S1 on a region covering ~5–8 s

# 4. Triage gates (SC-002/003)
uv run python scripts/analyze_audio.py clean.wav --enhancement auto     # no enhanced_16k/ dir
uv run python scripts/analyze_audio.py silence.wav                       # run_state == "no_speech"
```

## Reading a region's story

```bash
RID=r2_utterance_0
jq --arg r "$RID" '.entries[] | select(.region_id==$r)' <run_dir>/final/iterations.json
python - <<'PY'
import pandas as pd
df = pd.read_parquet("<run_dir>/rounds/3/belief/utterance.parquet")
print(df[df.status != "open"][["start","end","aggregated_uncertainty","epistemic",
                               "aleatoric_floor","status","irreducible_reason","round"]])
PY
```

An `irreducible: overlapping_speech` region with `aleatoric_floor ≈ residual` is the loop saying:
models disagree because two people are talking at once — more models will not fix this; separation
(v2 `--enable-overlap-separation`) or human review will.

## Development order

Phases in [plan.md](./plan.md): A (harvest/aggregate split, behavior-neutral) → B (triage + gating) →
C (loop core) → D (identity/overlap + fusion) → E (v2). Run `/speckit.tasks` to generate tasks.md.
