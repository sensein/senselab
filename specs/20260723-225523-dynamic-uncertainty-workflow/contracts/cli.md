# Contract: CLI (`scripts/analyze_audio.py`)

All new flags default to values that preserve today's artifact set (FR-024). Existing flags unchanged.

## New flags

| Flag | Default | Meaning |
|---|---|---|
| `--max-rounds N` | `3` | Total rounds incl. triage+baseline. `1` = baseline only: no triage gating, no interventions, no `rounds/`≥2, `final/` still emitted from round-1 belief. |
| `--enhancement {auto,always,never}` | `always` | C1. `auto` gates the enhanced pass on triage quality; `always` = today's behavior; `never` ≡ existing `--no-enhancement` (kept as alias). |
| `--policy PATH` | packaged `default.yaml` | Policy file (FR-027). Overrides below win over the file. |
| `--budget-medium N` / `--budget-heavy N` | from policy | Per-run intervention budgets (FR-018). |
| `--max-region-rounds N` | from policy | Per-region intervention cap (FR-017). |
| `--region-top-n N` | from policy | Regions per round (FR-010). |
| `--reserve-asr-models M [M…]` | from policy | U2 escalation pool. |
| `--enable-overlap-separation` | off | v2 U4 rule (heavy). |
| `--no-adaptive-outputs` | off | Suppress `rounds/` + `final/` (debug/regression aid). |

## Semantics & compatibility

- **Golden-compat mode**: `--max-rounds 1 --enhancement always` → stage execution order, cache keys,
  task JSONs, 9 uncertainty parquets, LS bundle, disagreements.json, summary.json pre-existing keys all
  match today's outputs (SC-005). New outputs (`rounds/1/`, `final/`, additive summary keys, additive
  parquet columns) appear unless `--no-adaptive-outputs`.
- **Cache-key stability**: triage and baseline reuse the existing task names and params in `cache_key`
  (`scripts/analyze_audio.py:786`) — a triage AST/quality result is the same cache entry the baseline
  would create; adaptive runs share cache with legacy runs of the same audio.
- **`--skip` interaction**: `--skip comparisons` disables the belief store and therefore all rounds ≥ 2
  and `final/` (a warning states this). Skipping a task removes its voters, nothing else.
- **Exit codes**: unchanged (0 success incl. `no_speech`; 2 usage/model-unavailable errors, e.g. the
  existing brouhaha hard-fail stays).
- **Progress output**: each round prints one summary line (regions, fired/deferred, budget) — no
  per-bucket spam.

## Examples

```bash
# Full adaptive run, default policy
uv run python scripts/analyze_audio.py audio.wav

# Today's exact behavior + adaptive artifacts suppressed
uv run python scripts/analyze_audio.py audio.wav --max-rounds 1 --enhancement always --no-adaptive-outputs

# Cost-capped unattended batch
uv run python scripts/analyze_audio.py audio.wav --enhancement auto --budget-medium 8 --budget-heavy 0

# Deep interrogation of one hard file
uv run python scripts/analyze_audio.py audio.wav --max-rounds 5 --region-top-n 16 \
    --reserve-asr-models openai/whisper-large-v3-turbo ibm-granite/granite-speech-3.3-8b
```
