# Quickstart — Reviewer recipes for the comparator outputs

Every recipe assumes you have just run:

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav
```

…with the default config. The comparator stage runs at the end of the pipeline and emits
nine parquets, one ranked `disagreements.json`, one timeline plot, and matching LS-bundle
tracks.

```text
artifacts/analyze_audio/<run_id>/
├── raw_16k/
│   └── uncertainty/{presence,identity,utterance}.parquet
├── enhanced_16k/
│   └── uncertainty/{presence,identity,utterance}.parquet
├── uncertainty/
│   └── raw_vs_enhanced/{presence,identity,utterance}.parquet
├── disagreements.json
├── timeline.png
├── labelstudio_config.xml
└── labelstudio_tasks.json
```

## Recipe 1 — "Where did models disagree on whether someone was speaking?"

Read the presence axis:

```python
import pandas as pd
df = pd.read_parquet("artifacts/analyze_audio/<run_id>/raw_16k/uncertainty/presence.parquet")
df.sort_values("aggregated_uncertainty", ascending=False).head(10)
```

The model_votes column carries each contributor's binary speech-presence vote — useful for
seeing which model is the outlier on a given bucket.

In Label Studio: scrub the `raw_16k__uncertainty__presence` track and look for the `high`
bins (red).

## Recipe 2 — "Where is the speaker identity unclear?"

```python
df = pd.read_parquet("artifacts/analyze_audio/<run_id>/raw_16k/uncertainty/identity.parquet")
df.query("aggregated_uncertainty >= 0.5").head(20)
```

The row-level breakdown:

- `model_votes["pyannote/..."].speaker_label` vs `model_votes["nvidia/diar_sortformer..."].speaker_label`
  — cross-model speaker-label disagreement.
- `model_votes["speechbrain/spkrec-ecapa-voxceleb"].embedding_cosine_to_prev` —
  high values flag a possible speaker change.
- The raw_vs_enhanced/identity parquet adds `raw_vs_enh_disagrees` for "did enhancement
  flip the speaker label here?".

In LS: `raw_16k__uncertainty__identity` and `pass_pair__uncertainty__identity` tracks.

## Recipe 3 — "Where am I least confident in the transcript?"

```python
df = pd.read_parquet("artifacts/analyze_audio/<run_id>/raw_16k/uncertainty/utterance.parquet")
top = df.sort_values("aggregated_uncertainty", ascending=False).head(10)
for _, row in top.iterrows():
    print(f"{row['start']:6.2f}–{row['end']:6.2f}  u={row['aggregated_uncertainty']:.2f}")
    for model_id, vote in row["model_votes"].items():
        if vote.get("text"):
            print(f"  {model_id}: {vote['text']!r}  avg_logprob={vote.get('avg_logprob')}")
```

In LS: `raw_16k__uncertainty__utterance` Labels track + `raw_16k__uncertainty__utterance__text`
TextArea sibling. The TextArea carries the per-bucket transcript consensus + dissenting
model transcripts so you can audit which words drove the WER.

## Recipe 4 — "Did enhancement help on each axis?"

Three raw-vs-enhanced parquets:

```python
import pandas as pd
for axis in ("presence", "identity", "utterance"):
    df = pd.read_parquet(f"artifacts/analyze_audio/<run_id>/uncertainty/raw_vs_enhanced/{axis}.parquet")
    print(f"{axis}: {(df['aggregated_uncertainty'] >= 0.33).sum()} buckets show raw≠enhanced")
```

In LS: the three `pass_pair__uncertainty__*` tracks. Empty stretches mean no
disagreement — i.e. enhancement was a no-op there for that axis.

## Recipe 5 — "Show me the worst N buckets across the whole run"

```python
import json
with open("artifacts/analyze_audio/<run_id>/disagreements.json") as f:
    idx = json.load(f)
for entry in idx["entries"][:10]:
    print(f"#{entry['rank']}  {entry['axis']:9s}  {entry['pass']:18s}  "
          f"{entry['start']:6.2f}–{entry['end']:6.2f}  u={entry['aggregated_uncertainty']:.2f}")
    print(f"   {entry['summary']}")
```

Top-N is set by `--disagreements-top-n` (default `100`, `0` to disable).

## Recipe 6 — "Look at the timeline plot"

```bash
open artifacts/analyze_audio/<run_id>/timeline.png
```

5 rows top-to-bottom:

1. **presence uncertainty** — raw solid + enhanced dashed on `[0, 1]`.
2. **identity uncertainty** — same overlay.
3. **utterance uncertainty** — same overlay.
4. **raw-vs-enhanced delta** — three colored bands, one per axis. High bands = enhancement
   changed the answer there.
5. **reference** — raw diarization speaker bars + raw ASR token spans for context.

## Recipe 7 — "I changed a flag — what re-runs?"

The cache key includes:

```
(audio_signature, axis, pass_set, model_set, comparator_params, wrapper_hash, senselab_version, schema_version)
```

So:

- Re-running with the same flags hits the cache and prints `cache="hit"` per axis.
- Bumping `--cross-stream-win-length` invalidates every comparator parquet (different
  `comparator_params` hash) but leaves the upstream task caches alone.
- Bumping `--uncertainty-aggregator` invalidates the comparator parquets but reuses the
  per-axis intermediate vote dicts (same upstream task results → same model_votes).
- Editing `scripts/analyze_audio.py` invalidates the comparator parquets via
  `wrapper_hash` (per Constitution §V).

## Recipe 8 — "Skip the comparator entirely"

```bash
uv run python scripts/analyze_audio.py path/to/audio.wav --skip comparisons
```

Output is bit-identical to a pre-comparator run: no `<pass>/uncertainty/`,
no `uncertainty/raw_vs_enhanced/`, no `disagreements.json`, no comparator LS tracks.

## Common gotchas

- **PPG is not provisioned** → utterance uncertainty falls back to ASR pairwise WER + Whisper
  avg_logprob only. The `incomparable_reasons` block in `disagreements.json` records this so
  reviewers know not to look for the PER signal.
- **Single-model presence** (you ran with `--asr-models openai/whisper-large-v3-turbo` only
  and dropped diar via `--skip diarization`) → only one binary vote per bucket; entropy is
  always 0; the row is still emitted with `aggregated_uncertainty = 1 − native_confidence`
  (Whisper) or `0.0` if no native signal.
- **Text-only ASR with alignment skipped** (`--no-align-asr`) → Granite / Canary-Qwen
  contribute only at the per-clip level, not per-bucket. Their `text` field on every bucket
  is empty unless the bucket happens to overlap a tokenized chunk.
