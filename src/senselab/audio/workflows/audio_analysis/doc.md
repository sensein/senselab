# `senselab.audio.workflows.audio_analysis`

Three-axis uncertainty for analyze_audio runs. Reads the per-task pipeline outputs
(diarization, ASR, scene classification, alignment, PPG, speaker embeddings) and emits a
single `[0, 1]` uncertainty scalar per bucket on each of three axes:

- **presence_uncertainty** — was there a speaker?
- **identity_uncertainty** — was it the same speaker?
- **utterance_uncertainty** — what was said?

Every model whose output naturally encodes an axis votes (max-inclusive). The vote
collapse is per-axis: Shannon entropy for presence (binary votes); cross-model label
disagreement + cosine across-time for identity; pairwise mean WER + Whisper avg_logprob
+ PPG-vs-ASR phoneme-error-rate for utterance. Sub-signals within each axis fold via the
shared `--uncertainty-aggregator` flag (default `min` over confidences ≡ `max` over
uncertainties).

Output:

- 9 parquets (3 axes × 2 passes + 3 raw_vs_enhanced deltas) — see
  `contracts/uncertainty-row.parquet.md`.
- `disagreements.json` — top-N ranked across all parquets — see
  `contracts/disagreements.json.md`.
- `timeline.png` — 5-row figure (presence / identity / utterance overlaid raw-vs-enhanced
  + delta strip + reference) — see `contracts/ls-bundle.md` for the matching LS tracks.

See `specs/20260508-173136-compare-uncertainty/spec.md` for the full design and
`specs/20260508-173136-compare-uncertainty/quickstart.md` for reviewer recipes.

## Public API

```python
from senselab.audio.workflows.audio_analysis import (
    BucketGrid,
    compute_uncertainty_axes,
    build_disagreements_index,
    build_aligned_timeline_plot,
    attach_uncertainty_tracks_to_ls,
    write_axis_parquet,
)
```

`compute_uncertainty_axes(passes, grid, params, *, audio, speaker_embedding_models,
aggregator)` is the workflow entry point. It is a pure function: callers (such as
`scripts/analyze_audio.py`) handle the surrounding I/O (cache lookup, parquet writing,
disagreements.json + plot, LS bundle extension).
