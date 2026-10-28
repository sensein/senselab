# SQUIM over 10 s windows in the general feature-extraction API

Issue: sensein/senselab#614. Follows PR #615, which windowed SQUIM inside triage's `task_squim`.

## Decision

`torchaudio_squim.extract_objective_quality_features_from_audios` reads SQUIM's objective head over
fixed windows of a span of each recording and reports the median over windows. The default window is
10 s. `window_s=None` keeps a single pass over the span. This is a breaking change: the result gains
`windows`, `windows_n` and `span_s`, and the default values are now medians over windows instead of one
pass over the whole waveform. No compatibility path is kept (pre-1.x major release).

```python
extract_objective_quality_features_from_audios(
    audios, window_s=10.0, min_window_s=3.0, start_s=None, end_s=None, floor_s=1.0, device=None
)
```

## Why windows

A comparison over 10,554 recordings (#614, #615) found three problems with one whole-extent pass:

- it includes non-speech;
- runtime grows as length^1.74: 232 s and 12 GB of memory at 120 s of audio, and about 1,100 s and
  31 GB at 300 s;
- it tracks quality covariates slightly worse than a median over 10 s windows.

A window bounds the per-call cost, so runtime grows linearly with length.

## Window plan

`plan_windows(start_s, end_s, window_s, min_window_s, floor_s, short_tail)` is the one implementation.
Full windows tile from the span start. A trailing remainder of at least `min_window_s` is its own
window. A shorter remainder is folded into the previous window (`short_tail="fold"`, the general
function) or dropped (`short_tail="drop"`, triage). A span shorter than `min_window_s` is one window,
and one shorter than `floor_s` gets none. Windows are cut by sample index: the start rounds down and the
end rounds up, clamped to the waveform, the rule triage's `sample_bounds` uses.

The general function folds where triage drops. Triage's published `task_squim` values were computed
with the drop rule, and they must not change. Both policies share every other rule.

## Floor

The default `floor_s` is 1.0 s, the same as triage's `data/task_squim.yaml` `floor_s`. Its derivation
is in that file. In the `specs/20260817-triage-workflow-dag/benchmarks/squim.md` table, 1.0 s is the
shortest span length, and a 0.9 to 1.6 s speech span already reads as speech. A span under the floor
returns NaN for all three heads with `windows_n == 0`.

`floor_s` is a parameter because three callers deliberately read sub-second spans one pass at a time,
and their values must not change: triage `PREPROCESS` and `SPEECH` per-span SQUIM, and the
`audio_analysis` temporal features. Those callers pass `window_s=None, floor_s=0.0`. Triage's
`task_squim` scorer passes the same values, because it plans and cuts its windows itself.

## Callers

- `features_extraction.api.extract_features_from_audios`: `torchaudio_squim` accepts a dict of the keys
  above. The default is the windowed behaviour, and `device` is now forwarded.
- `explore_conversation`: passes a dict `torchaudio_squim` config through instead of coercing it to a
  bool.
- `features_extraction.temporal` (analyze_audio): pins one pass per sliding window. Its rows gain
  `windows_n` and `span_s`, and `CACHE_SCHEMA_VERSION` is bumped 25 -> 26.
- triage `task_squim`: `task_windows` delegates to `plan_windows(short_tail="drop")`. A test checks the
  windows against the previous implementation on a grid of extents. It also checks the measurement's
  windows and medians against the values the pre-refactor code wrote on a synthetic chirp.
