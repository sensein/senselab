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
    audios, window_s=10.0, min_window_s=3.0, start_s=None, end_s=None, floor_s=0.0, device=None
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

The default `floor_s` is 0 (owner decision, 2026-10-28): every span the model accepts gets a value,
single words included. Values from short spans vary more. In the #614 reporter's comparison, medians
over speech-only spans, mostly under 0.5 s, agreed with medians over 10 s windows only at Pearson
r ≈ 0.55–0.76. Callers that want a cut-off set `floor_s`. A span under it returns NaN for all three
heads with `windows_n == 0`.

Triage's `task_squim` keeps its own `floor_s = 1.0` from `data/task_squim.yaml`. It applies that floor in
`plan_windows` and then reads each planned window with `window_s=None` at the default floor. Its windows
and medians are unchanged; the identity tests in `task_squim_test.py` check this.

An empty span (`start_s == end_s`, or an empty audio) raises `ValueError`, like any span outside
`0 <= start_s < end_s <= duration`. NaN is kept for "read nothing by request" (under `floor_s`), and an
empty span is a caller error. The model raises `RuntimeError` on inputs shorter than its 64-sample
kernel (4 ms); that error propagates as before.

The single-pass callers pass only `window_s=None`. These are triage `PREPROCESS` and `SPEECH` per-span
SQUIM, analyze_audio's temporal features and the triage `task_squim` scorer. At the default floor that
is the previous whole-waveform call for any non-empty input. A zero-sample clip in triage's per-span
readings is now recorded as `unmeasured: ValueError` where the model's `RuntimeError` was recorded
before.

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
