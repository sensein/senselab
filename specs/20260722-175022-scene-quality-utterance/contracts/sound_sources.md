# Contract: sound-source categorization

## A. Category map — `workflows/audio_analysis/data/audioset_source_map.json` (new)

Schema in `data-model.md §3`. Categories `{speech, people, machine, environment}`, default `environment`, keys = AudioSet display names covering AST (527) ∪ YAMNet (521).

**Invariant (SC-003)**: every emittable class maps to exactly one category. Enforced by test (see D).

## B. Enabling full distributions (mandatory prerequisite)

Today `_classify_windowed` truncates to `top_k=5`. The AST + YAMNet call sites in `scripts/analyze_audio.py` must pass the full label count:

```python
classify_audios(..., win_length=w, hop_length=h, top_k=scene_top_k)   # scene_top_k default = full (527 / 521)
```

- New CLI param `--scene-top-k` (default: full per model). Top-1 consumers (`speech_presence_labels`, YAMNet veto) are unaffected — they read `labels[0]`.
- Stored per-window dict is unchanged in shape (`labels`/`scores` parallel arrays), just longer.

## C. Harvester — `workflows/audio_analysis/sound_sources.py` (new)

```python
def harvest_source_categories(
    *,
    pass_summary: dict[str, Any],   # reads pass_summary['ast'], pass_summary['yamnet']
    grid: BucketGrid,               # presence reporting grid
    category_map: SoundSourceCategoryMap,
) -> list[dict[str, Any]]:
    """One dict per presence bucket:
    {'start','end', src_speech, src_people, src_machine, src_environment,
     src_dominant, '_raw': {per-category class contributions}}."""
```

### Rules

- For each classifier window overlapping a bucket, sum `scores` into category masses via `category_map`; combine AST + YAMNet (mean of available; YAMNet-authoritative tie-break consistent with existing `_speech_window_mask` policy is **not** required here — masses are additive, not vetoed).
- Normalize the four masses to sum ≈ 1 per bucket; `src_dominant = argmax`.
- Project classifier windows onto buckets with the existing center→nearest-window logic (`presence.py:335`).
- Both classifiers absent → all `src_*` = `None` (FR-023).
- Unmapped class → `default` category + one logged warning per unique class (not per occurrence).

## D. Coverage test (SC-003)

`sound_sources_test.py::test_category_map_covers_all_classes`:
- Load AST `id2label` (`AutoConfig.from_pretrained(AST_ID).id2label`) and the YAMNet class list (vendored copy or model asset).
- Assert `set(classes) ⊆ set(map.keys())` OR every class resolves (mapped or default) with **zero** silent gaps, and each maps to exactly one of the 4 categories.
- Assert masses from a synthetic window sum to 1.
