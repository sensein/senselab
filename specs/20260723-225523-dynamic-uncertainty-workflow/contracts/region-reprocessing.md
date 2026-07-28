# Contract: Region re-processing (crop → run → merge back)

## Crop construction

1. Input: Region with grid-quantized `core_start/core_end` (multiples of the axis reporting hop).
2. `crop_start = core_start − pad_s`, `crop_end = core_end + pad_s`, clipped to `[0, duration]`.
3. Trough snapping: within each pad, if presence p_voice has a local minimum < 0.2, move the crop edge
   outward to that trough (cut in silence, not mid-word). Snapped edges re-quantized to the grid.
4. Extract with `extract_segments([(audio, [(crop_start, crop_end)])])`
   (`senselab.audio.tasks.preprocessing`) from the **elected stream's** 16 kHz mono waveform.

## Caching

- The crop is an `Audio` whose `audio_signature` (`scripts/analyze_audio.py:741`) is content-derived —
  existing `cache_key`/`align_cache_key` work unchanged (FR-014). No cache schema change.
- Grid quantization (step 1/3) guarantees identical regions across rounds/runs produce identical crop
  signatures → cache hits.
- Provenance for crop-scoped outcomes adds: `{"crop": {"start": ..., "end": ..., "pad_s": ...,
  "stream": ..., "region_id": ..., "parent_audio_signature": ...}}`.

## Timestamp mapping

- All model outputs on a crop are crop-local; merge-back adds `crop_start` to every start/end.
- Mapping is exact for word/segment timestamps; frame arrays record `frame_offset_s = crop_start`.

## Merge-back (midpoint rule)

- A word/segment/frame merges into the vote store iff its midpoint ∈ `[core_start, core_end)`.
  Padded-context outputs are discarded (edge-effect quarantine, D2).
- Merged votes get `scope = region:<region_id>` and shadow per contracts/belief-store.md §3.
- A model that returns nothing over the core (silence per that model) merges an explicit
  `{speaks: false}` presence vote — absence of output is evidence, same convention as file scope.

## Minimum-length and applicability rules

| Runner | Min crop (post-pad) | Notes |
|---|---|---|
| ASR (any backend) | 1.0 s | Below this, skip U1/U2 (guard) |
| Forced alignment | same as its ASR text | Only for text-only outputs, unchanged from script |
| segmentation-3.0 / Brouhaha posteriors | 0.2 s | Chunked path already handles arbitrary length |
| Speaker embeddings (fine hop) | 2 × embedding window (default 2.0 s → 4.0 s) | I1 guard |
| YAMNet scene re-check | 0.96 s | Short-crop scene evidence |
| AST | never on crops | 10.24 s native window (`scripts/analyze_audio.py:61-67`) |
| Diarization | never on crops | D3 — global clustering context |

## Failure

A crop-run failure (model error, empty output where text was expected) follows D11: logged in
iterations.json, no vote changes, region's `interventions_remaining` still decremented (the attempt
consumed budget).
