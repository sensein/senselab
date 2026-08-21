# senselab audio I/O audit — 2026-08-19

Read-only investigation. No repo changes; nothing filed upstream.

| file | contents |
|---|---|
| `upstream.md` | **Part 1.** Upstream status: torchcodec #1576 (clamping — acknowledged, intended, closed `not_planned`, with the maintainer's verbatim rationale); the unreported bit-depth gap; the active chunking/seeking workstream (#1448/#1449/#1601/#1610/#1614/#1645); pytorch/audio #4211; the torchaudio I/O position; and a measured **0.11.1 vs 0.16.0** comparison. |
| `matrix.md` | **Part 2a.** The format matrix: what each of 4 encoders writes for 11 targets, extension→codec maps, input-type constraints, float→int16 rounding, round-trip exactness per cell, out-of-range behaviour, sample-rate/channel/length/metadata preservation, video-container support, and senselab's measured end-to-end behaviour. |
| `invariants.md` | **Part 2b + Part 3.** The decode side: source-independence, no normalisation (whole-file *and* per-segment at 60 dB), range-transparency on read, and chunk-equals-slice per format with max abs differences. Ends with the invariant list a test can pin, and the recommendation argued against the alternatives. |
| `torchcodec-issue-draft.md` | Two **drafts, not filed**: (A) the unreported MP3 1105-sample timeline disagreement; (B) the unreported request for sample-format control. Explains why the clamping issue must *not* be re-filed. |
| `scripts/` | Reproduction scripts, runnable via `uv run --project <senselab>`. |
| `raw/` | Raw outputs: `matrix.json`, the printed matrices, the three invariant runs, and the 0.16.0 comparison. |

## The four things that most changed the picture

1. **`torchaudio.save`/`load` are not a fallback — they are torchcodec.** In 2.11.0 both import
   from torchcodec and raise `ImportError` without it, and `torchaudio.info`,
   `list_audio_backends`, `io`, `backend` and `AudioMetaData` are gone. senselab's
   "torchcodec else torchaudio" branch has no second branch.
2. **Upstream will not add the range check** (#1576, closed `not_planned`, on `O(n)` cost
   grounds). The check must live in senselab — which is what PR #570 does.
3. **Decode is clean and the invariants hold**: one dtype, one amplitude convention, nothing
   normalised at any scope, out-of-range float survives bit-exactly, and chunk == slice
   bit-for-bit for WAV and FLAC. The exception is MP3 through
   `get_samples_played_in_range`: a constant −1105-sample shift, unreported upstream, still
   present on 0.16.0.
4. **The pinned 0.11.1 gets chunked resampling wrong** — `AudioDecoder(sample_rate=N)` read in
   chunks differs from one-go by up to 0.996 in amplitude. senselab does not currently trip it
   (it always decodes at native rate), but it forecloses the obvious speedup. Fixed in 0.16.0 by
   PR #1614 — which in turn introduces an AAC tail discrepancy 0.11.1 did not have.
