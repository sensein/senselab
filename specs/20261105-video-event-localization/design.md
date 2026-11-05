# Video event localization: design and measurements

`senselab.video.tasks.event_localization` finds when labelled events occur in a recording that is longer than a
video-language model can take in one request. It owns the model-independent parts; the media cut and the model call stay
with the caller (`ask(window) -> reply`).

## Why windows with absolute offsets

Qwen3-Omni encodes time as position ids tied to the clip it is given: 13 position ids per second for audio and video, the
video temporal grid index times `second_per_grid` (temporal patch 2 over the container fps) times 13, and an interleaving
chunk of 2 s for audio-video (checkpoint `config.json`, `modeling_qwen3_omni_moe.get_rope_index`). The prompt carries no
wall-clock, and the clock restarts at 0 for every request, so a reply's times are relative to the window and the caller must
add the window start. The window therefore has to start exactly at its planned time (stream copy can only cut on key
frames) and must carry the container's true frame rate. The engine's cap of 25,088 video embedding tokens per item is about
34 s of 720p30 at the server's own pixel budget, so windows are 30 s with a 10 s overlap by default; a refused window is
halved (`WindowTooLarge`).

## Closed label set and constrained decoding

`segment_list_schema` expresses the label set as a JSON-schema enum so a server with structured outputs can only emit
labels from the set. Range rules (times inside the window, end after start) are not in the schema because grammar backends
differ in what they support; `parse_reply` enforces them and counts what it drops (`bad_label`, `bad_item`, `bad_order`,
`clamped`), what it repairs (`folded` label spelling, `salvaged` objects of a reply cut off by the token limit) and what it
cannot read (`unparsed`). `strict=True` accepts only exact labels and numeric times, for a server that does not enforce the
schema.

## Merge and score

Per label, segments of all windows that overlap or lie within `gap_s` (default 1 s) are chained. Generated segments carry no
probability, so the score is the fraction of the windows covering the merged segment that contributed to it; it only ranks
segments seen twice above segments seen once, and mAP computed from it has many ties. Segment and point-event precision,
recall and F1 at a fixed tIoU or tolerance are the primary readout.

## Measurements (synthetic clips, Qwen3-Omni-30B-A3B-Instruct, vLLM 0.32, greedy)

24 synthetic 90 s clips (25 and 30 fps), five known events each, window 30 s, stride 20 s. Events matched at tIoU 0.5 of 20
per condition:

| Clip and event asked for | Matched | Median start error |
|---|---|---|
| red square on silent video, asked for the square | 20 | -0.2 s |
| square plus tone together, asked for the square | 16 | -0.1 s |
| tone on a static picture, asked for the tone | 0 (no prediction) | none |
| square plus tone together, asked for the tone | 2 | -1.2 s |
| the same four with a burned-in `t=` clock | 19, 18, 0, 6 | 0.0, 0.0, none, -0.2 s |

A window cut was checked separately: every window held exactly its source frames and a tone sat at its expected
window-relative time to within 0.1 s. Visual events localize; audio-only tone events do not, so a caller that needs sound
events should not rely on this task alone.
