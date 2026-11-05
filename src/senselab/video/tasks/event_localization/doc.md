# Event localization

Finds *when* labelled events occur in a long recording by asking a video- or audio-language model about overlapping
windows, and scores the result against annotated segments. The model call and the media cutting stay with the caller; this
module holds the parts that do not depend on a model.

```python
from senselab.video.tasks.event_localization import build_prompt, localize_events, segment_list_schema

labels = ["hands flapping", "jumping", "rocking", "spinning"]
family = {"task": "List every interval in which the child shows one of the behaviours below.",
          "labels": {l: "definition" for l in labels}}
schema = segment_list_schema(labels)          # JSON schema with `label` as an enum, for constrained decoding


def ask(window):                              # window = absolute (start, end) in seconds
    clip = cut(video, *window)                # caller's media cutting
    return model(clip, build_prompt(family, window[1] - window[0]), json_schema=schema)


result = localize_events(duration_s, ask, labels, win_s=30, stride_s=20)
result.segments       # [{"label", "start_s", "end_s", "score", "n_windows", ...}] on the recording's time axis
result.counts()       # parse counts: unparsed, salvaged, bad_item, bad_label, folded, bad_order, clamped, empty
```

* `plan_windows` / `split_window` / `shift_segments`: windows that cover the recording with a fixed overlap, halving of a
  refused window, window-relative to absolute times.
* `segment_list_schema` and `parse_reply`: the closed label set as a JSON-schema enum, and validation of a reply (labels
  outside the set, bad or reversed times, clamping, a reply cut off mid-list). `strict=True` accepts only exact labels.
* `merge_windows`: per label, segments that overlap or lie within `gap_s` are chained; the score is the fraction of the
  covering windows that saw the segment.
* `metrics`: mAP over tIoU thresholds (ActivityNet / OpenTAD convention), segment and point-event precision / recall / F1,
  per-label presence with prevalence, balanced accuracy and MCC, repeat-count error, variant confusion, shuffled and
  whole-recording floors, `rasterize` for frame-wise protocols.

Design, measurements and the window time-axis findings: `specs/20261105-video-event-localization/design.md`.
