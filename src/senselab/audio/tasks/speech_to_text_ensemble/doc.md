# Speech-to-text ensemble (transcript fusion)

Combine word-timestamped transcripts from multiple ASR systems into a single
consensus transcript with per-word confidence and alternates (ROVER-style
time-slot voting). Promoted from the adaptive uncertainty workflow
(spec `20260723-225523-dynamic-uncertainty-workflow`, architecture-review T050);
model-independent and dependency-free.

- `fuse_word_streams(word_streams, weights=...)` — time-slot voting; `weights`
  lets callers down-weight correlated systems (e.g. senselab's model-family
  weights) or weight by validation accuracy.
- `load_calibrator(profile)` — logistic / piecewise confidence calibration maps.
- `iter_word_leaves(node)` — recursive word-leaf extraction from serialized
  `ScriptLine` trees (dicts); for `ScriptLine` *instances* use
  `ScriptLine.iter_leaves()`.
