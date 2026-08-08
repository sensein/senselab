# Speech-to-text ensemble (transcript fusion)

Combine word-timestamped transcripts from multiple ASR systems into a single
consensus transcript with per-word confidence and alternates (ROVER-style
time-slot voting). Promoted from the adaptive uncertainty workflow
(spec `20260723-225523-dynamic-uncertainty-workflow`, architecture-review T050);
model-independent and dependency-free.

- `fuse_word_streams(word_streams, weights=...)` — time-slot voting; `weights`
  lets callers down-weight correlated systems (e.g. senselab's model-family
  weights) or weight by validation accuracy.
- Per-word `corroboration` in `[0, 1]` — optional external evidence that something was
  said there, supplied by the caller. It enters the vote weight *and* the coverage
  term (for a one-member slot `share` is identically 1.0, so vote weight alone would
  be a no-op on exactly the words the mechanism exists for) and is floored at
  `MIN_CORROBORATION`: an uncorroborated word is attenuated, never dropped. Absent or
  `None` means *unmeasured* and applies no discount. Alternates are gated on a second,
  uncorroborated tally, so attenuation can decide who wins but never who is recorded.
- `load_calibrator(profile)` — logistic / piecewise confidence calibration maps.
- `iter_word_leaves(node)` — recursive word-leaf extraction from serialized
  `ScriptLine` trees (dicts); for `ScriptLine` *instances* use
  `ScriptLine.iter_leaves()`.
