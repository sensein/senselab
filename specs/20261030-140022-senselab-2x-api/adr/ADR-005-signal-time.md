# ADR-005: Signal and time model

**Status:** accepted, 2026-10-30.

## Context
Architect review §7 (R-43): a `Signal` was one array with one sampling rate, spans had no units,
and time maps came only from synchronisation. That cannot cover mixed-rate wearables,
variable-frame-rate video or long recordings.

## Decision
- **The data model is protocol-shaped.**
  - Tasks accept protocols, not concrete classes: `RegularSeries`, `IrregularSeries`,
    `FrameSequence`, `Text`.
  - Concrete types (`Audio`, `ECG`, `Video`, …) implement them and validate units and layout.
- **A `Recording`** holds named **channel groups**.
  - Each group is regular (its own sampling rate) or irregular (per-sample timestamps).
  - A wearable with ECG at 250 Hz and accelerometry at 50 Hz is one recording with two groups.
- **Time** is integer nanoseconds on a named `clock_id`.
  - A `TimeAnchor` ties a clock to absolute time (it may be date-shifted for privacy).
  - A `TimeMap` (offset plus drift, or piecewise, with uncertainty) relates two clocks. It can come
    from metadata or from a synchronisation task, and is used as a declared input (ADR-002).
- **Spans** carry `{start_ns, end_ns, clock_id}`. Durations and values carry units.
- **Video** with a variable frame rate is a `FrameSequence` with frame timestamps.
- **Long recordings** are accessed lazily, by time range and in chunks. Identity is chunked
  (ADR-002), so a multi-day recording is never hashed or loaded whole.

## Consequences
- Generic tasks (P-26) are written against the protocols and run on any group that implements
  them.
- Integer time removes float drift in long recordings.
- Frame-rate and sample-rate conversions are explicit tasks.

## Alternatives rejected
- **One class with a single rate.** It cannot express mixed-rate recordings.
- **Float seconds.** Precision loss over days.

## Note 2026-11-01: where whole-file hashing lives

Stamping a file on read hashes every byte, and that now lives in the runtime (`runtime/audio_io.py`), not the
kernel. It still conflicts with "never hashed whole" above; that conflict remains R-109 and is decided in
Phase 2.

## Amendment 2026-11-01: chunked stamps resolve the whole-file conflict

- A source stamp records the per-chunk sha256 digests of the file on the `content_identity` grid (64 MiB) beside
  their root, which stays the recording's identity. Taking a stamp hashes the file once, to take its identity; a
  windowed read then verifies only the chunks the decoder reads, each read whole, hashed and compared before any of
  its bytes is decoded, and `check` of a stamp taken elsewhere verifies the same way, reading only the chunks the
  sound header lies in. A `cp -p` rewrite inside a chunk a read touches still raises `StaleSource`; one in a chunk it
  does not touch is caught by the next read that touches it, or by `check` of a stamp this process took (its ctime).
  The stamp change bumps the signal wire formats to `recording@2.0`, `audio@2.0` and `regular-group@2.0` (ADR-006).
  This closes the note above (R-109, P2-21; Phase 2 Task 11b).

## Amendment 2026-11-02: NaN in signals (H-R38)

This states in the ADR a rule that `design.md` §2 held alone (R-130).

- No signal holds an infinite sample.
- A regular group, audio included, refuses NaN when built and when decoded: a NaN in audio is a producer defect
  (0/0 on silence).
- An irregular group holds NaN as its missing-sample marker (pose and physiological dropouts), as the one quiet
  NaN of ADR-006. `Events.value` admits NaN.

## Amendment 2026-11-02: decoded samples and their stamp (Phase 2 Task 34a)

- Held samples and a file's stamp are exclusive unless the reader issues both: `RegularGroup.decoded_from(group,
  reader)` decodes a source group and holds the samples with the stamp and `decoded`, their `npy@1` digest. Only such
  a group is identified as its file is; validation checks the held samples still have `decoded`. Otherwise a held
  group's identity comes from its samples (R-171). Signal wire formats `@2.1` (ADR-006).
