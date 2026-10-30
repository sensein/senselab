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
