# Span rules

## The anchor is local and absolute

Two anchorings were tried and rejected before the current one.

**Global floor, max-normalised envelope.** `floor + 18 dB`, floor = median envelope over
YAMNet-silence windows. Fails under noise: the events keep their positions relative to each other while
the floor climbs, so the gate crosses the quieter event. `|floor| − speech contrast` measured constant
across added noise — 22.1, 22.0, 21.9, 21.6, 20.9, 19.2 dB — while the floor moved −53.5 → −23.1 dB.

**Global peak.** `peak − 25 dB`, on the reasoning that only the floor moves. Rejected: the peak moves too.

| test | result |
| --- | --- |
| peak spread within one recording, sliding 1 s window | 49.1 dB (reference file), 38.4 dB (speech-only file) |
| the same constant across recordings | speech sits 22.1 dB below peak here; on a speech-only file the peak **is** speech |
| one injected 30 ms full-scale click | speech moves 22.1 → **42.7 dB** below the new peak; the gate stops proposing it |
| p99 instead of the maximum, same click | shifts **−15 to −27 dB** |

The last row locates the fault: **normalising the envelope by its own maximum**, which lets one loud
sample rescale everything. Not the statistic.

**Current rule: dBFS envelope, rolling 3 s 10th-percentile floor.** Immune to the transient:

| rule | clean file | + 30 ms click |
| --- | --- | --- |
| `peak − 25 dB`, max-normalised | speech found | **speech lost** |
| local dBFS floor `+ 18 dB` | speech found, IoU 0.89 | **speech found, IoU 0.89** |

## Propose threshold `K`

| `K` | speech detected to | spans on the clean file | survives the click |
| --- | --- | --- | --- |
| 18 dB | +20 dB SNR | 6 | yes |
| 12 dB | +10 dB SNR | 6 | yes |
| 8 dB | +5 dB SNR (merged, IoU 0.10) | **2 — merged** | yes |

Lower is not freely better: at 8 dB the clean-file set collapses by merging. Hence 18 dB for AIRWAY
(events 53–57 dB above the floor) and 12 dB for SPEECH (events ~22 dB lower).

An earlier claim that peak-anchoring bought SNR reach was a **misattribution**: under noise
`peak − 25 dB` sat 5.2 dB lower than `floor + 18 dB`, and the reach came from the lower threshold.

## Onset — peak-anchored

Walk back from the peak to `peak − 15 dB`.

| onset rule | inside the labels' declared onset windows |
| --- | --- |
| Δ above an estimated floor | **2 / 6** |
| peak-anchored, `peak − 15 dB` | **5 / 6** (5/5 excluding speech) |

Same envelope both times, so this is the anchor and not the smoothing. Widening the envelope's bandwidth
makes onsets *worse* — median error 144 ms at 320 Hz against 63 ms at 40 Hz — because a wider band tracks
pre-event fluctuation a fixed threshold then fires on.

Every scored peak was located inside a labelled span. Running the rules unsupervised over a whole
envelope is the harder problem.

## Offset — a fraction of the event's own range

`peak − 0.7 × (peak − floor)`, closing after 120 ms continuously below.

| offset rule | median \|error\| | worst |
| --- | --- | --- |
| fixed `peak − 10 dB` | 573.9 ms | 1285.1 ms |
| fixed `peak − 30 dB` | 206.5 ms | 1066.3 ms |
| `0.7 × (peak − floor)`, 250 ms hangover | **84.3 ms** | 417.7 ms |

A fixed drop cannot serve both: the mouth sound peaks 20.0 dB above the floor and cough 1 peaks 56.8 dB,
so one drop is at once too shallow for a cough and unreachable for the click, which at `−40 dB` runs
6.9 s to end of file.

**The hangover must be shorter than the shortest event to be bounded.** At 250 ms it overshoots a 202 ms
mouth click by 418 ms — the rule must observe more silence than the event lasts before it closes. Hence
per-consumer, and 120 ms shipped.

## Proposal is where both defects were

A first attempt grew regions from a `floor + 8 dB` gate. It merged both coughs and the unresolved
6.60–7.10 s stretch into one 4551 ms span, and dropped the mouth click entirely because its
minimum-duration test applied to the gate crossing rather than the derived span. Peak-picking fixed
both. A low-contrast peak's offset threshold sits near the floor, making its walk effectively unbounded
— which is what merged two coughs 1.7 s apart into one 6 s span at 12 dB.
