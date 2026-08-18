# What a real recording resolves — measured, 2026-08-17

One 14.03 s file, 48 kHz mono, close-miked (Brouhaha C50 median 28.5 dB, so essentially dry), holding
two coughs, two breaths, one short utterance and a handling click. Asserted by the file-level taxonomy
to contain breathing, coughing and speech. Detectors run locally; outputs under the job scratch dir.

## The events

| event | onset (s) | 10-90% rise | level step | detected as |
| --- | --- | --- | --- | --- |
| handling click | 0.893 | — | ~13 dB | transient |
| breath (exhale) | 2.275 | 60 ms | 28.6 dB | breath |
| breath (exhale) | 5.308 | 127 ms | 20.0 dB | breath |
| cough 1 | 7.924 | 17 ms | 44.9 dB | cough |
| cough 2 | 9.609 | **9 ms** | 48.5 dB | cough |
| speech | ~11.62 | — | — | "There's something going on." |

## Finding 1 — only DSP resolves an onset; no classifier comes close

Cough 2's rise is bounded to about ±5 ms on a 1 ms envelope. Independent flux detectors at a 5.33 ms
hop land within ~20 ms. Against that:

| detector | cough response width at 10% | speech leading edge error |
| --- | --- | --- |
| YAMNet 0.96 s / 0.48 s | 0.48 s | 1.06 s early |
| AST 0.96 s / 0.48 s | 0.96 s | 1.06 s early |
| AST 0.96 s / 0.10 s | 0.90 s | 1.34 s early |
| AST 0.48 s / 0.05 s | 0.65 s | 1.58 s early |

Shrinking the hop tenfold made the leading edge *worse*, not tighter: response width is set by the
window and the model's context, not by the hop. So a classifier cannot localise, at any hop, and
sliding it faster only buys sample density while looking like precision.

## Finding 2 — rise time separates cough from breath with no model at all

9-17 ms and 45-49 dB for the coughs; 60-127 ms and 20-29 dB for the breaths. The separation is
physiological — a cough is an explosive release against a closed glottis, a breath is turbulent flow —
and it is available from the envelope alone.

## Finding 3 — breath duration is not measurable, and neither is any offset here

Moving the offset threshold from floor+12 dB to floor+3 dB moves the breath offset by **2.03 s** and
**1.76 s**. The coughs carry 1.04-1.10 s of offset ambiguity. Any breath duration reported from this
file describes the threshold, not the breath. This is the same shape as the phonation-offset problem
in D12: for turbulent and aspirate events the offset is definitional, and a single-threshold rule
reports a choice as a measurement.

## Finding 4 — `pyannote/segmentation-3.0` calls the coughs speech

P(speech) saturates at 1.0 across [7.898, 10.226], covering both coughs, while Brouhaha's VAD stays
near 0.01 there and fires only on the real utterance. Brouhaha is right: SQUIM STOI is 0.18-0.44
across that region and two independent Whisper models transcribe nothing there. A cough's second
phase is voiced human sound carrying speaker identity, which is exactly what a speaker-segmentation
model is built to fire on. Used as a VAD it produces a 2.3 s false speech span, on a file whose real
speech is 1.5 s.

## Finding 5 — AST and YAMNet disagree sharply on the same event

YAMNet: `Cough` 1.000. AST: `Throat clearing` 0.93-0.96, `Cough` 0.11. Given a 9 ms rise and a 48 dB
step, YAMNet is right. Two consequences: the correlation risk accepted in D6 does not show up here —
these two failed differently, which is what makes them two families — and the taxonomy cannot assume
its confusable classes are separable by classifier vote, because on this file they are not.

## Finding 6 — periodicity measures are unavailable outside speech

Praat HNR returns nan nearly everywhere, with valid values only at the two cough onsets. pyin rails at
its 60 Hz floor through the quiet stretches, locking onto low-frequency rumble. Any design leaning on
HNR or F0 as a general vocal-evidence family must account for their being undefined wherever there is
no periodic content — which is most of an airway-branch recording.

## Also measured

No background talkers on this file: pyin voicing probability never exceeds 0.31 outside the utterance
and segmentation-3.0 shows no second speaker. 81.7% of energy sits below 1 kHz, consistent with
proximity effect. Stationary tones at 85.0, 108.4, 164.1, 1564.5 and 1757.8 Hz. Clipped fraction 0.000.

`pyannote/voice-activity-detection` is **gated (403)** for this account, so the dedicated VAD pipeline
could not run; raw `segmentation-3.0` frame posteriors were substituted, which is what surfaced
Finding 4.
