# Config derivations

Every measurement, failure and retraction behind
`src/senselab/audio/workflows/triage/data/config/default.yaml`.

This prose used to live in that file as a `derivation:` key. It was a config *value*, not a
comment, so the loader hashed it: 50,424 characters of English sat inside `config_hash`, and
correcting a word of it made two behaviourally identical runs report different identities. It moved
here so the hash names the parameters. The config now carries a short description per section and
per key; the reasoning is below, keyed by the same section names, and is reproduced verbatim.

CLAUDE.md: "the measurement behind a choice, the failure that drove it and the rejected
alternatives go in `specs/`."

## How to find a key

Sections appear in the config's own order. A key's derivation is in its section; where one
paragraph derives keys in several sections it is filed under the section it chiefly governs, and
the cross-cutting nulls are gathered under [unset](#unset).

## resample

The one sampling rate every downstream model takes.

Resample target 16000 Hz -- benchmarks/preprocess-params.md. Every downstream model is 16 kHz
native (YAMNet, HeAR, AST, CrisperWhisper, SQUIM), so the choice is one named resample here or N
inside N backends. At least 96.3% of every labelled event's energy survives an 8 kHz Nyquist; the
thin margin is the airway events, whose 95% points sit at 6981 and 7272 Hz, not speech, which is
band-limited to 632 Hz.

## preemphasis

The first-difference filter applied before the primary envelope.

Pre-emphasis 0.97 -- conventional in speech analysis, not fitted here. It raises event-to-floor
contrast on every labelled event and most on the two hardest, cough 1 by +10.95 dB and the mouth
sound by +7.36 dB, both of which carry 14-16% of their energy in 4-8 kHz.

## envelope

The amplitude envelope the primary span pass is proposed from.

Envelope lowpass 40 Hz, zero-phase -- benchmarks/preprocess-params.md. Sweeping the cutoff against
six labelled events, a wider band makes onsets worse (median 144 ms at 320 Hz against 63 ms at
40 Hz) because it tracks pre-event fluctuation a fixed threshold then fires on. 40 Hz is the
modulation bandwidth the envelope is for, not an onset-precision choice. Zero-phase beats causal,
63.5 ms against 90.1 ms median, which makes the envelope offline-only.

Envelope filter order 4 -- conventional Butterworth order, not fitted. Applied forward-and-backward,
so the effective order is doubled and the phase response is exactly flat.

## floor

The single global noise floor every rise is measured against.

Floor: a single global value per recording, in dBFS -- the 5th percentile of the ENVELOPE (not the
raw waveform), one number for the whole file rather than a rolling one. Owner-directed this
session, replacing a rolling 3 s / 10th-percentile-of-the-envelope floor (benchmarks/spans.md),
which on continuous real speech with no genuine internal silence tracked whichever moment happened
to be quietest inside whatever 3 s window currently surrounded the walk -- a swing of 20+ dB across
one uninterrupted utterance, verified on had_that_curiosity.wav, because "the 10th percentile of
recent speech" is not "the background noise floor" when there is no pause long enough to expose
one. It also fed a span's offset threshold from the floor at the peak's own sample, never re-read
as the walk advanced away from it, so a peak sitting where that rolling floor happened to read low
produced an unusually permissive threshold far from where the walk ended up -- the mechanism behind
a real multi-scene composite collapsing into one merged span once an unrelated gain-curve bug
stopped masking it. A single global number removes the staleness entirely: there is no longer a
floor value tied to one sample's position for a later sample to have drifted away from. The
percentile is read from the envelope specifically, not the raw waveform, so it stays the same
statistic ``rise`` (envelope minus floor) already compares -- a first attempt read it from
``|samples|`` instead and was caught before shipping: a stationary noise floor's raw-sample 10th
percentile and its own Hilbert-envelope median differ by a fixed ~19.8 dB regardless of the noise's
actual level (the Rayleigh-vs-half-normal ratio of the same process), which put ordinary background
noise above both spans.k_db values with no real event present at all -- verified directly: a plain
noise-bed-plus-one-burst fixture measured a median ``rise`` of 19.4 dB with no burst in sight. 10.0
was carried over from the retired rolling floor's own percentile, not re-derived for the new
definition; lowered to 5.0 this session, owner-directed, tightening the floor toward the true
quietest stretch of the recording rather than a broader low-decile band. Read by the two amplitude
envelopes only; spectral continuity no longer references a floor at all, since its rank cut
(spans.continuity_cut_percentile) is scale-free. Not re-derived from a corpus fit either.

## spans

Span proposal: the gate, the walk, the length filters and the four sources.

Span walk-stop: 12 dB above the floor, sustained for 30 ms, the same rule for onset (walking
backward) and offset (walking forward) -- owner-directed this session, replacing the asymmetric
pair this file previously carried: a peak-anchored onset (benchmarked at 5 of 6 correct against
2 of 6 for a floor-referenced rule) paired with a 0.7-of-range offset and a 120 ms hangover
(median offset error 84.3 ms against 573.9 ms for a fixed peak-10 dB threshold), both
benchmarks/spans.md, both measured against the rolling local floor retired above. That pair's
asymmetry -- onset checked one sample against two thresholds, offset required a sustained window
against a threshold fixed once at the peak's own (rolling) floor value -- is what let a
multi-scene recording over-merge once the floor's own staleness stopped being masked by an
unrelated gain-curve bug. A floor that is now one global number removes the motivation for a
peak-anchored rule at all: neither walk can close for a reason the other would not also close for.
12 dB and 30 ms are this session's chosen values, not a corpus fit -- the prior benchmark measured
a different rule against a different floor and does not carry over; both numbers are provisional
and expected to be refit. 30 ms (owner-directed, tightened from an initial 300 ms) is short enough
that a walk does not keep bridging past a real gap between two nearby events on this narrower
duration -- shorter than spans.min_duration_ms (50 ms), so it cannot itself force a span past that
floor. spans.min_separation_ms must track it exactly (see that key's own entry below): a raw
threshold-crossing gap the pre-merge stage does not absorb has to be at least this long, or a real
gap could reach the walk already merged away, never getting a chance to stop it.

Span propose K, one shared value, 6 dB -- owner-directed this session, provisional and expected to
be refit. PREPROCESS proposes one set of foreground-acoustic-event spans, not an airway-specific
set and a separately-thresholded general set side by side -- a span carries no notion of which
downstream branch it is "for", only whether it overlaps a clip. Two prior values are retired
rather than kept as second keys: 18 dB (benchmarks/spans.md, events at 53-57 dB above the
then-rolling floor) was fitted for airway events specifically and does not serve a quiet breath or
a soft word; 12 dB was tried for a general "any vocal task" gate but never measured against this
floor definition. Neither carries over cleanly to a shared, population-general threshold, so 6 dB
is a deliberately permissive placeholder -- low enough that the pre-emphasised primary pass (see
the envelope/floor derivation) rarely misses a real event, at the cost of also passing more of the
background than a fitted value would. AIRWAY no longer derives a population-specific gate of its
own either (retired this session, see the airway derivation below): it takes PREPROCESS's general
span set directly, at the one shared threshold, and reads PREPROCESS's own per-span HeAR/YAMNet
labels rather than re-deriving evidence at a second, separately-configured K. SPEECH no longer
reads these spans at all -- it derives its own from word timings.

Span source: primary, supplementary, continuity and ASR -- owner-directed this session. PREPROCESS
proposes spans up to four times, from four different measures of the same recording, and keeps
the union, each source only ever adding a candidate none of the earlier sources already cover:
primary from the pre-emphasised signal's amplitude envelope (envelope.lowpass_hz/.filter_order,
ButterworthSmoothing, ``energy_envelope`` below), supplementary from the dynamically-normalized
signal's own amplitude envelope (``normalization.envelope_smoothing``, PercentileSmoothing,
``normalized_envelope`` below), and continuity from spectral_continuity.spectral_continuity over
the pre-emphasised signal (see the paragraph below). Primary rather than the normalized signal,
reversing this session's own earlier design: measured directly, the AGC can compress a real
recording's local dynamic range enough that no span-gate value clears it at all -- a five-breath
recording's rise-over-floor topped out at 9 dB after normalization against 23 dB on the same
recording pre-normalization, with the same k_db and floor definition throughout. That risk is why
normalization stays supplementary and optional rather than the primary pass -- not a reason to
keep it off: the pre-emphasised signal needs no optional step to exist, so spans are never
entirely absent for want of a normalization fit, and normalization.* is now on by default (see
the normalization.* paragraph below) precisely because the supplementary and continuity passes,
where either finds something, only ever *add* a candidate no earlier pass found (typically a quiet
event AGC boosted into range, or a sustained tonal/harmonic production too soft to clear any
amplitude gate), never replace or override one -- so the AGC-compression failure mode can only ever
cost a missed addition, never a wrong primary span.

Spectral continuity -- owner-directed this session, added specifically to catch breathing and
glides that an amplitude-only gate can miss. Originally its own window_s/hop_s pair; now reuses
spectrogram_narrowband's own magnitude array directly (spectrogram.narrowband_window_ms/.hop_ms,
20 ms / 5 ms) rather than computing a second STFT at merely-matching parameters -- one recording
put through one STFT for both purposes, and `_spans` reads spectrogram_narrowband's output from
`state` instead of recomputing it. Narrowband rather than wideband, measured: see
benchmarks/preprocess-params.md, "Which representation feeds continuity". A 5 ms window is about
one glottal period, so the wideband array resolves the pulse train in time rather than resolving
harmonics, and its frame-to-frame comparison swings on glottal phase during steady phonation;
the 20 ms window does not. Owns no smoothing key of its own any more: the frame-to-frame trace
is smoothed with the identical ButterworthSmoothing(cutoff_hz=envelope.lowpass_hz,
order=envelope.filter_order) the primary amplitude envelope itself is built with, not a
separately-chosen scheme -- owner-directed, replacing an earlier MedianSmoothing(window_s=0.2)
picked independently for continuity alone. Continuity is compared against and deduplicated with
the amplitude envelope in the same `_spans` pass (see the priority-order paragraph above); sharing
the envelope's own filter means both measures agree on what counts as the same continuous event at
the same time-constant before that comparison happens, rather than each drawing its own boundary
on its own schedule. spans.continuity_cut_percentile 5.0 -- measured, and it replaced the retired
absolute pair spans.continuity_margin/.continuity_floor_margin (0.03 / 0.02) outright. Continuity
is a novelty function, so it is read for its dips, not its plateaus: a high value only says
nothing changed, which is equally true of steady silence and steady phonation, and no gate on the
plateaus can separate those. The lowest 5% of trace samples BY RANK are the change points and the
runs between them are the spans. Rank, not a comparison against the percentile's value: a value
cut lands on the plateau of a flat trace, where `>` admits no samples at all (an all-continuous
file measured zero spans) and `>=` admits every one of them. A rank cut is also scale-free, which
is what let the representation change at all -- an absolute 0.03 means something different against
every input array, and refitting it per representation was the blocker the rank cut removed. 5.0
is the owner's own proposal and measured: on Prolonged-vowel, median segment duration outside
speech divided by the same inside speech reads 7.6x at p5 against ~1.0x -- no separation -- for
every other representation and smoothing tested. p2 reads 8.8x, consistently but not decisively
better; 5.0 kept for the wider margin against a file with fewer genuine events. Honest limit: a
percentile gate cannot express "nothing here", since it always marks exactly that fraction of
frames as change points, however continuous the recording. Not a corpus fit -- three recordings.
The measure itself (frame-to-frame cosine similarity
of log-magnitude spectra, smoothed) worked well for sustained tonal/harmonic production under the
retired MedianSmoothing: on a High-to-Low glide it read 0.95-0.99 through the voiced glide against
0.86-0.92 in the silence flanking it, and on a held vowel it separated the same way, both
re-confirmed on the actual reused torchaudio magnitude array (previously validated via an
independent scipy STFT at matching parameters; the two agree to within one hop) -- not
re-measured under the new ButterworthSmoothing. Measured directly and found NOT to work for breath
specifically: on a five-breath recording, the measure reads 0.85-0.90 with no separation tracking
the breaths' own timing at all -- turbulent broadband noise does not carry the frame-to-frame
spectral coherence a harmonic sound does, so this mechanism helps glides and phonation-like
sustains on the evidence gathered so far, and does not, on the one breathing recording tested, help
breath detection the way it was asked to. Left in as a general continuity detector rather than
restricted to glides specifically, because nothing about the mechanism is glide-specific and a
future corpus may show it earning its keep on other sustained productions; the breath gap is
recorded here rather than hidden so it is not mistaken for a working detector later. No longer
reads spans.transition_window_ms or spans.min_separation_ms: a rank cut has no walk to stop and no
neighbouring proposals to merge, so both dropped out with the retired threshold pair.
spans.continuity_min_duration_ms (300 ms) is its own key because continuity events (a glide, a
sustained vowel) are typically much longer than the 50 ms amplitude spans.min_duration_ms admits,
and reusing the shorter value let noise-length blips through in testing. It is the only length
filter left on this source, and it is what the measured discrimination above was run under. Absent whenever
spectrogram_narrowband itself is, since continuity now has a hard data dependency on that block's
output rather than computing its own.

ASR spans -- owner-directed this session, the fourth and lowest-priority source. Unlike the three
above, this one measures nothing and needs no floor or gate: the consensus transcript's own word
timings (the lexical words of ``consensus_transcript``; a bracketed word proposes nothing) are the
evidence directly -- a recognizer
transcribing a stretch as speech already says something an envelope or a spectral-shape measure
would otherwise have to infer from the acoustics. Consecutive consensus words are grouped into runs
by speech.word_gap_ms, the identical mechanism and the identical config key SPEECH already uses for
its own word-timing spans (group_extents_into_runs, shared rather than duplicated so the two
speech.word_gap_ms -- DELETED 2026-09-05, parameter and key both. Words arrive from the recognizers
already timed, so their extents are the span; nothing is inferred and nothing needs a threshold to
become one. group_extents_into_runs now merges where extents touch or overlap and takes no gap
argument at all, a parameter that could only ever take one defensible value being worse than none.

What the key cost while it existed. It shipped null and PREPROCESS read it with `.get`, skipping the
ASR span source entirely when it was None -- silently, with the node still returning PASS. A
388-recording run over ten subjects produced 7,651 consensus words and zero ASR spans. SPEECH read
the same key with `require`, so the same null errored that branch outright on all 112 recordings of
the earlier stage-0 collection. One unmeasured key, two failures, neither of them announced.

Both call sites shared the key deliberately, so PREPROCESS and SPEECH could not drift onto two
notions of one span; they now share the parameterless rule instead. The concept the key expressed --
"one utterance", a run of words delimited by pauses -- is no longer expressed anywhere in the graph.
Nothing downstream was found to depend on it. Two test fixtures did, both by placing words with
deliberate holes between them and relying on a threshold to close them; both now place contiguous
words, which is what the helper's own docstring already claimed it did.

branches cannot drift onto two different notions of "one utterance"). speech.word_gap_ms ships null
in the packaged config -- "any value is a claim about what makes one utterance," per its own
derivation below -- so the ASR span source is absent by default alongside every other place that
reads the same key, not a special case. Also absent whenever consensus_transcript itself is (both
recognizers failed). Kept only where it does not already overlap a primary, supplementary or
continuity span: a stretch of real speech an amplitude gate and a spectral-shape gate both missed,
filled in by the one measure that read it directly rather than inferring it.

Span min_duration_ms 50, conventional and not fitted -- discards a proposal shorter than anything
the instruments here localise. min_separation_ms 30, changed this session from a conventional
150 ms and pinned equal to spans.transition_window_ms rather than left independent: it absorbs
threshold-crossings this close together into one proposal before either is walked, so one event's
own brief dips below k_db do not read as several, but at 150 ms it was absorbing more than that --
verified directly on a real recording (a DDK buttercup task) where a genuine ~149 ms gap, well
past the walk-stop's own 30 ms sustain requirement and dipping below the floor itself, sat 0.6 ms
under the old 150 ms cutoff and got glued into one seven-second span the walk-stop rule would
otherwise have split. The walk stops at k_db, the same level that opens a candidate, which makes
this provable rather than merely observed: any sample failing the crossing test also satisfies the
stop, so a raw gap of at least transition_window_ms is always long enough to stop a walk on its own
once it reaches it. Setting min_separation_ms to that same value closes the gap: nothing shorter than a
self-sufficient stop condition ever survives to the pre-merge stage un-merged, and nothing merged
away was ever going to stop a walk regardless. The two keys must move together -- if
transition_window_ms changes again, min_separation_ms should equal it, not sit independently at
whatever value was last fitted.

### floor_margin_db -- RETIRED 2026-09-04

The walk-stop used to have a level of its own, `spans.floor_margin_db`, shipped at 12.0 against a
`k_db` of 6.0. It never did anything. The stop test was *more permissive* than the open test, so
every sample that failed the `k_db` crossing already satisfied the stop, and no walk could extend a
span past its own crossing run at any `floor_margin_db >= k_db`. Measured on a real 79.51 s envelope
(1,272,082 samples, floor -78.28 dBFS): 12.0 and 6.0 both give 86 spans totalling 59.77 s, extents
byte-identical; only values *below* `k_db` change anything (3.0 -> 51 spans / 66.52 s, 0.0 -> 11
spans / 74.88 s).

Why it existed is still worth knowing: it replaced a peak-anchored onset paired with a
floor-fraction offset, whose asymmetry let an offset's stale, peak-anchored threshold outlive the
walk's own progress and collapse a real multi-scene recording into one merged span. The symmetric
floor-relative rule it introduced is what survives -- only its separate level is gone, folded into
`k_db`. `transition_window_ms` is untouched and remains load-bearing: it sets how far the walk looks
for a still-loud sample, and so whether neighbouring events bridge into one span (30 ms -> 86 spans,
100 ms -> 45, 300 ms -> 18 on the same envelope).

## clipping

Clip detection and how per-period plateaus are coalesced.

clipping.* -- audio/tasks/clipping, ClipDaT (Hansen, Stauffer & Xia, Speech Communication 134
(2021) 20-31, section 3). near_threshold 0.995 and leniency_samples 3 are the paper's own two
reported constants, not fitted here -- the algorithm has no others. minimum_extreme 1e-4 is not
from the paper: it guards a degenerate case the paper does not address, where the file's global
max or min sits at or near exact zero (a one-sided clip against a silent baseline, or true
silence) and the 99.5% band of that near-zero value excludes almost nothing, opening a spurious
event across the whole baseline. A numerical guard against a divide-by-near-zero class of input,
not a perceptual threshold. The detector no longer opens an event on a lone sample at a merely
relative extreme, so the caller-side duration filter that used to suppress that is gone: there is
no clipping.min_duration_ms. See benchmarks/preprocess-params.md for the run-length measurement
behind the detector's own gate.

clipping.merge_gap_ms 30 -- the surviving caller-side value, and the only one _clip_spans applies.
Its job is coalescing, not suppression: per-period clipping of voiced speech leaves one plateau
per glottal cycle, so a hard-limited 200 Hz tone yields 1200 separate events over 3 s and a 120 Hz
one yields 720. Merging within 30 ms collapses each to a single span covering the clipped region
(measured: 1 span, 2.999 s, for both), which is the correct description of one clipped vowel;
without it a single clipped vowel would render as hundreds of adjacent spans. 30 ms comfortably
exceeds one glottal period across the plausible F0 range while staying well below the separation
of distinct clipping episodes -- a 2 Hz sine hard-limited at 0.6, whose plateaus sit ~148 ms
apart, still yields 12 separate spans totalling 1.778 s rather than being bridged into one. The
value is conventional, not corpus-fitted, and is still owed a corpus.

## normalization

Dynamic-range normalization, the supplementary span source's signal.

normalization.* -- audio/tasks/envelope.dynamic_range_normalize. Owner-directed this session: on by
default, all eight keys given real values, so PREPROCESS's normalized_envelope measurement and its
supplementary spans actually run under the packaged config -- some real content (a quiet event AGC
boosts into range) is only reachable through this pass, not through the pre-emphasised primary
pass alone. The function's own reference implementation (a pasted sketch reviewed and reworked to
reuse hilbert_envelope_dbfs for both its macro and micro envelopes instead of a bespoke
rectify-and-filter) carried illustrative literals -- 0.2 Hz / 20 Hz cutoffs, 15 dB target range, a
compression ratio of 2, a -6 dBFS reference, a 10 Hz gain-smoothing cutoff, a 0.95 ceiling -- and
none of them was a measurement, only the shape of a reasonable algorithm; the cutoff-based pair was
since replaced by MedianSmoothing(window_s=...) (see the two paragraphs below) for a measured
reason, so those two keys carry window_s values (0.5 s / 0.05 s) rather than the original cutoffs.
The remaining five values (target_dr_db 15.0, compression_ratio 2.0, macro_target_dbfs -6.0 dBFS,
gain_smoothing.window_s 0.025 s, floor_dbfs -100.0 dBFS, ceiling 0.95) are carried over from the
original illustrative sketch as-is -- owner-directed, provisional, not corpus-fit, the same status
as spans.k_db/floor.percentile/spans.transition_window_ms elsewhere in this file. Turning this on
does not relax the AGC-compression finding two paragraphs above (a real recording's rise-over-floor
can still collapse under this AGC); it is safe specifically because normalization stays a
supplementary, additive span source that can only add a candidate, never replace the primary
pre-emphasised pass or override one of its spans. Re-verified with these exact eight values across
all four FiveBreaths recordings in the same subject's session before shipping this as the default:
primary/normalized rise-over-floor was 25.9/11.2 dB, 56.2/26.9 dB, 32.0/12.2 dB and 55.0/26.4 dB --
the compression is real and consistent (roughly half, matching the original 23/9 dB report) but
every normalized value still clears spans.k_db (6 dB) with room to spare, so the supplementary
source keeps contributing rather than going silent.

normalization.macro_smoothing.window_s, .micro_smoothing.window_s --
audio/tasks/envelope.MedianSmoothing, plugged into hilbert_envelope_dbfs for the two internal
envelopes dynamic_range_normalize derives its gain curve from. Chosen over a zero-phase Butterworth
(envelope.lowpass_hz's own ButterworthSmoothing) after comparing the two on the same word-burst
composite used elsewhere in this session's work on the target-span block: the Butterworth showed a clear overshoot
at a word's onset -- a dip past the pre-onset baseline that a forward-and-backward filtfilt spreads
to both sides of the transient it is reacting to -- because its output is a resonant filter's
response and can take a value the signal never did. A median's output is always one of the
window's own samples, so it cannot overshoot. envelope.lowpass_hz/.filter_order keep their own
fitted derivation above and stay Butterworth-only for the airway block, which this change does not
touch. Both window_s keys are null; nobody has fit a window size to a corpus, only compared the
two strategies' qualitative behavior on one file.

normalization.gain_smoothing.window_s -- audio/tasks/envelope.MedianSmoothing, replacing what this
file used to call gain_smooth_hz/.gain_filter_order (a Butterworth), on reasoning this file
originally rejected for the *envelope* case above but had not yet re-examined for the *gain curve*
case. The original argument was that a gain multiplier's failure mode is a discontinuity, not an
onset overshoot, so Butterworth's smoothness was worth keeping there even though it was rejected
for envelopes. Measured directly and found wrong: on a ~150 ms tone burst, the pre-smoothing gain
curve is already a clean, correct step (huge gain in silence, a flat and correct value for the
entire burst) -- Butterworth's own resonance still could not settle to that flat value within the
burst, applying several-hundred-percent excess gain for most of the event's duration rather than a
brief, edge-localized ringing artifact. Raising the cutoff to 200 Hz (20x the original 10 Hz)
still left the mid-burst gain ~3.5x too high; the residual is the filter's lag behind the
*upstream* macro-level transition, not a bandwidth problem a higher cutoff can buy down. A median
settles to the correct plateau almost immediately regardless of window width (5-50 ms all measured
within a few percent of the correct value), at the cost of the discontinuity-avoidance the original
reasoning wanted -- a real trade, made because the measured failure (wrong gain for most of a short
event) is worse than the risk it re-accepts (a bounded rather than ramped transition at the edges).
Null; not corpus-fit, only compared on one file the same way the keys above were.

normalization.envelope_smoothing.window_s, .percentile -- audio/tasks/envelope.PercentileSmoothing,
the envelope actually used for target-span detection (the two keys above only feed the internal
gain curve). Started as the same MedianSmoothing as macro/micro, on the reasoning directly above --
but a median is a central-tendency statistic, and plotted against the raw, unsmoothed
|hilbert(x)| on real speech, it reads 3-4x quieter than the signal's own true local peaks (e.g.
0.15-0.2 where the raw envelope reaches 0.6-1.0), because a median deliberately reports the middle
of its window, not the top. Where this pipeline needs to know "is foreground energy present", that
gap directly reduces the contrast target-span detection depends on. A plain rolling maximum was
tried next and rejected the opposite way: one loud sample pins the entire window to its own height
and holds it there, both before and after the sound occurred, because the statistic has no notion
of *how far away* the peak it is reporting sits -- verified directly (t=4.85s on the same
composite: the held value read 0.68 while the envelope's own value at that exact instant was
0.105, sourced from a peak 4.4 ms away) and it does not improve by widening or narrowing the
window, only by changing which single sample dominates. window_s=0.025 s, percentile=90.0 is what
a direct three-way comparison (median / rolling max / percentile, at several window widths, on the
same real speech) converged on: high enough to sit close to the true local peak, but a percentile
rather than a maximum, so one outlier sample sixty milliseconds away cannot govern a window it
barely overlaps. The window stays at 25 ms, not the 5 ms the visual peak-hugging comparison alone
favoured: on the existing clip/target-span regression fixture (a clean tone burst against a near-
silent noise floor), a 5 ms window drops a real burst entirely and it is not the percentile causing
it -- the 50th-percentile (median) case at the same 5 ms window drops the identical burst, while
90th-percentile at 25 ms recovers all three spans a median at 25 ms already found. So the window
width, not the statistic, is what this content is sensitive to; 25 ms is the width already
exercised without incident. This is a chosen value from that comparison, not a corpus fit --
unlike the null keys throughout this file, whose fits still remain to be done.

## spectrogram

The two STFT resolutions; the narrowband array also feeds continuity.

Spectrograms 5 ms and 20 ms window, 5 ms hop -- benchmarks/preprocess-params.md. At F0 88.1 Hz the
glottal period is 11.4 ms, so 10 ms resolves neither harmonics (150 Hz against 88 Hz spacing) nor
pulses (0.88 of a period). Two windows rather than a compromise between them.

## gammatone

The auditory filterbank's channel layout.

Gammatone 40 ERB channels, 80-7800 Hz, 5 ms hop -- conventional auditory-filterbank settings.

## yamnet

YAMNet's silence and coverage readings, and its label-space size.

YAMNet silence_threshold and coverage_threshold, both 0.5 -- benchmarks/hear-yamnet.md and
taxonomy.md. Silence is bimodal across the reference recording's 29 windows: every score is <=0.36
or >=0.62. The labels that matter leave the same kind of gap containing 0.5 -- Cough jumps
0.84 -> 0.27, Speech 0.92 -> 0.14, Breathing 0.59 -> 0.36. A threshold in an empty interval is not
a fitted value.

YAMNet top_k 521 -- the full label space, which is a size, not a threshold. classify_audios
defaults windowed top_k to 5 and Silence is not always in the top 5 (capability-map 4.2), so a
truncated read silently reports zero for a label the model actually emitted.

## windows

Per-classifier window grids and the thresholds folding scores into label sets.

v2 window classifications -- preprocess.md's "sets, not accumulators" rule. A window's product is the
set of labels each clearing ITS OWN threshold, and the file-level product is the set-union with the
windows retained per label. The thresholds are windows.<classifier>.default_threshold with a
per-label override map; all six are null because no ROC over this corpus exists. The v1
taxonomy.presence_floor.{yamnet,ast,hear} 0.5 values are RETRACTED and deleted rather than carried
over: they were read off bimodal gaps in one reference recording's whole-file scores, and a
whole-file gap is not a per-window threshold. YAMNet's grid is not a key at all -- classify_audios
ignores win_length/hop_length for YAMNet and returns its own 0.96 s / 0.48 s frames, so the grid is
recorded as a fact on the pooled measurement.

HeAR window 2.0 s -- model-imposed, not chosen. The detector's graph rejects every other input
length outright, which is why a shorter span is placed in a buffer rather than passed as-is.

windows.ast.win_length_s 10.24 -- owner-directed: AST reads the recording in 10 s windows (10.24 s
is the model's 1024-frame input at a 10 ms hop, the nearest realisable width to the directive).
The audio_analysis workflow measured and prefers a 0.96 s slid window for its own purposes; that
workflow's notes do not govern triage, and this key exists precisely so the choice is declared and
hashed rather than inherited. A finer window remains reachable as an override. Two consequences a
reader should know: a recording shorter than the window is zero-padded by ASTFeatureExtractor, so
it yields one window covering the whole file; and AST's windows sit on a coarser grid than
YAMNet's, so the acoustic evidence line counts each classifier's windows on its own grid.
windows.ast.hop_s 10.24 -- owner-directed 10 s windows are read NON-OVERLAPPING until a hop is
fitted, so the hop equals the window and AST reads the recording once end to end. It is a declared
default rather than a null because a null hop stopped AST running at all: require() raised inside
the scores block, so the model never ran under the packaged config and the expensive output was
lost along with the threshold fold it was supposed to survive. A fitted hop from 8.0 s through
10.24 s remains an override; a smaller hop would make heavily overlapping AST contexts look like
independent time-local evidence and is rejected by config validation.

windows.ast.top_k 527 -- the full AudioSet label space AST was fine-tuned on. A SIZE, not a
threshold, and the same reason yamnet.top_k is 521: classify_audios does `top_k=top_k or 5` on its
windowed path (classification/api.py:135), so passing None does not mean "keep everything", it means
"keep five". Five of 527 is a RANKING over the vocabulary, which is the one operation
preprocess.md's set rule forbids -- it would make "the set of labels over threshold" silently mean
"the set of the top five labels that are also over threshold", and a label the model emitted at 0.9
in a busy window would vanish. HeAR is unaffected: detect_health_acoustic_events takes a different
path on which top_k=None does keep all eight labels.

HeAR's 2 s window is model-imposed and lives in the module, as HEAR_WINDOW_SECONDS; only its hop
is a key here.
windows.hear.hop_s 2.0 -- non-overlapping 2 s windows, by the same ruling as windows.ast.hop_s and
for the same reason: a null hop meant HeAR never ran under the packaged config. A fit on spans
HeAR's input does not have to be padded to fill is still owed, and lands as an override. hear.placement and
hear.label_floor are DELETED with the code that read them: placement was only ever an argument to
span_to_hear_buffer, which branch-airway.md removes from the graph by confining HeAR to PREPROCESS,
and label_floor is replaced by windows.hear.default_threshold. speech.agreement_flag_floor is
deleted likewise: branch-speech.md replaces the aggregate-agreement flag with per-word recognizer
membership, so there is no aggregate for a floor to gate.

## phonation_spans

Praat's settings for PREPROCESS's F0 and formant tracks. The detector these keys were named for was
removed on 2026-09-04; the five surviving keys are the track settings, and the section keeps its
name only until a rename is worth the config-hash churn.

### The detector -- REMOVED 2026-09-04

Owner-directed. Eight of this section's keys were its criterion parameters -- `f0_stability_cents`,
`formant_stability_hz`, `glide_min_excursion_cents`, `hangover_ms`, `voicing_strength_floor`,
`mixed_voiced_fraction`, `unvoiced_max_formant_bandwidth_hz`, `word_aligned_min_evidence_fraction`
-- and every one shipped null, so `_propose_phonation_spans` raised on `require()` on every run and
was swallowed as a cascading absence. The pass was dead code that read as a measurement.

Two defects are on the record and neither is repairable by fitting those eight values, which is why
the detector went rather than being given numbers:

- **The glide criterion cannot classify a pitch glide on steady formants.** Its two continuity limbs
  are OR'd and the formant limb alone suffices, so an F0 sweep of 150-600 Hz over a steady vowel
  reads `sustained` at every rate probed -- and the clinical glissando task is exactly that shape.
- **A sustained span needs no voicing to be admitted.** Formant stability is defined on any audio,
  so the formant limb admitted whole-file `sustained`/`unvoiced` spans with `voiced_fraction=0.0` on
  a breath recording and on free speech. A span whose voiced fraction is zero and whose family is
  `phonation` is a contradiction in terms.

**What it cost, recorded rather than hidden.** The detector was the only producer of
`family="phonation"` spans, and two readers depended on them:

- TAXONOMY's `voice` kind had exactly one evidence line, fed by those spans. It now reports
  `unavailable` with a `why` naming the retirement, which folds to `uncertain` and never to
  `absent`. The gating keys `taxonomy.voice_min_duration_s` and `.voice_uncertain_duration_s` are
  consequently read by nothing.
- The VOICE branch's whole subject is those spans. In production it now finds none and says so, in
  a `why` that names the retirement rather than the bare "no phonation span in the store" a reader
  would take for a property of the recording.

**A structural consequence worth stating plainly:** with `voice` permanently `uncertain`,
TAXONOMY's fold can reach neither `FAIL` (every kind absent) nor `PASS` (nothing uncertain). Every
recording flags, whatever it contains. That is an artefact of the retirement, not of the audio, and
it lifts when voice is reworked to read `consensus_taxonomy` -- which is owed a decision nobody has
made: which consolidated labels express the voice kind, and how they map to a state.

phonation_spans -- the sustained-phonation and glide detector, moved from PREPROCESS to TAXONOMY
this session (owner-directed): PREPROCESS now only measures F0 and formant tracks over the whole
stream (``phonation_tracks``, no boundary decided); TAXONOMY reads that measurement back and
proposes the spans over it, the same functions and the same ``phonation_spans.*`` keys as before,
just applied by a different node. Its seven criterion
parameters are null; the five described at the end of this paragraph -- hop_s, formant_max_hz,
max_formants, formant_window_s and formant_preemphasis_hz -- ship declared defaults.
f0_stability_cents and formant_stability_hz are the two limbs of the continuity criterion (a frame
continues a sustain when F0 moves less than the first across one hop, OR F1 and F2 both move less
than the second); glide_min_excursion_cents is the monotone excursion separating a glide from drift;
hangover_ms is how long the criterion must fail continuously before the span closes.
voicing_strength_floor is the Praat pitch strength above which a frame counts as voiced, and
mixed_voiced_fraction is the voiced-frame fraction separating voiced from mixed from unvoiced -- both
are needed because a disordered voice sustains with little or no periodicity and a detector that
required a periodicity floor would measure exactly the voices least in need of measurement.
unvoiced_max_formant_bandwidth_hz requires the F1/F2 poles that carry an aperiodic sustain to be
narrow enough to be resonant evidence; stable Burg poles alone also arise in broadband noise and
must not promote it to phonation. It is null until fitted, is not a diagnostic proxy, and does not
apply to the periodic F0 limb.
word_aligned_min_evidence_fraction is the fraction of frames in a timed consensus-word segment that
must show periodic or narrow-resonant evidence before that acoustic segment becomes a phonation
span. Word text never enters the test, and word evidence is complementary: no word cannot suppress
a sustained-phonation span. It is null until fitted.
formant_max_hz 5000.0, max_formants 5, formant_window_s 0.025 and formant_preemphasis_hz 50.0 are
praat_parselmouth.py:888-891's own to_formant_burg defaults -- conventional, not fitted here. hop_s 0.01
is Praat's documented time_step default, the same value phonation.hop_s already carries.

## words

Word-level vocabularies read off the consensus transcript.

words.onomatopoeic_tokens -- the vocabulary of cough- and breath-like renderings a recognizer emits
as ordinary words ("khh", "ahem", "uh-huh-huh"). Null: it is owed the corpus it was drawn from, and
seeding it from three remembered examples would be a vocabulary nobody fitted. While null, only
already-bracketed tokens become bracketed words, and an onomatopoeic rendering is counted as a
lexical word -- which is the honest state, not a safe default. A token in the vocabulary becomes a
bracketed word (`khh` -> `[KHH]`) with the raw token kept in the word's `readings`.

## stimulus

The consensus word stream aligned against what the recording declared it expected.

stimulus.sentence_terminators `".?!"` -- the characters that close a structure unit inside one
declared prompt. Orthographic, not fitted to any corpus: these are the three sentence-final marks of
the Latin-script orthographies the corpus records (the 4.0-release adult tree carries `language`
`en` and `es`), and the value is a declaration about writing systems, not a threshold over a
measurement. It is not null, because a null here would make the whole derivative absent on every
recording rather than making one projection unavailable; and it is not a threshold, because nothing
about it was chosen to separate two populations.

`AudioHints.expected_speech` is already a *list*, and its docstring says why -- "Ordered and
separate rather than one concatenated string, because 'which sentence was skipped' is a different
question from 'how close was the whole thing'". A caller who declares six sentences as six entries
therefore gets six units with no splitting at all. The terminator split exists so a caller who
declares one multi-sentence passage as one entry gets the same units: measured on the 4.0-release
adult tree, `rainbow-passage` carries all four of its sentences in one recording-grain
`stimulus_text`, and `caterpillar-passage` more than ten. Without the split those are one unit and
the per-sentence boundaries the VOICE consumer wants do not exist.

**No other key was added, and that is the finding, not an omission.** The alignment's three
outcomes -- realised, substituted, absent -- are the aligner's own path, not a cut over a score, so
no operating point separates them. Every number the consumers named in
`expected-patterns.md` (`p_omission_score_max`, `p_repeat_overlap_min`, `p_echo_overlap_max`,
`p_verbatim_overlap_max`) is a branch decision over this derivative's output, and belongs to the
branch that makes it. Inventing a PREPROCESS cut here would have been an unmeasured decision with a
public interface.

The edit costs the alignment runs under are not new either: it reuses `align_pair`
(`audio_analysis/harmonize.py`), whose sclite costs (match 0, substitution 4, indel 3) are derived
in `transcript-alignment.md`. They were designed for exactly the reference-against-hypothesis case
this derivative is, which is a weaker assumption than the ASR-against-ASR case they are already
serving.

## routing

Which declared tags force which kind's branch.

routing.hint_branch_map -- which may_contain tags and which metadata.speech_type values force which
kind's branch. A vocabulary, null, owed the corpus it was drawn from. A tag matching no entry forces
nothing and is recorded as unmapped; forcing only ever ADDS a branch.

## airway

AIRWAY's labels of interest and what may confirm or contest them.

Airway labels_of_interest {Cough, Breathe} -- branch-airway.md's default, from HeAR's eight. The
confirmation map Cough -> {Cough}, Breathe -> {Breathing, Sigh, Gasp} is branch-airway.md's step 2
table: which AudioSet labels corroborate each HeAR label. Vocabulary, not thresholds.

airway v2 -- branch-airway.md. airway.k_db, airway.k_db_by_task and airway.k_margin_db are retired
this session: they matched spans by a stored k_db attribute value, which PREPROCESS's spans no
longer carry uniformly (continuity and ASR spans carry none at all) and which no longer means
"the threshold this branch's own candidates were proposed at" now that PREPROCESS proposes one
general span set at one shared spans.k_db for every reader. AIRWAY now takes that general span set
directly and reads PREPROCESS's own per-span span_hear/span_yamnet measurements rather than
re-running HeAR itself at a second, separately-configured gate -- removing a redundant model pass
along with the stale threshold. airway.contest_labels is the declared set of YAMNet labels that may
contest a HeAR label; it is null and, when supplied, is refused at load if it intersects
the airway evidence set derived from taxonomy.airway_ontology_roots -- a label cannot be both
airway evidence and a contest of airway evidence.

## phonation

Praat's harmonicity and pitch settings for the F0/formant tracks.

Praat harmonicity settings hop_s 0.01, silence_threshold 0.1, periods_per_window 4.5 --
Praat's own documented defaults for the cc method, which extract_harmonicity_descriptors
(praat_parselmouth.py:639, the call at :680) already uses. Conventional, not fitted here. Two lengths follow from
them and are not the same length, which cost a round: periods_per_window / f0_min_hz is the
analysis window (VOICE's RMS track averages over it), while to_harmonicity_cc refuses any segment
shorter than (periods_per_window + 1) / f0_min_hz -- 1.2222x the window. Binary-searched at
f0_min 150 Hz: the shortest sound it accepts is exactly 587 samples at 16 kHz = 36.69 ms, against
a 30.0 ms window and a 36.667 ms analytic bound. VOICE therefore prunes on the larger figure and
counts what it dropped in the verdict's short_intervals_n. Pruning on the window instead left the
band [window, 1.2222 x window) unguarded, which is where real fragmented envelopes sit (measured:
median residual fragment 33.6 ms, 20 of 41 under 30 ms, and a measured unguarded band of
[60 ms, 73.3 ms) under the b2ai override's f0_min).

Praat's point process needs 3 / f0_min_hz, a third length again, also binary-searched: 640 samples
= 40.0 ms at f0_min 75 Hz and 320 = 20.0 ms at 150 Hz, both exactly 3 / f0_min. A gate run is a
run of hop_s frames, so one- to four-frame runs are ordinary, and every one of them was shorter
than this. VOICE skips period_marks for such a run and counts it in marks_skipped_short_n; the
run's marks measurement records unmeasured rather than a count of zero, because nobody looked.

phonation.period_doubling_factor 2.0 -- the definition of period doubling, an identity rather
than a threshold: a run is ambiguous when its median F0 times or divided by this factor also
lies inside the caller's declared range.

## praat_features

Praat's settings for PREPROCESS's whole-file feature set over the `enhanced` stream, plus the five
coefficients of the per-recording F0 narrowing. All of them are forwarded verbatim to
`extract_praat_parselmouth_features_from_audios`; the five `pitch_*` keys are also read by the
phonation-spans node's `derive_f0_range` and by VOICE's, through `f0_range_parameters`
(`nodes/common.py:357`, over `PITCH_NARROWING_KEYS` at `:344-351`), so the three call sites cannot
hold coefficients that drift. They are config keys rather than module constants because a coefficient
that decides a measurement's range is a parameter of the run, and `praat_parselmouth.py` sits in general senselab
and cannot read this config — so it carries them as keyword arguments with library defaults, and the
triage path passes these values.

praat_features.time_step_s 0.005 -- the frame shift every frame-based descriptor is computed on:
the pitch, intensity, harmonicity, formant and CPP tracks the forty scalars are pooled from. It is
the value the extractor itself has shipped since the module was written and the value every
Praat-derived number in senselab has been produced at; declaring it makes the pooling reproducible
from the config rather than from a signature default. Half the `phonation_spans.hop_s` used for the
F0/formant tracks, which is a different measurement with a different consumer (TAXONOMY reads the
tracks per frame; this section reads only pooled scalars), so the two are not required to agree.

praat_features.window_length_s 0.025 -- the analysis window the spectral moments are computed over,
25 ms, the same conventional short-time window `spectrogram.wideband_window_ms` names. Also the
extractor's own shipped value.

Neither is a threshold: nothing is compared against them and no verdict turns on them. They are
here because the measurement's provenance has to name the settings it was taken at, and a signature
default is not part of a run's recorded configuration.

praat_features.pitch_ceiling_quartile_multiplier 2.5 -- the only cited coefficient of the four.
Hirst 2011 §2.1 gives both values in the same lineage: `1.5 * q3`, credited to De Looze 2010, and
then "In the most recent implementation ... 2.5 * q3." Hirst & De Looze 2021 §13.3.4 splits them by
material -- `1.5 · q3` for non-emphatic speech, "something like 2.5 q3" for emphatic -- and Hirst's
shipped plugin (Nakala `doi:10.34847/nkl.5fb7xhhc`) matches that: `automatic_min_max_f0.praat`
computes `max_f0 = ceiling((q75 * 1.5)/10)*10` and switches to 2.5 under an `Expanded_pitch_range`
boolean. So the honest statement is not "the paper says 2.5, the script ships 1.5" -- both offer
both. It is: 1.5 is the non-emphatic default, 2.5 is the most recent implementation's value and the
emphatic one, and we take 2.5 because brief high excursions are the finding in this corpus rather
than noise to be smoothed away. Do not write `ceilFac`; that variable is in the third-party
`parantes/better-f0`, not in Hirst's tree. Hirst's first pass is 50--700 Hz in the paper and
60--750 Hz in the plugin; senselab's is `voice.f0_search_range_hz`, and that is the one real
divergence from his rule. What is adopted from Hirst is the two-pass structure and this coefficient,
and nothing else -- his floor is `0.75 * q1`, which senselab does not use, because the asymmetry is
his own empirical finding: "if the Pitch Floor is too low then we are likely to get octave errors
[...] Setting the Pitch Ceiling too high does not, however, seem to lead to any systematic errors."

praat_features.pitch_floor_divisor 1.5 -- senselab's own; cite nobody. The floor is
`max(search_floor, p5 / 1.5)`, which is -7.02 semitones off the 5th percentile and lands below
`0.75 * q1` on every source measured. Wider is the right direction for a corpus enriched for
pathological voices, where a range that excludes the voice is the failure that matters, and Hirst's
quartile floor is what misses a 100->400 Hz glide's low end at `0.75 * q1 = 107.2`. The q15/q65 pair
sometimes cited for this is real but never bare -- `q15 * 0.83` and `q65 * 1.92`, fitted in De
Looze's 2010 thesis against hand-annotated extrema and stated in De Looze & Hirst 2010 §3.1. Attach
no speaker count to it; the thesis's counts are 68, 53 and 10, and the 2008 paper concluded q25/q75,
with q15 appearing there only as a rejected floor candidate at coefficient 0.78.

praat_features.pitch_excursion_multiplier 1.5 -- senselab's own, and it has no derivation, which has
to be written rather than left to look like one. The floor's 1.5 has one (-7.02 semitones); this is
the same number reused as headroom above p95, and the adversarial case set does not discriminate it:
sweeping it over 1.0 ... 2.5 gives 13/13 at every step, printed by
`src/tests/audio/tasks/f0_range_probe.py`. Thirteen is the whole denominator: the probe's fourteen
sources include a 45 Hz fry that places no pitch at a 50 Hz floor and is counted as an absence, not
as a miss. So it is a declared convention whose only measured property is that the result is
insensitive to it across that span. State both halves --
insensitivity is not derivation, and it is also the evidence that no operating point was fitted
against this corpus. What the term is *for* is measured, and the two ceiling terms rescue opposite
cases: on a 0.4 s register break 110->440 Hz the contour is 110 Hz almost everywhere, so `q3 = 110`
and `2.5 * q3 = 275` clips the break while `1.5 * p95 = 660` (clamped to the search ceiling) keeps
it; on a 0.35 s emphatic peak 150->330 Hz the peak is too brief to reach p95 at all, `q3 = 150` and
`p95 = 165`, so `2.5 * q3 = 375` keeps the peak while `1.5 * p95 = 247.5` clips it. A sustained
excursion moves p95 and a brief one moves neither statistic much, so the larger of the two terms is
the one that has not been diluted -- which is the whole argument for the `max`.

praat_features.pitch_pinned_octave_ratio 2.0 -- senselab's own; cite nobody. When p95 falls below
twice the search floor the whole contour sits within an octave of the bottom of the search range,
the narrowing is untrustworthy, and the unnarrowed search range is used instead. Note what the
action is: the fallback does not raise the floor, it abandons narrowing. It is what catches deep
fundamental capture, which Hirst's ceiling coefficient alone does not -- measured, 330 Hz under a
strong 55 Hz component gives `[50.0, 137.5]` and 440 Hz under 60 Hz gives `[50.0, 157.1]`, both
ranges the voice never enters. Sweeping the ratio over 1.5 ... 3.0 gives 13/13 at every step on the
same thirteen tracked cases, so this too is insensitive across its plausible span rather than
fitted. Both sweeps come from the probe named above; do not quote a denominator it does not print. Three consequences to state rather than
discover. It fires on a clean 90 Hz buzz, because p95 = 90 is under 2 x 50 -- an ordinary low male
voice, not only pathology -- which is safe (a wide range never excludes the voice) but costs those
recordings the octave-error robustness narrowing buys, and the population it captures moves if
`voice.f0_search_range_hz[0]` moves. It fires on a 55 Hz fry, which is the right outcome:
contamination pushing the range wider is the safe direction. And residual capture is owed: the
fallback does not catch second-harmonic capture against a higher voice -- measured, a 220 Hz voice
under a 120 Hz-dominant hum gives p95 ~ 110 against 2 x 50 = 100, so it does not fire, and Hirst's
ceiling coefficient is what rescues that case here. That residual is the state of the art rather
than a gap in this work: Edlund & Heldner 2006 (`/nailon/`) §4.4 says "Correction for octave errors
is planned to go here as well, but not currently implemented", and Portnova et al. 2025, JSLHR
68:3568--3582 states this exact failure mode, reviews strategies for it including the two-pass, and
then resolves it by manual labelling. Mertens' *Polytonia* (2014) §5.5 is a published range
estimator with designed octave handling (discarding syllables >= 18 ST from the median), so "no
published estimator corrects this" is overstated; the citable claim is that no published method
tests whether its own first-pass distribution was octave-halved and acts on the answer.

Two mechanisms are recorded here as rejected rather than left to be re-proposed. A second-pass
median comparison -- re-run at the narrowed range and widen back when the two medians differ by an
octave -- is unreachable by construction: the narrowed range puts the first median near its centre,
so the second can deviate by at most a factor of `pitch_floor_divisor`, 0.585 octave. Measured, it
fired in 0 of 120 conditions, and it cannot fire at all when `p95 < 2 * search_floor / 1.5`, 66.7 Hz
at the current floor, which both mains frequencies sit below. And a log-Hz MAD trim before the
percentiles does not do the job it was added for -- an octave is 1.0 in log2 and a bimodal contour
widens the MAD, so a 2-MAD window keeps both modes -- while corrupting `pitch_frames`: measured on a
clean 110 Hz buzz it discarded 36 of 188 voiced frames, so the percentiles were of the post-trim set
and `pitch_frames` was not the voiced-frame count its consumers read it as. The 5th and 95th
percentiles already trim 10%.

The three percentiles (5th, upper quartile, 95th) stay module constants in `praat_parselmouth.py`
rather than config keys. They name *which statistic* each term is taken from, not how far it is
moved: changing one changes the rule's structure, and the keys above are the coefficients that scale
it. Percentiles commute with any monotone transform, so linear Hz and log-Hz give identical cut
points; the ratio margins are what carry the scale-freedom.

praat_features.pitch_pinned_percentile 95.0 -- the one percentile that is a key, because it is half
of a branch predicate rather than only an input to a ratio. `p95 < pitch_pinned_octave_ratio *
search_floor` decides whether the recording is narrowed at all, and moving 95 to 90 moves that
decision boundary as much as moving the ratio from 2.0 to 1.8 does, so by the criterion above it is
a threshold. It is deliberately one statistic serving two places: the same percentile is the
ceiling's excursion term, and the pair cannot move independently. That coupling is the design -- the
branch asks "does the top of this contour clear the search floor by an octave" and the ceiling term
asks "how much headroom above the top of this contour", and both questions are about the same top.
95 rather than the maximum because a single octave-halved frame would otherwise set both, and
rather than q3 because the excursion cases the ceiling term exists for (a 0.4 s register break) sit
above q3 by construction. Its own insensitivity is not separately swept; the two ratios that
multiply it are, and the sweep in the probe holds this percentile fixed.

The unit the pitch descriptors are reported in is deliberately not a key. It is baked into the
returned key names (`mean_f0_hertz`), so making it configurable would make the measurement's own
attribute names depend on the config, and every consumer would have to read the config to know what
to look up.

### praat_features.cpps

The twelve keys under `praat_features.cpps` are every setting the smoothed cepstral peak prominence
is computed under. They exist because `extract_cpp_descriptors` stopped calling Praat's `Get CPPS...`
and now computes the measure directly — log-power spectrum, cepstrum, robust trend fit, peak
prominence, frame by frame, with the two smoothing windows that are the *S* — so every number Praat
used to hold inside that one call is now a number this repo chose and has to declare. The rule the
implementation follows and the defects it replaces are in `praat-instrument-audit.md` step 4; what
is here is where each value came from. The triage path reads them through `cpps_settings`
(`nodes/common.py`), which builds the `CppsSettings` the extractor takes, and PREPROCESS stamps every
field into the `praat_features` measurement's own parameters under a `cpps_` prefix — so a stored
scalar names the settings it was taken at, which is the whole point of moving them out of the
function body.

**Two of the twelve are inherited rather than derived, and that has to be said rather than dressed
up.** `time_averaging_s` 0.01 and `quefrency_averaging_s` 0.001 are the literals the retired wrapper
passed, and they depart from Praat's own form defaults (0.02 and 0.0005) and from Hillenbrand's.
Carrying them forward is continuity — it keeps the new implementation comparable to the values
already in the corpus, which is the only thing being held constant across a change that alters
everything else — but most published CPPS was collected under different windows, so a value here is
not interchangeable with a published one. That is finding 11's trap stated in advance: a derivation
is evidence a decision was recorded, not that it was correct. Neither has been swept.

praat_features.cpps.peak_search_range_hz [60.0, 700.0] -- the F0 band the cepstral peak is searched
in, and the one value here that is a correction rather than a carry-forward. The retired code
hardcoded 60--330 regardless of the caller's ceiling (finding 4), penalising a 420 Hz voice by
3.8 dB -- roughly the whole normal-to-dysphonic span. 500 was considered and rejected in step 4:
untrained falsetto routinely exceeds 700 Hz, so an upward glide's endpoint sits above it and the
truncation is moved rather than removed. Measured here on a 420 Hz harmonic buzz: 23.53 dB at
60--700 against 18.98 dB at 60--330, a 4.55 dB penalty, which is the same defect finding 4 measured
at 3.8 dB on a different source. **This band is 100 Hz above `voice.f0_search_range_hz`'s ceiling of
600**, and that tension is owed rather than settled -- step 4 records it, and the honest statement
is that if falsetto above 700 Hz is real enough to move this band then finding 4's logic applies at
600 too. A second non-comparability is owed with it: at 700 Hz the peak quefrency is 1.43 ms, only
0.4 ms above the trend fit's 1 ms origin, where source and filter quefrencies are not separable, so
a CPPS at F0 700 is not the same quantity as one at F0 120.

**The band is fixed across recordings, not derived from each one.** An earlier draft of step 4 said
"from the recording's own derived F0", and that fails three ways: `derive_f0_range` *raises* on a
type-3 voice (`phonation/api.py`), so a derived band is unavailable on exactly the population this
reimplementation exists to serve; a prominence is measured against a regression over a quefrency
range, so a per-recording range would change the value and destroy the cross-recording comparability
`branch-conventions.md` requires of a per-measure band; and the published convention is a fixed wide
search.

praat_features.cpps.trend_range_s [0.001, 0.0] -- the quefrency range the trend line is fitted
over, and **a different thing from the peak-search band**. Conflating them is the error step 4
names: an earlier version read "declare 60--500 Hz" as the regression range, and fitting the trend
over 2--16.7 ms instead of 1 ms-to-end changes every value. 0.001 s is Praat's own fit origin and
the retired wrapper's; 0.0 is Praat's spelling for "to the end of the quefrency axis", not a
measurement of zero. The axis ends at 51.2 ms with the packaged `max_frequency_hz` and window, so
the fit spans 1--51.2 ms and the peak band 1.43--16.67 ms sits inside it.

praat_features.cpps.time_averaging_s 0.01 and praat_features.cpps.quefrency_averaging_s 0.001 --
the two smoothing windows, in that order: first a moving average across frames, then one across
quefrency within a frame. They are what makes the measure CPP**S** rather than CPP, and an earlier
version of step 4 omitted both, which would have shipped a differently-named quantity. Inherited,
not derived; see the paragraph above.

praat_features.cpps.max_frequency_hz 5000.0 -- the upper edge of the analysed band; the signal is
resampled to twice it before framing. The retired wrapper's value, and **this is finding 7's defect
declared rather than fixed**: 5 kHz was an unnamed Praat default that band-limits the spectral
moments too. Changing it is a separate decision with its own re-derivation, and it is a key here so
that the decision is visible instead of buried in a positional argument.

praat_features.cpps.pitch_floor_hz 60.0 -- sets the analysis window rather than the peak search;
the two were the same number in the retired code and are separate keys now because they answer
different questions. The window's effective width is three periods of this floor and its physical
Gaussian duration twice that, which is Praat's rule and stays a module constant
(`CPPS_WINDOW_PERIODS`, `CPPS_GAUSSIAN_WIDTH_FACTOR`) by the same criterion the three pitch
percentiles do: it names how the window is *shaped*, not how far a value is moved. At 60 Hz that is
a 100 ms window, which is why a recording under 100 ms places no frame and reports
`cpp_frames = 0.0` rather than a number.

praat_features.cpps.time_step_s 0.002 -- the hop between cepstrogram frames, the retired wrapper's
value. Not the same as `praat_features.time_step_s` (0.005), which is the frame shift for the pitch,
intensity, harmonicity and formant tracks: those are pooled per frame of a different analysis, and
the two are not required to agree. It is also the denominator of the time-averaging window, so the
first smoothing spans five frames at the packaged pair.

praat_features.cpps.preemphasis_from_hz 50.0 -- the pre-emphasis corner applied before framing, the
retired wrapper's value and Praat's form default.

praat_features.cpps.robust_tolerance 0.05 -- the relative slope change below which the trend fit's
reweighting has converged. The retired wrapper's value, passed to Praat's own robust fit; **the
direct implementation's fit is a Huber M-estimator, which is not Praat's**, so the number is carried
across a change of estimator and means "converged" rather than reproducing Praat's arithmetic. The
Huber tuning constant 1.345 (95% efficiency at the Gaussian) and the 50-iteration cap are module
constants for the same reason the window shape is: they name which estimator, not how far a value
moves. Validation that the whole chain is right rather than merely self-consistent: on
`src/tests/data_for_testing/audio_48khz_mono_16bits.wav` the direct implementation reads 5.376 dB
against Praat's own `Get CPPS...` 5.063 dB over the identical settings and band, a 0.31 dB
difference attributable to the estimator and the edge handling of the moving averages, and it places
2411 frames where Praat's cepstrogram places 2411.

praat_features.cpps.subtract_tilt_before_smoothing false, praat_features.cpps.tilt_line_type
straight, praat_features.cpps.peak_interpolation parabolic -- the retired wrapper's three values,
all previously unmentioned and all value-setting. `tilt_line_type` is worth naming twice: Praat's
CPPS convention is an exponential decay and this has always been a straight line, so the incumbent
was already not computing Praat's default CPPS. **The implementation supports only the declared
value of each and raises on any other**, which is deliberate: these are keys so that a stored scalar
names what it was computed under, and a setting that a run silently ignores is worse in a provenance
record than a setting that is absent. Implementing the alternatives is a separate decision, and each
would owe its own re-derivation.

Two things that are deliberately **not** keys. The voicing gate is gone rather than configurable:
the retired code computed CPPS per voiced interval, which is why it returned nothing on the
aperiodic voices the measure was promoted to find, and reintroducing the gate behind a flag would
reintroduce the defect. And there is no minimum value: the retired `> 4` cut deleted the dysphonic
range outright (finding 2), and a configurable floor is the same selection on the dependent variable
with a knob on it.

## diarization

Whole-file diarization as one shared PREPROCESS derivative: how many voices the recording holds,
and where each of them is. Owed by
[`specs/20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)'s
"Whole-file diarization as a shared derivative", and the derivative that closes it. The division of
labour it lands under: **PREPROCESS measures, branches refine their task spans, QUALITY judges
multi-voice against those refined spans after every branch.** Nothing in this section is a
threshold on a decision, because this block takes none.

diarization.model `pyannote/speaker-diarization-community-1`. Not a new choice: it is the default
`senselab.audio.tasks.speaker_diarization.diarize_audios` already resolves to when a caller passes
no model, and the block calls that task rather than pyannote directly. Naming it here rather than
inheriting the task's default is what makes a run's `config_hash` change when the backend changes.
The **claim that pyannote is reliable at the single-versus-multi-speaker distinction is uncited**,
as `20260913-branch-contract-and-hints/design.md` already records; `benchmarks/diarization.md` and
`benchmarks/glides-diarization.md` are the measurements to read it against. What *is* measured is
its ceiling: the seed-17 speaker-ceiling probe (TTS corpus, k=1..8, 20 sessions/k) found no
structural saturation -- predicted counts at k=8 spanned {5,6,7,8} -- and that backend's
`DiarizationCapabilities` record names it the strongest of the six at low k, which is the range
this derivative lives in.

diarization.revision `main`. The ref the spec is pinned *from*, not the ref anything loads through.
`PyannoteAudioModel` resolves it to an immutable commit at construction, and
`speaker_diarization/pyannote.py` resolves again and passes `revision=<sha>` to
`Pipeline.from_pretrained`, so the load is commit-addressed and the model agent records that SHA.
CLAUDE.md's rule -- a load must pass a SHA, never a ref -- is satisfied by the backend, which is
the reason the block goes through the task instead of calling pyannote itself. Pinning a literal
SHA here instead would freeze the corpus to one checkpoint and is the change to make when a
campaign wants that; it is not the default because nothing has yet been measured against a
specific commit of this checkpoint.

diarization.streams `[enhanced, residual]`. **Owner-directed on 2026-09-15, and settled.** The
question `20260913-branch-contract-and-hints/design.md` held open was raw-versus-enhanced, on the
grounds that the derivative's headline use is catching a quiet background talker and enhancement
suppresses exactly that -- which is also this project's standing conclusion for off-target speaker
detection (`SPEECH_DETECTION_SOTA_REVIEW_2026.md`). The owner's resolution dissolves the dilemma
rather than picking a side: **enhancement does not destroy the background talker, it partitions the
recording.** `residual = plain - g*enhanced` is already computed, lag-aligned, gain-fitted, written
as its own stream and already classified beside `enhanced` (`residual_yamnet_scores` and friends)
and already read by the ruleset (`airway.breath`'s `[residual, energy_fraction]`,
`taxonomy.ruleset.emptiness.peak_streams`). So both halves are diarized:

- **`enhanced`** answers *how many voices survived enhancement*, which for a single-participant
  protocol recording should be one.
- **`residual`** answers *whether a voice was removed*, which is the evidence that a quiet
  background talker was there at all.

Neither alone answers the owner's question and **the two counts are never summed**: one measurement
per stream, `enhanced_diarization` and `residual_diarization`, each with its own `signal`, its own
`derived_from` and its own sidecar. A disagreement between them is itself the finding. Cost roughly
doubles against a single-stream pass, which is why this is a list a campaign can shorten to one
entry rather than a literal.

Adding a stream costs one line and no code: `diarization: {streams: [enhanced, residual, plain]}`.
The measurement records the stream it was taken on, so two readings can never be compared as if
they were the same.

diarization.exclusive `false`. The overlapping view, not pyannote's exclusive partition. This is
the one setting here that is a correctness choice rather than a preference:
`speaker_diarization/pyannote.py` spells out that under the exclusive view "a per-instant speaker
count derived from these segments is capped at 1 by construction, so it reports *no overlap* as a
confident measurement rather than as something the input could not express". `community-1` computes
overlap internally -- its local segmentation model is `segmentation-3.0` -- and the exclusive view
discards it. Two people talking at once is precisely the evidence a multi-voice reading wants, so
the block takes the view that can express it. The speaker set is identical under either view; only
`overlap_s` and `max_concurrent_speakers` differ, and under the partition both would be structural
zeros.

diarization.min_speakers, diarization.max_speakers: null, meaning **no bound is handed to the
clustering**. Null here is a stated choice, not an unmeasured one, and it is the same shape as
`windows.yamnet.label_thresholds` and `voice.f0_range_by_population`: an override a caller may
supply, absent by default. Neither bound is fitted on this corpus, and a `max_speakers: 2` would
be worse than unfitted -- it would cap the measurement at the answer, making "three voices"
unreportable on the very recordings a multi-voice reading exists to find. Pyannote is the only
backend of the six that acts on these hints at all
(`DiarizationCapabilities.honors_speaker_hints`), so stating one is a real change to the estimate
and owes its own measurement.

**No minimum duration key.** One was considered and is deliberately absent. A floor would turn a
measurable recording into an absence, and the corpus is short -- median 7.0 s, mean 15.0 s -- so
any floor worth having would fall inside the bulk of it. Probed directly on 2026-09-15 against
community-1 on CPU, head-truncated clips of a real b2ai recording at 0.05, 0.1, 0.25, 0.5, 1.0 and
2.0 s all returned without raising, so there is no structural length below which the model refuses
and no measurement a floor could be read off. The measurement carries `duration_s`, so a consumer
that wants to discount a short reading has the number to do it with.

**What is recorded, and where.** Per stream: `n_speakers` is the headline the owner asked for
("pyannote to see if it's 1 or more"), with `speakers`, `n_segments`, `per_speaker_s`, `speech_s`,
`overlap_s` and `max_concurrent_speakers` beside it -- all attributes, because a consumer answering
"one voice or more" must not have to open a file. The segment table is an `.npz` sidecar under
`derivatives/<stream>_diarization.npz`, named by the entity through its path and SHA-256, following
the posteriorgram: what grows with the recording's length is written beside the run, never inlined.
Its `starts`/`ends`/`speakers`/`streams` are four parallel columns of one table, so concatenating
the two streams' files gives a self-describing segment list a consumer intersects with a span by
arithmetic alone.

One measurement per stream rather than one carrying both, so `signal` keeps meaning what it means
everywhere else in the store and each entity's `derived_from` names exactly the stream it was
measured on -- the same shape `enhanced_yamnet_scores` and `residual_yamnet_scores` already use. It
also makes each stream its own block, so a run whose `residual` was never written still gets its
`enhanced` reading and records the other as its own absence, and an extend pass that widens
`streams` measures only what is missing.

The segments are deliberately **not** written as `span` entities, which would be the other
idiomatic home for timed regions: `live_entities(store, "span")` is the branches' set of candidate
task spans, and a per-speaker time partition is not a candidate for anything -- putting it there
would widen an existing consumer's input set as a side effect of adding a measurement. Attribution
of a speaker *to* a span is a live design question with a contract tension; it is written up in
[`../20260915-preprocess-diarization/design.md`](../20260915-preprocess-diarization/design.md) and
is not decided here.

**Zero is a measurement.** Measured on the b2ai sample on 2026-09-15: every `Respiration-and-cough`
recording (breath, cough, five-breaths, three-quick-breaths) diarized to **zero** speakers and zero
segments, while every `Story-recall` and `Harvard-Sentences` recording returned exactly one. A
recording whose segmentation finds no speech has no speaker, and that is an answer with a value,
not an absence -- the distinction this repository has paid for before.

## voice

VOICE's F0 ranges and task-duration expectations.

VOICE additions (plan-nodes-2 N21, N22, N25); period_doubling_factor and voice.hint_tags are
derived in the paragraphs above. UNSET, continued:
  phonation.hnr_floor_interval_db, phonation.rms_floor_interval: the near-edge intervals were
  measured as normalised autocorrelation -- the same (0.44, 0.933) and (0.0007, 0.0161) named
  above -- and the gate now reads Praat harmonicity in dB and RMS, so the units do not transfer.
  Praat calibrates neither: its harmonicity has a silence_threshold relative to the global peak
  (already `phonation.silence_threshold`), and no dB floor or RMS interval at all, so unlike the
  F0 range these two cannot be resolved by deriving them per recording and stay null pending a
  measurement in the implementation's own units. While null the near-edge flag is inert.
  gate_interval is three-valued: "measured" iff both intervals are supplied, "partial" when exactly
  one is, "unmeasured" when neither -- each family's near-edge check arms independently with its own
  interval.

voice.f0_search_range_hz replaces voice.f0_range_hz, which replaced phonation.f0_min_hz and
phonation.f0_max_hz. It is [50.0, 600.0] Hz: the wide first pass each recording's own
[floor, ceiling] is narrowed from. The two-pass structure is Hirst 2011's, whose own first pass is
50--700 Hz in the paper and 60--750 Hz in his plugin; the coefficients that do the narrowing are the
five `praat_features.pitch_*` keys above, one of them his and four senselab's own. PREPROCESS and
VOICE both narrow it the same way, off `plain`, and both read the coefficients through
`f0_range_parameters` (`nodes/common.py:357`), so the two cannot hold ranges that drift. What was wrong
with the key it replaces is its premise, not its value: a fixed corpus-wide range had to be null,
because no single range serves both a low adult male fundamental and an infant voice -- but no fixed
range is needed, because the range is derivable per recording. Do not cite
`doi:10.3758/BRM.41.2.318` for any of that; Vogel et al. (2009) recommends fixed sex-specific
settings and rejects per-recording derivation as impractical at scale. Two bounds are owed rather
than settled: the 600 Hz ceiling against the CPPS band's 700 Hz (`praat-instrument-audit.md` step 4),
and the 50 Hz floor, which decides which ordinary low voices take the pinned-contour fallback. See
`specs/20260911-ppg-praat-batch/design.md` and `praat-instrument-audit.md` step 2.

voice v2 -- branch-voice.md. voice.f0_range_by_population replaces the derived range per declared
age and sex; null, owed a fit per population, and a range spanning too wide an interval makes any
period-doubling test on it vacuous. voice.f0_range_ratio_max is the f0_max / f0_min above which the
period-doubling check reports nothing because it flags everything; it is null, and a configuration
exceeding it is REFUSED AT LOAD rather than run and flagged -- a check that fires on every file
transports no information, and running it anyway would put that non-information into every verdict.
While the ratio is null no configuration is refused, which is the honest state: nobody has fixed the
bound. voice.task_duration_ranges is the expected duration range per declared task, against which a
span outside it flags with the declared range named; null, because no corpus here establishes what a
maximum-phonation-time task should produce.

## speech

SPEECH's enrollment, separation, diarization and non-target settings.

speech v2 -- branch-speech.md. speech.enrollment_model names the speaker-embedding model AND its
revision that enrollment is estimated with; null, and while null an enrollment is refused rather than
compared. speech.separation_backend chooses between unasdiff in speech_sound mode and
MossFormer2_SS_16K; null until the two are ranked on this corpus, and while null separation does not
run. speech.separation_sound_class is the FSD class name unasdiff's sound slot is conditioned on.
It is null for a DIFFERENT reason from every other null here, and the distinction matters: nobody
needs to measure anything for it. branch-speech.md says the slot stands for any background and
should not be conditioned on a class, and separate_audios refuses speech_sound without one ("index 0
is 'Hi-hat'"), so THE CAPABILITY IS ABSENT UPSTREAM. It is settled by adding an unconditioned sound
slot to unasdiff, or by someone naming a defensible class and saying why -- not by a ROC, a corpus
or a fit. Until then the unasdiff option cannot run. speech.nontarget.{level_db,tilt_db_per_octave,d_to_r_db} are the proximity
leg's three thresholds, each null; until all three exist the legs are measured and reported per span
and nontarget_speech_s is written as null rather than zero.

Hint tag vocabularies (speech.hint_tags, voice.hint_tags) -- which may_contain tags count as a
caller asserting each kind, seeded from the design documents' own member names. Vocabulary, not
fitted; extended by override.

## taxonomy

`taxonomy.consolidation_floor` **0.2** -- owner-directed, applied to every classifier, both to
TAXONOMY's consolidation and to the union that selects the figure's raster rows.

Raised from 0.1 on 2026-09-05. The reasoning is about what these numbers are: YAMNet's and HeAR's
outputs are **probabilities**, and this corpus is single-speaker clinical recordings, not complex
mixtures with many concurrent sources. A label a classifier gives 0.15 to, in a recording that
contains one person doing one task, is far more likely to be the tail of a 500-label distribution
than a second sound genuinely present. The floor was originally 0.1 and applied only to YAMNet, on
the argument that HeAR's eight labels are all health events worth seeing; that exemption was
withdrawn when a rendered page showed HeAR painting all eight in every span, which made the real
coughs unfindable.

Measured on `Story-recall-(v2)`, whose HeAR label peaks over the file are Speech 0.781, Laugh 0.717,
Snore 0.677, Baby Cough 0.414, Breathe 0.377, Sneeze 0.190, Throat Clear 0.139, Cough 0.127. At 0.1
all eight survive; at **0.2 the surviving five are Speech, Laugh, Snore, Baby Cough and Breathe**,
and the three that go are the ones under 0.2. YAMNet is unaffected on this recording: only Silence
and Speech clear either value, everything else being under 0.08.

This is a declared default, not a fitted one. What would upgrade it is per-window labelled verdicts
over more than three subjects -- the ROC that `windows.<classifier>.default_threshold` is still
owed. Note the two are different quantities: this floor governs which labels are worth
*consolidating and drawing*, while `default_threshold` governs which become labelled window
entities in the store.


The evidence fold: per-kind presence floors and label vocabularies.

Taxonomy label vocabularies -- semantic mappings, not thresholds: which of each detector's labels
can express each kind, read off the label inventories (AudioSet's 521, HEAR_EVENT_LABELS' eight,
CrisperWhisper's bracketed non-lexical tokens). Not fitted; overridable. benchmarks/taxonomy.md
records why no single AudioSet roll-up label exists.

taxonomy.airway_ontology_roots [Respiratory sounds] -- the airway vocabulary is no longer a list of
labels. It is the AudioSet ontology subtree below these roots, resolved through the
classifier-ontology profile: the AudioSet evidence labels are the closure minus every node in
neither AudioSet's released 527 nor YAMNet's 521, and the HeAR evidence labels are those whose
mapped node falls inside the same closure. One key, so the two vocabularies cannot drift; the two
hand-listed keys it replaces had, and carried Sigh while missing Pant and Snort. Not fitted;
widening the kind is a second root. specs/20260910-classifier-ontology-mapping/design.md records the
three findings, the emittability rule and what evidence would put Sigh back.

taxonomy v2 -- taxonomy.md. TAXONOMY runs no models and folds stored evidence only, so the v1
min_families committee and its per-detector floors are deleted rather than re-derived. Each kind now
has named evidence LINES with their own floors: speech has authoritative lexical (word count) evidence
plus an acoustic (window count) corroboration line; airway has health_acoustic (HeAR window count) and
acoustic (AudioSet window count); voice has neither, being classified from phonation-span duration alone.
presence_floor values are counts, not scores, and all four are null. voice_min_duration_s and
voice_uncertain_duration_s are the two
duration cutoffs, both null, both owed a fit across voiced, unvoiced and mixed production.
taxonomy.speech_labels replaces audioset_speech_labels and is null because the v2 spec owes it the
AudioSet speech FAMILY and the v1 list carried one member. The consensus transcript is authoritative
for speech presence: a completed empty consensus remains speech-absent despite an isolated acoustic
label, which is retained as corroboration. lexical_airway_tokens is deleted: airway's lexical evidence
line no longer exists, because a bracketed event is not a word and carries no lexical evidence at all.

taxonomy.airway_bracket_tokens [breath, cough, throatclearing, sniff] -- which typed bracketed
consensus tokens are an airway event, read by the family taxonomy ruleset's airway.bracketed_event
gate. This does NOT reinstate lexical_airway_tokens or a lexical evidence line for airway: a
bracketed event is still not a word, and this key names it as an acoustic detection with a timing
rather than as lexical content. [uh] and [um] are deliberately absent -- they are fillers and they
separate the other way. The counts behind the set, and why the gate reads extracted typed counts
rather than the capped consensus transcript, are in family-taxonomy-ruleset.md.

taxonomy.ruleset -- the whole ruleset block (reference_family_set, branch_gates, branch_flags,
emptiness, gates) is derived in family-taxonomy-ruleset.md, keyed by gate name, and is not repeated
here.

## quality

SQUIM and disruption tolerances, plus QUALITY's clip-consistency check. The four SQUIM and
disruption keys are read by nothing and no derivation was written for them.

quality.clip_contradiction_margin 0.005 and quality.clip_edge_guard_samples 3 --
specs/20260912-quality-clip-consistency/design.md. Both belong to the clip-contradiction rule, which
is read twice on the same numbers: PREPROCESS's `_clip_spans` withdraws a candidate whose peak an
unclipped sample elsewhere in the recording exceeds, and QUALITY contests any written span that
still does. One key each, read by both, so the audit's count stays a statement about the detector.
Neither is fitted; both are restatements of ClipDaT's own two constants on the same signal. The margin is
1 - clipping.near_threshold: a sample within 0.5% of a clip level is one the detector would have
counted as part of that run had it been contiguous with it, so it is not evidence against the level
(the exactly symmetric figure is 1/0.995 - 1 = 0.005025). The guard is clipping.leniency_samples:
a run closes at the sample where its own leniency was exceeded, so the samples immediately outside
an edge are that run's decay at the detector's temporal resolution, not independent evidence. The
margin is relative and int16 quantisation is absolute, so for clip levels below 6.1e-3 the margin is
narrower than one quantisation step; clipping.minimum_extreme is 1e-4, which leaves that window open
for near-silent files. It is named in the design rather than closed with a second, unmeasured
absolute floor.

## disruptions

Clip, dropout and discontinuity counting over each span.

Disruption parameters -- clip_headroom, min_clip_run and min_dropout_ms are conventional. A single
sample at full scale is not clipping, which is what min_clip_run is for. The counts these produce
are exact; what has no measured value is the tolerance, which is quality.disruption_* below.
All of them are measured on the ORIGINAL recording rather than on the plain stream:
peak-normalisation and resampling between them destroy the flat plateaus clipping consists of, and
clipped_runs read 0 on every campaign file including four whose originals peak at exactly
0.0 dBFS. A resampled copy is the wrong instrument for a defect of the recording.

disruptions.discontinuity_local_factor 10.0 and discontinuity_window_ms 20.0 --
benchmarks/disruptions.md. The criterion this replaces was an absolute 0.5 sample-to-sample jump,
which measures high-frequency energy rather than defects: a full-scale 3 kHz tone at 16 kHz steps
1.18 between neighbouring samples at every zero crossing, and the rule called 11999 of them
discontinuities in one second, 800 on one campaign recording of loud speech. A jump is now
referenced to the signal's own local variation -- the larger standard deviation of the two
20 ms windows flanking it, neither containing the jump, and a standard deviation rather than an
RMS so a constant offset is not read as variation. Measured over 806142 samples of peak-normalised
clean speech from three recordings at 16, 24 and 48 kHz: the largest ratio observed was 9.64 and
the 99.9th percentile 3.90, so a factor of 10 flags none of them, against 376 flagged by the
absolute rule on the same audio. The 20 ms window is the conventional short-term analysis frame,
the same length spectrogram.narrowband_window_ms uses, and is not separately fitted.

disruptions ZCR -- zero_crossing_rate is a plain reading of each span in crossings per second,
reported and never gated. It has no threshold because nobody has derived one.

## pii

The detector set a PII scan is required to have run.

pii.required_detectors [gliner, presidio, rules] -- vocabulary, not fitted: it is the output of
the pii_detection module's own default_detectors(), which is the set scan_for_pii runs when the
caller names none. Nothing about the list is measured; what it buys is that "the scan is complete"
stops depending on the host. A detector that was never ATTEMPTED recorded neither a scan nor a
failure, so two hosts disagreed silently about the same recording: one ran [presidio, rules] with
failed={} and read as complete, the other attempted gliner, recorded its failure and withheld.
Completeness is now required-subset-of-scanned AND failed empty, with a detector in required but
neither scanned nor failed recorded as missing. Narrowing the list is how an operator who runs
fewer detectors says so; the config_hash then names that choice. The local-LLM detector is
deliberately absent for the same reason it is absent from default_detectors(): it is never
default-on, so requiring it would make every scan incomplete.

## redaction

What a released transcript replaces, and by how much padding.

Redaction rules, which are validity checks rather than values. redaction.padding_ms is required to
be a non-negative whole number of milliseconds at REDACT's entry: a negative margin narrows every
extent instead of widening it, so the audio keeps the name while the verdict and the transcript both
report a redaction, and neither verification channel can see the difference. REDACT runs no
recognizer: verification is a re-scan of the redacted consensus text, judged complete by the same
pii.required_detectors rule the planning scan is, and a re-scan that skipped a required detector is
a flag. A consensus word the store places nowhere overlaps no planned extent, so it is released as
[UNPLACED] rather than verbatim and counted in unplaced_words_n -- text of unknown location cannot
be shown to be safe.

redaction.fill -- redact.md leaves this DEFERRED: which of silence, noise or bleep is least damaging
to the measurements taken downstream of a released artifact has not been measured, so the key ships
null and a run must declare the fill it used. silence and bleep are implemented; noise raises rather
than shipping an unmeasured spectral shape, because "speech-shaped" names a shaping nobody here has
fitted. redaction.bleep_hz 1000.0 is the conventional broadcast censor tone -- a presentation
choice, declared rather than defaulted silently, and not fitted. The bleep is scaled to the
extent's own PEAK, so its RMS is peak/sqrt(2) and the masked extent comes out louder than what it
replaced by that extent's crest factor over sqrt(2): measured 1.00x on a pure tone and 3.19x on
the gaussian-noise fixture the node's own tests use (a review probe on speech read 2.34x,
consistent with the closed form, but no fixture in this tree reproduces it). That is a
presentation choice too, and equally unfitted -- matching the extent's RMS instead would change
how loud a released artifact's redactions are without anyone having measured which is preferable.

## residual

The background-residual PREPROCESS block: `plain` and a lag-aligned, gain-fitted FRCRN_SE_16K
enhancement, both written as streams and both classified. See `preprocess.md`'s own section for the
algorithm. There is no energy-fraction gate here any more -- see the note below.

residual.enabled false. Off by default: ~22 s of GPU work per recording, and the residual is not
wired into any branch or decision yet -- the owner has not taken that decision -- so it would cost
every run without being read by anything.

residual.max_lag_ms 200.0. The cross-correlation search half-window FRCRN's output is aligned to
`plain` within. Not fitted: 200 ms is generously wider than any lag observed (0 ms on every
recording measured, both in the buzz-separation comparison and the no-speech benchmark), so it is a
safety margin rather than a value read off data.

residual.bands_hz `[[0, 200], [200, 1000], [1000, 4000], [4000, 8000]]`. The same four-band split
`subtract.py`'s own reporting uses in the buzz-separation comparison, reused rather than refitted so
the residual's band report reads against that comparison's own numbers directly.

**No gate here any more.** This section previously carried three energy-fraction keys
(`min_enhanced_energy_fraction: 0.50`, `max_energy_fraction: 0.90`, `min_energy_fraction: 0.005`)
derived from `residual-without-speech-2026-09-08.md`'s eight-recording local measurement. The owner
removed them: PREPROCESS measures, it does not judge whether a residual means "background" -- that
question is answered by looking at what `enhanced` and `residual` were each classified as
(`enhanced_yamnet`/`ast`/`hear` and `residual_yamnet`/`ast`/`hear`), which now exist for every run
this block completes, not by a threshold on an energy ratio computed before any classifier runs.
The bimodal local behaviour the 0.50 gate was fitted to (94-98% enhanced-energy retained when FRCRN
passes an input through, 0.01-6.7% when it nulls one, nothing observed in between) is still true as
an observation -- reproduced again this session, unchanged, both by rerunning the original benchmark
against the now-shared library and by an independent raw-vs-`plain`-feed rerun. But it is no longer
reconciled with the cluster run: 61 of 70 non-speech recordings there had FRCRN's enhanced output
retain ~100% (pass-through), not null, where this session's local measurement -- on the identical
recordings, both fed raw and fed the pipeline's own `plain` conditioning -- keeps nulling them.
Feeding FRCRN the raw file vs. the `plain` stream was tested directly and ruled out as the cause (see
`benchmarks/residual-without-speech-2026-09-08.md`'s 2026-09-07 update); the best-supported remaining
account is a difference in FRCRN's execution environment between this machine (CPU) and the cluster
(most likely GPU), surfacing only on non-speech input that sits outside the model's training
distribution -- not confirmed by directly reproducing the cluster's numbers on a GPU in this session.
**The 0.50 gate's derivation does not survive**, independent of the gate being removed: it was fitted
on a measurement that does not reproduce across the two environments the pipeline actually runs in,
and no account has been confirmed for why.

The `speech_present`, `n_consensus_words` and `speech_coverage_fraction` fields this block writes on
the `residual` measurement carry no config key of their own: `speech_present` is
`n_consensus_words > 0`, an identity rather than a fitted threshold, matching the "no new threshold"
rule the `speech_overlap == 0.0` windows split already follows.

## report

The presentation form the summary is written in.

report.format pdf. report.md calls it a presentation choice owed no measurement, but one that "must
be declared rather than defaulted silently". Declaring it here IS the declaration: a null would make
the one product report.md requires on EVERY file and EVERY outcome unreachable under the packaged
config, which is a worse failure than a defaulted presentation choice -- the run would emit nothing
for the file it most needs to emit something for, the one ADMIT refused. It shipped png first, on
the reasoning that a single uncut canvas is the only form in which the lanes and the spectrogram
stay registered against each other and that paginating would break the alignment the summary exists
to show. The measurement that settled it against png is the same one that motivated it: the first
real rendering was about 32 inches tall, and the owner's objection was that nobody reads a 32-inch
image. The blocks are what made it that tall, and they share no axis with anything -- they are
prose beside the picture, not a lane in it. So the pdf splits on THAT seam and only that seam:
page one is the aligned panels, uncut and registered exactly as before, and page two is every
block. No alignment is broken because nothing aligned was paginated. png stays one override away
and carries identical claims, with the blocks back on the canvas; it is the better choice for a
viewer that scrolls rather than pages.

## unset

Keys deliberately left null, and what each one owes.

UNSET, and why -- benchmarks/open.md carries each of these:
  redaction.padding_ms: must exceed the *worst* consensus-word edge error, which is unquantified. The
    median will not do -- of the two boundary failures, an audible fragment of a name and a clipped
    neighbour, only one is recoverable.
  speech.second_diarizer: no measured ranking of second diarizers exists; while null, a count of
    not-1 records second_diarizer "not_consulted" and still flags.
  speech.target_match_cosine: no similarity threshold has been derived; a hint carrying a target
    embedding under this null is refused rather than answered with an invented cut.
  speech.speech_test_stoi_floor, speech.speech_test_si_sdr_floor: SQUIM thresholds over speech
    spans are unmeasured; while null each span's corroboration records squim_vote "not_evaluated"
    and YAMNet coverage alone decides. Distinct from quality.stoi_floor/pesq_floor: these gate
    step 3's "is this span speech" vote, those gate step 8's quality reading.
  phonation.hnr_floor_interval_db, phonation.rms_floor_interval: benchmarks/voice.md measured a
    near-edge interval in normalised-autocorrelation units, which do not transfer to the Praat-dB
    implementation, and Praat self-calibrates neither, so neither can be resolved by derivation;
    while null the near-edge row is inert and the verdict records "unmeasured".
  quality.stoi_floor, quality.pesq_floor, quality.disruption_*: no labelled quality verdicts exist,
    so SPEECH's quality fail is unreachable by design until they do. Reserved: read by nothing yet;
    SPEECH step 8 reports without gating until these are measured AND wired.
