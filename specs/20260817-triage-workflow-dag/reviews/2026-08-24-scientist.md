# Speech/voice-scientist review — code-first, then spec (2026-08-24, @ bec94fe9)

Protocol: phase 1 read only the implementation, the underlying task modules, and the 28-file
campaign stores (`artifacts/triage_b2ai28_v2/`, all numbers recomputed from the per-run
`store.jsonl`); phase 2 cross-checked the specs. Corpus context: behavioural task recordings,
quiet indoor settings, single intended target speaker, but multiple speakers and background
conversations do occur. One methodological bound on everything below: one speaker, one session,
28 files, seven override parameters declared as guesses — refutations are licensed on this
evidence; new thresholds are not (`benchmarks/voice.md:13`).

## Findings, ranked by scientific severity

**S1 — The fabrication test has measured sensitivity 0/28 on exactly its target population, and
the spec's planned completion (periodicity) cannot fix it.** `speech.py:367-379` tests only
envelope-over-floor inside the word's extent; hallucinated words sit on vowels and DDK bursts
20-35 dB above floor. 0 candidates across 28 stores while 10 wordless files produced PII findings.
A sustained vowel is the most periodic signal in the corpus — the periodicity leg would CONFIRM
the hallucinations. The missing legs are lexical/temporal: measured on the campaign,
(i) word duration — 0/519 lexical words > 960 ms vs all 5 words > 1 s on non-lexical files
(3.15–15.36 s); (ii) token-repetition degeneracy — lexical max-repeat/n 0.05–0.12 vs DDK
0.38–1.00; (iii) SQUIM STOI — lexical spans 0.91–0.99 vs breath/cough/vowel 0.38–0.65 (fails on
DDK 0.98-0.99, which is acoustically speech-like). MPT-2: one 15.36 s "word" covering 15.4 s of an
18.3 s file at YAMNet Speech coverage 0.00 was scanned, produced PERSON, and withheld the release.
YAMNet-silence overlap would not help: 26/28 files have 0% certified-silent windows in a quiet
room.

**S2 — The hint's most informative fields are never read.** `targeted_speaker_count`,
`environment`, `metadata.speech_type` are unread anywhere in the tree; hints are one-directional
(absence→flag only); `count != 1` is hard-coded (`speech.py:525`) rather than compared to the
declared count. The task identity — known with certainty — would have prevented every false PII
withhold.

**S3 — HeAR's label floor is refuted by the campaign's own distribution.** Claimed
(`default.yaml:87-89`): winners 0.940–0.996, runner-up ≤ 0.41, floor 0.5 in an empty interval.
Measured: 170 spans → 14 labels (8.2%), winners 0.501–0.821, none above 0.821, two within 0.005 of
the floor. Recall inversion: the two dedicated cough files (spans up to 81.9 dB over floor) got
ZERO labels; DDK bursts scored Cough 0.780/0.528; a sustained vowel got Breathe 0.501. Mechanism:
71% of spans are < 2 s and zero-padded — which `hear.py:56-61` itself says "destroys the
representation"; spans > 2 s take the unmeasured sliding path under the same floor.

**S4 — The K=18 dB airway gate has ~1 dB of margin on quiet breaths and was fitted on coughs.**
Breath-1's events: 18.5, 18.9, 19.0, 21.1 dB over floor. The benchmark's 53–57 dB is a cough
figure. The respiration protocol spans a 30 dB level spread (RMS −38.5 to −6.8 dBFS), so detection
is level-limited. Also the offset rule merged ThreeQuickBreaths-2's three breaths into one 6.58 s
span (the mechanism behind comparison.md's "unexplained" airway miss on that file).

**S5 — VOICE's HNR gate uses a per-fragment reference level, inflating voiced time on breath
6.7×.** `voice.py:310-320` runs `hnr_track` on each residual interval as its own Sound; Praat's
silence threshold is relative to that Sound's own maximum, so slicing first renormalises the
criterion. Probe: Breath-1 frames ≥ 5 dB HNR = 39.4% per-fragment vs 3.1% whole-file; MPT-1 92.4%
vs 82.5%. This is the mechanism behind Breath-1's 132 "voiced" runs / 5.36 s at reported f0 397–412
Hz, and it directly contradicts `benchmarks/voice.md:46-48` ("Breath is unvoiced, the gate rejects
it, and that is correct"). Fix is one line: compute tracks once on the stream, slice the arrays.

**S6 — One breath inside a speaker turn zeroes the file's speaker count.** pyannote returns one
exclusive segment on connected speech; any-overlap withdrawal (`speech.py:114`, N10) is therefore
all-or-nothing: Story-recall (110 words, one Breathe label) → count 0. speaker_count = 0 on 10/28
files — outside the declared codomain {1,2,≥3}; count==2 and ≥3 never fired. Cascade: all words
in_withdrawn → no target → "pii found and no target speaker known" (14 files) → withhold. The
withdrawal rule shipped untested on the case it exists for (`benchmarks/diarization.md:35` admits
the benchmark recording lacked it).

**S7 — REDACT's verification is a second sample from the hallucination process on a DIFFERENT
signal.** Known half: planner scans consensus words, verifier re-transcribes (F-E). Unstated half:
PREPROCESS transcribes `plain` (mono, 16 kHz, peak-normalised) while REDACT re-transcribes
`recording` (original rate and level) — and `model-to-branch.md:60-67` already documents that this
exact difference changes CrisperWhisper's output. Part of the verification's false-positive rate
is a preprocessing artefact; verify on the same conditioned stream.

**S8 — Corroboration is inert by design.** No code path or spec sentence makes span survival
depend on the vote; with SQUIM floors null there is nothing to agree with; 13 YAMNet disconfirms
across 8 files changed nothing. Unit conflation: `yamnet.coverage_threshold` = 0.5 serves as a
per-window score floor (`speech.py:181`), a fraction-of-windows floor (`:431`), and a score-only
floor in AIRWAY — one key, three semantics, one derivation (score only).

**S9 — MPT is not recoverable from VOICE's output.** `branch-voice.md:91` defines MPT as a run
duration; `:64` forbids merging; a 10 ms-hop double-threshold gate without hysteresis chatters:
MPT-1 = 37 runs over 7.09 s. MPT-2's vowel was consumed by the hallucinated ASR word, leaving
VOICE 0.34 s — the clinical measurement determined by where a hallucination landed. On connected
speech, 73–99% of the residual lies inside the diarization interval (median run 25–40 ms): it is
the target's own inter-word voicing, and `f0_median_hz` (356–420 Hz) is computed from the biased
long-run minority and reported unqualified.

**S10 — The period-doubling alias check is vacuous by construction at this range, and the stated
boundary is off by an octave.** Unflagged band is (f0_max/2, 2·f0_min), non-empty iff
f0_max < 4·f0_min — TWO octaves, not the override comment's one. At 75–500 Hz: 100% flagged.
Policy: require f0_max/f0_min < 4 at validation; set ranges per task/population; two-pass estimate
when sex unknown; better, replace the range-property test with a mark-sequence doubling detector
(alternating periods / bimodal histogram / subharmonic energy).

**S11 — YAMNet confirm/contest is decided by window-scale mismatch and an incomplete map.** The
winner is the count-argmax over all 521 labels in 0.96 s windows: a cough inside a sentence loses
to Speech (5 of 9 contests), a quick breath reads Explosion. `Snoring` is airway evidence in
`taxonomy.audioset_airway_labels` yet contests Breathe in AIRWAY's confirmation map — the same
label supports and contests airway simultaneously; the nasal-breath→Snore prior is documented in
the project's own benchmarks and unhandled.

**S12 — Instrument validity notes.** SQUIM separates lexical from breath/cough well but fails on
DDK and vowels (speech-like); a single global STOI floor is unsafe — pair it with SI-SDR or
duration. YAMNet coverage is structurally wrong for < 1 window spans (60 ms span → coverage 0.00);
no minimum-span rule exists. Clipping detection runs on `plain` after peak-normalise + resample,
which destroys flat plateaus: clipped_runs = 0 on every file including four peaking at 0.0 dBFS —
and the 7 wordless files get no disruption measurement at all (span-scoped).
`discontinuity_threshold` 0.5 absolute measures loudness (800 hits on Picture-description), not
defects.

## Multi-speaker / background conversation (the stated recording reality)

- Nothing distinguishes target from background speech at the presence level; with min_families=1,
  one YAMNet window or one transcribed word suffices, and `benchmarks/snr.md:17-19` shows YAMNet
  Speech holds 0.987–0.998 at SNRs where the envelope fails — a background conversation is
  indistinguishable from the target.
- The measured 12 dB speech envelope gate is dead code (`default.yaml:36-37`): there is no
  acoustic speech-activity detector independent of ASR. Untranscribed background talkers don't
  exist; transcribed ones enter spans, PII scan and redaction plan as the target's.
- Overlap is structurally undetectable: `speech.py:481` uses the exclusive diarization view, whose
  own module docstring warns per-instant count is capped at 1 by construction.
- The [first word, last word] crop hides pre/post-task bystanders unless ASR transcribed them.
- Unattributed words' PII is "treated as the target's" (N12) — privacy-conservative,
  scientifically wrong: a bystander's identifier attributed to the participant.
- The archived off-target design had the right instrument (level, spectral tilt,
  direct-to-reverberant proximity leg — `archive/decisions.md:364-366`) and was retired as moot
  (`open.md:56-59`) with nothing replacing it. For this corpus, the largest capability gap.

## Spec-change list (top of 20; full table in the review transcript)

1. `branch-speech.md` §1 + open.md: replace the fabrication test's stated completion — three
   lexical/temporal legs (duration, repetition, SQUIM speech-test), gating the PII scan, not
   flagging. Blocking measurement mostly done from campaign artifacts; needs a second session
   before thresholds ship.
2. `branch-speech.md` §7: add the fourth limit — the scan is also an UPPER bound; hallucinated
   identifiers withhold releases nothing justifies.
3. `preprocess.md`: record `no_speech_prob`/`avg_logprob`/`token_entropy` (already on ScriptLine,
   discarded) as evidence.
4. `branch-speech.md` §4 + N10: portion-only (or fractional) withdrawal; codomain gains 0 with its
   own reading; record withdrawn_s/segment_s.
5. `branch-speech.md` §4: state exclusive=True's consequence or switch and report an overlap track.
6. New `benchmarks/nontarget.md` + branch-speech §9: reinstate the presence-level non-target axis
   (level/tilt/D-R proximity leg); `nontarget_speech_s` verdict field.
7. `benchmarks/voice.md`: retract "breath is unvoiced, the gate rejects it"; require
   whole-stream track computation (fix is one line).
8. `branch-voice.md`: derived `longest_run_s` + `voiced_s_bridged(gap ≤ G)` so MPT is recoverable;
   hysteresis/minimum-run as derived parameters.
9. N21 + benchmarks/voice.md + override: correct the alias vacuity boundary to two octaves;
   validate f0_max/f0_min < 4; per-task ranges; mark-sequence doubling detector.
10. `benchmarks/hear-yamnet.md` + default.yaml: retract the "wide empty interval" claim for
    hear.label_floor; record 0.501–0.821 winners and the zero-label cough files; mark 0.5
    unmeasured on this corpus.
11. `benchmarks/spans.md`: add quiet-breath contrasts (18.5–23.2 dB) beside the cough figure;
    per-task K or lower-K-with-merge-guard; report merge rate with recall.
12. `branch-airway.md` §2: restrict the YAMNet contest winner to a declared candidate set;
    minimum window-overlap rule; resolve the Snoring inconsistency.
13. `branch-speech.md` §3: state what a corroboration vote does; split coverage_threshold into
    score and fraction keys; minimum span duration for a coverage vote.
14. `redact.md`: verification on the same conditioned stream as planning; the path out of the
    permanently-unreleasable state.
15. `branch-speech.md` §8 + foundation: clipping on `recording` not `plain`; local-RMS
    discontinuity threshold; state wordless files get no disruption measurement.
16. `benchmarks/open.md`: split the SQUIM row — the speech-test floor is now largely measurable
    (report separations AND failure cases); quality floor remains unmeasured.
17. `verdict.md`: a calibration statement — flag 27/28 transports no information; state intended
    flag rate and which reasons may flag alone.
18. `branch-speech.md` §7 + taxonomy.md: hints condition positives (a PERSON finding on a declared
    non-lexical task is hallucination-suspect); compare count to declared targeted_speaker_count.

Refutations licensed on n=1 (one counterexample suffices): the HeAR empty-interval claim (S3),
"breath is unvoiced, the gate rejects it" (S5), the {1,2,≥3} codomain (S6), and the one-octave
alias comment (S10). Everything proposing a new threshold needs a second session first.
