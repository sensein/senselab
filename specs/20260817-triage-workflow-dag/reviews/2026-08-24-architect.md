# Architect review — code-first, then spec (2026-08-24, @ bec94fe9)

Protocol: phase 1 read only the implementation and the 28-file campaign evidence
(`artifacts/triage_b2ai28_v2/`); phase 2 cross-checked the specs. Reference model under test
(project owner's): each branch assesses its own type's presence; VOICE assesses only when airway
and speech both say absent; PII detection and redaction belong only to the speech branch.

## Phase 1 — goals as implemented

- **ADMIT** (`nodes/admit.py:35-112`): is this file measurable at all — decode, all-zero,
  constant-per-channel. The graph's only gate: on fail, `run.py:272-274` skips everything but
  VERDICT.
- **PREPROCESS** (`nodes/preprocess.py:108-…`, verdict `:646-651`): what shared derivatives exist.
  Unconditional, always pass; 13 independent blocks, failures land in `absent`. Sole writer of
  per-recognizer `word` entities and transcripts — PII-bearing text enters the store here, before
  any branch decides anything.
- **TAXONOMY** (`nodes/taxonomy.py:71-302`): which kinds does a detector committee predict.
  Screens two kinds (`SCREENED_KINDS`, `:36`); family A = YAMNet + AST, B = CrisperWhisper lexical,
  C = HeAR (airway only). Writes a third `kind` for `voice_no_words` as `not_screened`
  (`:279-287`) — a fact about a kind it does not assess. Gates nothing; nothing reads it but
  VERDICT.
- **AIRWAY** (`nodes/airway.py:85-370`): do the K=18 dB spans carry cough/breath. HeAR labels,
  YAMNet confirms/contests, lexical contamination. Decides `kind="airway"`.
- **SPEECH** (`nodes/speech.py:266-840`): are there words, whose, how good, does the text carry
  PII. Runs no ASR — fuses PREPROCESS's word streams; spans from word timings; pyannote over
  [first, last word]; withdraws segments overlapping AIRWAY labels; separation at count==2; PII
  scan per span per recognizer. `fail` only on no-words (`:317-322`); the no-words path still
  writes an empty `pii_scan` (`:324-337`).
- **VOICE** (`nodes/voice.py:173-433`): is there voiced content in the energetic residual nobody
  claimed. Residual = envelope>floor minus AIRWAY labels minus SPEECH spans (`:242-256`). Runs
  unconditionally (`run.py:196`).
- **REDACT** (`nodes/redact.py:337-499`): can a de-identified pair be released. Reads SPEECH's
  findings + `pii_scan`; silences the **source** `recording` stream; re-transcribes with the
  declared recognizers and **re-scans with `scan_for_pii` itself** (`:196-205`). Runs
  unconditionally.
- **VERDICT** (`nodes/verdict.py:192-264`): what does the store say overall; two axes.
  `triage` (`vocabulary.py:123-131`): fail iff ADMIT failed or every kind ABSENT; flag on any FLAG
  reason; else pass. A REDACT fail is invisible on this axis (evidence: FiveBreaths-4 is
  triage=pass with REDACT:fail). `release` (`verdict.py:141-155`): releasable iff REDACT passed.

Control flow: `run.py:31` is a fixed linear order, not a DAG; dependencies exist only as store
reads. Real edges: SPEECH ← AIRWAY labels; VOICE ← AIRWAY + SPEECH; nothing reads TAXONOMY except
VERDICT — `TAXONOMY → AIRWAY` in the declared order is pipeline order mistaken for dependency.

## Reference-model assessment

**Bullet 1 — each branch assesses its own type: PARTIALLY IMPLEMENTED.**
Each branch is the only node writing a verdict with its kind, but: (a) TAXONOMY independently
predicts airway/speech presence and the prediction survives into the file `kinds` map, able to
outrank the branch — `vocabulary.py:105-110` (PRESENT + branch FAIL → kind stays PRESENT, flag) vs
`:111-115` (ABSENT + branch PASS → rewritten PRESENT): the tie-break depends on the disagreement's
direction. (b) **A branch FLAG never resolves its own kind** — `vocabulary.py:116-120` handles
UNDECIDED only for PASS/FAIL; FLAG is the modal branch outcome (AIRWAY 10/28, SPEECH 18/28).
Measured: MPT-2 and MPT-3 end `speech=absent` while carrying words, a span, a PERSON finding and a
planned redaction; DDK-TA ends `airway=absent` with two uninvalidated Cough label assertions.
(c) Branches read each other (SPEECH ← AIRWAY labels; VOICE ← both), and TAXONOMY's speech family B
is the same evidence class SPEECH uses, on one recognizer instead of the fused pair.

**Bullet 2 — VOICE conditional on both-absent: NOT IMPLEMENTED.**
Unconditional (`run.py:196`). Campaign: VOICE measured gated runs on 27/28, including all 21
speech-present files (Picture-description: 76 runs from 140-word speech). On Cough-2 VOICE's pass
promoted `voice_no_words` to PRESENT on a file the fold also calls speech-present. Counterfactual:
removing VOICE's verdict changes triage on exactly 1 of 28 files — its cost here is compute plus
one spurious flag, but 21 files of Praat analysis is work the reference model would never do.

**Bullet 3 — PII/redact speech-only: PARTIALLY IMPLEMENTED.**
Planning-side detection is SPEECH-only. But REDACT runs a second, independent PII detector outside
the branch — the one that gates release — and the two scanners read different text over the same
audio (F-E: DDK-KA planner 0 findings, verifier 2 categories → permanently unreleasable). REDACT
runs on wordless recordings: 7 respiration files got withheld releases via the incomplete-scan row
(F-F: structurally unreachable pass). PREPROCESS writes transcripts unconditionally, so PII-bearing
text exists in every store regardless of any branch's finding.

## Architecture findings (ranked)

- **A-1** Presence decided twice; resolution asymmetric with a hole at FLAG (`vocabulary.py:88-121`).
  11/28 contradiction flags; 4/28 kinds stuck at the screen's value under a branch FLAG. The row
  justifying the advisory pattern (ABSENT + branch PASS) fired 0/28.
- **A-2** The advisory taxonomy has no usable configuration on this corpus: AST abstained 28/28
  (null floor) so family A = YAMNet alone; min_families=1 turns "presence needs agreement" into a
  one-detector OR; the packaged null (unanimity) reads undecided on nearly everything; the nextflow
  prototype's admissible set excludes 1 for airway while the code accepts it (`taxonomy.py:230-234`).
- **A-3** Unconditional execution is the dominant cost: TAXONOMY re-runs HeAR whole-file while
  AIRWAY runs it per span; AST never voted; on the 7 wordless files every expensive stage ran for a
  predetermined verdict set.
- **A-4** The runner is a fixed sequence; the declared DAG has a false edge (TAXONOMY→AIRWAY) and
  hides the real ones (AIRWAY→SPEECH→VOICE). `ran` computed twice; the store-derived ERRORED
  signature is dead code under the runner.
- **A-5** Single-writer discipline leaks for spans (four writers, disambiguated by activity-node
  string), words (two writers, three consumer filters), and the `not_screened` voice kind — which
  is what lets VOICE's pass promote the kind with no screen behind it.
- **A-6** Release failure invisible on the triage axis (FiveBreaths-4: triage=pass, withheld).
- **A-7** Spec self-contradiction on wordless recordings: `verdict.md:89-90` (no scan → not_assessed)
  vs `redact.md:49-55` (empty scan → fail/withheld); code follows the latter; NOT_ASSESSED is
  unreachable in practice.
- **A-8** The packaged default config cannot complete a run (`speech.word_gap_ms`,
  `redaction.padding_ms` null through `require`) — every real run is an override run.

## Phase 2 — classification (code-vs-spec)

| # | divergence | class |
|---|---|---|
| D-1 | VOICE unconditional subtraction | spec and code agree; reference model is the outlier — decide |
| D-2 | TAXONOMY co-decides presence | spec'd; qualify the model or change the fold |
| D-3 | Branch FLAG never resolves its kind | **spec gap** |
| D-4 | REDACT on wordless files | **internal spec contradiction**; code follows redact.md |
| D-5 | REDACT re-scans itself | spec'd; distinguish detection-for-planning vs verification |
| D-6 | `voice_no_words` written by TAXONOMY | spec'd; blocks conditional VOICE — change |
| D-7 | min_families=1 accepted | **spec gap** (admissible set unstated) |
| D-8 | AST abstains forever, family A = 1 member | **spec gap** |
| D-9 | Quality on `plain`, every span, never separated stream | **code deviates from branch-speech.md §8** |
| D-10 | `alignment` never read by SPEECH | **code deviates from branch-speech.md §1** |
| D-11 | No SPEECH figure | **code deviates** |
| D-12 | Separation only at count==2 | code is the sane reading; fix spec wording |
| D-13 | TAXONOMY→AIRWAY declared edge | **spec error** |
| D-14 | REDACT fail invisible in triage | by design; document explicitly |

## Spec-change list

- **S-1** `branch-voice.md` §residual + `store.md`: either adopt conditionality (guarded runner
  call; drop the subtraction; stop mapping not_screened→UNDECIDED or every skip flags) or restate
  the model as "VOICE claims only unclaimed energy" and document pass-on-speech-present.
- **S-2** `taxonomy.md` + `verdict.md`: remove TAXONOMY's voice_no_words kind or make
  `not_screened` a first-class state meaning "no screen; branch is sole authority".
- **S-3** `verdict.md` resolution table: add the FLAG row and the PRESENT+FAIL precedence rule
  explicitly; `vocabulary.py:105-120` gains both.
- **S-4** `taxonomy.md`: state the admissible min_families set normatively; enforce in the loader;
  record `single_source` when a family member abstains.
- **S-5** `redact.md` + `verdict.md`: resolve the wordless contradiction. If the reference model
  wins: REDACT gated on SPEECH's outcome; wordless → no REDACT verdict → release not_assessed;
  delete SPEECH's empty pii_scan write. 7 campaign files move withheld → not_assessed.
- **S-6** `redact.md`: name the planner/verifier one-way street (F-E) and specify the path out
  (feed verifier findings back for a second planning pass, or an explicit `unremediable` reason).
- **S-7** `store.md`: correct the declared graph; if a real DAG runner is wanted, nodes declare
  reads and the order is derived — TAXONOMY becomes parallelisable.
- **S-8** `branch-speech.md` §1/§8/Product: bring spec down to what runs (quality on plain,
  edges from fused words, no figure) or mark unimplemented; fix "count is not 1" → "exactly 2".
- **S-9** `verdict.md` two-axes: one sentence that triage=pass never implies the release check
  passed.
