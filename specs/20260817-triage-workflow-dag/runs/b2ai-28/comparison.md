# b2ai-28: three lanes over the same 28 recordings

Three independent screenings of one subject's one session (28 wav files, `sub-17cee767…_ses-DA790C5A…`).
Nothing here was recomputed; this is a join of three finished runs.

| lane | what it is | where |
|---|---|---|
| **SL·mac** | senselab triage workflow, local (macOS), commit `9e4ab8bc`, config_hash `e7893648350055d1` | `artifacts/triage_b2ai28_v2/summary.json` |
| **SL·eng** | the same workflow and the same config hash on Engaging | `/orcd/scratch/orcd/013/satra/triage_b2ai28/b2ai28-v2-20260823/` |
| **LLM** | agentharness comparator — deliberately weaker instruments: one recognizer, one audio classifier, no HeAR, no pyannote, no SQUIM | `/orcd/scratch/orcd/013/satra/agentharness-triage-comparator/b2ai-28-combined-20260823/outcomes.jsonl` |

Two files are read from a later Engaging run: **Diadochokinesis-PA** and **Diadochokinesis-Pataka**
errored on `REDACT` and `SPEECH` in the v2 pass (`ValueError: extent ends at 11.32s, past the
5.7585625s this audio decoded to`; `ValueError: no PII scan measurement in the store (N15)`) and were
re-run at commit `316bbff2` in `b2ai28-v3-ddk/`. Those two rows therefore carry a commit skew against
every other SL·eng row — noted again under caveats.

The LLM lane has **no release axis**. It reports triage, three kinds, and a boolean `pii_found`; it
does not model artifact release, redaction verification, or scan completeness. Every release figure
below is SL·mac against SL·eng only. No LLM counterpart is constructed, because there is none.

Figure: `~/Downloads/triage_threeway_28.png`.

## Per-file table

`**≠**` marks a row where the lanes carrying a comparable value disagree. `undec` = undecided.
`rel` = releasable, `held` = withheld.
[1m[33mwarning[39m[0m[1m:[0m [1m`--no-sync` has no effect when used outside of a project[0m

**Diadochokinesis**

| task | hint tags | triage SL·mac / SL·eng / LLM | release SL·mac / SL·eng | LLM pii | airway | speech | voice_no_words |
|---|---|---|---|---|---|---|---|
| Diadochokinesis-KA | voice | flag / flag / flag | held / held | no | **≠** pres / pres / abs | **≠** pres / pres / **undec** | **≠** **undec** / **undec** / pres |
| Diadochokinesis-PA | voice | **≠** flag / flag / pass | **≠** rel / held | no | abs / abs / abs | **≠** pres / pres / abs | **≠** **undec** / **undec** / pres |
| Diadochokinesis-Pataka | voice | **≠** flag / flag / pass | held / held | no | abs / abs / abs | pres / pres / pres | **≠** **undec** / **undec** / pres |
| Diadochokinesis-TA | voice | flag / flag / flag | **≠** held / rel | no | abs / abs / abs | **≠** pres / pres / **undec** | **≠** **undec** / **undec** / pres |
| Diadochokinesis-buttercup | speech · voice | flag / flag / flag | held / held | no | abs / abs / abs | pres / pres / pres | **undec** / **undec** / **undec** |

**Glides / phonation**

| task | hint tags | triage SL·mac / SL·eng / LLM | release SL·mac / SL·eng | LLM pii | airway | speech | voice_no_words |
|---|---|---|---|---|---|---|---|
| Glides-High-to-Low | phonation · voice | **≠** flag / flag / pass | **≠** held / rel | no | **≠** pres / pres / abs | **≠** pres / pres / abs | **≠** **undec** / **undec** / pres |
| Glides-Low-to-High | phonation · voice | **≠** flag / flag / pass | **≠** rel / held | no | abs / abs / abs | **≠** pres / pres / abs | **≠** **undec** / **undec** / pres |
| Loudness | phonation · voice | flag / flag / flag | held / held | no | **≠** pres / pres / abs | **≠** pres / pres / **undec** | **≠** **undec** / **undec** / pres |
| Maximum-phonation-time-1 | sustained-vowel · phonation · voice | **≠** flag / flag / pass | held / held | no | **≠** pres / pres / abs | **≠** pres / pres / abs | **≠** **undec** / **undec** / pres |
| Maximum-phonation-time-2 | sustained-vowel · phonation · voice | flag / flag / flag | **≠** held / rel | no | **≠** pres / pres / **undec** | abs / abs / abs | **≠** **undec** / **undec** / pres |
| Maximum-phonation-time-3 | sustained-vowel · phonation · voice | **≠** flag / flag / pass | held / held | no | **≠** pres / pres / abs | abs / abs / abs | **≠** **undec** / **undec** / pres |
| Prolonged-vowel | sustained-vowel · phonation · voice | flag / flag / flag | held / held | no | **≠** pres / pres / **undec** | pres / pres / pres | **≠** **undec** / **undec** / pres |

**Connected speech**

| task | hint tags | triage SL·mac / SL·eng / LLM | release SL·mac / SL·eng | LLM pii | airway | speech | voice_no_words |
|---|---|---|---|---|---|---|---|
| Free-speech-1 | speech | **≠** flag / flag / pass | rel / rel | no | **≠** pres / pres / abs | pres / pres / pres | **≠** **undec** / **undec** / pres |
| Free-speech-2 | speech | flag / flag / flag | rel / rel | no | **≠** pres / pres / abs | pres / pres / pres | **undec** / **undec** / **undec** |
| Free-speech-3 | speech | **≠** flag / flag / pass | rel / rel | no | **≠** pres / pres / abs | pres / pres / pres | **≠** **undec** / **undec** / abs |
| Picture-description | speech | **≠** flag / flag / pass | held / held | no | **≠** pres / pres / abs | pres / pres / pres | **≠** **undec** / **undec** / abs |
| Rainbow-Passage | read-speech · speech | **≠** flag / flag / pass | rel / rel | no | abs / abs / abs | pres / pres / pres | **≠** **undec** / **undec** / abs |
| Story-recall | speech | **≠** flag / flag / pass | held / held | no | **≠** pres / pres / abs | pres / pres / pres | **≠** **undec** / **undec** / abs |

**Respiration & cough**

| task | hint tags | triage SL·mac / SL·eng / LLM | release SL·mac / SL·eng | LLM pii | airway | speech | voice_no_words |
|---|---|---|---|---|---|---|---|
| R&C Breath-1 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** pres / pres / **undec** | **undec** / **undec** / **undec** |
| R&C Breath-2 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** pres / pres / **undec** | **undec** / **undec** / **undec** |
| R&C Cough-1 | cough · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** pres / pres / **undec** | **≠** **undec** / **undec** / abs |
| R&C Cough-2 | cough · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** pres / pres / **undec** | **≠** pres / pres / abs |
| R&C FiveBreaths-1 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** pres / pres / **undec** | **undec** / **undec** / **undec** |
| R&C FiveBreaths-2 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | **≠** abs / abs / **undec** | **≠** **undec** / **undec** / abs |
| R&C FiveBreaths-3 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | abs / abs / abs | **undec** / **undec** / **undec** |
| R&C FiveBreaths-4 | breathe · airway | **≠** pass / flag / pass | held / held | no | pres / pres / pres | abs / abs / abs | abs / abs / abs |
| R&C ThreeQuickBreaths-1 | breathe · airway | flag / flag / flag | held / held | no | pres / pres / pres | abs / abs / abs | **undec** / **undec** / **undec** |
| R&C ThreeQuickBreaths-2 | breathe · airway | **≠** flag / flag / pass | held / held | no | **≠** abs / abs / pres | abs / abs / abs | **≠** **undec** / **undec** / abs |

## Agreement

### Pairwise triage

| pair | agree | rate | disagreeing files |
|---|---|---|---|
| SL·mac vs SL·eng | 27 / 28 | **96.4%** | R&C FiveBreaths-4 |
| SL·mac vs LLM | 16 / 28 | **57.1%** | DDK-PA, DDK-Pataka, Glides-H2L, Glides-L2H, MPT-1, MPT-3, Free-speech-1, Free-speech-3, Picture-description, Rainbow-Passage, Story-recall, R&C ThreeQuickBreaths-2 |
| SL·eng vs LLM | 15 / 28 | **53.6%** | the same twelve, plus R&C FiveBreaths-4 |

Marginals explain most of the second and third rows: SL·mac is flag on 27 of 28 and SL·eng on 28 of
28, while the LLM splits 15 flag / 13 pass. A lane that flags everything cannot disagree by calling
something pass, so the whole senselab-vs-LLM triage gap is one-directional — every one of the twelve
is senselab flag against LLM pass.

### Per-kind

| kind | SL·mac vs SL·eng | SL·mac vs LLM | SL·eng vs LLM |
|---|---|---|---|
| airway | 28 / 28 (**100%**) | 15 / 28 (53.6%) | 15 / 28 (53.6%) |
| speech | 28 / 28 (**100%**) | 15 / 28 (53.6%) | 15 / 28 (53.6%) |
| voice_no_words | 28 / 28 (**100%**) | 8 / 28 (**28.6%**) | 8 / 28 (28.6%) |
| all three kinds, per file | 28 / 28 (**100%**) | 38 / 84 cells (45.2%) | 38 / 84 cells (45.2%) |

The two senselab lanes produce **identical kind vectors on all 28 files** — 84 of 84 cells. All three
lanes agree on the complete kind vector for only **4 of 28** files: DDK-buttercup, R&C FiveBreaths-3,
R&C FiveBreaths-4, R&C ThreeQuickBreaths-1.

### Release (SL·mac vs SL·eng only — the LLM has no such axis)

23 / 28 (**82.1%**). Disagreements: DDK-PA, DDK-TA, Glides-H2L, Glides-L2H, MPT-2. Note the shape:
the two hosts agree perfectly on what they measured and disagree most on what may be released.

| lane | releasable | withheld |
|---|---|---|
| SL·mac | 6 | 22 |
| SL·eng | 7 | 21 |

The releasable sets overlap on only four files (Free-speech-1/2/3, Rainbow-Passage).

### PII

| lane | files with findings | findings |
|---|---|---|
| SL·mac | 15 / 28 | 53 redactions; survivals on 9 files |
| LLM | **0 / 28** | `pii_found: false` on every file |

## Disagreement analysis

### 1. `voice_no_words`: senselab is blinded, and says so — 20 of 28 files

SL·mac and SL·eng report `undecided` on 26 of 28 (`present` once, `absent` once). The LLM decides on
21 of 28 (`present` 12, `absent` 9). This single axis is the largest disagreement class in the
comparison and the reason the per-kind rate drops to 28.6%.

Two established causes, both in the run's own override:

- `override.yaml` sets `phonation.f0_min_hz 75 / f0_max_hz 500` and states in its own comment that
  *any range wider than one octave makes the period-doubling alias check fire on clean phonation, so
  `ambiguous_runs_n` carries no information under this guess*. The prediction holds exactly: the
  `VOICE` node's why is `period_doubling_alias in range for run at …` on **26 of 28 files**.
- The override raises `taxonomy.min_families` for `airway` and `speech` only. `voice_no_words` keeps
  the packaged null, under which presence needs unanimity across three families with a bracketed
  non-lexical ASR token — so `present` is very nearly unreachable regardless of the audio.

Direction matters here. On the 12 files where the protocol declares phonation or voice (all five DDK,
both Glides, Loudness, all three MPT, Prolonged-vowel), the LLM's `present` matches the declaration
and senselab's `undecided` does not. This is senselab measuring nothing, not the LLM measuring wrong.

### 2. `airway`: the `min_families = 1` guess, then two real sensitivity differences — 13 files

Senselab says `present` where the LLM says `absent` on 10 files, and `present` where the LLM says
`undecided` on 2. Splitting by senselab's own `AIRWAY` why separates three mechanisms:

- **8 files** (DDK-KA, Glides-H2L, Loudness, MPT-1, MPT-3, Free-speech-1, Free-speech-3,
  Picture-description) carry `contradiction: airway predicted present, AIRWAY found no subject`. Under
  the override's `taxonomy.min_families.airway = 1`, one family's `present` vote is accepted as
  presence; the evidence node then found no subject and reported the contradiction. Senselab's
  `airway = present` on these files is a taxonomy prediction its own measurement node contradicts.
  Attribution: the override's admitted guess. The LLM's `absent` is the better reading of the audio.
- **2 files** (Free-speech-2, Story-recall) carry `yamnet contests Cough/Breathe with Speech`. Here
  senselab has weak breath or cough labels inside connected speech and marks them contested; the LLM's
  single classifier scored Breathing at 0.042 (Story-recall) and never reached its 0.5 floor. This is a
  genuine sensitivity difference on real evidence, not a guess artefact.
- **2 files** are LLM `undecided`, and its two undecideds arise from opposite conditions. On
  Prolonged-vowel it is pure threshold placement: Breathing peaked at **0.4985**, immediately under its
  own 0.5 floor (senselab's why on that file is `yamnet contests Breathe with Speech` — the same
  evidence, kept alive rather than rounded off). On MPT-2 it is the reverse plus a coverage hole:
  Breathing scored **0.86**, well *above* the floor, but the classifier was never run over the first
  0.595 s, so the lane refused to call absence on a region it had not measured (senselab's why on MPT-2
  is the `contradiction` above, so its `present` there is still the `min_families` artefact).

### 3. `speech` on respiration files: the conservatism flips — 9 files

The LLM says `undecided` where senselab says `present` on 8 files, five of them respiration
(Breath-1, Breath-2, Cough-1, Cough-2, FiveBreaths-1) plus DDK-KA, DDK-TA and Loudness.

The established cause is the LLM's recognizer transcribing breaths, and the LLM lane names it in its
own reasons: a 445-word / 6-distinct degenerate loop over near-floor energy on Breath-1, a
single-token ×223 loop on Cough-1, with many zero-length timestamps. It refuses to count degenerate
output as lexical content and leaves `speech` undecided.

Senselab, with two recognizers and a YAMNet cross-check, still records `speech = present` — because
`taxonomy.min_families.speech = 1` accepts one family's vote. On Cough-2 and FiveBreaths-1 the
`SPEECH` node's own why is `contradiction: speech predicted present, SPEECH found no subject`; on
Breath-2 and Cough-1 it is `yamnet disconfirms span … (Speech coverage 0.00)`. So on this axis the
richer lane is the *less* conservative one: senselab promotes hallucination-derived tokens to presence
and flags the contradiction, while the weaker lane declines to promote them at all. Whichever
convention is preferred, the direction of caution is not a property of instrument count — it is a
property of the folding rule.

### 4. `speech` present-vs-absent on non-lexical audio: one recognizer against two — 4 files

DDK-PA, Glides-H2L, Glides-L2H, MPT-1: senselab `present`, LLM `absent`. On both Glides files the
LLM's recognizer returned **zero** words while senselab's two recognizers produced words that the NER
ensemble then flagged (one `PERSON` redaction each). The audio carries no lexical content in either
lane's reading, so the outcome is decided by what a recognizer happens to emit on non-lexical input —
recognizer hallucination nondeterminism, sampled once in one lane and twice in the other.

### 5. PII: 53 findings against zero — and the two halves are different arguments

Senselab found and redacted 53 items across 15 files. The LLM reported `pii_found: false` on all 28.
Split by whether the audio carries lexical content, the disagreement is two unrelated things:

- **10 non-lexical files** (DDK-KA / Pataka / buttercup, both Glides, Loudness, all three MPT,
  Prolonged-vowel) — 12 redactions, with `PERSON` / `NAME` / `LOCATION` surviving verification on six.
  These files have no words to carry a name. The findings are recognizer hallucinations on non-lexical
  audio that the NER detectors then took at face value. Senselab withholds anyway, and that is the
  correct behaviour for a gate that cannot distinguish a hallucinated name from a real one — but it is
  not a PII catch, and it should not be counted as one. On these ten files the LLM's answer about the
  underlying audio is right.
- **5 lexical files** — Free-speech-1/2/3 (6 + 6 + 5 findings, **all redacted, re-scan verified clean
  under both recognizers**), Picture-description (11), Story-recall (13). Here the disagreement is
  substantive and the LLM never fires. Its reasons name what it saw and dismissed: kinship and
  common-noun scene content on Picture-description, a relational phrase inside recalled content on
  Story-recall. Part of the gap is therefore a definitional disagreement about whether relational and
  kinship terms are identifying, and part is an instrument gap — one recognizer, no NER ensemble, no
  redact-and-re-verify pass. **This report cannot separate the two halves without reading the findings'
  content, which it does not do.** What is decidable without reading content: the LLM lane has no
  mechanism that could produce a `yes` independently of its single transcript, and it produced no `yes`
  on any of 28 files. A detector that never fires earns no credit for a negative.

### 6. Release withheld for scan-incompleteness: not a conflict, an axis the LLM lacks — 7 files

R&C Cough-2, FiveBreaths-1/2/3/4, ThreeQuickBreaths-1/2 are withheld with
`the store's pii scan is incomplete (required detectors were not attempted: gliner, presidio, rules; no
detector ran); an unchecked recording is not a clean one (N15)`. No detector ran because no words
reached the store. Senselab treats unscanned as unchecked.

The LLM lane is not in conflict with this. Its own reasons repeatedly state that *a clean text scan is
a statement about the text, not a clearance of the audio* — it holds the same epistemic position and
simply has no release axis on which to act. Counting these as LLM misses would be inventing a
counterpart the lane does not have.

### 7. SL·mac vs SL·eng release: recognizer nondeterminism reaching the release gate — 5 files

Same commit, same config hash, identical kind vectors, different release on DDK-PA, DDK-TA,
Glides-H2L, Glides-L2H, MPT-2. The `REDACT` whys give the mechanism directly. On DDK-TA, Glides-H2L
and MPT-2, SL·mac reports `the redacted output re-scans clean, but verification re-ran only
Qwen/Qwen3-ASR-1.7B; nyralabs/CrisperWhisper2.0_turbo wrote no word at a resolved commit to re-run at`
→ REDACT flag → withheld, while SL·eng reports `every finding redacted; the redacted output re-scans
clean` → releasable. Glides-L2H runs the other way (a `PERSON` survived on Engaging, not locally).

So which of the two recognizers happens to emit a word on non-lexical audio determines whether the
verification pass can re-run both systems, and that determines release. The release axis inherits the
recognizers' nondeterminism through the verification-coverage requirement. DDK-PA is confounded by the
commit skew and is not a clean host comparison.

### 8. The one triage disagreement between hosts: R&C FiveBreaths-4

Identical kinds on both hosts (`airway present, speech absent, voice absent`). SL·mac's `SPEECH` why
is `no words from either recognizer; this branch has no subject` → fail, folded as a no-subject fail →
**triage pass**. SL·eng's is `yamnet disconfirms span 1.16-1.30s (Speech coverage 0.00); speaker count
0 != 1` → flag → **triage flag**. One host's recognizer emitted a 140 ms span the other's did not.
Same class as §7.

## Genuinely unexplained

Four items. None of these is force-attributed to a known cause, because none of the sources supports one.

1. **R&C ThreeQuickBreaths-2, airway.** Both senselab lanes say `absent`
   (`spans exist but none clears the label floor; a hint declares airway content not found`); the LLM
   says `present`. This is the only file where senselab misses a protocol-declared kind and the LLM
   catches it. Both lanes run a YAMNet-family classifier, so it is not an instrument-inventory gap —
   it is a label-floor or span-proposal difference that neither summary explains. Open.
2. **R&C Cough-2, `voice_no_words`.** Both senselab lanes say `present`, and this is the one file
   where senselab's `VOICE` node returned a clean `voiced runs measured; nothing contested` rather
   than the alias flag. The LLM says `absent`, reporting **0 frames above its 5 dB
   HNR floor in every one of the five loud spans**, max voiced_fraction 0.31, and F0 pinned at the
   500 Hz search edge. Given how negative the LLM's phonation numbers are, the two
   lanes cannot be measuring the same interval, and nothing in either summary says which interval
   senselab's voiced run occupies. Open.
3. **The LLM's own split across Free-speech-1/2/3**, nominally identical elicited-speech material:
   `voice_no_words` = present / undecided / absent, and triage = pass / flag / pass. No stated reason
   distinguishes the three. Unexplained within-lane variance.
4. **DDK-Pataka and DDK-PA, `speech`.** The LLM calls Pataka `present` and PA `absent` on two files of
   the same task family with the same hint and the same declared kinds. Its PA reason cites a
   degenerate transcript and its Pataka reason does not. Whether that is a real acoustic difference
   between the two recordings or recognizer sampling is not resolvable from these artifacts.

## What each lane is for

**The two senselab lanes are one instrument sampled twice, not two opinions.** They produce identical
kind vectors on 84 of 84 cells. Everything they disagree about — one triage call, five release calls —
traces to recognizer nondeterminism on non-lexical audio. Running both is a reproducibility probe, and
it earned its keep: it localised the nondeterminism to the verification-coverage requirement in
`REDACT` (§7) rather than to the measurement nodes.

**What senselab catches that the LLM cannot.** The release axis, entirely. 53 PII findings redacted,
of which the 17 on the three Free-speech files were redacted *and* re-verified clean under two
independent recognizers — an outcome the LLM lane has no pass capable of producing. Seven files
withheld because nothing scanned them. And on Free-speech-2 and Story-recall, weak breath and cough
evidence inside connected speech that the LLM's single classifier scored below its floor.

**What the LLM catches that senselab misses.** Three things, and they are not small:

- The whole `voice_no_words` axis. Senselab is blinded on it by an admitted guess and returns
  `undecided` 26 times; the LLM decides 21 times, and on all 12 protocol-declared phonation and DDK
  files its `present` matches the declaration.
- The respiration `speech` axis. On 8 files senselab promoted degenerate recognizer output to
  `present`; the LLM identified the degeneracy signature by name and declined.
- R&C ThreeQuickBreaths-2's airway — the one declared kind senselab missed on both hosts.

**R&C FiveBreaths-4 is the file all three lanes like.** Identical kind vectors (`airway present,
speech absent, voice absent`), and two of three lanes call triage pass. It is also the cleanest
demonstration that unanimity is not clearance: senselab still withholds it, because no PII detector
ever ran on it. Three lanes agreeing about content says nothing about whether the artifact may leave
the building.

## Caveats

- **n = 1 speaker, one session, 28 files.** Every rate in this document is a rate over one person's
  recordings. Nothing here estimates a population property, and no rate should be read as an accuracy.
- **The senselab run stands on seven admitted guesses.** `override.yaml` opens with *every value below
  is an ADMITTED GUESS, not a measurement* and labels each one: `speech.word_gap_ms 500`,
  `phonation.f0_min_hz 75` / `f0_max_hz 500`, `hnr_floor_db 5`, `rms_floor 0.001`,
  `redaction.padding_ms 200`, `taxonomy.min_families {airway: 1, speech: 1}`. Two of the three largest
  disagreement classes above (§1, §2, §3) are consequences of two of those guesses. This comparison
  therefore measures a configuration at least as much as it measures a workflow.
- **The LLM lane is one sample per file.** The only file with two complete comparator runs is R&C
  Cough-1: `voice_no_words` came out `undecided` in run `20260823-124623-21033611` and `absent` in
  `rerun6-20260823-163920`, with triage and the other two kinds unchanged. One repeat, one flip — the
  lane's per-file verdicts carry unmeasured variance, and the disagreement rates against it are upper
  bounds on the disagreement of the underlying method.
- **The LLM lane's 28 verdicts are not 28 equal-effort attempts.** Six first attempts stopped
  incomplete (five `max_images`, one `client_error`) with no verdict and were re-run: Breath-1,
  Breath-2, Cough-1, Cough-2, FiveBreaths-1, FiveBreaths-3. All six are respiration files, which is
  also where §3's disagreement class lives.
- **Commit skew on two files.** DDK-PA and DDK-Pataka's SL·eng rows come from `316bbff2`, not
  `9e4ab8bc`, because the v2 pass errored on `REDACT` and `SPEECH` for both. Their config hash is
  unchanged (`e7893648350055d1`), so the difference is code, not configuration. DDK-PA is one of the
  five release disagreements in §7 and should be discounted there.
- **No transcript text or finding content was read for this comparison.** PII appears here only as
  categories and counts, which is why §5 leaves the definitional half of its disagreement unresolved.
