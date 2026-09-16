# What TAXONOMY has to read to route a recording — 62,550 recordings, run of 2026-09-10

The routing question, measured: which candidate detector, at which threshold, decides that a
recording carries speech, airway or voice content well enough to send it to that branch. **No
branch was run and none should be.** Everything below is read from PREPROCESS's and TAXONOMY's
stored evidence.

## The corpus and how it was resolved

A completed triage run of the b2ai adult tree at
`/orcd/scratch/bcs/002/satra/triage_full_20260908/out/` (config hash `05a76991f1a7fcff`). Each
recording was resolved through its own `<stem>.summary.json`'s `run_root`, remapped onto the local
directory by basename; run directories were never globbed, because preemption left partial
duplicates a glob would pick up and no summary names.

| | |
| --- | --- |
| `*.summary.json` resolved | 62,550 |
| stores read | 62,547 |
| stores absent | 3 |

The three absent stores are PREPROCESS hard failures whose summary carries `ok: false` and no
`run_root`: two `asr_crisperwhisper` `"No position encodings are defined for positions >= 448"`
overflows and one torchcodec `decodeAVFrame` failure. They are excluded, not counted as negatives.

**Two counts in the analysis brief did not reproduce.** The brief said 47 families over 805
sanitized task ids; collapsing only trailing numeric segments gives **48 families over 796 ids**.
Its two spot figures both check out exactly — `harvard-sentences-list` 13,705 and `high-to-low` 43 —
and the family counts sum to 62,550. The extra family is `high-to-low` (43), which is distinct from
`glides-high-to-low` (1,554); `picture-description`, `-option1` and `-option2` are likewise three
families, not one.

## The reference standards, and what is wrong with each

**There are no branch labels. Nobody annotated which recordings contain speech.** Two standards are
used and they are not equally strong.

| standard | positive when | n positive | weakness |
| --- | --- | --- | --- |
| `agreed_asr` | ≥1 live consensus `word` with `outcome: agreement` | 36,569 | measures *lexical agreement between two recognizers*, not "someone spoke"; see the diadochokinesis result below |
| `declared_speech` | family asks the participant to produce speech | 41,224 | a declaration, not an observation |
| `declared_lexical_speech` | as above, excluding the ten diadochokinesis families | 33,235 | as above |
| `declared_airway` | family asks for a breath, cough or throat manoeuvre | 13,017 | as above |
| `declared_voice` | family asks for sustained or glided phonation | 8,306 | as above |

`agreed_asr` is the operational definition the owner's framing implies and is the only standard
that is an observation of the recording. The `declared_*` standards are priors. A detector
disagreeing with one is not thereby wrong, which is why every disagreement is enumerated below
rather than folded into an error rate.

## TAXONOMY as it actually ran — the baseline

| kind | `present` | `absent` | `uncertain` | no `kind` entity |
| --- | --- | --- | --- | --- |
| speech | 0 | 0 | 62,518 | 29 |
| airway | 0 | 0 | 62,518 | 29 |
| voice | 0 | 0 | 62,518 | 29 |

ROUTING runs a branch unless the kind is `absent` (`routing.py:185-187`), so every branch would run
on every recording:

| reference | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| `agreed_asr` | 36,569 | 25,978 | 0 | 0 | 1.000 | **0.000** |
| `declared_speech` | 41,224 | 21,323 | 0 | 0 | 1.000 | **0.000** |
| `declared_airway` | 13,017 | 49,530 | 0 | 0 | 1.000 | **0.000** |
| `declared_voice` | 8,306 | 54,241 | 0 | 0 | 1.000 | **0.000** |

**TAXONOMY routes nothing.** Every candidate below is an improvement on specificity 0.

### Why, mechanically — seven nulls, not one

- `windows.{yamnet,ast,hear}.default_threshold` are null (`data/config/default.yaml:92,95,101`).
  `_windows()` opens with `config.require(f"windows.{classifier}.default_threshold")`
  (`preprocess.py:892`), so the block never completes and no `<classifier>_windows` measurement
  reaches the store. TAXONOMY's speech acoustic line reads exactly that measurement
  (`taxonomy.py:98`) and is therefore always `unavailable`. The same nulls leave every `span_hear`
  and `span_yamnet` measurement carrying `labelled: false`, which is what `_span_label_evidence`
  tests at `taxonomy.py:200` before declaring both airway lines unavailable.
- `taxonomy.presence_floor.*` are all null (`default.yaml:165-171`). `_line_state` returns
  `unavailable` whenever the floor is None (`taxonomy.py:269`), so even an available line could not
  reach `present`.
- `taxonomy.voice_min_duration_s` is null and unread; `_retired_voice_line` (`taxonomy.py:311`)
  returns `unavailable` unconditionally.

## Speech

### The owner's rule and its alternatives, whole corpus

`speech.words_agreement ≥ 1` scored against `agreed_asr` is the definition of the standard and is
reported only for completeness (J = 1.000). Against the declaration:

`speech.words_agreement ≥ n` vs `declared_speech` (n = 41,224 of 62,547):

| n | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 33,985 | 2,584 | 18,739 | 7,239 | 0.8244 | 0.8788 |
| 2 | 33,282 | 1,614 | 19,709 | 7,942 | 0.8073 | 0.9243 |
| 4 | 31,131 | 290 | 21,033 | 10,093 | 0.7552 | 0.9864 |
| 10 | 13,777 | 65 | 21,258 | 27,447 | 0.3342 | 0.9970 |

`speech.words_lexical ≥ n` — every non-bracketed consensus word, **any** outcome — vs
`declared_speech`:

| n | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 39,789 | 5,579 | 15,744 | 1,435 | 0.9652 | 0.7384 |
| 2 | 39,360 | 3,093 | 18,230 | 1,864 | 0.9548 | 0.8549 |
| 4 | 37,368 | 1,431 | 19,892 | 3,856 | 0.9065 | 0.9329 |
| 5 | 36,953 | 707 | 20,616 | 4,271 | **0.8964** | **0.9668** |

Best Youden over every speech detector and either declaration standard is `words_lexical ≥ 5`
(J = 0.8632). Requiring *agreement* costs 14 points of sensitivity at n = 1 and is the single
largest effect measured on the speech side.

### Classifiers cannot substitute for ASR, and only AST comes close

Against `agreed_asr`, best threshold per detector:

| detector | best thr | sens | spec | J | at 0.2 (`consolidation_floor`) sens / spec |
| --- | --- | --- | --- | --- | --- |
| `speech.ast_peak.plain` | 0.5 | 0.8989 | 0.9152 | **0.8141** | 0.9563 / 0.7900 |
| `speech.ast_peak.enhanced` | 0.5 | 0.9068 | 0.9117 | 0.8186 | — |
| `speech.yamnet_peak.plain` | 0.99 | 0.9381 | 0.7694 | 0.7075 | 0.9965 / **0.2168** |
| `speech.hear_peak.plain` | 0.2 | 0.8840 | 0.6696 | 0.5536 | 0.8840 / 0.6696 |
| `speech.ast_peak.residual` | 0.2 | 0.7615 | 0.7878 | 0.5492 | — |
| `speech.yamnet_peak.residual` | 0.95 | 0.0981 | 0.9449 | **0.0429** | — |
| `speech.hear_peak.residual` | 0.2 | 0.1881 | 0.8686 | **0.0567** | — |
| `speech.ast_peak.consensus` | — | 0.0000 | 1.0000 | **0.0000** | — |

**YAMNet `Speech` at 0.2 is useless for routing**: specificity 0.217. It fires on 78% of the
recordings with no agreed word, and its per-family firing fraction never drops below 0.560 in any
of the 48 families — its minimum is `respiration-and-cough-breath` 0.560, its maximum
`animal-fluency` 1.000. Only at 0.99 does it become a discriminator, and AST at 0.5 has the higher
Youden there.

**Every `*.consensus` AST detector has sensitivity 0.0000 at every threshold.** `consensus_taxonomy`
is built from `PER_SPAN_CLASSIFIERS = {yamnet: span_yamnet, hear: span_hear}` (`taxonomy.py:380`);
there is no `span_ast` anywhere in `preprocess.py`, so AST contributes nothing to the fused
taxonomy. This is a property of the fusion, not of AST.

**The residual stream is not a speech detector.** Both `speech.yamnet_peak.residual` (J = 0.043)
and `speech.hear_peak.residual` (J = 0.057) are near-worthless, which is the intended behaviour:
the residual is what enhancement removed.

### Speech disagreements, enumerated

**2,584 recordings carry an agreed word in a family that does not ask for words.**

| family | with an agreed word | n | fraction |
| --- | --- | --- | --- |
| `prolonged-vowel` | 1,258 | 1,604 | 0.784 |
| `glides-low-to-high` | 269 | 1,596 | 0.169 |
| `respiration-and-cough-cough` | 239 | 1,788 | 0.134 |
| `maximum-phonation-time` | 237 | 2,696 | 0.088 |
| `respiration-and-cough-v2-hardcough` | 127 | 698 | 0.182 |
| `glides-high-to-low` | 126 | 1,554 | 0.081 |
| `respiration-and-cough-breath` | 73 | 1,788 | 0.041 |
| `maximum-phonation-time-v2` | 72 | 813 | 0.089 |
| `respiration-and-cough-fivebreaths` | 64 | 3,576 | 0.018 |
| `voluntary-cough` | 44 | 327 | 0.135 |
| `respiration-and-cough-threequickbreaths` | 30 | 1,718 | 0.017 |
| `respiration-and-cough-v2-breath` | 13 | 699 | 0.019 |
| `respiration-and-cough-v2-threebreathsmouth` | 9 | 699 | 0.013 |
| `respiration-and-cough-v2-threebreathsnose` | 9 | 699 | 0.013 |
| `breath-sounds` | 6 | 326 | 0.018 |
| `respiration-and-cough-v2-threebreaths` | 6 | 699 | 0.009 |
| `high-to-low` | 2 | 43 | 0.047 |

Reading their transcripts splits them into three causes, and only one is a detection error.

**(a) The protocol asks for speech before the non-speech manoeuvre.** In `prolonged-vowel`, 938 of
the 1,258 transcripts begin "One two three" — 569 `'One two three [UH]'`, 225 `'One two three. Ah'`,
97 `'One two three.'`, 38 `'One two three [UH] [UH]'`, 29 `'One two three [UH] [UH] [UH]'`. The
participant counts aloud and then sustains. This is real speech in a declared-voice file and the
speech branch should have it.

**(b) Genuine off-task speech — the goal-2 case.** Every `respiration-and-cough-breath` and
`-fivebreaths` disagreement has a non-bracketed agreed word (73/73 and 62/64) and the transcripts
are conversation, much of it apparently the examiner:
`"I'll have you do that one more time. [breath]"`, `"Okay. [breath]"`, `"So just breathe."`,
`"Now do I stop?"`, `"Good job."`, `"That's good."`. In `breath-sounds`, one recording reads
`"So far have been general surgery and then cardiac surgery. But we'll see. I have a lot of time
for this side."` — a conversation held during a breath elicitation. These are the recordings the
pipeline exists to flag, and no rule should suppress them.

**(c) The manoeuvre transcribed as a word.** `respiration-and-cough-cough` has 239 recordings with
an agreed word but only **37** with a non-bracketed one, and those 37 read `'cough cough cough'` —
the cough rendered as the ordinary English word. `maximum-phonation-time` gives 55 `'[UH]'` and 50
`'Ah.'`; `glides-low-to-high` gives 145 `'E.'`, 26 `'E-'`, 18 `'E'` — the glide vowel transcribed
as a letter name. This is exactly the case `taxonomy`'s proposed `words.onomatopoeic_tokens`
vocabulary is for, and it is measurable: it is the whole of the cough-family disagreement.

**449 recordings across 12 families have an agreed word where every agreed word is bracketed** —
`'[cough]'` (76), `'[UH]'` (73), `'[cough] Cough Cough Cough Cough'` (26) and similar, concentrated
in `respiration-and-cough-cough` (202), `-v2-hardcough` (120), `maximum-phonation-time` (67) and
`voluntary-cough` (34). The owner's rule as stated ("at least one consensus word whose outcome is
`agreement`") fires on all 449. Requiring the word to be non-bracketed as well removes them at a
cost of one true positive (`words_agreement_lexical` at n = 1: sens 0.8244, spec 0.8998 against
`declared_speech`, against 0.8244 / 0.8788).

**7,239 recordings in a speech-declared family carry no agreed word**, and 5,677 of them — 78% —
are diadochokinesis:

| family | no agreed word | n | fraction |
| --- | --- | --- | --- |
| `diadochokinesis-v2-kuh` | 668 | 702 | 0.952 |
| `diadochokinesis-ka` | 834 | 896 | 0.931 |
| `diadochokinesis-v2-puh` | 642 | 702 | 0.915 |
| `diadochokinesis-v2-puhtuhkuh` | 608 | 701 | 0.867 |
| `diadochokinesis-pa` | 688 | 896 | 0.768 |
| `diadochokinesis-v2-tuh` | 520 | 702 | 0.741 |
| `diadochokinesis-pataka` | 663 | 896 | 0.740 |
| `diadochokinesis-ta` | 617 | 896 | 0.689 |
| `diadochokinesis-buttercup` | 247 | 896 | 0.276 |
| `diadochokinesis-v2-buttercup` | 190 | 702 | 0.271 |
| `harvard-sentences-list` | 1,004 | 13,705 | 0.073 |

**Their transcripts are not empty.** `diadochokinesis-ka` reads
`'Ca-ca-ca-ca-ca-ca-...'`, `'Ca-caxacaca-caxacaca-...'`, `'Ca- ca- ca- ca- ...'` — the two
recognizers transcribe the syllable train and disagree on how, so every word carries
`outcome: variant` and none carries `agreement`. The buttercup variants, whose target is a real
word, disagree at only 0.27 against 0.93 for `/ka/` — the same family type with and without a
lexical target, which is the cleanest available demonstration that **`agreed_asr` measures lexical
agreement, not speech presence.** This is the reference standard's principal weakness and it is not
repairable by choosing a different threshold.

The remaining 1,562 are largely genuinely wordless: of the 1,004 silent `harvard-sentences-list`
recordings, **916 have a transcript that is empty or nothing but bracketed markers** — 456 `'[UM]'`,
210 `'[breath]'`, 117 `'[laughter]'`, 94 empty, 36 `'[UH]'` — and only 88 carry any unbracketed
token at all, those being recognizer disagreements over a real reading
(`'Either light knows the poor tide.'`, `'Do you have a room child flap?'`) or short interjections
(`'Yeah.'` ×7). A 6.7% produced-nothing rate in a reading task is a finding in itself.

## Airway

`declared_airway` positive on 13,017 of 62,547.

| detector | best thr | tp | fp | tn | fn | sens | spec | J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `airway.yamnet_peak.plain+hear≥0.2` | 0.2 | — | — | — | — | 0.8713 | 0.8506 | **0.7219** |
| `airway.residual_energy_fraction+hear≥0.2` | 0.05 | — | — | — | — | 0.8388 | 0.8828 | 0.7216 |
| `airway.residual_energy_fraction+no_agreed_word` | 1e-4 | — | — | — | — | 0.9268 | 0.8275 | 0.7543 |
| `airway.yamnet_peak.plain` | 0.3 | 10,474 | 6,501 | 43,008 | 2,535 | 0.8051 | 0.8687 | 0.6738 |
| `airway.residual_energy_fraction` | 0.1 | 10,206 | 6,242 | 43,265 | 2,803 | 0.7845 | 0.8739 | 0.6585 |
| `airway.hear_peak.residual` | 0.4 | — | — | — | — | 0.8846 | 0.7692 | 0.6538 |
| `airway.ast_peak.plain` | 0.15 | — | — | — | — | 0.8077 | 0.8296 | 0.6373 |
| `airway.hear_peak.plain` | 0.6 | 10,884 | 15,275 | 32,964 | 1,392 | 0.8866 | 0.6833 | 0.5700 |
| `airway.hear_peak.span` | 0.7 | — | — | — | — | 0.7360 | 0.6302 | **0.3661** |
| `airway.yamnet_peak.enhanced` | 0.3 | — | — | — | — | 0.4957 | 0.8612 | 0.3569 |
| `airway.ast_peak.consensus` | — | 0 | 0 | 49,473 | 12,990 | 0.0000 | 1.0000 | **0.0000** |
| `airway.residual_enhanced_energy_fraction` | — | — | — | — | — | 0.9999 | 0.0000 | **−0.0001** |

Full sweeps for `airway.residual_energy_fraction` vs `declared_airway`:

| thr | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| 1e-5 | 12,936 | 39,516 | 9,991 | 73 | 0.9944 | 0.2018 |
| 1e-3 | 11,862 | 20,784 | 28,723 | 1,147 | 0.9118 | 0.5802 |
| 0.01 | 11,179 | 13,254 | 36,253 | 1,830 | 0.8593 | 0.7323 |
| 0.05 | 10,605 | 8,281 | 41,226 | 2,404 | 0.8152 | 0.8327 |
| 0.1 | 10,206 | 6,242 | 43,265 | 2,803 | 0.7845 | 0.8739 |
| 0.2 | 9,617 | 4,253 | 45,254 | 3,392 | 0.7393 | 0.9141 |
| 0.5 | 7,577 | 1,875 | 47,632 | 5,432 | 0.5824 | 0.9621 |
| 0.9 | 2,201 | 389 | 49,118 | 10,808 | 0.1692 | 0.9921 |

### Airway is two phenomena and one detector does not serve both

Firing fraction per family, `airway.residual_energy_fraction ≥ 0.1` against
`airway.yamnet_peak.plain ≥ 0.3`:

| family | n | residual ≥ 0.1 | YAMNet airway ≥ 0.3 |
| --- | --- | --- | --- |
| `respiration-and-cough-v2-breath` | 697 | **0.977** | 0.674 |
| `respiration-and-cough-v2-threebreathsnose` | 699 | **0.961** | 0.858 |
| `breath-sounds` | 326 | **0.948** | 0.902 |
| `respiration-and-cough-breath` | 1,788 | **0.904** | 0.554 |
| `respiration-and-cough-fivebreaths` | 3,572 | **0.887** | 0.821 |
| `respiration-and-cough-threequickbreaths` | 1,717 | 0.630 | 0.874 |
| `respiration-and-cough-cough` | 1,787 | 0.569 | **0.825** |
| `voluntary-cough` | 327 | 0.621 | **0.963** |
| `respiration-and-cough-v2-hardcough` | 698 | **0.352** | **0.855** |
| `word-color-stroop` | 472 | 0.356 | 0.417 |
| `maximum-phonation-time` | 2,695 | 0.260 | 0.473 |

**The residual catches breathing and misses coughing** (0.352 on `hardcough`); **YAMNet catches
coughing and is weakest on quiet breathing** (0.554 on `respiration-and-cough-breath`). Neither
threshold in either sweep repairs the other's blind spot.

### Per-label firing at 0.2, whole file, `plain` stream

| family | HeAR `Cough` | YAMNet `Cough` | HeAR `Breathe` | YAMNet `Breathing` | HeAR `Snore` |
| --- | --- | --- | --- | --- | --- |
| `voluntary-cough` | 0.978 | 0.872 | 0.469 | 0.453 | 0.469 |
| `respiration-and-cough-cough` | 0.940 | 0.749 | 0.145 | 0.351 | 0.158 |
| `respiration-and-cough-v2-hardcough` | 0.928 | 0.719 | 0.130 | 0.211 | 0.208 |
| `respiration-and-cough-v2-threebreathsnose` | 0.044 | 0.027 | 0.961 | 0.870 | 0.884 |
| `respiration-and-cough-v2-threebreaths` | 0.206 | 0.247 | 0.938 | 0.928 | 0.826 |
| `breath-sounds` | 0.095 | 0.187 | 0.951 | 0.914 | 0.914 |
| `respiration-and-cough-breath` | 0.109 | 0.031 | 0.808 | 0.597 | 0.910 |
| `harvard-sentences-list` | 0.162 | **0.006** | 0.048 | **0.019** | 0.139 |
| `free-speech` | 0.435 | **0.019** | 0.291 | 0.158 | 0.531 |
| `word-color-stroop` | 0.828 | **0.061** | 0.841 | 0.462 | 0.983 |
| `story-recall-v2` | 0.746 | 0.067 | 0.556 | 0.277 | 0.804 |
| `maximum-phonation-time` | 0.040 | 0.046 | 0.498 | **0.504** | 0.873 |
| `prolonged-vowel` | 0.128 | 0.029 | 0.346 | 0.318 | 0.791 |

- **YAMNet `Cough` is the most specific single label measured**: 0.006 on `harvard-sentences-list`
  and 0.019 on `free-speech` against 0.75–0.87 on the cough tasks.
- **HeAR `Cough` is not usable alone**: 0.83 on `word-color-stroop`, 0.75 on `story-recall-v2`,
  0.71 on `caterpillar-passage`. It is sensitive and it fires on long speech.
- **HeAR `Snore` discriminates nothing.** It fires on 0.98 of `word-color-stroop`, 0.92 of
  `animal-fluency`, 0.89 of `random-item-generation` and 0.80 of `story-recall-v2` — all speech
  tasks — as well as on 0.91 of `respiration-and-cough-fivebreaths` and 0.87 of
  `maximum-phonation-time`. Its lowest values are on `diadochokinesis-buttercup` (0.075) and
  `respiration-and-cough-cough` (0.158). The earlier account that its `Snore` "peaks on sustained
  voicing, not respiration" is **not what this corpus shows**: it peaks on both, and on connected
  speech as well. It must not raise or corroborate any kind.
- **YAMNet `Breathing` confuses sustained phonation**: 0.504 on `maximum-phonation-time`, against
  0.019 on `harvard-sentences-list`.
- **Five of HeAR's eight labels can never corroborate an AudioSet one.** `_write_consensus_taxonomy`
  merges rows on the exact label string (`taxonomy.py:461`). Checked against a run's own 521 YAMNet
  label keys, only `Cough`, `Sneeze` and `Speech` are spelled the same; `Baby Cough`, `Breathe`,
  `Laugh`, `Snore` and `Throat Clear` have only near-misses (`Breathing`, `Laughter`, `Snoring`,
  `Throat clearing`), so no `consensus_taxonomy` row for them can reach `n_classifiers: 2`.

## Voice

`declared_voice` positive on 8,306 of 62,547.

| detector | best thr | tp | fp | tn | fn | sens | spec | J |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `voice.yamnet_peak.plain` | 0.05 | 7,522 | 3,258 | 50,956 | 782 | **0.9058** | **0.9399** | **0.8457** |
| `voice.yamnet_chant_peak.plain` | 0.02 | 7,555 | 4,337 | 49,877 | 749 | 0.9098 | 0.9200 | 0.8298 |
| `voice.yamnet_mantra_peak.plain` | 0.02 | 7,551 | 4,582 | 49,632 | 753 | 0.9093 | 0.9155 | 0.8248 |
| `voice.yamnet_peak.enhanced` | 0.05 | — | — | — | — | 0.8663 | 0.9364 | 0.8027 |
| `voice.yamnet_peak.plain+amplitude≥3s` | 0.02 | — | — | — | — | 0.8371 | 0.9580 | 0.7951 |
| `voice.ast_peak.plain` | 0.005 | — | — | — | — | 0.7914 | 0.9092 | 0.7005 |
| `voice.longest_amplitude_span` | 3 s | 7,138 | 11,394 | 42,847 | 1,168 | 0.8594 | 0.7899 | 0.6493 |
| `voice.total_amplitude_span` | 5 s | — | — | — | — | 0.7615 | 0.5510 | 0.3125 |
| `voice.yamnet_peak.residual` | 0.02 | — | — | — | — | 0.2445 | 0.9547 | 0.1992 |
| `voice.longest_continuity_span` | 4 s | 31 | 39 | 54,202 | 8,275 | 0.0037 | 0.9993 | **0.0030** |

`voice.longest_amplitude_span ≥ t`, the owner's proposed rule, in full:

| t (s) | tp | fp | tn | fn | sens | spec |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,140 | 43,185 | 11,056 | 166 | 0.9800 | 0.2038 |
| 2 | 7,841 | 24,569 | 29,672 | 465 | 0.9440 | 0.5470 |
| 3 | 7,138 | 11,394 | 42,847 | 1,168 | 0.8594 | 0.7899 |
| 4 | 6,172 | 5,365 | 48,876 | 2,134 | 0.7431 | 0.9011 |
| 5 | 5,280 | 2,750 | 51,491 | 3,026 | 0.6357 | 0.9493 |
| 7 | 3,824 | 1,023 | 53,218 | 4,482 | 0.4604 | 0.9811 |

**A 2 s floor is much worse than a 3 s floor** — 24,569 false positives against 11,394 — which
reproduces the earlier small-sample finding at 62,550. Its false positives are connected speech:
`caterpillar-passage` 0.712, `story-recall-v2` 0.664, `cinderella-story` 0.659,
`picture-description` 0.640, `free-speech` 0.509 all clear 3 s.

**The YAMNet chant family beats the duration rule outright.** At 0.05 it is 4.5 points more
sensitive and 15 points more specific than the 3 s duration floor, and its per-family separation is
clean:

| family | `voice.yamnet_peak.plain ≥ 0.05` |
| --- | --- |
| `prolonged-vowel` (declared) | 0.930 |
| `maximum-phonation-time-v2` (declared) | 0.927 |
| `maximum-phonation-time` (declared) | 0.911 |
| `glides-low-to-high` (declared) | 0.907 |
| `glides-high-to-low` (declared) | 0.862 |
| `high-to-low` (declared) | 0.860 |
| `random-item-generation` | 0.306 |
| `diadochokinesis-v2-puh` | 0.291 |
| `cinderella-story` | 0.275 |
| every other family | ≤ 0.242 |

At the individual-label level (peak ≥ 0.2), YAMNet `Chant` reaches 0.847 / 0.844 / 0.841 on
`maximum-phonation-time-v2` / `maximum-phonation-time` / `prolonged-vowel`, 0.591 / 0.581 / 0.515
on the three glide families, and at most 0.121 in any other family.

**`voice.longest_continuity_span` is useless and worse than useless below 3 s** — Youden is negative
at every threshold from 0.25 s to 2 s, i.e. a long continuity span is *anti*-correlated with a
declared voice task. This corroborates the existing note in `dag.md` that a continuity candidate
overlapping an amplitude span is absorbed as `corroborated_by` rather than surviving as a span; the
consequence is that the standalone continuity span count carries no phonation signal at all.

## Detectors that are useless, stated plainly

| detector | best J | why |
| --- | --- | --- |
| `{speech,airway,voice}.ast_peak.consensus` | 0.0000 | AST has no `span_ast` measurement; it cannot enter `consensus_taxonomy` (`taxonomy.py:380`) |
| `airway.residual_enhanced_energy_fraction` | −0.0001 | fires on 99.99% of everything at every threshold |
| `voice.longest_continuity_span` | 0.0030 | negative below 3 s |
| `speech.yamnet_peak.residual` | 0.0429 | the residual is what enhancement removed |
| `speech.hear_peak.residual` | 0.0567 | as above |
| `voice.yamnet_peak.residual` | 0.1992 | as above |
| `airway.hear_peak.span` | 0.3661 | the per-span HeAR reading is a worse airway detector than the whole-file one (0.5700) |
| `airway.yamnet_peak.enhanced` | 0.3569 | enhancement removes the airway content along with the noise |

`airway.hear_peak.span` is the one that matters operationally: `_span_label_evidence`
(`taxonomy.py:167`) is written to read exactly that per-span HeAR measurement as airway's
authoritative line, and on this corpus it is the weakest usable airway detector measured.

## Reproducing

```bash
uv run python scripts/analyze_routing_evidence.py <run_dir> <out_dir> [--workers N] [--expect N]
```

The run reported here, on ORCD (one node, 32 workers, ~3 minutes wall):

```bash
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"
/orcd/scratch/bcs/002/satra/senselab-bench/.venv/bin/python \
  scripts/analyze_routing_evidence.py \
  /orcd/scratch/bcs/002/satra/triage_full_20260908/out \
  /orcd/scratch/bcs/002/satra/routing_scratch/final_out \
  --workers 32 --expect 62550
```

`prevalence.json`, `sweeps.json` and `index.json` from that run are beside this file. The full set —
`sweeps_by_family.json` (2.0 MB), `label_prevalence.json` (0.5 MB), `disagreements.json` (9.9 MB,
every case with its transcript and evidence), `summary.md` and the 573 MB `features/features.jsonl`
— stays at `/orcd/scratch/bcs/002/satra/routing_scratch/final_out/`.
