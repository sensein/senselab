# What r3 moved

Measured with `scripts/free_speech_review_page.py compare` over the r2 and r3 extracts, both of the
same 11,701 free-response recordings. Counts only; no detected string appears here.

## Totals

| | r2 | r3 | |
| --- | ---: | ---: | ---: |
| recordings | 11,701 | 11,701 | — |
| findings | 22,319 | 22,319 | — |
| marks | 15,595 | 15,595 | — |
| `in_stimulus: true` | 5,477 | 7,508 | **+2,031** |
| `in_stimulus: false` | 12,305 | 13,263 | +958 |
| `in_stimulus: null` | 4,537 | 1,548 | **−2,989** |
| releasable | 3,719 | 3,841 | +122 |
| withheld | 2,972 | 2,850 | **−122** |
| nothing to redact | 5,009 | 5,009 | — |
| not assessed | 1 | 1 | — |

The finding and mark counts are identical, which is the right shape: near-match and
`expected_names` change *how a finding is answered*, not whether it is found. A change in the
detector population would have meant something else moved as well.

The three stimulus states balance exactly: the 2,989 that became checkable split 2,099 matched and
890 unmatched, and a further 68 moved from matched to unmatched elsewhere — so
+2,031 matched, +958 unmatched, −2,989 uncheckable.

## Per family

| family | uncheckable r2 → r3 | matched r2 → r3 | withheld r2 → r3 |
| --- | ---: | ---: | ---: |
| `cinderella-story` | 2,989 → **0** | 0 → **2,099** | 179 → 142 (−37) |
| `picture-description` | 748 → 748 | 0 → 0 | 99 → 99 |
| `picture-description-option1` | 308 → 308 | 0 → 0 | 42 → 42 |
| `picture-description-option2` | 436 → 436 | 0 → 0 | 42 → 42 |
| `productive-vocabulary` | 56 → 56 | 810 → 902 (+92) | 277 → 270 (−7) |
| `story-recall` | 0 → 0 | 1,493 → 1,491 (−2) | 262 → **185 (−77)** |
| `story-recall-v2` | 0 → 0 | 2,318 → 2,330 (+12) | 72 → 71 (−1) |
| `free-speech` | 0 → 0 | 298 → 203 (−95) | 1,090 → 1,091 (+1) |
| `free-speech-v2` | 0 → 0 | 555 → 483 (−72) | 864 → 863 (−1) |
| `open-response-questions` | 0 → 0 | 3 → 0 (−3) | 45 → 45 |

**`cinderella-story` is answered.** All 2,989 of its previously uncheckable findings are now
checkable, and 2,099 of them — **70.2%** — are accounted for by the declared cast of 20. At mark
level that is 1,350 of its 1,964 marks matched, with 614 checked and not in the cast. Its withheld
count falls 179 → 142.

The 614 are the interesting remainder, and the review layer is what would classify them: a name a
participant supplied while retelling, a mis-parse, or a cast member the declaration missed.

**The negative control holds exactly.** `picture-description` and both options are unchanged on
every column: 1,492 uncheckable findings, zero matched, 183 withheld. Nothing is matching there that
should not, which is the result that makes the cinderella movement trustworthy rather than merely
large.

**The largest withheld fall is `story-recall`, not `cinderella-story`** — 262 → 185, −77 against
−37. That is near-match, not the cast: `story-recall` already declared its source story, so the
widened edit budget exempts findings that previously survived redaction and forced a withhold.
Across the corpus 122 recordings moved from withheld to releasable, and 114 of those are the two
`story-recall` families plus `productive-vocabulary`.

## The one result that ran against expectation

`in_stimulus: true` was expected to rise everywhere near-match applies. It **fell** on four families:
`free-speech` −95, `free-speech-v2` −72, `open-response-questions` −3, `story-recall` −2.

That is not a regression. Near-match tightens as well as loosens, and the two populations separate
cleanly by surface length. Separating the three mechanisms at mark level:

| mechanism | marks | median length | under 5 chars | 8+ chars |
| --- | ---: | ---: | ---: | ---: |
| the declared cast answered it | 1,350 | 10 | 5.4% | 67.4% |
| near-match **admitted** it | 185 | 9 | 3.8% | 69.7% |
| near-match **withdrew** it | **313** | **3** | **56.9%** | 30.0% |

`exact_below: 5` requires a token under five characters to match the stimulus exactly. The previous
check was substring containment against the joined stimulus text, so a short token matched whenever
its letters appeared anywhere inside it — including inside an unrelated longer word. Those are the
313, and withdrawing them is the fitting working as specified. What near-match admits sits at the
other end of the distribution, where the one- and two-edit budgets apply: median 9 characters, 69.7%
of eight or more.

**On this subset near-match is net negative on its own: 185 admitted against 313 withdrawn.** The
whole of the +2,031 is the declared cast. That does not contradict the fitted corpus-wide yield of
44.59% → 46.34%, which is measured over every family including the syllable tasks; it does mean the
free-response families are the part of the corpus where the tightening dominates, because their
stimulus text is a question or a single cue word — short common tokens, few real names — rather than
a story. `productive-vocabulary` and `story-recall` gain (103 and 58 marks); `free-speech` and
`free-speech-v2` lose (99 and 112).

## What did not move, and should not have

**Bracketed marks are unchanged**: 542 in both passes, `PERSON` 340 in both, and every other
category identical. The bracket exclusion was already in place before r2; near-match and
`expected_names` do not touch which tokens reach the scanner, only how a finding is answered
afterwards. A change here would have meant one of them was reaching further than its specification.

*(These are marks. The same corpus is 1,007 bracketed words of which 723 `PERSON`; the two counts
are not interchangeable and comparing across them would show a fall that never happened.)*

## The reviewer did not run

| | r2 | r3 |
| --- | ---: | ---: |
| `disabled` | 3,719 | **6,691** |
| `not_run` | 2,972 | **0** |
| no annotation | 5,010 | 5,010 |
| `absent` / `clean` / `flagged` | 0 | **0** |

`not_run` has gone to zero and `disabled` has absorbed it exactly, which is the ladder change
visible in the data: with the short-circuit removed, the only thing standing between a recording and
a review is the config flag, and `redaction.llm_check.enabled` is still false. r3 was CPU-only.

**No recording in this corpus was reviewed.** Nothing here has been corroborated or contradicted by
a reviewer, and no reviewer verdict exists to read. The page states this twice — once in the rail
over the whole corpus, computed from the state tally rather than asserted, and once per card — and
the banner retires itself the moment any recording carries `absent`, `clean` or `flagged`. A pass
with the flag set and a GPU is what would produce a verdict; that has not been authorised.

## One movement in the diff is mine, not the graph's

1,102 marks read `checked, did not match` in r2 and `nothing to check against` in r3 — 1,050 of them
`picture-description` and 45 `productive-vocabulary`. **No finding changed.** This is the boolean-to-
tri-state fix below: those marks were always uncheckable, and the old boolean could not say so.

It is called out because it is exactly the shape of a real result and is not one. The finding-level
table above is untouched by it: `in_stimulus` on a finding has been tri-state in both extracts, so
every number in the totals and per-family tables compares like with like. Only the mark-level flip
census needed separating, and separating it is what made the three mechanisms legible.

## A defect this measurement found in the page

The mark-level stimulus state was boolean while the finding-level one was tri-state, so a mark
nothing could be checked against rendered identically to one that was checked and did not match —
the same collapse fixed on findings a pass earlier, left in place one level up. It surfaced because
the mark-level flip census could not separate "uncheckable became matched" from "unmatched became
matched", which is the distinction `cinderella-story` turns on. Fixed, with the finding panel now
naming all three states on a mark.
