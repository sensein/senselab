# The LLM re-read, run for the first time

2026-09-19, on ORCD `pi_satra`. `redaction.llm_check` had never executed against a model: every
test stubs `review_redacted_text` or calls `parse_completion` directly. This is what it does when it
is switched on over real recordings.

**It runs.** The shipped path — `redact.py` → `redaction_review.review_redacted_text` → the
`pii-redaction-review` subprocess venv → `google/gemma-4-31B-it-qat-w4a16-ct` — worked on the first
attempt, with no change to production code. Nothing here required a fix to make the run happen.

No transcript, no detected string and no model reasoning appears in this document. Counts,
categories, shapes and timings only; the illustrations are synthetic. The run's own outputs stay on
ORCD scratch under `/orcd/scratch/bcs/002/satra/llmcheck_20260919/`.

## What it took to get there

Neither checkpoint is gated. Both were fetched unauthenticated, with no `HF_TOKEN`, into
`HF_HOME=/orcd/scratch/bcs/002/satra/hf`:

| checkpoint | size | resolved commit | staged in |
| --- | --- | --- | --- |
| `google/gemma-4-31B-it-qat-w4a16-ct` | 23.3 GB | `52f3f65b…8219d` | 221 s |
| `google/gemma-4-31B-it` | 62.6 GB | `842da379…ff475` | 558 s |

`ensure_venv` built `pii-redaction-review-cu128` in 177 s: Python 3.12.13, torch 2.8.0+cu128,
**transformers 5.17.0**, accelerate 1.15.0, compressed-tensors 0.15.0.1. The host venv that drove
the graph carries transformers 5.5.4 and torch 2.11.0+cu130 — a different stack, which is what the
isolation is for.

Hardware: `pi_satra` holds `node2803` with 4× H100 80 GB HBM3 and `node3805` with 8× A100 80 GB
PCIe. **Both card types are 80 GB.** Every number below that says "GPU" was measured on an H100.

## The four questions

### Does it find residue the detector missed?

**On this corpus, no — not once.** The re-read ran on 12 recordings and returned `clean` on all 12,
in one iteration each. It flagged nothing that the detector cascade had left behind.

That is a real answer, not a null one, and it is only interesting because the same 12 transcripts had
just been through a detector that fired hard: two diadochokinesis recordings carried **20 and 18**
detector findings each, on transcripts of 14 characters. The model read what was left and said there
was nothing identifying in it.

The honest caveat is that the redacted transcripts reaching the re-read are *short*. Their median
released length is 32 characters; only three exceed 300. A quasi-identifier argument — the thing this
step exists for — needs several attributes in one text, and a 14-character DDK transcript cannot
contain them. The step was asked the question it was designed for on roughly three recordings out of
26.

### Does it flag task material as PII?

**No, and it argues explicitly about why not.** This was the failure mode most worth fearing, given
that the *detector* flags task material constantly (below). On a Cinderella retelling the model
identified the characters as folk-tale archetypes rather than real people and declined; on an
open-response recording it weighed whether a role plus an institution type was jointly identifying,
concluded it was not, and named what would have changed its mind. Across all 12 reasonings, 12 of 12
refer to the task or stimulus context.

By contrast the detector cascade, re-run over 579 stored consensus transcripts spread across 120
subjects, flags **266 (45.9%)** — including 21 of 25 maximum-phonation recordings, 19 of 25 glides,
and 18 of 25 diadochokinesis, whose transcripts are a handful of syllables. Detector spans found:
PERSON 261, DATE_TIME 147, NAME 75, LOCATION 72, MISC 15, LOC 12, NRP 6, and single digits of
VEHICLE_IDENTIFIER / UNIQUE_IDENTIFIER / DATE / AGE. **The false-positive problem on this corpus is
the detector's, and the re-read is the only component that pushed back on it** — though only in its
reasoning, since it can withhold but not un-redact.

### How long does it take?

Two very different answers, and the gap between them is the most actionable thing in this document.

| | per review round | per checked recording |
| --- | --- | --- |
| **as shipped** (a fresh subprocess and a fresh model load per round) | ~62 s | ~75 s |
| **weights resident** (same prompt, same parse, same loop; model loaded once) | 1.9–18.5 s, median 4.6 s | ~7 s |

The shipped figure is measured directly: one `review_redacted_text` call on a warm venv with the
snapshot already staged took **61.7 s** (BF16) and ~64 s (QAT, after subtracting the 177 s venv
build from a 240.7 s first call). The resident figure comes from the same code's prompt, parse and
masking driven over a persistent worker: **load 9.9 s (QAT) / 13.7 s (BF16), then 40.2 ms and
39.0 ms per output token**, with a median of 117 output tokens per round.

So roughly **90% of the shipped per-round cost is process startup and model load**, paid again for
every round and again for every recording. End to end, the full triage graph took 145–170 s on short
recordings and 312–425 s on long ones; the re-read is a ~60–100 s addition to that.

### Is the captured reasoning worth storing?

**On anything with content, yes. On three-syllable transcripts, no.**

Measured over the 12 real reasonings: 190–632 characters, median 456. Pairwise word-set Jaccard has
a median of 0.27 — they are not one template. 8 of 12 contain an explicit weighing word
("considered", "whether", "however"), 9 of 12 refer to the redaction placeholders they were handed,
and 12 of 12 name the task context.

Split by length, though: among transcripts ≥150 characters the reasonings are all distinct (max
pairwise Jaccard 0.43), while among transcripts under 60 characters **one pair is word-for-word
identical** (Jaccard 1.00) — two different DDK recordings, 14 characters each, got the same
paragraph. That is boilerplate, and it is exactly what you would expect: there is nothing to reason
about, so the model says the same thing twice. The design rests on the chain of thought being
informative, and it is — in proportion to how much text there is.

## Per recording

26 recordings: 20 the detector flagged (10 with transcripts ≥150 chars, 10 below, spread across
families), 6 controls it did not. Each run end to end through `scripts/triage_audio.py` with a
two-line `--config` override setting `redaction.llm_check.enabled: true` and nothing else.

| family | detector findings | detector categories | REDACT | llm_check | iters | released chars | reasoning chars |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cape-v-sentences | 2 | NAME, PERSON | pass | clean | 1 | 27 | 456 |
| cinderella-story | 2 | DATE_TIME | pass | clean | 1 | 888 | 632 |
| diadochokinesis-pa | 2 | LOCATION, PERSON | pass | clean | 1 | 18 | 239 |
| diadochokinesis-pataka | 20 | NAME, PERSON | pass | clean | 1 | 14 | 301 |
| diadochokinesis-buttercup | 18 | NAME, PERSON | pass | clean | 1 | 14 | 301 |
| free-speech | 3 | DATE_TIME | pass | clean | 1 | 327 | 624 |
| harvard-sentences | 2 | NAME, PERSON | pass | clean | 1 | 36 | 410 |
| loudness | 1 | PERSON | pass | clean | 1 | 9 | 190 |
| open-response | 2 | DATE_TIME, MISC | pass | clean | 1 | 414 | 598 |
| productive-vocabulary-1 | 4 | PERSON | pass | clean | 1 | 190 | 705 |
| productive-vocabulary-2 | 5 | NRP, PERSON | pass | clean | 1 | 19 | 371 |
| productive-vocabulary-5 | 3 | LOCATION, PERSON | pass | clean | 1 | 190 | 564 |
| animal-fluency | 6 | NAME, PERSON | fail | not_run | 0 | — | — |
| caterpillar-passage | 3 | DATE_TIME | fail | not_run | 0 | — | — |
| picture-description | 8 | PERSON | fail | not_run | 0 | — | — |
| picture-description | 7 | LOCATION, NAME, PERSON | fail | not_run | 0 | — | — |
| productive-vocabulary-1 | 1 | PERSON | fail | not_run | 0 | — | — |
| productive-vocabulary-3 | 6 | PERSON | fail | not_run | 0 | — | — |
| 8 further recordings | 0 | — | *skipped* | *no record at all* | — | — | — |

**Only 12 of 26 recordings reached the re-read.** Six had REDACT withhold (its own verification
re-scan still found a surviving category), and eight had no detector finding at all, so REDACT never
ran. On the sample as a whole the enabled step produced a model verdict on 46% of recordings.

## The properties, checked against a real run

| property | verdict |
| --- | --- |
| the model is the QAT w4a16 31B checkpoint | yes — `model_id` in all 12 annotations |
| a ref is resolved to a commit and the load happens at the commit | yes — `resolve_revision` returned `52f3f65b…`, the worker loaded the snapshot directory named by that SHA, and all 12 annotations carry a 40-hex `revision` |
| the step iterates, bounded by `max_iterations` | **not exercised by corpus data** — every real recording stopped at round 1. Exercised on synthetic residue: 2 and 3 rounds, bound never reached |
| the chain of thought is captured and stored | yes — one `redaction_llm_review` per round carrying the reasoning verbatim, and `report.py` renders it |
| it can only withhold, never mint a redaction | yes — **0 of 12 released transcripts contain an `[LLM_…]` placeholder**; the loop's masking never leaves the loop |
| an unavailable model is recorded absent | yes — probed deliberately (below) |
| it does not run when the detector path already withheld | yes — all 6 REDACT `fail` recordings recorded `not_run` with `revision: null` |
| REDACT's verdict carries no `llm_check` field | yes — 0 of 26 |
| VERDICT reads the annotation, not the release axis | consistent — no flagged annotation arose to test the raise |

### The absent path, probed deliberately

Since the model was available on every real call, the `absent` path was exercised on purpose: one
recording re-run with `redaction.llm_check.model_id` pointing at a repo that does not exist. The
result is exactly what the design asks for and nothing more:

- the round's measurement carries `available: false` and a `failure` naming a
  `RevisionResolutionError` and the Hub's 401, with an empty `reasoning`;
- the annotation carries `status: absent`, `revision: null`, and the same failure string;
- **REDACT's own outcome is unchanged (`pass`) and the release stays `releasable`** — the detector
  path's answer stands, and the recording's fate does not depend on whether a GPU was reachable;
- nothing about it is silent: the failure is in the store, in the summary JSON's `llm_annotation`,
  and in the file verdict's `llm_redaction`.

One gap in the record worth naming: when the *runner* skips REDACT (no detector finding), nothing at
all is written — no activity, no annotation. The spec's "written on every path" holds inside
`redact()`, but from the store alone "the check was never reached" and "the node never ran" look the
same; only `run.json`'s node outcomes separate them.

## QAT against the full BF16 checkpoint

The QAT variant was chosen because 62.5 GB of BF16 "does not fit a 40/48 GB card". Every card this
account can reach is 80 GB, so that constraint does not bind here, and both were run over the same
15 texts.

| | QAT w4a16 | full BF16 |
| --- | --- | --- |
| load (warm cache, H100) | 9.9 s | 13.7 s |
| per output token | 40.2 ms | 39.0 ms |
| median round | 4.6 s | 4.2 s |
| verdicts on 12 real transcripts | 12 clean | 12 clean |
| synthetic residue probes | both flagged | both flagged |
| categories flagged on the probes | identical | identical |
| iterations on probe 1 | 3 | 2 |
| reasoning length (median) | 564 chars | 544 chars |

**They agree on every verdict and every flagged category.** The only difference is that on one
synthetic probe the QAT model needed a third round to settle where BF16 settled in two, and on the
shipped-path smoke BF16 listed 7 findings where QAT listed 6. There is no measured quality reason to
switch, and no memory reason to stay on 80 GB cards — so the QAT default is fine, but the derivation
should say "it also fits smaller cards", not "the alternative does not fit".

## CPU

Same code, same prompt, same loop, 32 cores and 256 GB on `mit_preemptable`, QAT checkpoint — the
only realistic CPU candidate, and for a reason that only shows up when you run it.

**The w4a16 packing is a storage format, not a runtime one.** On CPU the load reports 9.5 s and is
then followed by 45 s of "Decompressing model", after which the resident parameters are `bfloat16`.
So the QAT variant does not buy 23 GB of host memory; it buys 23 GB of *disk* and then expands to
roughly the same footprint as the full checkpoint. Anyone choosing QAT for a memory-constrained CPU
host on the strength of the file size would be choosing it for a property it does not have there.

The matched comparison, same text, same model:

| | QAT on H100 | QAT on 32 CPU cores | ratio |
| --- | --- | --- | --- |
| synthetic residue probe 1 | 34.7 s, 3 iterations, flagged | **718.4 s**, 3 iterations, flagged | **20.7×** |
| synthetic residue probe 2 | 17.0 s, 2 iterations, flagged | **382.2 s**, 2 iterations, flagged | **22.5×** |

The verdicts, the flagged categories and even the iteration counts are identical; only the clock
differs. That is the useful half of the answer: **CPU does not change the behaviour, it changes the
price by a factor of about 20.**

Scaled to the corpus that is decisive. At ~145 s per checked recording (the GPU-amortised ~7 s times
~21), ≈15,900 recordings is **≈640 node-hours on a 32-core node** — a week on one node, and that is
before the shipped per-call model load, which on CPU also pays the 45 s decompression every time.

**So the step is a GPU step, and the design should say so plainly rather than implying it.** CPU is
viable for a spot check — 2 to 12 minutes for one recording — and not for a corpus.

The CPU arm was still working through the remaining texts when this was written; the ratio rests on
the two matched multi-round probes and on the load behaviour, all measured. The GPU arms completed
all 15.

## What the corpus would cost

From a census of all 62,578 recordings in the Sept-8 run: **62,521 carry a `consensus_transcript`
and 60,207 carry a non-empty one.** Median transcript length varies 250-fold across families —
caterpillar 1044 chars, Cinderella 961, story-recall 529, picture-description 419, free-speech 280,
harvard-sentences 39, respiration 8, maximum-phonation 4 — so a sample skewed to one family would
mislead, and the estimate below is weighted family by family.

Applying each family's measured detector hit rate to its corpus population gives **≈23,800
recordings (39.5%) that would reach REDACT**. In this sample REDACT then passed 12 of 18 times, so
**≈15,900 recordings would actually get a model verdict** — a quarter of the corpus.

| | per recording | corpus | on 4 H100s |
| --- | --- | --- | --- |
| as shipped (load per round) | ~75 s | ~330 GPU-hours | **~3.5 days** |
| weights resident | ~7 s | ~31 GPU-hours | **~8 hours** |

Assumptions: 1.2 rounds per checked recording (measured); generation cost tracks output tokens, which
barely move with input length in this range (median 117 output tokens whether the transcript is 14 or
888 characters — the model writes a paragraph either way); perfect parallel efficiency across GPUs;
no queue wait. The soft edge is the iteration count: every corpus recording stopped at one round, but
a corpus with more long-form speech would flag more and iterate more, and each extra round is another
full cost unit.

The single biggest lever is not the model or the card. **It is that the shipped path pays a process
start and a model load per round.** Amortising the load turns a three-and-a-half-day job into an
overnight one.

## What I would change

1. **Amortise the model load.** `_llm_check` calls `review_redacted_text` once per iteration, and
   each call spawns a subprocess that loads 23 GB from scratch. Even within one recording, a
   3-iteration check pays three loads. A worker that stays alive for the duration of a run — or a
   batch entry point that takes many transcripts — is worth ~10× on corpus-scale use, and it changes
   the step from "cluster-only" to "plausible on any GPU host".
2. **Record the step's own cost.** The `llm_check` activity is written with no `started`/`ended`,
   and no per-round timing reaches the store. For a step that can take minutes, the store cannot
   answer "how long did this take" at all. Every timing in this document had to be measured from
   outside the graph.
3. **Raise the transformers floor.** `REVIEW_REQUIREMENTS` asks `transformers>=4.57`; the shipped
   checkpoint declares `transformers_version: 5.8.0.dev0` and architecture `Gemma4ForConditionalGeneration`.
   The run worked because the resolver happened to choose 5.17.0. A host that resolves an older
   satisfying version fails at `from_pretrained`, not at install, which is the worse place to fail.
4. **Say plainly what the step is for, given what it does here.** It found no residue in 12 tries and
   its main observed value was the opposite: reasoning that correctly declines to treat task material
   as identifying. If that is the value, the step is a check on the *detector's* false positives as
   much as on its false negatives — and it cannot act on that, because it may only withhold. Whether
   an "the detector over-redacted this" annotation should exist is a design question this run raises
   and does not answer.
5. **Note in the derivation that the QAT choice is no longer forced.** Both checkpoints fit the
   hardware and agree on every verdict measured here.

## Reproducing

```
/orcd/scratch/bcs/002/satra/llmcheck_20260919/
  stage.sbatch     one checkpoint, resolved and staged
  screen.sbatch    detector cascade over stored consensus transcripts, to pick a sample
  pipeline.sbatch  scripts/triage_audio.py per recording with llm_on.yaml
  arena.sbatch     the three timing arms over the redacted transcripts the pipeline produced
  work/            selection, harvest, census and arena outputs (carry transcript text — private)
  llm_on.yaml      redaction.llm_check.enabled: true
```
