# The 2026-09-19 corpus run

Everything needed to launch, resume, aggregate and read it.

## What it runs

The whole graph — ADMIT → PREPROCESS → TAXONOMY → routing → AIRWAY/SPEECH/VOICE → QUALITY →
REDACT → VERDICT → REPORT — over all 62,578 recordings in
`/orcd/scratch/bcs/002/satra/clipfix_20260913/manifest_all.jsonl`. Hints on, `task_token` off.

This is the first run of the branches at corpus scale. The 2026-09-08 run stopped at TAXONOMY:
`specs/20260817-triage-workflow-dag/corpus-speech-census.md` says what that one does and does not
hold.

## Where everything is

| what | where |
|---|---|
| code | `/orcd/scratch/bcs/002/satra/senselab-corpus`, detached at `f2729d7c` |
| job | `/orcd/scratch/bcs/002/satra/triage_design_20260919/corpus.sbatch` |
| driver | `.../triage_design_20260919/corpus_driver.py` |
| output | `.../triage_design_20260919/run/` — `rows/` and `out/` |
| logs | `.../triage_design_20260919/logs/triage-corpus-*` |

The checkout is the run's own, pinned, and the job refuses to start if `git rev-parse HEAD` is not
`f2729d7c`. A shared checkout was reset under fifteen running jobs on 2026-09-19; the pin and the
check exist because of it.

## Shape and sizing

1,024 slices of ~61 recordings, 200 concurrent on `mit_preemptable`, 8 CPUs and 32 GB each, CPU only.
Sized on a 20-recording smoke that ran the full graph: median **254 s** per recording, mean 265 s,
max 480 s. A slice is therefore ~4.5 h against a 6 h limit, and the corpus is **~23 h** of wall clock.

The limit is deliberately under twice the expected slice: two tasks of an earlier array sat three
hours on one node without flushing a line, and a stalled slice must release its allocation rather
than hold it to a longer wall. Array tasks stagger their start over the first two minutes so 200
tasks do not open the same shared caches at once.

## Resuming

The driver's `complete()` treats a recording as done only when its row says `ok` **and** every
product it names is on disk, and `sweep_partial` removes a half-written run root before a retry. So
resuming is just resubmitting the same array over the same output root — a preemption, a stall or a
wall-clock kill costs one recording.

```bash
sbatch /orcd/scratch/bcs/002/satra/triage_design_20260919/corpus.sbatch
```

## Reading the result

Every row carries the whole decision, not just the two axes — `FileVerdict.record()`, the same
projection the verdict entity and each `run.json` carry — plus the header duration. So the corpus
questions are answered from the rows without reopening a store:

```bash
uv run python scripts/triage_corpus_report.py \
  /orcd/scratch/bcs/002/satra/triage_design_20260919/run --out <somewhere>
```

That writes `corpus_decisions.json` and `corpus_decisions.md`: the two axes and the discard ground,
declared family and which families flag, route state against finding per branch, conformance per
node and per family with what it is about, deviation types, the config paths nobody measured, the
gates no branch could read, the LLM re-read, duration buckets crossed against triage, and every
contributing verdict. Every share carries its denominator.

`build_index.py` remains for the operational join — timings, hosts, hint and language, failures and
node errors — and reads REPORT's `summary.json` rather than the rows.

## What is known to be wrong going in

- **VOICE's flag grounds are under measurement.** In the smoke, 14 of 23 flag records said a branch
  found nothing, 12 of them VOICE, and every `maximum-phonation-time` recording flagged. If that is
  a defect in VERDICT's fold rather than in the branch, the branch evidence in these stores stays
  good and only the fold needs redoing.
- **PII findings are not yet read against the stimulus.** See
  `specs/20260919-pii-against-the-stimulus/design.md`. Expect the raw PII flag rate in this run to
  carry the false positives that document measures, until the annotation lands.
- **Other branches clamp nothing.** `voice.py`, `ddk.py`, `airway.py` and `preprocess.py` all
  compose extents that can run past the decode and, unlike SPEECH, fail silently rather than
  raising. Listed in `specs/20260919-diarization-turns-past-the-decode/design.md`.
