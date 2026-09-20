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

**400 slices of ~157 recordings, 200 concurrent, on `mit_preemptable`, 8 CPUs and 32 GB each, CPU
only, 16 h per slice.** Sized on a 20-recording smoke and a 2,291-recording pilot that both ran the
full graph: median **254 s** per recording, mean 265 s, max 480 s. A slice is therefore ~11.6 h
against a 16 h limit, and the corpus is **~23 h** of wall clock.

The slice count is set by the queue, not by preference: `mit_preemptable`'s QOS caps a user at
**448 submitted jobs**, and every task of an array counts against it. 1,024 slices and 450 slices
are both refused with `QOSMaxSubmitJobPerUserLimit`; 400 leaves headroom for the report and prep
jobs that have to run beside it.

Array tasks stagger their start over the first two minutes so 200 tasks do not open the same shared
caches at once.

**Roughly one slice in fifty stalls.** Four have been seen across three arrays: allocated, running,
and never flushing a single line of output — 5h12m, 2h38m and 2h14m before being cancelled. The
cause is not diagnosed. The time limit and the resume path are what contain it: a stalled slice
dies at its wall and a resubmission picks up its recordings. **Plan on two passes**, not one.

## Resuming

The driver's `complete()` treats a recording as done only when its row says `ok` **and** every
product it names is on disk, and `sweep_partial` removes a half-written run root before a retry. So
resuming is just resubmitting the same array over the same output root — a preemption, a stall or a
wall-clock kill costs one recording.

```bash
sbatch /orcd/scratch/bcs/002/satra/triage_design_20260919/corpus.sbatch
```

### Resume is keyed on completion, not on the commit

`complete()` asks whether a recording's row says `ok` and its products are on disk. It does not ask
which commit produced them. So resubmitting over an existing output root resumes an interrupted run
— and would skip every recording if the code has changed underneath it, silently returning the old
run's answers as the new one's.

**A run on new code gets a new output root.** `run/` for the corpus, `pilot/` for the first pilot,
`pilot2/` for the next, and so on. The commit is recorded in each row and in each slice's log header,
and the job refuses to start on the wrong checkout, but neither of those saves a run pointed at a
root full of older answers.

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
