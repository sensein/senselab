# Resuming the review pass

## Settled, 2026-09-26 — read this first; everything below is the 2026-09-25 history

The campaign is complete at `ef2bb815`. Nothing is running.

| | path |
|---|---|
| corpus (r5, 62,550 = every in-scope BIDS WAV) | `/orcd/scratch/bcs/002/satra/triage_r5_20260925/out` |
| checkout, pinned | `/orcd/scratch/bcs/002/satra/senselab-r5` @ `ef2bb815` |
| review manifest (15,208 with lexical residue) | `triage_r5_20260925/review/review_manifest.jsonl` |
| final re-fold, all 62,550, `review_on.yaml` | `triage_r5_20260925/refold/` |
| parquet, schema 7 | `recording_vectors_r5_v7/`; laptop `~/Downloads/recording_vectors_20260926_v7/` |
| evaluations | `evaluations_r5_20260926_v2/`; laptop `~/Downloads/evaluations_r5_20260926_v2/` |
| page | `free_speech_page_20260926_v2/`; laptop `~/Downloads/free_speech_review_20260926_v2/` |
| 2 corpus-fill recordings (run at `b7d882a9`, replayed at r5) | `tmp_complete/`; both `flag / not_assessed`, one ASR hypothesis only |

What changed since the note below: the PII pathway sees only lexical residue
(`specs/20260925-lexical-only-pii-pathway/`); a reviewer reading is residue only where it proposes a redaction,
on both axes and on every release path. Outcome: 9,930 flagged (baseline 8,915), 3,700 withheld, 864 reviewer
redaction proposals all withheld, 0 released with one.

Open for the owner: 4,369 free-response readings propose only un-hiding (detector over-redaction in open speech);
2,836 withholdings come from REDACT failing where the reviewer proposed nothing to hide; the two fill
recordings need a one-hypothesis consensus or another recogniser for a real verdict.


Written 2026-09-25, mid-campaign. What is running, what is pinned, what is decided, what is open,
and the exact commands to carry on. Branch `design/triage-workflow-dag`.

## In flight right now

| what | job | state |
|---|---|---|
| REVIEW over the corpus | **23879493** | 128 slices, `%36`, a100, 31/128 done, ~97 elements left |

It is **resumable**: a recording whose store already carries a `redaction_llm_annotation` written by
a **REVIEW** activity comes back `present` and costs nothing. So a cancelled or preempted array is
resumed by resubmitting the same sbatch — never by starting over.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/triage_review_20260925 && sbatch review.sbatch'
```

Watch it:

```bash
ssh orcd 'ls /orcd/scratch/bcs/002/satra/triage_review_20260925/rows/slices/*.summary.json | wc -l'
```

## Monitors do not survive the session — restart them first

Every watcher in this campaign is a background shell held by the session that started it. A new
session inherits **none** of them, and nothing on the cluster notices: the array keeps running and
no one is told when it finishes. So the first act on resuming is to re-arm a watcher, before
anything else, or the pass completes silently and sits idle.

The watchers live in the job's scratch directory, which is **also not durable** — it goes when the
job is deleted. Treat them as disposable and rewrite them; the shape is what matters:

```bash
# poll the array, report only real progress, exit when it leaves the queue
prev=-1
while :; do
  read -r slices rows state <<<"$(ssh orcd '
    R=/orcd/scratch/bcs/002/satra/triage_review_20260925
    s=$(ls $R/rows/slices/*.summary.json 2>/dev/null | wc -l)
    r=$(cat $R/rows/slices/*.jsonl 2>/dev/null | wc -l)
    q=$(squeue -j <JOBID> -h -r -o "%T" 2>/dev/null | wc -l)
    echo "$s $r $q"')"
  [ "$state" = "0" ] && { echo "done: $slices/128 slices, $rows rows"; break; }
  bucket=$(( slices / 16 ))
  [ "$bucket" != "$prev" ] && { echo "slices $slices/128 · rows $rows"; prev=$bucket; }
  sleep 900
done
```

Two things learned the hard way about these:

- **Report on progress, not on state jitter.** A watcher keyed to the running-count fires every few
  minutes on a preemptable partition and says nothing. Bucket by completed slices instead.
- **A waiter that treats `PREEMPTED` as terminal exits early**, because a requeued job passes
  through that state and then carries on. Wait for the job to leave the queue.

Re-arm one for whichever of the three steps below is in flight, and one for the next.

## Pinned checkouts — do not cross them

Two checkouts, two jobs, and they must stay separate. Checking one out to a different commit while
a job runs from it fails that job's guard (`exit 74`), and I did exactly this once today.

| checkout | used by | pin |
|---|---|---|
| `/orcd/scratch/bcs/002/satra/senselab-review` | the REVIEW array | `5165925f` |
| `/orcd/scratch/bcs/002/satra/senselab-rvec` | parquet, re-fold, page | current HEAD |

Both now have `origin` at GitHub. `senselab-rvec` pointed at another scratch checkout until today.

## Trees and artefacts

| | path |
|---|---|
| corpus (r4, verified 62,548) | `/orcd/scratch/bcs/002/satra/triage_r4_20260924/out` |
| original corpus, for `run.json` | `/orcd/scratch/bcs/002/satra/triage_design_20260919/run/out` |
| hints (required by every driver) | `/orcd/scratch/bcs/002/satra/triage_design_20260919/scope` |
| review manifest, 62,519 rows | `/orcd/scratch/bcs/002/satra/triage_review_20260925/review_manifest.jsonl` |
| reviewer-on config | `/orcd/scratch/bcs/002/satra/triage_review_20260925/review_on.yaml` |
| parquet shards | `/orcd/scratch/bcs/002/satra/recording_vectors_r4` |
| page extract + html | `/orcd/scratch/bcs/002/satra/free_speech_page_20260925` |
| **on the laptop** | `~/Downloads/recording_vectors_20260925/`, `~/Downloads/free_speech_review_20260925/` |

The r4 tree is a **mirror**: `store.jsonl`, `streams/`, a `derivatives` symlink, and **no
`run.json`**. That is why the manifest carries `source` — `source_of()` has nothing to read there.

## The three remaining steps, in order

Each needs the r4 tree to be settled, so do them after the review array leaves the queue. A census
taken while the array writes is a snapshot mid-write; that contaminated the r4 report once already.

**1. Final re-fold**, to make the whole corpus one fold. See the open question below for the config.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/triage_refold_20260925 && sbatch refold.sbatch'
```

**2. Parquet at schema 6**, then merge and retrieve mode 600.

```bash
ssh orcd 'cd /orcd/scratch/bcs/002/satra/recording_vectors_r4 && sbatch rvec_r4.sbatch'
# then
ssh orcd '... python scripts/triage_recording_vectors.py --merge <dir> --out <dir>'
```

**3. The two evaluations** the owner asked for, with `~/evaluate_flags.py` (staged on the cluster):
flags per family against the **pre-review baseline** (`~/flags_r4_prereview.json`, 8,915 of 61,113
flagged — taken before any reviewer data landed, so it separates the reviewer's effect from the
r3→r4 code), and the free-response population on its own.

Then rebuild the page from the settled tree.

## Decided, and measured

- **story-recall**: 93.7% → 5.0% flagged, `coverage_min` gone from the failing-gate table, 45 of 47
  other families within 1pp. The gate change did one thing.
- **release split**: 26,757 / 12,016 / 4,677 / 19,098 landing exactly on its predicted grounds.
- **non-lexical cleared** (owner, 2026-09-25): `not_assessed` 19,098 → **1**. Totals then reproduce
  r3's permissive classification but with a named ground per recording.
- **brackets dropped for the reviewer**: 29.9% of recordings render empty and cost no GPU. Not a
  false-positive fix — a probe over 20 marker-only recordings read all 20 clean.
- **loop convergence**: 99.2% of multi-round recordings gained nothing after round one, because
  masking an empty proposal is a no-op and the next round re-read an identical string. 39% of GPU
  time. Fixed.

## Open for the owner

**`verdict.llm_redaction_withholds`.** Marked `UNFITTED` in the packaged config. On, essentially
every flagged reading becomes a withholding: the sample put withheld at 22.6% against 7.5% before,
so roughly 14,000 rather than 4,647. The review array runs with it **on** (`review_on.yaml`); the
last re-fold ran with the packaged config, which is **off**, and silently reverted it.

This is cheap either way and that is the point: the **reading** is the expensive, permanent half,
and the weighting is pure fold. Whichever way it goes, one re-fold settles the corpus.

## Traps this campaign has actually hit

- A census over a tree a job is writing is a snapshot mid-write.
- `MaxSubmitPU` counts **expanded array elements** (380 once), not job ids.
- Entitlement is not availability: 36 cards allowed, 8–13 obtainable.
- The L40S OOMs this checkpoint at 38.3 GiB; it needs an 80 GB card.
- `--gres=gpu:1` untyped reaches an L40S first.
- Print a record's keys before filtering on them. `tokens_n`, not `n_words`.
- A substring assertion cannot see a JavaScript syntax error. 112 of them missed a broken page.
- Derive a commit SHA from `git rev-parse`, never from recall.
