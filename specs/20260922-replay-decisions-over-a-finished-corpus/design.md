# Replaying the decisions over a finished corpus

Re-run `TAXONOMY → routing → AIRWAY/SPEECH/VOICE → QUALITY → REDACT → VERDICT → REPORT` over the
stores the 2026-09-19 corpus run left behind, reading PREPROCESS out of each store instead of
recomputing it, so the changes made since that run apply to all ~62,548 recordings.

## Why a replay and not a resubmission

The brief this work started from assumed triage inference is content-addressably cached, so that a
plain resubmission of the array would be nearly free upstream of the changed nodes. It is not.
`senselab.utils.tasks.cached_inference` is imported by exactly two modules, both under
`workflows/audio_analysis/` (`stage_context.py`, `stages.py`); nothing under `workflows/triage/`
references it at any level. `specs/20260817-triage-workflow-dag/praat-instrument-audit.md:210-213`
already recorded this and drew the same conclusion.

Three measurements confirm it rather than merely restating the code.

**The whole corpus, per node.** Every one of the 62,518 rows the corpus run wrote carries
`node_seconds` from the driver's own wrapper around the node callables. Summed:

| node | n | mean s | median s | total s | share |
| --- | --- | --- | --- | --- | --- |
| PREPROCESS | 62,489 | 149.93 | 117.91 | 9,368,832 | 90.4% |
| SPEECH | 43,431 | 12.05 | 12.25 | 523,326 | 5.1% |
| REDACT | 17,970 | 18.16 | 14.53 | 326,405 | 3.2% |
| REPORT | 62,518 | 1.83 | 1.19 | 114,526 | 1.1% |
| routing | 62,486 | 0.08 | 0.06 | 5,015 | 0.05% |
| TAXONOMY | 62,486 | 0.07 | 0.04 | 4,255 | 0.04% |
| ADMIT | 62,518 | 0.07 | 0.05 | 4,162 | 0.04% |
| AIRWAY | 24,120 | 0.13 | 0.04 | 3,254 | 0.03% |
| VOICE | 22,623 | 0.09 | 0.04 | 2,039 | 0.02% |
| VERDICT | 62,518 | 0.01 | 0.00 | 372 | 0.004% |
| QUALITY | 62,486 | 0.00 | 0.00 | 190 | 0.002% |

Total 10,358,336 recording-seconds, 2,877 task-hours, over a 51.72 h array window.

**Twelve recordings re-run at the branch tip**, spread over three duration strata and four
routed-branch profiles, timed with the corpus driver's own instrumentation. No node was skipped and
no node was cheaper for having run before; the re-run cost between 1.00 and 1.24 times the original,
a spread dominated by node heterogeneity rather than by reuse.

**The same recording twice inside one process** — the node-independent control, since it removes
the machine from the comparison. The second pass costs 0.97 of the first. That 3% is the one-off
model load amortised over a second recording; there is no other reuse to find.

So a resubmission pays PREPROCESS again, and PREPROCESS is 90.4% of the corpus. A replay that reads
it out of the store pays the other 9.6% — 275 task-hours against 2,877, a factor of 10.4.

## What changed since the corpus run

Against the corpus run's commit `b7d882a9`, only four things downstream of the audio changed
behaviour. Everything else in the 54-file diff is docstring and rationale movement, a reindentation
of `classifier_ontology/2026-09-10.json`, and the viewer.

1. **Gates moved into VERDICT and became task-group-resolved.** The twenty flat threshold keys under
   the branch sections became `gates: {default, by_group, by_family}`, resolved family, then group,
   then default. VERDICT now answers conformance from the readings a branch reported.
2. **SPEECH's PII scan haystack widened.** The scan was already conditional on words outside the
   stimulus at `b7d882a9`; the tip adds the task's carrier to the haystack, so a syllable task's
   carrier counts as asked-for. Strictly fewer recordings reach REDACT.
3. **Diarization narrowed to the enhanced stream.** `diarization.streams` went from
   `[enhanced, residual]` to `[enhanced]`.
4. **REPORT's pagination** draws text pages at the printed margin.

Change 3 is the only one upstream of TAXONOMY, and it is purely subtractive: it removes
`residual_diarization`, which a sweep of the package found no reader for. A store carrying it is a
strict superset of what the tip's PREPROCESS would write, so reading PREPROCESS out of the corpus
store loses nothing a replayed node could have read. This is what makes the re-entry legitimate
rather than merely cheap.

## Re-entry point

At TAXONOMY, not later. The five nodes between TAXONOMY and SPEECH — TAXONOMY, routing, AIRWAY,
VOICE, QUALITY — cost 14,753 s over the whole corpus, 0.142% of it. Re-entering after them would
save nothing measurable and would oblige the driver to reconstruct each one's state from the store
rather than letting the node write it. Re-running them keeps the replayed store internally
consistent at no cost worth naming.

## Are the replayed nodes' inputs all available?

Node by node, against the code rather than by assumption. Two sources: the store, and sidecar files
under the run's own `run/` directory. A finished run root holds `run/store.jsonl`, `run/run.json`,
`run/streams/` (5 FLAC) and `run/derivatives/` (20 `.npz`/`.json`), plus `released/` and `summary/`.

| node | reads | where from | available |
| --- | --- | --- | --- |
| TAXONOMY | `<classifier>_scores` measurements; `span_yamnet`/`span_hear` `raw_scores` | store + `derivatives/*_scores.json` | yes |
| routing | TAXONOMY summaries, `hint` | store + hint | yes, hint rebuilt |
| AIRWAY | `hear_scores` sidecar, streams, `hint.metadata["task_token"]` | store + `derivatives/` + `streams/` | yes |
| SPEECH | consensus transcript, diarization `.npz`, streams, `hint`, `enrollment` | store + sidecars | yes; enrollment was None corpus-wide |
| VOICE | `derivatives/phonation_tracks.npz`, `hint.metadata["task_token"]` | store + sidecar | yes |
| QUALITY | `clip_amplitude`, the `recording` stream | store + `streams/` | yes |
| REDACT | SPEECH's live `pii` entities, the `recording` stream, `hint.expected_speech` | store + `streams/` + hint | yes |
| VERDICT | every node's verdict/report, `ruleset_routing`, `hint`, `ran` | store + hint | yes; `ran` from the driver |
| REPORT | the whole store, sidecar paths | store + `run/` | yes |

Two things are not in the store and must be supplied by the driver.

**`hint`.** `AudioHints` is a `run_triage` parameter that ADMIT accepts and does not store
(`nodes/admit.py:53`, "Unread"). routing, AIRWAY, SPEECH, VOICE, REDACT and VERDICT all read fields
off it. Nothing in `store.jsonl` or `run.json` reconstructs it. The corpus run built it from the
BIDS sidecar beside each recording with its own `build_hint`, which is deterministic given the
sidecar, so the replay rebuilds it the same way — 0.021 s per recording, measured. This is the one
genuine gap in the run's self-description and it is closed by re-reading the input tree, not by the
store.

**`enrollment`.** Also a `run_triage` parameter with no serialized form. It was `None` for every
recording in the corpus run, so the replay passes `None`. A corpus run that used enrollment could
not be replayed without carrying it in the manifest.

**Sidecar paths.** `resolve_stream` (`nodes/common.py:566-570`) honours an absolute stored path and
falls back to `run_dir / path` for a relative one; `taxonomy.py:76`, `airway.py:342` do
`run_dir / str(path)`, which pathlib resolves to the absolute path when the stored one is absolute.
`voice.py:162` is the exception: it reads a hardcoded relative `derivatives/phonation_tracks.npz`.
So the `run_dir` the replay hands the nodes must physically contain `derivatives/`; it cannot be
made to resolve by rewriting stored paths.

## Writing somewhere other than the run it read

`extend.py`'s contract is in-place: `write_store` atomically replaces `run/store.jsonl` under the
run root it was read from. That is right for a driver that adds a measurement, and the replay
defaults to it.

It is not usable for the 2026-09-19 corpus, whose tree must not be modified. The driver therefore
takes `--out-root`, under which each run gets a fresh root that reads through to the finished one:

- `run/derivatives` is a symlink to the finished run's `derivatives/` — nothing among the replayed
  nodes writes there, only TAXONOMY, AIRWAY and VOICE read from it.
- `run/streams` is a real directory seeded with a symlink per finished stream, because REDACT does
  write there (`redact.py:803-804`, `write_stream(redacted, run_dir, STREAM_NAME)`) and so, when
  separation is configured, does SPEECH (`speech.py:1707`). Separation is off in the packaged
  config, but the directory must be writable regardless.
- `store.jsonl`, `run.json`, `released/` and `summary/` are new files in the new root.

Seven inodes per run plus the products, ~875k inodes and ~100 GB over the corpus, against 26 TB free
on the group scratch.

## Superseding the old decisions

The replay re-decides, which is what makes it different in kind from the six drivers that precede
it: those add a measurement the store lacked, and none of them has to retire a decision that is
still correct-looking.

**Retire before replaying, not after.** The established pattern in `extend_ppg_praat.py:350-357` and
`extend_diarization.py:173-180` writes the new measurement first and supersedes the old one only
when the new id differs. That ordering is safe there because the two drivers add one measurement
each and nothing downstream reads it within the same pass. It is not safe here. Between TAXONOMY and
VERDICT the replayed nodes read each other: routing reads TAXONOMY's summaries, REDACT gates on
SPEECH's live `pii` entities (`run._speech_found_pii`), and VERDICT reads every node's verdict and
report. If the previous pass's outputs are still live while the new ones are being written, VERDICT
— the node whose gates are the point of this exercise — sees two of everything.

**What makes retire-first safe is the run id.** Every id is a digest that includes
`ProvStore.run_id` (`prov_store.py:302, 327, 361`). The replay reads the finished store under a run
id of its own rather than the run root's name, so every entity it writes takes an id distinct from
its predecessor's even when the content is byte-identical. Retiring the old entity therefore cannot
retire the new one by collision, which is the failure the write-first ordering exists to avoid.

The cost of that choice is that an unchanged decision does not keep its id, so a differential over a
replayed store compares content, not ids. The corpus replays under
`specs/20260921-gates-in-verdict/` and `specs/20260921-required-and-typical-counts/` already compare
readings as a name-to-value mapping for an unrelated reason, so this costs nothing in practice.

**Scope of the retirement** is every live entity whose generating activity's `node` is one of the
replayed nodes, found through `store.generated_by` and the activity's `node` field. Keying on the
node rather than on a hand-written list of entity types means a node that starts writing a new kind
of entity is covered without the driver being edited.

**The edge.** `extend.supersede` writes `wasInvalidatedBy(entity, activity)` against a new activity
naming what was retired and why, and deletes nothing. A reader after the replay sees the new
decision through the live-entity filters every node already uses, and the old decision plus the edge
between them by reading the store directly. That is what makes the re-run auditable rather than
merely newer.

## Idempotence and re-entry

A preempted slice must resume, and a slice run twice must not stack two supersessions.

The fingerprint predicate the append-only drivers use (`extend_quality.py:125-134`: run the
derivation, write only if `store.fingerprint()` moved) does not work here, because a replay always
moves the fingerprint — that is its purpose.

The predicate instead is a **marker activity** the replay writes last, carrying the config hash and
the code commit it ran under. `ProvStore.activity` derives the activity id from
`(run_id, node, step, parameters)`, so writing the same marker twice is a set-union no-op. A run
whose store already carries a marker with this config hash is `SKIPPED` and nothing is written.

Because `write_store` is atomic and is reached only after every replayed node has run, a task killed
part-way leaves the previous complete store on disk with no marker, and re-entry redoes that
recording from the beginning. There is no partial state to reconcile, and a second supersession
cannot stack because the first one never reached disk.

## The LLM PII check does not ride along

`redaction.llm_check.enabled` is `false`. Enabling it is a GPU pass over roughly 13,464 recordings
and belongs in a second, GPU-partitioned driver, for three reasons.

**Concurrency.** The replay is CPU-only and sized by `MaxTRESPU cpu=1024` on `mit_preemptable` —
128 concurrent tasks at 8 CPUs, which is the ceiling the corpus run itself hit (measured maximum
instantaneous concurrency 132, p95 128). The GPU ceiling on the same partition is
`gres/gpu=4` per user, and on `pi_satra` the group pool is `gres/gpu=6`. Folding the LLM check into
the replay would put all 62,548 recordings behind a 4-to-6-way GPU gate instead of a 128-way CPU
one, for the benefit of the 21% that need it.

**Batching shape.** `redaction.llm_check.keep_worker_resident` was added alongside the gates change
and its own comment says it needs a GPU to itself. Holding one loaded worker across many recordings
is a different driver shape — batch within a process, one process per GPU — from the replay's
one-recording-at-a-time transaction.

**It is a separate decision.** Turning the check on changes what REDACT is given, so it deserves its
own before-and-after over the corpus rather than arriving inside a pass whose question is what the
gates did. Running it second also lets it read the replayed stores, so it starts from the PII
findings the widened-haystack SPEECH actually produced rather than the corpus run's.

## Cost

275 task-hours for the replay against 2,877 for a resubmission. The components, from the corpus-wide
node table plus a measured 0.099 s per recording for the store round trip
(`read_store` 0.045 + `build_hint` 0.021 + `write_store` 0.034, over 200 sampled stores):

| | recording-seconds | task-hours |
| --- | --- | --- |
| everything except PREPROCESS and ADMIT | 985,342 | 273.7 |
| store round trip, 62,518 × 0.099 s | 6,189 | 1.7 |
| replay total | 991,531 | 275.4 |
| a resubmission, for comparison | 10,358,336 | 2,877.3 |

At the measured 128-task ceiling and the corpus run's own 43.5% packing efficiency that is 5.0 h of
wall clock, against 51.7 h for a resubmission; at ideal packing, 2.2 h against 22.5 h. The replay's
tasks are far shorter than the corpus run's 6.8 h median slice, so its packing should be nearer the
ideal — the two existing partial replays over this corpus
(`specs/20260921-required-and-typical-counts/`, two 64-task arrays) each finished a full pass in
2.9 and 3.2 task-hours of a much narrower derivation, with a median task of 158 s.

## What is not measured

- **The original run's cost per PREPROCESS step.** `ProvStore.activity` accepts `started`/`ended`
  and every caller leaves them `None`, so all 51 activities in a finished store carry null times.
  The corpus run's PREPROCESS can be attributed to nodes but not to steps. The step attribution used
  here comes from the re-run probe, which patches `ProvStore.activity` to stamp the interval since
  the previous registration; it is the branch tip's step cost, not the corpus run's.
- **How much the narrowed diarization saves.** It removes one of two pyannote passes, but the corpus
  run's per-step cost is unrecoverable for the reason above, so the saving is bounded by the tip's
  own `enhanced_diarization` step rather than measured on the run that paid for both.
- **How many recordings the widened PII haystack keeps out of REDACT.** It can only be counted by
  running the replay; REDACT's 17,970 recordings and 326,405 s are an upper bound for the tip.
