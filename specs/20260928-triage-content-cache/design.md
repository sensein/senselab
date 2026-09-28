# A result cache for every process the triage graph reruns

Owner, 2026-09-28: *"caching should be on for all processes that operate on the same input with the
same process (changes in either invalidates the cache)."*

## What was recomputed, and why nothing caught it

Measured on r7's replay (`extend_replay_decisions.py` at `77e9273e`, from `triage_design_20260919/run/out`).

A 20-recording sample from slice 5, 6 of which separate, was replayed under `cProfile` on node2803
(job 24176348). It took 885 s wall:

| call | node | calls | total | per call | share |
|---|---|---|---|---|---|
| ClearerVoice `MossFormer2_SS_16K` separation, subprocess worker | SPEECH step 5 | 6 | 589 s | 98 s | 67% |
| PII detectors (GLiNER + Presidio + rules), `subprocess.run` | SPEECH step 7, REDACT verify | 12 | 156 s | 13 s | 18% |
| everything else (fold, report, prov export) | — | — | ~140 s | — | 15% |

No second diarizer or speaker-embedding call ran in the sample; the label scores AIRWAY, TAXONOMY
and REPORT read are stored window entities, not model calls.

Three facts made this cost recur on every replay:

1. **Triage has no persistent cache.** `cached_inference` served only `audio_analysis`. Triage had
   `lru_cache` over in-process lookups and nothing across runs.
2. **The replay recomputes every stream a replayed node writes.** `mirror_run_root` links the streams
   a replay only reads and deliberately leaves the ones a replayed node writes (`separated_*`,
   `redacted`), so SPEECH separated the same audio again in each of r4, r5, r6 and r7.
3. **The input really was identical.** For `sub-2bb59c22…_task-picture-description`, `separated_*`
   in r4, r5, r6 and r7 all came from `enhanced` (a symlink to the design run's file), commit
   `407cb030…`. The four outputs agree to 1.6e-6 (correlation 1.00000000): CPU float noise between
   nodes, not a different result.

The owner's suspicion holds: the separation reads `enhanced` (`signal: enhanced`, the stream
diarization counted speakers on), and each replay redid it. It had not been done by the design run
for this recording (no `separated_*` there); the first replay did it, and every later one repeated it.

Two further obstacles would have defeated a cache keyed the way `audio_analysis` keys its own:

- `senselab_version` is `hatch-vcs`'s distance-from-tag string (`1.3.1a45.dev1283` locally,
  `dev1234` on the cluster checkout). It changes with every commit, so a key containing it misses on
  every replay at a new commit whether the process changed or not.
- `prune_unreachable_entries` deletes any entry whose task is not in `audio_analysis`'s
  `STAGE_VERSIONS`, so triage entries in the same directory would be wiped by the next
  `analyze_audio` run.

## The rule

An entry is keyed on **what the process reads** and **what the process is**, and on nothing else:

- the input's content: `audio_signature` of the prepared audio the worker receives (after resample
  and downmix), or `transcript_signature` of the exact text scanned;
- the process name and its behaviour version (`RESULT_PROCESS_VERSIONS`), bumped by hand when what
  it returns for the same input changes;
- the model id and the 40-hex commit its weights resolved to;
- every parameter that shapes the result (for PII: detector set, Presidio entities and threshold,
  GLiNER labels, threshold and label map, the digests of the worker script and `rules.py`, the
  venv's pinned requirements and Python).

A change in any of them is a different key. The senselab version, host, device and run id are
recorded on the entry as provenance, not keyed: they do not change what the process computes (the
device changes the float noise, measured above at 1.6e-6).

## Where the cache sits

In the library calls, not in the triage nodes, so every caller gets it: `run_clearvoice_over_audios`
(enhancement, separation, super-resolution) and `detect_pii_via_subprocess`. Both key per input:
a batch sends the worker only the inputs it lacks, and a fully held batch starts no subprocess, stages
no model and builds no venv.

Only a complete PII scan is stored: a detector that failed to load leaves a result a retry could
improve.

The store lives beside `SENSELAB_CACHE` at `results/schema-<CACHE_SCHEMA_VERSION>/`, one directory per
key holding `result.json` and lossless `.npy` arrays. A schema bump reads a fresh directory rather
than wiping one 128 concurrent array tasks may be reading. `SENSELAB_RESULT_CACHE` overrides the root
or switches the cache off (`off`). Writes are atomic (assembled privately, renamed into place; the
first writer of a key wins).

## Provenance of a reuse

A library call cannot name the provenance graph it runs in, so the triage node that does names itself:
on a miss, SPEECH (separation, PII scan) and REDACT (verification scan) call `annotate_result_origin`
with their run id and activity id. On a later hit the store records, on the separated stream and in the
`pii_scan` measurement and REDACT's verdict detail, the cache key, `hit: true`, and the origin's run and
activity. The separation activity carries `cache_key` and `cache_hit`.

## Measured

The same 20 recordings, replayed twice into scratch copies with an empty cache, on node2803
(job 24177422, `cProfile`):

| pass | wall | ClearerVoice worker | separation | PII scans | subprocess.run |
|---|---|---|---|---|---|
| cold | 834 s | 6 calls, 568 s | 12 streams, all miss | 21, all miss | 75 calls, 700 s |
| warm | 135 s | not started | 12 streams, all hit | 21, all hit | 21 calls, 0.06 s |

41.7 s → 6.8 s a recording. The warm pass made the same decision (triage, release, release ground) on
all 20 recordings, and its 12 separated streams equal the cold pass's bit for bit (they are the same
cached arrays). 25 entries took 28 MB.

## Not done

- A resident PII worker. With the cache, the subprocess starts only for unseen text; a resident worker
  would still help a first run, but needs a request/response protocol over the worker's stdio, which is
  more than this change.
- `audio_analysis`'s own `cache_key` still includes `senselab_version`, so it misses after every
  commit. The same reasoning applies to it; changing it is outside this change.
