# GPU against CPU for the reviewer, under the resident-worker review-only policy

2026-09-24. What the existing figures say, why none of them answers the question, and the
measurement now running.

## Why the existing figures do not answer it

| figure | source | why it does not apply |
|---|---|---|
| **20.7× GPU over CPU** | `specs/20260817-triage-workflow-dag/llm-check-first-run.md:206` | **n = 1.** One synthetic residue probe, 3 iterations, 34.7 s on an H100 against 718.4 s on 32 cores. Not a real transcript. The CPU arm paid a 45 s bfloat16 decompression of the w4a16 checkpoint *per process*, which a resident worker pays once. The doc's own caveat says the CPU arm had not finished its remaining texts. |
| **3.6 s median per recording** | `llm-check-amortised-load.md`, arm B | **H100**, 12 recordings, and the *passing* population only. Under the old contract: one text, `FINDINGS:` array, 92 output tokens median. |
| **6.1 s median, 80 output tokens** | this session's pass | **A100**, and the old contract. |
| **718 s per recording on 32 cores** | `llm-check-first-run.md` | load-per-round, with the decompression inside it. |

CPU has never been measured under a resident worker at all. Every CPU number in the tree includes a
per-process model load.

## Why the new contract makes even the GPU figure stale

The reviewer now reads two texts rather than one, carries the task's own facts, and emits three
judgments and a proposal rather than one array. The fixed part of the request grew:

| | characters | whitespace words |
|---|---:|---:|
| prompt, old contract | 679 | 105 |
| prompt, new contract | 1,864 | 286 |
| context block (a task with a four-name cast) | +277 | — |

So the fixed part is 2.7–3.2× longer, and the variable part roughly doubles because the original
transcript is sent alongside the redacted one. Input and output tokens are now recorded separately
on every round (`input_tokens`, `output_tokens` in `review_payload`) precisely so this is measured
rather than scaled. **The old contract recorded no input-token count at all**, so the "before"
figure for input tokens does not exist; the character counts above are the measurable part of the
before/after, and the token counts from this run are the new baseline.

## The concurrency each arm can actually obtain

This is the number that decides it, and it is why CPU looked competitive. Read from
`sacctmgr -n -P show qos format=Name,MaxTRESPU,GrpTRES,MaxSubmitPU` on 2026-09-24:

| QOS | GPU cap (per user) | CPU cap (per user) | MaxSubmitPU |
|---|---|---|---:|
| `ou_bcs_low` | `a100=32, h100=32, gpu=64` | `cpu=1792` | 256 |
| `ou_bcs_normal` | `a100=8, h100=8, gpu=16` | `cpu=448` | 256 |
| `pi_satra` | `a100=4, h100=2, gpu=6` (GrpTRES) | `cpu=192` | 256 |
| `mit_preemptable` | `gpu=4` | `cpu=1024` | 448 |
| `mit_normal` | — | `cpu=96` | 448 |

The partitions **share nodes** — `ou_bcs_normal ⊂ ou_bcs_low`, `pi_satra` is two of those nodes, and
all are also in `mit_preemptable` — but the caps are per-QOS and therefore add.

**GPU arm**, one card and 8 cores per task: the cap is `64 + 16 + 6 + 4 = 90` concurrent cards. The
cap is not the constraint; free cards are. Snapshot 2026-09-24 21:00 EDT, unallocated on usable
nodes: `ou_bcs_low` 13 a100 / 0 h100, `ou_bcs_normal` 6 a100 / 0 h100 (a subset of those 13),
`pi_satra` 0, `mit_preemptable` 20 a100 / 16 h100 but capped at 4. **Obtainable now ≈ 17.**

**CPU arm**, 32 cores per task: `1792/32 + 1024/32 + 448/32 + 192/32 + 96/32 = 56 + 32 + 14 + 6 + 3
= 111` concurrent workers. Free cores at the same moment were 2,542 / 18,535 / 1,906 / 112, so the
caps bind and not availability. **Obtainable now ≈ 111.**

So CPU can obtain roughly **6.5× the concurrency** of GPU. A per-recording ratio below ~6.5× would
make CPU the faster arm in wall clock, whatever the per-unit speed says. That is the whole reason
this has to be measured as wall clock over the population and not as a ratio.

Projection, once per-recording seconds are in hand, over the 42,030 recordings carrying lexical
speech:

```
wall_clock_hours = 42030 * seconds_per_recording / concurrency / 3600
```

## The measurement now running

Both arms run the **same command over the same 40 recordings**, differing only in device, under
`keep_worker_resident: true` and the review-only driver (`scripts/extend_llm_review.py`, no
`--apply`, so no audio is written and no graph node but REVIEW runs).

The sample is stratified 10 / 10 / 10 / 10 across the four populations the reviewer now spans —
findings present, scanned clean, scan declined, no lexical word — seeded `random.Random(0)` over
the full manifest. `bench/sample_strata.json` records what it drew.

| arm | partition | resources | job |
|---|---|---|---|
| GPU | `ou_bcs_normal` | `--gres=gpu:a100:1 --cpus-per-task=8 --mem=96G` | **23714619** |
| CPU | `mit_preemptable` | `--cpus-per-task=32 --mem=256G`, `CUDA_VISIBLE_DEVICES=""` | **23714628** |

Pinned at `304c13c208f5dd1eaa1a04e982476b84e57f9858`, asserted in both job scripts with the full
40 characters; a short SHA failed 108 tasks earlier today.

Per-recording seconds, tokens per second, and memory come from the store the driver writes:
`elapsed_s`, `load_s`, `input_tokens`, `output_tokens`, `peak_reserved_mib` and `resident_mib` on
each `redaction_llm_review` measurement, plus `elapsed_s` on the slice summary. CPU memory comes
from `sacct -j <id> --format=JobID,Elapsed,MaxRSS,State`, since `peak_reserved_mib` is a CUDA
counter and reads 0 on CPU.

## The caveat that applies to both arms

**The machine is not idle.** At submission the user held 42 running CPU-only array tasks across
`pi_satra`, `ou_bcs_normal` and `mit_preemptable`, and the `triage-r4-*` arrays were actively
churning. Timings taken under that load are upper bounds, not clean measurements. The *ratio*
between the arms is the more robust quantity, because both arms contend with the same background —
but the CPU arm contends for the resource the background is consuming, and the GPU arm does not, so
even the ratio is biased against CPU here. Both figures should be re-taken on a quiet machine before
anything irreversible is sized from them.

Measured under the same load, the **replay driver** — the thing this one replaces — completed 394
recordings in 9,486 s on one A100 (`triage-llm-23686415_0`, 2026-09-24 17:36 to 20:14), i.e. **24.1 s
per recording**, with the reviewer's own venv started exactly once. That is the figure the
review-only driver has to beat, and it is measured on the same machine on the same day.
