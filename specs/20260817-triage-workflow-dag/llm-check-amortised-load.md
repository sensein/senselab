# Amortising the LLM re-read's model load

2026-09-19, following `llm-check-first-run.md`, which measured the step for the first time and named
this as the first of the five things to change:

> **Amortise the model load.** `_llm_check` calls `review_redacted_text` once per iteration, and each
> call spawns a subprocess that loads 23 GB from scratch. […] A worker that stays alive for the
> duration of a run — or a batch entry point that takes many transcripts — is worth ~10× on
> corpus-scale use.

This document is that change, the second item beside it (the step recording its own cost), the third
(the `transformers` floor), and what all three measure.

No transcript text, no detected string and no model reasoning appears here. Counts, lengths and
timings only.

## A worker, not a batch entry point

Both were on the table. The worker wins on how the graph is actually driven, and the batch entry
point loses on something structural rather than on effort.

**How the step is reached.** `redact()` calls `_llm_check` once, at the end of REDACT, on a
recording whose detector path passed. REDACT is the last node of the SPEECH branch, and VERDICT —
which reads the annotation — runs after it in the same graph. A batch entry point taking many
transcripts would therefore have to hold every recording's REDACT open until every other recording's
REDACT had finished, and only then let any VERDICT run. That is a two-pass graph, and the graph is
one pass per recording by construction: one store, one run directory, one verdict, written as the
recording is processed. Batching the re-read means either buffering N stores in memory or writing
each recording's verdict twice, and the second is the thing `llm-check.md` already refuses ("a field
in a verdict the verdict does not act on is a second record able to disagree with the first").

**Where the loads actually are.** The two drivers differ, and only one of them is where a corpus is
run:

| driver | recordings per process | what a per-process worker amortises |
| --- | --- | --- |
| `scripts/triage_audio.py` | 1 | the rounds of one check — up to `max_iterations`, 3 by default |
| the corpus driver (`corpus_driver.py`, one Slurm array task per slice) | one slice, ~156 on a 400-slice run | every checked recording in the slice, rounds included |

So a worker that lives as long as the process gets the whole win on the corpus driver — which is the
only thing that runs at corpus scale — and the within-recording win on the single-file path, with
**no change to either entry point**. `review_redacted_text` keeps its signature; the amortisation is
underneath it. That is what makes the single-recording path impossible to break: it is the same
call, and a process that makes exactly one review behaves exactly as it did, one load and all.

What that argument missed, and the measurements below found, is that the corpus driver is also the
one place the worker has to share a GPU with the rest of the graph — and it cannot. So the design
stands, the win is real, and it is gated on a config key rather than taken by default.

## What the worker is

One subprocess in the `pii-redaction-review` venv, keyed on `(model_id, resolved commit)`, started
by the first review in a process and reused by every later one.

The protocol is one JSON object per line over the worker's stdin and stdout. The worker reads a load
payload, loads the weights, replies `{"ready": true, …}`, then answers one review per request line
until its stdin closes. Every reply is prefixed with a marker, and the worker redirects its own
`sys.stdout` to stderr before importing anything, because `transformers` writes progress to both
streams and a line the loader wrote must never be read as the model's answer. A dedicated thread
drains stderr into a bounded tail, so a chatty loader cannot fill its pipe and deadlock the worker,
and a failure message can say what the worker was last doing.

**The ref never reaches a load, unchanged.** `resolve_revision(model_id, ref)` still runs in the
parent before anything is spawned, and only the 40-hex SHA goes to the worker. The worker is keyed
on that SHA, which makes the pinning *stronger* than it was: the commit a worker loaded is fixed for
its lifetime, so a later review is served by the weights that SHA names or by a worker that was
restarted and re-resolved — never by a pointer that moved under a running one.
`revision_pinning_guard_test.py`'s allowlist entry says so.

### How long it lives

`redaction.llm_check.keep_worker_resident`, shipped `false`. The rounds of one check always share
one loaded model — that part needs no permission and cannot hurt anything, because the check is the
only thing running while it runs. Whether the *next recording* inherits the weights is the key, and
it ships off because a resident worker on an 80 GB card starves everything else on it; the
measurement is below, and the derivation is in `config-derivations.md` under `redaction.llm_check.*`.
`_llm_check` releases the worker in a `finally`, so a check that raises does not strand the memory.

### The three failure shapes, and why they differ

| what happened | what the worker does | what the caller gets | why |
| --- | --- | --- | --- |
| the load failed — no venv, no weights, no device | recorded once per process | `available=False`, the same `failure` on every later call, **no second load attempt** | a 23 GB load that failed will fail again, and at ~13,600 checked recordings (below) a `timeout_s` of 1800 s paid per recording is ~6,800 GPU-hours of waiting for the same answer |
| the review raised or the worker died | the worker is closed | `available=False`; the **next** call starts a fresh worker | an out-of-memory poisons the process but not the next transcript, and one reload is cheap |
| the review did not answer within `timeout_s` | the worker is killed | `available=False`, the failure names the timeout | unchanged from the shipped contract, except that it is now the generation that is bounded rather than the generation plus the load |

The sticky refusal is the only one of the three that is a *new* behaviour rather than a new
implementation of an old one, and it is the one worth arguing. The alternative — retry the load for
every recording — is what the shipped code did, and on a host with no reachable GPU it turns an
enabled step into a per-recording stall. `shutdown_review_worker()` clears it, so a deliberate retry
is available and a silent one is not.

**The absent path is unchanged where it matters.** REDACT's outcome is still its detector path's, the
release still stands, the annotation still carries `status: absent` with `revision: null` and the
failure string, and the `available: false` measurement is still in the store. What changed is only
how many times the graph pays to discover it.

### What it costs to hold — the thing that nearly made this not worth having

A worker kept alive keeps a GPU. How much of one turned out to be the most surprising measurement
here, and it changed the code.

Measured on an H100 80 GB, loading the QAT checkpoint directly and generating 64 tokens twice:

| | GiB |
| --- | --- |
| the checkpoint on disk | 21.67 |
| parameters resident after load (`int32` packed weights + `bfloat16` scales) | 19.04 |
| allocator **allocated** after load | 19.08 |
| allocator **reserved** after load | 23.07 |
| allocator reserved after one 64-token generation | **71.66** |
| …after `torch.cuda.empty_cache()` | **70.35** |
| …after a second generation, then emptied again | 70.35 |
| nvidia-smi, whole process, steady state | **72,735 MiB** |

Two things fall out of that, one of them good news and one of them the reason this section exists.

**The w4a16 packing does survive the load on the GPU.** `llm-check-first-run.md` measured the CPU
path decompressing to `bfloat16` after load — "the QAT variant does not buy 23 GB of host memory" —
and the obvious fear was that the same happened on device. At load it does not: the resident
parameter dtypes are 3.66 G `int32` and 2.90 G `bfloat16`, 19.04 GiB, close to the packed size on
disk, and the card reads 24.2 GB.

**But the first forward pass costs 51 GiB that never comes back.** After one generation the
allocator holds 71.66 GiB, and `empty_cache()` returns **1.31 GiB** of it — so the other 70.35 GiB
is *allocated*, not merely cached, and a second generation adds nothing further. The CPU behaviour
the first run measured is not absent on GPU; it is deferred to first use. A warm worker's true
steady-state footprint on this checkpoint is **~70 GiB, not ~23 GB**, and the arena saw exactly that
held flat across all twelve reviews.

That is a property of the checkpoint's runtime, not of the amortisation: the per-round path reached
the same peak and only avoided holding it by exiting. What the amortisation changes is that the peak
is now held continuously rather than for twenty seconds at a time.

The worker still empties the allocator after each generation. It is free, it does return the
transient 1.3 GiB, and it is what makes `resident_mib` mean "what this worker holds between reviews"
rather than "its high-water mark" — which is the number that decides whether anything else fits on
the card. Both readings reach the round's measurement in the store beside the timings, because after
this exercise it is plain that **memory, not time, is the binding constraint on this step**, and the
store should be able to answer it without an `nvidia-smi` outside the run.

**What this means for where the step should run.** On an 80 GB card a resident worker leaves about
8 GB for everything else, which is not enough for the rest of the triage graph. So the amortisation
argues for the step running as its **own pass over stored redacted transcripts** — which is already
the owner's plan — rather than co-resident with the graph inside the corpus driver. Running it in
the driver still works and is still ~5× cheaper than the shipped path, but only with a GPU it does
not have to share.

The obvious next thing to try, and deliberately not tried here because it changes numerics and needs
its own measurement, is loading with `run_compressed=True` so the packed weights are used in place
rather than materialised: that is the difference between a ~24 GB worker and a ~72 GB one.

`shutdown_review_worker()` remains the way to hand it all back. The worker also exits on its own
when its stdin reaches EOF, so a killed or preempted parent does not leave the weights on a shared
node.

## The step now records its own cost

The second item in `llm-check-first-run.md`: *"Every timing in this document had to be measured from
outside the graph."*

| where | what it carries |
| --- | --- |
| the `REDACT`/`llm_check` activity | `started` and `ended`, ISO 8601 with a zone, **on every path** — disabled and not-run included, so a stamp is never the thing that distinguishes them |
| each `redaction_llm_review` measurement | `elapsed_s` for that round, `load_s` — how much of it was the weights, `0.0` on an amortised round — `output_tokens`, and the two device-memory readings |
| `report.py`'s `_llm_reviews`, the summary JSON, the rendered block | the same, so a sizer reads them from the document rather than from the store |

**No timing reaches the annotation**, and that is deliberate rather than an oversight.
`corpus_report.aggregate` counts each `llm_redaction` field's *values* into a `Counter`: a float per
recording would be ~13,600 single-count buckets in the corpus report. The annotation stays controlled
vocabulary; the timings live one level down, on the rounds, where nothing counts them by value.

Between them the two answer both questions the store could not answer before — *how long did the
re-read take on this recording* (`ended - started`) and *how much of that was the model load*
(the rounds' `load_s`) — which is what makes the next person sizing this read it rather than
re-measure it.

## The `transformers` floor

The third item. `REVIEW_REQUIREMENTS` asked `transformers>=4.57`; the checkpoint's `config.json`
declares `transformers_version: 5.8.0.dev0`, `model_type: gemma4` and architecture
`Gemma4ForConditionalGeneration`. The first run worked because the resolver happened to pick 5.17.0.

Checked rather than assumed, two ways.

**Where `gemma4` enters the library**, by listing `src/transformers/models/` at each release tag:
absent at v5.0.0, v5.2.0, v5.3.0 and v5.4.0; present from **v5.5.0** (2026-04-02) onward. So
`>=4.57` admitted every version from 4.57.0 to 5.4.x, none of which can build this architecture, and
each of which installs cleanly and then fails at `from_pretrained` — which the spec correctly names
as the worse place to fail.

**Whether each version can actually read this checkpoint**, by installing it pinned and loading the
config and the model class on the meta device — no weights, no GPU — plus validating the
`quantization_config` against `compressed_tensors.quantization.QuantizationConfig`:

| pinned | `AutoConfig` | model class on meta | `quantization_config` |
| --- | --- | --- | --- |
| transformers 5.4.0 | **ValueError: model type `gemma4` … not recognize** | same | ok |
| transformers 5.5.0 | ok | ok | ok |
| transformers 5.7.0 | ok | ok | ok |
| transformers 5.8.0 | ok | ok | ok |
| transformers 5.17.0 | ok | ok | ok |
| compressed-tensors 0.12.0 | ok | ok | **ValidationError: 3 validation errors for QuantizationConfig** |
| compressed-tensors 0.13.0 | ok | ok | ok |
| compressed-tensors 0.15.0.1 | ok | ok | ok |

(A 0.14.0 cell is missing: that case failed during install rather than at the probe, and 0.13.0
already brackets the floor, so the cause was not chased.)

So the *measured* minima are transformers 5.5.0 and compressed-tensors 0.13.0, and the old
`transformers>=4.57` admitted an eight-release band — 4.57.0 through 5.4.x — that installs cleanly
and then fails at `from_pretrained`, which is the spec's complaint, now with a measurement under it.

The floor nonetheless ships **higher than the measured minimum**: `transformers>=5.8` and
`compressed-tensors>=0.15`, the versions the checkpoint's own `config.json` declares it was written
by (`transformers_version: 5.8.0.dev0`, `quantization_config.version: 0.15.1.a20260521`). That is a
deliberate choice and worth naming as one: 5.5.0–5.7.x pass every check that can be run without
loading 23 GB of weights, and are excluded on the checkpoint author's assertion rather than on an
observed failure. The asymmetry is what decides it — a floor set too high costs nothing, because the
resolver picks the newest satisfying version either way, while a floor set too low costs a
`from_pretrained` failure on a host nobody is watching.

Raising either floor changes the requirement set `ensure_venv` records in its `.senselab-installed`
marker, so **the first run after this change rebuilds `pii-redaction-review`** — about three minutes,
once per host. That is the mechanism working: the marker compares the sorted requirements, and a
changed floor is supposed to invalidate the tree.

## What was measured

Two arms on one H100 80 GB on `pi_satra`, against the redacted transcripts the first run's 12 checked
recordings produced — the same texts, the same node loop (`redact._llm_check`), the same config, the
same order. The only difference between the arms is whether the worker is shut down after each
review:

- **A — a load per round.** `review_redacted_text` is wrapped so `shutdown_review_worker()` runs
  after every call. That is the shipped cost model exactly: one spawn and one load per round.
- **B — amortised.** The worker is left up.

A warm-up review is run and discarded before either arm, so both see a fully warm page cache and the
snapshot already staged. Arm A does *not* reproduce the first run's ~62 s per round and it is worth
saying why before the numbers rather than after: it is three times faster, because the warm-up put
21.67 GiB of weights in the page cache first. Arm A is therefore the shipped path with its most
expensive component made as cheap as it can be, which makes every ratio below a lower bound.

Then, separately, the shape that matters at corpus scale: **the whole graph over many recordings in
one process**, the corpus driver's shape, with `llm_check` on and the per-recording step cost read
back out of each run's own store rather than from a stopwatch outside it.

### The step, arm against arm

12 recordings, one H100 80 GB, `node2803` on `pi_satra`, `max_iterations: 3`, every recording
settling in one round.

| | arm A — a load per round | arm B — amortised |
| --- | --- | --- |
| total for 12 recordings | 244.2 s | **60.4 s** |
| mean per recording | 20.3 s | 5.0 s |
| median per recording | 20.3 s | **3.6 s** |
| range | 18.0 – 22.3 s | 2.0 – 21.0 s (the 21.0 s is the one that loaded) |
| seconds spent loading | 167.0 s | 14.4 s |
| **load as a share of the arm** | **68.4%** | 23.8% |
| output tokens, median | 92 | 92 |
| verdicts | 12 clean | 12 clean |
| device held between reviews | 4 MiB | 70,858 MiB |

**4.04× over 12 recordings**, and **5.84×** comparing arm A's median against arm B's median *after*
the one recording that paid the load — which is the figure a slice of 156 recordings converges on,
since the load is paid once per process however long the slice is.

Two honest qualifications, both of which make this a conservative reading:

1. **Arm A is cheaper here than the shipped path was when the first run measured it.** That run
   reported ~62 s per round and ~75 s per checked recording; arm A measures 20.3 s, three times
   faster, on twelve recordings rather than one. The difference is the page cache: a warm-up load
   runs immediately before either arm, so every arm-A load reads 21.67 GiB that is already in RAM
   (13–15 s), where the same load cold takes 38.8 s — measured, as the first arena's warm-up. The
   first run's single 61.7 s figure was not preceded by a load. So the true saving on a cold host is
   larger than 4–6×, not smaller; this measurement simply refuses to claim the difference.
2. **Every transcript settled in one round.** The spec's 1.2-rounds-per-recording assumption is not
   exercised here, and each extra round is another full load in arm A and none in arm B — so more
   iteration widens the gap rather than narrowing it.

### The step, end to end, in the corpus driver's shape

The arena times the step in isolation. The corpus driver's shape — 26 recordings through the *whole*
graph in one process, `llm_check` on, on one H100 — is where the amortisation has to survive contact
with everything else the graph wants the card for. **It did not.**

With `keep_worker_resident: true`:

| | recordings |
| --- | --- |
| completed, PREPROCESS through REDACT | **3** |
| errored in PREPROCESS with `asr_qwen: RuntimeError: CUDA error: out of memory` | **19** |

The three that completed are recordings 1, 2 and 3; the check first ran on recording 3, and **every
recording after it failed**. Recording 3's own store carries the step exactly as designed —
`activity_span_s: 21.587`, one round, `load_s: 14.587`, 134 output tokens, the 40-hex revision — and
then the worker it left behind held 70.4 GiB and PREPROCESS's ASR could not allocate.

That is the measurement behind `redaction.llm_check.keep_worker_resident`, and it is why the key
ships `false`. It is also worth saying plainly what the default costs: with one round per recording,
releasing the worker after every check is arm A, 20.3 s per checked recording — the amortisation
then buys only the multi-round case, and on this corpus no recording had one. **The corpus-scale win
is real but it is not free: it requires giving the step a GPU of its own**, which is what the owner's
separate-pass plan already does.

The same list, the same process, the same card, rerun with the shipped
`keep_worker_resident: false` — the weights handed back when each check ends:

| | resident | released (shipped) |
| --- | --- | --- |
| recordings driven | 22 (job ended) | 26 |
| `PREPROCESS` completed | 3 | **26** |
| `asr_qwen: RuntimeError: CUDA error: out of memory` | **19** | **0** |
| reached REDACT | 3 | 18 |
| got a model verdict | **1** | **12** |
| load per checked recording | 14.6 s, once | 14.6 – 15.8 s, each |
| the step's own span (`ended - started`) | 21.6 s | 19.5 – 23.5 s, median 21.4 s |
| node errors of any kind | 19 | **0** |

The released arm is the whole sample working: 26 of 26 completed, 18 reached REDACT, 12 got a model
verdict — the same 12 of 26 the first run reported — and the whole thing took 3,434 s wall including
the graph. The resident arm produced one verdict and nineteen failures on the identical list.

The cost of releasing is exactly what arm A predicted: **69.9% of each checked recording's 21.4 s is
the model load**, paid again every time. That is the price of not having a GPU to spare, and it is
the thing a dedicated pass stops paying.

Two smaller things this arm settled. The store's records are complete and legible on every path —
`activity_span_s` 0.0 with `status: not_run` on the six recordings the detector path withheld, a
real span and a real `load_s` on the twelve that ran, and no activity at all on the eight where the
runner skipped REDACT, which is the gap `llm-check-first-run.md` already named and this change does
not close. And the timings are consistent enough to size from: twelve loads spanning 14.6–15.8 s and
twelve rounds spanning 18.3–22.3 s, on transcripts from 8 to 887 characters.

## The corpus, re-derived under the gate

`llm-check-first-run.md` estimated ≈23,800 recordings reaching REDACT and ≈15,900 getting a model
verdict, from a 20-recording detector-rate sample extrapolated family by family. That estimate
predates the PII-scan gate: `nodes/speech.py` step 7 now scans a recording only where at least one
lexical word is **not** in its declared stimulus, or where the family's expected pattern is a free
response. A recording that produced only the words its task asked for is never scanned, so it never
reaches REDACT.

The denominator is therefore a **read** rather than an extrapolation from a sample of 20: the
62,578-recording corpus run of 2026-09-19 writes one row per recording carrying its whole decision,
and `decision.ran.REDACT` says whether the recording reached the node. Read over the rows that
existed while this was written:

| | rows | share | scaled to 62,578 |
| --- | --- | --- | --- |
| recordings read | 5,531 | — | — |
| reached REDACT | 1,606 | **29.0%** | ≈18,200 |
| REDACT passed, so a model verdict | 1,204 | **21.8%** | **≈13,600** |
| (pass given reached) | | 75.0% | |

Against `llm-check-first-run.md`'s ≈23,800 reaching and ≈15,900 getting a verdict, the gate removes
about a quarter of the reaching population and about 14% of the verdicts. It is a smaller correction
than the gate's own headline numbers suggest, and the reason is visible family by family: the
families the gate was aimed at are not the ones that dominate. Grouped over the same rows:
diadochokinesis reaches REDACT on **398 of 469 (84.9%)** — ASR renders repeated syllables as words
no stimulus contains, so the gate opens almost every time — and free speech on **244 of 327
(74.6%)** by design, while respiration-and-cough reaches it on **8 of 777 (1.0%)** and glides on
**13 of 209 (6.2%)**. The gate closes hard on the wordless families and barely at all on the one
whose transcripts the first run found nothing to reason about.

**The caveat on this read, stated rather than buried.** The run is in flight, and its 400 slices
stride the manifest (`entries[idx::nsl]`), so the rows that exist early are spread across the corpus
rather than clustered — but they are the *faster* recordings within each slice, and shorter
recordings are less likely to produce a word outside their stimulus. The rate rose while the early
rows accumulated and has since settled: reach 27.8% at 2,108 rows, 28.8% at 2,598, 29.1% at 3,779,
**29.0% at 5,531**, with the verdict rate flat at 21.8% across the last three. That is a converged
number rather than a trend, but it is still a 9% sample of a run that has 14 hours left; re-read
`decision.ran.REDACT` over the completed run before spending anything on the strength of it.

### What the second pass would cost

The re-read as its own pass over only the recordings that carry a finding and clear the detector
path — ≈13,600 of 62,578 — at 1.0 rounds each, the rate every recording measured here and in the
first run settled at:

| | per recording | corpus | on 4 H100s |
| --- | --- | --- | --- |
| as shipped, as the first run measured it | ~75 s | ~284 GPU-hours | **~3.0 days** |
| as shipped, as arm A measures it here | 20.3 s | ~77 GPU-hours | **~19 hours** |
| **amortised (arm B)** | **3.6 s** | **~13.5 GPU-hours** | **~3.4 hours** |

So against the first run's own table the change is the difference between a three-day job and an
afternoon; against the more conservative arm-A baseline measured here it is nineteen hours against
three and a half. Both comparisons hold the population, the model, the config and the card fixed;
only the load policy differs.

Assumptions, all of them the first run's except where marked: one round per checked recording
(measured, 12 of 12 here and 12 of 12 there); generation cost tracks output tokens, which barely
move with transcript length (median 92 tokens across 8- to 887-character transcripts here);
perfect parallel efficiency across GPUs; no queue wait. **New assumption, and the one most worth
challenging:** that each of the four GPUs can be given wholly to this pass, since one worker holds
~70 GiB of an 80 GB card.

## What this does not change

- **`redaction.llm_check.enabled` still ships `false`.** Whether the step runs is the owner's, and a
  separate pass is the plan. Nothing here turns it on.
- **The annotate-don't-decide rule.** REDACT's outcome is its detector path's on every path; the
  re-read's summary is an annotation and VERDICT reads it under `verdict.llm_redaction_flags`.
- **The loop.** Review, mask, review again, bounded by `max_iterations`; whether *any* round flagged
  is what decides the check.
- **What the step is for, and whether it earns its place.** Items 4 and 5 of
  `llm-check-first-run.md` are design questions this change does not answer and does not try to: it
  makes the step affordable enough that the question can be asked over a corpus rather than over 12
  recordings.

## Tests

`src/tests/text/tasks/pii_redaction_review_test.py` replaces the worker *script* with a few lines of
pure Python that speak the shipped protocol and load nothing, and leaves the subprocess, the line
protocol, the threads and the lifetime real. That is what makes spawn counts, restarts, timeouts and
the refusal record observable on a laptop; a stub of `review_redacted_text` — which is what
`redact_test.py` uses, and rightly — sits above all of them and can see none.

The timings' own tests are in `redact_test.py`, next to the rest of the step's store contract,
including one that fails if a timing ever reaches the annotation.

### Mutations

Twenty-two, each applied to the shipped source, run against the named suite, and reverted.
**Twenty-one caught.**

| | mutation | caught by |
| --- | --- | --- |
| M1 | the worker is never reused — a fresh load per review, the shipped behaviour | worker suite |
| M2 | a refused load is retried for every recording | worker suite |
| M3 | a dead worker is reused rather than replaced | worker suite |
| M4 | the reply marker is ignored, so a line the loader wrote is read as a reply | worker suite |
| M5 | a review that never answers is waited on for ever | worker suite |
| M6 | `shutdown_review_worker` does not clear the recorded refusal | worker suite |
| M7 | `load_s` is not reported, so load and generation cannot be separated | worker suite |
| M8 | the worker exiting without answering is not noticed | worker suite |
| M9 | the `llm_check` activity carries no `started`/`ended` | redact suite |
| M10 | the stamp is naive, so two hosts cannot be compared | redact suite |
| M11 | per-round timings never reach the store | redact suite |
| M12 | the report reader drops the timings the store holds | redact suite |
| M13 | a timing leaks into the annotation `corpus_report` counts by value | redact suite |
| M14 | the allocator is never emptied | **not caught — see below** |
| M15 | the device footprint never reaches the caller | worker suite |
| M16 | the device footprint never reaches the store | redact suite |
| M17 | the worker is held across recordings regardless of the config | redact suite |
| M18 | the worker is always released, so a dedicated pass cannot amortise either | redact suite |
| M19 | the release is not on the failure path | redact suite |
| M20 | the packaged config asks for residency | redact suite |
| M21 | releasing between recordings also forgets a refused load | redact suite |
| M22 | the refusal is cleared however shutdown was asked for | worker suite |

**M14 is not catchable by this suite, by construction, and saying so is more useful than pretending
otherwise.** `torch.cuda.empty_cache()` lives inside the worker *script* — the string the fake
worker replaces wholesale — and its effect is a CUDA fact. No laptop test can observe it. What the
suite does pin is the contract around it (M15, M16: both readings reach the caller and the store).
The behaviour itself was measured on the GPU instead, and the measurement is why the claim in the
code comment was weakened: emptying returns 1.31 GiB of 71.66, because the rest is allocated rather
than cached.

Two are worth recording as lessons about the tests rather than about the code.

**M4 survived its first attempt**, because the only unmarked line the fake worker wrote was prose,
which the JSON parse rejects whether the marker is checked or not. A well-formed object that is
*not* a reply is what makes the marker load-bearing, and the test now writes one.

**M21 and M22 exist because adding the residency key introduced a defect that every test then
passing failed to see.** `shutdown_review_worker()` cleared the recorded start-up refusal — that is
what it is for — and the node now calls it after every check, so on a host with no reachable GPU the
refusal was forgotten once per recording and the 23 GB load was re-attempted 13,600 times: exactly
the stall the record was added to prevent. Neither half was wrong; the interaction was. It is now a
keyword (`forget_failure`), and the test that fails without it runs three checks in a row against a
worker that cannot start and asserts one load attempt, which is the only shape that can see it.

## Reproducing

```
/orcd/scratch/bcs/002/satra/llmamort_20260919/
  env.sh          ffmpeg and uv from the campaign's env_common.sh; the HF home holding gemma-4
  prep.sbatch     uv sync for the pinned checkout
  floor.sbatch    work/floor_probe.py — transformers and compressed-tensors, pinned, config only
  arena2.sbatch   work/arena.py — arm A against arm B over the first run's redacted transcripts
  memprobe.sbatch work/memprobe.py — what the checkpoint holds on an H100, load and after generate
  run3.sbatch     work/driver.py — 26 recordings through the whole graph in one process
  work/           floor_probe.json, arena2.json, memprobe.json, driver.json, driver3.json
```

Three pinned checkouts, one per measured commit, so nothing ran from a tree another job was using:
`senselab-llmamort` (the worker), `senselab-llmamort2` (the allocator reporting),
`senselab-llmamort3` (the residency key). The corpus denominator is a read of
`decision.ran.REDACT` over `/orcd/scratch/bcs/002/satra/triage_design_20260919/run/rows/`, which
belongs to a different run and was not written to.

The `driver*.json` files carry counts and timings only; the run directories under `run/` and
`run3/` carry transcript text and stay on scratch.
