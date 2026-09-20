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

### The three failure shapes, and why they differ

| what happened | what the worker does | what the caller gets | why |
| --- | --- | --- | --- |
| the load failed — no venv, no weights, no device | recorded once per process | `available=False`, the same `failure` on every later call, **no second load attempt** | a 23 GB load that failed will fail again, and at 12,800 checked recordings a `timeout_s` of 1800 s paid per recording is 6,400 GPU-hours of waiting for the same answer |
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

### What it costs to hold

The worker holds the weights for the process's lifetime — ~23 GB of device memory on the QAT
checkpoint — from the first recording that reaches the check to the end of the slice. On the 80 GB
cards this account can reach that sits alongside the rest of the graph without contention; on a
smaller card it is a real constraint, and `shutdown_review_worker()` is the way to hand it back
early. The worker also exits on its own when its stdin reaches EOF, so a killed or preempted parent
does not leave 23 GB resident on a shared node.

## The step now records its own cost

The second item in `llm-check-first-run.md`: *"Every timing in this document had to be measured from
outside the graph."*

| where | what it carries |
| --- | --- |
| the `REDACT`/`llm_check` activity | `started` and `ended`, ISO 8601 with a zone, **on every path** — disabled and not-run included, so a stamp is never the thing that distinguishes them |
| each `redaction_llm_review` measurement | `elapsed_s` for that round, `load_s` — how much of it was the weights, `0.0` on an amortised round — and `output_tokens` |
| `report.py`'s `_llm_reviews`, the summary JSON, the rendered block | the same three, so a sizer reads them from the document rather than from the store |

**No timing reaches the annotation**, and that is deliberate rather than an oversight.
`corpus_report.aggregate` counts each `llm_redaction` field's *values* into a `Counter`: a float per
recording would be ~12,800 single-count buckets in the corpus report. The annotation stays controlled
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
snapshot already staged; arm A's agreement with the first run's independently-measured ~62 s per
round is what establishes that the wrapper reproduces the shipped path rather than approximating it.

Then, separately, the shape that matters at corpus scale: **the whole graph over many recordings in
one process**, the corpus driver's shape, with `llm_check` on and the per-recording step cost read
back out of each run's own store rather than from a stopwatch outside it.

<!-- MEASUREMENT -->

## The corpus, re-derived under the gate

`llm-check-first-run.md` estimated ≈23,800 recordings reaching REDACT and ≈15,900 getting a model
verdict, from a 20-recording detector-rate sample extrapolated family by family. That estimate
predates the PII-scan gate: `nodes/speech.py` step 7 now scans a recording only where at least one
lexical word is **not** in its declared stimulus, or where the family's expected pattern is a free
response. A recording that produced only the words its task asked for is never scanned, so it never
reaches REDACT.

<!-- CORPUS -->

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

Thirteen mutations were applied and all thirteen were caught; they are listed in the measurement
section. One of them was not caught on the first attempt and is worth recording as a lesson about
the tests rather than about the code: the marker check on the worker's replies survived a mutation
that ignored the marker, because the only unmarked line the fake worker wrote was prose, which the
JSON parse rejects either way. A well-formed object that is *not* a reply is what makes the marker
load-bearing, and the test now writes one.
