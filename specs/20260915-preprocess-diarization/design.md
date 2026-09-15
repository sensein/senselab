# Diarization as a shared PREPROCESS derivative

Closes the derivative owed by
[`../20260913-branch-contract-and-hints/design.md`](../20260913-branch-contract-and-hints/design.md)'s
"Whole-file diarization as a shared derivative". The owner's ask, in their words:

> the preprocessing needs an extension similar to ppg that generates diarization on each enhanced
> file (using pyannote to see if it's 1 or more). pyannote is good at detecting single speaker. so a
> deviation could also be flagged as more than one speaker in this recording.

## Division of labour

**PREPROCESS measures. Branches refine their task spans. QUALITY judges multi-voice against those
refined spans, after every branch has run.** Speaker *identity* stays with SPEECH.

This change is the first clause only. There is no multi-voice gate, no verdict, and no threshold on
the count anywhere in it — the count is written, and what it *means* is decided by a node that can
see the refined spans. Every parameter is in
[`../20260817-triage-workflow-dag/config-derivations.md`](../20260817-triage-workflow-dag/config-derivations.md)'s
`diarization` section; nothing below repeats a derivation that lives there.

## What landed

| piece | where |
| --- | --- |
| the block, one per configured stream | `nodes/preprocess.py`: `diarization`, registered as `<stream>_diarization` |
| which streams | `diarization_streams` over `diarization.streams` — `[enhanced, residual]` |
| the model spec | `diarization_model` — a `PyannoteAudioModel`, commit-resolved at construction |
| the input | `diarization_input` — one stream, read **back out of the store**, mono at 16 kHz |
| the fold | `speaker_activity` and `diarized_segments` — pure, store-free, separately tested |
| the write | `write_diarization` — attributes plus one `derivatives/<stream>_diarization.npz` |
| the typed absence | `SpeakerDiarizationUnavailable(ValueError)`, in `extend.py`'s `UNAVAILABLE` |
| the extend path | `scripts/extend_diarization.py` |

It calls `senselab.audio.tasks.speaker_diarization.diarize_audios`. It does **not** call pyannote
directly, and that is load-bearing for more than tidiness: the backend resolves the ref to a commit
and passes `revision=<sha>` to `Pipeline.from_pretrained`, which is CLAUDE.md's two-call rule.
Reimplementing the load here would have reimplemented that, badly.

## Both halves of the partition

Enhancement does not destroy a quiet background talker; it **partitions** the recording into
`enhanced` and `residual = plain - g·enhanced`. Both are diarized:

- **`enhanced`** — how many voices *survived* enhancement. One, for a single-participant protocol.
- **`residual`** — whether a voice was *removed*. This is the evidence that a background talker was
  there at all.

Neither alone answers the owner's question. **The two counts are never summed and never collapsed
into one number**, in the measurement or in the slice log: a store carrying only a total could not
say whether enhancement removed a voice.

One measurement per stream, `<stream>_diarization`, rather than one measurement carrying both. Three
reasons, in order of weight:

1. `signal` is a single-stream field everywhere else in the store. A multi-stream measurement would
   have to abuse it, and `derived_from` would name two streams for a number that came from one.
2. It is the shape the repository already uses for a per-stream whole-file model product:
   `enhanced_yamnet_scores` / `residual_yamnet_scores`, not one `yamnet_scores` with a nested map.
3. It makes each stream its own block, so a run whose residual was never written still gets its
   enhanced reading and records the other as its own absence — and an extend pass that widens
   `diarization.streams` measures only what is missing.

## Attributes versus sidecar

Per stream, `n_speakers` is the headline. It sits on the entity with `speakers`, `n_segments`,
`per_speaker_s`, `speech_s`, `overlap_s` and `max_concurrent_speakers`, because a consumer answering
"one voice or more" must not have to open a file to do it — the same reason `praat_features` puts
forty-five scalars on the entity and writes no sidecar at all.

The segment table goes to `derivatives/<stream>_diarization.npz`, named by the entity through its
path and SHA-256. The rule it follows is the posteriorgram's: **what grows with the recording's
length is written beside the run, never inlined.** `starts`, `ends`, `speakers` and `streams` are
four parallel columns of one table, so concatenating the two files gives a complete, self-describing
segment list — intersecting it with a span is arithmetic, with nothing to look up.

They are deliberately **not** `span` entities, which is the other idiomatic home for timed regions
(`clip_spans` and the `spans` block both write them). `live_entities(store, "span")` is the
branches' set of *candidate task spans*. A per-speaker time partition is not a candidate for
anything, and putting it there would silently widen an existing consumer's input set as a side
effect of adding a measurement.

## The overlapping view, not the partition

`diarization.exclusive` is `false`. This is the only setting in the section that is a correctness
choice rather than a preference, and `speaker_diarization/pyannote.py` already states why:

> With the exclusive view, *no downstream consumer can detect overlap at all* — a per-instant
> speaker count derived from these segments is capped at 1 by construction, so it reports "no
> overlap" as a confident measurement rather than as something the input could not express.

Two people talking at once is the evidence a multi-voice reading most wants, so the block takes the
view that can express it. The speaker set is identical under either view; only `overlap_s` and
`max_concurrent_speakers` differ, and under the partition both are structural zeros. **It has two
consequences for SPEECH's reuse**, recorded below.

## Zero speakers is a value

Measured on real b2ai recordings (below): **every** `Respiration-and-cough` recording diarizes to
zero speakers and zero segments. That is the correct answer — a breath or a cough is not a voice —
and it is recorded as `n_speakers: 0` with the measurement present, not as an absence.

**This is also how an empty residual reads**, which is the ordinary case. "The residual holds no
voice" is a positive finding (enhancement took nothing out) and is strictly more informative than an
absence; an *absence* on the residual means the stream could not be read or the model could not be
obtained, which is a different fact and must stay distinguishable from it. The absent-versus-zero
distinction is one this repository has paid for before.

## What SPEECH must read

The owner has directed that *"speech should not have to rerun pyannote to do this, just take the
output and use it"*, so `speech.py`'s own pass goes away and this derivative becomes the **only**
diarization in the graph — no fallback, no second opinion. **That change is not in this commit.**
What follows is what the read-swap needs, read off `speech.py` rather than assumed, so whoever makes
it is doing a mechanical swap.

### What SPEECH does today

`speech.py:641-681`. It takes `interval = clamp_extent((min lexical word start, max lexical word
end), plain)`, crops `plain` to it, and calls `diarize_audios([cropped], model=diarizer)` with
`pyannote/speaker-diarization-community-1` — the same checkpoint, at the `diarize_audios` default
of `exclusive=True`. From the result it builds exactly one structure:

```python
speaker_segments: list[tuple[str, str, tuple[float, float]]]   # (entity_id, speaker, absolute extent)
```

offsetting each segment by `interval[0]` back onto `plain`'s timebase. Everything downstream reads
only that list and its `count = len({speaker})`:

| use | site | what it needs |
| --- | --- | --- |
| a `speaker` entity per segment, into the view | `:672-679` | label + absolute extent |
| the speaker count | `:681` | distinct labels |
| `flags.append("speaker count N != 1")` | `:687` | the count |
| the `second_diarizer` consultation | `:684-702` | the count |
| the separation gate (`not_needed` / `count_N_exceeds_backend`) | `:705-717` | the count |
| word → speaker attribution by overlap | `:777-798` | label + extent, per segment |
| per-speaker audio for the enrollment embedding | `:826-838` | label + extent, to slice `plain` |
| `span.attributed_to` and `span.nontarget` | `:876-890` | the per-word attribution above |

**Every one of those is label + absolute extent + the derived count.** The derivative carries all
three: `speakers`, `n_speakers`, and the `(start, end, speaker, stream)` sidecar table in absolute
seconds on the stream's own timebase. **There is no field SPEECH reads that the derivative cannot
supply.** No blocker.

### What the migration gains

Verified against the code, not assumed: SPEECH crops to the lexical word hull before diarizing, so
it **structurally cannot** see a speaker outside `[first word start, last word end]` — which is
where an interrupting or background voice sits. The derivative is whole-file. Coverage is strictly
larger, and the extra region is exactly the one SPEECH was blind to.

### Three consequences the migration must handle

1. **The stream changes: `plain` → `enhanced`.** SPEECH diarizes `plain`; nothing in
   `diarization.streams` is `plain`. Two sub-points, and the second is a real trap:
   - SPEECH slices `plain` itself for the enrollment embedding, so the *embedding* is unaffected —
     only the *labels* would come from a different signal. Either accept that (enhancement is meant
     to help the diarizer, and the residual half catches what it removes), or add `plain` to
     `diarization.streams`: one config line, no code, one more model pass per recording. **A
     decision for the owner, not a defect.**
   - **The timebases coincide only when the residual block's lag is non-negative.**
     `residual.py::align` trims the *reference*'s head when `lag < 0` (`reference[-lag:]`), which
     shifts `enhanced`'s and `residual`'s t=0 forward relative to `plain` by `-lag` samples. Every
     lag observed so far is 0 — including on the store measured here, whose `residual` measurement
     records `lag_samples: 0` and whose six streams all carry the extent `(0.0, 88.0733125)` — so
     in practice extents transfer unchanged. But a consumer must read `lag_samples` off the
     `residual` measurement rather than assume it, and the migration should do so explicitly.
2. **`exclusive: false` makes words straddle.** SPEECH's `_overlaps` marks a word that overlaps more
   than one segment as `straddles` and attributes it to nobody (`:783-784`). Under its own
   `exclusive=True` partition that could never happen for two *different* speakers at one instant,
   because the partition resolves concurrency away. Under the overlapping view it can, and will,
   wherever there is genuine overlap. That is arguably the honest answer — a word spoken over
   another voice does have ambiguous attribution — but it is a **behaviour change**, and the
   migration should decide it deliberately rather than discover it.
3. **`exclusive: false` can contaminate the enrollment embedding.** `:826-838` concatenates `plain`
   slices per label; under the overlapping view a slice can contain two voices, which the partition
   could not produce. If that matters, SPEECH should drop segments that overlap another speaker
   before slicing — a filter over the same table, not a different measurement.

`speech.second_diarizer` (`default.yaml`, null → `not_consulted`) becomes a question about the
shared derivative rather than about SPEECH. Noted, not acted on.

## Open design question: attribution onto spans

The owner raised, as an alternative or complement: *"or diarization can attribute speaker to
spans."* **Not decided here, and nothing in this change writes a span attribution.**

The tension is a contract question, not a detail. `speech.py:790-793` already writes an `assertion`
with `verb: "attribute"`, and `"attribute"` is **not** among the branch contract's verbs — `propose`
plus the four annotating ones (`label`, `contest`, `refine`, `trim`), as
[`../20260817-triage-workflow-dag/branch-conventions.md`](../20260817-triage-workflow-dag/branch-conventions.md)
§ *`propose` versus `refine`* scopes them. (`deviate` is a sixth, forward-declared there and not yet
read by REPORT.) So attribution-to-spans has a precedent in the code and no standing in the
contract. Three things need settling together:

1. **Who may attribute?** If PREPROCESS does it, a measurement node is asserting something about
   another node's spans — which is what "PREPROCESS measures, branches refine, QUALITY judges" was
   written to prevent. It would also be attributing to spans that do not exist yet: PREPROCESS runs
   before the branches that refine them.
2. **Under which verb?** Either `attribute` joins the contract's five, or the existing use in
   `speech.py` is a violation to retire.
3. **Against which spans?** A span refined by a branch is not the span PREPROCESS could have seen.

**Recommendation: keep segments as the primary product and let the attributing node attribute.**
The derivative emits `(start, end, speaker, stream)` in absolute seconds, so any node that owns a
span can intersect it in a few lines — which is precisely what `speech.py:780` already does today
with its own segments, and what QUALITY will do against branch-refined spans. That keeps PREPROCESS
measuring, leaves the verb question to the contract, and loses nothing: an attribution is derivable
from the segments, but the segments are not derivable from an attribution.

## Measurements, 2026-09-15

Host: Apple silicon laptop, CPU only, **under load from a concurrent pytest run in another
worktree** — so every timing here is an upper bound, and the standing note about resource
measurements needing isolation applies in full. Model:
`pyannote/speaker-diarization-community-1` at commit `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee`.

### Speaker counts on real recordings

48 recordings from `~/Downloads/b2ai_v31_bids_07_01_v3`, three subjects, every `Story-recall`,
`Harvard-Sentences` and `Respiration-and-cough` file they carry, conditioned exactly as the block
conditions them (these are raw session audio, so this is a single-stream reading, not the
`enhanced`/`residual` pair):

| task family | files | `n_speakers` | segments |
| --- | ---: | --- | --- |
| `Story-recall` | 3 | **1** on all three | 11, 3, 1 |
| `Harvard-Sentences` | 20 | **1** on all twenty | 1 on seventeen, 2 on two, 3 on one |
| `Respiration-and-cough` | 25 | **0** on all twenty-five | 0 |

**Nothing read above one speaker in those 48.** These are single-participant protocol recordings, so
that is the expected result: no intruding voice, no interviewer, no false split. Twenty-five
zero-speaker files are breath and cough tasks with no speech in them.

Separate the multi-*segment* single-speaker readings from multi-*speaker* ones: a Harvard sentence
split into three segments is one voice with pauses, and `n_speakers` says so. A consumer that
counted segments would have read three of those twenty as multi-voice.

### The one complete store, through the extend driver

`~/Downloads/triage_prov_sample/` (copied), `sub-004d42e9…_task-Story-recall`, 88.07 s, run end to
end through `scripts/extend_diarization.py` with the real model on both streams:

| stream | `n_speakers` | segments | `speech_s` | `overlap_s` | model time |
| --- | ---: | ---: | ---: | ---: | ---: |
| `enhanced` | **2** | 27 | 66.37 | 9.53 | 63.3 s |
| `residual` | **1** | 10 | 48.16 | 0.00 | 63.8 s |
| `plain` (measured separately, for comparison, not written) | **2** | 15 | 64.78 | 3.54 | ~70 s |

**`enhanced`'s second speaker is a false split, not a second person.** `SPEAKER_00` holds 66.37 s
across ten segments; `SPEAKER_01` holds 9.53 s across seventeen, none longer than 1.35 s, one of
them 0.02 s, and **every one of them nested wholly inside a `SPEAKER_00` segment**. An interviewer
or an intruding voice produces turns *between* the participant's, not seventeen sub-second fragments
interleaved inside them. The likely mechanism is the clustering splitting one speaker's own
variation — a register change, or a passage at a different level — into a second cluster. It is
exactly the failure mode the uncited single-versus-multi claim needs testing against, and it
appeared on the first real store tried.

**`residual`'s one speaker is leaked target speech, not a background talker.** Its ten segments run
2.41–15.44 s, 17.40–21.68 s, 30.54–36.03 s, 57.47–65.59 s and so on — every one of them *inside* a
`SPEAKER_00` segment of the enhanced stream, none of them in a gap. The store's own `residual`
measurement says why: `enhanced_energy_fraction` is **0.109** and `energy_fraction` is **0.825**, so
FRCRN kept 11% of the energy and 82% went into the residual. That is the near-nulling mode
`20260817-triage-workflow-dag/config-derivations.md`'s `residual` section already documents. The
residual half is reporting imperfect separation on this recording, not a second person. A genuine
background talker would show up where the target is silent.

Both streams' extents are `(0.0, 88.0733125)`, the same as `plain`'s, and the residual
measurement's `lag_samples` is **0** — so on this recording the timebases coincide exactly. See the
caveat in "Three consequences" below: that is a property of this recording's lag, not a guarantee.

**Both readings are why `n_speakers` alone must not become a gate.** A QUALITY rule reading
"`enhanced` ≥ 2 means a second person" would have flagged this recording; one reading "`residual`
≥ 1 means a voice was removed" would have flagged it too. What distinguishes a real second voice
from either artefact is the **segment geometry** — per-speaker totals, segment counts, and whether
one speaker's segments nest inside another's or fill its gaps — which is in the sidecar precisely so
whoever writes that rule can use it. The 48-file probe says nothing about the false-positive rate;
this one store says it is not zero on either stream.

### Cost

**0.217 s of CPU per second of audio**, over 586.7 s of audio in 127.4 s, on the 48-file probe. The
ratio is strongly bimodal and duration-dependent, which matters for slicing:

| population | files | ratio (s CPU / s audio) |
| --- | ---: | --- |
| no speech found (`Respiration-and-cough`) | 25 | 0.009 – 0.083 |
| short speech (3–6 s, `Harvard-Sentences`) | 20 | 0.054 – 0.099 |
| long speech (25–80 s, `Story-recall`) | 3 | 0.60 – 0.76 |

A recording the segmentation rejects costs almost nothing; the embedding and clustering stages are
the whole cost and only run where there is speech. **The budget scales with *speech* duration, not
audio duration** — the property the PPG pass's own sizing note warns about, one step further along.

### Short input

Head-truncated clips at 0.05, 0.1, 0.25, 0.5, 1.0 and 2.0 s all returned without raising. There is
no structural length below which the model refuses, so **no `min_duration_s` key ships** — a floor
would turn a measurable recording into an absence and could not be read off any measurement. The
entity carries `duration_s` for a consumer that wants to discount a short reading.

## The tests discriminate — mutation table

Twenty-two single-edit mutations of the implementation, each run against
`preprocess_test.py`, `extend_diarization_test.py` and `config_test.py`. **Every one is caught.**
The table is the evidence that no test here passes either way.

| # | mutation | tests that fail |
| --- | --- | ---: |
| M1 | `n_speakers` counts segments, not distinct speakers | 1 |
| M2 | per-speaker totals sum a speaker's overlapping turns twice | 1 |
| M3 | overlap counts any active instant, not concurrent ones | 5 |
| M4 | `max_concurrent_speakers` pinned at 1 (the exclusive-partition bug) | 2 |
| M5 | a line naming no region is kept as a speaker | 1 |
| M6 | segments inlined as attributes instead of a sidecar | 2 |
| M7 | the diarizer's failure escapes as itself, not a typed absence | 1 |
| M8 | an unresolvable model spec escapes as itself | 2 |
| M9 | the typed absence is not a `ValueError`, so it reads as a node failure | 2 |
| M10 | the stream is hardcoded to `enhanced` instead of read from config | 7 |
| M11 | the speaker bounds are never handed to the diarizer | 1 |
| M12 | the packaged config takes the exclusive partition | 2 |
| M13 | the driver never skips a store that already holds a reading | 3 |
| M14 | the driver re-derives but never retires what it replaced | 1 |
| M15 | the driver drops the per-recording speaker count from its log | 4 |
| M16 | the sidecar drops the per-row `streams` column | 1 |
| M17 | every stream writes to one shared sidecar path | 6 |
| M18 | the measurement name ignores the stream | 13 |
| M19 | only the first configured stream is measured | 5 |
| M20 | the packaged `streams` list drops `residual` | 10 |
| M21 | the driver sums the two streams' counts into one | 4 |
| M22 | an unreadable `diarization.streams` registers nothing instead of an absence | 1 |

The first run of this table found a real gap: M1 passed, because no test then combined "one speaker"
with "more than one segment". `test_one_speakers_own_repeated_turns_are_one_voice_counted_once` was
strengthened to assert both, and M1 fails now. That is the one change mutation testing forced.

## Sizing for the corpus pass

Corpus: **62,547 recordings, median 7.0 s, mean 15.0 s, 250.2 h total** (900,720 s).

**Per stream**, so the figure halves if `diarization.streams` is shortened to one entry:

| term | per stream | both streams |
| --- | ---: | ---: |
| model, at 0.217 s/s over 900,720 s | 195,456 s ≈ **54 CPU-h** | ≈ **108 CPU-h** |
| per-recording I/O (`read_store` 0.19 s + `write_store` 0.12 s + `export_prov` 0.17 s, from the PPG pass's measurement — paid once per recording, not once per stream) | — | ≈ 0.5 s × 62,547 ≈ **8.7 CPU-h** |
| total | | **≈ 117 CPU-hours** |

**Do not assume the residual half is cheaper.** The intuition is that most residuals hold no speech
and the segmentation rejects them at the 0.01–0.02 ratio above — but on the one real store measured
it was not: 63.8 s for the residual against 63.3 s for the enhanced stream, because FRCRN left 48 s
of the participant's voice in it. Budget the two halves equally until a slice says otherwise. One
store is not the corpus, and this is the first number the pilot slice should be read for.

**No GPU needed.** At ≈117 CPU-hours a 64-way array is under two hours per task. A GPU would cut the
model term, but pyannote's GPU throughput was **not measured here** — this host has none — so no
speedup figure is quoted and the sizing does not depend on one. Ask for CPU; it queues sooner.

Two cautions carried from the PPG pass, both sharper here:

- `rows[i::n]` equalises **row counts**, not audio, and here not even audio — it equalises neither
  speech nor cost. With the bimodal ratio above, a slice that draws the long `Story-recall` tail
  costs many times one that draws breath recordings. The `--time` below has room for the unlucky
  slice, not the mean one.
- The 0.217 figure is a **loaded-laptop upper bound** on a sample whose task mix is not the
  corpus's. Run one slice first and read its `elapsed_s` before trusting the array's shape.

### The sbatch I would submit (not submitted)

```bash
#!/bin/bash
#SBATCH --job-name=triage-diarization
#SBATCH --partition=mit_normal
#SBATCH --array=0-63
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/triage-diarization-%A_%a.out
#SBATCH --requeue

set -euo pipefail

# The four omissions that have killed arrays here before: PATH, the ffmpeg shared libraries
# torchcodec dlopens by soname, HF_HOME/HF_TOKEN (pyannote's checkpoints are gated), and a
# stable run id so every task pins the same model commit.
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$(readlink -f ~/orcd/scratch)/miniforge/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$(readlink -f ~/orcd/scratch)/hf-cache-$USER"
mkdir -p "$HF_HOME"
export HF_TOKEN="$(cat ~/.cache/huggingface/token)"
export SENSELAB_RUN_ID="triage-diarization-20260915"

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPO_DIR"
[ -d src/senselab ] || { echo "not the senselab checkout: $REPO_DIR" >&2; exit 1; }
mkdir -p logs

MANIFEST="${DIARIZATION_MANIFEST:?set DIARIZATION_MANIFEST to the ppg pass's manifest JSONL}"

uv sync --all-extras
uv run --no-sync python scripts/extend_diarization.py "$MANIFEST" \
    --slice-index "$SLURM_ARRAY_TASK_ID" \
    --slice-count 64 \
    --device cpu
```

**Warm the Hub cache once before submitting**, on a login node or one interactive task, or 64 tasks
race the same gated download:

```bash
uv run python -c "
from senselab.utils.data_structures import DeviceType, PyannoteAudioModel
from senselab.audio.tasks.speaker_diarization.pyannote import PyannoteDiarization
m = PyannoteAudioModel(path_or_uri='pyannote/speaker-diarization-community-1', revision='main')
PyannoteDiarization._get_pyannote_diarization_pipeline(model=m, device=DeviceType.CPU)
print(m.commit_sha)
"
```

Unlike the PPG driver there is **no venv gate**, because there is no subprocess venv: pyannote runs
in the main environment. The pre-warm is a Hub-download courtesy, not a correctness requirement — a
task that has to download will succeed, just slowly, and 64 at once will rate-limit.

Run one slice first (`--slice-count 64 --slice-index 0`), read its `speaker_counts` histogram (per
stream) and `elapsed_s` out of `slices/slice-0-of-64.summary.json`, and size the rest from that. The
histogram is also the first corpus-scale evidence on the false-split rate the one store above
suggests is non-zero.
