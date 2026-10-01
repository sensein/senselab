# Nimble as a second opinion beside the reviewer

Owner-approved 2026-10-01. The redaction reviewer (Gemma, REVIEW, prompt v6) is one model reading one
transcript. A second model that answers some of the same questions typed, with a probability, gives
two things. First, a way to route a recording to a human where the two disagree with confidence.
Second, a calibrated number per question that a later fit can use. It decides nothing on release.

## The model

Bespoke Labs' Nimble 9B. It is a decision model: given a `state` (named text fields) and typed
`questions`, it returns, per question, a choice with class probabilities (`choice`) or a single
probability of yes (`noul`). It is served by Ollama 0.35's `POST /v1/systemone`.

| | |
|---|---|
| name:tag | `nimble:latest`, the tag the pull filed it under (locates the manifest only) |
| weights blob | `sha256:bbf1d6fc03bb0ed24d88f4c214ed7b5d1768aeb43d5cf433fb69eff0c8578013` (9.5 GB) |
| image config | `sha256:2c26ca580a1a2f29728af87c30258cfc12c0fe010ad4dc8fc5db5afcdfb406c3` |
| manifest file | `sha256:24e550a16a7081881be2f1f0d91e8cc13a597472735c04119f035a0a85c67e0c` |
| system / params layers | a classify-only system prompt; `{"num_ctx":8194}` |
| server | Ollama 0.35.0, run inside the job, loopback only, `OLLAMA_NOPRUNE=1`, never pulls |
| sampling | `options: {temperature: 0, seed: second_opinion.seed}`: accepted and measured inert (below) |

## Pinning

An Ollama tag is mutable, so the pin is three digests, the way an HF load is pinned by commit.
`verify_pin` (`senselab.text.tasks.decision_model.ollama`) reads the local manifest and refuses
the store, raising `PinMismatchError`, if any of these fails:

- the manifest file does not hash to `manifest_digest`. The manifest names the system-prompt and
  parameters layers, and those change the answers under the same weights;
- the config digest is not the pinned one;
- a small layer (system, params, license) does not hash to its digest;
- there is not exactly one model layer, or its digest is not the pinned blob;
- the blob's size differs from the manifest's;
- the blob does not hash to the pinned digest.

The full hash costs about 30 s on 9.5 GB. It runs once per (digest, size, mtime), and a marker file
in `--verified-dir` records the result. A replaced file therefore hashes again. The driver verifies
before the array task reads a row, so a mismatched store fails the task with exit 2 instead of
writing `absent` into every store.

The ProvStore model agent needs a 40-hex git commit, and a blob digest is not one. So the agent
records `model_id=ollama:nimble:latest@sha256:…`, `version=<blob digest>` and an `unresolved_reason`
saying the pin is a blob digest. Passing the sha256 off as a commit would make the provenance look
right while being wrong.

### Durable location (proposed, not yet executed)

The pilot store lives in scratch: `/orcd/scratch/bcs/002/satra/tmp_nimble/{ol,models}`, 2.1 GB of
binary and 8.9 GB of models. The group model store `/orcd/data/satra/002/models` is already where
other weights live, but its `blobs/` directory is not group-writable. The proposal is a
self-contained subtree:

    /orcd/data/satra/002/models/ollama-senselab/
        bin/ollama          # 0.35.0, with its lib/
        models/             # OLLAMA_MODELS: manifests/registry.ollama.ai/library/nimble/latest, blobs/
        verified/           # verify_pin markers

The copy commands are in the r9 RUN.md section "Nimble second opinion". `verify_pin` against the
copy is the check that it arrived whole.

### Determinism, measured

On 2026-10-01 a short H100 srun against the scratch store asked one transcript six ways: no
`options`, `{temperature: 0, seed: 7}` twice, seed 99, `{temperature: 1.0, seed: 3}`, and an unknown
option key. All six returned HTTP 200 and byte-identical answers, for example other_voice
P(more_than_one) = 0.026351785774690627 and named_diagnosis 0.9962437042755286. `usage` reports 3
output tokens: the probabilities are read off the logits, not sampled. So `options` is accepted and
has no effect. Determinism comes from the pinned weights, prompt layers and input. Cold load plus
the first answer took 8.7 s; warm answers took 0.9–1.9 s.

### End to end, measured

On 2026-10-01 `extend_second_opinion.py` at dcedb091 ran on an H100 (node3209) over scratch copies
of five r9 run roots: story-recall, free-speech-1, free-speech-2, word-color-stroop and
productive-vocabulary-1. The model was served from the pilot store.

- **First pass.** All five came back `ok`, and each store was re-folded. The first row took 20.7 s
  (pin hash plus server start plus load); the rest took 4.7–8.3 s each, re-fold and REPORT included.
- **Second pass.** All five were `present`. `server_started` was false; the slice took 1.4 s.
- **Story-recall.** other_voice was 0.91 against the reviewer's `one`, so it flagged under the new
  ground. Its instructions_spoken was 0.83, not compared, because that store's annotation is
  prompt v4.
- **The other four.** No disagreement. Free-speech named_diagnosis read 0.77 and 0.98, beside
  reviewer readings that were `flagged`.

## The questions (QUESTION_SET_VERSION 1)

Four questions, over the same original transcript (`transcript_texts`) and the same task context
(`review.task_context`) that the reviewer reads. The context supplies `task`, `speech_type`,
`instructions` and `stimulus` (the `asked_to_say` text). The wording is in
`decision_model/second_opinion.py:QUESTIONS`.

| question | type | read as | compared with the reviewer |
|---|---|---|---|
| `other_voice` | choice: one / more_than_one / unclear | P(more_than_one) | yes: `speakers == more_than_one`, or `unclear` with quoted other speakers |
| `instructions_spoken` | noul | P(yes) | yes: a non-empty `instructions_spoken`, only where the reading carries the part (prompt v5 and later) |
| `named_diagnosis` | noul | P(yes) | yes: a `CONDITION` entry with action `redact` in the proposal |
| `safe_harbor_identifier_present` | noul | P(yes) | **no** — recorded only |

Safe Harbor is not compared. The reviewer's identifier reading is per span, with a proposal and
categories. The model's is one number per transcript. On the pilot it also disagreed with the
reviewer where the reviewer was right: 0.19 on a recording carrying a named venue. A fitted rule
would need span-level labels first.

The pilot asked under different question names (`diagnosis`, `safe_harbor`), with a state of only
task, instructions and transcript. The node asks the wording above. The pilot's numbers motivate the
thresholds; they do not calibrate them.

## The measurement

SECOND_OPINION writes one `nimble_opinion` measurement per call, on every path:

| attribute | |
|---|---|
| `status` | `ok` / `disabled` / `nothing_to_read` / `absent` (asked, no answer: `failure` says why) |
| `probabilities` | question → P(yes); `other_voice` is P(more_than_one) |
| `other_voice_choice` | the class the model chose |
| `question_set_version`, `model_id`, `blob_digest`, `config_digest`, `manifest_digest`, `seed` | identity |
| `transcript_chars` | the length of the text asked about (see the context window, below) |
| `context_keys` | which task-context keys were present |
| `result_cache` | `{key, hit, stored}` |

The activity is `SECOND_OPINION/decide`. It uses `consensus_transcript` and is associated with the
software and model agents.

## The cache key

`result_cache_key` has these parts:

- `input_signature`: the hash of the original transcript plus `\x1f` plus the context's sorted JSON;
- `process`: `nimble_opinion`, at `RESULT_PROCESS_VERSIONS` 1;
- `model_id`: `ollama:nimble:latest`;
- `commit_sha`: the blob digest;
- `params`: `{question_set_version, seed, config: config_digest, manifest: manifest_digest}`.

Changing the wording bumps `QUESTION_SET_VERSION`, and a new release changes the digests, so
neither can serve a stale answer. A malformed answer (a missing question, or a choice without class
probabilities) raises before the store, so it is never cached.

## The fold

`verdict.nimble_disagreement_flags` (owner: true) adds a flag ground:

    the second-opinion model confidently disagrees with the redaction reviewer: other_voice p=0.85 reviewer=no

A question disagrees when `p >= verdict.nimble_confident_yes` and the reviewer said no, or when
`p <= verdict.nimble_confident_no` and the reviewer said yes. The comparison runs only when the
opinion is `ok` and the annotation is `clean` or `flagged`. If either threshold is null, nothing
is compared. The release axis is unchanged. The fold record carries `second_opinion`:
`{status, probabilities, disagreements, model_id, blob_digest}`. That is where the parquet columns
come from.

## The thresholds: UNFITTED, 0.8 / 0.2

The 20-recording pilot (2026-09-30, `tmp_nimble/pilot_out.json`, each recording asked twice, both
answers identical 20/20) gave these probabilities on the three compared questions:

| family | other_voice | instructions_spoken | diagnosis | reviewer (pre-v6) |
|---|---|---|---|---|
| story-recall | **0.85** | 0.93 | 0.02 | one, no diagnosis |
| cinderella-story ×3 | 0.02 | 0.02 | 0.01 | one |
| free-speech | 0.01 | 0.01 | **0.10** | one, diagnosis |
| free-speech ×4 | 0.01–0.02 | 0.01–0.05 | 0.99–1.00 | one, diagnosis |
| free-speech-v2 ×7 | 0.01–0.03 | 0.01–0.03 | 0.00–0.04 | one, no diagnosis |
| free-speech | 0.01 | 0.02 | 0.01 | one |
| prolonged-vowel | 0.36 | 0.03 | 0.02 | more_than_one |
| word-color-stroop | **0.05** | 0.03 | 0.04 | more_than_one |
| productive-vocabulary | 0.75 | 0.34 | 0.01 | more_than_one |

Of the 60 answers, 50 are ≤ 0.10 and six are ≥ 0.84. The other four are 0.103, 0.34, 0.36 and
0.75. Cuts at 0.8 and 0.2 fall in the gaps. Under them three recordings flag (bold):

- the story-recall examiner, which the v6 reviewer is now also expected to catch;
- the Stroop recording, where the model is probably wrong, since the examiner's prompts sit in
  that transcript;
- a free-speech diagnosis at 0.103, where the reviewer is right.

That is 15% of a pilot chosen to include hard cases, not a population rate. Story-recall reads
instructions_spoken = 0.93, but the pre-v5 reviewer never answered the question, so it is not
compared.

To fit: run r9, then put a human verdict on every recording that disagrees at the looser 0.6/0.4,
by question. Each cut goes where "model right" stops outnumbering "reviewer right". If the model is
never right on a question, drop that question from `SECOND_OPINION_QUESTIONS`.

## Cost

Warm requests took 0.71–1.39 s (median about 1.0 s) on an H100. The first request loads the
weights: 74.5 s in the pilot, 8.7 s on the 2026-10-01 test. Over the ~15,165-recording review manifest that is about 4.2 GPU-hours, plus one
load per task. At ~370 rows a slice, a task runs about 7–8 minutes. The model is 9.5 GB, so a
smaller GPU would serve it; the array asks for a typed H100 per the owner's preference.

## Not done

- The context window is the manifest's `num_ctx: 8194` tokens. That holds a transcript of roughly
  25,000 characters beside the questions; the pilot's longest was 1,248. The node sends the
  transcript whole and records `transcript_chars`. Ollama truncates an over-long prompt silently,
  so check the r9 distribution of `transcript_chars` before reading a probability from the far
  tail. A cap would be a new config key, with its own derivation.
- Thresholds are unfitted, as above.
- `ollama serve` stdout goes to a per-task log beside the slice log. It is not parsed, so a server
  that falls back to CPU shows up only as latency.
