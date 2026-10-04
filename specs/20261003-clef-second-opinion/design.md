# Clef as the second opinion

2026-10-03. Owner: "replace nimble with clef". SECOND_OPINION now asks Cloudflare's Clef 27B instead
of Bespoke Labs' Nimble 9B. The node, its four questions, the comparison with the reviewer and the
flag ground are unchanged. The design they follow is
`specs/20261001-nimble-second-opinion/design.md`. This note records only what the swap changed.

## The model

| | |
|---|---|
| Model | `clef:27b`. Ollama library tags `27b` and `latest` resolve to the same manifest. |
| Licence | Apache-2.0. Released 2026-10-01; the Hugging Face repo is `Cloudflare/clef`. |
| Image config | `model_family` qwen35, `model_type` 27.0B, `file_type` Q4_K_M, `requires` 0.35.1, `capabilities` decision and vision |
| Manifest | `sha256:2bb11a61d1fb5d7a51f136ad9f569f970727377cf7b96037a06285d81fa25b73` |
| Config | `sha256:7b967ed607c3e1a56d0883e2af077e7cff88f4e8b08ea9f796abcf450d4f8ab2` |
| Weights | `sha256:6c02216a0055c1e1e92994d240e37c440f1c09b30281f538f4a38ebab0743da9`, 17,059,634,656 bytes |
| Other layers | vision projector `f0e2930e…` (927,607,360 bytes, unused here), licence `50cbab8a…`, params `58e1b82a…` = `{"num_ctx":16384}` |

The digests were read from `registry.ollama.ai/v2/library/clef/manifests/27b` on 2026-10-03, and
the manifest file hashes to the pinned manifest digest. `verify_pin` hashes every layer the
manifest names, so the context length (16384) and the projector are part of the identity.

`clef-flash` (9B) is not used. ollama/ollama#18769 reports it failing on `/v1/systemone` with
"Clef: non-finite logit" on CUDA and "cannot open model" on CPU, while `clef:27b` answers on the
same endpoint.

## The request

The question set needed no change. Clef serves the same `/v1/systemone` endpoint, and its library
page documents the same request: `model`, a `state`, and named `questions`. Each question has a
`type` (`choice`, `noul` or `score`), `instructions` and `criteria`. The answers come back under
`answers`, per type: a choice with per-option probabilities, a noul with a probability, a score
with a legend. `QUESTION_SET_VERSION` stays 1, because neither the questions nor the state changed.

No Nimble answer can be reused. The result-cache key carries the process name, which is now
`second_opinion_answers`, and the weights digest.

## Names

The names are now model-neutral, so a further swap renames nothing:

| was | now |
|---|---|
| measurement `nimble_opinion` | `second_opinion_answers` |
| `verdict.nimble_disagreement_flags`, `nimble_confident_yes`, `nimble_confident_no` | `verdict.second_opinion_*` |
| parquet `nimble_*` (schema 17) | `second_opinion_*`, plus `second_opinion_model_id` (schema 18) |

## What is unmeasured

- **Thresholds:** 0.8 and 0.2 are kept from the Nimble pilot, and the owner said to leave them
  (2026-10-02). Clef is trained with a Brier term for calibration, but where its answers fall on
  this corpus is unknown until a run.
- **Determinism:** Clef scores with a non-autoregressive head, so the seed should be inert as it
  was for Nimble. This is not measured.
- **Speed:** the cold load and the warm per-request time, at 18 GB of weights against Nimble's
  9.5 GB, are not measured. `workers` 4 with `num_ctx` 16384 each is assumed to fit an 80 GB
  H100 or A100 beside the weights. `require_gpu` refuses to serve the model unless it is wholly
  resident.

The pilot (below) is the first measurement.

## Staging

Ollama 0.35.1 or later and the pinned store live under
`/orcd/data/satra/002/models/ollama-senselab/`, beside the 0.35.0 binary and the Nimble blobs,
which are kept.
