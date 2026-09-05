# Audio Flamingo 3: batched generation, and what it leaves of the vLLM gap

Measured 2026-09-03 on ORCD, H100 80GB HBM3, node2803, partition `pi_satra`, model SHA
`7d4bae64ee29878af6504ae6f6bb3e40492838ad`, transformers 5.5.4 / torch 2.14.0+cu130, attention
`sdpa` (flash-attn is unavailable on this cluster). Job 21903328, commit `cf2f0672`.

Inputs are the **same 40 clips** the vLLM comparison used — `af3_vllm_bench/manifest.json`, 396.0 s
of audio, 14 from `breath1.wav`, 13 from `maximum-phonation-time-1.wav`, 13 from
`prolonged-vowel.wav`, deliberately varied so prefix caching cannot inflate a result — read through
the same harness and the same timing points as `bench_tf.py`, so the rows sit beside the earlier
ones rather than near them.

## The defect

`describe_with_audio_flamingo` looped `for audio in audios`, issuing one `apply_chat_template`, one
`generate` and one `decode` per clip. The transformers path could not batch whatever the caller
passed. The vLLM comparison therefore measured **vLLM batched against senselab serialised**, and
attributed the whole difference to vLLM.

## The sweep

| batch_size | wall s | clips/s | RTF | peak alloc GB | peak reserved GB |
| --- | --- | --- | --- | --- | --- |
| 1 (control) | 36.60 | 1.093 | 10.8 | 15.52 | 15.73 |
| 4 | 21.09 | 1.897 | 18.8 | 15.77 | 16.12 |
| 8 (default) | 9.04 | 4.426 | 43.8 | 16.11 | 17.05 |
| 16 | 6.28 | 6.374 | 63.1 | 16.78 | 19.06 |
| 40 | 4.06 | 9.851 | 97.5 | 18.80 | 22.68 |

`batch_size=1` is the control and reproduces the earlier serial arm — 36.60 s against 38.26 s,
within 4% — which is what licenses reading the rest of the table against the vLLM row.

Single-clip generation 0.678 s, load 12.923 s (earlier arm: 0.79 s and 13.908 s).

## What survives of the 30.1×

| comparison | factor |
| --- | --- |
| original claim, serial transformers vs vLLM (38.26 / 1.27) | 30.1× |
| **batching alone, serial vs batched transformers (38.26 / 4.06)** | **9.4×** |
| **vLLM vs batched transformers at bs=40 (4.06 / 1.27)** | **3.2×** |
| vLLM vs batched transformers at the bs=8 default (9.04 / 1.27) | 7.1× |
| single-clip generation, vLLM vs transformers (0.678 / 0.368) | 1.84× |

**Roughly nine of the thirty were ours.** A real gap remains — 3.2× at equal batch, and vLLM also
ships FlashAttention 3 internally where this cluster cannot build flash-attn for transformers at
all, so even 3.2× is not a like-for-like kernel comparison. Load cost still inverts the single-clip
case: 12.9 s against vLLM's 51.2 s, so per-file work favours transformers end to end.

Memory is not what limits the batch here: bs=1 → bs=40 costs 3.3 GB of allocated peak (15.52 →
18.80 GB) on an 80 GB card. The default of 8 is set well below what the hardware allows, because the
caller controls clip length and 600 s clips window into 20× the features of a 30 s clip.

## Batching perturbs the answers, and that is not a vLLM property

Identical answers against the `batch_size=1` outputs, same clips, same prompt:

| against bs=1 | identical |
| --- | --- |
| bs=4 | 29/40 |
| bs=8 | 28/40 |
| bs=16 | 30/40 |
| bs=40 | 30/40 |

The earlier report recorded vLLM as "28/40 identical, mean similarity 0.920" and read it as vLLM
failing to be a drop-in. **Batched transformers lands in the same place, 28–30/40**, so the
perturbation belongs to batching, not to vLLM.

It is numerical rather than sampling noise: `generation_config` carries `do_sample: null` and the
single-clip output is byte-identical on a repeat call, so generation is deterministic for a *fixed
batch shape*. What changes an answer is the shape — padding width and batched-kernel reduction
order. Exact reproducibility therefore requires `batch_size=1` on either backend; it was never a
reason to prefer transformers.

Differences are mostly paraphrase at the same meaning (`'...before you respond'` against
`'...before your respond'`), with occasional genuine divergence (clip 7: "in a flat tone, while a
buzzing sound is heard in the background" against "in a slow and deliberate manner").

## An upstream inconsistency the loop was built on

The serial loop was justified in `doc.md` and in a code comment by: *`strip_prefix` is exposed on
the processor's `decode` and not on `batch_decode`, which is why generations are decoded one at a
time.* Both halves are false, verified against `transformers` 5.5.4:

- `AudioFlamingo3Processor.batch_decode` is `return self.decode(*args, **kwargs)` — a plain alias,
  so `strip_prefix` reaches it.
- `decode` is `decoded = self.tokenizer.decode(*args, **kwargs)` followed by
  `[self._strip_assistant_prefix_and_quotes(t) for t in decoded]`. That comprehension assumes
  `decoded` is a list. For a **1-D** sequence `tokenizer.decode` returns a `str`, so it iterates the
  string's characters: `strip_prefix=True` on a single sequence returned a 49-element list of
  single characters, and `.strip()` on that list raised `AttributeError`.

So `strip_prefix=True` was broken on the path the comment said was the only one supporting it, and
correct on the path it said did not. Passing a **2-D** batch is what makes it work
(`['hello world', 'hello world']`). The mocked test covering the flag returned a fixed string from a
stand-in `decode`, so it asserted the flag was forwarded and could not see that forwarding it broke
the call — the same shape of blind spot that let the CUDA dtype bug (`0d5f5792`) ship.

Padding was the other assumption worth checking, and it held: batched `apply_chat_template`
left-pads (the processor's `common_kwargs` default `padding_side: "left"` overrides the tokenizer's
own `right`), confirmed by all-leading zeros in the attention mask on clips of 1 s, 7 s and 3 s. All
rows' prompts therefore occupy the same columns and one shared offset slices every continuation; no
per-row mask arithmetic is needed. The value is now passed explicitly, because a silent upstream
change of that default would corrupt generations rather than fail.

## Not decided here

Whether to add a vLLM subprocess-venv backend. The corrected baseline this document establishes is
what that decision should be taken against; 3.2× at equal batch is the number, not 30.1×.
