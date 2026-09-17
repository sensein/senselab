# The optional LLM check over the redacted transcript

Owner-directed, 2026-09-17. Off by default; a cluster run turns it on.

## What it is for

The detector cascade marks *tokens*. Presidio, GLiNER and the rules cascade each ask "is this span a
name / a phone number / a date", and REDACT redacts what they mark. None of them asks the question a
human reviewer asks: **taken together, does what is left identify this person?** A birth month, a
street, an employer and a diagnosis are four spans no detector flags and one person.

An instruction-tuned model can ask that question, and can say *why*. Its reasoning is therefore the
product of this step, not a by-product of it. A verdict of "clean" from a model whose reasoning
nobody kept is a claim with no evidence behind it.

## What it is not

It is not a redactor. It never edits a released artifact, never widens a planned extent, and never
produces a span the audio is masked from. The masking it does do is **local to its own loop**: the
substring a round flags is replaced in the text the *next round* reviews, so the model sees the
effect of its own concern, and that masked text is discarded when the loop ends.

The reason is that an LLM's substring does not map reliably onto a consensus word, and a consensus
word is what carries an extent. Turning a flagged substring into a redaction would mean guessing at
a time extent from a text match, which is exactly the class of inference the graph refuses
elsewhere. Withholding needs no such guess.

## The rule: it annotates, and VERDICT decides — owner correction, 2026-09-17

> "the llm is part of a branch, so it can only annotate (with provenance)."

The same contract the branches took the day before — *"a refusal is a decision. a branch does not
decide. it's an authority on task/branch specific detection."* This check is a detector. Detectors
annotate.

**What was wrong.** `redact.py` carried

```python
if outcome is Outcome.PASS and llm.status == "flagged":
    outcome = Outcome.FLAG
```

and `vocabulary._release_from` reads REDACT's outcome as the release state. So an LLM flag withheld
the release. An unmeasured model was deciding whether a recording could be handed on.

**What it is now.** REDACT's outcome is its own — the detector path, the mask, the `scan_for_pii`
re-verification — on every path. The re-read's summary is a `redaction_llm_annotation` measurement
and nothing else.

| state | what REDACT's outcome does | what VERDICT does with the annotation |
| --- | --- | --- |
| `disabled` | nothing | nothing. The config left it off. |
| `not_run` | nothing | nothing. The detector path had already withheld. |
| `clean` | nothing | nothing. |
| `flagged` | **nothing** | raises `triage` to `flag` under `verdict.llm_redaction_flags`. Never `release`. |
| `absent` | nothing | nothing. Carried in `llm_redaction`; the detector path's answer stands. |

**Why the triage axis and not release.** They answer different questions. Triage asks whether a human
should look at the recording; release asks whether an artifact may be handed on. "A birth month, a
street and an employer are still here, and together they are one person" is a claim a human reviewer
should read — a triage question. It is not evidence that a detector missed a span, which is the only
thing the release axis is a reading of. The safety signal survives; the unmeasured gate does not.

`verdict.llm_redaction_flags` ships `true`: a false "residue" flag costs one review, a missed one
costs a disclosure. **No false-positive rate has been measured for this model on this corpus.**
`config-derivations.md` § verdict says so in the same words, because a default argued from an
asymmetry rather than from a rate has to say which it is.

`absent` deserves its argument, unchanged. An enabled step that cannot reach its model has two honest
options: withhold everything, or let the answer the detector path already reached stand and say
loudly that the extra check did not happen. Withholding would mean a cluster whose GPU queue is full
releases nothing at all, and the release would then depend on machine availability rather than on the
recording. So the answer stands — and it is never a *silent* pass: the annotation's `status` is
`absent`, its `failure` names the reason, an `available: false` measurement is in the store, the
review activity's model agent carries `unresolved_reason` rather than a commit, and the file
verdict's `llm_redaction` carries the whole record. **If an operator wants an enabled-and-absent
check to flag, that is a different rule and needs its own config key with its own derivation; it is
not the shipped one.**

A round that flags something and is then followed by a round that could not run keeps the flag. A
concern already raised is not withdrawn because the next round failed.

## The loop

1. review the redacted transcript;
2. if the round flagged nothing, stop — `clean` if this was round one, otherwise `flagged`;
3. otherwise mask each flagged substring with `[LLM_<CATEGORY>]` and review the masked text;
4. stop at `redaction.llm_check.max_iterations`.

Whether **any** round flagged is what decides the check, not whether the last one did. A model whose
single concern is resolved by masking it has still raised a concern about the text that would be
released, which does not carry that mask.

A substring the model names but the text does not contain is left alone. Nothing is guessed at on
the model's behalf.

## What reaches the store

One `REDACT`/`llm_check` activity, written on **every** path — disabled and not-run included, where
it carries `enabled` and no model agent — so the step's state is a record rather than an inference
from an absence. When the step ran, the activity is associated with a model agent carrying the
resolved commit (or an `unresolved_reason`).

Under it: one `redaction_llm_review` measurement per round, carrying the iteration, `available`, the
reasoning verbatim, the findings with their categories and one-sentence reasons, the failure if any,
the model id and the revision. And one `redaction_llm_annotation` measurement — status, iterations,
flagged **categories**, model id, revision, failure — which is what VERDICT reads. No reasoning
reaches it: a `why` is controlled vocabulary and a chain of thought quotes the transcript.

REDACT's verdict carries **no** `llm_check` field. It used to, and a field in a verdict the verdict
does not act on is a second record able to disagree with the first.

`report.py` reads both back (`_llm_annotation`, `_llm_reviews`) and renders status, concerns and
reasoning under the REDACT block, so the captured reasoning is reachable from the document a reader
opens rather than only from the store. The summary JSON carries the rounds under `llm_check` and the
summary under `llm_annotation`. The file verdict carries the same summary under `llm_redaction`.

None of this reaches `artifacts_dir`. The store, the run directory and the summary are the private
side; the release directory is the only thing gated on a pass, and the reviewer writes nothing into
it.

## The model

`google/gemma-4-31B-it-qat-w4a16-ct`. The owner asked for "gemma4 (32b)"; there is no 32B in the
family (12B / 26B-A4B / 31B / E2B / E4B), and 31B is the one meant. The memory arithmetic that picks
the QAT variant over the full BF16 checkpoint is in `config-derivations.md` under `redaction`.

Heavy dependencies live in the `pii-redaction-review` subprocess venv, built by the shared
`ensure_venv`, which is how every other heavy backend in this repo is isolated. The parent resolves
the ref to a commit and stages it via `hf_subprocess_env` before spawning, so the worker loads
offline from a snapshot directory named by that commit; on the online fallback the worker is given
`revision=<sha>`. The ref never reaches a load. Both revision allowlists in
`src/tests/utils/revision_pinning_guard_test.py` and `hf_load_coverage_test.py` carry the file with
the review that put it there — and both guards fired on the new file before it was added, which is
the mechanism working as designed.

## Tests

None of them needs the model. `redact_module.review_redacted_text` is stubbed with a scripted list
of `ReviewResult`s, which is what lets the iteration, the chain-of-thought capture, the
annotate-don't-decide rule and the absent path all be pinned on a laptop. Two of them run REDACT and
then VERDICT over the same store, which is the only place the release axis and the triage axis can be
read against one flagged re-read at once. The backend's own parsing
(`parse_completion`) is tested directly, including the case where the reasoning quotes the
transcript's `[CATEGORY]` placeholders — splitting the answer on the first `[` truncated it there.
