# Contract: utterance estimator improvements

## A. Whisper token logits → `ScriptLine` fields

**Edit** `src/senselab/audio/tasks/speech_to_text/huggingface.py` (Whisper path):

```python
pipe(..., generate_kwargs={
    "language": ..., "num_beams": 1,
    "return_dict_in_generate": True,
    "output_scores": True,
})
```

- From returned `scores` (per-step logits), compute per-token softmax entropy and `avg_logprob`; extract Whisper `no_speech_prob` where available.
- Attach onto each emitted `ScriptLine` (new optional fields — data-model §6): `avg_logprob`, `no_speech_prob`, `token_entropy`.
- Non-Whisper backends leave these `None` (graceful degradation, FR-017).

**Edit** `src/senselab/utils/data_structures/script_line.py`: declare the three optional fields (default `None`) and map them in `from_dict`. This also **revives the existing dead `avg_logprob`/`no_speech_prob` reads** in `harvesters.py` (both presence and utterance).

## B. Overlap grid handling — `utterance.py`

- Utterance already receives its own `utterance_grid` (default 1.0 s / 0.5 s overlap) and already excludes boundary-straddling words (`asr_text_in_window(..., fully_contained=True)`). Contract confirms/keeps this and adds the token-entropy vote:

```python
votes[m] = {
    "text": ..., "phoneme_sequence": ..., "avg_logprob": ...,
    "alignment_ctc_score": ...,
    "token_entropy": mean_token_entropy_in_window(resolved, start, end),  # NEW, None-safe
}
```

- `mean_token_entropy_in_window`: mean of per-token entropies whose timestamp midpoint ∈ `[start,end)`; `None` when the backend didn't supply token entropy.

## C. Aggregator sub-signal — `aggregate.py::aggregate_utterance`

Add one sub-signal to the existing fold (pairwise phoneme distance + Whisper `1−exp(avg_logprob)` + PPG):

```python
# token entropy → uncertainty in [0,1]
te = mean over contributing models of normalized token entropy   # normalize by log(vocab) or a fitted temperature
if te is not None:
    sub_signals.append(te)
```

- Combined via the existing `--uncertainty-aggregator` (min/mean/max) — unchanged mechanism.
- Confidences from different backends mapped to a common calibrated `[0,1]` scale (FR-018) using the `CalibrationProfile.temperature` hook.

## D. Scene-quality coupling — `compute.py` / `utterance.py`

- After presence quality columns exist for the bucket's time span, compute a coupling multiplier:

```python
coupling = 1.0 + w_q * quality_snr_at(span) + w_s * (src_machine + src_environment)_at(span)
utterance_uncertainty_coupled = min(1.0, utterance_uncertainty * coupling)
```

- `w_q`, `w_s` are parameters with documented defaults. The multiplier is written to the `scene_quality_coupling` column (recorded, not hidden — FR-019). The pre-coupling value remains available in `model_votes`/`raw_aggregated_uncertainty`.

## E. Backward compatibility

- When token entropy is `None` for all models (no Whisper / non-Whisper only), utterance falls back to today's sub-signals exactly (SC-008).
- `aggregated_uncertainty` column meaning is preserved; coupling is an additional column, and applied to the reported utterance uncertainty only per the documented rule.
