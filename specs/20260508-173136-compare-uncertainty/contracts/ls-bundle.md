# Label Studio Bundle Contract — comparator additions

The comparator stage extends the existing `<run_dir>/labelstudio_tasks.json` and `<run_dir>/labelstudio_config.xml` produced by PR #510. Existing tracks are unchanged.

## XML config additions (`labelstudio_config.xml`)

For every (pass, comparison_kind, task_or_pair) that produced a non-empty parquet, append one `<Labels>` block to the existing `<View>` element:

```xml
<Labels name="raw_16k__compare__asr__whisper_vs_granite" toName="audio">
  <Label value="agree"/>
  <Label value="disagree"/>
  <Label value="incomparable"/>
  <Label value="one_sided"/>
</Labels>
```

For ASR-vs-ASR specifically, **also** append a TextArea block carrying the WER + both transcripts so reviewers can read the actual difference:

```xml
<TextArea name="raw_16k__compare__asr__whisper_vs_granite__text"
          toName="audio" perRegion="true" editable="false"
          placeholder="WER and transcripts for this disagreement"/>
```

## Tasks JSON additions (`labelstudio_tasks.json`)

Each existing pass-level task gains additional `result[]` entries. One Labels region per non-`agree` row in the comparator's parquet that is also in the disagreements top-N (so we don't blow up the LS bundle for noisy regions):

```json
{
  "id": "raw_16k__compare__asr__whisper_vs_granite__0042",
  "from_name": "raw_16k__compare__asr__whisper_vs_granite",
  "to_name": "audio",
  "type": "labels",
  "value": {
    "start": 12.30,
    "end": 12.50,
    "labels": ["disagree"]
  }
}
```

For ASR-vs-ASR, the same `id` MUST also produce a paired TextArea region:

```json
{
  "id": "raw_16k__compare__asr__whisper_vs_granite__0042__text",
  "from_name": "raw_16k__compare__asr__whisper_vs_granite__text",
  "to_name": "audio",
  "type": "textarea",
  "value": {
    "start": 12.30,
    "end": 12.50,
    "text": ["WER 0.6: A=\"four little rabbits\" | B=\"four white rabbits\""]
  }
}
```

## Track-name convention

The full grammar:

```text
<pass>__compare__<task_or_pair>__<short_a>_vs_<short_b>
```

Where:

- `<pass>` is `raw_16k`, `enhanced_16k`, or `pass_pair` (for `raw_vs_enhanced` rows).
- `<task_or_pair>` is `diarization`, `ast`, `yamnet`, `asr`, or for cross-stream: `asr_vs_diarization`, `ast_vs_diarization`, `yamnet_vs_diarization`, `asr_vs_ppg`.
- `<short_a>`/`<short_b>` are short identifiers derived from the model id by `_safe(model_id)` (the same helper PR #510 introduced for diarization/ASR track names).

## Backwards compatibility

- All existing tracks (`<pass>__diarization__<model>`, `<pass>__asr__<model>`, etc.) are preserved bit-identically.
- New tracks are appended at the end of the `<View>` element so existing track ordering is stable.
- A run with `--skip comparisons` produces no new tracks (FR-005, SC-005).

## Region-volume control

To keep the LS bundle reviewable, only rows that meet one of these conditions emit an LS region:

1. The row is in the disagreements top-N (default 100).
2. `comparison_status != "ok"` (incomparable / one_sided rows are always shown so the reviewer sees coverage gaps).

This is a deliberate cap. Reviewers wanting full per-row visibility read the parquet directly. The LS bundle is for triage.
