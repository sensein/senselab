# Label Studio Bundle Contract — comparator additions

The comparator stage extends `<run_dir>/labelstudio_tasks.json` and
`<run_dir>/labelstudio_config.xml` with three Labels tracks per pass plus three raw-vs-enhanced
tracks plus one TextArea track for utterance.

## XML config additions (`labelstudio_config.xml`)

Six `<Labels>` blocks (3 per pass × 2 passes) plus three for `raw_vs_enhanced`:

```xml
<Labels name="raw_16k__uncertainty__presence" toName="audio">
  <Label value="low"/>
  <Label value="medium"/>
  <Label value="high"/>
  <Label value="incomparable"/>
  <Label value="unavailable"/>
</Labels>

<Labels name="raw_16k__uncertainty__identity" toName="audio">
  <Label value="low"/>
  <Label value="medium"/>
  <Label value="high"/>
  <Label value="incomparable"/>
  <Label value="unavailable"/>
</Labels>

<Labels name="raw_16k__uncertainty__utterance" toName="audio">
  <Label value="low"/>
  <Label value="medium"/>
  <Label value="high"/>
  <Label value="incomparable"/>
  <Label value="unavailable"/>
</Labels>

<!-- … same three blocks for enhanced_16k … -->

<!-- raw_vs_enhanced delta tracks -->
<Labels name="pass_pair__uncertainty__presence" toName="audio">
  <Label value="low"/><Label value="medium"/><Label value="high"/>
  <Label value="incomparable"/><Label value="unavailable"/>
</Labels>
<!-- … identity and utterance same shape … -->
```

Plus one TextArea sibling for the utterance tracks (one per pass + one for `pass_pair`):

```xml
<TextArea name="raw_16k__uncertainty__utterance__text"
          toName="audio" perRegion="true" editable="false"
          placeholder="Per-bucket transcript consensus + dissenting models"/>
```

## Bin mapping

`aggregated_uncertainty` from each parquet row → label value:

| `aggregated_uncertainty` | LS label |
|---|---|
| `< 0.33` | `low` |
| `[0.33, 0.66)` | `medium` |
| `≥ 0.66` | `high` |
| (any) with `comparison_status == "incomparable"` | `incomparable` |
| (any) with `comparison_status == "unavailable"` | `unavailable` |

## Tasks JSON additions (`labelstudio_tasks.json`)

For each row in each of the 9 parquets, append one Labels region to the existing pass-level
task (the `pass_pair` rows are appended to the `raw_16k` task by convention, since LS doesn't
have a notion of "pass pair"):

```json
{
  "id": "raw_16k__uncertainty__utterance__25",
  "from_name": "raw_16k__uncertainty__utterance",
  "to_name": "audio",
  "type": "labels",
  "value": {
    "start": 12.5,
    "end": 13.0,
    "labels": ["high"]
  }
}
```

Plus, for every row in the **utterance** parquets, also append one TextArea region carrying
the per-bucket transcript consensus + dissenting-model transcripts:

```json
{
  "id": "raw_16k__uncertainty__utterance__25__text",
  "from_name": "raw_16k__uncertainty__utterance__text",
  "to_name": "audio",
  "type": "textarea",
  "value": {
    "start": 12.5,
    "end": 13.0,
    "text": [
      "consensus: \"hello world\"\nwhisper-turbo: \"hello world\"\ngranite: \"hello word\"\ncanary-qwen: \"hello\"\nqwen3: \"hello world\""
    ]
  }
}
```

## Region-id naming

`<pass>__uncertainty__<axis>__<row_idx>` — joins to `disagreements.json` entries via the
`ls_region_id` field. The TextArea sibling appends `__text`.

## Region-emission policy

- Every parquet row is emitted as an LS region (no top-N filtering at the LS layer — the
  `disagreements.json` top-N drives the *ranking* but every bucket is still scrubbable).
- For high-volume runs (long clips, many models), reviewers can hide low-uncertainty rows
  in LS via the standard label filter (e.g. show only `high` + `medium`).
