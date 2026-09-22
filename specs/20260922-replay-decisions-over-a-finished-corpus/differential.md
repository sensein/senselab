# The differential: what the replay actually changed

`scripts/triage_replay_diff.py` over `senselab.audio.workflows.triage.replay_diff`. It answers, for
the whole corpus, whether a decision moved, which grounds appeared and disappeared, what each node
did differently, what the PII change did, and — the majority case — how much did not move at all.

## It needs the replayed store and nothing else

The claim was worth checking rather than assuming, because the whole shape of the tool depends on
it. It holds, and this is why.

A replay retires rather than deletes. `extend.live_decisions` finds every **live** entity whose
generating activity's node is in `REPLAYED_NODES`, and `retire_decisions` writes one
`wasInvalidatedBy` edge per entity against an activity of node `REPLAY`, step `decision_superseded`.
Only then does it run the nodes again. Two consequences fix the partition exactly:

- The set of entities carrying an edge to a `REPLAY`/`decision_superseded` activity **is** the set
  that was live the instant before the replay began. That is the corpus run's decision, entire.
- Every entity a replayed node generated that does **not** carry such an edge was written by the
  replay. It cannot collide with what it replaced, because `ProvStore` mixes `run_id` into every id
  digest and the replay reads under `replay_run_id(run_root, config_hash)`.

So `split_generations` needs no second tree, no original store and no `run.json`. Confirmed on real
data: probe over `triage_replay_20260922/smoke_out`, 16 recordings, every one yielding a `before`
and an `after` set with identical `(node, prov_type)` profiles and disjoint ids, both carrying their
own `VERDICT` verdict entity with the whole `FileVerdict.record()` on it. `ADMIT` and `PREPROCESS`
entities are in neither set, which is what a replay reading them off disk should leave behind.

Two limits, both recorded rather than worked around:

- **A store replayed twice under different configurations cannot be split into three.** Every
  retirement activity carries node `REPLAY` and step `decision_superseded` whatever pass wrote it,
  so a second pass's retirements are indistinguishable from the first's. The row carries status
  `ambiguous` when more than one marker config hash is present, and is still counted; none of the
  corpus's stores is in that state (all 16 smoke stores, and every landed `out/` store checked,
  carry exactly one marker: config hash `9bf3416760a0a72b`, commit `f3da4081…`).
- **The stem is not in the replayed run root.** The replay writes `store.jsonl` and `prov/` into the
  mirrored root and no `run.json`, so the driver takes the stem from the same manifest the replay
  took and derives the mirrored root the way `mirror_run_root` did: `out_root / entity_subdir(stem)
  / <finished run dir name>`.

## What a move is

The comparison is of content, not ids — the replay's run id makes an unchanged decision take a new
id, so ids say nothing.

`FileVerdict.record()` grew a field between the two commits. A record with `gates` compared against
one without it is not a decision that moved, so the row separates three things: `changed_keys` (keys
both records carry whose value differs), `new_keys` (keys only the replay's record carries) and
`dropped_keys`. `identical` is true when `changed_keys` is empty and nothing moved on the two axes
the record does not carry — the deviation assertions and the PII scan. On the 16 smoke recordings,
all 16 are identical with `new_keys == ["gates"]`, which is the correct reading of a run where the
gates moved into VERDICT and changed no outcome.

## The PII axis, and the direction that leaks

`_speech_found_pii` gates REDACT on live `pii` entities, and the replay retires the old ones before
SPEECH runs, so each pass's findings are cleanly its own. Whether the scan ran is read off SPEECH's
`pii_scan` measurements: the skipped path writes one carrying `scanned: false` and a controlled
reason, and every run also writes a second, unconditional `pii_scan` naming the detectors. So
`scanned` is false when **any** `pii_scan` in that generation carries `scanned: false`, true when
the pass wrote one and none does, and None when the pass wrote none (SPEECH did not run).

`pii_scanned_then_not` — scanned before, not scanned now — is counted apart from everything else and
named stem by stem in the report, because that is the direction in which a mistake leaks. Its
mirror, `pii_not_scanned_then_now`, and REDACT's own run state in both directions, are counted
beside it.

## Nothing that is content leaves a store

The report is the one most at risk of quoting what it should not, since it exists to judge a PII
change. Every value a row carries is read from a named categorical field: outcome, release, route
and run-state names; declared deviation types; `node|ground` where the ground is `reason_ground` of
a `why` the vocabulary guarantees is controlled; config paths; PII detector categories; and stems.
No attribute dictionary is copied wholesale, which is what would otherwise carry a word's `text` or
a finding's surface.

Two tests hold it: a fixture store whose `word`, `pii` and `deviate` entities all carry a
distinctive string asserts that string is in the store on disk and in neither the serialised row nor
the rendered report.

## Interface

```bash
# One array task's stride: read the replayed stores, write one row each.
uv run python scripts/triage_replay_diff.py rows MANIFEST \
    --slice-index I --slice-count N --out-root REPLAY_OUT [--log-dir DIR]

# The fold: every row under a tree into replay_diff.json and replay_diff.md.
uv run python scripts/triage_replay_diff.py report ROWS_DIR [--out DIR]
```

`MANIFEST` is the manifest the replay itself took — `stem` and `enhanced` per line — and
`--out-root` is the `--out-root` the replay was given. Rows go to `<log-dir>/slices/`, beside the
manifest by default; nothing is written under the corpus tree or the replay tree.

Row statuses: `ok`, `not_replayed` (no marker yet), `no_store` (the replay has not written this run),
`unreadable`, `nothing_retired` (the corpus run left no live decision to retire), `ambiguous`,
`error`. A task exits nonzero only on `error`, so running it against a partly-finished replay is a
normal thing to do.
