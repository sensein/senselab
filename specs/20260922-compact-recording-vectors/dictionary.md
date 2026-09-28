# The data dictionary

Added 2026-09-27 on the owner's request: "for the parquet add a data dictionary that describes how
each variable was computed. verify in code. make this info available in the parquet viewer."

## Where it lives and how it is kept true

- **One file:** `src/senselab/audio/workflows/triage/data/recording_vectors/dictionary.yaml`. It has
  explicit entries for the single columns and two families, `measurement` and `gate`, whose
  templates expand over a member list. `recording_vectors.dictionary()` expands the families.
- **The dtype is not in the YAML.** It is taken from `schema()`, so it cannot drift from what the
  writer emits.
- **The load refuses** an entry with no column, a column with no entry, a column described twice,
  and an entry missing a field.
- **Tests in `recording_vectors_dictionary_test.py`:**
  - every `source` names a function, method, class or module constant that exists in the file it
    names, resolved by AST;
  - the extractor is always the first source;
  - the families' members are exactly `MEASUREMENTS` with their kinds, and exactly `GATE_NAMES` with
    `GATE_SPECS`' readings and directions;
  - every closed vocabulary listed under `values` equals the constants the code declares: `Triage`,
    `Release`, the route states, the residue methods, the condition kinds, REVIEW's statuses, and
    the reviewer's judgment states;
  - no located gate is a conformance or flag gate, which is why, since `schema_version` 10, it carries
    only its bound, read from the fold's `gates.bounds`.
- **The tests check that names resolve, not that prose is right.** Each `computation` was written by
  reading the cited code at `e0eadb37`. A change to what a node computes can leave its entry wrong
  while every test passes, so an edit to a writer should be read against its entry.

## The file carries it

- **Metadata:** `schema()` puts the expanded dictionary, as JSON, under
  `senselab.recording_vectors.dictionary`, beside `senselab.recording_vectors.schema_version`.
- **Coverage:** every shard carries it, and so does the merged file. The merge refuses shards whose
  dictionaries differ.
- **Size:** 115 kB per file, once, in the footer.
- **No `schema_version` bump:** no column, layout or vocabulary changed, and a reader that ignores
  the key reads the file as before.
- **Checked on real stores:** three r6 stores were built into a shard and merged. Both files carried
  227 entries in schema order, and the viewer's own reader (`readDictionary`, over the vendored
  hyparquet) read them back.

## The viewer

- **Reading it:** `decode.js:readDictionary` reads the key from the footer. It returns null for a
  file without one, so an older file still opens and the dictionary controls are disabled.
- **Showing it:** the page has a searchable panel over every entry. An info button beside a
  column's name on the axes, the facets, the measurements table and the decision record opens that
  column's entry.

## Where schema.md disagreed with the code

Each was corrected in `schema.md`; the code was not changed.

| schema.md said | the code does |
| --- | --- |
| `release_ground` is one of seven grounds, three behind `release_without_redaction` and four behind `not_assessed` | 18 grounds across four tuples: 8 without redaction, 4 not assessed, 3 withheld, 3 with redaction (`vocabulary.py`) |
| the version table ended at 5 | 6–9 each added columns (reviewer; residue and scan; ledger counts; condition kind and `task_words_n`) |
| `pii_findings_n` counts the store's `pii` entities | it counts live `pii` entities carrying an extent, and only where a detector ran |
| `m_<name>` is null ⟺ `m_<name>_n == 0`, enforced by a test | a scalar whose every reading is non-finite (a NaN peak over floor) is null with `_n > 0`. No r6 row hits it, and no test covers it |
| a located gate reaches its three columns through the fold's record | no located gate is a conformance or flag gate, and `gate_readings` skips a gate with no reading. The fold never records one, so its columns are always null (0 non-null in r6) |
| `gate_family` is null when the recording declared no family (the out-of-family mode) | a group resolves only for a declared family, so a resolved group always carries its family |
| `gate_applied_n` is `0` only when no group resolved | it is also `0` when the owning branch left no in-family report (2,519 r6 rows, against 29 with no group) |
| (the `_reviewer_columns` docstring) every `llm_*` column is None without an annotation | `llm_flagged_categories` is `[]` and `llm_flagged_n` is `0`. The docstring was corrected |
