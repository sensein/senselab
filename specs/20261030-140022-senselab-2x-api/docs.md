# Documentation and example data (P-41)

The docs are senselab's public explanation and the running view of 2.x implementation work.
Every PR gets a preview.

## Audiences and structure

The docs follow the Diátaxis split (tutorials, how-to guides, explanation, reference), entered from
three audience paths that share the same pages:

| Audience | Starts from | Needs |
|---|---|---|
| **Scientists** | a research question (`use-cases.md` §1) | which pipeline answers it, what each measure means, quality control, provenance and reproducibility, how to cite |
| **Clinicians** | a measure or a finding | what it captures and what it misses, quality and limitations, listening and visual examples, license and privacy constraints. Results come first; code comes second |
| **Engineers** | an API or an extension point | reference generated from declarations, adding a task, backend, dataset adapter or substrate, execution on a laptop and on Slurm, contributing (`engineering.md`) |

- **Reference** is generated from the task, backend and profile declarations (design §4.10):
  schemas, default models, stacks, license records, measured resource tables. It cannot drift
  from code.
- **Tutorials and how-tos** are the use-case pipelines (`testing.md` §4) rendered as pages. They
  are tested by construction.
- **Explanation** pages come from the specs: concepts, identity and reuse policy, the flywheel
  roadmap. Rationale stays in `specs/` (P-21) and is linked.
- **Prose standard:** P-31, reviewed in CI by the prose model.

## Rich examples on public data

- Each example is a senselab pipeline run on a **public dataset subset**. It shows outputs
  (tables, plots, and audio or video snippets where the license allows) with a provenance link.
  Examples are executed, never hand-written.
- Datasets are drawn from the verified open sets in `benchmarks.md`. Candidates:
  - LibriSpeech, AMI, CREMA-D, PVQD and COUGHVID for speech and voice;
  - PTB-XL and MIT-BIH for ECG;
  - Sleep-EDFx and Capture-24 for sleep and actigraphy;
  - GazeBase for eye tracking.

  Each dataset's terms apply. For example, Capture-24 forbids re-identification, and some sets
  allow publishing aggregate results only. The license policy (P-23) is checked when the docs are
  built.
- **Where examples run:**
  - CPU examples run on small subsets during every docs build.
  - GPU examples run on ORCD on a schedule (`testing.md` cadence). Their outputs are published as
    versioned artifacts with provenance, which the docs reference. They are never committed.

## Showing implementation progress

- **Every PR gets a preview** at `sensein.group/senselab/pr-<N>/`, which the existing
  `docs-preview.yaml` provides. 2.x extends it from the pdoc API reference to the full site.
- **A generated "2.x status" page** shows:
  - each task's state, read from the registry: declared, implemented, ported, or benchmarked;
  - phase exits met;
  - the consolidation inventory (P-37);
  - the migration ledger marks (P-39);
  - open review-register items.
- **Each page carries a status badge**: available in 2.x, 1.x only, or planned.

## Data policy: toy fixtures versus pulled data

**In the repository**
- **Toy fixtures only.** They are for logic and contract tests.
- **Prefer generators over files.** Synthetic tones, noise, chirps, simulated ECG and
  accelerometry are generated in code, so they take zero bytes and the tests can be parametrised.
- **Committed files** are tiny, public-domain or CC0 clips, each with a license record.
- **Budget (design call):** at most 256 KB per file and 2 MB in total, checked in CI.
- **Moves out of the repo:** today's 16 MB `data_for_testing/`, notably the 10.7 MB mp4 and the
  1 MB WAVs, moves to pulled data. Its fixtures are replaced by generators where possible.

**Pulled at build or test time, never committed**
- **Data manifests:** versioned YAML files (`name`, `version`, source URL and mirrors, `sha256`
  per file, license record, terms, subset spec). They live beside their consumers (`docs/`,
  `benchmarks/`, use-case tests).
- **One fetch implementation** serves docs, benchmarks and use-case tests (P-37):
  `senselab data fetch <manifest>`.
  - It downloads into a content-addressed cache outside the repo and verifies every hash.
  - It refuses a mismatch and accepts a mirror only on a matching hash.
  - A cache hit trusts a blob, without reading it, only if `lstat` shows a regular file (not a symlink) whose size, mtime, inode and ctime equal the `BlobStamp` recorded when the dataset was completed; any other blob is re-hashed and refetched on a mismatch. A full re-hash of every blob is opt-in (`--verify`), so a cached fetch of a large dataset does not re-read it. Blobs and markers are read-only, and manifest paths are compared by Unicode canonical caseless match (NFD, full case folding with `str.casefold`, NFC), so two paths that a case-insensitive filesystem (APFS, NTFS) may store as one file are refused. Full folding is wider than NTFS's simple upcase table (it joins `ß` and `ss`); refusing such a pair is the safe side. If two writers commit one blob and the first `os.link` loses to a file that appeared meanwhile, that file is kept only if its bytes hash to the blob's name; otherwise the verified staged copy replaces it.
  - A dataset adapter (P-11) reads the result.
- **Identity:** a pulled file's identity is its content hash, so results computed on it key
  correctly (ADR-002). CI caches by manifest hash.
- **Failure:** an unavailable source fails the build with the manifest entry and the fix. The
  build never silently substitutes other data.

## Phasing

- **Phase 1:**
  - the data manifest and fetch, used first by the benchmark smoke suite;
  - the fixture budget check;
  - the generators that replace the committed fixtures.
- **Phase 2:** reference generated from declarations; the status page and badges in the PR
  preview.
- **Phase 3 onwards:** each use-case pipeline ported in a phase lands with its docs page and a
  public-data example in the same phase.
- **Phase 5:** the 1.x notebooks are retired once their pages exist.
