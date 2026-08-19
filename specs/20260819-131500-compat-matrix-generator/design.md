# The generator was overwriting a different document

## What was wrong

`scripts/generate-compat-matrix.py` wrote `generate_matrix_markdown()` to
`docs/compatibility-matrix.md`. The file at that path is not what the generator produces.

| | tracked `docs/compatibility-matrix.md` | `generate_matrix_markdown()` |
|---|---|---|
| lines | 78 | 143 |
| bytes | 4536 | 15962 |
| subject | Python support; per-package min/max version bounds | one row per public function: required deps, GPU, isolated venv |
| provenance | hand-maintained; lower bounds verified by pinning each package to its minimum in an isolated venv, upper bounds by running the suite | derived from `COMPATIBILITY_MATRIX` |

They share a filename and nothing else. Running the script replaced a document recording
measurements no generator can reproduce — "lower bounds verified by pinning" is the result of
having done that work — with an unrelated table.

The divergence was diagnosed in `specs/20260818-093000-drop-pre-4x-pyannote/decision.md`, which
declined to run the generator for exactly this reason and recorded it as "a separate defect,
unfiled here". This files it.

## How it happened

`specs/20260419-133236-test-classification-deps` created the path as generated output (its T004,
T024 and T037 all treat it as such). Later work — the diarization backends plan, then the pyannote
4.x change — edited the committed file by hand to register new backends and bump
`pyannote-audio >=4.0`. Both were reasonable against the file as it then read. Neither noticed that
a generator claimed the same path, because nothing failed: the script is not run in CI, and the
only symptom is destruction at the moment someone runs it.

## Fix

The generated table moves to `docs/function-dependencies.md`, named for what it contains, with its
H1 retitled to match. The path is one constant, `compatibility.GENERATED_DOC`, imported by the
script rather than repeated in it. `docs/compatibility-matrix.md` stays hand-maintained and is
written by nothing.

`src/tests/utils/compatibility_doc_test.py` holds three checks: the committed generated document
equals what the generator produces (so regenerating is a no-op and the table cannot drift from the
matrix it describes); the generator's target is not the hand-maintained path, and its output does
not equal that file's contents; and the script imports the constant rather than carrying its own
literal.

The first check is the one that matters — it converts "someone must remember to regenerate" into a
test failure. It also means a change to `COMPATIBILITY_MATRIX` that forgets the doc now fails CI
with the command to run.

## Rejected

**Fold the hand-maintained content into the generator.** The version bounds are measurements, not
facts derivable from `COMPATIBILITY_MATRIX`: "verified by pinning each package to its minimum in an
isolated venv" is a record of work performed on a host. A generator would have to invent them.

**Delete the generator.** The per-function table is useful and is genuinely derived; the defect was
its destination, not its existence.

**Leave the script and document the hazard.** A comment does not prevent the next person from
running a script named `generate-compat-matrix.py` to generate the compatibility matrix.
