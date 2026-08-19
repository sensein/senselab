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

## The second half, found when CI failed

The first fix pointed the generator at `docs/function-dependencies.md` and asserted the committed
copy matched it. CI failed: **the file was missing.** `docs/` is gitignored — `.gitignore:192`,
under the comment `# pdoc documentation` — because it is pdoc's output directory, written by
`pdoc src/senselab -o docs` in both `docs.yaml` and `docs-preview.yaml`.

That is the rest of the explanation for why the collision survived. `docs/compatibility-matrix.md`
was tracked only because someone force-added it past the ignore rule, so:

- it never appeared in `git status`, and an accidental overwrite showed up as nothing at all;
- it lived in a directory a CI job writes to, alongside generated HTML;
- and a generator aimed at the same directory looked, from the outside, entirely reasonable.

A hand-maintained document inside build output cannot be defended by naming; it has to be moved.

## How it happened

`specs/20260419-133236-test-classification-deps` created the path as generated output (its T004,
T024 and T037 all treat it as such). Later work — the diarization backends plan, then the pyannote
4.x change — edited the committed file by hand to register new backends and bump
`pyannote-audio >=4.0`. Both were reasonable against the file as it then read. Neither noticed that
a generator claimed the same path, because nothing failed: the script is not run in CI, and the
only symptom is destruction at the moment someone runs it.

## Fix

The two documents are separated by kind, not just by name.

**The hand-maintained one moves out of build output** to `COMPATIBILITY.md` at the repo root, beside
`README.md`, `SECURITY.md` and `CHANGELOG.md`. It is tracked, visible in `git status`, and no build
writes to its directory. It gained two lines naming its generated counterpart, so a reader who wants
the per-function view knows where it is.

**The generated one stays in `docs/` and is not committed.** It is derived from
`COMPATIBILITY_MATRIX` on every docs build, so a committed copy could only ever be a stale duplicate
of what the build already produces. Both `docs.yaml` and `docs-preview.yaml` now run the generator
immediately after pdoc, which is what puts the table on the published site. The path is one constant,
`compatibility.GENERATED_DOC`, imported by the script rather than repeated in it.

`src/tests/utils/compatibility_doc_test.py` holds four checks:

1. `COMPATIBILITY.md` exists, is tracked, and is **not** under any gitignore rule. This is the check
   that would have caught the original defect, and it fails if anyone moves a hand-maintained
   document back into build output.
2. `GENERATED_DOC` is neither the hand-maintained path nor the old shared one, **is** ignored, and
   its content differs from the hand-maintained document.
3. Both docs workflows invoke the generator — since the table is no longer committed, the build is
   the only thing that can produce it, and a workflow publishing `docs/` without it would ship a
   site missing the table.
4. The script imports the path constant rather than carrying its own literal.

Regenerating is no longer something anyone must remember, because nothing is committed to go stale.
That is a stronger guarantee than the idempotence check the first attempt reached for — which could
not have worked anyway, since the file it wanted to compare cannot be committed.

## Rejected

**Fold the hand-maintained content into the generator.** The version bounds are measurements, not
facts derivable from `COMPATIBILITY_MATRIX`: "verified by pinning each package to its minimum in an
isolated venv" is a record of work performed on a host. A generator would have to invent them.

**Delete the generator.** The per-function table is useful and is genuinely derived; the defect was
its destination, not its existence.

**Leave the script and document the hazard.** A comment does not prevent the next person from
running a script named `generate-compat-matrix.py` to generate the compatibility matrix.

**Force-add the generated table past the ignore rule**, as the hand-maintained file had been. That is
the practice that hid this defect; repeating it for a second file in the same directory would be
choosing the same trap knowingly.

**Keep the hand-maintained document at `docs/compatibility-matrix.md` and only redirect the
generator.** This was the first attempt, and CI rejected it. It leaves a hand-maintained file inside
pdoc's output directory, invisible to `git status` and adjacent to generated HTML — so the next
collision is as quiet as this one was.
