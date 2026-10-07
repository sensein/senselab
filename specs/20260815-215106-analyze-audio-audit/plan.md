# `analyze_audio` audit — Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to execute this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a verified findings register for
`src/senselab/audio/workflows/audio_analysis`, so the changes it implies can be decided against
the triage-workflow ("graph") design rather than guessed at.

**Architecture:** Four discovery sweeps (prose, computation, orchestration, assumptions) run as
parallel subagents over three measured layers, each emitting candidate findings to its own file.
Candidates then pass two gates — an adversarial refutation attempt, then a reproduction attempt —
before landing in the register, tiered by which gates they cleared. No source file is edited.

**Tech Stack:** Python 3.12, `uv`, `ast` for static sweeps, `pytest` for reproductions,
markdown for deliverables. Subagents via the Agent tool.

## Global Constraints

Copied from `design.md`. Every task's requirements implicitly include these.

- **This audit changes no code.** No fixes, no deletions, no file moves, no prose edits to source
  files. Sweep A *identifies and stages* prose; it does not edit. A task that modifies anything
  under `src/senselab/` has failed.
- **Every finding carries `file:line` plus a concrete failure scenario** (inputs → wrong output).
  A finding without a failure scenario is an unverified concern, not a finding.
- **Every sweep reports what it checked and found clean**, so silence is distinguishable from
  absence of looking.
- **Nothing reaches the register without surviving a refutation attempt.** The refuter defaults
  to refuted under uncertainty.
- **Verified-latent findings must name the exact experiment that would settle them** — the
  corpus, the hardware, the configuration. "Needs more investigation" is not an experiment.
- Every Python command runs through `uv run --no-sync`. Never bare `python`/`pip`.
- Never run `pytest -n auto`.
- Never `git add -A` unqualified. Always list paths explicitly.
- Reproduction scripts live in `specs/20260815-215106-analyze-audio-audit/repro/`, never in
  `src/`. They may import from `src/senselab` but must not modify it.
- Do not run Python with a working directory inside `audio_analysis/` — `types.py` there shadows
  the stdlib `types` module and every import fails with a confusing circular-import error. Run
  from the repository root. (This is itself a candidate finding; Task 3 records it.)

## Measured surface

From `design.md`, re-derivable by the script in Task 1:

| Layer | Files | Code lines |
| --- | --- | --- |
| Orchestration (imports a senselab task) | 20 | 5,423 |
| Computation (imports no task) | 61 | 11,721 |
| Prose (docstrings + comments) | — | 10,888 |

Orchestration files: `compute.py`, `labelstudio.py`, `stages.py`, `adaptive/plot.py`,
`background_mask.py`, `embeddings.py`, `adaptive/fusion.py`, `harvesters.py`, `l1_plot.py`,
`speech_presence.py`, `quality.py`, `adaptive/backends.py`, `asr.py`, `stage_context.py`,
`adaptive/evaluate.py`, `sound_sources.py`, `perturbations.py`, `adaptive/audio_io.py`,
`foreground.py`, `aggregate.py`.

Largest computation files: `contracts.py` (1,241), `adaptive/interventions.py` (910), `fuse.py`
(713), `adaptive/belief.py` (645), `plot.py` (543), `adaptive/loop.py` (534), `speaker.py` (337),
`run_config.py` (318), `speaker_identity.py` (310), `noise_floor.py` (302), `harmonize.py` (295).

## File structure

| Path | Responsibility | Action |
| --- | --- | --- |
| `specs/20260815-215106-analyze-audio-audit/measure.py` | Re-derives the layer table; run to refresh counts | Create |
| `.../candidates/sweep-a-prose.md` | Sweep A raw candidates | Create |
| `.../candidates/sweep-b-computation.md` | Sweep B raw candidates | Create |
| `.../candidates/sweep-c-orchestration.md` | Sweep C raw candidates | Create |
| `.../candidates/sweep-d-assumptions.md` | Sweep D raw candidates | Create |
| `.../candidates/deduped.md` | Merged, de-duplicated candidate list with stable ids | Create |
| `.../verdicts/refutation.md` | Per-candidate refutation verdicts | Create |
| `.../repro/` | Reproduction scripts, one per attempted finding | Create |
| `.../verdicts/reproduction.md` | Per-candidate reproduction outcomes | Create |
| `.../register.md` | The deliverable: findings by tier and severity | Create |
| `.../summary.md` | Layer measurements, patterns, implications for the graph | Create |
| `.../prose-migration.md` | Rationale worth keeping, relocated out of the code | Create |

Candidates, verdicts and the register are separate files because they are produced by different
agents at different times and merging them would make provenance unrecoverable.

---

### Task 1: The measurement script and the candidate scaffold

Establishes the numbers every later task cites, so no agent re-derives them differently.

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/measure.py`
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/.gitkeep`
- Create: `specs/20260815-215106-analyze-audio-audit/repro/.gitkeep`
- Create: `specs/20260815-215106-analyze-audio-audit/verdicts/.gitkeep`

**Interfaces:**
- Consumes: nothing.
- Produces: `measure.py`, runnable as
  `uv run --no-sync python specs/20260815-215106-analyze-audio-audit/measure.py`, printing the
  layer table and per-file breakdown. Later tasks cite its output rather than recomputing.

- [ ] **Step 1: Write the script**

```python
"""Re-derive the audit's layer measurements.

The audit slices `analyze_audio` by layer rather than by module, and every sweep's scope is
defined by this split, so the numbers have to come from one place that anyone can re-run. A
sweep that recomputed them differently would silently audit a different surface than the one
the design describes.

Run from the repository root:

    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/measure.py

Never run it with a working directory inside ``audio_analysis/``: that package contains a
``types.py`` which shadows the stdlib module of the same name, and every import then fails with
a circular-import error that names ``weakref`` rather than the real cause.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path("src/senselab/audio/workflows/audio_analysis")


def _counts(path: pathlib.Path) -> tuple[int, int, int, bool]:
    """Return ``(code, docstring, comment, imports_a_task)`` for one file.

    Docstrings are counted via the AST rather than by matching quotes, because a triple-quoted
    string used as a value is not a docstring and a regex cannot tell the difference.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                doc += text.count("\n") + 1
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith("#"))
    code = len(lines) - blank - comment - doc
    imports_task = "senselab.audio.tasks" in src or "senselab.utils.tasks" in src
    return code, doc, comment, imports_task


def main() -> int:
    """Print the layer table and the per-file breakdown."""
    if not ROOT.is_dir():
        print(f"not found: {ROOT} (run from the repository root)", file=sys.stderr)
        return 1

    orch: list[tuple[int, str]] = []
    comp: list[tuple[int, str]] = []
    total_doc = total_comment = 0
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        code, doc, comment, imports_task = _counts(path)
        total_doc += doc
        total_comment += comment
        rel = str(path.relative_to(ROOT))
        (orch if imports_task else comp).append((code, rel))

    orch.sort(reverse=True)
    comp.sort(reverse=True)
    orch_code = sum(c for c, _ in orch)
    comp_code = sum(c for c, _ in comp)
    prose = total_doc + total_comment

    print(f"orchestration : {len(orch):3d} files  {orch_code:6d} code")
    print(f"computation   : {len(comp):3d} files  {comp_code:6d} code")
    print(f"prose         :              {prose:6d}  ({total_doc} docstring + {total_comment} comment)")
    print(f"prose:code    : {prose / max(orch_code + comp_code, 1):.2f} : 1")
    print()
    print("orchestration files:")
    for code, name in orch:
        print(f"  {code:6d}  {name}")
    print()
    print("computation files (top 25):")
    for code, name in comp[:25]:
        print(f"  {code:6d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it and confirm it reproduces the design's numbers**

```bash
uv run --no-sync python specs/20260815-215106-analyze-audio-audit/measure.py | head -6
```

Expected: `orchestration` 20 files / 5423 code; `computation` 61 files / 11721 code; prose 10888;
ratio 0.64. If any number differs, `alpha` has moved — update `design.md`'s table in the same
commit and note the drift, rather than leaving the two documents disagreeing.

- [ ] **Step 3: Create the output directories**

```bash
mkdir -p specs/20260815-215106-analyze-audio-audit/{candidates,repro,verdicts}
touch specs/20260815-215106-analyze-audio-audit/{candidates,repro,verdicts}/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/measure.py \
        specs/20260815-215106-analyze-audio-audit/candidates/.gitkeep \
        specs/20260815-215106-analyze-audio-audit/repro/.gitkeep \
        specs/20260815-215106-analyze-audio-audit/verdicts/.gitkeep
git commit -m "audit: the layer measurement every sweep's scope depends on

The audit slices analyze_audio by layer, so the split has to come from one
re-runnable place. A sweep that recomputed it differently would audit a
different surface than the design describes."
```

---

### Task 2: Sweep A — prose

Runs first because its output changes what the other sweeps read: a stale docstring found here
tells Sweep B and C not to trust that module's stated contract.

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md`

**Interfaces:**
- Consumes: `measure.py`'s output.
- Produces: candidates with ids `A-1`, `A-2`, … Each carries `class` ∈ {`restates-code`,
  `rationale-to-migrate`, `stale-or-false`}.

- [ ] **Step 1: Dispatch the sweep agent**

Dispatch one agent (model: sonnet) with this prompt verbatim:

> Working directory: `/Users/satra/software/sensein/senselab`, branch `feat/analyze-audio-audit`.
>
> READ-ONLY. Do not modify any file under `src/`. You are producing a candidate list, not edits.
>
> Audit the prose in `src/senselab/audio/workflows/audio_analysis` (81 files). It currently
> carries 8,796 docstring lines and 2,092 comment lines against 17,144 lines of code — a ratio of
> 0.64:1, reaching 2.76× in `estimates.py`. The repository's convention is to explain *why* rather
> than *what*; the judgement is that it has been applied past the point of usefulness.
>
> Classify every prose block you flag into exactly one of three classes:
>
> 1. **`restates-code`** — describes what readable code plainly does. Example shape: a docstring
>    reading "Return the fraction of sessions where the backend reported exactly `true_k`" above
>    `def exact_count_accuracy(...)`. These are deletion candidates.
> 2. **`rationale-to-migrate`** — states a measurement, a failure that drove a choice, or a
>    rejected alternative. Load-bearing, but does not need to live wrapped around a function.
>    These are migration candidates; record the destination module-group summary you would put it
>    in.
> 3. **`stale-or-false`** — says something the code contradicts. **This class is a defect, not a
>    cleanup, and it is the most valuable output of this sweep.** Four such instances were found
>    and fixed in the two days before this audit: a module docstring claiming a 2.0 s / 1.0 s
>    default against a 1.0 s / 0.5 s signature; a module documenting two detectors where three
>    run; a "no boolean anywhere in its output" claim contradicted by a `bool` field; and
>    `p_voice` framing that outlived its own consumer's removal. Look for the same shape:
>    docstrings naming counts, defaults, or invariants that the code no longer honours.
>
> For every candidate emit exactly this block:
>
> ```
> ### A-<n>
> - class: restates-code | rationale-to-migrate | stale-or-false
> - location: <path>:<line>
> - quote: <the prose, trimmed to the relevant sentence>
> - why: <one sentence>
> - failure: <for stale-or-false only: what a reader who trusts this would get wrong>
> - destination: <for rationale-to-migrate only: which summary doc it belongs in>
> ```
>
> Prioritise `stale-or-false` — sweep for it across all 81 files before spending effort on the
> other two classes, which you may sample rather than exhaust (say explicitly what you sampled).
>
> End your file with a `## Checked and clean` section naming the files you read whose prose you
> judged proportionate and accurate. Silence must be distinguishable from not looking.
>
> Write to `specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md`. Return only:
> a count per class, and the three you consider most serious. Under 12 lines.

- [ ] **Step 2: Confirm the output exists and is well-formed**

```bash
grep -c "^### A-" specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md
grep -c "^- class: stale-or-false" specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md
grep -c "^## Checked and clean" specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md
```

Expected: a non-zero candidate count, and exactly one `Checked and clean` section. If the clean
section is missing, re-dispatch — a sweep that does not say what it cleared is not usable.

- [ ] **Step 3: Verify no source file was touched**

```bash
git status --short src/
```

Expected: empty. If not, the agent violated the read-only constraint: revert with
`git checkout -- src/` and re-dispatch with the constraint restated.

- [ ] **Step 4: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md
git commit -m "audit: prose candidates

Three classes with three fates: prose that restates readable code, rationale
that is load-bearing but misplaced, and prose the code contradicts. The third
is a defect class -- four instances were found and fixed in the two days before
this audit."
```

---

### Task 3: Sweep B — the computation layer

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md`

**Interfaces:**
- Consumes: `measure.py`'s computation file list; Sweep A's `stale-or-false` candidates (so a
  module whose docstring is known false is read with that in mind).
- Produces: candidates `B-1`, `B-2`, … each with `kind` ∈ {`unfitted-threshold`,
  `misnamed-statistic`, `unearned-confidence`, `promotion-candidate`}.

- [ ] **Step 1: Dispatch the sweep agent**

Dispatch one agent (model: sonnet) with this prompt verbatim:

> Working directory: `/Users/satra/software/sensein/senselab`, branch `feat/analyze-audio-audit`.
>
> READ-ONLY. Do not modify any file under `src/`.
>
> Audit the **computation layer** of `src/senselab/audio/workflows/audio_analysis`: the 61 files
> that import no senselab task — 11,721 lines of statistics, contracts, fusion and plotting. The
> largest are `contracts.py` (1,241 code lines), `adaptive/interventions.py` (910), `fuse.py`
> (713), `adaptive/belief.py` (645), `adaptive/loop.py` (534), `speaker.py` (337),
> `run_config.py` (318), `speaker_identity.py` (310), `noise_floor.py` (302), `harmonize.py`
> (295). Audit this as **mathematics**, not as workflow.
>
> Read `specs/20260815-215106-analyze-audio-audit/candidates/sweep-a-prose.md` first: any module
> listed there as `stale-or-false` has a docstring the code contradicts, so do not trust its
> stated contract.
>
> Four kinds:
>
> 1. **`unfitted-threshold`** — a numeric literal that gates a decision, with no written
>    derivation. The repository's rule is that thresholds belong in `data/` with a derivation, and
>    it names two defects that came from literals nobody fitted: a silhouette coefficient read
>    directly as a probability, and a 2→10 dB HNR ramp under which ordinary voiced speech (median
>    8.12 dB) read as only partly voiced.
>    **Trap: not every literal is a threshold.** A window length chosen for memory, a batch size,
>    a Hann overlap satisfying COLA are operational knobs, not decision gates. Report only numbers
>    that change a verdict. `data/run_config/default.yaml` is the good case — its values carry
>    derivations inline — so compare against it.
> 2. **`misnamed-statistic`** — a quantity whose name or type claims more than the computation
>    supports. The live precedent, fixed two days ago: `p_voice = 0.5 * (silhouette + 1)` rescaled
>    a partition-and-metric-dependent index into something every reader took for a probability.
>    **Trap: this needs the consumer, not just the producer.** A defensible computation can be
>    misread downstream, which is only visible by tracing where the value goes.
> 3. **`unearned-confidence`** — agreement computed over sources that are not independent; a
>    confidence that does not degrade when an input is missing; `0.0` returned where `None` is
>    meant, collapsing "we could not check" into "we checked and it was clean".
> 4. **`promotion-candidate`** — computation with no workflow dependency that belongs in
>    `utils/tasks/` or `audio/tasks/`. Record the target layer and what currently blocks the move.
>    Do not move anything.
>
> For every candidate emit exactly this block:
>
> ```
> ### B-<n>
> - kind: unfitted-threshold | misnamed-statistic | unearned-confidence | promotion-candidate
> - location: <path>:<line>
> - defect: <one sentence>
> - failure: <concrete inputs -> wrong output>
> - consumer: <for misnamed-statistic / unearned-confidence: where the value is read>
> - target: <for promotion-candidate: the module it should live in>
> ```
>
> A candidate without a concrete `failure` is an unverified concern — mark it
> `failure: UNVERIFIED` rather than inventing one. Inventing a plausible mechanism for a real
> defect is the specific failure mode this audit exists to avoid.
>
> End with `## Checked and clean` naming the files you read and judged sound.
>
> Write to `specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md`. Return
> only: a count per kind, and the three most serious. Under 12 lines.

- [ ] **Step 2: Confirm the output and the read-only constraint**

```bash
grep -c "^### B-" specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md
grep -c "^## Checked and clean" specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md
git status --short src/
```

Expected: non-zero candidates, exactly one clean section, empty `git status`.

- [ ] **Step 3: Record the `types.py` shadowing observation**

Append to the sweep file, since it was found while writing this plan and belongs in the same
candidate pool:

```markdown
### B-shadow
- kind: promotion-candidate
- location: src/senselab/audio/workflows/audio_analysis/types.py:1
- defect: The module is named `types`, which shadows the stdlib module of the same name for any
  Python process whose working directory is this package.
- failure: `cd src/senselab/audio/workflows/audio_analysis && python -c "import ast"` fails with
  `ImportError: cannot import name 'GenericAlias' from partially initialized module 'types'` —
  an error naming `weakref`, not the real cause. Observed while writing this plan.
- target: Rename, or accept and document. Not a promotion in the layering sense; recorded here
  because it is a naming defect in the computation layer with no better home.
```

- [ ] **Step 4: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md
git commit -m "audit: computation-layer candidates

11,721 lines that import no task, audited as mathematics: literals that gate a
decision without a derivation, statistics that claim more than they support,
confidence that does not degrade when its inputs vanish, and computation that
belongs in utils/ where another workflow could reuse it."
```

---

### Task 4: Sweep C — the orchestration layer

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/sweep-c-orchestration.md`

**Interfaces:**
- Consumes: `measure.py`'s orchestration file list.
- Produces: candidates `C-1`, `C-2`, … with `kind` ∈ {`model-in-control-flow`,
  `contract-violation`, `ordering-dependency`, `call-site-mismatch`}.

- [ ] **Step 1: Dispatch the sweep agent**

Dispatch one agent (model: sonnet) with this prompt verbatim:

> Working directory: `/Users/satra/software/sensein/senselab`, branch `feat/analyze-audio-audit`.
>
> READ-ONLY. Do not modify any file under `src/`.
>
> Audit the **orchestration layer** of `src/senselab/audio/workflows/audio_analysis`: the 20 files
> that import a senselab task — 5,423 lines. They are `compute.py`, `labelstudio.py`, `stages.py`,
> `adaptive/plot.py`, `background_mask.py`, `embeddings.py`, `adaptive/fusion.py`,
> `harvesters.py`, `l1_plot.py`, `speech_presence.py`, `quality.py`, `adaptive/backends.py`,
> `asr.py`, `stage_context.py`, `adaptive/evaluate.py`, `sound_sources.py`, `perturbations.py`,
> `adaptive/audio_io.py`, `foreground.py`, `aggregate.py`.
>
> Four kinds:
>
> 1. **`model-in-control-flow`** — a specific backend named in a branch rather than selected
>    through config; a hardcoded default that cannot be overridden; a stage that silently requires
>    one model's output shape. This is the sweep that scopes the model-pluggability work, so be
>    thorough.
>    **Trap: prefix dispatch to a backend-specific worker is legitimate.** Routing
>    `"nvidia/diar_sortformer"` to a Sortformer worker is correct. The finding is when a *decision*
>    depends on which model ran, or when adding a model requires editing control flow rather than
>    config.
> 2. **`contract-violation`** — a stage's output consumed downstream in a way its producer does
>    not guarantee. Read the producer's stated contract, then read every consumer.
> 3. **`ordering-dependency`** — a stage that must run before another with nothing enforcing it.
> 4. **`call-site-mismatch`** — a helper that is correct while its callers pass the wrong thing.
>    **Give this explicit attention rather than letting it emerge.** In a review of the PR that
>    produced much of this code, this single class accounted for four of nine findings and all
>    three defects that review missed: a helper that no-ops on a SHA while callers passed SHAs; a
>    function taking `list[str]` handed a model object; a `revision="main"` hardcoded beneath a
>    caller that had resolved a commit. The shape is always the same — the helper is right, the
>    call site is not, and the helper's own unit test passes.
>
> For every candidate emit exactly this block:
>
> ```
> ### C-<n>
> - kind: model-in-control-flow | contract-violation | ordering-dependency | call-site-mismatch
> - location: <path>:<line>
> - defect: <one sentence>
> - failure: <concrete inputs -> wrong output>
> - callers: <for call-site-mismatch / contract-violation: every call site you checked>
> ```
>
> Mark `failure: UNVERIFIED` rather than inventing a mechanism. A real defect with an invented
> mechanism is the specific failure mode this audit exists to avoid.
>
> End with `## Checked and clean`.
>
> Write to `specs/20260815-215106-analyze-audio-audit/candidates/sweep-c-orchestration.md`.
> Return only: a count per kind, and the three most serious. Under 12 lines.

- [ ] **Step 2: Confirm the output and the read-only constraint**

```bash
grep -c "^### C-" specs/20260815-215106-analyze-audio-audit/candidates/sweep-c-orchestration.md
grep -c "^## Checked and clean" specs/20260815-215106-analyze-audio-audit/candidates/sweep-c-orchestration.md
git status --short src/
```

- [ ] **Step 3: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/candidates/sweep-c-orchestration.md
git commit -m "audit: orchestration-layer candidates

5,423 lines that call senselab tasks: models named in branches rather than
selected by config, contracts violated downstream, unenforced ordering, and
helpers that are correct while their call sites are not -- the class that
produced four of nine findings in the review of this code's own PR."
```

---

### Task 5: Sweep D — assumptions

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/sweep-d-assumptions.md`

**Interfaces:**
- Consumes: both layer lists; Sweep B's `unfitted-threshold` candidates (a threshold with no
  derivation is often an assumption in disguise).
- Produces: candidates `D-1`, `D-2`, … with `kind` ∈ {`adult-speech-assumption`,
  `read-speech-assumption`, `lifespan-gap`}.

- [ ] **Step 1: Dispatch the sweep agent**

Dispatch one agent (model: sonnet) with this prompt verbatim:

> Working directory: `/Users/satra/software/sensein/senselab`, branch `feat/analyze-audio-audit`.
>
> READ-ONLY. Do not modify any file under `src/`.
>
> Audit `src/senselab/audio/workflows/audio_analysis` (all 81 files, both layers) for assumptions
> about **who is speaking and what they are doing**. The motivating goal is making this workflow
> work across the lifespan — infant through elderly — so the question is what currently presumes
> an adult reading connected prose.
>
> Read `specs/20260815-215106-analyze-audio-audit/candidates/sweep-b-computation.md` first: a
> threshold with no derivation is often an assumption in disguise, and you should say so rather
> than re-report it.
>
> Three kinds:
>
> 1. **`adult-speech-assumption`** — behaviour correct only for adult voices: VAD tuned on adult
>    speech, speaker embeddings whose separation was validated on adults, pitch or formant ranges,
>    a diarizer whose training population is adult conversational speech.
> 2. **`read-speech-assumption`** — behaviour correct only for fluent connected speech: language
>    models assuming continuous prose, thresholds fitted on read passages applied to spontaneous
>    talk, sustained phonation, or a breathing task.
> 3. **`lifespan-gap`** — a stage whose validated age range is unknown to its consumers.
>
> **Two traps, and this sweep is worthless if you fall into either.**
>
> First: the assumption usually lives in a **constant or a model choice, not in prose**.
> `PROFILE_WINDOW_S = 2.0` embeds a claim about how long a stable voiced segment is — true for an
> adult sustaining a vowel, questionable for a two-year-old. Reason about what each parameter
> presumes. Do not grep for the word "adult"; that finds documentation, not assumptions.
>
> Second: **"unvalidated" is not by itself a finding**, or this sweep returns 81 shrugs. Nearly
> every stage is unvalidated across the lifespan; saying so 81 times is noise. The finding is
> where an unvalidated output is **used as if validated** — a speaker count fed into a decision
> without carrying its own uncertainty, a model whose training population is never surfaced to the
> consumer that acts on its output, a threshold applied uniformly across ages with no per-age term.
>
> For every candidate emit exactly this block:
>
> ```
> ### D-<n>
> - kind: adult-speech-assumption | read-speech-assumption | lifespan-gap
> - location: <path>:<line>
> - assumption: <what the code presumes about the speaker or the task>
> - population: <who it holds for, and who it does not>
> - used-as-if-validated: <where the unvalidated output is consumed as if it were sound>
> - experiment: <the measurement that would settle it: what corpus, what metric, what comparison>
> ```
>
> The `experiment` field is mandatory and is the point of this sweep. Most of these will be
> impossible to reproduce without a child-voice corpus, so they will land as verified-latent
> findings — and a latent finding without a stated experiment is a permanent maybe rather than a
> measurement task.
>
> End with `## Checked and clean`.
>
> Write to `specs/20260815-215106-analyze-audio-audit/candidates/sweep-d-assumptions.md`. Return
> only: a count per kind, and the three most serious. Under 12 lines.

- [ ] **Step 2: Confirm the output, the clean section, and that every candidate has an experiment**

```bash
grep -c "^### D-" specs/20260815-215106-analyze-audio-audit/candidates/sweep-d-assumptions.md
grep -c "^- experiment:" specs/20260815-215106-analyze-audio-audit/candidates/sweep-d-assumptions.md
git status --short src/
```

Expected: the two counts are equal. A candidate without an experiment is not usable — re-dispatch
for the missing ones rather than accepting them.

- [ ] **Step 3: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/candidates/sweep-d-assumptions.md
git commit -m "audit: assumption candidates

What the workflow presumes about who is speaking and what they are doing. The
finding is not that a stage is unvalidated across ages -- nearly all are -- but
that an unvalidated output is consumed as if it were sound. Every candidate
names the experiment that would settle it."
```

---

### Task 6: Dedupe and assign stable ids

Four sweeps over an overlapping surface will report the same defect more than once, in different
vocabularies. Merging is a judgement task and gets its own gate.

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/candidates/deduped.md`

**Interfaces:**
- Consumes: all four candidate files.
- Produces: one list with ids `F-1`, `F-2`, …, each recording which sweep(s) raised it.

- [ ] **Step 1: Dispatch the dedupe agent**

Dispatch one agent (model: sonnet) with this prompt verbatim:

> Working directory: `/Users/satra/software/sensein/senselab`.
>
> READ-ONLY on `src/`. Read all four candidate files in
> `specs/20260815-215106-analyze-audio-audit/candidates/` and merge them into one list.
>
> Two candidates are **the same finding** when fixing one would fix the other. They are
> **different findings** when they share a location but would need separate fixes — a function
> with both a stale docstring and an unfitted threshold is two findings at one `file:line`.
>
> Merge conservatively: when unsure, keep them separate and note the possible duplicate. A
> wrongly merged pair loses one of them silently; a wrongly separated pair costs one extra
> verification.
>
> Emit for each:
>
> ```
> ### F-<n>
> - raised-by: <A-3, C-7 — every source candidate id>
> - layer: prose | computation | orchestration | assumption
> - location: <path>:<line>
> - defect: <one sentence, in one vocabulary>
> - failure: <concrete inputs -> wrong output, or UNVERIFIED>
> - experiment: <for assumption findings, carried through from sweep D>
> ```
>
> Then add a `## Cross-sweep patterns` section: any defect **class** that appears in three or more
> locations. A repeated pattern is more actionable than three separate rows, and it is the thing
> the summary will lead with.
>
> Write to `specs/20260815-215106-analyze-audio-audit/candidates/deduped.md`. Return: the merged
> count, how many were merged away, and the patterns you found. Under 12 lines.

- [ ] **Step 2: Sanity-check the merge**

```bash
grep -c "^### F-" specs/20260815-215106-analyze-audio-audit/candidates/deduped.md
for f in a-prose b-computation c-orchestration d-assumptions; do
  grep -c "^### [ABCD]-" specs/20260815-215106-analyze-audio-audit/candidates/sweep-$f.md
done
```

The merged count must be ≤ the sum and > the largest single sweep. A merged count equal to the
sum means nothing merged (suspicious with four overlapping sweeps); a count below the largest
single sweep means over-merging.

- [ ] **Step 3: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/candidates/deduped.md
git commit -m "audit: dedupe four sweeps into one candidate list

Merged where one fix would resolve both, kept separate where a shared location
still needs two fixes. Conservative on purpose: a wrongly merged pair loses a
finding silently, a wrongly separated pair costs one extra verification."
```

---

### Task 7: Gate 1 — adversarial refutation

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/verdicts/refutation.md`

**Interfaces:**
- Consumes: `deduped.md`.
- Produces: per-candidate `SURVIVED` / `REFUTED` with evidence.

- [ ] **Step 1: Dispatch one refuter per candidate, in batches**

For each candidate `F-<n>`, dispatch an agent (model: sonnet) with this prompt, substituting the
candidate's fields:

> Working directory: `/Users/satra/software/sensein/senselab`.
>
> READ-ONLY. Your job is to **refute** the following claimed defect. You are not evaluating
> whether it is interesting; you are trying to show it is wrong. **Default to REFUTED when you
> cannot establish it.**
>
> Claim `F-<n>`, at `<location>`: `<defect>`
> Claimed failure: `<failure>`
>
> Refute it by any of:
> - The code does not say what the claim says it says. Read it.
> - Something upstream already prevents the failure — a guard, a validator, a caller that never
>   passes the triggering value.
> - The claimed failure scenario cannot occur: the input is impossible, the branch unreachable,
>   the consumer does not exist.
> - The claim's *mechanism* is wrong even if something nearby is broken. **This matters as much
>   as the rest.** In the review of the PR that produced much of this code, three findings were
>   real while their stated mechanisms were fiction: a claim about a senselab version that never
>   existed, a claim that `chmod` modifies directories the caller does not own (it returns EPERM
>   and is swallowed), and a claim that a ref pointer was never written when it was written by a
>   different code path. If the defect is real but the mechanism is wrong, report
>   `SURVIVED-CORRECTED` and give the true mechanism.
>
> Return exactly:
>
> ```
> ### F-<n>
> - verdict: SURVIVED | SURVIVED-CORRECTED | REFUTED
> - evidence: <file:line and what you read>
> - corrected-mechanism: <for SURVIVED-CORRECTED only>
> - note: <one sentence>
> ```
>
> Under 15 lines.

Batch these — dispatch up to six at a time in one message so they run concurrently. Append each
verdict to `verdicts/refutation.md`.

- [ ] **Step 2: Tally**

```bash
grep -c "verdict: SURVIVED$" specs/20260815-215106-analyze-audio-audit/verdicts/refutation.md
grep -c "verdict: SURVIVED-CORRECTED" specs/20260815-215106-analyze-audio-audit/verdicts/refutation.md
grep -c "verdict: REFUTED" specs/20260815-215106-analyze-audio-audit/verdicts/refutation.md
```

Every candidate in `deduped.md` must appear exactly once. A candidate with no verdict has not been
gated and must not proceed.

- [ ] **Step 3: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/verdicts/refutation.md
git commit -m "audit: gate 1, adversarial refutation

Every candidate faced an agent trying to disprove it, defaulting to refuted under
uncertainty. SURVIVED-CORRECTED exists because a real defect with an invented
mechanism is not good enough: three findings in the review of this code's own PR
were real while their stated mechanisms were fiction, and a wrong reason gets
copied into the next fix."
```

---

### Task 8: Gate 2 — reproduction

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/repro/F-<n>.py` (one per attempt)
- Create: `specs/20260815-215106-analyze-audio-audit/verdicts/reproduction.md`

**Interfaces:**
- Consumes: candidates whose refutation verdict is `SURVIVED` or `SURVIVED-CORRECTED`.
- Produces: `DEMONSTRATED` / `LATENT` per candidate.

- [ ] **Step 1: Dispatch reproduction agents in batches**

For each surviving candidate, dispatch an agent (model: sonnet) with this prompt:

> Working directory: `/Users/satra/software/sensein/senselab`.
>
> Write a script that **demonstrates** this defect by executing real code, and run it.
>
> Claim `F-<n>`, at `<location>`: `<defect>`
> Failure scenario: `<failure>`
>
> Rules:
> - The script goes in `specs/20260815-215106-analyze-audio-audit/repro/F-<n>.py`. It may import
>   from `src/senselab` but must not modify anything under `src/`.
> - Run it with `uv run --no-sync python specs/20260815-215106-analyze-audio-audit/repro/F-<n>.py`
>   **from the repository root** — never with a working directory inside `audio_analysis/`, where
>   `types.py` shadows the stdlib and every import fails with a misleading circular-import error.
> - **Load no model and download nothing.** Construct inputs directly or stub at the model
>   boundary. A reproduction that pulls a checkpoint is not acceptable.
> - The script must print `DEFECT REPRODUCED` on success, with the wrong value beside the right
>   one, and exit non-zero if it cannot.
>
> If you **cannot** reproduce it without a resource you do not have — a child-voice corpus, a GPU,
> a cold HF cache, a multi-node allocation — do not fake it. Report `LATENT` and state the exact
> experiment: what data, what hardware, what comparison, what result would confirm or refute.
>
> Return exactly:
>
> ```
> ### F-<n>
> - outcome: DEMONSTRATED | LATENT
> - script: repro/F-<n>.py | none
> - observed: <the wrong value, and what it should be>
> - experiment: <for LATENT only: the measurement that would settle it>
> ```
>
> Under 15 lines.

- [ ] **Step 2: Verify every DEMONSTRATED script actually runs**

```bash
for f in specs/20260815-215106-analyze-audio-audit/repro/F-*.py; do
  echo "== $f"
  uv run --no-sync python "$f" 2>&1 | tail -3
done
```

Every script claiming `DEMONSTRATED` must print `DEFECT REPRODUCED` when run from the repository
root. One that does not is not demonstrated — downgrade it to `LATENT` with a note, or drop it.

- [ ] **Step 3: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/repro/ \
        specs/20260815-215106-analyze-audio-audit/verdicts/reproduction.md
git commit -m "audit: gate 2, reproduction

Findings that execute wrong are DEMONSTRATED. Findings that survive refutation
but need a child-voice corpus, a cold cache or a multi-node run are LATENT, and
each carries the experiment that would settle it -- requiring reproduction of
everything would have discarded exactly the class the lifespan work needs."
```

---

### Task 9: The register, the summary, and the prose migration

**Files:**
- Create: `specs/20260815-215106-analyze-audio-audit/register.md`
- Create: `specs/20260815-215106-analyze-audio-audit/summary.md`
- Create: `specs/20260815-215106-analyze-audio-audit/prose-migration.md`

**Interfaces:**
- Consumes: `deduped.md`, both verdict files, `measure.py` output.
- Produces: the deliverable.

- [ ] **Step 1: Write `register.md`**

One table, sorted by tier then severity. Columns exactly:

`id | layer | location | defect | failure | tier | severity | graph_implication`

Tier assignment is mechanical, not a judgement:

| Refutation | Reproduction | Tier |
| --- | --- | --- |
| SURVIVED / SURVIVED-CORRECTED | DEMONSTRATED | **demonstrated** |
| SURVIVED / SURVIVED-CORRECTED | LATENT | **verified-latent** |
| REFUTED | — | **unverified concern** (separate section, below the table) |

`graph_implication` takes exactly one of: `consumed` (the triage workflow reads this signal),
`routed-around` (the workflow can avoid it), `irrelevant`. This field is why the register exists —
a defect in a signal the graph never reads is a different priority from one the review flag
depends on.

Every `verified-latent` row must carry its experiment. Add a `## Refuted` section listing what did
*not* survive and why, so the same candidate is not re-raised later.

- [ ] **Step 2: Write `summary.md`**

Lead with the cross-sweep patterns from `deduped.md`, not with individual findings — a class
appearing in six places is the actionable unit. Then, in order:

1. The layer measurements from `measure.py`, and what the 11,721 lines of task-free computation
   inside a workflow package imply for reuse.
2. The prose ratio, and how much of it Sweep A classified as `restates-code`.
3. What the register implies for the graph: which signals it can build on, which to route around.
4. What it implies for each deferred concern — model pluggability, the 1-vs-more speaker
   decision, new extraction models, lifespan validation — with the finding ids that scope each.

- [ ] **Step 3: Write `prose-migration.md`**

Every Sweep A `rationale-to-migrate` candidate, grouped by destination module-group summary, with
the prose quoted verbatim and its source `file:line`. This is the document that makes deleting the
in-code copy safe later — nothing is migrated in this task, only staged.

- [ ] **Step 4: Verify the register is complete and internally consistent**

```bash
uv run --no-sync python - <<'PY'
import pathlib, re
d = pathlib.Path("specs/20260815-215106-analyze-audio-audit")
dedup = {m.group(1) for m in re.finditer(r"^### (F-\S+)", (d/"candidates/deduped.md").read_text(), re.M)}
reg = (d/"register.md").read_text()
missing = [i for i in sorted(dedup) if i not in reg]
print("candidates:", len(dedup), "| missing from register:", missing or "none")
latent = re.findall(r"^\|.*verified-latent.*$", reg, re.M)
no_exp = [r for r in latent if "experiment" not in r.lower() and "→" not in r]
print("verified-latent rows:", len(latent), "| without an experiment:", len(no_exp))
PY
```

Expected: no candidate missing from the register (every one is either a row or in `## Refuted`),
and no verified-latent row lacking an experiment.

- [ ] **Step 5: Confirm no source file changed across the whole audit**

```bash
git diff --stat origin/alpha...HEAD -- src/
```

Expected: **empty**. The audit's central constraint is that it changes no code; this is the check
that proves it. Any output here is a violation to be reverted before the deliverable is committed.

- [ ] **Step 6: Commit**

```bash
git add specs/20260815-215106-analyze-audio-audit/register.md \
        specs/20260815-215106-analyze-audio-audit/summary.md \
        specs/20260815-215106-analyze-audio-audit/prose-migration.md
git commit -m "audit: the register, the summary, and the staged prose migration

Findings by tier: demonstrated (reproduced by executing code), verified-latent
(survived refutation, needs a corpus or hardware we lack, carries the experiment
that would settle it), and refuted (recorded so they are not re-raised).

Every row says whether the triage workflow consumes that signal, can route
around it, or is unaffected -- the register exists to inform that design, not to
be a fix list. No file under src/ was modified."
```

---

## Self-review

**Spec coverage.** Every section of `design.md` maps to a task: the four sweeps → Tasks 2-5;
the two verification gates → Tasks 7-8; the register format including `graph_implication` →
Task 9 Step 1; the three deliverables → Task 9; the measured layer table → Task 1. Dedupe (Task 6)
is not named in the spec but is required by it — four sweeps over an overlapping surface cannot
produce one register without it.

**Placeholder scan.** No TBD/TODO. Every dispatch prompt is verbatim and complete. Every
verification step is a runnable command with a stated expected result. The one deliberate
"discover rather than assume" step is Task 8's per-candidate reproduction, where the script cannot
be written in advance because it depends on the finding.

**Type consistency.** Candidate ids flow `A-n`/`B-n`/`C-n`/`D-n` → `F-n` at dedupe (Task 6) and
stay `F-n` through both gates and the register. Field names are consistent across prompts:
`location`, `defect`, `failure`, `experiment`. The tier vocabulary (`demonstrated`,
`verified-latent`, `unverified concern`) matches `design.md` exactly.

**One risk, flagged not hidden.** Tasks 7 and 8 dispatch one agent per candidate, so their cost
scales with what the sweeps find. If the sweeps return more than ~40 candidates, batch the
refutation by grouping related candidates into one agent rather than dispatching 40+ — but never
group a candidate with the agent that raised it, since the refuter's independence is the point of
the gate.
