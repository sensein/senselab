# Reproduce driver: pinned code, locked venvs, sandboxed steps

## Why

A lab result usually lives in someone else's repository, on a branch, behind an environment nobody
wrote down. Rerunning several such results at once fails for two reasons:

1. **Pins conflict.** Two analyses can pin different torch, numpy or CUDA-extension versions, so one
   environment cannot serve both.
2. **Nothing is pinned tightly enough.** Even where an analysis runs, a later rerun silently picks up
   newer packages, a moved branch or a changed input.

`senselab.utils.reproduce` handles both. It keeps the subprocess-venv idea already used by the
backends (one venv per component, called across a process boundary). On top of that it adds:

- hashed locks
- code pinned to a commit
- a sandbox that makes "offline" and "read-only data" true at the OS level
- a content-addressed result cache, so that an unchanged step is never rerun

## What it does not use

- **`ensure_venv`.** It picks the torch index by probing the host when it builds, and names the venv
  by that probe. A venv built on a login node is then the wrong variant for a GPU job, and old torch
  versions with no matching index fall back to CPU silently.
- **`_clean_subprocess_env`.** It passes the caller's whole environment through.

The lock header format and the hashed-compile recipe are shared with `venv_lock`.

## The unit: a step

A **manifest** lists steps. Each step names:

- a repository and a 40-hex commit, plus whether submodules are needed
- a kind: python, notebook, r or ir_notebook
- an env spec (below)
- an argv template, with placeholders `{out}`, `{tmp}`, `{checkout}`, `{venv}`, `{cache}`, `{root:<name>}` and `{up:<step>}`
- inputs and staged files, both drawn from named data roots
- patches and an adapter
- `needs`, the upstream steps
- resources, and the expected metrics

A **profile** is supplied by the caller and kept outside this repository. It names:

- the data roots
- the results, runs and job-log roots
- the keyed environment map
- the partitions
- the R module
- a deny-pattern file

Nothing site-specific lives in this package.

## Environments

Every env is built from hashed locks, ahead of time, by `prefetch`. A run never installs anything.
There are up to four locks:

1. **`torch.lock`.** The torch family and the exact CUDA-variant package names.
   - When the source is a repository `uv.lock`, it is derived from that lock's own entries as
     URL-pinned, hashed lines. Recompiling against the PyTorch index can give a different file
     for the same version.
   - Otherwise it is compiled against an explicit index URL. The compile passes
     `--exclude-newer`, plus `--exclude-newer-package <name>=false` for every CUDA-variant package,
     because the index publishes them with no upload time.
2. **`body.lock`.** Everything else.
   - From a repository `uv.lock`, via `uv export --frozen --no-emit-project --no-dev`, with the
     torch family and the sdist-only packages excluded by name.
   - Otherwise compiled from `requirements.in` with `--universal --generate-hashes --exclude-newer`.
     A uv config file may give static metadata for sdists, so locking never runs their `setup.py`.
3. **`wheels.lock`.** One line per sdist-only package:
   - inputs: the sdist URL and its sha256, the hashed build-env lock, patches, build variables,
     CUDA module and arch list
   - output: the wheel sha256

   The wheel is built once in a separate build venv, recorded, and reused afterwards. CUDA builds
   are not bit-reproducible, so a rebuild only happens when explicitly asked for, and it changes the
   lock.
4. **Overlay locks** (test tools, notebook tools). These are installed together with the base locks
   into a separate *derived* venv. The base venv stays exactly its locks, and `uv pip check` sees
   the whole derived venv. It cannot see `--target` directories.

The install runs four stages, each `--no-deps`. Stages 1, 2 and 4 also use `--require-hashes`; stage 3 installs a wheel only after the driver has checked its sha256 against `wheels.lock`:

1. torch
2. body (`--only-binary :all:`)
3. the recorded wheels
4. the overlay, into the derived venv

**Integrity is checked before anything is installed:**

- Every lock file's sha256 must match the manifest. This is the tamper guard:
  `--require-hashes` alone accepts a lock repinned to another real release.
- Requirements-derived bodies must match their recorded header digest.
- Overlay pins must agree with the base pins.
- The torch pins must agree with the resolve the body came from.

How `locks.check_integrity` reads those rules:

- Every manifest path is relative with no `..`, and must also resolve inside the manifest
  repository, so a symlink cannot point a hashed entry elsewhere. Patches are `{file, sha256}`
  like locks: an unhashed patch would let a changed diff through every other check.
- The header digest (`requirements_digest_for`) covers the requirement lines, Python, CUDA cap,
  `exclude_newer`, the config and overrides sha256s and `extra_no_emit`. It leaves out the backend
  venvs' IPC packages, because a reproduce env holds only what the original env held. Lines are
  normalised (continuations joined, comments stripped, whitespace collapsed) so that layout edits do
  not change the digest; a repeated line is refused rather than collapsed.
- Lock parsing skips `--index-url`, `--extra-index-url` and `--find-links` lines, which an
  index-compiled torch lock carries, and refuses every other option (`-e`, `-r`, `-c`, …), since
  those would install something the lock does not pin. `===` is refused, and a URL pin must name a
  wheel of its own project.
- Overlays fail closed. A name the base locks pin more than once (marker variants) may not be pinned
  by an overlay at all, because comparing markers would need marker evaluation. Otherwise pin,
  marker text and hash set must be identical. Overlays may not pin `sdist_allow` or `extra_no_emit`
  names, nor torch-family names that `torch.lock` lacks: those come from other install stages.
- Torch versions drop a `+local` tag only when one side lacks it (a resolve may spell `2.8.0`
  where the wheel says `2.8.0+cu128`); when both carry one, `cu126` is not `cu128`. Against a
  repository `uv.lock` each torch-lock entry needs hashes, and they must be among that version's
  wheel hashes: the same version on another index can be a different file. The check runs both
  ways: a torch-family name in the reference that `torch.lock` lacks is refused. Against a
  repository `uv.lock` the family is `torch`/`torchvision`/`torchaudio` from `torch.lock` plus the
  `nvidia-`, `triton`, `pytorch-triton` and `cuda-` named packages in the transitive closure of their
  `uv.lock` dependencies, over every marker. A name rule alone would demand packages that an optional
  extra pulls in for another reason (a repository lock carried `cuda-bindings` and
  `cuda-pathfinder` from a toolkit extra, which the correct torch lock lacks). The same closure
  decides which names an overlay may not pin. Against a first resolve, which holds only selected
  packages, the name rule is used.
- The repository `uv.lock` lives in the checkout, not the manifest repository, so the caller passes
  its path.
- `check_integrity` returns the sha256 of each file it read. The bytes can change after the check,
  so the installer copies each file into `{tmp}`, checks the copy against that mapping and installs
  from the copy. Step patches are checked the same way, with `verify_files`, when the checkout is
  patched.

- The header digest refuses `-r`, `-c`, `-e` and other non-index options in `requirements.in`: the
  files they name are not hashed, so editing one would change the compile without changing the
  digest. Index options are digested with the requirement lines.
- Each file is read once. The header, pin and torch checks parse the bytes that were hashed, so a
  file swapped between the hash and a re-read cannot pass.

How `lockgen` produces the locks:

- Each builder runs one uv command with relative paths from a fixed working directory (the manifest
  repository, or the checkout for `uv export`) and `--no-header`, then writes its own
  `# reproduce lock` header with the command. The lock bytes therefore do not depend on where the
  manifest repository sits on disk.
- A manifest `exclude_newer` date becomes `<date>T23:59:59Z`. uv reads a bare date in the local time
  zone, which would make the lock depend on the host.
- The torch pins for an index compile are the torch-family pins of the first resolve, written to a
  `.in` file beside `torch.lock`, so the recorded command can be rerun. Every non-root family name
  (`nvidia-*`, `triton`, …) gets `--exclude-newer-package <name>=false`, as the spec says; the spike
  needed it for the `nvidia-*` names, and on exact pins it is harmless for `triton`.
- A `uv.lock`-derived torch lock and a uv-export body both use the `uv.lock` torch closure of the
  roots the env installs, the same set the integrity check uses, so the two cannot disagree. Each
  name gets the single linux x86_64 manylinux wheel for the env's `cpXY` (or a `py3-none` wheel);
  zero or several matches is an error.
- Overlays leave out the names `torch.lock` holds and the `sdist_allow`/`extra_no_emit` names: the
  base venv provides or deliberately lacks them, and the integrity check refuses an overlay that pins
  them differently. Notebook tools compile with `torch.lock` and `body.lock` as `--constraints`;
  their `--index-url` lines do not change the compile index.

**The venv name** is `repro-<analysis>-<env_key8>`. `env_key` is derived from inputs only:

- the lock sha256s
- the exact interpreter version
- the declared maximum CUDA version
- the sdist build inputs

Wheel sha256s are outputs, so a rebuild never renames a venv.

**R steps are the one exception to "locked":**

- Packages are installed by `prefetch` from a dated CRAN snapshot into a step-private library,
  under a named module.
- `sessionInfo()` is recorded, and its digest is part of the step key.
- `prefetch` fails if the installed versions differ from the declared ones.
- A run fails if its `sessionInfo()` differs from the one captured at prefetch.
- The module's `PATH` and `LD_LIBRARY_PATH` are captured at prefetch and keyed.

## Sandbox

Every step runs as
`unshare -rnm --propagation private sh -c 'set -e; <mounts>; /usr/sbin/ip link set lo up; exec "$@"' sh <argv>`.
With `set -e`, any failed mount aborts the step before it runs, so read-only data cannot fail open.
The literal `sh` is `$0`; without it the first word of argv is dropped.

**Mounts:**

1. Each data root is bound to itself.
2. Each writable directory under a data root is bound next. That includes the step's own output
   dir, and the order matters: binding it after the parent is remounted read-only gives a
   read-only directory.
3. The data roots are remounted read-only.
4. The checkout and the venv are also mounted read-only.

**Network:** there is none.

- Loopback is raised, because notebook kernels talk over 127.0.0.1.
- A canary TCP connect at step start must fail.
- If unprivileged user namespaces are unavailable, the driver refuses to run, except in a test-only
  mode that marks the run unreportable.

**Environment:** it is built from an allowlist, never inherited.

- **Keyed:** the profile map, the offline variables, `LANG`, `TZ`, and every path variable in
  *template* form (`PATH`, `PYTHONPATH`, the Jupyter dirs, `R_LIBS`, the cache dirs,
  `CUDA_CACHE_PATH`).
- **Recorded only:** volatile values (job IDs, device lists, temp dirs, expanded paths).
- **Always set:** `REPRO_OUT` and `REPRO_TMP`.
- **Refused:** keys that look like secrets.

stdout and stderr always go to files under the step's output dir, never to the caller's terminal.

## Checkouts and patches

**Mirrors:** each repository and its submodules are mirrored bare into the cache.

**Worktrees:** one detached worktree per (commit, patch-set sha256), created at prefetch.

- Notebook outputs are cleared as plain JSON and committed, then each patch is applied and
  committed, all on a throwaway ref in the mirror.
- The resulting tree sha is recorded. A run checks that the worktree still has that tree and a
  clean status.
- Patches may not touch notebook outputs, and they are scanned with the profile's deny patterns.
- Fixes to someone else's analysis are therefore patch files with recorded hashes, never commits
  to their repository.

## Results and cache

**Step key:** sha256 over these fields, and nothing volatile:

- code commit and submodule SHAs
- `env_key`
- `wheels.lock` and overlay sha256s
- keyed env digest
- argv template, cwd
- patch-set, adapter and input-set digests
- weight digests, R session digest
- expectations, replicate index
- upstream keys

**Result dir:** each step writes to `<results_root>/<step_key>/`. A step whose result dir holds a
completed summary with exit 0 is skipped, and its result dir is linked into the dated run. Downstream steps
therefore read the same files on any day.

**Input-set digest:** sha256 over the sorted (sha256(relative path), sha256(content), size)
triples, so it reveals no file names. It is taken before and after the step, and a change fails
the step.

## Metrics, expectations, provenance

**Adapters.** Each step's adapter turns the analysis's own outputs into `metrics.json`, from a fixed
vocabulary:

- headline metrics, and per-class values
- confusion counts
- named counts
- degrees of freedom

Unknown keys and non-numeric values are rejected. Count fields below a minimum cell size are
suppressed.

**Expectations.** Only the driver writes `target`, `tolerance`, `delta` and `pass`, by evaluating the
step's expectations: a literal value, an upstream step's metric, or a number read from a
file under a data root. A failed expectation is a non-zero exit, so a
dependent job never starts.

**`summary.json`** holds:

- the exit code
- the exception class and the last frame, as a checkout-relative `file:line`
- fixed error codes
- the interpreter-prefix check
- the torch version
- the canary result

It holds no messages, and no paths other than checkout-relative code frames.

**Provenance** is recorded with `prov_store` and exported as BIDS BEP028:

- the code commit, as a software agent
- the venv and the host as environments
- inputs and outputs as `artifact` entities that carry a checksum without a path

## Commands

`python -m senselab.utils.reproduce` takes these subcommands:

- `lock`: compile locks
- `prefetch`: mirrors, worktrees, venvs, wheels, R libraries. `--sources-only` creates worktrees only.
- `build-one`: one sdist build, as a batch job
- `run`: run steps locally, in DAG order
- `submit`: one batch job per step, chained with `afterok`
- `lint`: check manifests and patches
- `status`: per step: key, executed, skipped or never-ran, exit, codes
