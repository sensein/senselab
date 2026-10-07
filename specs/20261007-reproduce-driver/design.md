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
  `.in` file beside `torch.lock`, so the recorded command can be rerun. Only `nvidia-*` names get
  `--exclude-newer-package <name>=false`: the spike found that the PyTorch index links those wheels
  with no upload time, so `--exclude-newer` drops them all, while `torch`, `torchaudio` and `triton`
  carry upload times and pass the date filter. Exempting more would only weaken the date bound.
- A `uv.lock`-derived torch lock and a uv-export body both use the `uv.lock` torch closure of the
  same roots, the set the integrity check uses, so the two cannot disagree. The torch lock records
  its `roots` and the `uv.lock` sha256 in its header; the export reads the roots back from
  `torch.lock` and refuses one with none. Each name gets the single linux x86_64 manylinux wheel for
  the env's `cpXY` (or a `py3-none` wheel); zero or several matches is an error.
- The integrity check refuses a body entry that `torch.lock` names or that belongs to the torch
  family (the `uv.lock` closure, or the name rule against a first resolve), so a package cannot come
  from two install stages.
- Overlays leave out every name in `torch.lock` and `body.lock` and the `sdist_allow`/`extra_no_emit`
  names, so an overlay holds only names the base lacks. Keeping shared names failed the integrity
  check on real data: a universal compile emits every wheel hash where the base holds a subset, and
  its markers differ (`pycparser`, `cffi`, `markupsafe` against a uv-export body; `decorator` against
  a requirements body). Notebook tools compile with `torch.lock` and `body.lock` as `--constraints`,
  which still bound the versions of the names left out.

How `build` produces and reuses sdist wheels:

- The build venv is separate from the step venv, so setuptools, wheel and pip (chumpy's `setup.py`
  imports pip) never reach the step venv unless its own locks list them. It is made from the
  hashed `build_env` lock, plus the env's torch lock when the build imports torch (mmcv-full).
  Locks and patches are copied into the work directory, the copies are hashed, and only the copies
  are used, as in the integrity check. The sdist's bytes are read once, hashed and unpacked from
  memory, so nothing can change between the check and the unpack.
- Every subprocess gets an **allowlisted environment**, never the caller's: `PATH=/usr/bin:/bin`
  (the build adds the build venv's `bin`), `HOME` and `TMPDIR` under the work directory,
  `LANG`/`LC_ALL=C.UTF-8`, `SSL_CERT_FILE`, and the caller's `UV_CACHE_DIR` and
  `UV_PYTHON_INSTALL_DIR` (asked of uv when unset, because a private `HOME` would otherwise move
  uv's cache and interpreters). The build adds the declared `build_env_vars`, which may not
  replace any of these, and the fixed ones below. A `PYTHONPATH`, `CFLAGS`, `LD_LIBRARY_PATH`,
  `CUDA_HOME` or `PIP_*`/`UV_*` setting in the caller's shell changes the wheel without changing
  any recorded input. The slow test plants a broken `setuptools` on `PYTHONPATH` and a `CFLAGS`;
  as a control, letting that `PYTHONPATH` through makes the same build fail.
- Every uv call passes `--no-config`, so a user or project `uv.toml` cannot change an index or a
  build setting behind the lock.
- **No network during the build.** The sdist fetch and the build-venv installs happen first; the
  `uv build` then runs with `UV_OFFLINE=1`, `PIP_NO_INDEX=1` and `http(s)_proxy` pointed at
  `127.0.0.1:9`. This stops cooperative tools only. Once B5's sandbox is merged, B7 also runs the
  build inside the network namespace, which stops everything.
- The build runs with `SOURCE_DATE_EPOCH=315532800` (zip dates of 1980) and a fixed umask of
  `007`. Tar and zip members unpack with their archive mode under that umask. File modes are
  stored in the wheel: the first slow-test build, unpacked with tarfile's `data` filter, differed
  from the spike's wheel only in `top_level.txt` (0644 against 0660). The spike unpacked with `tar`
  under umask 007. With the fixed umask the wheel is bit-identical to the spike's under umask 022
  and 007 alike. `007` was chosen because it reproduces the wheels the spike already recorded.
- Patches are `-p1` unified diffs against the unpacked sdist root, applied with GNU `patch
  --fuzz=0`, before the build venv is created. The spike's mmcv patch carries `srcp/…` paths and
  has to be regenerated with `a/`, `b/` paths when its manifest row is written.
- **Modules load without dotfiles:** `bash --noprofile --norc -c '. /etc/profile.d/modules.sh &&
  module load <m…> && exec "$@"'`. `nvcc` is required only when the build is a CUDA build, that is,
  when `torch_cuda_arch_list` is set or a `cuda/*` module is listed. Then the script fails unless
  `nvcc` is on `PATH`, and exports `CUDA_HOME` derived from it, as in the spike's mmcv job.
  The spike's mmcv build got gcc 12.2 from the login shell's default modules. With dotfiles and the
  caller's `PATH` gone, the compiler is `/usr/bin/gcc` (8.5 here) unless the manifest loads one,
  e.g. `build_modules: [gcc/12.2.0, cuda/12.9.1]`. Each wheel gets a `<wheel>.build-info.json`
  recording the uv, gcc and nvcc versions, the modules and `python_exact`, so the toolchain is
  visible either way.
- **Planning amendment: `build_modules` replaces `cuda_module`.** Spec §2.3.3 named one optional
  `cuda_module` per sdist. A build needs a compiler module as well as a CUDA module, so `SdistSpec`
  carries `build_modules: [name…]`, loaded in order. Each entry is one module name with no shell
  syntax, and entries are unique. The list, in order, goes into `build_config_sha256` and so into
  `env_key`'s sdist inputs.
- **Content-addressed wheelhouse:** `<wheelhouse>/<wheel sha256>/<wheel file>`, never overwritten.
  The wheel is written to a temp file in that directory and hard-linked into place, so no partial
  file is ever visible and a file that appears meanwhile is not replaced. Other bytes already at a
  wheel's address are an error. A rebuild adds a new directory and
  repoints the line; the old wheel stays, so an earlier `wheels.lock` still resolves.
- `wheels.lock` has nine columns: `name version sdist_url sdist_sha256 build_env_sha256
  patch_sha256s build_config_sha256 wheel_path wheel_sha256`. `wheel_path` is relative to the
  wheelhouse. The spike's seven-column file named the build lock by file name. This one records
  the build lock's sha256 and every patch sha256, and folds `python_exact`, the torch lock sha256,
  the build variables, the `build_modules` list and the arch list into `build_config_sha256`, since an arch
  list may contain spaces and the variables are a mapping.
- Reuse compares every input column with the manifest. A line recorded for other inputs, a missing
  wheel, or a wheel whose sha256 differs is an error that names `--rebuild`, never a silent rebuild:
  CUDA builds are not bit-reproducible, so a rebuild is a new wheel and changes the lock.
- **`wheels.lock` contract with the manifest.** Readers take a shared `flock` on the sidecar
  `wheels.lock.lock` and writers an exclusive one, because several build jobs of one env can
  finish together. Writes go to a temp file followed by `os.replace`. Each read is one read of the
  bytes, checked against the manifest's `wheels.lock_sha256` when the manifest has one.
  `record_wheel` and `ensure_wheel` return the new file sha256. The caller (`prefetch`, B13) writes
  it into the manifest's `wheels.lock_sha256`; until it does, the next integrity check refuses the
  env. The sidecar is not content; the manifest repository should list it in `.gitignore`.
- B7 installs a recorded wheel through `stage_wheel`. That copies it into the step's `{tmp}`,
  checks the copy's sha256, and installs from the copy.
- The batch job starts from an empty environment (`--export=NONE`). It exports `PATH`,
  `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `SENSELAB_REPRO_CACHE`, the driver interpreter, and a
  `TMPDIR` under `/dev/shm/<job>` that is removed on exit. It also exports `REPRO_UV`, the uv path
  resolved to an absolute path when the script is rendered. uv usually lives in the user's
  `~/.local/bin`, which a minimal `PATH` lacks; `find_uv` reads `REPRO_UV` first, and every build
  subprocess runs that absolute uv. The script loads no modules: under `set -u`, sourcing
  `/etc/profile.d/modules.sh` dies on an unset `MODULEPATH`. `build-one` loads `build_modules`
  itself, in the wrapper above. A test runs the rendered prologue under `env -i` with `set -u`. `build-one` builds in a temporary work
  directory under that `TMPDIR`, removed afterwards.

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

How `env` installs (B7):

- **Copies, not originals.** `check_integrity` returns the sha256 of the bytes it hashed;
  `prefetch_env` copies every verified file into a private temp dir, re-hashes each copy against
  that mapping and installs only from the copies. A test swaps the body lock between the check and
  the copy and expects a refusal with no install. Stage 3 wheels go through `stage_wheel` the same
  way, so neither the manifest repository nor the wheelhouse is read by uv.
- **uv runs with `build.base_environment`** (system `PATH`, private `HOME`/`TMPDIR`, the uv cache
  and Python dirs, the CA bundle) and `--no-config`, so a caller's `UV_INDEX_URL`, `PYTHONPATH` or
  `uv.toml` cannot change an install. The test plants both and checks the env keys.
- **Stage argv.** Torch: `--no-deps --require-hashes`, with `--index-url <torch.index_url>` only
  when the lock has `==` pins (`build.torch_index_url`); URL-pinned locks need no index. Body and
  overlays: `--no-deps --require-hashes --only-binary :all: --index-url <body.index_url>`; the
  overlays are compiled against the body index, and binaries-only keeps a stray sdist (and its
  setuptools) out of the step venv. Wheels: `--no-deps <copy>`.
- **The torch lock goes to every sdist build** when the env has one. `SdistSpec` has no "needs
  torch" field and a build that silently lacks torch fails late or, worse, builds without its
  extension; the cost is a temporary torch install in the build venv.
- **Freeze equals locks.** After installing, the parsed `uv pip freeze` must equal the lock entries
  whose markers hold for the env's Python on this host, plus the stage 3 wheels, with `+cuXXX`
  dropped on both sides. A mismatch fails the prefetch and leaves no marker. `installed.json` keeps
  the raw freeze lines (`name @ file://…` included) and the parsed map.
- **Completion and reuse.** A venv is complete when `.reproduce-venv.json` names its `env_key` and
  overlay sha256; it is written last. A complete venv is reused without any uv call; anything else
  at that path is removed and rebuilt, under a `flock` on `<venv>.lock`. Wheels are only reused or
  built when a venv actually needs building. Two overlays with identical bytes share one derived
  venv, since the name is the overlay's sha256.
- **pip check.** Each `The package \`X\` …` line becomes `(normalised X, kind, detail)`, where detail
  is the rest of the line without backticks; `requires …` lines are `requirement`, lines mentioning
  the platform are `platform`, anything else is `other` and can never be declared. The comparison
  is exact both ways: an undeclared violation and a declared one that did not occur both fail.
- **Smoke catches what pip check cannot.** The slow test's sdist wheel (python_speech_features 0.6)
  declares no requirements but imports scipy, which no lock holds: `uv pip check` is clean and the
  sandboxed import fails. Smoke runs one `python -c 'import m'` per module through `sandbox_argv`,
  the venv bound read-only, offline.
- **R prefetch** loads the module with `bash --noprofile --norc -c '. /etc/profile.d/modules.sh &&
  module load "$1" …'` (the plan's `bash -lc` read dotfiles, B6), with an allowlisted env and no
  `set -u` (the init script fails under it). `Rscript --vanilla` keeps `Renviron.site` and
  `Rprofile.site` from resetting `R_LIBS_SITE` or the repository; B12 should run R steps the same
  way. The R script writes plain text (installed TSV, `.libPaths()`, `sessionInfo()`, R version),
  and Python writes `SESSION.json` with sorted keys, so its sha256 is stable and the script needs no
  JSON package. `SESSION.json` is written last and only when every declared version is installed
  and the library is first on `.libPaths()`; it also marks the library complete for reuse.
- **Merge order** (`r_env_vars`): `PATH` is the caller's prefix (`{venv}/bin`), then the module's
  entries, then a suffix; `LD_LIBRARY_PATH` is the module's entries, then the native lib dirs, each
  entry kept at its first position.

## Sandbox

Every step runs as
`/usr/bin/unshare -rnm --pid --fork --kill-child --mount-proc --propagation private /bin/sh -c 'set -e; <mounts>; /usr/sbin/ip link set lo up; exec /usr/bin/setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs "$@"' sh <argv>`.
With `set -e`, any failed mount aborts the step before it runs, so read-only data cannot fail open.
The literal `sh` is `$0`; without it the first word of argv is dropped.

The step runs with every capability dropped. `unshare -r` makes the script root in its namespace,
with full capabilities there, and a step that kept them could
`mount -o remount,bind,rw` a data root, or `umount` the read-only binds, and then write to the
data. The B5 review did exactly this on scratch NFS. `setpriv` clears the bounding, inheritable
and ambient sets and sets `no_new_privs` before it execs the step, so remount and umount fail with
"permission denied". The step is still uid 0 in the namespace (the caller's uid outside), so
ordinary writes to `{out}` keep working. A missing `setpriv` fails the exec, and the step never
runs.

**Processes:** the step also gets its own PID namespace.

- `--fork` makes the script, and then the step it execs, pid 1.
- `--mount-proc` mounts a fresh `/proc` before the script runs, so `/proc` lists only the
  sandbox's processes. It does not conflict with the bind mounts, which come after it in the same
  mount namespace. After the capability drop the step cannot remount `/proc` either.
- A host process is invisible to the step, so `kill <host pid>` finds no such process.
- `--kill-child` SIGKILLs the step when `unshare` itself dies, e.g. on a driver timeout. Without it,
  the namespace's pid 1 would outlive the killed `unshare`.
- Verified 2026-10-07: pid 1, /proc shows only sandbox pids, a host `sleep` survives `kill`, and a
  one-cell `python -m nbconvert --execute` notebook runs (kernel over loopback).

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
- **Recorded only:** volatile values (four Slurm job variables, the device list, temp dirs,
  expanded paths).
- **Always set:** `REPRO_OUT` and `REPRO_TMP`.
- **Refused:** keys that look like secrets.

stdout and stderr always go to files under the step's output dir, never to the caller's terminal.

How `sandbox` reads those rules:

- Paths are resolved (`realpath`) before the script is written, so the nesting checks see what
  `mount` will see. Two layouts are refused instead of mounted:
  - a read-only path under (or equal to) a writable dir, because the non-recursive rw bind would
    hide the read-only mount beneath it;
  - a writable dir under the checkout or venv, because their read-only bind comes last and would
    hide it.
- A writable dir is bound only when it lies under a data root. Elsewhere it is writable already.
- The checkout and venv are bound after the roots are remounted. A checkout under a data root is
  then read-only either way.
- Every tool the sandbox itself runs is called by absolute path: `/usr/bin/unshare`, `/bin/sh`,
  `/usr/bin/mount`, `/usr/sbin/ip` and `/usr/bin/setpriv`. None is looked up through the step's
  `PATH`, so a step's venv or native dirs cannot shadow them.
- Mount lists (roots, writable dirs, checkout/venv) are written shallow-first. A nested root given
  before its parent would otherwise be hidden by the parent's later bind, and its remount would fail
  (exit 32).
- **Native tools are per step and opt-in.**
  - The profile's `native` map names entries, each with explicit `bin_dirs` and `lib_dirs`
    (absolute, no `:`).
  - A step selects entries by name in `resources.native`. `spec.check_step_profile` refuses a name
    the profile lacks, and `build_env` calls it.
  - Only the selected entries' bin dirs go on `PATH`, after `{venv}/bin:/usr/bin:/bin`, in
    selection order.
  - Only their lib dirs go into a keyed `LD_LIBRARY_PATH`. A step that selects none gets
    `PATH={venv}/bin:/usr/bin:/bin` and no `LD_LIBRARY_PATH`.
  - `LD_LIBRARY_PATH` is never taken from the profile map.
  - An earlier rule put every native `bin` dir's sibling `lib` on every step's `LD_LIBRARY_PATH`.
    The B5 re-review found that a conda-installed FFmpeg prefix's `lib/` holds 961 libraries,
    including `libstdc++`, `libgomp`, `libssl` and `libz`. They shadow the system copies for git,
    curl and R.
  - The intended way to expose FFmpeg is therefore a **curated lib dir**: a directory of symlinks to
    only the needed `.so` files (`libav*`, `libsw*`, `libpostproc*`), named in that entry's
    `lib_dirs`.
- **For B7, R steps:** the R module's `LD_LIBRARY_PATH` is captured at prefetch. When a step has
  both, the order is the module's entries first, then the selected native lib dirs. This module
  merges nothing yet; B7 implements the merge.
- Keyed templates: `PYTHONPATH` (the step's entries; an entry containing `:` is refused, since it would split), `R_LIBS`
  (`r_lib`, as the plan's B7 says, with `R_LIBS_SITE` keyed as an empty constant so no site library
  is seen), `JUPYTER_PATH`
  (`{venv}/share/jupyter`), `JUPYTER_DATA_DIR`/`JUPYTER_CONFIG_DIR`/`XDG_CACHE_HOME` (under
  `{tmp}`), `HF_HOME`/`TORCH_HOME` (under `{cache}`), `CUDA_CACHE_PATH` (`{out}/cuda-cache`, as spec
  §15.6 says), `REPRO_OUT`/`REPRO_TMP`.
- Keyed constants: `LANG=C.UTF-8`, `TZ=UTC`, the three HF offline variables, every proxy variable
  (both cases) set to `http://127.0.0.1:9`, and `no_proxy`/`NO_PROXY` set empty.
- Recorded only: `TMPDIR`, `HOME`, `IPYTHONDIR`, `JUPYTER_RUNTIME_DIR` (all under `{tmp}`, expanded),
  plus `SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`, `SLURM_CPUS_PER_TASK` and
  `CUDA_VISIBLE_DEVICES`. These are the only names read from the caller's environment. The Slurm set
  is narrow on purpose. Variables such as `SLURM_NTASKS` and `SLURM_PROCID` make launchers (Lightning,
  mmcv/mmengine) assume a Slurm-managed distributed job and try to set it up, and inside the
  namespace a step cannot call Slurm.
- The keyed sha covers the profile map, the constants and the templates. The recorded sha covers
  the full expanded env.
- The profile map may not name any driver-owned variable, whatever the step uses:
  - the names above;
  - anything starting with `SLURM_`, `REPRO_`, `JUPYTER_`, `HF_`, `XDG_` or `R_LIBS`;
  - any proxy name, in either case;
  - `LD_PRELOAD`, `LD_LIBRARY_PATH`, `PYTHONHOME`, `PYTHONSTARTUP` and `PYTHONNOUSERSITE`.

  The driver's values cannot be overridden, and the refusal does not depend on the step.
- Templates expand with `spec.render`, the same grammar `spec.placeholders` validates (`{{`/`}}`
  escape a brace). Native dirs are escaped before they enter the `PATH` and `LD_LIBRARY_PATH`
  templates.
- Departures from the approved spec's wording (spec §2.3.4), settled in the B5 reviews:
  - The R library is keyed as `R_LIBS={r_lib}` with `R_LIBS_SITE=""`, per plan B7. It is not
    `R_LIBS_USER`, which on its own leaves the site library on the search path.
  - A narrow Slurm allowlist (`SLURM_JOB_ID`, `SLURM_ARRAY_JOB_ID`, `SLURM_ARRAY_TASK_ID`,
    `SLURM_CPUS_PER_TASK`, recorded only) replaces "`SLURM_*` recorded only", for the launcher
    reason above.
  - The profile's `native_dirs` list became the `native` map, selected per step through
    `resources.native`.
- The canary is a `python -c` snippet; the host may be `name` or `name:port`. It prints `{"canary_failed": true|false}` and always exits 0,
  so the caller decides what a connect means.

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
