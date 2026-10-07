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
  `127.0.0.1:9`, and since the B7 review it runs inside the step sandbox (`sandbox.sandbox_argv`):
  no network namespace route out, the build venv bound read-only, every capability dropped. The
  modules are loaded outside, through the same `bash --noprofile --norc` wrapper around `env -0`
  (`module_environment`), and their changed variables are passed into the build as values, the
  module `PATH` after the venv's `bin`. Lmod is not run inside the namespace. A slow test builds an
  sdist whose `setup.py` exits when it can connect to PyPI (the host itself can, as a control): it
  builds. The python-speech-features wheel is still bit-identical to the spike's.
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
- **Installed and reuse.** `.reproduce-venv.json` names the `env_key`, the overlay sha256, the
  `{sdist: wheel sha256}` map and the sha256 of `installed.json`; it is written last. The venv is
  reused, without any uv call, only when all of these match: the key and overlay, `installed.json`
  against its recorded sha256, and the wheel map against the verified `wheels.lock` lines
  (`build.reuse_wheel`). The venv name leaves wheel sha256s out (they are outputs), so a rebuilt
  wheel would otherwise leave a venv holding the old one under the same name; the B7 review found
  that. `rebuild=True` rebuilds every venv as well as every wheel. Anything else at that path is
  removed and rebuilt, under a `flock` on `<venv>.lock`. Two overlays with identical bytes share
  one derived venv, since the name is the overlay's sha256.
- **Complete means checked.** The marker also records `checks: {digest, pip_check_ok, smoke_ok}`,
  where the digest covers `pip_check_expected` and `smoke_imports` (neither is in `env_key`, so a
  changed expectation re-runs the checks). Only `prefetch_env` with a profile runs them and sets
  them true; without a profile a venv is installed but never complete. `venv_checked` is the
  predicate, and B12 must refuse a venv for which it is false.
- **`wheels.lock` and the manifest.** With `env.wheels` unset, `wheels_lock` names where the first
  prefetch records wheels, and an existing file there is refused: its bytes are not covered by any
  manifest sha256. `PrefetchResult.record` is true whenever the file's sha256 now differs from the
  manifest's (or the manifest has none), and the caller must write it into `env.wheels`.
- **pip check.** Each `The package \`X\` …` line becomes `(normalised X, kind, detail)`, where detail
  is the rest of the line without backticks; `requires …` lines are `requirement`, lines mentioning
  the platform are `platform`, anything else is `other` and can never be declared. The comparison
  is exact both ways: an undeclared violation and a declared one that did not occur both fail.
- **Smoke catches what pip check cannot.** The slow test's sdist wheel (python_speech_features 0.6)
  declares no requirements but imports scipy, which no lock holds: `uv pip check` is clean and the
  sandboxed import fails. Smoke runs one `python -c 'import m'` per module through `sandbox_argv`,
  the venv bound read-only, offline, with an empty private working directory: `python -c` puts the
  working directory first on `sys.path`, so a same-named module in the caller's directory would
  otherwise satisfy the import (a test shows the control import succeeding there and the smoke
  failing). An import past the timeout raises `E_SMOKE_TIMEOUT`.
- **R prefetch** loads the module with `bash --noprofile --norc -c '. /etc/profile.d/modules.sh &&
  module load "$1" …'` (the plan's `bash -lc` read dotfiles, B6), with an allowlisted env and no
  `set -u` (the init script fails under it). `Rscript --vanilla` keeps `Renviron.site` and
  `Rprofile.site` from resetting `R_LIBS_SITE` or the repository; B12 should run R steps the same
  way. The R script writes plain text (installed TSV, `.libPaths()`, `sessionInfo()`, R version),
  and Python writes `SESSION.json` with sorted keys, so its sha256 is stable and the script needs no
  JSON package. `SESSION.json` is written last and only when every declared version is installed,
  the module's R is `profile.r.version` and the library is first on `.libPaths()`. Its `key`
  (modules, R version, snapshot, declared packages) marks the library complete for reuse; when it
  is missing or differs, `SESSION.json` and the whole library are removed before the install, so no
  package from an earlier snapshot survives.
- **`R_LIBS_USER=<r_lib>`** as well as `R_LIBS`: without it R adds a user library under `HOME`
  (the temp home at prefetch), which then appears in `.libPaths()` and makes `SESSION.json` differ
  between two otherwise identical prefetches. The B7 reviewer checked on the real module that
  `.libPaths()` is then exactly the library and R's own. The sandbox keys `R_LIBS_USER` the same way.
- **`profile.r.build_modules`** are loaded before the R module, for the env capture and the
  install, so packages with C++ sources (lme4) build with a newer compiler than the system's and
  find its runtime libraries at run time through the captured `LD_LIBRARY_PATH`.
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

**Mounts: read-only by default (B12 review, D1).**

1. `mount --make-rprivate /`.
2. Each data root, then each writable directory, then the checkout and venv are bound to themselves
   (shallow-first), so a missing one aborts the script. The step's tmp is also bound on `/tmp`.
3. The sweep: the driver's own interpreter (`-I -S`, one process, ctypes `mount(2)`) reads
   `/proc/self/mountinfo` and remounts the topmost mount at every mount point
   `MS_REMOUNT|MS_BIND|MS_RDONLY`, keeping that mount's per-mount flags (`nosuid`, `nodev`, `noexec`,
   the atime flags), which a user namespace may not clear. Skipped: the writable binds (by path) and
   the fresh `/proc`. A refusal is fatal (the script exits before the step) unless the mount is a
   kernel pseudo filesystem (`PSEUDO_FS`: sysfs, cgroup, tracefs, binfmt_misc, autofs, and the
   like), which holds no data and whose writes need host privileges anyway. `hugetlbfs` and `mqueue`
   are not exempt (B12 re-review): both hold data a step could share with the host, both remount
   read-only on this cluster, and if either refuses somewhere the step aborts. An IPC namespace that
   would make `mqueue` private was the alternative; not exempting them is simpler and fails closed. Measured on a compute
   node with 733 mounts (438 NFS, 227 autofs): 0.07 s; the same loop as one `mount` process per entry
   took 11 s. Two mounts refuse there, tracefs and binfmt_misc. Taking only the topmost entry per path
   matters: remounting by path always reaches the topmost mount, and the flags of a lower entry
   (an autofs trigger under an NFS mount) would be refused.
4. A fresh private tmpfs on `/dev/shm`, so multiprocessing works and nothing reaches the host's.

A nested user namespace does not undo this. A step that calls `unshare(CLONE_NEWUSER|CLONE_NEWNS)`
gets every capability in the new namespace, but the mounts it inherits are locked: remounting a
read-only root read-write, or bind-mounting it and remounting the bind read-write, fails with
`EPERM`, and writes keep failing with `EROFS`. Checked on the RHEL 8 (4.18) compute nodes, and kept
as a regression test (`test_a_nested_user_namespace_cannot_make_a_read_only_root_writable`).
`unshare -Urm` from the shell cannot even write its uid map, and a shell started with `-Um` loses its
capabilities at exec; the test uses the syscall directly so it exercises the strongest case.

So everything is read-only except the step's `work/` and `tmp/` dirs (and, for the R session check
and the adapter, their own driver-made scratch dirs). That includes the cache with its R libraries,
Hugging Face and torch homes and venvs, native dirs, the driver's interpreter and the senselab
source, `$HOME`, the results, runs and Slurm-log roots, and scratch. A step that needs another
writable location must put it under `{out}`; no declaration mechanism exists, since no planned step
needs one. Hugging Face and torch read cached files offline without writing; if a step turns out to
need writable model caches, the fix is to point that library at a dir under `{tmp}`, not to make the
cache writable. `/tmp` is the step's tmp unless a path the step needs (a root, the result dir, the
cache, the checkout, the venv, the driver's interpreter) lies under `/tmp`, where the bind would hide
it; then `/tmp` stays read-only and `TMPDIR` still points at `{tmp}` (the test suite runs this way,
since pytest's `tmp_path` is under `/tmp`). Sandboxed wheel builds (B6) get `UV_CACHE_DIR` under their
own tmp for the same reason.

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
  and `R_LIBS_USER` (`r_lib`, as the plan's B7 says, with `R_LIBS_SITE` keyed as an empty constant so no site library
  is seen), `JUPYTER_PATH`
  (`{venv}/share/jupyter`), `JUPYTER_DATA_DIR`/`JUPYTER_CONFIG_DIR`/`XDG_CACHE_HOME` (under
  `{tmp}`), `HF_HOME`/`TORCH_HOME` (under `{cache}`), `CUDA_CACHE_PATH` (`{tmp}/cuda-cache`; spec §15.6
  put it under `{out}`, and B12's review moved it into the step's scratch so it is writable without
  being an output), `REPRO_OUT`/`REPRO_TMP`.
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
  - The R library is keyed as `R_LIBS={r_lib}` and `R_LIBS_USER={r_lib}` with `R_LIBS_SITE=""`, per
    plan B7 and its review. `R_LIBS_USER` alone would leave the site library on the search path.
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

`source.py` details:

- **Layout.** Mirrors at `<cache>/src/<name>` (the URL's last component without `.git`; a second
  URL with the same name is refused rather than renamed, so the path stays predictable for
  authoring). Worktrees at `<cache>/wt/<name>-<commit[:8]>-<patch set[:8]>`, with a sidecar
  `<worktree>.json` written only after the worktree verifies.
- **Ref namespaces.** Upstream branches are fetched into `refs/upstream/heads/*`, not
  `refs/heads/*`, so an upstream branch can never collide with or prune a `refs/heads/repro/*`
  ref. Fetches also pass `--no-prune`. A commit reachable from no branch or tag is fetched by sha
  into `refs/upstream/pinned/<sha>`, so gc keeps it.
- **Patch-set sha256.** sha256 of canonical JSON over the ordered `[manifest-relative path,
  sha256]` pairs. The full path, not the basename, so `patches/a/x.patch` and `patches/b/x.patch`
  with identical bytes are two patch sets with two worktrees rather than one marker that matches
  neither (B8 review). The empty set's value is `EMPTY_PATCH_SET_SHA256`, which names the
  authoring ref. A marker is reused when its (repo, commit, patch-set sha256) matches.
- **Reproducible shas.** Commits use a fixed identity and the 1980 epoch, no system or global git
  config, and hooks off, so the same (commit, patch set) gives the same head and tree sha in any
  cache. The clear commit is made even when no notebook changes (`--allow-empty`), so every chain
  has the same shape: `commit`, clear, one commit per patch.
- **Git environment.** Allowlisted, with a private `HOME`. Fetches keep the caller's `HOME` and
  `SSH_AUTH_SOCK`/`GIT_SSH_COMMAND`/proxy variables, because upstreams are private SSH remotes and
  ssh reads its keys there; global config is still not read, so credential helpers configured
  there do not apply.
- **Clearing** drops code-cell `outputs`, nulls `execution_count`, and also drops notebook-level
  `metadata.widgets` (saved widget state can embed output data) and the `attachments` of markdown
  and raw cells (embedded images can show data). Plain JSON, no nbconvert.
- **Submodules** are worktrees of their own mirrors at the gitlink sha, not clones: no second copy
  of the objects, no file-protocol submodule clone. Notebooks inside submodules are not cleared,
  because a clear commit there would change the superproject's gitlink. Instead the checkout fails
  closed: a submodule notebook that clearing would change raises `submodule-notebook-outputs`
  unless the caller passes `allow_submodule_notebook_outputs`, and the paths are then recorded in
  the checkout.
- **Patch bytes.** Verified with `verify_files`, read again and re-hashed, then fed to
  `git apply --check --index -` and `git apply --index -` on stdin; no fuzz.
- **After each patch**, before its commit, every tracked notebook is run through the clear; if
  any would change, the checkout fails with `patch-lint`. The line lint below is an early report,
  not the guarantee: it cannot see outputs in compact one-line JSON or a hunk that starts inside an
  outputs array without context, and this check can.
- **Lint.** Deny patterns are matched with `grep -E` so their dialect is the pre-commit hook's,
  over every patch line (headers and context too). The outputs check tracks `"outputs": [`
  blocks within each notebook hunk, and also flags any changed line naming `"output_type"`, for a
  hunk that starts inside an outputs block. Binary patches are refused because they cannot be
  scanned, and so is creating a symlink or changing a mode to `120000`, because a link could point
  a checkout path at data outside it. C-quoted header paths (git quotes non-ASCII names) are
  unquoted before the `.ipynb` test. Findings carry line numbers and pattern indices only.
- **Reuse.** A worktree with a marker is verified (head, ref, first-parent chain, tree, clean
  status, submodule heads) and refused on any mismatch rather than rebuilt, so tampering is never
  papered over. A worktree without a marker is an interrupted build and is removed and rebuilt.

## Results and cache

**Step key:** sha256 over these fields, and nothing volatile:

- the reproduce source digest (`senselab/utils/reproduce` sources)
- code commit and submodule SHAs
- `env_key`
- `wheels.lock` and overlay sha256s
- keyed env, native-dir content digests, GPU architectures
- argv template, cwd, inputs, stage, outputs
- patch-set, adapter, adapter-env and input-set digests
- `hf_weights` (commit-pinned, per-file sha256s), R session and R module-env digests
- metrics schema, expectations, replicate index
- upstream keys

**Result dir:** each step writes to `<results_root>/<step_key>/`. A step whose result dir holds a
completed summary with exit 0 is skipped, and its result dir is linked into the dated run. Downstream steps
therefore read the same files on any day.

**Input-set digest:** sha256 over the sorted (sha256(relative path), sha256(content), size)
triples, so it reveals no file names. It is taken before and after the step, and a change fails
the step.

- Hashing the relative path (not storing it) is what keeps the digest safe to record in provenance
  and `summary.json`; hashing it at all is what makes a rename change the digest.
- Errors are a fixed code (`E_DIGEST_NO_ROOT`, `E_DIGEST_BAD_PATTERN`, `E_DIGEST_UNREADABLE`) with
  no path, because the exception class and message can reach `summary.json`.
- Each pattern is walked from its longest wildcard-free directory, and a pattern with no wildcard is
  hashed directly, so a narrow pattern on a large data root does not walk the whole root.
- Non-regular files are refused rather than opened: opening a FIFO blocks.
- Exclusions are whole path components, so the output digest can leave out `{out}/cuda-cache`
  (written by the CUDA JIT during the step) without also hiding a sibling such as `cuda-cache-x`.
- Patterns and exclusions with an empty, `.` or `..` component are refused, so one file cannot be
  named two ways and nothing outside the root can be reached by spelling.
- `require_nonempty` (`E_DIGEST_EMPTY`) lets a caller refuse an input set that matched nothing; a
  typo in a pattern would otherwise digest the empty set the same way before and after.
- **Symlinks are asymmetric, by construction.** A pattern's wildcard-free leading components are
  resolved by path, so a symlinked directory named there (or in a pattern with no wildcard at all)
  is followed. Below that prefix the walk does not descend into symlinked directories, so a glob
  never reaches one; this keeps a symlink loop or a link out of the data root from being walked
  just because `**` matched it. A manifest that needs files behind a directory symlink names the
  link in the pattern's fixed prefix.

**Step keys, result dirs, claims and ordering (B11).** `cache.py` and `dag.py`.

- **What the key covers.** `step_key_payload` names every field explicitly (no whole-model dump of
  `StepSpec`), so a field added to the manifest later is unkeyed until someone decides it should be
  keyed. The payload holds:
  - `reproduce_source_sha256`: a sha256 over every installed `senselab/utils/reproduce/**/*.py`
    file (sorted relative path and content sha256), taken from the package directory and computed
    once per process. It stands for everything the driver decides: metrics validation, min-cell
    suppression, expectation evaluation, the GPU and canary guards, the child preamble and the
    sandbox script. A content digest replaced an earlier hand-bumped number because a forgotten
    bump serves stale results silently, while a spurious recompute only costs time. senselab's
    version is not keyed: hatch-vcs changes it on every commit, so keying it would recompute every
    multi-hour step after any unrelated senselab commit (the reason the analyze-audio cache dropped
    it at schema 24). The digest covers only `utils/reproduce`, so **B12 rule: adapters import
    nothing from senselab outside `senselab.utils.reproduce`, and B12 enforces it** (for example by
    running the adapter with an import hook or a `sys.modules` check that refuses other
    `senselab.*` modules); otherwise a change to such a module would change results under an
    unchanged key.
  - the commit and submodule commits, and the patch-set sha256 (`source.patch_set_sha256`, the
    same value that names the worktree)
  - `env_key` (which already covers the torch and body lock sha256s, the sdist inputs and
    `python_exact`, so they are not repeated), the `wheels.lock` and overlay (`test`,
    `notebook_tools`) lock sha256s
  - the keyed env (`sandbox.keyed_templates`: the profile `env` map, the fixed and offline
    variables, and every path variable in template form, including `PYTHONPATH`, `PATH`,
    `LD_LIBRARY_PATH` and `R_LIBS`)
  - a content digest of every selected native dir (`native`): `cache.native_digests` runs
    `digest.input_set_digest(dir, patterns, require_nonempty=True)` over each `bin_dirs` then
    `lib_dirs` entry of the selected `NativeSpec`s. The keyed env holds only the dir paths, and a
    module upgraded in place (a new ffmpeg in the same prefix) would otherwise keep the key. The
    patterns are the `NativeSpec`'s `bin_patterns` (default `["*"]`: every top-level entry, which is
    what `PATH` lookup reaches) and `lib_patterns` (default `["**/*.so", "**/*.so.*"]`: shared
    objects, versioned or not, at any depth, so plugin subdirs count but headers, pkg-config files
    and docs do not). `__pycache__` dirs and `*.pyc` are always excluded (`digest.input_set_digest`'s
    `exclude_patterns`), since a Python run beside a native tool rewrites them. A dangling symlink
    or an unreadable file among the matches gives `E_DIGEST_UNREADABLE`, and a dir matching nothing
    gives `E_DIGEST_EMPTY`. A wrapper script in a bin dir (a shell script that execs a binary
    elsewhere, as conda and module prefixes often install) is digested as the script's own bytes; what
    it runs is not followed. A profile whose tool is a wrapper names the real binary's dir as well
  - `resources.gpu_arch` (sorted). A step limited to sm_80 can produce different numbers from one
    run on sm_90 (cuBLAS and cuDNN pick kernels, and so reduction orders, per architecture), so widening or narrowing the allowed
    architectures recomputes. `min_driver` is left out: it is a floor checked at run time, and
    any driver above it runs the same kernels for the same CUDA runtime in the venv, which the
    locks already pin; keying it would recompute when someone merely tightens the floor
  - kind, argv template and cwd; inputs (root, sorted patterns, and one input-set digest each),
    stage (root, sorted patterns, dest) and outputs
  - the adapter sha256 and `adapter_env_sha256`, the identity of the interpreter that runs the
    adapter. B12 passes the step venv's `installed_sha256` when the adapter runs there, or
    `cache.driver_env_sha256()` when it runs in the driver's env: a sha256 over `sys.version` and
    the sorted `name==version` of senselab's runtime dependency closure (its requirements with no
    extra, followed transitively; a requirement that asks for extras, such as `dep[x]`, also brings
    in `dep`'s requirements for `x`) as installed in the interpreter's own `purelib`/`platlib`.
    Only `.dist-info` distributions are read: a legacy `.egg-info` dist has no `METADATA` file and is
    skipped, so a package installed that way does not move the identity.
    `sys.path`, the cwd and `PYTHONPATH` are never read, so a stray checkout cannot shift it;
    extras and dev groups are left out because the adapter may not import them. senselab itself
    is left out, its reproduce code being `reproduce_source_sha256`.
  - `hf_weights`, sorted by (repo, repo_type), each entry's files sorted by path
  - the R `SESSION.json` and module-env sha256s (spec §15 item 5 keys both)
  - the metrics schema and expectations, `replicate`, and the upstream keys
- **Staged data-root files are keyed through `inputs` (rule "stage covered by inputs").** Every
  pattern of a `stage` entry drawn from a data root must equal a pattern of some `inputs` entry with
  the same root, possibly spread over several entries (`E_SPEC_STAGE_UNDIGESTED`, a `StepSpec`
  validator), so one digest set covers both and editing a staged file moves the key. Requiring the
  repeat, rather than digesting stage entries separately, keeps a single list of digests per step. `checkout` stages are keyed by the commit
  and patch set, `up:` stages by the upstream key.
- **Pattern order is not keyed.** `digest.input_set_digest` digests the union of the matches (a
  file matching several patterns counts once, and entries are sorted), so pattern order carries no
  meaning and is sorted in the payload. `gpu_arch` and HF repositories and files are sets too. List
  order is kept where it means something: argv, patches, the `inputs` list itself (paired with
  `input_digests` and with provenance's `inputs[i]`), and expectations.
- **What it leaves out, and why.** Volatile variables never reach it, by construction: the key reads
  the keyed env from `keyed_templates`, which never reads `os.environ`, and the profile may not set
  `SLURM_*`, `CUDA_VISIBLE_DEVICES`, `TMPDIR`, `HOME` and the rest (sandbox `RESERVED`). A second
  "refuse volatile names" check in `cache.py` would guard a path that cannot be taken, so there is
  none. Also left out: the date, the data-root locations (argv keeps `{root:<name>}` as a template;
  content is covered by the input digests), the results root, the partition and limits,
  `min_driver` (above), the step's own `name` field, and the repository URL. Step names still enter
  the key wherever a step refers to another: its upstream keys are keyed by need name, and
  `{up:<name>}` stays a template in argv and stage. Only the step's own name is out, so two steps
  declared identically (same replicate, same references) share one result dir, which is what
  content addressing should do; the claim makes the second wait and then skip. Leaving the URL out
  means moving a mirror does not recompute anything, because the commit already pins the code.
  `smoke_imports` and `pip_check_expected` are build checks, not result inputs.
- **Expectation numbers.** `Expect.value`, `Expect.tolerance` and `tolerance_from`'s `a` and `k`
  refuse NaN and infinity, and `-0.0` is stored as `0.0`, so a key never depends on a float spelling
  JSON cannot write or that compares equal.
- **The wheels lock comes from the manifest.** An env with `sdist_allow` and no `env.wheels` cannot
  be keyed (`E_KEY_NO_WHEELS`): the first prefetch records the wheel sha256s, and the manifest must
  carry that file's sha256 before a run. Taking the post-prefetch sha256 as an argument instead
  would let a key depend on an uncommitted file.
- **Serialization.** `canonical_json`: `json.dumps(sort_keys=True, separators=(",", ":"),
  ensure_ascii=True, allow_nan=False)`, tagged `"schema": "reproduce-step-key-3"`. Mappings
  (submodules, upstream, env, native) are dicts, so their caller-side order cannot matter. Floats
  in expectations are Pydantic-coerced, so `77` and `77.0` key alike. Changing what the payload
  holds bumps the schema tag; changing driver code changes `reproduce_source_sha256` by itself.
- **`hf_weights`.** Required on `StepSpec`, with no default, so every manifest states `[]` for a
  step that loads no weights (all tier-1 steps). Each entry is `{repo: "<org>/<name>", repo_type:
  model|dataset, revision: <40-hex commit>, files: [{path, sha256}, ...]}`. The revision is a
  commit, never a branch or tag (senselab's rule that a load passes a commit SHA). `files` lists
  every file the step reads with its sha256, which prefetch checks after download. It needs at
  least one file, no duplicate paths, and one entry per repository.
- **Completion marker.** A result dir is done when `summary.json` parses as an object with integer
  `exit` 0 (a JSON `true` is not 0). `write_summary` writes a temporary file with `O_EXCL` and
  mode `0o660` (umask permitting), fsyncs it, renames it over `summary.json` and fsyncs the
  directory, so a reader sees the old summary or the new one and never a torn file. It is the last
  write of a run, so any dir without it is incomplete. A failed run's summary (exit non-zero)
  stays readable for diagnosis and is cleared by `reset_result_dir(results_root, key)` when the
  step is retried. It wipes only when `summary.json` is absent or parses as an object with a
  non-zero integer exit; any other read error (a summary that does not decode as UTF-8 included), or a
  summary that is not such an object, is
  `E_RESULT_UNREADABLE` and nothing is deleted, so a transient NFS or permission error can never
  erase a finished result. A done dir is never reset (`E_RESULT_DONE`), a symlink or non-directory at the
  path is refused rather than removed (`E_RESULT_NOT_DIR`), and a failed removal or creation is
  `E_RESULT_RESET` with the `OSError` (which names the path) suppressed.
- **Claims.** `claim` holds a `SharedFileLock` (heartbeat, stale takeover) on
  `<results_root>/.claims/<key>`. The lock files live in `.claims/` rather than beside each result
  dir, so the results root holds only result dirs and `.claims`. `manage_dir_mode=False`: the
  default chmods the lock's directory to `2775`, which would make a confidential results tree
  world-traversable. A timeout raises `E_CLAIM_HELD` with the `TimeoutError` (whose message names
  the lock path) suppressed. The caller re-checks `is_done` inside the claim, because a waiter
  usually gets the claim right after the holder marked the step done.
- **Run links.** `<runs_root>/<YYYY-MM-DD>/<analysis>/<step>` is a relative symlink to the result
  dir (computed between real paths), so the runs and results trees can move together. Relinking to
  the same target is a no-op. A different target replaces the link by renaming a new symlink over
  it, so a reader never finds the link missing. A real file or directory at the link path is
  refused rather than removed.
- **Ordering.** `toposort` is Kahn's algorithm with a heap on step names, so the order depends only
  on the set, not on file or step order. A repeated name, a need naming no step in the set, and a
  cycle are refused. The cycle error names only the steps on one cycle, not every step downstream
  of it. `toposort` also runs `check_expect_sources`, so one call validates the whole set.
- **Cross-step expectations.** A `StepSpec` validator cannot see other steps, so `dag.py` checks
  that a `from_step.metric` is an address the upstream `metrics_schema` declares
  (`metrics.metric_address_error`). It also checks that the address names the same quantity as the
  step's own metric (`metrics.metric_quantity`: rate, mcc, statistic or count), so a rate is never
  compared with a count. Units (`fraction`, `pp`) are per metrics entry and only known at run
  time, so the manifest-time unit check is that any expectation `unit` sits only on a rate or
  `mcc`, the quantities that take one. That check needs only the step itself, so it is a
  `StepSpec` validator (`E_SPEC_EXPECT_UNIT`) and applies to every expectation, not only
  `from_step` ones. Each `tolerance_from` step must declare the expectation's own metric, since the
  spread is taken over that metric of the calibration replicates.
- **Constraints on B12/B13.**
  - Input digests are keyed, so a step's key exists only once its inputs can be hashed. An `up:`
    input or stage reads the upstream result dir, so a downstream key can be computed only after
    the upstream step finished; `submit` cannot precompute every key at submission time and must
    let each job compute its own key when it starts.
  - Every skip check re-hashes the step's input sets and native dirs. "Done" is never decided from
    a remembered key, so a skip on a large data root costs a full read of its declared inputs.
  - The stage copy selects files with `digest.pattern_matches` and walks with the same rule as
    `input_set_digest`: a pattern's wildcard-free prefix is resolved by path, and below it no
    symlinked directory is descended. Otherwise the copied set and the keyed set could differ.
  - `SharedFileLock` logs a stale-takeover warning on the `senselab` logger, and it carries the
    `.claims` lock path under the results tree. B12 routes that logger to `{out}/logs/driver.log`
    only, never to the terminal.

## Metrics, expectations, provenance

**Adapters.** Each step's adapter turns the analysis's own outputs into `metrics.json`, from a fixed
vocabulary:

- headline and secondary metrics, and per-class values
- confusion counts
- named counts
- degrees of freedom

Unknown keys and non-numeric values are rejected. Count fields below a minimum cell size are
suppressed.

**Why the vocabulary has the shape it has.**

- `headline` (and `secondary`) is a list of entries, not a map, because one step can report the same metric for
  several systems (told apart by a declared `label`) or several test statistics of one name (told
  apart by their degrees of freedom).
- Every name, label and count name an adapter may emit is declared in the manifest's
  `metrics_schema`. The declared names must all appear, so a pass rule cannot silently pass on a
  missing number.
- **Balanced accuracy is the only classification headline (Satra, 2026-10-07: "Raw accuracy is
  unhelpful without a null rate, so only a balanced accuracy headline is helpful.").** A
  classification step's `headline` is exactly `["balanced_accuracy"]`; anything else is the schema
  error `E_SPEC_HEADLINE_NOT_BA`. Accuracy, top-1, macro F1, AUROC, PR-AUC, MCC and F1 may be declared
  as `secondary` metrics. Secondary entries go through the same validation as headline entries
  (value range, `floor` checked against `expected_floor`, `lift` required), so accuracy and top-1
  always travel with their null rate, the majority-class prevalence. `macro_recall` was removed
  from the vocabulary outright: it is the same quantity as balanced accuracy, and pre-alpha means
  no aliases, so an adapter maps a source's `recall_macro` to `balanced_accuracy`.
- **Names are bound to the step's kind** (`HEADLINES_BY_KIND`, `SECONDARY_BY_KIND`): classification
  headline {balanced_accuracy} and secondary {accuracy, top1, macro_f1, auroc, pr_auc, mcc, f1};
  detection headline {avg_map, map_at}; statistic {f_stat}; count {count}. Otherwise a step could
  declare itself `statistic` to escape the imbalance rules while reporting an accuracy. Detection,
  statistic and count steps take no secondary metrics (`E_SPEC_SECONDARY_NOT_ALLOWED`): none of the
  tier-1 rows needs one, and a detection secondary would add names whose floors are presence-only
  anyway. Adding them later is a vocabulary edit.
- A classification step declares at least two classes. A baseline such as a Bernoulli or
  majority-class predictor is a classification step like any other: the N4 baseline declares
  headline `balanced_accuracy` (floor 1/k) and secondary `accuracy` with the majority-class floor,
  and `threshold_tuning: none`.
- Free text is never accepted. The only strings are declared labels and fixed words: `unit`
  (`fraction`, `pp`), `threshold_tuning`, `protocol`, and the split's `unit` and `stratify`, each
  from an enum in `metrics.py`. A new unit or stratification is added by editing that enum, in a
  reviewed change, never by a manifest.

**Imbalance.** A headline number on an imbalanced split means little without its floor, so a
classification or detection step's output is refused unless it carries:

- `prevalence` in [0, 1] for every declared class; for classification (single-label by definition
  here) also `recall` and `precision` for every class, and prevalences that sum to 1 within 1e-6
- `floor` and `lift` on every headline and secondary entry; `lift` must equal value minus floor
- `threshold_tuning`: `validation`, `none` or `test`. `test` is accepted with the warning
  `W_TUNED_ON_TEST` (returned beside the clean object, not an error) so that a reproduction can
  record honestly that the source tuned its threshold on the test split; refusing it would force
  the adapter to misreport the source.

**Which floors are checked.** With k the number of declared classes **present** on the scored
split (prevalence above 0) and that split's prevalences, the floor
is checked within 1e-6 (scaled by 100 for a `pp` entry):

| metric | floor |
|---|---|
| balanced_accuracy | 1/k |
| auroc | 0.5 |
| mcc | 0 |
| accuracy, top1 | the largest class prevalence (majority class) |
| pr_auc | p, the positive-class prevalence, when the schema declares `positive` |
| f1 | 2p/(1+p), the all-positive predictor, when the schema declares `positive` |

`macro_f1`, `avg_map` and `map_at` are checked for presence only: their trivial-classifier value
depends on choices (which trivial predictor, which IoU thresholds, per-class averaging) that the
vocabulary does not carry, so a fixed formula would be wrong for some rows. `positive` may be
declared only on a binary classification step. `pr_auc` and `f1` are single-class quantities, so a
step that declares either as secondary must be binary (`E_SPEC_SECONDARY_BINARY_ONLY`) and declare
`positive` (`E_SPEC_POSITIVE_REQUIRED`); their floors are then always compared, never presence-only.

**Split protocol.** `split_report.protocol` is `seen_subject`, `unseen_subject` or `none`.
`seen_subject` and `unseen_subject` are both legitimate evaluations. A seen-subject split asks
whether a model works on new clips of people it was trained on; an unseen-subject split asks whether
it works on new people. The protocol travels with every number so that a seen-subject result is
never read as generalization to new people. `invariant_ok` records whether the split met its own
protocol's invariant (no subject in two partitions for unseen; every test subject in train and no
session or video in two partitions for seen). A `false` is a finding about the original split,
reported with the number, not a schema error.

- `none` labels a source split that has no protocol at all (for example a row-level split), and is
  accepted only with `invariant_ok: false`, so it can never read as a clean split.
- A classification or detection step must emit a `split_report`. The only way out is an explicit
  schema opt-out, `split: {none: whole_dataset}`, for a step scored on the whole dataset with no
  held-out partition (a baseline); a step that opts out may not also emit a report.
- For classification, every fold's `prevalence` must give every declared class (and sum to 1).

**Suppression and its residual risk.** `n`, `counts.*`, `confusion.*`, a `count` headline value and
a fold's `n_groups` below the minimum cell become `"<k>"` rather than rejecting the file: a small
cell is expected in a real result and is not an adapter bug. `n_groups` is suppressed for every split
unit, not only `subject`/`family`: an evaluation that splits by recording or clip (a seen-subject
protocol) still reports `n_groups` as its number of subjects, so a fold of three children under unit
`recording` would otherwise be emitted as is. Degrees of freedom and `n_folds` are never suppressed.

Accepted residual risk: a prevalence multiplied by an unsuppressed `n` (or a recall times a
class size) can reconstruct a small cell that was suppressed elsewhere. Closing that would mean
rounding or suppressing prevalences and rates, which are the numbers the imbalance rules exist to
report. The output stays aggregate-only and is read only through the narrow reruns exception.

The driver validates twice, once the adapter's output and once the final object after writing the
`expect` results (`writer="driver"`), and must pass the same `min_cell` both times: a token
`"<5"` is accepted again only under `min_cell=5`.

**Absent classes.** A declared class with prevalence 0 does not count toward k, matching
scikit-learn's balanced accuracy, which averages recall over the classes present in `y_true`.
Counting it would lower the chance floor (two present classes of three declared would accept 1/3
instead of 1/2) and inflate the lift. Such a class gives the warning `W_CLASS_ABSENT`.

**Keys bound to kinds.** `per_class` (and the schema's `per_class`/`positive`) belong only to
classification and detection, and `confusion` only to classification. Elsewhere the key is refused
with `E_METRICS_KIND_KEY`, a code distinct from `E_METRICS_UNKNOWN_KEY` because the key is in the
vocabulary but not for this kind. Without this, a step declared `statistic` could report per-class
recall and a confusion matrix free of the prevalence, floor and split rules.

**Value ranges.** Headline values are range-checked (`E_METRICS_RANGE`): classification and detection
rates in [0, 1], or [0, 100] with `unit: pp`; `mcc` in [-1, 1], scaled likewise; `f_stat` at least 0;
and `ci_low <= value <= ci_high` when given. This catches a percentage written without `unit: pp`.
`floor` takes the value's bounds for every headline entry, presence-only floors included, and
`lift` lies within plus or minus the range width under the same scaling. Per-class recall,
precision and f1 lie in [0, 1], as prevalence already does.

**Sub-sample prevalence.** When the top-level `n` is given, a class prevalence above 0 but below
`1/n` stands for less than one example and is refused (`E_METRICS_PREVALENCE_SUBSAMPLE`); it would
otherwise let an adapter dodge `W_CLASS_ABSENT` and the present-class floor rule with a tiny
non-zero value.

**Driver-only keys.** `delta`, `target`, `tolerance` and `pass` are refused anywhere in adapter
output. Only the driver, evaluating `expect`, writes them; otherwise an adapter could report its own
pass.

**Metric addresses.** `expect` and `from_step` name a value as `<name>`, `<name>@<label>` (a headline
or secondary metric; the two name sets are disjoint, so a bare name is unambiguous),
`counts.<name>`, `per_class.<label>.<field>` or `confusion.<cell>`. A manifest's `expect` metrics are
checked against its own `metrics_schema` when the manifest loads, and `expect.unit` is `fraction`
or `pp`. `from_step.metric` names a metric of the upstream step, so it is checked against the
upstream step's schema when the DAG is assembled (B11), not here. A bare name that matches
several entries is refused when read, rather than picking one.

**Notes for `run.py` (B12).**

- `validate_metrics` returns error codes (`E_*`) and warnings (`W_*`, `WARNINGS` in `metrics.py`)
  in one sorted list. Only an `E_*` code fails the step; `clean` is None exactly then. `W_*` codes
  (`W_TUNED_ON_TEST`, `W_CLASS_ABSENT`) are recorded in `summary.json` and provenance next to the
  numbers and do not change the exit code.
- Declared inputs are digested with `input_set_digest(root, patterns, require_nonempty=True)`, so a
  pattern that matches nothing fails with `E_DIGEST_EMPTY` instead of digesting the empty set the
  same way before and after the step. Output digests cover `work/` only, which holds no CUDA cache.
- The final object, after the driver writes the `expect` results, is validated again with
  `writer="driver"` and the same `min_cell`.

**Host record.** Hostname, CPU model and counts, the first GPU's name and driver from `nvidia-smi`
with the GPU count, and the venv's Python, torch and torch CUDA versions, asked of the venv's own
interpreter in isolated mode with a minimal environment so the caller's `PYTHONPATH` cannot answer
for it. Anything unavailable is None.

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

**Provenance record (B10).** `provenance.record_step` writes `{out}/prov/store.jsonl` and the
`prov-reproduce_{io,act,soft,env}.json` files.

- **No path, anywhere.** Data file names and directory names can carry participant identifiers, so
  the record holds no input or output path, file name, data-root-relative path or glob pattern
  (patterns are data-root-relative paths too). An input or output is named only by its manifest
  position (`inputs[<i>]`, `outputs[<i>]`), its role, its data-root name for an input, its
  input-set digest and its file count. The test writes files under distinctive names, records them
  and greps every written file for each name component and the temp root.
- **Why `artifact` is a new type.** The store required a `path` beside every checksum, so that a
  checksum always said which file it described. An input set has no single file, and its file names
  must not be written, so the exemption is bound to one new type rather than relaxed for all: every
  other type still refuses a pathless checksum. The design spec's rev-4 text proposed a separate
  `content_sha256` attribute; reusing `checksum_sha256` on a dedicated type keeps one checksum term
  and one BEP028 mapping (`Checksum`, SPDX SHA-256). The value is an input-set digest, not a single
  file's sha256; `file_count` beside it says how many files it covers. An artifact sits under
  `prov:Entity`, not `Files`, because BEP028's `Files` are located files.
- **The command is withheld.** BEP028's `Command` is required; a rendered argv names data files, and
  even the argv template may hold a data-root-relative path, so `Command` stays null, the activity
  `Description` says why, and the argv is recorded as `argv_sha256`. The `Description` was hardcoded
  to triage's wording; it is now a `description` argument of `to_bep028_graph` and
  `convert_store_file`, with triage's wording as the default. `write_bep028_files` serializes an
  already-built graph and has no activity to describe, so it takes no `description`.
- **The step environment** is recorded as its sorted variable names plus `env_sha256`, never values:
  `HOME`, `TMPDIR` and templated variables expand to paths.
- **The code commit** is a software agent with `commit_sha` set and `version` `<step> <commit>`, so
  the BEP028 `Software` node reads `Label` = step name, `Version` = the commit, which is the mapping
  model agents already use; the serializer is unchanged for software agents.
- **Environments.** The venv: its directory name (`repro-<analysis>-<key8>`, under the cache, not a
  data root), the Python its interpreter reports and its installed packages. The host: the
  hostname, the driver's own Python, `platform.platform()` and senselab's version. The rest of the
  host record (CPU, GPU, driver, torch, job id) goes into the activity's `parameters`, because
  `Environment` has no fields for it.
- **What identifies the env (spec §2.3 step 6).** The activity's `parameters` carry
  `torch_lock_sha256`, `body_lock_sha256`, `overlay_sha256`, `env_key` and `installed_sha256`.
  `installed_sha256` is the value in the venv's marker, read through `env.installed_sha256`, which
  also checks `installed.json` still hashes to it; a venv whose file no longer matches is refused.
  The two can then be cross-checked. An earlier draft recorded a `dependencies_digest` over the
  parsed package map instead; it was removed, because `parse_freeze` drops `+cuXXX` and so the
  digest could not tell a `cu126` venv from a `cu128` one, while `installed.json` keeps the raw
  freeze lines.
- **Closed forms.** Every value `record_step` records is checked before anything is written, and a
  non-string where a string is expected is a `ValueError` naming the field, never a `TypeError`:
  - hex keys and digests (64-hex sha256s, a 40-hex commit, a hex step key), the overlay sha256
    hex or None;
  - step, venv and root names: letters, digits, `_`, `-`, `.`, no leading dot, so `.`, `..` and
    `up:..` are refused; a root may carry the `up:` prefix;
  - package names in PEP 503 normalised form, versions from `[A-Za-z0-9.+!_-]` (no `/`, `:` or
    whitespace);
  - the host record's exact keys; counts are non-negative integers (not `bool`); strings match a
    per-key pattern (hostname, numeric driver and CUDA versions, version strings, a numeric job id,
    and single-spaced words without `/` for the CPU and GPU names), at most 128 characters;
  - `E_*`/`W_*` codes, variable names, an integer exit code (not `bool`), and full ISO 8601
    datetimes with a timezone.

  Errors name the field, never the value. Step stdout and logs are not inputs to `record_step` at all.
- **Write order and reruns.** The store and graph are built and the graph passes `check_graph`
  before any file is written. `store.jsonl` is then replaced, the BEP028 groups written, and any
  `prov-reproduce_*.json` the new record did not produce (the `_io` group of a step that now
  declares no inputs or outputs) is removed, so a rerun into the same `prov/` never leaves a group
  from an earlier record beside the new one. Other files in `prov/` are left alone. Removing
  stale groups was preferred to always writing every group, because an empty group file would
  claim a record kind with no records.

## Running a step (B12)

`run.py`: `run_step` for one step, `run_steps` for a set of manifests in DAG order.

- **Result dir layout (B12 review, D2).** `R = <results_root>/<key>`:
  - `R/work/`: the only dir the step may write. `{out}` (argv, `REPRO_OUT`, `outputs` patterns, and a
    dependent's `{up:<step>}` and `up:` inputs) means `R/work`. Stage files go to `R/work/stage`, so a
    script that writes next to its inputs still writes only into `{out}` (spec §15.3); a read-only
    stage bind would break that.
  - `R/tmp/`: `{tmp}`, `TMPDIR`, `/tmp` when possible, `HOME` and the Jupyter dirs, `CUDA_CACHE_PATH`.
    Removed after the run (never following a symlink in it); the driver never reads it after the step.
  - Driver-owned: `R/logs/`, `R/prov/`, `R/adapter/` (the adapter copy and the R session script),
    `R/metrics.json`, `R/summary.json`, and `<runs_root>/<date>/<analysis>/run.json`. They are never
    writable in a sandbox, since `R` itself is read-only there.
  - Step stdout and stderr go to `R/logs/step.{stdout,stderr}.log`, opened by the driver
    (`O_CREAT|O_EXCL|O_NOFOLLOW`) before the sandbox starts.
  - Every driver create under `R` uses `O_EXCL|O_NOFOLLOW`; `metrics.json` is written to such a file
    and renamed into place, refusing a symlink at the target; `record_step` refuses a symlinked
    `prov/`. These are defence in depth: the step cannot write `R` at all.
  - Reserved names: after the step, any `summary.json`, `metrics.json` or `run.json` anywhere under
    `work/` (file, dir or symlink) is deleted and the step fails with `E_RESERVED_NAME`, so nothing a
    step forged can be read as a driver file. A forged `R/summary.json` cannot exist: `R` is read-only.
  - Agents read only `R/summary.json`, `R/metrics.json` and `run.json` (plan amendment,
    2026-10-07).
- **Key before skip.** Inputs and native dirs are keyed (B11), so the order of the plan's B12 list
  changes: resolve the prefetched worktree (`source.prefetched_checkout`, which never creates one:
  `E_SRC_MISSING`, or `E_SRC_MISMATCH` when it no longer verifies), open each data root once so an
  automounted root is mounted before the namespace is entered (`E_ROOT_MISSING`), digest every
  declared input with `require_nonempty=True` and the native dirs, compute the key, and only then
  check done. A done dir is linked and the step is `skipped`, with no user namespace, venv or GPU
  needed. These input digests are the "before" digests.
- **Refusals before the reset.** Inside the claim, after done is checked again: `E_NO_USERNS`; the
  venv must load from its marker (`env.load_venv`, `E_ENV_MISSING`) and pass `venv_checked`
  (`E_ENV_UNCHECKED`; notebook kinds use the derived `notebook_tools` venv); `gpu_arch` or `min_driver`
  with no GPU is `E_GPU_MISSING`, a driver below `min_driver` `E_DRIVER_OLD`. These return a code and
  leave the result dir untouched, so a previous run's diagnostics survive; only then is the dir reset.
  The GPU guard runs before the step, not after as the plan listed it, so a GPU job is not spent on a
  host that fails it. After the step, torch built without CUDA on a `gpu_arch` step is
  `E_GPU_TORCH_CPU` and a torch CUDA above `max_cuda` is `E_MAX_CUDA`. `gpu_arch` is not matched
  against the host's architecture: the host record has no compute capability, and the partition
  choice (B13) is what places the step.
- **Host record.** `host.host_record(None)` probes only the host (hostname, CPUs, `nvidia-smi`); no
  venv interpreter runs outside the sandbox. Python, torch and torch CUDA come from the preamble.
- **Stage.** Files are chosen by `digest.matching_files`, the same walk as `input_set_digest`, and
  copied by content. A rendered `dest` must be relative with no `..` (`E_STAGE_BAD_DEST`); two entries
  writing one file is `E_STAGE_CONFLICT`; digest codes read `E_STAGE_*`.
- **R steps (D3).** The env gets the prefetch `MODULE_ENV.json` merge (`env.r_env_vars`: venv `bin`, the
  module's `PATH`, then the system and native dirs). Before the step, `Rscript --vanilla
  <env.r_session_script> <r_lib> <R/rsession>` runs in the same sandbox with the same env (its out dir
  is the only extra writable dir), `env.read_r_session` reads the result, and every field of
  `env.R_SESSION_FIELDS` (installed versions, every loaded namespace with its `packageVersion`,
  `.libPaths()`, R version, `sessionInfo()`) must equal `SESSION.json`'s, or the step fails with
  `E_R_SESSION` and does not run. The namespace versions catch a namespace-only package (Matrix,
  MASS, nlme) updated in place in the module's site library, which the installed versions of the
  step's own library cannot see. Prefetch records the same fields with the same script
  (`env.r_session_script`) in a fresh `Rscript` after the install, with `TZ=UTC` as the step has: the
  install process itself has loaded namespaces of its own (tools, compiler and the like), so its
  record would never match a run. The reader still drops `sessionInfo()`'s `loaded via a namespace`
  text block, since the namespaces are compared through their name and version map.
- **The sandboxed run.** A template naming a missing placeholder is `E_RENDER_PLACEHOLDER`, a
  malformed one `E_RENDER_BAD`; the cwd must lie in the checkout or `work/` (`E_CWD`). The sandbox
  execs `sh` which `cd`s into the cwd after the mounts (so it is on the read-only bind), then execs
  the venv's `python -I -B -c PREAMBLE <fd> <step argv…>`. The preamble runs the canary, imports torch,
  writes `{sys_prefix, python, torch, torch_cuda, canary_failed}` to the driver's pipe `<fd>`, closes
  it, and `execvp`s the step only when the canary's connect failed. The step therefore never holds the
  pipe, and cannot rewrite what the driver reads to dodge `E_MAX_CUDA` or `E_SYS_PREFIX`. Parsing is
  strict: one line, one object, those five keys, a boolean canary, version-shaped strings; anything
  else, nothing, or a connect that succeeded is `E_CANARY`. A prefix that is not the venv is
  `E_SYS_PREFIX`. The driver drains the pipe in a thread (capped at 1 MiB) so a writer never blocks.
- **After the step.** The inputs are digested again (`E_INPUT_MUTATED`); `verify_checkout` must still
  hold (clean status, `HEAD^{tree}` the prefetch tree: `E_SRC_MUTATED`); the venv marker must still name
  the same `env_key` and overlay and `installed.json` hash to the same value as before the step
  (`E_ENV_CHANGED`), the cross-check of the marker against the `VenvResult` before `installed_sha256`
  is recorded; reserved names are refused. `E_STEP_FAILED` is added only when the step itself ran.
- **Adapter.** Contract: `adapter.py <work dir> <metrics path>`; it reads `work/` (read-only) and
  writes one JSON object to the metrics path, which is `/proc/self/fd/<n>`, the driver's pipe. It runs
  in the driver's interpreter, so `adapter_env_sha256` is `cache.driver_env_sha256()`, in the same
  read-only sandbox with `-I -B` (`-I` ignores `PYTHONDONTWRITEBYTECODE`, so `-B` does that job) and
  its own scratch dir `R/adapter-tmp` as `HOME` and `TMPDIR`, removed afterwards. The runner puts
  senselab's source dir first on `sys.path`. The script is read, re-hashed against the keyed sha
  (`E_ADAPTER_CHANGED`) and copied to `R/adapter/adapter.py`. A non-zero exit is `E_ADAPTER_FAILED`,
  no readable JSON `E_ADAPTER_NO_METRICS`.
- **The adapter import rule, enforced.** The runner first imports every `senselab.utils.reproduce`
  module. It then wraps `builtins.__import__` and `importlib.import_module`: an import made by code
  outside the `senselab` package of a senselab module outside `senselab.utils.reproduce` raises
  `ImportError`, and `from senselab.utils import reproduce` is the only `from` form on a parent
  package that passes. After the adapter returns, any senselab module loaded since the start and
  outside the package also counts. Either case exits 86, read as `E_ADAPTER_IMPORT` whatever the
  adapter did with the exception. This stops the accidental dependency the rule is for. Evasion
  classes it does not stop, all deliberate: reaching an already-loaded module through `sys.modules`
  or as an attribute of `senselab` (for example after `import senselab.utils.reproduce.cache`, whose
  own imports loaded `senselab.utils.file_lock`); loading a module file by path
  (`importlib.util.spec_from_file_location`, `runpy`, `exec` of its source), which loads it under
  another name; `importlib._bootstrap` and other private machinery; and setting the adapter's own
  `__name__` to a `senselab.` name. Adapters live in a reviewed repository.
- **Validation and expectations.** The adapter's object is validated with `writer="adapter"`, then
  `evaluate_expect` writes the results, then the whole object is validated again with
  `writer="driver"` and the same `min_cell`, and only then written to `R/metrics.json`. Only `E_*`
  codes fail a step; `W_*` codes are recorded. An expectation is compared in its own `unit` when it
  gives one, else in the unit of the step's metric entry, converting upstream values (pp = 100 ×
  fraction); per-class rates are fractions. `target`, `tolerance` and `delta` are then written in the
  metric entry's own unit (converted back), so every number in an entry shares the entry's `unit`;
  recording an `expect_unit` instead would have put two units in one entry. `exact` compares with
  `==`, or within a relative 1e-9 when a conversion happened (so 0.7 fraction equals an exact 70 pp,
  which `0.7 * 100` misses by one ulp). The tolerance from `tolerance_from` is `max(a, k·sd)` with the
  sample standard deviation. Results go into the metric's own entry (headline or secondary, or the
  per-class entry); a place addressed by several expectations gets only `pass`; the top level gets the
  step's overall `pass` when there are expectations, so a count or confusion expectation writes only
  its `target`, `tolerance` and `delta` there, and only when it is the only one there. Satra
  (2026-10-07): a classification step may pass on an expectation targeting a secondary metric, as long
  as balanced accuracy is always reported, which the schema already guarantees; no further rule.
  `pass: false` is `E_EXPECT`.
- **Exit and summary.** `summary.json` is `{exit, exc_class, frame, codes, pass, sys_prefix_ok, torch,
  torch_cuda, canary_failed, executed}`. `pass` is null when the step has no expectations or they were
  never evaluated. `exit` is the step's own exit when that is non-zero (97 when the preamble stopped
  it), else 1 when any `E_*` code is present, else 0, so `is_done` and Slurm `afterok` see every
  failure. `exc_class` and `frame` come from the last Python traceback in the step's stderr: a dotted
  class name ending in `Error`, `Exception`, `Interrupt`, `Exit`, `Warning`, `Iteration` or `Timeout`
  (anything else is null, so a message never passes for a class), and the last frame inside the
  checkout as a checkout-relative `file:line`. `executed` is true when the step's own process ran.
- **Order at the end, and provenance's exit.** Outputs are digested (`work/` only; a successful step
  whose pattern matches nothing is `E_OUTPUT_MISSING`), the scratch dirs removed, then the exit is
  fixed and `record_step` is called with it, so the record's `exit_code` is the summary's. If the
  record is refused (`E_PROV_REFUSED`, for a closed-form `ValueError`) or fails (`E_DRIVER`, for an
  `OSError` or `TypeError`), nothing is recorded and the summary's exit becomes non-zero, so the two
  never disagree.
- **Driver failures.** Any `OSError`, `TypeError` or `ValueError` anywhere in `run_step` (log setup,
  staging, `write_summary`, provenance, scratch removal, run links) becomes `E_DRIVER`, with the
  traceback in `driver.log` only. After the reset, a summary is still written when possible (an
  `E_DRIVER` summary if the first write failed). For the whole of `run_step` the `senselab` logger has
  a `NullHandler` and no propagation: without any handler, `logging.lastResort` would print a record
  (and its traceback, which names paths under the runs and results roots) to stderr. A failure caught
  outside the step's `_driver_log` (a run link, for example) is appended straight to `driver.log` when
  the result dir has one, and dropped otherwise. The test runs in a subprocess, because pytest's
  logging plugin installs handlers that hide `lastResort`. In `run_steps`, a `run.json` that cannot
  be written adds `E_DRIVER` and a non-zero exit to every step of that analysis, instead of raising.
- **Statuses and links.** `executed`, `skipped`, `failed` (refused, or the preamble did not hand over)
  and `never-ran` (a need is not executed-or-skipped with exit 0). `run_steps` writes `run.json` with
  each step's key, status, exit and codes and links each result dir. A `never-ran` step, or a failed
  one with no key, removes any same-day `<runs_root>/<date>/<analysis>/<step>` link, which would
  otherwise point at an older result.
- **The test-only `no_netns` mode** drops the network namespace (`sandbox_argv(netns=False)`: `unshare
  -rm`, no loopback step; mounts, the read-only sweep and the capability drop stay). Its runs always
  carry `E_NOT_REPORTABLE` and so a non-zero exit: such a result is never done, never skipped to, and
  never feeds a dependent. Keying the mode instead would have put a test switch into every key.
- **Off-pattern host values (decision).** `provenance.drop_unrecordable_host_values` sets each host
  value `record_step` would refuse to null before recording, and the run records
  `W_HOST_VALUE_DROPPED` (the key names go to `driver.log`). The result stays valid and cacheable: a
  host value describes where the step ran, it is not an input, and refusing would leave one oddly named
  host unable ever to record a run.
- **Provenance venv names.** A derived venv is `repro-<analysis>-<key8>+<overlay sha8>`; the record's
  closed form for venv names accepts exactly that `+<8 hex>` suffix.
- **Logging.** While a step is claimed and run, the `senselab` logger has one handler and does not
  propagate: records are buffered until `R/logs` exists and then appended (`O_NOFOLLOW`) to
  `driver.log`. A claim's stale-takeover warning, which names the lock path under the results tree,
  therefore never reaches a terminal. Before the key exists there is no result dir, and those failures
  return a code only.

## Commands

`python -m senselab.utils.reproduce` takes these subcommands:

- `lock`: compile locks
- `prefetch`: mirrors, worktrees, venvs, wheels, R libraries. `--sources-only` creates worktrees only.
- `build-one`: one sdist build, as a batch job
- `run`: run steps locally, in DAG order; `--step` runs one batch job's step, `--finalize` settles a submission
- `submit`: one batch job per step, chained with `afterok`
- `lint`: check manifests and patches
- `status`: per step: key, executed, skipped, failed, never-ran or pending, exit, codes

Exit codes, for every subcommand: `0` success (for `lint`, no findings; for `run` and `status`, every step
executed or skipped with exit 0); `1` lint findings, a step that did not succeed, or a failed lock,
prefetch, build or submission; `2` bad arguments or an unreadable named file or run dir; `3` (`status`
only) nothing failed but a step is still `pending`. `lint`, `prefetch` and `submit` print one finding per
line, `<where>: <code> <detail>`; a patch finding names the lint code, line and deny-pattern index, never
patch text.

## Batch jobs and the command line (B13)

`slurm.py` and `cli.py`.

- **No key at submission.** A downstream step's key covers its `up:` input digests, which exist only once
  the upstream job has finished (B11 constraint), so `submit` computes no key. Each job runs `run
  <manifests> --step <name> --date <submission date>`. `run_steps(only=…)` runs the named step and
  resolves every step it depends on with `run.resolve_step`: the upstream key is computed again from its
  inputs (a key is never remembered) and its result dir must be done, else the upstream reads
  `E_UPSTREAM_NOT_DONE` and the job's step is `never-ran`. An upstream step is never run by a downstream
  job. The date is fixed at submission, so a job that starts after midnight writes the same run dir.
- **Held submission (B13 review and re-review).** `submit` runs, in this order:
  1. the cross-checks;
  2. rendering of every script (each step's primary and fallback partition, and the finalize job), so a
     render error raises before anything is probed, written or submitted;
  3. the **reservation**: for each analysis in sorted order, under that `run.json`'s lock, it refuses any
     `pending` row of another submission (`E_SUBMIT_OVERLAP`; settle a dead one with `run --finalize
     --submission <id>`) and writes a placeholder `pending` row per step tagged with this submission id (job
     null), remembering the row each replaced. Check and write share one lock, so of two submissions started
     together exactly one reserves a run dir; an overlap found at a later analysis undoes the reservations
     already made (restoring the replaced rows) and nothing is submitted. A test races two threads against a
     fake sbatch that blocks between calls;
  4. the dated log dir (mode `2770`); the partition probes; every step job `--hold` and
     `--comment=<submission id>` in DAG order, each `--dependency=afterok:<needs' job ids>` (in `needs` order)
     with `--kill-on-invalid-dep=yes`; the job ids, filled into this submission's rows under the lock; the
     finalize job (`afterany` every step job, also `--hold` and tagged); one `scontrol release` of all of them.

  Steps 3 and 4 run under `umask 007` with SIGTERM and SIGHUP raising `Terminated`, so an interrupt during
  the reservation also undoes it. Any interrupt prints `interrupted: submission <id>` (the id only) on
  stderr. A slow test runs two real `submit` processes against one run dir on NFS with a fake sbatch on
  `PATH`: exactly one wins.

  Because nothing runs before the release, no upstream can have finished or been purged when a dependent
  is submitted, so an `afterok` dependency is never invalid at submit time. Any exception, Ctrl-C, SIGTERM
  or SIGHUP in step 4 starts the abort, which runs with SIGINT, SIGTERM and SIGHUP ignored (a second signal
  cannot interrupt it; the handlers are restored afterwards): `scancel` of the union of the job ids recorded
  so far (the finalize job included) and the user's jobs `squeue` lists under this submission's `--comment`
  tag, which catches a job slurmctld accepted whose id never reached the driver. The scancel exit is
  checked (a failure is reported as `E_SCANCEL_FAILED` with the leftover job ids, ids only). Then each of this submission's still-`pending` rows gets back the row its
  placeholder replaced, or becomes `never-ran` with `E_SUBMIT_ABORTED`. Rows of other submissions are never
  touched. The command exits 1.
  A job's exit is non-zero exactly when its step is not `StepOutcome.ok`, which is what `afterok` reads.
- **`run.json` under jobs.** Rows are merged, never rewritten, under an exclusive `file_lock` on
  `<run dir>/.run.json.lock` (`run.update_run_record`); a merge keeps a row's `job` and `submission`. Every
  job gets `--submission <id>`, and its own row is written only over a row of that submission or an
  untagged one (a local run's), so a duplicate or stale job cannot overwrite another submission's row. A
  step that never ran carries why: its failed needs' codes, else `E_UPSTREAM_NOT_DONE`
  (`run.upstream_codes`). A job whose step did not succeed turns its dependents' `pending` rows of the same
  submission into `never-ran` with that step's codes.
- **Finalize.** The finalize job (on the `build` partition, 1 CPU, `1G`, 10 minutes), or `run --finalize
  --submission <id>` by hand, first `scancel`s the job ids recorded in that submission's still-`pending`
  rows together with every job `squeue` lists under its `--comment` tag (never the calling job itself).
  From the finalize job this is a no-op, since its dependencies have ended. **By hand on a live submission
  it cancels that submission's queued and running jobs as well**, then settles their rows; that is the way
  to stop a submission. A failing scancel is reported with the ids. The rows are then settled in DAG order:
  a placeholder row (no job id, never submitted) is `never-ran` with `E_SUBMIT_ABORTED`; a step with a need that did not succeed is
  `never-ran` with `E_UPSTREAM_NOT_DONE`; any other step's own job ended without writing its row, so it is
  `never-ran` with `E_JOB_ENDED_WITHOUT_RECORD` and, where `sacct` answers, the job's final state word as
  `slurm_state` (`TIMEOUT`, `NODE_FAIL`, `CANCELLED`). It never touches another submission's rows.
- **Local runs.** A local `run` (full, or `--step` without `--submission`) refuses with `E_SUBMIT_OVERLAP`
  while any submission has `pending` rows in one of its run dirs, and it never rewrites a row a submission
  tagged: a full local run merges its rows into the record (rows of other steps are kept) and leaves tagged
  rows as they are.
- **Requeue.** `never-ran` written by finalize, or by a failed upstream, is not final against a requeued
  job: a requeued job writes its own row when it ends (same submission tag), so its later row replaces the
  `never-ran`, and `status` then shows the step's true state. A job killed after its step started and not
  requeued reads `never-ran`; its result dir is not done, so the next submission reruns it.
- **Pinned manifests.** `submit` hashes each manifest's bytes and parses those same bytes; each job gets
  `--pin <sha256>` per manifest and refuses to run when a file no longer hashes the same: it writes its row
  `failed` with `E_MANIFEST_CHANGED`, turns its dependents' `pending` rows into `never-ran` with the same
  code (where the changed manifests still parse), keeps the record's row order, and exits 1, which cancels
  its dependents. A full local `run` (no
  `--step`) still rewrites the analysis's whole record.
- **Scripts.** Each script goes to sbatch on standard input, so nothing is written to the submit dir.
  `--output`/`--error` are `<slurm_logs_root>/<date>/%j.{out,err}` (spec §2.3 item 6), and every job (step,
  finalize and build) runs there (`--chdir`), with the driver started as `python -P` so that directory is
  not on `sys.path`. The driver creates the dir with mode `2770` and runs sbatch under `umask 007`, so the
  `.out`/`.err` files are never world-readable. Checked with a real trivial job through `submit` from a shell with umask `022`: the
  job's `.out` and `.err` came out `0660`, so the submit-time umask reaches them (through `SLURM_UMASK`). `--export=NONE`, then `set -euo pipefail`, `umask 007`,
  `PATH=/usr/bin:/bin`, and explicitly `UV_CACHE_DIR`, `UV_LINK_MODE=hardlink`, `UV_PYTHON_INSTALL_DIR`,
  `SENSELAB_REPRO_CACHE` and, when the profile declares native lib dirs, `LD_LIBRARY_PATH` (all of them, in
  profile order), because a batch job inherits nothing. These reach the driver only: the step's own env is
  built from scratch by the sandbox. The uv dirs are the submitter's, or what `uv cache dir` and `uv python
  dir` print. A per-job `TMPDIR` under `/dev/shm` is removed on exit. Every value is a plain word, a
  comma-separated list of plain words for the Slurm fields (`gpu:a100:1,shard:1`, a partition list), or
  `shlex`-quoted; anything else is refused at render time and reported by lint first.
- **What the queue shows.** `squeue` and `scontrol show job` are readable by every cluster user. They show
  `JobName=reproduce-<step>` (the analysis name stays out), and the `StdOut`, `StdErr` and `WorkDir` paths
  under `<slurm_logs_root>/<date>/`, which sits in the confidential results area: path names only, never
  contents. The command line (manifest, profile and cache paths, the submission id and pins) is in the
  batch script, which only the owner and admins can read. Manifest rule: analysis and step names must not
  be sensitive.
- **Feasibility probe (spec §15 item 8, amended by the coordinator).** A partition entry with a `fallback`
  is probed with `sbatch --test-only` (the step's script without its dependency lines). The fallback is
  used only when the probe rejects the primary partition (unavailable, or the request cannot be satisfied
  there) or the probe itself errors or times out. A `Preempts:` line lists other jobs ours would preempt;
  on the site partitions both the primary and the fallback have `PreemptMode=OFF`, so our jobs are not
  preempted there, and preempting guest jobs is no reason to fall back. An entry without a fallback is
  never probed; the probe applies to CPU entries too.
- **Cross-checks (`slurm.step_findings` and `slurm.profile_findings`, used by `lint`, `prefetch` and
  `submit`).** `resources.profile` must be a partition key (`partition-unknown`). A `fallback` is a
  partition *name*, not a key (the site profile names another partition there), so it is checked as a plain
  word that differs from its own partition (`fallback-invalid`). A partition, mem, time, gres or fallback
  value that rendering would refuse is `resource-invalid`, so lint catches it first; the profile's log root
  and `build` partition are checked once (`logs-invalid`, `resource-invalid`). `gpu_arch` or `min_driver`
  on an entry with no `gres` is `gpu-no-gres`. Every `inputs`/`stage` root other than `checkout` and `up:`
  and every `{root:<name>}` in argv, cwd, `pythonpath` and stage `dest` must be a profile root
  (`root-unknown`); `native-unknown` as in `spec.check_step_profile`. `submit` refuses on any finding.
  `Name` (analysis and step names) refuses `.`, `..` and a leading dot, since names become directories.
- **`lock`.** Per distinct env of a manifest, the builders run in the order resolve, torch, body, overlays
  (B4 review): for a requirements body `first_resolve` (when the env names one), `torch_lock_from_index`,
  `body_lock_from_requirements`; for a uv-export body `torch_lock_from_uvlock` with the torch names present
  in the checkout's `uv.lock` as roots, then `body_lock_from_uvlock_export` with the `--extra`/`--group`
  options read from the draft's `export_cmd`, which is then replaced by the command that ran; then the `test`
  overlay (the checkout's dev group) and the `notebook_tools` overlay, compiled from the file beside its
  lock with the `.in` suffix. The checkout, needed only for a uv-export body or a test overlay, gets the
  profile's deny patterns. A draft states placeholder sha256s; only the written locks' sha256s are filled
  in. Input sha256s (`requirements_in`, config, overrides) stay the author's, since they are what the
  header digest and the integrity check protect.
- **`prefetch`.** Lints first (refusing on findings), then checks every step out with the profile's deny
  patterns (B8) and calls `env.prefetch_env` once per distinct env **with the profile** (B7), so the pip
  check and smoke imports run and `venv_checked` can become true. A uv-export env with a torch lock gets the
  checkout's `uv.lock`. An env with sdists and no `wheels` records its first `wheels.lock` beside the body
  lock (`<body dir>/wheels.lock`); two such envs with body locks in one directory collide, and the second is
  refused by `env.prefetch_env`, since the file then exists unrecorded. When `PrefetchResult.record` is set,
  the new sha256 is written into `env.wheels` of every step with that env (B6/B7 contract). R steps then get
  `env.prefetch_r` (an `ir_notebook` step with the derived notebook venv). CUDA sdists (`build.needs_nvcc`)
  with no reusable wheel are first built as `build-one` batch jobs on the `build` partition, one at a time
  with `sbatch --wait`, because each build changes `wheels.lock` and so the manifest's sha256 for the next;
  `--local-builds` builds them in-process instead. The wait has no timeout (a CUDA build takes minutes).
  Each build job is named `reproduce-build-<sdist>-<random>` on the sbatch command line, so if the driver
  is interrupted (Ctrl-C, SIGTERM, any exception) it runs `scancel --name` on it: no orphan job finishes
  later and rewrites the manifest. The job gets the `--base-dir` prefetch used. Steps declaring `hf_weights` are refused: downloading
  pinned weights is out of the current scope, and no tier-1 step needs it. `--sources-only` reads a draft
  (one source mapping, a list, or a `{name, steps}` manifest whose steps may lack `env`; only `name`,
  `repo`, `commit` and `submodules` are read) and creates only the cleared worktrees with the empty patch
  set.
- **`build-one`** (the batch entry point `build.render_build_job` names) takes the manifest's `wheels`
  location, or the default above when it has none (refusing an unrecorded existing file), calls
  `build.ensure_wheel` with the wheelhouse `<cache>/wheelhouse` and sdist cache `<cache>/sdists`, and writes
  a changed `wheels.lock` sha256 into the manifest. It gets prefetch's `--base-dir`; the cache comes from
  `SENSELAB_REPRO_CACHE`, which the build job exports.
- **Manifest writes (`spec.rewrite_manifest`).** The edit is applied to the parsed data. When every change
  replaces a plain word (a sha256) occurring once in the file, the text is edited in place and must re-parse
  to the edited data, so comments, anchors and layout survive; otherwise the data is dumped again with
  `yaml.safe_dump(sort_keys=False)` (comments dropped, anchors re-derived), or as indented JSON. The result
  must validate as a `Manifest`, and the file is replaced atomically.
- **Defaults.** `--base-dir` defaults to the git top level of the first manifest's directory; `--cache` to
  `$SENSELAB_REPRO_CACHE`; `--date` to today in UTC.
- **`status <run dir>`** reads `run.json`, and for a row with a key, `<run dir>/<step>/summary.json` when
  the link resolves to the dir named by that key (opened with `O_NOFOLLOW`); the summary's exit and codes
  win. It opens nothing else: no log, no metrics, no output. A step name, key, status or code of an
  unexpected shape prints as `?`.
- **Errors.** `main` maps every error to one `error:` line and exit 1: the message of a usage, lock,
  source, Slurm or validation error, the code of an error that carries one (`cache.CacheError`, digest and
  DAG errors), `interrupted`, and otherwise `E_DRIVER (<exception class>)`. No traceback is printed.
- **`__main__.py`** guards its call to `main`, because the adapter runner imports every module of the
  package and would otherwise run the command line inside the adapter sandbox.
