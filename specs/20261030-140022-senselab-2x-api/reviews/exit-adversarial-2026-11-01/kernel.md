# Phase 1 exit attack: kernel, identity, gates (target c54e9712)

Probes: p1..p8.py in this directory (run via run.sh against a detached worktree, now removed).

## K-1 (high) Two different local weight sets get the same key: symlinked subdirectories are not hashed
Invariant: a model key is the digest of the weights; different weight sets must not collide (ADR-002, P-4; models.py
`_local_identity`, `_files_under`).
Repro (p1.py): dir a = {top.bin="T", sub -> realsub{w.bin="A"}}, dir b = {top.bin="T", sub -> realsub2{w.bin="B"}}.
`resolve(ModelRef.parse("local:<a>")).identity == resolve("local:<b>").identity` -> True.
Cause: `_files_under` uses `os.walk(path)` with followlinks=False; a symlink to a directory is listed in dirnames
and never in filenames, so every file under it is omitted from `weights_identity`. File symlinks are hashed, dir
symlinks are dropped silently (the weights_identity docstring says links are "hashed by what they point to").
Observed: collision. Expected: identity covers the files reachable, or refuse a directory symlink.

## K-2 (medium) Local directory identity hangs on a FIFO and is not a "weights" identity
Repro (p5.py): `local:<dir>` containing `mkfifo pipe` -> `resolve` blocks forever in `open()` (alarm fired at 3 s).
Same function hashes everything under the dir, not just weight files: the same weights with `.DS_Store` or the
`.cache/huggingface/download/*.metadata` dir that `hf download --local-dir` writes (etag + timestamp) give a different
key from the clean copy (p5: False). A dangling symlink raises FileNotFoundError, not the documented
ModelResolutionError. Expected: only regular files, an explicit ignore rule or manifest, documented errors.

## K-3 (medium) `get_cached_commit_hash` reads `refs/<ref>` with an unvalidated ref: path traversal pins identity from any file
Invariant: an hf key is the hub repo plus the commit the hub/cache vouches for (models.py `resolve`, revision.py).
`ModelRef` accepts any `[^\s@]+` ref. Repro (p8.py): `o/m@../../../victim.txt` with victim.txt = 40 x "c" ->
`resolve(...).identity == "c"*40` with the hub unreachable. Observed: a commit attested from an arbitrary file
under the filesystem. Expected: refs with `..`/leading `/` refused (the hub's ref grammar forbids them).

## K-4 (medium) "Kernel does no I/O" is enforced only as a module blocklist; real I/O passes the contract
Invariant: ADR-009 / `.importlinter` contract kernel-does-no-io.
Repro: add `kernel/zz_probe.py` using builtin `open()`, `pathlib.Path.read_bytes()`, `numpy.fromfile` and
`senselab_worker.digest.file_digest`; `lint-imports --config .importlinter` -> 8 kept, 0 broken.
Cause: forbidden list is soundfile/os/io/shutil/tempfile/importlib/hashing; `pathlib`, `subprocess`, `socket`,
`urllib`, builtin `open`, numpy file readers and `senselab_worker.digest` (kernel already imports it) are not covered,
and `allow_indirect_imports = True`. Today's kernel reads nothing (checked: no open/Path read in kernel/*.py), but nothing
stops the next change. Expected: an AST/behavioural guard (e.g. import with `open`/`Path.*` patched) or a wider contract.

## G-1 (medium) License gate: a package can pass as a "bundled library" or hide a native library
`check_licenses.bundled_libraries` + `policy.bundled` (p2.py):
- Records are keyed by library name only, not (package, file, hash). Any package shipping `libffi-<hash>.so.8`,
  `libopenblas-*.so`, `libgfortran-*.so`, `libsndfile.so` gets that library's MIT/BSD/LGPL/GPL-with-exception record,
  whatever the file actually is.
- Disguise: a shared object whose name matches `_EXTENSION` (`libgpl.cpython-312-x86_64-linux-gnu.so`,
  `libgpl.abi3.so`, any `.pyd`) is treated as a Python extension and needs no record; `.so.1.2-x`, `.SO`, `.DLL`, `.a`,
  `.node`, `.bundle`, framework binaries are not detected at all (all return {} in p2).
- Libraries are read from the RECORD of whatever is installed on the CI platform, so other platforms' wheels
  (delvewheel `.libs`, macOS dylibs) are not seen; `dist.files is None` (no RECORD) silently yields no libraries.
- The license of a package is read from the installed distribution, not the locked version: `dist.version` is never
  compared with the lock, so a relicensed locked version passes if the environment holds an old one.
Expected: record keyed by package+name(+sha256), extension exclusion limited to the package's own extension
modules, installed==locked check.

## G-2 (low) Release gate reads versions from file names only
Repro (p3.py): three wheels+sdists named 2.0.0rc1 whose METADATA says `Name: <x>-LIES`, `Version: 9.9.9.dev0`;
`check_versions(... tag="2.0.0-rc.1", prerelease=True)` and `check_pins` both pass. Only the install step (uv) could
catch it; not run offline here. Expected: METADATA Name/Version must equal the file name's. Also: `--tag` and
`--prerelease` are optional (omitted on workflow_dispatch, which does not publish, so no live hole).
Grammar probes (version_problem/tag_problem, 40 inputs): no bypass found; `1.0.0-rc.1` is a tag-only spelling that
`Version()` equates with `1.0.0rc1`, so two tag spellings name one release (low).

## E-1 (low) Equivalence table parsing accepts silent data loss and unscoped classes
Repro (p6.py): (a) duplicate YAML class key: first class and its derivation vanish, last wins, no error;
(b) backend keys `1` and `'1'` collapse into one, second overwrites first; (c) backend `on`/`yes`/`no` parse as
booleans and become "True"/"False", never matching `key.backend`; (d) overlapping classes (same fingerprint in two classes)
accepted; (e) derivation is any non-blank string (`TODO` passes) and is not in `_equivalence_digest`, so
editing/blanking-by-rewording it changes no recorded digest; (f) a class is scoped to a backend only, not a model or task.
Invariant: "every class must carry a derivation" holds only in form. Expected: a strict loader that rejects duplicate keys
and non-string keys, and overlapping classes.

## S-1 (low) Spec-id check is evadable and incomplete
(p2.py) Not matched: lowercase `f-12`/`h-r12`, `r​12`, non-ASCII hyphen `F‑12`/`F−12`, `x_R12`, `R12a`, `H10`, `r123`,
`re_review`, `full review`, soft-hyphen `re­attack`; `F-１２` (fullwidth digits) IS matched. Id families that exist in the
specs and are not scanned: `D-n` (decisions), `I-n`, `C-n`, `M-n`, `T-n`; none present in packages/ today. Files scanned only by suffix
(no `.sh .html .js .ts .csv`, no-suffix files); `main(argv)` silently skips paths not beginning with `packages/`
(`./packages/x.py`, absolute) and non-existent paths, exit 0. Untracked files are invisible without argv.

## B-1 (low) Fixture and size budgets
Fixture: code files (.py/.pyi/.ipynb) are excluded from the 2 MB total, so any number of 255 KB `.py`/`.ipynb` files
carrying embedded data pass (only the per-file limit applies). Size: `exempt` is keyed on (path, function name), so any
new function of that name in that file is exempt; a function written with `;` on few lines evades the line budget
(ruff statement cap is the only bound); decorators are not counted.

## Duplicates (P-37)
- `check_fixture_budget.load_budget` and `check_size_budgets.load_budget`: same format/derivation/positive-int loop.
- `check_spec_ids.tracked_files` and `check_size_budgets.tracked_python`: same `git ls-files -z packages/` call and split.
- `release_gate.DISTRIBUTIONS`/`DEPENDS_ON` and `check_licenses.DISTRIBUTIONS`: same three-name list in two scripts.
- `check_licenses._norm` and `packaging.utils.canonicalize_name` (already used by release_gate).
- `models.parse_weights` and `identity.parse_equivalence`: both a format-tagged YAML table loader with the same
  checks; `models.verify_weights` and `verify_model` both compare `file_digest` to an expected hash and raise WeightsMismatch.
- `models._files_under` vs the listing a caller would give `weights_identity`; K-1 lives in the one copy.
- `identity._pinned_identity_error` and `ModelIdentity._check_pinned`/`ResolvedModel._check_identity` repeat the
  source-to-pattern choice at three call sites.

## Held (tried, invariant held)
- LogicalKey digest: 2^53+1 vs 2^53, 2^64, 1e21, NaN, bytes all refused; NFC collapses equal; True vs 1, None vs absent,
  case differ; model None vs weights, schema_version differ.
- ModelIdentity/ResolvedModel bypasses (model_copy update, model_construct, setattr, dict) all refused at LogicalKey.
- ModelRef round trip: `#sha256=` inside a url id, trailing `#sha256=`, local/ref mixes all refused.
- Same weights as url (declared hash) and as local file give one key (intended); file vs one-file dir give two (low, by design);
  `Org/N@main` vs `org/n@main` give two keys with one commit (low; hub ids are case-insensitive, forks share commits).
- Release grammar: trailing newline, uppercase, `v` prefix, `.post`, `+local`, epoch, dev, leading zeros, unicode digits refused.
  Pins: duplicate sibling, extras, markers, wildcards refused.
- Kernel reads no file today (grep of kernel/*.py: no open/read/stat; Path in signals.py is a type only).
- Not tested: hatch-vcs/git-describe tag behaviour (setuptools_scm not in the venv, no installs allowed); uv's own
  name/version consistency check in the install step.
