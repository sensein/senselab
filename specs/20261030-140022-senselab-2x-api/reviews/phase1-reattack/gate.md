# H8c surface G findings (commit b42d3cf0, macOS)

Probe scripts: probes/p1..p14.py, harness probes/h.py. Throwaway tagged repos: ws, ws2, ws3, fbrepo (all under this dir; no real-repo tags touched).
Reproductions of gate cases: `h.fresh(name)` copies a real `uv build --all-packages` dist (tag 2.0.0 in throwaway ws), `rewrite_wheel/rewrite_sdist` mutate it, `run()` calls `release_gate.main` (full install) or the static checks.

## XG-1 (high) any module outside the digest allowlist can hash serialised data
Invariant: "No first-party module outside the allowlist can produce a digest of serialised data" (H-R17).
The data-flow check (`untagged_digests`) runs only over ALLOWED modules; the capability check only looks for imports of hashlib-like modules. A non-allowlisted module imports the allowlisted helper instead.
Repro (planted `src/senselab/zz/probe_a.py`, then `pytest packages/senselab-core/tests/lint/tagged_digest_test.py`): 61 passed.
```
from senselab_worker.hashing import sha256_bytes
def record_id(x): return sha256_bytes(json.dumps(x, sort_keys=True).encode())
import uuid; uuid.uuid5(uuid.NAMESPACE_URL, json.dumps(x))   # sha1 of serialised data, no capability import
```
Expected: reported. Observed: passes. Also not capabilities: uuid, joblib.hash, zlib/binascii, sklearn murmurhash, dask.tokenize.
Regression of closed A-row: no (H-R17 design gap).

## XG-2 (medium) strict serialisation->digest flow misses ordinary code inside allowlisted modules
`untagged_digests(src, "packages/a.py", strict=True)` returns nothing for each of (p4.py):
`hashlib.sha384/sha224/sha3_512/shake_256(json.dumps(x).encode())` (HASH_NAMES lacks them), `hashlib.file_digest(BytesIO(json.dumps(x).encode()),"sha256")`, `hmac.digest(k, json.dumps(x).encode(), "sha256")`,
`d: bytes = json.dumps(x).encode(); sha256(d)` (AnnAssign), tuple-assign, augmented assign, `self.d = ...`, helper `def ser(x): return json.dumps(x).encode()` then `sha256(ser(x))`,
`m.model_dump_json().encode()`, `f"{x}".encode()`, `",".join(...)`, unannotated hasher parameter `def feed(h, x): h.update(json.dumps(x).encode())`, `n = m; n.update(...)`, `mk().update(...)`.
Applies to hashing.py/provenance.py/canonical.py, the three allowlisted 2.x modules, so a contributor edit passes.

## XG-3 (medium) capability_imports misses plain re-spellings
Not found (p4.py): `importlib.import_module(name="hashlib")` (keyword; the moved-module guard handles `name=`, this one does not), `from importlib import import_module; im, z = import_module, 1`, annotated alias `im: object = import_module`,
`functools.partial(import_module)`, `getattr(importlib, "import_module")`, dict-held importer, `from senselab_worker.hashing import hashlib` (re-export), `sys.modules[...]`, `importlib.util.find_spec/module_from_spec`, `exec("import hashlib")`.
Also DYNAMIC_ALLOWED is a set of argument strings: a second `import_module(module.name)` site in an allowlisted file is invisible.

## XG-4 (high) release gate accepts an epoch version, defeating the 1.x floor and tag binding
Invariant: H-R24 "floor 2.0.0a0 (1.x refused)", version bound to ref_name.
Repro (ws2, throwaway): `git tag '1!1.0.0'; uv build --all-packages; release_gate.py dist --tag '1!1.0.0' --prerelease false` -> "release gate passed", rc=0. Artifacts are senselab-1!1.0.0*. Epoch 1 outranks every 2.x release permanently on PyPI.
Expected: refuse `version.epoch != 0`. Observed: pass (check_versions compares Version objects, epoch dominates `< floor`).

## XG-5 (medium) gate cannot see missing files: no completeness check, import smoke is shallow
Wheel of senselab-core without `senselab_core/kernel/identity.py`, `testing/synth.py`, `cli.py`, `runtime/storage.py` -> full gate passes in a clean env (p3.py): contents check only proves members are tracked, never that tracked package files are present; the smoke is `import senselab_core`, which does not import kernel/runtime. A tracked file silently dropped by hatch (e.g. matched by .gitignore, force-added) ships broken.

## XG-6 (medium) gate runs in the caller's environment beyond UV_*/PIP_*/VIRTUAL_ENV
`gate_environment` leaves PYTHONPATH, PYTHONHOME, GIT_* etc.
(a) p8.py: worker wheel without `senselab_worker/canonical.py` is refused in a clean env, but with `PYTHONPATH=<source trees>` the gate passes (import resolves from source, not the installed dist).
(b) p9.py: untracked `evil.py` inside a wheel is refused; with `GIT_INDEX_FILE=<alt index containing it>` exported, `_tracked()` honours it and the gate passes.
Invariant violated: "installed standalone ... from dist bytes only, whatever the caches or environment hold".

## XG-7 (medium) sdists are barely gated
- sdist PKG-INFO `Requires-Dist` is never checked: replacing `senselab-worker==2.0.0` with `senselab-worker>=0` plus an extra `evilpkg` in the core sdist passes (p2.py, full gate). PyPI serves sdist PKG-INFO to resolvers.
- sdist filenames unbound: `senselab_core-2.0.0.tar.gz` renamed to `anything-else-7.tar.gz` passes; the core and worker sdists with swapped filenames pass.
- sdists are never built or installed by the gate (only wheels).
- sdist member top directory is not bound: member `othertop/pyproject.toml` (tracked content) passes.

## XG-8 (medium) wheel/METADATA/WHEEL are generated files with free content; filename tags unchecked
All pass the full gate (p1.py, p13.py): wheel renamed `senselab-2.0.0-py2.py3-none-any.whl` while WHEEL says py3-none-any; WHEEL rewritten to `Root-Is-Purelib: false`, `Tag: cp39-cp39-win32` under a py3-none-any filename; duplicate `Version:` and `Name:` headers in METADATA; METADATA `Requires-Python: >=3.6`, an extra third-party `Requires-Dist: evilpkg`, `Provides-Extra: x` (METADATA is not compared with the pyproject).
PyPI rejects tag/filename mismatches and duplicate Version headers; `uv publish dist/*` uploads sequentially, so the failure leaves an incomplete set published. Expected: parse with packaging.metadata (validate) and compare WHEEL Tag to the filename, METADATA to the pyproject.

## XG-9 (low) tar members `../x` and `/x` with tracked content pass
`_candidates` partitions on the first `/` and never checks the top component: sdist members `../pyproject.toml`, `/pyproject.toml`, `../README.md` (identical to tracked files) pass for the root sdist (p2.py). Extractors without a data filter write outside the target. (Wheel `..`, absolute and case-duplicate members are refused; held.)

## XG-10 (medium) fixture budget leaks
(p5.py, p6.py)
- `.mjs` is in SOURCE_SUFFIXES (not a fixture) but not in CODE_SUFFIXES (not code): 40 x 250 KB `.mjs` under `src/tests/` = 10 MB, no total applies; same with `.json` is refused.
- `generated_files` glob `src/senselab/utils/data/venv_locks/*.txt`: fnmatch `*` crosses `/`; a 20 MB `venv_locks/sub/payload.txt` and `venv_locks/payload.txt` pass. No check that the file is a lock.
- Tracked gitlink (mode 160000) `src/tests/data/bigsubmodule` with 30 MB inside: rc 0 (only is_file/is_symlink paths are scanned).
- Mode 120000 is never consulted: tracked symlink absent from the worktree passes (rc 0); on a checkout with core.symlinks=false it is a regular file and passes.
- Totals apply only to the test trees: 60 x 250 KB files under `src/senselab/audio/data/` (15 MB) pass; so do many 500 KB `.py` files outside test trees.
- Policy file `scripts/data/fixture_budget.yaml` (caps, generated globs) is not bound to the freeze branch; only the allowlist is.

## XG-11 (medium) workflow interpolation guard inspects only `run`/`script` keys
(p7.py) `interpolation_offenders` returns [] for `${{ github.event.pull_request.title }}` in: `shell: bash -c "..." {0}`, `with.command`, `with.args` (docker), azure `with.inlineScript`, `working-directory`, a `uses:` path. `matrix.python-version` is allowlisted in `run` but the matrix definition is never checked, so `matrix: python-version: ${{ fromJSON(github.event.pull_request.title) }}` makes the allowlisted expression untrusted.

## XG-12 (medium) moved-module guard: red at HEAD, and bypassable
- At b42d3cf0 `src/tests/utils/moved_modules_test.py::test_no_moved_module_paths_remain` FAILS (run with --noconftest, no planted files): `packages/senselab-core/tests/adversarial/phase1_regression_test.py: senselab/utils/file_lock` (line 91, added by b42d3cf0). The guard excludes only its own file. Regression of the guard by the A-row tests.
- Bypasses (p14.py): `importlib.import_module(".file_lock", "senselab.utils")`, `import_module(name=".model_revision", package="senselab.utils")`, `getattr(senselab.utils, "prov_store")` -> no hit. Non-`.py`/`.md` files (`.sh`, `.yaml`, pyproject entry points: `python -m senselab.utils.cuda_probe`) are never text-scanned.

## What held
- sibling twice in dist (second wheel with build tag, second sdist, any filename): refused (count != 1).
- wheel members: `..`, absolute, case-variant duplicates, duplicate names, extra dist-info files (INSTALLER, licenses/*), untracked file, __pycache__: refused.
- tar: symlink/hardlink members, second PKG-INFO, nested `PKG-INFO`: refused.
- pins: extras, marker, URL, loose `>=`, duplicate pin, extra sibling edge, `==2.0.0.0`/spaces (equal version): handled correctly.
- tag/prerelease/dev/post/local/floor mismatches: refused; swapped wheel filenames, wheel filename version != metadata: refused (install or stem check).
- tracked symlink in package tree: hatch failed to build the wheel, gate refused.
- publish.yaml: gate precedes publish, upload only on `release`, dispatch gate-only, ref_name/prerelease via env; `--tag` value starting `--` fails closed.
- import-linter: 4 contracts kept; dependencies_test catches undeclared imports in worker/core.
- hatch pin hook: local version refused; loader prefers the in-sdist copy, content bound by the gate alias check.
- senselab_core exact-version check: editable exemption matches H-R22 (mixed editable/wheel mismatch accepted by ruling).
