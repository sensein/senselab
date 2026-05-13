# Phase 0 — Research: NeMo Canary torch/torchaudio CUDA mismatch

## D1 — How to detect host CUDA without importing `torch`

**Decision**: Probe order is `nvidia-smi` (no args; parse `CUDA Version: X.Y` from the default header line) → `nvcc --version` (parse `release X.Y`) → "no CUDA, use CPU index".

**Rationale**: At `ensure_venv` time the target venv doesn't exist yet, so `import torch` from inside it is impossible. The host's main venv may have a `torch` build that disagrees with the host's runtime CUDA (e.g. CPU-only torch pinned by senselab's main extras), so trusting `torch.version.cuda` from the parent process is wrong. `nvidia-smi`'s default output prints `CUDA Version: X.Y` on the header line — direct, no driver→CUDA mapping table to maintain. `nvcc` is a developer-toolkit binary; useful as a fallback in driver-only installs that have CUDA toolkit linked through `LD_LIBRARY_PATH`. Both probes report the CUDA version as a string we can parse — no mapping required.

**Alternatives considered**:
- `nvidia-smi --query-gpu=driver_version` + driver→CUDA lookup table. Rejected — requires maintaining an NVIDIA driver→CUDA mapping that lags real driver releases; `nvidia-smi`'s default output already prints CUDA version directly with zero mapping work.
- Parsing `/usr/local/cuda/version.txt` — fragile, distro-dependent, doesn't work in containers with bind-mounted CUDA.
- Reading `LD_LIBRARY_PATH` for `libcudart.so.X` — fragile and ambiguous; multiple cudart libraries are common.
- Spinning up a one-shot venv with just `torch` to probe — slow, defeats the purpose.

## D2 — Mapping host CUDA to a PyTorch wheel index

**Decision**: Static map of supported indexes, sorted high-to-low: `cu128`, `cu126`, `cu124`, `cu121`. Pick the highest index whose CUDA version `≤` host CUDA. No match → `cpu`. Host CUDA 12.9 picks `cu128`.

**Rationale**: PyTorch's `cuXX` wheels link against CUDA toolkit XX but are forward-compatible to newer driver CUDA via the NVIDIA driver's built-in compat shim. The static list is short and exact, avoiding any HTTP probe. It must be updated by hand when PyTorch publishes a new index (e.g. `cu129`) — that's an explicit one-line PR.

**Alternatives considered**:
- HTTP-probe the index URL listings at install time — adds network dependency and is slower than the static list; the savings of not maintaining the list don't outweigh the cost.
- Use PyPI's `python_version` + `platform_tag` resolution alone — that's what causes today's bug; PyPI's torch wheel selection doesn't enforce same-toolchain matching of torch+torchaudio.

## D3 — Plumbing the index URL into `uv pip install`

**Decision**: Inside `ensure_venv`, prefix the existing `uv pip install --python <venv-python> <reqs>` argv with `--index-url <chosen>` and `--extra-index-url https://pypi.org/simple`.

**Rationale**: Single call site; applies uniformly to every subprocess backend. The `--extra-index-url` keeps `nemo_toolkit[asr,tts]` etc. resolvable from PyPI while `torch`/`torchaudio` go through the PyTorch index. uv resolves both in one atomic pass.

**Alternatives considered**:
- Two `uv pip install` calls (torch via special index, everything else from PyPI). Rejected — increases install time; non-atomic; uv could end up with two different `torch` builds in the same env if a transitive requirement also names `torch`.
- Per-package index pinning via PEP 631 / `--index-strategy unsafe-first-match`. Rejected — more knobs than needed; uv's `--index-url + --extra-index-url` is the documented pattern for this use case.

## D4 — Communicating "no compatible CUDA wheels exist"

**Decision**: Attempt the install once. On `uv pip install` failure with a recognizable signature (no matching distribution, no compatible wheel), wrap the error into `SenselabCudaCompatibilityError` carrying:
- detected CUDA version,
- attempted index URL,
- failing package(s) (parsed from uv's stderr),
- one-line action: "downgrade CUDA, fall back to CPU by setting `SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`, or wait for upstream wheels".

**Rationale**: Satisfies FR-004 (single named actionable error) and SC-003 (no deep stack traces). Reusing the `uv pip install` failure as the trigger means we don't have to maintain a separate "is this wheel available" probe.

**Alternatives considered**:
- Pre-flight HEAD-request to the index URL — adds a network call, still doesn't catch "index exists but specific version missing" cases.
- Trial-and-error fallback to CPU when CUDA index fails. Rejected — silent CPU fallback is the wrong default; the user should be told explicitly so they can decide.

## D5 — Stale marker / partial-install recovery

**Decision**: Add a `torch_index` field to the marker dict. Absence of `torch_index` in an existing marker counts as mismatch → rebuild. No explicit `schema_version` field — the field-presence check is sufficient and the marker is internal state, not a public contract that needs versioning.

**Rationale**: Satisfies FR-005 and SC-004. Reuses the existing `stored.get("requirements") == sorted(requirements)` rebuild path with one additional clause. No upgrade-compatibility ceremony needed for an internal cache file.

**Alternatives considered**:
- Try `import torch` from inside the venv and catch failures. Rejected — brittle in subtle import-error landscape; would need a maintenance burden to keep up with different failure modes.

## D6 — Env-var override

**Decision**: `SENSELAB_TORCH_INDEX_URL`. When set, used verbatim as `--index-url`. When unset, the static map (D2) applies. The resolved override is recorded in the marker so changing the env var triggers a venv rebuild.

**Rationale**: Constitution VIII (No Hardcoded Parameters) — operators on air-gapped hosts or internal PyPI mirrors can route through their own index without code change. Standard pattern in Python tooling.

**Alternatives considered**:
- A config file at `~/.config/senselab/torch_index.toml`. Rejected — heavier than necessary; env var is sufficient and matches the existing project pattern (`HF_TOKEN`, etc.).

## Open questions (none blocking)

None. All six decisions are resolved by this research pass. The implementation can proceed directly to Phase 2 (`/speckit.tasks`).

## Out-of-scope notes

- **Multiple CUDA versions on one host**: e.g. user has both CUDA 12.4 and CUDA 12.9 installed and wants senselab to use the older. `SENSELAB_TORCH_INDEX_URL` (D6) is the documented escape hatch.
- **AMD ROCm**: PyTorch publishes `rocm6.x` wheels; out of scope for this fix. The static map (D2) can be extended with ROCm entries when the project starts targeting AMD hardware.
- **Aarch64 (e.g. Grace-Hopper)**: PyTorch's CUDA wheels are mostly x86_64. ARM CUDA hosts are out of scope; an ARM CUDA user can set `SENSELAB_TORCH_INDEX_URL` to point at NVIDIA's PyPI mirror.
