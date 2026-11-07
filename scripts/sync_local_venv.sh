#!/usr/bin/env bash
# The controller's only sync of the shared local .venv: install the locked set and record uv.lock's digest, which
# packages/conftest.py compares with the uv.lock under test (STALE VENV).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
uv sync --locked --all-extras --group dev
venv="$(uv run --no-sync python -c 'import sys; print(sys.prefix)')"
uv run --no-sync python -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  uv.lock > "$venv/senselab-lock.sha"
