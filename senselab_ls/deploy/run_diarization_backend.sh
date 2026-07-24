#!/usr/bin/env bash
# Start the diarization backend by hand (systemd is preferred; see diarization.service).
# Loads backend.env from this directory, then launches on PORT (default 9090).
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="${here}/backend.env"
[ -f "$env_file" ] || { echo "missing $env_file (copy backend.env.example and fill in)"; exit 1; }

set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

repo_root="$(cd "${here}/../.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:${repo_root}"
cd "$repo_root"
exec "${LSML_VENV:-/opt/lsml-venv}/bin/label-studio-ml" \
  start senselab_ls/backends/diarization -p "${PORT:-9090}" --host 0.0.0.0
