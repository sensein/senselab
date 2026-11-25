#!/usr/bin/env bash
# Sets `present` in $GITHUB_OUTPUT for the 2.x workflows.
#   no packages/ directory                               -> present=false, exit 0
#   packages/ with every expected package and its src/   -> present=true, exit 0
#   packages/ missing an expected package or its src/    -> exit 1
set -euo pipefail

expected=(senselab-worker senselab-core)
out=${GITHUB_OUTPUT:-/dev/stdout}

if [ ! -e packages ]; then
  echo "present=false" >> "$out"
  echo "::notice::no packages/ tree on this ref; the 2.x package jobs skip"
  exit 0
fi

missing=0
for pkg in "${expected[@]}"; do
  for path in "packages/$pkg/pyproject.toml" "packages/$pkg/src"; do
    if [ ! -e "$path" ]; then
      echo "::error::packages/ exists but $path is missing"
      missing=1
    fi
  done
done
[ "$missing" = 0 ] || exit 1
echo "present=true" >> "$out"
