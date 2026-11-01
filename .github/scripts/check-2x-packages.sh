#!/usr/bin/env bash
# Fails unless packages/ holds every expected package with its pyproject.toml and src/.
set -euo pipefail

expected=(senselab-worker senselab-core)

missing=0
for pkg in "${expected[@]}"; do
  for path in "packages/$pkg/pyproject.toml" "packages/$pkg/src"; do
    if [ ! -e "$path" ]; then
      echo "::error::$path is missing"
      missing=1
    fi
  done
done
exit "$missing"
