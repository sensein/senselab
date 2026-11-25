#!/usr/bin/env bash
# Fetches the 1.x freeze branch, whose commit the fixture budget check compares against.
set -euo pipefail
git fetch --no-tags --depth=1 origin freeze-1x-2026-10-30:refs/remotes/origin/freeze-1x-2026-10-30
