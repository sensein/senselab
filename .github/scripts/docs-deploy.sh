#!/usr/bin/env bash
# Replace one part of the `docs` branch and push it as a single orphan commit.
#
# Usage (run inside a clone whose remote can push to `docs`):
#   docs-deploy.sh [--namespace <ns>] preview <site-dir> <pr-number>   replace <ns>/pr-<N>/ with <site-dir>
#   docs-deploy.sh release <site-dir>               replace everything except pr-*/ and the namespace dirs
#   docs-deploy.sh [--namespace <ns>] cleanup <open-pr-number>...   remove every <ns>/pr-<N>/ whose N is not listed
#
# Without --namespace, previews live at the root as pr-<N>/ (1.x). A namespace is a top-level directory
# holding its own pr-<N>/ previews (`2x` for the 2.x line). A cleanup touches only its own namespace's
# previews, and a release keeps every namespace directory as it is. Namespaces are listed in
# `namespaces` below; a namespace not listed there is refused.
#
# Each attempt reads the current tip of `docs`, rebuilds the tree from it with only this
# deploy's part replaced, commits that tree with no parent, and pushes with
# --force-with-lease pinned to the tip it read. A lease failure means another deploy moved
# the branch in between; the attempt is redone on the new tip.
#
# Environment:
#   DOCS_REMOTE               remote to fetch from and push to (default: origin)
#   DOCS_BRANCH               branch to deploy to (default: docs)
#   DOCS_DEPLOY_ATTEMPTS      attempts before giving up (default: 8)
#   DOCS_DEPLOY_BACKOFF       base backoff in seconds; attempt k sleeps k*base plus up to base jitter (default: 2)
#   DOCS_DEPLOY_BEFORE_PUSH   test hook: command evaluated after the tree is built and before the push
set -euo pipefail

remote=${DOCS_REMOTE:-origin}
branch=${DOCS_BRANCH:-docs}
attempts=${DOCS_DEPLOY_ATTEMPTS:-8}
backoff=${DOCS_DEPLOY_BACKOFF:-2}
namespaces=(2x)

: "${GIT_AUTHOR_NAME:=github-actions[bot]}"
: "${GIT_AUTHOR_EMAIL:=github-actions[bot]@users.noreply.github.com}"
: "${GIT_COMMITTER_NAME:=$GIT_AUTHOR_NAME}"
: "${GIT_COMMITTER_EMAIL:=$GIT_AUTHOR_EMAIL}"
export GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL

die() {
  echo "docs-deploy: $*" >&2
  exit 1
}

usage="usage: $0 [--namespace <ns>] preview <site-dir> <pr-number> | release <site-dir> | [--namespace <ns>] cleanup <open-pr-number>..."

ns=""
if [ "${1:-}" = --namespace ]; then
  [ $# -ge 2 ] || die "$usage"
  ns=$2
  shift 2
  known=0
  for candidate in "${namespaces[@]}"; do
    if [ "$candidate" = "$ns" ]; then known=1; fi
  done
  [ "$known" = 1 ] || die "unknown namespace '$ns' (known: ${namespaces[*]})"
fi
# Path prefix of this namespace's previews: "" at the root, "<ns>/" inside a namespace.
prefix=${ns:+$ns/}

mode=${1:-}
[ -n "$mode" ] || die "$usage"
shift

site_tree=""
case "$mode" in
  preview)
    [ $# -eq 2 ] || die "preview takes <site-dir> <pr-number>"
    site_dir=$1
    pr=$2
    [[ "$pr" =~ ^[0-9]+$ ]] || die "pr number must be numeric, got '$pr'"
    message="Docs site: ${ns:+$ns }preview of PR #$pr"
    ;;
  release)
    [ -z "$ns" ] || die "release takes no namespace"
    [ $# -eq 1 ] || die "release takes <site-dir>"
    site_dir=$1
    message="Docs site: release"
    ;;
  cleanup)
    open_prs=" $* "
    message="Docs site with ${ns:+$ns }previews of open PRs only"
    ;;
  *)
    die "unknown mode '$mode'"
    ;;
esac

scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
export GIT_INDEX_FILE="$scratch/index"

if [ "$mode" != cleanup ]; then
  [ -d "$site_dir" ] || die "site dir '$site_dir' does not exist"
  site_dir=$(cd "$site_dir" && pwd)
  # Hash the built site into a tree object once; every attempt reuses it.
  rm -f "$GIT_INDEX_FILE"
  git --work-tree="$site_dir" add -A --force .
  site_tree=$(git write-tree)
fi

# Prints the tip of the remote branch, or nothing when the branch does not exist yet.
fetch_tip() {
  if git ls-remote --exit-code --heads "$remote" "$branch" >/dev/null 2>&1; then
    git fetch --quiet --no-tags --depth=1 "$remote" "+refs/heads/$branch:refs/docs-deploy/tip" >&2
    git rev-parse refs/docs-deploy/tip
  fi
}

# Entry names directly under <tree-ish>:<dir> ($1, $2; $2 empty for the root) that match pr-<digits>.
pr_dirs() {
  local tree=$1 dir=${2%/}
  if [ -n "$dir" ]; then
    git cat-file -e "$tree:$dir" 2>/dev/null || return 0
    tree="$tree:$dir"
  fi
  git ls-tree --name-only "$tree" | grep -E '^pr-[0-9]+$' || true
}

# Top-level entries the release site never owns: root previews and namespace directories.
preserved_dirs() {
  pr_dirs "$1" ""
  local name
  for name in "${namespaces[@]}"; do
    if git cat-file -e "$1:$name" 2>/dev/null; then echo "$name"; fi
  done
}

# Builds the new root tree from tip $1 (may be empty) and prints its id.
build_tree() {
  local tip=$1 name num
  rm -f "$GIT_INDEX_FILE"
  case "$mode" in
    preview)
      if [ -n "$tip" ]; then
        git read-tree "$tip"
        git rm -r --cached --quiet --ignore-unmatch -- "${prefix}pr-$pr" >/dev/null
      else
        git read-tree --empty
      fi
      git read-tree --prefix="${prefix}pr-$pr/" "$site_tree"
      ;;
    release)
      git read-tree "$site_tree"
      while read -r name; do
        [ -n "$name" ] || continue
        git rm -r --cached --quiet --ignore-unmatch -- "$name" >/dev/null
      done < <(preserved_dirs "$site_tree")
      if [ -n "$tip" ]; then
        while read -r name; do
          [ -n "$name" ] || continue
          git read-tree --prefix="$name/" "$tip:$name"
        done < <(preserved_dirs "$tip")
      fi
      ;;
    cleanup)
      [ -n "$tip" ] || return 0
      git read-tree "$tip"
      while read -r name; do
        [ -n "$name" ] || continue
        num=${name#pr-}
        if [[ "$open_prs" != *" $num "* ]]; then
          git rm -r --cached --quiet -- "$prefix$name" >/dev/null
          echo "docs-deploy: removing $prefix$name" >&2
        fi
      done < <(pr_dirs "$tip" "$ns")
      ;;
  esac
  git write-tree
}

for ((attempt = 1; attempt <= attempts; attempt++)); do
  tip=$(fetch_tip)
  tree=$(build_tree "$tip")
  if [ -z "$tree" ] || { [ -n "$tip" ] && [ "$tree" = "$(git rev-parse "$tip^{tree}")" ]; }; then
    echo "docs-deploy: $branch already up to date"
    exit 0
  fi
  commit=$(git commit-tree "$tree" -m "$message")
  if [ -n "${DOCS_DEPLOY_BEFORE_PUSH:-}" ]; then
    export DOCS_DEPLOY_ATTEMPT=$attempt
    eval "$DOCS_DEPLOY_BEFORE_PUSH"
  fi
  if push_out=$(git push --porcelain --force-with-lease="refs/heads/$branch:$tip" "$remote" \
    "$commit:refs/heads/$branch" 2>&1); then
    echo "docs-deploy: pushed $commit to $branch (attempt $attempt)"
    exit 0
  fi
  echo "$push_out" >&2
  if ! grep -qE 'stale info|fetch first|non-fast-forward|cannot lock ref|failed to update ref|incorrect old value' <<<"$push_out"; then
    die "push failed for a reason other than a moved branch"
  fi
  if ((attempt < attempts)); then
    delay=$((attempt * backoff + RANDOM % (backoff + 1)))
    echo "docs-deploy: $branch moved since ${tip:-<none>}; retrying in ${delay}s" >&2
    sleep "$delay"
  fi
done
die "gave up after $attempts attempts"
