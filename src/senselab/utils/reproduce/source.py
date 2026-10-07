"""Pinned checkouts: bare mirrors, cleared notebook outputs, verified patches, detached worktrees.

`mirror` keeps a bare mirror of a repository at ``<cache>/src/<name>``, and of every submodule
named at the pinned commit. Upstream branches are fetched into ``refs/upstream/heads/*``, tags into
``refs/tags/*``, and a commit reachable from neither into ``refs/upstream/pinned/<commit>``.
Fetches never prune, so the throwaway ``refs/heads/repro/*`` refs survive every refresh.

`checkout` gives one detached worktree per (commit, patch-set sha256) at
``<cache>/wt/<name>-<commit[:8]>-<patch set[:8]>``. On top of ``commit`` it commits
``repro: clear outputs`` (every tracked notebook with outputs dropped and execution counts nulled,
as plain JSON), then one ``repro: <patch name>`` commit per patch, and points
``refs/heads/repro/<commit[:8]>-<patch set[:8]>`` at the result with ``git update-ref``. The refs
live only in the cache mirror. After each patch no tracked notebook may carry outputs. Submodules
are worktrees of their own mirrors at the recorded gitlinks; their notebooks are never cleared, so
one with outputs is refused unless the caller allows it. A sidecar ``<worktree>.json`` marks a
complete worktree; a complete worktree is verified and reused, an incomplete one is removed and
rebuilt, under a `SharedFileLock` on the worktree path.

Every git command runs with an allowlisted environment, no system or global config, hooks
disabled, and a fixed identity and date, so the commit and tree shas of a checkout are the same in
every cache. Errors carry a fixed code (`ReproduceSourceError.code`); the message never quotes
patch or notebook content.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, NamedTuple

from senselab.utils.file_lock import SharedFileLock
from senselab.utils.reproduce.locks import ReproduceLockError, verify_files
from senselab.utils.reproduce.spec import FileRef, SourceSpec, StepSpec

SRC = "src"
WT = "wt"
REPRO_REF_PREFIX = "refs/heads/repro/"
UPSTREAM_HEADS = "refs/upstream/heads/"
UPSTREAM_PINNED = "refs/upstream/pinned/"
PATCH_SET_SCHEMA = "reproduce-patch-set-1"
MARKER_SCHEMA = "reproduce-checkout-2"
CLEAR_MESSAGE = "repro: clear outputs"
SYSTEM_PATH = "/usr/bin:/bin"
REPRO_NAME = "reproduce"
REPRO_EMAIL = "reproduce@localhost"
REPRO_EPOCH = 315532800
REPRO_DATE = f"@{REPRO_EPOCH} +0000"
REPRO_IDENTITY_LOG = f"{REPRO_NAME} <{REPRO_EMAIL}> {REPRO_EPOCH}|{REPRO_NAME} <{REPRO_EMAIL}> {REPRO_EPOCH}"
FETCH_PASSTHROUGH = (
    "SSH_AUTH_SOCK",
    "GIT_SSH_COMMAND",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
)
GIT_OPTIONS = (
    "core.hooksPath=/dev/null",
    "core.fsmonitor=false",
    "core.autocrlf=false",
    "commit.gpgsign=false",
    "gc.auto=0",
    "fetch.prune=false",
    "fetch.pruneTags=false",
    "advice.detachedHead=false",
    "init.defaultBranch=main",
)

GIT_FAILED = "git-failed"
COMMIT_MISSING = "commit-missing"
MIRROR_URL_CONFLICT = "mirror-url-conflict"
BAD_REPO_NAME = "bad-repo-name"
SUBMODULE_UNKNOWN = "submodule-unknown"
NOTEBOOK_UNREADABLE = "notebook-unreadable"
PATCH_DOES_NOT_APPLY = "patch-does-not-apply"
PATCH_LINT = "patch-lint"
PATCHES_NEED_BASE_DIR = "patches-need-base-dir"
HEAD_MISMATCH = "head-mismatch"
CHAIN_MISMATCH = "chain-mismatch"
TREE_MISMATCH = "tree-mismatch"
WORKTREE_DIRTY = "worktree-dirty"
WORKTREE_MISMATCH = "worktree-mismatch"
WORKTREE_MISSING = "worktree-missing"
REF_MISSING = "ref-missing"
DENY_PATTERN_INVALID = "deny-pattern-invalid"
SUBMODULE_NOTEBOOK_OUTPUTS = "submodule-notebook-outputs"

LINT_NOTEBOOK_OUTPUTS = "notebook-outputs"
LINT_DENY_PATTERN = "deny-pattern"
LINT_BINARY = "binary-patch"
LINT_SYMLINK = "symlink"

_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_OUTPUTS_RE = re.compile(r'^(\s*)"outputs"\s*:\s*\[(.*)$')
_OUTPUT_TYPE_RE = re.compile(r'"output_type"\s*:')
_HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")
_TAIL = 2000
_SYMLINK_MODE_RE = re.compile(r"^(new file mode|new mode) 120000\s*$")
_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


class ReproduceSourceError(RuntimeError):
    """A checkout, patch or lint failure, identified by a fixed ``code``.

    ``detail`` holds the git output for the driver log; ``str()`` never includes it.
    """

    def __init__(self, code: str, message: str = "", detail: str = "") -> None:
        """Record the code, a content-free message and the detail for the log."""
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {message}" if message else code)


class Checkout(NamedTuple):
    """A prepared worktree and what it was built from."""

    worktree: Path
    mirror: Path
    repo: str
    commit: str
    patch_set_sha256: str
    patches: list[tuple[str, str]]
    ref: str
    head: str
    tree_sha: str
    submodules: dict[str, str]
    submodule_notebooks_with_outputs: list[str]


class LintFinding(NamedTuple):
    """One lint hit: a fixed code, a 1-based patch line, and the deny pattern's index (else -1)."""

    code: str
    line: int
    pattern: int = -1


# --- git ---------------------------------------------------------------------------------------


def git_environment(home: Path, *, network: bool = False) -> dict[str, str]:
    """The allowlisted environment for every git subprocess.

    Local operations get a private ``home``. Network operations keep the caller's ``HOME`` (ssh
    reads its keys and known hosts there) and the variables in `FETCH_PASSTHROUGH`. Either way git
    reads no system or global config, and identity and dates are fixed.

    Args:
        home: A private directory for ``HOME`` and ``XDG_CONFIG_HOME``.
        network: Whether the command talks to a remote.

    Returns:
        A new mapping.
    """
    (home / ".config").mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": SYSTEM_PATH,
        "HOME": str(home),
        "XDG_CONFIG_HOME": str(home / ".config"),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": REPRO_NAME,
        "GIT_AUTHOR_EMAIL": REPRO_EMAIL,
        "GIT_AUTHOR_DATE": REPRO_DATE,
        "GIT_COMMITTER_NAME": REPRO_NAME,
        "GIT_COMMITTER_EMAIL": REPRO_EMAIL,
        "GIT_COMMITTER_DATE": REPRO_DATE,
    }
    if network:
        if os.environ.get("HOME"):
            env["HOME"] = os.environ["HOME"]
        env.update({k: os.environ[k] for k in FETCH_PASSTHROUGH if os.environ.get(k)})
    return env


class _Git:
    """git bound to the cache's private home."""

    def __init__(self, cache: Path) -> None:
        self.home = cache / ".git-home"

    def run(
        self, cwd: Path, *args: str, stdin: bytes | None = None, network: bool = False
    ) -> subprocess.CompletedProcess[bytes]:
        argv = ["git", *(a for o in GIT_OPTIONS for a in ("-c", o)), "-C", str(cwd), *args]
        env = git_environment(self.home, network=network)
        return subprocess.run(argv, env=env, input=stdin, capture_output=True, check=False)

    def ok(
        self, cwd: Path, *args: str, stdin: bytes | None = None, network: bool = False, code: str = GIT_FAILED
    ) -> str:
        result = self.run(cwd, *args, stdin=stdin, network=network)
        if result.returncode != 0:
            detail = (result.stdout + result.stderr).decode(errors="replace").strip()[-_TAIL:]
            raise ReproduceSourceError(code, f"git {args[0]} ({result.returncode})", detail)
        return result.stdout.decode().strip()

    def has_commit(self, repo: Path, commit: str) -> bool:
        return self.run(repo, "cat-file", "-e", f"{commit}^{{commit}}").returncode == 0


# --- names and patch sets ----------------------------------------------------------------------


def repo_name(url: str) -> str:
    """The mirror name of a repository: its URL's last component without ``.git``.

    Args:
        url: A git URL, scp-like address or local path.

    Returns:
        The name.

    Raises:
        ReproduceSourceError: If the name is empty or has characters outside ``[A-Za-z0-9_.-]``.
    """
    tail = re.split(r"[/:]", url.rstrip("/"))[-1]
    name = tail[: -len(".git")] if tail.endswith(".git") else tail
    if not _NAME_RE.match(name):
        raise ReproduceSourceError(BAD_REPO_NAME, "repository URL gives no usable name")
    return name


def resolve_submodule_url(parent: str, url: str) -> str:
    """Resolve a ``.gitmodules`` URL against the superproject's URL, as git does.

    Args:
        parent: The superproject's URL.
        url: The submodule URL; ``./`` and ``../`` forms are relative to ``parent``.

    Returns:
        An absolute URL or path.
    """
    if not (url.startswith("./") or url.startswith("../")):
        return url
    base = parent.rstrip("/")
    parts = url.split("/")
    while parts and parts[0] in (".", ".."):
        if parts.pop(0) == "..":
            cut = max(base.rfind("/"), base.rfind(":"))
            base = base[:cut] if cut >= 0 else ""
            if cut >= 0 and parent[cut] == ":" and "/" not in base.split(":")[-1]:
                base += ":"
    rest = "/".join(parts)
    return base + rest if base.endswith(":") else f"{base}/{rest}"


def patch_set_sha256(patches: Sequence[FileRef]) -> str:
    """The sha256 naming an ordered patch set: each patch's manifest-relative path and content sha256.

    Args:
        patches: The step's patches, in application order.

    Returns:
        A hex sha256; `EMPTY_PATCH_SET_SHA256` for no patches.
    """
    entries = [[p.file, p.sha256] for p in patches]
    payload = {"schema": PATCH_SET_SCHEMA, "patches": entries}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


EMPTY_PATCH_SET_SHA256 = patch_set_sha256([])


def _tag(commit: str, patch_set: str) -> str:
    return f"{commit[:8]}-{patch_set[:8]}"


def mirror_dir(cache: Path, url: str) -> Path:
    """Where `mirror` keeps ``url``.

    Args:
        cache: The reproduce cache root.
        url: The repository URL.

    Returns:
        ``<cache>/src/<name>``.
    """
    return cache / SRC / repo_name(url)


def worktree_dir(cache: Path, url: str, commit: str, patch_set: str) -> Path:
    """Where `checkout` puts the worktree for one (commit, patch set).

    Args:
        cache: The reproduce cache root.
        url: The repository URL.
        commit: The pinned 40-hex commit.
        patch_set: The patch-set sha256.

    Returns:
        ``<cache>/wt/<name>-<commit[:8]>-<patch set[:8]>``.
    """
    return cache / WT / f"{repo_name(url)}-{_tag(commit, patch_set)}"


# --- mirrors -----------------------------------------------------------------------------------


def _submodules_at(git: _Git, repo: Path, rev: str) -> list[tuple[str, str, str]]:
    """``(path, url, gitlink sha)`` for every gitlink in ``rev``'s tree."""
    listing = git.ok(repo, "ls-tree", "-r", "-z", rev)
    links = {}
    for entry in listing.split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        mode, kind, sha = meta.split()
        if mode == "160000" and kind == "commit":
            links[path] = sha
    if not links:
        return []
    config = git.run(
        repo, "config", "--blob", f"{rev}:.gitmodules", "-z", "--get-regexp", r"^submodule\..*\.(path|url)$"
    )
    paths: dict[str, str] = {}
    urls: dict[str, str] = {}
    for item in config.stdout.decode().split("\0"):
        if not item:
            continue
        key, _, value = item.partition("\n")
        name, field = key[len("submodule.") :].rsplit(".", 1)
        (paths if field == "path" else urls)[name] = value
    by_path = {p: urls.get(n) for n, p in paths.items()}
    found = []
    for path, sha in sorted(links.items()):
        url = by_path.get(path)
        if not url:
            raise ReproduceSourceError(SUBMODULE_UNKNOWN, "a gitlink has no .gitmodules URL")
        found.append((path, url, sha))
    return found


def _ensure_mirror(git: _Git, path: Path, url: str, commit: str) -> None:
    if not (path / "HEAD").is_file():
        path.mkdir(parents=True, exist_ok=True)
        git.ok(path, "init", "-q", "--bare")
        git.ok(path, "config", "remote.origin.url", url)
        git.ok(path, "config", "remote.origin.fetch", f"+refs/heads/*:{UPSTREAM_HEADS}*")
        git.ok(path, "config", "--add", "remote.origin.fetch", "+refs/tags/*:refs/tags/*")
        git.ok(path, "config", "remote.origin.prune", "false")
    elif git.ok(path, "config", "--get", "remote.origin.url") != url:
        raise ReproduceSourceError(MIRROR_URL_CONFLICT, f"mirror {path.name} has another origin")
    if git.has_commit(path, commit):
        return
    git.ok(path, "fetch", "-q", "--no-prune", "origin", network=True)
    if git.has_commit(path, commit):
        return
    git.run(path, "fetch", "-q", "--no-prune", "origin", f"{commit}:{UPSTREAM_PINNED}{commit}", network=True)
    if not git.has_commit(path, commit):
        raise ReproduceSourceError(COMMIT_MISSING, f"{commit[:8]} not found in {path.name}")


def mirror(url: str, commit: str, *, cache: Path, submodules: bool = True) -> Path:
    """Mirror ``url`` bare under ``<cache>/src`` so it holds ``commit``, with its submodules.

    No network is used when the mirror already has ``commit`` and every submodule commit.

    Args:
        url: The repository URL or local path.
        commit: The 40-hex commit that must be present.
        cache: The reproduce cache root.
        submodules: Whether to mirror the submodules named at ``commit``, recursively.

    Returns:
        The mirror's path.

    Raises:
        ReproduceSourceError: ``commit-missing`` if the upstream lacks the commit, ``mirror-url-conflict``
            if the mirror name is taken by another URL, ``git-failed`` otherwise.
    """
    return _mirror(_Git(cache), url, commit, cache, submodules, set())


def _mirror(git: _Git, url: str, commit: str, cache: Path, submodules: bool, seen: set[tuple[str, str]]) -> Path:
    path = mirror_dir(cache, url)
    seen.add((url, commit))
    with SharedFileLock(path):
        _ensure_mirror(git, path, url, commit)
    if submodules:
        for _, sub_url, sha in _submodules_at(git, path, commit):
            resolved = resolve_submodule_url(url, sub_url)
            if (resolved, sha) not in seen:
                _mirror(git, resolved, sha, cache, True, seen)
    return path


# --- notebooks ---------------------------------------------------------------------------------


def clear_notebook_outputs(text: str) -> tuple[str, bool]:
    """Drop code-cell outputs, null execution counts, drop widget state and cell attachments, as plain JSON.

    Notebook-level ``metadata.widgets`` and the ``attachments`` of markdown and raw cells are
    removed too.

    Args:
        text: The notebook's JSON text.

    Returns:
        ``(text, changed)``. The text is re-serialised (``indent=1``, sorted keys, a trailing
        newline) only when something changed; otherwise it is returned as given.

    Raises:
        ReproduceSourceError: ``notebook-unreadable`` if the text is not a JSON notebook.
    """
    try:
        notebook = json.loads(text)
    except ValueError:
        raise ReproduceSourceError(NOTEBOOK_UNREADABLE, "notebook is not JSON") from None
    if not isinstance(notebook, dict):
        raise ReproduceSourceError(NOTEBOOK_UNREADABLE, "notebook is not a JSON object")
    cells: list[Any] = list(notebook.get("cells") or [])
    for sheet in notebook.get("worksheets") or []:
        cells.extend(sheet.get("cells") or [])
    changed = False
    metadata = notebook.get("metadata")
    if isinstance(metadata, dict) and "widgets" in metadata:
        del metadata["widgets"]
        changed = True
    for cell in cells:
        if isinstance(cell, dict) and cell.get("cell_type") in ("markdown", "raw") and "attachments" in cell:
            del cell["attachments"]
            changed = True
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        if cell.get("outputs") != []:
            cell["outputs"] = []
            changed = True
        for key in ("execution_count", "prompt_number"):
            if key in cell and cell[key] is not None:
                cell[key] = None
                changed = True
        if "execution_count" not in cell and "prompt_number" not in cell:
            cell["execution_count"] = None
            changed = True
    if not changed:
        return text, False
    return json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n", True


def _notebooks(git: _Git, worktree: Path) -> list[str]:
    """Tracked regular-file notebooks of ``worktree``, unquoted (``ls-files -z``)."""
    found = []
    for entry in git.ok(worktree, "ls-files", "-z", "-s").split("\0"):
        if not entry:
            continue
        meta, path = entry.split("\t", 1)
        if meta.split()[0] in ("100644", "100755") and path.endswith(".ipynb"):
            found.append(path)
    return found


def _notebooks_needing_clear(git: _Git, worktree: Path) -> list[str]:
    """Tracked notebooks that `clear_notebook_outputs` would change."""
    return [
        path
        for path in _notebooks(git, worktree)
        if clear_notebook_outputs((worktree / path).read_text(encoding="utf-8"))[1]
    ]


def _clear_outputs(git: _Git, worktree: Path) -> None:
    for path in _notebooks(git, worktree):
        file = worktree / path
        text, changed = clear_notebook_outputs(file.read_text(encoding="utf-8"))
        if changed:
            file.write_text(text, encoding="utf-8")
    git.ok(worktree, "add", "-u")
    git.ok(worktree, "commit", "-q", "--no-verify", "--allow-empty", "-m", CLEAR_MESSAGE)


# --- lint --------------------------------------------------------------------------------------


def load_deny_patterns(path: Path) -> list[str]:
    """Read a deny-pattern file: one extended regex per line; blank and ``#`` lines skipped.

    Args:
        path: The file.

    Returns:
        The patterns, in file order.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line for line in lines if line and not line.startswith("#")]


def _deny_hits(text: bytes, patterns: Sequence[str]) -> list[LintFinding]:
    """Lines matching each pattern, using ``grep -E`` so the regex dialect is the hook's."""
    findings = []
    env = {"PATH": SYSTEM_PATH, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    for index, pattern in enumerate(patterns):
        result = subprocess.run(
            ["grep", "-E", "-n", "-a", "-e", pattern], input=text, env=env, capture_output=True, check=False
        )
        if result.returncode > 1:
            raise ReproduceSourceError(DENY_PATTERN_INVALID, f"deny pattern {index} is not a valid regex")
        for line in result.stdout.decode(errors="replace").splitlines():
            findings.append(LintFinding(LINT_DENY_PATTERN, int(line.split(":", 1)[0]), index))
    return findings


def unquote_c_path(path: str) -> str:
    r"""Undo git's C-style quoting of a path (``"caf\303\251.ipynb"``); unquoted paths pass through.

    Args:
        path: A path as git prints it in a diff header.

    Returns:
        The path, with octal escapes decoded as UTF-8.
    """
    if len(path) < 2 or not (path.startswith('"') and path.endswith('"')):
        return path
    body, out, i = path[1:-1], bytearray(), 0
    while i < len(body):
        char = body[i]
        if char == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567" and re.match(r"[0-7]{3}", body[i + 1 : i + 4]):
                out.append(int(body[i + 1 : i + 4], 8))
                i += 4
                continue
            if nxt in _C_ESCAPES:
                out.append(_C_ESCAPES[nxt])
                i += 2
                continue
        out.extend(char.encode())
        i += 1
    return out.decode(errors="replace")


def _is_notebook(header: str) -> bool:
    name = header[4:]
    name = name if name.startswith('"') else name.split("\t", 1)[0]
    return unquote_c_path(name.rstrip()).endswith(".ipynb")


def _hunk_hits(lines: list[str], start: int, old: int, new: int) -> tuple[list[LintFinding], int]:
    """Findings in one hunk body starting at index ``start``; also the index after the hunk."""
    findings = []
    in_outputs: str | None = None
    i = start
    while i < len(lines) and (old > 0 or new > 0):
        line = lines[i]
        i += 1
        kind, content = (line[0], line[1:]) if line else (" ", "")
        if kind == "\\":
            continue
        if kind not in " +-":
            break
        old -= kind in " -"
        new -= kind in " +"
        changed = kind in "+-"
        hit = changed and (in_outputs is not None or bool(_OUTPUT_TYPE_RE.search(content)))
        match = _OUTPUTS_RE.match(content)
        if match and not match.group(2).strip().startswith("]"):
            hit = hit or changed
            in_outputs = match.group(1)
        elif in_outputs is not None and re.match(rf"^{re.escape(in_outputs)}\]\s*,?\s*$", content):
            in_outputs = None
        if hit:
            findings.append(LintFinding(LINT_NOTEBOOK_OUTPUTS, i))
    return findings, i


def _outputs_hits(lines: list[str]) -> list[LintFinding]:
    """Changed lines inside, or opening, a non-empty ``"outputs"`` block of a notebook hunk.

    Hunk bodies are consumed by their header's line counts, so a body line that looks like a file
    header is never misread as one.
    """
    findings: list[LintFinding] = []
    notebook = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            target = lines[i + 1] if not lines[i + 1].startswith("+++ /dev/null") else line
            notebook = _is_notebook(target)
            i += 2
            continue
        hunk = _HUNK_RE.match(line)
        if hunk:
            old = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new = int(hunk.group(2)) if hunk.group(2) is not None else 1
            found, i = _hunk_hits(lines, i + 1, old, new)
            if notebook:
                findings += found
            continue
        i += 1
    return findings


def lint_patch(patch: str | bytes, deny_patterns: Sequence[str]) -> list[LintFinding]:
    """Check a unified diff: no notebook ``"outputs"`` hunk, deny-pattern hit, binary or symlink.

    Every line of the patch is scanned for deny patterns, headers and context included. Findings
    carry line numbers and pattern indices only, never patch content.

    Args:
        patch: The patch text or bytes.
        deny_patterns: Extended regexes, as from `load_deny_patterns`.

    Returns:
        The findings, empty when the patch passes.

    Raises:
        ReproduceSourceError: ``deny-pattern-invalid`` if a pattern does not compile.
    """
    data = patch.encode() if isinstance(patch, str) else patch
    lines = data.decode(errors="replace").splitlines()
    findings = _outputs_hits(lines)
    findings += [LintFinding(LINT_BINARY, n) for n, line in enumerate(lines, start=1) if line == "GIT binary patch"]
    findings += [LintFinding(LINT_SYMLINK, n) for n, line in enumerate(lines, start=1) if _SYMLINK_MODE_RE.match(line)]
    findings += _deny_hits(data, deny_patterns)
    return sorted(findings)


# --- checkouts ---------------------------------------------------------------------------------


def _read_patches(patches: Sequence[FileRef], base_dir: Path | None) -> list[tuple[str, bytes]]:
    """Verified patch bytes: `verify_files` first, then the bytes read again must have that sha."""
    if not patches:
        return []
    if base_dir is None:
        raise ReproduceSourceError(PATCHES_NEED_BASE_DIR, "patches given without the manifest directory")
    verified = verify_files(patches, base_dir)
    out = []
    for ref in patches:
        data = (base_dir / ref.file).read_bytes()
        if hashlib.sha256(data).hexdigest() != verified[ref.file]:
            raise ReproduceLockError(f"file {ref.file}: changed after verification")
        out.append((PurePosixPath(ref.file).name, data))
    return out


def _marker_path(worktree: Path) -> Path:
    return worktree.with_name(worktree.name + ".json")


def _write_marker(path: Path, co: Checkout) -> None:
    data = co._asdict()
    data.update(schema=MARKER_SCHEMA, worktree=str(co.worktree), mirror=str(co.mirror))
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _read_marker(path: Path) -> Checkout:
    data = json.loads(path.read_text())
    if data.pop("schema", None) != MARKER_SCHEMA:
        raise ReproduceSourceError(WORKTREE_MISMATCH, f"{path.name} has another schema")
    data["worktree"], data["mirror"] = Path(data["worktree"]), Path(data["mirror"])
    data["patches"] = [(file, sha) for file, sha in data["patches"]]
    return Checkout(**data)


def _add_submodules(
    git: _Git, cache: Path, url: str, worktree: Path, notebooks: list[str], prefix: str = ""
) -> dict[str, str]:
    """Add submodule worktrees recursively; return their shas and append dirty notebook paths."""
    shas: dict[str, str] = {}
    for path, sub_url, sha in _submodules_at(git, worktree, "HEAD"):
        resolved = resolve_submodule_url(url, sub_url)
        sub_mirror = mirror(resolved, sha, cache=cache)
        target = worktree / path
        git.ok(sub_mirror, "worktree", "prune")
        git.ok(sub_mirror, "worktree", "add", "-q", "--detach", str(target), sha)
        if git.ok(target, "rev-parse", "HEAD") != sha:
            raise ReproduceSourceError(HEAD_MISMATCH, f"submodule {path} is not at its gitlink")
        shas[prefix + path] = sha
        notebooks += [f"{prefix}{path}/{nb}" for nb in _notebooks_needing_clear(git, target)]
        shas.update(_add_submodules(git, cache, resolved, target, notebooks, f"{prefix}{path}/"))
    return shas


def _remove_worktree(git: _Git, cache: Path, mirror_path: Path, worktree: Path) -> None:
    git.run(mirror_path, "worktree", "remove", "--force", "--force", str(worktree))
    if worktree.exists():
        shutil.rmtree(worktree)
    git.run(mirror_path, "worktree", "prune")
    src = cache / SRC
    if src.is_dir():
        for other in src.iterdir():
            if other != mirror_path and (other / "worktrees").is_dir():
                git.run(other, "worktree", "prune")


def _first_parent_chain(git: _Git, worktree: Path, length: int) -> list[str]:
    return git.ok(worktree, "rev-list", "--first-parent", f"--max-count={length}", "HEAD").split()


def verify_checkout(co: Checkout) -> None:
    """Check that a worktree is still what `checkout` recorded.

    ``HEAD`` and the ``repro/*`` ref equal ``co.head``; the first-parent chain, one clear commit
    plus one commit per patch long, ends at ``co.commit``; ``HEAD^{tree}`` is ``co.tree_sha``; the
    status, untracked files and submodules included, is clean; each submodule is at its sha.

    Args:
        co: The recorded checkout.

    Raises:
        ReproduceSourceError: ``head-mismatch``, ``chain-mismatch``, ``tree-mismatch`` or
            ``worktree-dirty``.
    """
    git = _Git(co.mirror.parent.parent)
    if not co.worktree.is_dir() or git.ok(co.worktree, "rev-parse", "HEAD", code=HEAD_MISMATCH) != co.head:
        raise ReproduceSourceError(HEAD_MISMATCH, "worktree HEAD is not the recorded head")
    if git.run(co.mirror, "rev-parse", "--verify", "-q", co.ref).stdout.decode().strip() != co.head:
        raise ReproduceSourceError(HEAD_MISMATCH, "repro ref is not the recorded head")
    length = 2 + len(co.patches)
    chain = _first_parent_chain(git, co.worktree, length)
    if len(chain) != length or chain[-1] != co.commit:
        raise ReproduceSourceError(CHAIN_MISMATCH, "first-parent chain does not end at the commit")
    if git.ok(co.worktree, "rev-parse", "HEAD^{tree}") != co.tree_sha:
        raise ReproduceSourceError(TREE_MISMATCH, "tree differs from the recorded tree")
    status = git.ok(co.worktree, "status", "--porcelain", "--untracked-files=all", "--ignore-submodules=none")
    if status:
        raise ReproduceSourceError(WORKTREE_DIRTY, "worktree has changes")
    for path, sha in co.submodules.items():
        if git.ok(co.worktree / path, "rev-parse", "HEAD") != sha:
            raise ReproduceSourceError(HEAD_MISMATCH, f"submodule {path} moved")


def _build(
    git: _Git,
    cache: Path,
    url: str,
    commit: str,
    mirror_path: Path,
    worktree: Path,
    patches: list[tuple[str, bytes]],
    refs: Sequence[FileRef],
    patch_set: str,
    submodules: bool,
    allow_submodule_notebook_outputs: bool,
) -> Checkout:
    worktree.parent.mkdir(parents=True, exist_ok=True)
    git.ok(mirror_path, "worktree", "prune")
    git.ok(mirror_path, "worktree", "add", "-q", "--detach", str(worktree), commit)
    if git.ok(worktree, "rev-parse", "HEAD") != commit:
        raise ReproduceSourceError(HEAD_MISMATCH, "worktree HEAD is not the commit")
    _clear_outputs(git, worktree)
    for name, data in patches:
        git.ok(worktree, "apply", "--check", "--index", "-", stdin=data, code=PATCH_DOES_NOT_APPLY)
        git.ok(worktree, "apply", "--index", "-", stdin=data, code=PATCH_DOES_NOT_APPLY)
        dirty = _notebooks_needing_clear(git, worktree)
        if dirty:
            raise ReproduceSourceError(PATCH_LINT, f"patch {name} leaves notebook outputs", "\n".join(dirty))
        git.ok(worktree, "commit", "-q", "--no-verify", "-m", f"repro: {name}")
    head = git.ok(worktree, "rev-parse", "HEAD")
    ref = REPRO_REF_PREFIX + _tag(commit, patch_set)
    notebooks: list[str] = []
    shas = _add_submodules(git, cache, url, worktree, notebooks) if submodules else {}
    if notebooks and not allow_submodule_notebook_outputs:
        raise ReproduceSourceError(
            SUBMODULE_NOTEBOOK_OUTPUTS, f"{len(notebooks)} submodule notebook(s) carry outputs", "\n".join(notebooks)
        )
    co = Checkout(
        worktree=worktree,
        mirror=mirror_path,
        repo=url,
        commit=commit,
        patch_set_sha256=patch_set,
        patches=[(r.file, r.sha256) for r in refs],
        ref=ref,
        head=head,
        tree_sha=git.ok(worktree, "rev-parse", "HEAD^{tree}"),
        submodules=shas,
        submodule_notebooks_with_outputs=sorted(notebooks),
    )
    git.ok(mirror_path, "update-ref", ref, head)
    verify_checkout(co)
    return co


def checkout(
    repo: str,
    commit: str,
    patches: Sequence[FileRef],
    *,
    cache: Path,
    base_dir: Path | None = None,
    submodules: bool = True,
    deny_patterns: Sequence[str] = (),
    allow_submodule_notebook_outputs: bool = False,
) -> Checkout:
    """Prepare the detached worktree for (``commit``, ``patches``); run at prefetch, never at run.

    Patches are verified (`locks.verify_files`) and linted before any git command runs. A complete
    worktree is verified and reused; anything else is rebuilt from the mirror.

    Args:
        repo: The repository URL or local path.
        commit: The pinned 40-hex commit.
        patches: The step's patches, in order, relative to ``base_dir``.
        cache: The reproduce cache root.
        base_dir: The manifest repository; required when there are patches.
        submodules: Whether to check out submodules from their mirrors.
        deny_patterns: Extended regexes no patch line may match.
        allow_submodule_notebook_outputs: Accept notebooks with outputs inside submodules, which
            are never cleared; their paths are recorded in the checkout. Refused otherwise.

    Returns:
        The checkout, with its tree sha and submodule shas.

    Raises:
        ReproduceLockError: If a patch's sha256 differs from the manifest.
        ReproduceSourceError: With a fixed code for every other failure.
    """
    data = _read_patches(patches, base_dir)
    for _, text in data:
        if lint_patch(text, deny_patterns):
            raise ReproduceSourceError(PATCH_LINT, "a patch fails lint")
    patch_set = patch_set_sha256(patches)
    git = _Git(cache)
    mirror_path = mirror(repo, commit, cache=cache, submodules=submodules)
    worktree = worktree_dir(cache, repo, commit, patch_set)
    marker = _marker_path(worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    with SharedFileLock(worktree):
        if marker.is_file():
            co = _read_marker(marker)
            if (co.repo, co.commit, co.patch_set_sha256) != (repo, commit, patch_set):
                raise ReproduceSourceError(WORKTREE_MISMATCH, f"{marker.name} records another checkout")
            verify_checkout(co)
            if co.submodule_notebooks_with_outputs and not allow_submodule_notebook_outputs:
                raise ReproduceSourceError(
                    SUBMODULE_NOTEBOOK_OUTPUTS,
                    f"{len(co.submodule_notebooks_with_outputs)} submodule notebook(s) carry outputs",
                    "\n".join(co.submodule_notebooks_with_outputs),
                )
            return co
        _remove_worktree(git, cache, mirror_path, worktree)
        try:
            co = _build(
                git,
                cache,
                repo,
                commit,
                mirror_path,
                worktree,
                data,
                patches,
                patch_set,
                submodules,
                allow_submodule_notebook_outputs,
            )
        except BaseException:
            _remove_worktree(git, cache, mirror_path, worktree)
            raise
        _write_marker(marker, co)
        return co


def checkout_step(step: StepSpec, *, cache: Path, base_dir: Path, deny_patterns: Sequence[str] = ()) -> Checkout:
    """`checkout` for a manifest step.

    Args:
        step: The step.
        cache: The reproduce cache root.
        base_dir: The manifest repository.
        deny_patterns: Extended regexes no patch line may match.

    Returns:
        The checkout.
    """
    return checkout(
        step.repo,
        step.commit,
        step.patches,
        cache=cache,
        base_dir=base_dir,
        submodules=step.submodules,
        deny_patterns=deny_patterns,
    )


def prefetched_checkout(step: StepSpec, *, cache: Path) -> Checkout:
    """The worktree prefetch made for a step, verified; a run never creates one.

    Args:
        step: The step.
        cache: The reproduce cache root.

    Returns:
        The recorded checkout, after `verify_checkout`.

    Raises:
        ReproduceSourceError: ``worktree-missing`` when no complete worktree exists for the step's
            (repository, commit, patch set); ``worktree-mismatch`` when its marker records another
            checkout or submodules the step disables; otherwise as `verify_checkout`.
    """
    patch_set = patch_set_sha256(step.patches)
    marker = _marker_path(worktree_dir(cache, step.repo, step.commit, patch_set))
    if not marker.is_file():
        raise ReproduceSourceError(WORKTREE_MISSING, "no prefetched worktree for the step")
    try:
        co = _read_marker(marker)
    except (OSError, ValueError, TypeError, KeyError):
        raise ReproduceSourceError(WORKTREE_MISMATCH, f"{marker.name} cannot be read") from None
    if (co.repo, co.commit, co.patch_set_sha256) != (step.repo, step.commit, patch_set):
        raise ReproduceSourceError(WORKTREE_MISMATCH, f"{marker.name} records another checkout")
    if co.submodules and not step.submodules:
        raise ReproduceSourceError(WORKTREE_MISMATCH, f"{marker.name} has submodules the step disables")
    verify_checkout(co)
    return co


def checkout_sources(sources: Iterable[SourceSpec], *, cache: Path) -> dict[str, Checkout]:
    """Check out draft sources with the empty patch set (``prefetch --sources-only``).

    Args:
        sources: The draft source specs.
        cache: The reproduce cache root.

    Returns:
        ``{source name: checkout}``, in input order.
    """
    return {s.name: checkout(s.repo, s.commit, [], cache=cache, submodules=s.submodules) for s in sources}


def authoring_worktree(repo: str, commit: str, dest: Path, *, cache: Path) -> Path:
    """Add a separate detached worktree at the cleared, unpatched ref, for writing patches.

    Runs ``git -C <mirror> worktree add --detach <dest> repro/<commit[:8]>-<empty set[:8]>``. The
    prefetch worktree itself is never edited.

    Args:
        repo: The repository URL or local path.
        commit: The pinned commit, already checked out with the empty patch set.
        dest: The new worktree's path; must not exist.
        cache: The reproduce cache root.

    Returns:
        ``dest``.

    Raises:
        ReproduceSourceError: ``ref-missing`` if the empty-patch-set checkout was never made.
    """
    git = _Git(cache)
    mirror_path = mirror_dir(cache, repo)
    ref = REPRO_REF_PREFIX + _tag(commit, EMPTY_PATCH_SET_SHA256)
    if not (mirror_path / "HEAD").is_file() or git.run(mirror_path, "rev-parse", "--verify", "-q", ref).returncode:
        raise ReproduceSourceError(REF_MISSING, "no cleared ref; run the sources-only prefetch first")
    dest.parent.mkdir(parents=True, exist_ok=True)
    git.ok(mirror_path, "worktree", "add", "-q", "--detach", str(dest), ref)
    return dest
