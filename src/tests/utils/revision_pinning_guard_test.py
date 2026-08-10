"""No load anywhere may pass a ref where a commit SHA belongs.

This is the regression guard for the design's central rule. Without it the codebase decays
back to ref-addressed loads one call site at a time, and the provenance keeps reporting
commits it did not actually load.

Three of the five call sites this task touches (``canary_qwen.py``,
``speech_emotion_recognition/api.py``, ``child_adult.py``) get their SHA from
``HFModel.commit_sha``, which is resolved once at model-construction time and simply
forwarded here -- there is no local resolution step left to unit-test in those beyond "the
field is forwarded, not ``model.revision``", which the static sweep below covers. A fourth
(``moss.py``) does the same via a local ``revision`` variable. ``brouhaha.py`` has no
``HFModel`` (it takes a bare ``model_id``/``revision`` pair), so it resolves for itself; that
resolution step is extracted into ``_build_worker_input`` specifically so it is unit-testable
here without spawning the subprocess venv.

``test_no_unreviewed_subprocess_revision_payload`` is what makes this durable against sites
added *after* this task: rather than hardcoding "these five files are fine" as the only
check, it statically discovers every subprocess-worker file whose parent-side payload
carries a revision-shaped key and fails if that file is not in
``REVISION_RESOLVED_SUBPROCESS_FILES`` -- the same conscious-human-checkpoint pattern
``hf_load_coverage_test.py`` uses for HF-load sites generally (new site -> test fails until
reviewed and added to the allowlist here).
"""

import ast
import re
from pathlib import Path

import pytest

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Subprocess-worker files where a resolved commit SHA has been verified to reach the
# worker's JSON payload. Add a file here only after checking its input_json construction the
# way the five below were checked -- do not add it just because it appears in
# hf_load_coverage_test.REVIEWED_SUBPROCESS, which tracks a different property (HF-cache
# safety, not revision-vs-ref). test_no_unreviewed_subprocess_revision_payload enforces that
# every subprocess file sending a revision-shaped payload key is in this set.
REVISION_RESOLVED_SUBPROCESS_FILES = {
    "audio/tasks/scene_quality/brouhaha.py",
    "audio/tasks/speech_to_text/canary_qwen.py",
    "audio/tasks/classification/speech_emotion_recognition/api.py",
    "audio/tasks/speaker_diarization/child_adult.py",
    "audio/tasks/speaker_diarization/moss.py",
}


def test_worker_input_json_carries_a_sha_not_a_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """Brouhaha's worker payload carries the resolved commit, never the ref it was given."""
    sha = "e" * 40
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    from senselab.audio.tasks.scene_quality import brouhaha

    payload = brouhaha._build_worker_input(  # noqa: SLF001 -- payload construction is what's under test
        chunks=[{"path": "/tmp/a.wav", "start_s": 0.0, "audio_idx": 0}],
        model_name="pyannote/brouhaha",
        revision="main",
        device="cpu",
        out_dir="/tmp",
    )
    assert SHA_RE.match(payload["revision"]), f"worker got a ref, not a commit: {payload['revision']!r}"


def test_subprocess_env_propagates_the_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker must join its parent's run rather than starting its own."""
    import senselab.utils.model_revision as mr

    # This module caches the run id process-wide (_RUN_ID); an earlier test in this session
    # may already have set it, which would make the assertion below pass regardless of what
    # hf_subprocess_env does. Reset it so run_id() actually re-reads SENSELAB_RUN_ID.
    mr._RUN_ID = None
    monkeypatch.setenv("SENSELAB_RUN_ID", "run-abc")
    monkeypatch.setattr("senselab.utils.dependencies.resolve_model", lambda *a, **k: ("f" * 40, "/tmp"))
    from senselab.utils.dependencies import hf_subprocess_env

    # base_env={}, not the default (a copy of os.environ): os.environ already carries
    # SENSELAB_RUN_ID from the monkeypatch.setenv above, so copying it verbatim would pass
    # this assertion even if hf_subprocess_env never injected the id itself.
    env = hf_subprocess_env("org/model", "main", base_env={})
    assert env["SENSELAB_RUN_ID"] == "run-abc", "a worker must join its parent's run, not start its own"


def test_revision_resolved_subprocess_files_still_resolve_before_sending() -> None:
    """The reviewed subprocess parents must keep resolving before they send.

    **Matched against the AST, not the file text, and that distinction is the whole test.**
    A substring search over the source cannot fail here: every reviewed file carries a
    comment explaining why it resolves, and those comments contain both
    ``resolve_revision`` and ``commit_sha``. So reverting the code to
    ``"revision": model.revision`` while leaving the comment in place would keep a text-based
    check green -- it would assert that the explanation exists, not that the behaviour does.

    Comments and string literals are invisible to ``ast``, so requiring a real
    ``resolve_revision(...)`` call or a real ``.commit_sha`` attribute access means only
    executable code can satisfy it.

    It still does not understand data flow, so it cannot prove the value reaching the payload
    is the resolved one -- ``test_worker_input_json_carries_a_sha_not_a_ref`` does that for the
    one builder callable without spawning a subprocess.
    """
    from tests.utils.hf_load_coverage_test import _SRC

    unresolved: list[str] = []
    for relpath in sorted(REVISION_RESOLVED_SUBPROCESS_FILES):
        tree = ast.parse((_SRC / relpath).read_text())
        resolves = any(
            (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "resolve_revision")
            or (isinstance(node, ast.Attribute) and node.attr == "commit_sha")
            for node in ast.walk(tree)
        )
        if not resolves:
            unresolved.append(relpath)

    assert not unresolved, (
        "File(s) have no executable resolve_revision(...) call and no .commit_sha access -- a raw "
        "ref may have crept back into the worker payload (a comment saying otherwise does not "
        "count):\n" + "\n".join(f"  {f}" for f in unresolved)
    )


def _revision_payload_files() -> set[str]:
    """Subprocess-worker files whose parent-side code builds a payload with a revision-ish key.

    AST-based, reusing ``hf_load_coverage_test.py``'s own subprocess-worker discovery
    (``_subprocess_worker_files``) as the candidate set. For each candidate, walks every
    dict literal in the *parent's* Python (not the worker string -- that is a separate
    string literal these dict literals feed into via ``json.dumps``) and flags any string
    key containing "revision" (covers ``"revision"``, ``"hf_revision"``,
    ``"model_revision"``, and any future name in that family). A file landing in this set is
    a candidate carrying a per-run identity across the subprocess boundary and must be
    reviewed the same way the five files in REVISION_RESOLVED_SUBPROCESS_FILES were.
    """
    from tests.utils.hf_load_coverage_test import _SRC, _subprocess_worker_files

    found: set[str] = set()
    for relpath in _subprocess_worker_files():
        try:
            tree = ast.parse((_SRC / relpath).read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and "revision" in key.value.lower():
                    found.add(relpath)
    return found


def test_no_unreviewed_subprocess_revision_payload() -> None:
    """Fail if a subprocess site sends a revision-shaped payload field without being reviewed.

    This is the check that generalises past the five files enumerated above: a *new*
    subprocess backend added later that copies the same ``"revision": model.revision``
    boilerplate is caught here automatically (via ``_revision_payload_files``'s AST sweep)
    rather than silently passing because nobody remembered to touch this test. Both
    directions are checked -- a newly-detected, unreviewed file, and a stale allowlist entry
    whose payload no longer carries a revision-shaped key -- mirroring
    ``hf_load_coverage_test.test_allowlists_have_no_stale_entries``.
    """
    detected = _revision_payload_files()
    missing = sorted(detected - REVISION_RESOLVED_SUBPROCESS_FILES)
    stale = sorted(REVISION_RESOLVED_SUBPROCESS_FILES - detected)
    assert not missing, (
        "Subprocess file(s) send a revision-shaped payload field but are not reviewed for "
        "the SHA-not-ref rule:\n"
        + "\n".join(f"  {f}" for f in missing)
        + "\n\nResolve via resolve_revision / model.commit_sha before building the worker "
        "payload, then add the file to REVISION_RESOLVED_SUBPROCESS_FILES in this test."
    )
    assert not stale, f"Remove stale REVISION_RESOLVED_SUBPROCESS_FILES entries (no revision-shaped key found): {stale}"


@pytest.mark.slow
def test_the_two_call_rule_holds_against_the_real_hub(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve yields a SHA; loading with that SHA needs no network.

    Touches the real Hub instead of mocking ``huggingface_hub``: a mock encodes our own beliefs
    about that library's behaviour and would pass just as happily if those beliefs are wrong.

    Neither ``pyproject.toml`` nor ``conftest.py`` registers a "slow"/"network" pytest marker in
    this repo, and CI runs ``pytest src/tests`` with no ``-m`` filter (``.github/workflows/tests.yaml``),
    so ``@pytest.mark.slow`` here is a label, not a skip mechanism -- the same as this repo's one
    existing precedent, ``@pytest.mark.large_model`` in ``speaker_verification_test.py``. The actual
    hermetic guard is the try/except below: any Hub/connection failure skips rather than fails or
    hangs, so an offline sandbox or air-gapped run is unaffected.

    ``local_files_only=True`` on the second call, not ``HF_HUB_OFFLINE``, is what forces the
    no-network check: ``huggingface_hub.constants.HF_HUB_OFFLINE`` is read once at import (see
    ``hf_subprocess_env``'s own docstring), so setting the env var this late in an already-running
    process is a no-op and the call would silently go to the network instead of proving anything.
    ``cache_dir`` is passed explicitly to both calls (not left to ``HF_HUB_CACHE``, frozen the same
    way) so the offline check is against a guaranteed-empty-until-populated temp dir, never this
    machine's real, possibly warm, HF cache.
    """
    cache_dir = str(tmp_path / "hub")
    # Keeps the run-resolution manifest (a *different* cache, see model_revision.py) out of this
    # machine's real ~/.cache/senselab/hf too, and out of any other test's manifest.
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "senselab"))
    import senselab.utils.model_revision as mr

    mr._RUN_ID = None
    mr._MEMO.clear()
    from senselab.utils.model_revision import resolve_revision

    repo = "hf-internal-testing/tiny-random-gpt2"
    try:
        sha = resolve_revision(repo, "main")
    except Exception as exc:  # noqa: BLE001 -- any Hub/network failure skips; see docstring
        pytest.skip(f"HuggingFace Hub unreachable, cannot verify the two-call rule live: {exc}")
    assert SHA_RE.match(sha)

    from huggingface_hub import snapshot_download

    snapshot_download(repo, revision=sha, cache_dir=cache_dir)  # populate
    assert snapshot_download(repo, revision=sha, cache_dir=cache_dir, local_files_only=True), (
        "a full SHA must resolve from cache with no network"
    )


def test_resolve_model_stages_the_manifest_commit_not_the_local_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """The loader must resolve through the run manifest, not this host's refs/<ref>.

    The regression this guards is the one that makes provenance lie. Before the fix,
    ``resolve_model`` passed the *ref* to ``ensure_hf_model``, so the load resolved against
    whatever this host's ``refs/main`` pointed at while the cache key and provenance resolved
    through the manifest. Two independent resolvers: identical on one warm node, divergent on the
    multi-node sweep the manifest exists for -- and the artifact then names a commit that never ran.
    """
    manifest_sha = "1" * 40
    staged: dict[str, str] = {}

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: manifest_sha)

    def _record(repo_id: str, revision: str = "main", token: object = None) -> str:
        staged["revision"] = revision
        return revision

    monkeypatch.setattr("senselab.utils.dependencies.ensure_hf_model", _record)

    from senselab.utils.dependencies import resolve_model

    sha, _ = resolve_model("org/model", "main")
    assert staged["revision"] == manifest_sha, (
        f"ensure_hf_model was handed {staged['revision']!r}; it must receive the manifest's commit, "
        "or the load and the recorded provenance can name different commits"
    )
    assert sha == manifest_sha
