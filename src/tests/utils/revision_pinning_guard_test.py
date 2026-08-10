"""No load anywhere may pass a ref where a commit SHA belongs.

This is the regression guard for the design's central rule. Without it the codebase decays
back to ref-addressed loads one call site at a time, and the provenance keeps reporting
commits it did not actually load.

Two of the three call sites this task touches (``canary_qwen.py``,
``speech_emotion_recognition/api.py``) get their SHA from ``HFModel.commit_sha``, which is
resolved once at model-construction time and simply forwarded here -- there is no local
resolution step left to unit-test in those two files beyond "the field is forwarded, not
``model.revision``", which the static sweep below covers. ``brouhaha.py`` has no ``HFModel``
(it takes a bare ``model_id``/``revision`` pair), so it resolves for itself; that resolution
step is extracted into ``_build_worker_input`` specifically so it is unit-testable here
without spawning the subprocess venv.
"""

import re
from pathlib import Path

import pytest

SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# Subprocess-worker files where a resolved commit SHA has been verified to reach the
# worker's JSON payload (task 6, hf-revision-pinning). Other files in
# hf_load_coverage_test.REVIEWED_SUBPROCESS may still forward `model.revision` unresolved
# (e.g. `speaker_diarization/child_adult.py`, `speaker_diarization/moss.py`, at the time this
# guard was written) -- fixing those is out of this task's scope. Add a file here only after
# checking its input_json construction the way the three below were checked; do not add it
# just because it appears in REVIEWED_SUBPROCESS.
REVISION_RESOLVED_SUBPROCESS_FILES = {
    "audio/tasks/scene_quality/brouhaha.py",
    "audio/tasks/speech_to_text/canary_qwen.py",
    "audio/tasks/classification/speech_emotion_recognition/api.py",
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


def test_revision_resolved_subprocess_files_send_a_sha_not_a_ref() -> None:
    """The subprocess parents fixed by this task must keep resolving before they send.

    Static, source-level sweep in the spirit of ``hf_load_coverage_test.py``: it does not
    understand data flow, so it cannot prove a specific value is a SHA, but it can catch the
    concrete anti-pattern this task removed -- ``X.revision or "main"`` fed straight into a
    ``"revision"`` payload key -- the moment it reappears in one of the three files this task
    fixed, and it requires each file to still reference ``resolve_revision``/``commit_sha`` at
    all. Deliberately scoped to ``REVISION_RESOLVED_SUBPROCESS_FILES`` rather than the full
    ``REVIEWED_SUBPROCESS`` set in ``hf_load_coverage_test.py``: other subprocess backends were
    not touched by this task and may still have the same bug (see that set's module comment).
    """
    from tests.utils.hf_load_coverage_test import _SRC

    anti_pattern = re.compile(r'"revision"\s*:[^,\n]*\.revision\b[^,\n]*or\s+"main"')
    offenders = []
    unresolved = []
    for relpath in sorted(REVISION_RESOLVED_SUBPROCESS_FILES):
        text = (_SRC / relpath).read_text()
        if anti_pattern.search(text):
            offenders.append(relpath)
        if "resolve_revision" not in text and "commit_sha" not in text:
            unresolved.append(relpath)

    assert not offenders, (
        'Subprocess worker file(s) send a bare ref (`X.revision or "main"`) as the worker\'s '
        "revision, instead of a resolved commit SHA:\n" + "\n".join(f"  {f}" for f in offenders)
    )
    assert not unresolved, (
        "File(s) no longer reference resolve_revision or commit_sha -- a raw ref may have "
        "crept back into the worker payload:\n" + "\n".join(f"  {f}" for f in unresolved)
    )


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
