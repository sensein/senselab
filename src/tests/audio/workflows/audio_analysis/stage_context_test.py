"""StageContext / PassPlan / STAGE_VERSIONS contract tests (T051 step 4).

The highest-value test here is `test_provenance_joins_to_build_cache_index`: the
adaptive loop indexes cached results on `provenance.audio_signature`, and a
mismatch against `summary.json` makes cache-replay escalation silently never
fire — no error, no log line.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from senselab.audio.workflows.audio_analysis.stage_context import (
    STAGE_VERSIONS,
    PassPlan,
    StageContext,
    stage_code_version,
)
from senselab.utils.model_revision import RevisionResolutionError
from senselab.utils.tasks.cached_inference import audio_signature, cache_store


@pytest.fixture(autouse=True)
def _stub_commit_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub HF revision resolution so `cache_key_for` never touches the network.

    `cache_key_for` resolves any Hub-shaped model id (containing ``/``, e.g.
    ``"facebook/mms-1b-all"``) to a commit SHA before computing the key. A handful of tests below
    use real-looking ids for exactly that reason; without this stub, resolving them would be a live
    Hub call whose outcome depends on whether this machine already has that repo cached locally —
    the same non-hermetic risk `HFModel` construction has, and the same fix.
    """
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda repo_id, ref="main", **kw: "f" * 40)


def _ctx(**kwargs: object) -> StageContext:
    base: dict[str, object] = {"perturbation": "raw", "audio_signature": "a" * 64, "senselab_ver": "1.2.3"}
    base.update(kwargs)
    return StageContext(**base)  # type: ignore[arg-type]


# ── STAGE_VERSIONS ────────────────────────────────────────────────────


def test_stage_versions_are_pinned() -> None:
    """A bump invalidates that stage's cache, so it should be a visible diff."""
    assert dict(STAGE_VERSIONS) == {
        "diarization": 1,
        "ast": 1,
        "yamnet": 1,
        "features": 1,
        "asr": 1,
        "alignment": 1,
        # Scene-quality level work. Each declares its own counter rather than
        # borrowing another stage's invalidation fate.
        "background_mask": 1,
        "noise_floor": 1,
        "background_sources": 1,
        "level_probe": 1,
    }


def test_stage_versions_is_immutable() -> None:
    """Nothing may mutate the table at runtime — keys must be reviewable in git."""
    with pytest.raises(TypeError):
        STAGE_VERSIONS["asr"] = 99  # type: ignore[index]


def test_stage_code_version_is_self_describing() -> None:
    """`cat`-ing a cache entry should tell you which stage version wrote it."""
    assert stage_code_version("asr") == "asr@1"


def test_unknown_stage_raises_rather_than_defaulting() -> None:
    """A new stage must declare a version, not inherit another stage's fate."""
    with pytest.raises(KeyError, match="STAGE_VERSIONS"):
        stage_code_version("brand_new_stage")


# ── device_label ──────────────────────────────────────────────────────


def test_device_label_is_auto_when_unset() -> None:
    """None → "auto". It's inside the cache key and the provenance."""
    assert _ctx().device_label == "auto"


def test_device_label_uses_the_enum_value() -> None:
    """A concrete device reports its senselab value."""
    from senselab.utils.data_structures import DeviceType

    assert _ctx(device=DeviceType.CPU).device_label == "cpu"


# ── keys ──────────────────────────────────────────────────────────────


def test_cache_key_is_stable_and_stage_scoped() -> None:
    """Same call → same key; a different task → a different key."""
    ctx = _ctx()
    first = ctx.cache_key_for("asr", "whisper", {"device": "auto"})
    assert first == ctx.cache_key_for("asr", "whisper", {"device": "auto"})
    assert first != ctx.cache_key_for("diarization", "whisper", {"device": "auto"})


def test_cache_key_tracks_the_audio_signature() -> None:
    """Different audio must never replay another clip's result."""
    a = _ctx(audio_signature="a" * 64).cache_key_for("asr", "m", {})
    b = _ctx(audio_signature="b" * 64).cache_key_for("asr", "m", {})
    assert a != b


def test_cache_key_tracks_the_resolved_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two commits of the same Hub model must not collide (the bug this task fixes)."""
    ctx = _ctx()
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "a" * 40)
    at_a = ctx.cache_key_for("asr", "openai/whisper-tiny", {})
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: "b" * 40)
    at_b = ctx.cache_key_for("asr", "openai/whisper-tiny", {})
    assert at_a != at_b


def test_commit_sha_for_a_non_hub_id_skips_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend name with no ``/`` (e.g. ``"yamnet"``) has nothing to resolve on the Hub.

    Distinct from a model-less stage's ``None`` -- both end up with no commit_sha, but for
    different reasons, and neither should attempt a Hub lookup.
    """

    def _boom(*a: object, **k: object) -> str:
        raise AssertionError("a non-Hub model id must never be resolved")

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", _boom)
    assert _ctx()._commit_sha_for("yamnet") is None  # noqa: SLF001
    assert _ctx()._commit_sha_for(None) is None  # noqa: SLF001


def test_commit_sha_for_a_hub_id_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``org/name``-shaped id is resolved through the run-scoped resolver."""
    sha = "c" * 40
    seen: list[str] = []

    def _record(repo_id: str, *a: object, **k: object) -> str:
        """Record the id it was asked to resolve, so the assertion can check what was passed."""
        seen.append(repo_id)
        return sha

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", _record)
    assert _ctx()._commit_sha_for("openai/whisper-tiny") == sha  # noqa: SLF001
    assert seen == ["openai/whisper-tiny"]


def _raising(exc: BaseException) -> Callable[..., str]:
    """Return a `resolve_revision` stand-in that fails the way `resolve_revision` itself fails.

    The wrapping matters to the three tests below, not just the exception type: `_resolve_uncached`
    raises `RevisionResolutionError(...) from <the Hub error>`, and `_commit_sha_for` reads
    `__cause__` to decide which of the three outcomes it is looking at. A stub that raised the Hub
    error bare would exercise a path production never takes.
    """

    def _stub(*a: object, **k: object) -> str:
        raise RevisionResolutionError("resolution failed") from exc

    return _stub


def test_commit_sha_for_a_hub_shaped_id_the_hub_lacks_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A definitive not-found is the one failure that may become ``None``.

    YAMNet's ``"google/yamnet"`` is the real case: it contains a ``/`` but loads via a TensorFlow
    subprocess venv, so the Hub 404s. The Hub *answered* — there is no commit — so ``None`` is the
    correct key component rather than a degradation, and it is the same answer every run, so no two
    commits can ever collide behind it. `__new__` builds the error without the httpx response its
    constructor wants.
    """
    from huggingface_hub.errors import RepositoryNotFoundError

    monkeypatch.setattr(
        "senselab.utils.model_revision.resolve_revision",
        _raising(RepositoryNotFoundError.__new__(RepositoryNotFoundError)),
    )
    assert _ctx()._commit_sha_for("google/yamnet") is None  # noqa: SLF001


def test_a_local_path_is_rejected_client_side_by_the_hub_client() -> None:
    """Pin the exception a local-path model id actually produces, unmocked and offline.

    The degrade branch keys off `HFValidationError`, so this asserts the premise the stub in the
    next test would otherwise be free to invent. `model_info` validates the repo-id *shape* before
    it opens a connection, so this runs with no network and cannot flake — which is the property
    that makes the verdict definitive rather than a "could not tell". It is also a live check on
    `huggingface_hub`: were a future release to raise something else (or to reach the network and
    return a 404), the degrade set here would need revisiting rather than silently reverting to the
    abort this PR removes.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    with pytest.raises(HFValidationError) as caught:
        HfApi().model_info(repo_id="/scratch/models/foo", revision="main")
    # Not a RepositoryNotFoundError, which is why it needs its own arm in the degrade set.
    assert not isinstance(caught.value, RepositoryNotFoundError)


def test_commit_sha_for_a_local_path_degrades_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """A local filesystem path is the *other* definitive not-a-Hub-model.

    ``/scratch/models/foo`` contains a ``/``, so it trips the Hub-id heuristic exactly as
    ``google/yamnet`` does — but the client rejects it before any request (see the test above).
    Deterministic and offline, so ``None`` is a stable key component; propagating instead would
    abort cache-key computation for every run that names a local checkpoint, which is the case the
    commit message cites as motivation.
    """
    from huggingface_hub.errors import HFValidationError

    monkeypatch.setattr(
        "senselab.utils.model_revision.resolve_revision",
        _raising(HFValidationError("Repo id must be in the form 'repo_name' or 'namespace/repo_name'")),
    )
    assert _ctx()._commit_sha_for("/scratch/models/foo") is None  # noqa: SLF001


def test_commit_sha_for_propagates_a_transient_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure must abort, because "we could not tell" is unsound for a *key*.

    Distinct from `signal.resolved_commit_sha`, which degrades this same failure to ``None`` — and
    correctly, because it fills in a provenance *record*. Here ``None`` is a value in the cache-key
    payload, so a 429 during one run and a success in the next puts two different commits' results
    in the same bucket. The load would very likely have succeeded, which is what makes swallowing
    this a silent-staleness bug rather than a crash avoided.
    """
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", _raising(ConnectionError("reset")))
    with pytest.raises(RevisionResolutionError):
        _ctx()._commit_sha_for("openai/whisper-tiny")  # noqa: SLF001


def test_commit_sha_for_propagates_a_gated_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gated repo propagates even though `GatedRepoError` *subclasses* `RepositoryNotFoundError`.

    The subclassing is the trap this test exists for: the repo demonstrably exists and has commits,
    we simply lack the licence to see them, so it is a "could not tell", not a not-found. A bare
    ``except RepositoryNotFoundError`` would silently take the degrade branch here.
    """
    from huggingface_hub.errors import GatedRepoError

    monkeypatch.setattr(
        "senselab.utils.model_revision.resolve_revision", _raising(GatedRepoError.__new__(GatedRepoError))
    )
    with pytest.raises(RevisionResolutionError):
        _ctx()._commit_sha_for("some/gated-model")  # noqa: SLF001


def test_align_key_differs_from_the_task_key() -> None:
    """Alignment keying stays independent of the ASR cache."""
    ctx = _ctx()
    align = ctx.align_key_for(
        transcript_sha="c" * 64, language="en", aligner_model_id="facebook/mms-1b-all", aligner_params={}
    )
    assert align != ctx.cache_key_for("alignment", "facebook/mms-1b-all", {})


def test_align_key_tracks_the_transcript() -> None:
    """A changed transcript must re-align rather than replay stale timestamps."""
    ctx = _ctx()
    kwargs = {"language": "en", "aligner_model_id": "facebook/mms-1b-all", "aligner_params": {}}
    assert ctx.align_key_for(transcript_sha="a", **kwargs) != ctx.align_key_for(transcript_sha="b", **kwargs)  # type: ignore[arg-type]


# ── provenance ────────────────────────────────────────────────────────


def test_provenance_records_the_stage_code_version() -> None:
    """The whole point of STAGE_VERSIONS is that a stale replay is diagnosable."""
    from senselab.utils.tasks.cached_inference import CACHE_SCHEMA_VERSION

    prov = _ctx().provenance_for("asr", "whisper", {"device": "auto"})
    assert prov["code_version"] == "asr@1"
    # Compared against the constant, not a literal: the point of the assertion is that the
    # recorded version tracks the real one, and a literal turns every legitimate bump into a
    # test edit that says nothing.
    assert prov["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert prov["pass"] == "raw"
    assert prov["device"] == "auto"


def test_provenance_records_the_commit_that_produced_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both "revision" and "commit_sha" travel: what was asked for, and what actually ran."""
    sha = "d" * 40
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    ctx = _ctx()
    prov = ctx.provenance_for("asr", "openai/whisper-large-v3-turbo", {"device": "cpu"})
    assert prov["commit_sha"] == sha
    assert prov["revision"] == "main"


def test_provenance_revision_is_none_when_nothing_is_pinned() -> None:
    """A model-less stage (e.g. "features") has no ref to have asked for, so both fields are None."""
    prov = _ctx().provenance_for("features", None, {})
    assert prov["commit_sha"] is None
    assert prov["revision"] is None


def test_provenance_revision_is_none_for_a_non_hub_backend() -> None:
    """A local backend name (no "/") is never resolved, so "revision" must not claim one was asked."""
    prov = _ctx().provenance_for("yamnet", "yamnet", {})
    assert prov["commit_sha"] is None
    assert prov["revision"] is None


def test_provenance_joins_to_build_cache_index(tmp_path: Path) -> None:
    """summary.json's audio_signature must resolve in the adaptive cache index.

    `adaptive/loop.py` reads summary["passes"][label]["audio_signature"];
    `build_cache_index` keys entries on provenance.audio_signature. If the two
    ever diverge the index misses silently and U2 escalation never fires.
    """
    from senselab.audio.workflows.audio_analysis.adaptive.interventions import build_cache_index

    audio = SimpleNamespace(waveform=torch.tensor([[0.1, 0.2, 0.3]]), sampling_rate=16000)
    sig = audio_signature(audio)
    ctx = StageContext(perturbation="raw", audio_signature=sig, cache_dir=tmp_path, senselab_ver="v")

    key = ctx.cache_key_for("asr", "openai/whisper-tiny", {})
    cache_store(
        tmp_path,
        key,
        {"status": "ok", "result": [], "provenance": ctx.provenance_for("asr", "openai/whisper-tiny", {})},
    )

    index = build_cache_index(tmp_path)
    assert (sig, "asr", "openai/whisper-tiny") in index, f"index keys: {list(index)[:3]}"


# ── sidecars ──────────────────────────────────────────────────────────


def test_write_sidecar_creates_nested_paths(tmp_path: Path) -> None:
    """Stages write per-model sidecars under the pass dir."""
    _ctx(out_dir=tmp_path).write_sidecar(Path("diarization") / "m.json", {"a": 1})
    assert (tmp_path / "diarization" / "m.json").exists()


def test_write_sidecar_is_a_noop_without_out_dir(tmp_path: Path) -> None:
    """Headless mode for the adaptive loop: no out_dir → no files, no error."""
    _ctx(out_dir=None).write_sidecar("x.json", {"a": 1})
    assert not list(tmp_path.iterdir())


# ── PassPlan ──────────────────────────────────────────────────────────


def test_pass_plan_defaults_to_running_nothing_expensive() -> None:
    """Absence means skip — an empty plan must not imply "run every model"."""
    plan = PassPlan()
    assert plan.diarization_models == () and plan.asr_models == ()
    assert plan.ast_model is None and plan.yamnet_model is None
    assert plan.features is False


def test_pass_plan_is_frozen() -> None:
    """Immutable so a plan can't be mutated mid-pass (the args.skip bug class)."""
    with pytest.raises(Exception):  # noqa: B017 — dataclasses raise FrozenInstanceError
        PassPlan().features = True  # type: ignore[misc]


def test_stage_context_is_frozen() -> None:
    """Same reasoning: the run environment must not drift between stages."""
    with pytest.raises(Exception):  # noqa: B017
        _ctx().perturbation = "other"  # type: ignore[misc]


# ── import weight ─────────────────────────────────────────────────────


def test_stage_context_import_stays_light() -> None:
    """Computing a cache key must not drag in torch/transformers.

    `DeviceType` is behind TYPE_CHECKING precisely for this: a caller that only wants a cache key
    shouldn't pay for the ML stack. Run in a subprocess because the parent test session has already
    imported everything.

    The assertion is made *after* a real `cache_key_for` call, not merely after the import, because
    the version of this guard that only imported the module could not fail: `_commit_sha_for`'s
    resolver import is deferred inside the function body, so an import of the whole ML stack from
    there was invisible to it — which is exactly how one got added and shipped. Resolution is
    stubbed so the call makes no Hub request; the stub is installed by importing `model_revision`
    directly, which is itself part of what must stay torch-free.
    """
    code = (
        "import sys; "
        "import senselab.utils.model_revision as r; "
        "r.resolve_revision = lambda repo_id, ref='main', **kw: 'f' * 40; "
        "import senselab.audio.workflows.audio_analysis.stage_context as m; "
        "ctx = m.StageContext(perturbation='raw', audio_signature='a' * 64, senselab_ver='1.2.3'); "
        "ctx.cache_key_for('asr', 'openai/whisper-tiny', {}); "
        "print('transformers' in sys.modules, 'torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    transformers_loaded, torch_loaded = out.stdout.strip().split()
    assert transformers_loaded == "False", "computing a cache key pulled in transformers"
    assert torch_loaded == "False", "computing a cache key pulled in torch"
