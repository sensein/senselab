"""Tests for the step provenance record: path-free artifacts, the code agent and both environments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.prov_bep028 import SHA256_ALGORITHM, check_graph, convert_store_file, store_id
from senselab.utils.prov_store import CHECKSUM_KEY, PATH_KEY, ProvStore
from senselab.utils.reproduce.digest import input_set_digest
from senselab.utils.reproduce.env import INSTALLED, MARKER, VenvResult, installed_sha256
from senselab.utils.reproduce.host import HOST_KEYS
from senselab.utils.reproduce.provenance import (
    DRAW_GIVEN,
    DRAW_KEYS,
    DRIVER_DESCRIPTION,
    KEY_INPUTS,
    LABEL,
    STORE_FILE,
    Artifact,
    drop_unrecordable_host_values,
    record_step,
)
from tests.utils.reproduce.run_test import needs_userns

COMMIT = "c" * 40
STEP_KEY = "a" * 64
TORCH_LOCK = "4" * 64
BODY_LOCK = "5" * 64
PROBE = "/data/ROOT/subjPX9931/rec.wav"
HOST_STRING_KEYS = ("hostname", "cpu_model", "gpu_name", "gpu_driver", "python", "torch", "torch_cuda", "slurm_job_id")
# Distinctive names that stand for participant identifiers in data paths; none may reach provenance.
SECRETS = (
    "ROOTqz7wx",
    "subjPX9931",
    "PX9931",
    "recPX9931task",
    ".wav",
    "OUTkk42vv",
    "predPX9931",
    ".csv",
    "STDOUTzz8",
    "LOGLINEyy3",
)


def _host() -> dict[str, Any]:
    return {key: None for key in HOST_KEYS} | {
        "hostname": "node042",
        "cpu_model": "Test CPU",
        "cpu_count": 8,
        "cpus_allowed": 4,
        "gpu_count": 0,
        "python": "3.12.11",
        "torch": "2.5.1",
    }


def _venv(tmp_path: Path) -> VenvResult:
    """A venv directory with an ``installed.json`` and the marker recording its sha256."""
    path = tmp_path / "cache" / "venvs" / "repro-demo-0123abcd"
    path.mkdir(parents=True, exist_ok=True)
    data = json.dumps({"freeze": ["numpy==2.1.0", "torch==2.5.1+cu128"]}).encode()
    (path / INSTALLED).write_bytes(data)
    (path / MARKER).write_text(json.dumps({"installed_sha256": hashlib.sha256(data).hexdigest()}))
    return VenvResult(
        path=path,
        env_key="e" * 64,
        overlay_sha256=None,
        reused=False,
        packages={"numpy": "2.1.0", "torch": "2.5.1"},
        checked=True,
    )


def _kwargs(tmp_path: Path, **change: object) -> dict[str, Any]:
    """A minimal valid ``record_step`` call, with ``change`` applied."""
    return {
        "step_key": STEP_KEY,
        "step": "classify",
        "commit_sha": COMMIT,
        "venv": _venv(tmp_path),
        "torch_lock_sha256": TORCH_LOCK,
        "body_lock_sha256": BODY_LOCK,
        "host": _host(),
        "inputs": [],
        "outputs": [],
        "started": "2026-10-07T10:00:00+00:00",
        "ended": "2026-10-07T10:05:00+00:00",
        "exit_code": 0,
    } | change


@pytest.fixture
def recorded(tmp_path: Path) -> dict[str, Any]:
    """A step whose input and output names carry identifiers, its logs, and the provenance written for it."""
    root = tmp_path / "ROOTqz7wx"
    (root / "subjPX9931").mkdir(parents=True)
    (root / "subjPX9931" / "recPX9931task.wav").write_bytes(b"RIFF....")
    out = tmp_path / "OUTkk42vv"
    out.mkdir()
    (out / "predPX9931.csv").write_text("id,label\nPX9931,1\n")
    (out / "logs").mkdir()
    (out / "logs" / "stdout.txt").write_text("STDOUTzz8 subjPX9931\n")
    (out / "logs" / "driver.log").write_text("LOGLINEyy3\n")
    in_digest, in_count = input_set_digest(root, ["subjPX9931/*.wav"], require_nonempty=True)
    out_digest, out_count = input_set_digest(out, ["*.csv"], require_nonempty=True)
    inputs = [Artifact(digest=in_digest, count=in_count, root="projects")]
    outputs = [Artifact(digest=out_digest, count=out_count)]
    written = record_step(
        out / "prov",
        **_kwargs(
            tmp_path,
            inputs=inputs,
            outputs=outputs,
            codes=["W_CLASS_ABSENT"],
            env_keys=["HOME", "PATH", "TZ"],
            env_sha256="1" * 64,
            argv_sha256="2" * 64,
        ),
    )
    return {"tmp": tmp_path, "out": out, "written": written, "inputs": inputs, "outputs": outputs}


def _graph(recorded: dict[str, Any]) -> dict[str, Any]:
    return convert_store_file(recorded["out"] / "prov" / STORE_FILE, label=LABEL, description=DRIVER_DESCRIPTION)


def test_writes_the_store_and_the_bep028_files(recorded: dict[str, Any]) -> None:
    """``store.jsonl`` plus the four ``prov-reproduce_*`` files, all inside the prov dir."""
    prov = recorded["out"] / "prov"
    names = sorted(path.name for path in recorded["written"])
    assert names == sorted(
        [STORE_FILE, "prov-reproduce_io.json", "prov-reproduce_act.json", "prov-reproduce_soft.json"]
        + ["prov-reproduce_env.json"]
    )
    assert all(path.parent == prov for path in recorded["written"])
    assert sorted(path.name for path in prov.iterdir()) == names


def test_no_path_or_file_name_reaches_any_provenance_file(recorded: dict[str, Any]) -> None:
    """No input or output path, file name, data-root-relative path, pattern or log content is exported."""
    forbidden = [*SECRETS, str(recorded["tmp"]), "subjPX9931/*.wav", "*.csv", "stdout.txt", "driver.log"]
    for path in recorded["written"]:
        text = path.read_text()
        assert [secret for secret in forbidden if secret in text] == [], path.name


def test_the_graph_is_valid_and_refers_only_to_itself(recorded: dict[str, Any]) -> None:
    """The exported graph passes the BEP028 checker."""
    assert check_graph(_graph(recorded)) == []


def test_inputs_and_outputs_are_path_free_artifacts(recorded: dict[str, Any]) -> None:
    """Each declared input and output is an ``artifact`` entity carrying only its digest and logical name."""
    store = ProvStore.read_jsonl(recorded["out"] / "prov" / STORE_FILE)
    artifacts = store.entities("artifact")
    assert len(store.entities()) == len(artifacts) == 2
    by_role = {entity.attributes["role"]: entity for entity in artifacts}
    assert by_role["input"].attributes == {
        "role": "input",
        "declared": "inputs[0]",
        "root": "projects",
        CHECKSUM_KEY: recorded["inputs"][0].digest,
        "file_count": 1,
    }
    assert by_role["output"].attributes == {
        "role": "output",
        "declared": "outputs[0]",
        CHECKSUM_KEY: recorded["outputs"][0].digest,
        "file_count": 1,
    }
    assert all(PATH_KEY not in entity.attributes for entity in artifacts)
    [activity] = store.activities()
    assert store.uses_of(activity.id) == [by_role["input"].id]
    assert store.generated_by(by_role["output"].id) == activity.id

    graph = _graph(recorded)
    assert "Files" not in graph["Records"]
    checksums = {node["Checksum"][0]["ChecksumValue"] for node in graph["Records"]["prov:Entity"]}
    assert checksums == {recorded["inputs"][0].digest, recorded["outputs"][0].digest}
    assert {node["Checksum"][0]["ChecksumAlgorithm"] for node in graph["Records"]["prov:Entity"]} == {SHA256_ALGORITHM}
    assert all("AtLocation" not in node for node in graph["Records"]["prov:Entity"])


def test_the_code_commit_is_a_software_agent(recorded: dict[str, Any]) -> None:
    """The step's code commit is the software agent that ran the activity."""
    store = ProvStore.read_jsonl(recorded["out"] / "prov" / STORE_FILE)
    [agent] = store.agents()
    assert agent.agent_type == "software"
    assert agent.commit_sha == COMMIT
    [activity] = store.activities()
    assert store.associated_with(activity.id) == [agent.id]
    [software] = _graph(recorded)["Records"]["Software"]
    assert software["Label"] == "classify"
    assert software["Version"] == COMMIT


def test_the_venv_and_the_host_are_environments(recorded: dict[str, Any]) -> None:
    """The venv carries its packages; the host carries the machine and the driver."""
    store = ProvStore.read_jsonl(recorded["out"] / "prov" / STORE_FILE)
    [venv] = store.environments("venv")
    assert venv.label == "repro-demo-0123abcd"
    assert venv.python_version == "3.12.11"
    assert venv.dependencies == {"numpy": "2.1.0", "torch": "2.5.1"}
    assert venv.dependencies_digest is None
    [host] = store.environments("host")
    assert host.label == "node042"
    assert host.operating_system
    assert host.senselab_version
    assert {node["Label"] for node in _graph(recorded)["Records"]["Environments"]} == {"node042", venv.label}


def test_the_activity_holds_the_run_and_withholds_the_command(recorded: dict[str, Any]) -> None:
    """Times, exit, codes, the host record and digests of argv and environment; ``Command`` stays null."""
    [activity] = _graph(recorded)["Records"]["Activities"]
    assert activity["Command"] is None
    assert activity["Description"] == DRIVER_DESCRIPTION.replace("{label}", "reproduce / classify")
    assert activity["StartedAtTime"] == "2026-10-07T10:00:00+00:00"
    assert activity["EndedAtTime"] == "2026-10-07T10:05:00+00:00"
    parameters = activity["Parameters"]
    assert parameters["step_key"] == STEP_KEY
    assert parameters["env_key"] == "e" * 64
    assert parameters["torch_lock_sha256"] == TORCH_LOCK
    assert parameters["body_lock_sha256"] == BODY_LOCK
    assert parameters["overlay_sha256"] is None
    assert parameters["installed_sha256"] == installed_sha256(
        recorded["tmp"] / "cache" / "venvs" / "repro-demo-0123abcd"
    )
    assert parameters["exit_code"] == 0
    assert parameters["codes"] == ["W_CLASS_ABSENT"]
    assert parameters["env_keys"] == ["HOME", "PATH", "TZ"]
    assert parameters["env_sha256"] == "1" * 64
    assert parameters["argv_sha256"] == "2" * 64
    assert parameters["host"] == _host()
    store = ProvStore.read_jsonl(recorded["out"] / "prov" / STORE_FILE)
    inputs = {entity.id for entity in store.entities() if entity.attributes["role"] == "input"}
    assert {store_id(uri) for uri in activity["Used"]} == inputs


def test_the_same_step_records_the_same_ids(tmp_path: Path) -> None:
    """Ids follow content, so rerunning a step with the same inputs records the same graph identities."""
    kwargs = _kwargs(tmp_path, inputs=[Artifact(digest="3" * 64, count=2, root="projects")])
    record_step(tmp_path / "a", **kwargs)
    record_step(tmp_path / "b", **kwargs)
    assert (tmp_path / "a" / STORE_FILE).read_text() == (tmp_path / "b" / STORE_FILE).read_text()


def test_a_rerun_removes_files_the_new_record_does_not_write(tmp_path: Path) -> None:
    """A rerun into the same prov dir leaves no stale BEP028 group from the earlier record."""
    prov = tmp_path / "prov"
    first = record_step(prov, **_kwargs(tmp_path, inputs=[Artifact(digest="3" * 64, count=2, root="projects")]))
    assert prov / f"prov-{LABEL}_io.json" in first
    (prov / "notes.txt").write_text("kept")
    second = record_step(prov, **_kwargs(tmp_path))
    assert prov / f"prov-{LABEL}_io.json" not in second
    assert sorted(prov.iterdir()) == sorted([*second, prov / "notes.txt"])


def test_the_installed_digest_is_the_markers(tmp_path: Path) -> None:
    """The recorded ``installed_sha256`` is the venv marker's, and a venv whose file no longer matches is refused."""
    kwargs = _kwargs(tmp_path)
    venv_path = kwargs["venv"].path
    record_step(tmp_path / "prov", **kwargs)
    [activity] = ProvStore.read_jsonl(tmp_path / "prov" / STORE_FILE).activities()
    marker = json.loads((venv_path / MARKER).read_text())
    assert activity.parameters["installed_sha256"] == marker["installed_sha256"]
    (venv_path / INSTALLED).write_text("{}")
    with pytest.raises(ValueError, match="installed_sha256"):
        record_step(tmp_path / "again", **kwargs)
    assert not (tmp_path / "again").exists()


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"step": "sub/01"}, "step"),
        ({"step": "."}, "step"),
        ({"step": ".."}, "step"),
        ({"step": ".hidden"}, "step"),
        ({"step_key": "../x"}, "step_key"),
        ({"step_key": 7}, "step_key"),
        ({"commit_sha": "main"}, "commit"),
        ({"torch_lock_sha256": PROBE}, "torch_lock_sha256"),
        ({"body_lock_sha256": None}, "body_lock_sha256"),
        ({"inputs": [Artifact(digest="0" * 63, count=1, root="projects")]}, "digest"),
        ({"inputs": [Artifact(digest="0" * 64, count=-1, root="projects")]}, "count"),
        ({"inputs": [Artifact(digest="0" * 64, count=True, root="projects")]}, "count"),
        ({"inputs": [Artifact(digest="0" * 64, count=1, root="data/subjPX9931")]}, "root"),
        ({"inputs": [Artifact(digest="0" * 64, count=1, root="up:..")]}, "root"),
        ({"inputs": [Artifact(digest="0" * 64, count=1, root="..")]}, "root"),
        ({"inputs": [Artifact(digest="0" * 64, count=1, root=".data")]}, "root"),
        ({"inputs": [Artifact(digest="0" * 64, count=1)]}, "root"),
        ({"outputs": [Artifact(digest="0" * 64, count=1, root="projects")]}, "root"),
        ({"host": {"hostname": "n"}}, "host"),
        ({"host": _host() | {"log": "x"}}, "host"),
        ({"host": _host() | {"cpu_model": ["a"]}}, "host"),
        ({"host": _host() | {"cpu_count": True}}, "host"),
        ({"host": _host() | {"cpu_model": "x" * 129}}, "host"),
        ({"host": _host() | {"cpu_model": "Test  CPU"}}, "host"),
        *[({"host": _host() | {key: PROBE}}, f"host {key}") for key in sorted(HOST_STRING_KEYS)],
        ({"codes": ["see /tmp/x"]}, "code"),
        ({"env_keys": ["A=B"]}, "env"),
        ({"env_sha256": "x"}, "env_sha256"),
        ({"env_sha256": 123}, "env_sha256"),
        ({"argv_sha256": "x"}, "argv_sha256"),
        ({"exit_code": True}, "exit_code"),
        ({"started": "/data/x"}, "started"),
        ({"started": "2026-10-07T10:00:00"}, "started"),
        ({"ended": "2026-10-07"}, "ended"),
    ],
)
def test_refuses_values_that_could_carry_a_path(tmp_path: Path, change: dict[str, Any], match: str) -> None:
    """Every free value is checked against a closed form before anything is written."""
    with pytest.raises(ValueError, match=match):
        record_step(tmp_path / "prov", **_kwargs(tmp_path, **change))
    assert not (tmp_path / "prov").exists()


@pytest.mark.parametrize(
    "change",
    [
        {"overlay_sha256": PROBE},
        {"packages": {PROBE: "1.0"}},
        {"packages": {"numpy": PROBE}},
        {"packages": {"NumPy": "1.0"}},
        {"packages": {"numpy": "1.0 rc"}},
        {"env_key": PROBE},
    ],
)
def test_refuses_venv_fields_that_could_carry_a_path(tmp_path: Path, change: dict[str, Any]) -> None:
    """The venv's overlay, key and package names and versions are closed forms too."""
    kwargs = _kwargs(tmp_path)
    kwargs["venv"] = kwargs["venv"]._replace(**change)
    with pytest.raises(ValueError, match="venv"):
        record_step(tmp_path / "prov", **kwargs)
    assert not (tmp_path / "prov").exists()


def test_the_store_file_is_json_lines(recorded: dict[str, Any]) -> None:
    """Every line of the store parses, so a reader never meets a half-written record."""
    lines = (recorded["out"] / "prov" / STORE_FILE).read_text().splitlines()
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_drop_unrecordable_host_values_nulls_only_off_pattern_values(tmp_path: Path) -> None:
    """Off-pattern strings, bad counts and unknown keys are dropped by name; the result always records."""
    host = _host() | {"gpu_name": PROBE, "cpu_count": True, "cpus_allowed": -1, "extra": "x"}
    clean, dropped = drop_unrecordable_host_values(host)
    assert dropped == ["cpu_count", "cpus_allowed", "extra", "gpu_name"]
    assert set(clean) == set(HOST_KEYS) and clean["gpu_name"] is None and clean["hostname"] == "node042"
    assert PROBE not in json.dumps(clean)
    record_step(tmp_path / "prov", **_kwargs(tmp_path, host=clean))
    assert drop_unrecordable_host_values(_host()) == (_host(), [])


def test_a_symlinked_prov_dir_is_refused(tmp_path: Path) -> None:
    """The record never writes through a symlink in place of ``prov/``."""
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "prov").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="prov_dir"):
        record_step(tmp_path / "prov", **_kwargs(tmp_path))
    assert list(elsewhere.iterdir()) == []


def test_a_derived_venv_name_is_recorded(tmp_path: Path) -> None:
    """A derived venv's ``+<overlay sha8>`` suffix is part of its closed-form name; other ``+`` uses are not."""
    kwargs = _kwargs(tmp_path)
    derived = kwargs["venv"].path.with_name("repro-demo-0123abcd+89abcdef")
    kwargs["venv"].path.rename(derived)
    record_step(tmp_path / "prov", **(kwargs | {"venv": kwargs["venv"]._replace(path=derived)}))
    odd = derived.with_name("repro+demo")
    derived.rename(odd)
    with pytest.raises(ValueError, match="venv name"):
        record_step(tmp_path / "prov2", **(kwargs | {"venv": kwargs["venv"]._replace(path=odd)}))


# --- draw provenance (SC8) ----------------------------------------------------------------------------

DRAW = {"seed": 7, "data_order_seed": 11, "sampling": {"temperature": 0.7, "top_k": 20, "do_sample": True}}

DRAW_STEP = """\
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
seen = {k: os.environ.get(k) for k in ("REPRO_SEED", "REPRO_DATA_ORDER_SEED", "REPRO_SAMPLING")}
(out / "result.json").write_text(json.dumps({"value": 1.0, "seen": seen}))
"""


def _draw_world(tmp_path: Path) -> Any:  # noqa: ANN401
    from tests.utils.reproduce.engine_test import _world
    from tests.utils.reproduce.run_test import _git

    world = _world(tmp_path)
    (world.repo / "draw_step.py").write_text(DRAW_STEP)
    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "draw")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    return world


def _draw_step(world: Any, **change: Any) -> dict[str, Any]:  # noqa: ANN401
    from tests.utils.reproduce.weights_test import REPO, REVISION, WEIGHTS, _weights

    fields: dict[str, Any] = {
        "argv": ["{venv}/bin/python", "{checkout}/draw_step.py", "{out}"],
        "hf_weights": _weights({"weights.bin": WEIGHTS}),
        "engine": {
            "name": "transformers-batched",
            "version": "4.57.1",
            "model": {"repo": REPO, "revision": REVISION},
            "concurrency": 1,
            "batch_size": 4,
        },
        "draw": DRAW,
        **change,
    }
    return dict(world.step("infer", "prep", "", **fields))


def _parameters(out: Path, prov: str = "prov") -> dict[str, Any]:
    records = [json.loads(line) for line in (out / prov / STORE_FILE).read_text().splitlines()]
    [parameters] = [r["parameters"] for r in records if "parameters" in r]
    return dict(parameters)


@needs_userns
def test_draw_and_engine_are_in_the_details(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record's ``draw`` holds seeds, sampling, engine, concurrency, batch size, GPU, CUDA and env lock together."""
    from senselab.utils import cuda_probe
    from senselab.utils.reproduce import engine
    from tests.utils.reproduce.engine_test import DRIVER, _gpu

    monkeypatch.setattr(
        cuda_probe, "detect_host_cuda", lambda: cuda_probe.HostCuda((13, 0), "nvidia-smi", "CUDA Version: 13.0")
    )
    world = _draw_world(tmp_path)
    manifest = world.manifest(_draw_step(world))
    world.prefetch(manifest)
    outcome = world.run(manifest, host_runner=_gpu)["infer"]
    assert outcome.ok and outcome.out is not None, outcome.codes
    parameters = _parameters(outcome.out)
    spec = manifest.steps[0].engine
    assert spec is not None
    assert parameters["details"]["engine"] == engine.engine_identity(spec)
    assert parameters["details"]["draw"] == {
        **DRAW,
        "engine": engine.engine_identity(spec),
        "concurrency": 1,
        "batch_size": 4,
        "gpu_name": "Test GPU",
        "gpu_driver": DRIVER,
        "gpu_count": 1,
        "host_cuda": "13.0",
        "torch_cuda": None,
        "env_key": parameters["env_key"],
        "body_lock_sha256": parameters["body_lock_sha256"],
        "torch_lock_sha256": None,
        "overlay_sha256": None,
        "installed_sha256": parameters["installed_sha256"],
        **{k: parameters["details"]["draw"][k] for k in KEY_INPUTS},
    }
    served = parameters["details"]["draw"]["served_model"]
    assert spec.model.revision in served and served == f"{spec.model.repo}@{spec.model.revision}"
    plain = world.manifest(world.step("plain", "prep", "{root:data}/in/a.txt"))
    world.prefetch(plain)
    bare = world.run(plain, host_runner=_gpu)["plain"]
    assert bare.ok and bare.out is not None, bare.codes
    draw = _parameters(bare.out)["details"]["draw"]
    assert (draw["seed"], draw["data_order_seed"], draw["sampling"], draw["engine"]) == (None, None, {}, None)
    assert (draw["concurrency"], draw["batch_size"], draw["gpu_driver"], draw["host_cuda"]) == (
        None,
        None,
        DRIVER,
        "13.0",
    )
    given = {k: draw[k] for k in DRAW_GIVEN}
    identity = engine.engine_identity(spec)
    ok = {**given, "engine": identity, "concurrency": 1, "batch_size": 4}
    record_step(tmp_path / "ok", **_kwargs(tmp_path, details={"engine": identity, "draw": ok}))
    recorded = _parameters(tmp_path / "ok", prov="")
    completed = recorded["details"]["draw"]
    assert set(completed) == set(DRAW_KEYS) and {k: completed[k] for k in ok} == ok
    for key in ("env_key", "body_lock_sha256", "torch_lock_sha256", "overlay_sha256", "installed_sha256"):
        assert completed[key] == recorded[key], key
    assert {k: completed[k] for k in ("gpu_name", "gpu_driver", "gpu_count")} == {
        k: recorded["host"][k] for k in ("gpu_name", "gpu_driver", "gpu_count")
    }
    bad: dict[str, Any]
    for bad in (
        {"seed": "7"},
        {"seed": True},
        {"seed": 2**40},
        {"sampling": {"temperature": "a b"}},
        {"sampling": {"temperature": "1e-7"}},
        {"sampling": {"stop": "a/b"}},
        {"extra": 1},
        {"gpu_name": "Test GPU"},
        {"engine": identity},
        {"concurrency": 0},
        {"manifest": {"path": "/abs/m.yaml", "sha256": "1" * 64}},
        {"manifest": {"path": "m.yaml"}},
        {"served_model": "org/model@main"},
        {"shard": [3, 3]},
        {"upstream": {"prep": "/x"}},
        {"submodules": {"../x": "c" * 40}},
        {"native": {"ffmpeg": ["/opt/ff/bin"]}},
        {"adapter_sha256": None},
        {"r_session_sha256": "x"},
    ):
        with pytest.raises(ValueError, match="draw"):
            record_step(tmp_path / "bad", **_kwargs(tmp_path, details={"draw": {**given, **bad}}))
    for bad in ({"concurrency": 2}, {"batch_size": 8}, {"engine": None}):
        with pytest.raises(ValueError, match="draw"):
            record_step(tmp_path / "bad", **_kwargs(tmp_path, details={"engine": identity, "draw": {**ok, **bad}}))


def test_draw_changes_the_key(tmp_path: Path) -> None:
    """Each seed and sampling parameter is keyed, and reaches the step env as a keyed literal; bad draws are refused."""
    from pydantic import ValidationError

    from senselab.utils.reproduce.cache import step_key, step_key_payload
    from senselab.utils.reproduce.sandbox import keyed_templates
    from senselab.utils.reproduce.spec import StepSpec
    from tests.utils.reproduce.cache_test import _kwargs as _key_kwargs
    from tests.utils.reproduce.cache_test import _profile, _step_data

    profile = _profile(tmp_path)

    def key(draw: dict[str, Any] | None) -> str:
        return step_key(StepSpec.model_validate(_step_data(draw=draw)), profile, **_key_kwargs())

    step = StepSpec.model_validate(_step_data(draw=DRAW))
    assert step_key_payload(step, profile, **_key_kwargs())["draw"] == DRAW
    keyed = keyed_templates(step, profile)
    assert (keyed["REPRO_SEED"], keyed["REPRO_DATA_ORDER_SEED"]) == ("7", "11")
    assert json.loads(keyed["REPRO_SAMPLING"]) == DRAW["sampling"]
    assert keyed["REPRO_SAMPLING"] == '{"do_sample":true,"temperature":0.7,"top_k":20}'
    bare = keyed_templates(StepSpec.model_validate(_step_data(draw={"seed": 1})), profile)
    assert bare["REPRO_SEED"] == "1" and "REPRO_DATA_ORDER_SEED" not in bare and "REPRO_SAMPLING" not in bare
    assert not {"REPRO_SEED", "REPRO_SAMPLING"} & set(keyed_templates(StepSpec.model_validate(_step_data()), profile))
    base = key(DRAW)
    assert key(None) != base and key({}) == key(None) and key({"sampling": {}}) == key(None)
    assert key({**DRAW, "sampling": {"temperature": 1}}) != key({**DRAW, "sampling": {"temperature": 1.0}})
    for change in (
        {"seed": 8},
        {"seed": None},
        {"data_order_seed": 12},
        {"sampling": {**DRAW["sampling"], "temperature": 0.8}},  # type: ignore[dict-item]
        {"sampling": {**DRAW["sampling"], "top_p": 0.9}},  # type: ignore[dict-item]
        {"sampling": {**DRAW["sampling"], "do_sample": 1}},  # type: ignore[dict-item]
    ):
        assert key({**DRAW, **change}) != base, change
    for bad in (
        {"seed": "7"},
        {"seed": True},
        {"seed": -1},
        {"sampling": {"temperature": float("nan")}},
        {"sampling": {"stop": "a b"}},
        {"sampling": {"Top-P": 0.9}},
        {"sampling": {"temperature": "1e-7"}},
        {"sampling": {"temperature": "inf"}},
        {"seed": 2**32},
        {"extra": 1},
    ):
        with pytest.raises(ValidationError):
            StepSpec.model_validate(_step_data(draw=bad))
    assert StepSpec.model_validate(_step_data(draw={"seed": 2**32 - 1, "sampling": {"mode": "greedy"}})).draw
    loaded = yaml.safe_load("seed: 1\nsampling: {temperature: 1.0e-7, top_k: 5, do_sample: true, mode: beam}\n")
    assert StepSpec.model_validate(_step_data(draw=loaded)).draw.sampling == {  # type: ignore[union-attr]
        "temperature": 1e-7,
        "top_k": 5,
        "do_sample": True,
        "mode": "beam",
    }
    with pytest.raises(ValidationError):
        StepSpec.model_validate(_step_data(draw=yaml.safe_load("sampling: {temperature: 1e-7}\n")))


@needs_userns
def test_draw_env_reaches_the_step(tmp_path: Path) -> None:
    """Inside the sandbox the step sees ``REPRO_SEED``, ``REPRO_DATA_ORDER_SEED`` and ``REPRO_SAMPLING``."""
    world = _draw_world(tmp_path)
    manifest = world.manifest(_draw_step(world))
    world.prefetch(manifest)
    outcome = world.run(manifest)["infer"]
    assert outcome.ok and outcome.out is not None, outcome.codes
    seen = json.loads((outcome.out / "work" / "result.json").read_text())["seen"]
    assert seen["REPRO_SEED"] == "7" and seen["REPRO_DATA_ORDER_SEED"] == "11"
    assert json.loads(seen["REPRO_SAMPLING"]) == DRAW["sampling"]
