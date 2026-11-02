"""Parity logic on synthetic data: the 1.x converter, held-out only, the comparison within a profile."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from senselab_core.kernel.formats import PARITY_OUTPUTS_FORMAT, PARITY_REFERENCE_FORMAT

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str, path: Path) -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


par = _load("parity_reference_under_test", ROOT / "benchmarks" / "parity_reference.py")


def _ids(split: str, n: int) -> list[str]:
    found: list[str] = []
    k = 0
    while len(found) < n:
        if par.split_of(f"u{k}") == split:
            found.append(f"u{k}")
        k += 1
    return found


def _records(directory: Path, task: str, ids: list[str], record: dict[str, Any]) -> None:
    for i in ids:
        (directory / f"{i}.json").write_text(json.dumps({"task": task, **record}), "utf-8")


def _profile(
    tmp_path: Path, items: list[str], tolerances: dict[str, tuple[str, float]], split: str = "calibration"
) -> Path:
    document = {
        "format": "tolerance@1.0",
        "task": "t",
        "backend": "b",
        "derivation": {
            "shas": ["a" * 40],
            "jobs": ["1"],
            "split": split,
            "items": items,
            "contexts": ["cpu/s/batch-1"],
        },
        "fields": [{"path": p, "kind": k, "spreads": [], "tolerance": t} for p, (k, t) in tolerances.items()],
        "findings": [],
    }
    path = tmp_path / "b.yaml"
    path.write_text(yaml.safe_dump(document), "utf-8")
    return path


@pytest.mark.logic
def test_converter_reads_1x_output_files(tmp_path: Path) -> None:
    """Each task's 1.x records become the 2.x-keyed flattened fields, in a `parity-reference@1.0`."""
    flatten = par.tolerance_module().flatten
    ids = _ids("heldout", 2)
    asr, dia, emb = tmp_path / "asr", tmp_path / "dia", tmp_path / "emb"
    for d in (asr, dia, emb):
        d.mkdir()
    _records(asr, "speech_to_text", ids, {"lines": [{"text": " hello ", "avg_logprob": -0.1}]})
    _records(
        dia,
        "diarize",
        ids,
        {"lines": [{"speaker": "B", "start": 1.0, "end": 2.0}, {"speaker": "A", "start": 0.0, "end": 0.5}]},
    )
    _records(emb, "speaker_embeddings", ids, {"vector": [[0.5, 1.5]]})
    kwargs = {"backend": "b", "alpha_sha": "f" * 40, "job": "9"}
    ref = par.convert(asr, task="speech_to_text", **kwargs)
    assert ref.format == PARITY_REFERENCE_FORMAT and ref.alpha_sha == "f" * 40
    assert {i.id for i in ref.items} == set(ids)
    fields = ref.items[0].fields
    assert fields["transcript.segments[0].text"] == '"hello"'
    assert fields["transcript.segments[0].confidence"][0] == pytest.approx(0.9048374, abs=1e-6)
    turns = par.convert(dia, task="diarize", **kwargs).items[0].fields
    assert turns == flatten(
        {"turns": {"start_ns": [0, 1_000_000_000], "end_ns": [500_000_000, 2_000_000_000], "label": ["A", "B"]}}
    )
    assert par.convert(emb, task="speaker_embeddings", **kwargs).items[0].fields == {"embedding.vector": [0.5, 1.5]}


@pytest.mark.logic
def test_converter_refuses_unreadable_records(tmp_path: Path) -> None:
    """A record of another task, or one with neither lines nor a vector, is not converted."""
    ids = _ids("heldout", 1)
    _records(tmp_path, "diarize", ids, {"lines": []})
    with pytest.raises(par.ParityError, match="another task"):
        par.convert(tmp_path, task="speech_to_text", backend="b", alpha_sha="f" * 40, job="1")
    _records(tmp_path, "diarize", ids, {"other": 1})
    with pytest.raises(par.ParityError, match="not one the converter reads"):
        par.convert(tmp_path, task="diarize", backend="b", alpha_sha="f" * 40, job="1")


@pytest.mark.logic
def test_parity_uses_heldout_only(tmp_path: Path) -> None:
    """Calibration items are refused by the converter, the 2.x run and the comparison; so is a held-out profile."""
    cal, held = _ids("calibration", 2), _ids("heldout", 2)
    _records(tmp_path, "speaker_embeddings", cal[:1], {"vector": [1.0]})
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.convert(tmp_path, task="speaker_embeddings", backend="b", alpha_sha="f" * 40, job="1")

    class Worker:
        def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG002
            return [{"id": i["id"], "outputs": {"embedding": {"vector": [1.0]}}} for i in items]

    kwargs: dict[str, Any] = {"task": "t", "backend": "b", "stack": "s", "device": "cpu", "sha": "a" * 40, "job": "1"}
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.run(Worker(), {cal[0]: {}}, {}, **kwargs)
    out = par.run(Worker(), {held[0]: {}, held[1]: {}}, {}, **kwargs)
    assert out.format == PARITY_OUTPUTS_FORMAT and set(out.items[0].batches) == {"1", "N"}
    profile = par.load_profile(_profile(tmp_path, cal, {"embedding.vector": ("numeric", 0.0)}))
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.compare_items({cal[0]: {"embedding.vector": [1.0]}}, {}, profile)
    with pytest.raises(par.ParityError, match="not a calibration-only profile"):
        par.load_profile(_profile(tmp_path, held, {"embedding.vector": ("numeric", 0.0)}))
    with pytest.raises(par.ParityError, match="not a calibration-only profile"):
        par.load_profile(_profile(tmp_path, cal, {"embedding.vector": ("numeric", 0.0)}, split="heldout"))


@pytest.mark.logic
def test_compare_judges_by_the_profile(tmp_path: Path) -> None:
    """Within tolerance passes; beyond it, a shape change, a missing or unprofiled path, an unequal string fail."""
    cal, held = _ids("calibration", 1), _ids("heldout", 1)[0]
    profile = par.load_profile(
        _profile(tmp_path, cal, {"v": ("numeric", 0.25), "t": ("exact", 0.0), "c": ("numeric", 0.0)})
    )
    expected = {"v": [1.0, 2.0], "t": '"a"', "c": [1.0]}
    assert par.compare(expected, {"v": [1.2, 2.0], "t": '"a"', "c": [1.0]}, profile) == []
    found = par.compare(expected, {"v": [1.3, 2.0], "t": '"b"', "c": [1.0, 1.0]}, profile)
    assert [f.split(":")[0] for f in found] == ["c", "t", "v"]
    assert "exceeds tolerance 0.25" in found[2] and "2 values" not in found[2]
    assert par.compare({"v": [1.0], "x": [1.0]}, {"v": [1.0]}, profile) == ["x: missing from the compared output"]
    assert par.compare({"x": [1.0]}, {"x": [1.0]}, profile) == ["x: the profile has no tolerance for it"]
    assert par.compare_items({held: expected}, {}, profile) == {held: ["item has no output"]}


@pytest.mark.logic
def test_reference_producer_imports_no_2x() -> None:
    """The 1.x producer runs under 1.x: it imports `senselab` and the standard library, nothing of 2.x."""
    tree = ast.parse((ROOT / "benchmarks" / "parity_reference_1x.py").read_text("utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert roots <= {"argparse", "json", "pathlib", "typing", "senselab"}


@pytest.mark.logic
def test_diarization_labels_compared_under_best_mapping(tmp_path: Path) -> None:
    """A speaker-label permutation alone is no mismatch; a turn moved in time or a merged speaker is."""
    flatten = par.tolerance_module().flatten

    def turns(start: list[int], end: list[int], label: list[str]) -> dict[str, Any]:
        flat: dict[str, Any] = flatten({"turns": {"start_ns": start, "end_ns": end, "label": label}})
        return flat

    cal = _ids("calibration", 1)
    paths = {"turns.start_ns": 0.0, "turns.end_ns": 0.0, **{f"turns.label[{n}]": 0.0 for n in range(3)}}
    profile = par.load_profile(
        _profile(tmp_path, cal, {p: ("exact" if "label" in p else "numeric", t) for p, t in paths.items()})
    )
    ref = turns([0, 10, 20], [10, 20, 30], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"])
    swapped = turns([0, 10, 20], [10, 20, 30], ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01"])
    assert par.compare(ref, swapped, profile) == []
    merged = turns([0, 10, 20], [10, 20, 30], ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"])
    assert any("label[1]" in f for f in par.compare(ref, merged, profile))
    shifted = turns([0, 10, 20], [10, 20, 31], ["SPEAKER_01", "SPEAKER_00", "SPEAKER_01"])
    assert any("end_ns" in f for f in par.compare(ref, shifted, profile))


@pytest.mark.logic
def test_real_script_path_resolves_every_suite_task() -> None:
    """The script's own task lookup, for one benchmark of each task of the smoke suite, finds the task and backend."""
    tol = par.tolerance_module()
    suite = tol._harness().Suite.load(ROOT / "benchmarks" / "suites" / "smoke.yaml")
    seen: dict[str, Any] = {}
    for bench in suite.benchmarks:
        seen.setdefault(bench.task, bench)
    assert {"speech_to_text", "diarize", "speaker_embeddings", "pii_spans"} <= set(seen)
    importlib.import_module("senselab")
    from senselab_core.runtime.call import packaged  # noqa: PLC0415

    for task, bench in seen.items():
        if task.startswith("signal."):
            continue
        backend = packaged().registry.backends(task)[0].name
        spec, found = tol.resolve(bench.task, backend)
        assert spec.name == task and found.name == backend
