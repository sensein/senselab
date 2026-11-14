"""Parity logic on synthetic data: the 1.x converter, held-out only, and the per-type comparison."""

import ast
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from senselab_core.kernel.formats import PARITY_OUTPUTS_FORMAT, PARITY_REFERENCE_FORMAT
from senselab_core.runtime.compare import RunConfigFile, Threshold, Thresholds, flatten

ROOT = Path(__file__).resolve().parents[4]


def _load(name: str, path: Path) -> Any:  # noqa: ANN401
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


par = _load("parity_reference_under_test", ROOT / "benchmarks" / "parity_reference.py")


_OWN = lambda item_id: item_id  # noqa: E731 - the item is its own split unit


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


THRESHOLDS = RunConfigFile.load({"thresholds": {"embedding": {"rtol": 0.0, "atol": 0.25}}}).thresholds
EXACT = Thresholds(by_name={n: Threshold(rtol=0.0, atol=0.0) for n in THRESHOLDS.by_name})


@pytest.mark.logic
def test_converter_reads_1x_output_files(tmp_path: Path) -> None:
    """Each task's 1.x records become the 2.x-keyed flattened fields, each kernel value with its format."""
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
    kwargs = {"backend": "b", "alpha_sha": "f" * 40, "job": "9", "unit_of": _OWN}
    ref = par.convert(asr, task="speech_to_text", **kwargs)
    assert ref.format == PARITY_REFERENCE_FORMAT and ref.alpha_sha == "f" * 40 and ref.flatten == "flatten@2"
    written = tmp_path / "reference.json"
    written.write_text(ref.model_dump_json(), "utf-8")
    assert par.load_reference(written) == ref
    stale = tmp_path / "stale.json"
    stale.write_text(ref.model_dump_json().replace('"flatten@2"', '"flatten@1"'), "utf-8")
    with pytest.raises(par.ParityError, match="flatten@1") as raised:
        par.load_reference(stale)
    assert str(stale) in str(raised.value)
    assert {i.id for i in ref.items} == set(ids)
    fields = ref.items[0].fields
    assert fields["transcript.segments[0].text"] == '"hello"' and fields["transcript.format"] == '"transcript@2.0"'
    assert fields["transcript.segments[0].confidence"][0] == pytest.approx(0.9048374, abs=1e-6)
    turns = par.convert(dia, task="diarize", **kwargs).items[0].fields
    assert turns == flatten(
        {
            "turns": {
                "format": "intervals@1.0",
                "start_ns": [0, 1_000_000_000],
                "end_ns": [500_000_000, 2_000_000_000],
                "label": ["A", "B"],
            }
        }
    )
    assert par.convert(emb, task="speaker_embeddings", **kwargs).items[0].fields == {
        "embedding.format": '"embedding@1.0"',
        "embedding.vector": [0.5, 1.5],
    }


@pytest.mark.logic
def test_converter_refuses_unreadable_records(tmp_path: Path) -> None:
    """A record of another task, or one with neither lines nor a vector, is not converted."""
    ids = _ids("heldout", 1)
    _records(tmp_path, "diarize", ids, {"lines": []})
    with pytest.raises(par.ParityError, match="another task"):
        par.convert(tmp_path, task="speech_to_text", backend="b", alpha_sha="f" * 40, job="1", unit_of=_OWN)
    _records(tmp_path, "diarize", ids, {"other": 1})
    with pytest.raises(par.ParityError, match="not one the converter reads"):
        par.convert(tmp_path, task="diarize", backend="b", alpha_sha="f" * 40, job="1", unit_of=_OWN)


@pytest.mark.logic
def test_parity_uses_heldout_only(tmp_path: Path) -> None:
    """Calibration items are refused by the converter, the 2.x run and the comparison."""
    cal, held = _ids("calibration", 2), _ids("heldout", 2)
    _records(tmp_path, "speaker_embeddings", cal[:1], {"vector": [1.0]})
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.convert(tmp_path, task="speaker_embeddings", backend="b", alpha_sha="f" * 40, job="1", unit_of=_OWN)

    class Worker:
        def run(self, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG002
            return [{"id": i["id"], "outputs": {"embedding": {"vector": [1.0]}}} for i in items]

    kwargs: dict[str, Any] = {
        "task": "t",
        "backend": "b",
        "stack": "s",
        "device": "cpu",
        "model": None,
        "sha": "a" * 40,
        "job": "1",
        "unit_of": _OWN,
    }
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.run(Worker(), {cal[0]: {}}, {}, **kwargs)
    out = par.run(Worker(), {held[0]: {}, held[1]: {}}, {}, **kwargs)
    assert out.format == PARITY_OUTPUTS_FORMAT and set(out.items[0].batches) == {"1", "N"}
    refused = par.ParityOutputs.model_validate(
        {**out.model_dump(), "items": [{"id": cal[0], "split": "calibration", "batches": {}}]}
    )
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.compare_stacks(refused, out, THRESHOLDS)


@pytest.mark.logic
def test_compare_judges_by_each_types_rule_and_threshold() -> None:
    """Within the threshold passes; beyond it, a shape change, a missing path or unequal text fails, naming the rule.

    A 2.x field 1.x never wrote is not judged.
    """
    held = _ids("heldout", 1)[0]
    expected = {"e.format": '"embedding@1.0"', "e.vector": [1.0, 2.0], "t": '"a"', "c": [1.0]}
    assert par.compare(expected, {**expected, "e.vector": [1.2, 2.0], "e.space": '"m"'}, THRESHOLDS) == []
    found = par.compare(expected, {**expected, "e.vector": [1.3, 2.0], "t": '"b"', "c": [1.0, 1.0]}, THRESHOLDS)
    assert [f.split(":")[0] for f in found] == ["c", "e.vector", "t"]
    assert "atol 0.25" in found[1] and found[1].endswith("(close embedding)") and found[2].endswith("(exact)")
    assert par.compare({**expected, "x": [1.0]}, expected, THRESHOLDS) == ["x: present on one side only (missing)"]
    assert par.compare_items({held: expected}, {}, THRESHOLDS) == {held: ["item has no output"]}


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
def test_diarization_labels_compared_under_best_mapping() -> None:
    """A speaker-label permutation alone is no mismatch; a turn moved in time or a merged speaker is."""

    def turns(start: list[int], end: list[int], label: list[str]) -> dict[str, Any]:
        flat: dict[str, Any] = flatten(
            {"turns": {"format": "intervals@1.0", "start_ns": start, "end_ns": end, "label": label}}
        )
        return flat

    profile = EXACT
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
    suite = par.harness().Suite.load(ROOT / "benchmarks" / "suites" / "smoke.yaml")
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
        spec, found = par.resolve(bench.task, backend)
        assert spec.name == task and found.name == backend


@pytest.mark.logic
def test_diarization_speaker_without_counterpart_is_a_mismatch() -> None:
    """An extra or a missing speaker never reads as a match, even when its name equals a mapped one."""

    def turns(label: list[str]) -> dict[str, Any]:
        flat: dict[str, Any] = flatten(
            {"turns": {"format": "intervals@1.0", "start_ns": [0, 10], "end_ns": [10, 20], "label": label}}
        )
        return flat

    profile = EXACT
    one = turns(["SPEAKER_00", "SPEAKER_00"])
    two = turns(["SPEAKER_01", "SPEAKER_00"])
    assert par.compare(one, two, profile)
    assert par.compare(two, one, profile)
    assert par.compare(two, turns(["SPEAKER_00", "SPEAKER_01"]), profile) == []


@pytest.mark.logic
def test_a_parity_decision_records_the_thresholds_of_its_run_config(tmp_path: Path) -> None:
    """A run config's threshold changes the decision, and the record carries that config's thresholds digest."""
    held = _ids("heldout", 1)[0]
    reference = par.ParityReference(
        flatten="flatten@2",
        task="t",
        backend="b",
        alpha_sha="f" * 40,
        job="1",
        items=(par.ReferenceItem(id=held, split="heldout", fields={"e.format": '"embedding@1.0"', "e.vector": [1.0]}),),
    )
    outputs = par.ParityOutputs(
        flatten="flatten@2",
        task="t",
        backend="b",
        stack="s",
        device="cpu",
        model=None,
        sha="a" * 40,
        job="2",
        items=(
            par.OutputItem(id=held, split="heldout", batches={"1": {"e.format": '"embedding@1.0"', "e.vector": [1.5]}}),
        ),
    )
    config = tmp_path / "run.yaml"
    config.write_text("thresholds:\n  embedding:\n    atol: 1.0\n", "utf-8")
    packaged = par.decide(outputs, reference, RunConfigFile.load().thresholds)
    mine = par.decide(outputs, reference, RunConfigFile.load(config).thresholds)
    assert not packaged.agree and packaged.thresholds == RunConfigFile.load().thresholds.canonical_digest()
    assert mine.agree and mine.thresholds == RunConfigFile.load(config).thresholds.canonical_digest()
    assert mine.format == "parity-comparison@1.0" and mine.against == "reference"


@pytest.mark.logic
def test_a_backend_the_profile_cannot_place_is_not_run_and_never_passes() -> None:
    """A cpu-only backend under a cuda-only profile has a placement gap, is recorded unrun, and `decide` refuses it."""
    importlib.import_module("senselab")
    importlib.import_module("senselab_core.runtime.fake_backend")
    profiles = importlib.import_module("senselab_core.runtime.profiles")
    laptop = profiles.ExecutionProfile.load("laptop")
    _, backend = par.resolve("fake.echo", "fake")
    assert par.placement_gap("fake.echo", None, backend, laptop.model_copy(update={"devices": ("cpu",)})) is None
    gap = par.placement_gap("fake.echo", None, backend, laptop.model_copy(update={"devices": ("cuda",)}))
    assert gap is not None and "'fake'" in gap
    out = par.ParityOutputs(
        flatten="flatten@2", task="fake.echo", backend="fake", stack=backend.stack, device="cpu", model=None,
        sha="a" * 40, job="1", not_placeable=gap, items=(),
    )  # fmt: skip
    with pytest.raises(par.ParityError, match="was not run on cpu"):
        par.decide(out, None, THRESHOLDS)


@pytest.mark.logic
def test_a_backend_runs_its_own_model_not_the_benchmarks_default() -> None:
    """`crisperwhisper` serves its own model, so a benchmark pinned to another model does not apply to it."""
    importlib.import_module("senselab")
    catalogue = importlib.import_module("senselab_core.runtime.call").packaged().catalogue
    _, crisper = par.resolve("speech_to_text", "crisperwhisper")
    _, other = par.resolve("speech_to_text", "transformers")
    own = catalogue.backend_entry(crisper).ref
    turbo = f"openai/whisper-large-v3-turbo@{'a' * 40}"
    assert own.id != "openai/whisper-large-v3-turbo"
    assert par.model_of(turbo, crisper, catalogue) == own
    assert par.model_of(None, crisper, catalogue) == own
    assert str(par.model_of(turbo, other, catalogue)) == turbo


@pytest.mark.logic
def test_cross_stack_pairs_are_one_task_and_model_whatever_the_backend() -> None:
    """PII on transformers and on vllm pair; another model, another device or the same stack do not."""

    def out(backend: str, stack: str, model: str | None, device: str = "cuda") -> Any:  # noqa: ANN401
        return par.ParityOutputs(
            flatten="flatten@2", task="pii_spans", backend=backend, stack=stack, device=device, model=model,
            sha="a" * 40, job="1", items=(),
        )  # fmt: skip

    a, b = out("transformers", "transformers", "m"), out("vllm", "vllm", "m")
    assert par.same_model_pairs([a, b]) == [(a, b)]
    assert par.same_model_pairs([a, out("t2", "transformers", "m")]) == []
    assert par.same_model_pairs([a, out("x", "vllm", "n")]) == []
    assert par.same_model_pairs([a, out("y", "y", "m", "cpu")]) == []


@pytest.mark.logic
def test_convert_writes_the_reference_where_the_parity_test_reads_it(tmp_path: Path) -> None:
    """`convert` names the file from task and backend under `--parity-dir`; the test reads the same path."""
    ids = _ids("heldout", 1)
    records = tmp_path / "rec"
    records.mkdir()
    _records(records, "speaker_embeddings", ids, {"vector": [1.0]})
    argv = ["convert", str(records), "--benchmark", "librispeech-speaker-trials", "--task", "speaker_embeddings"]
    argv += ["--backend", "speechbrain", "--alpha-sha", "f" * 40, "--job", "1", "--parity-dir", str(tmp_path / "p")]
    par.main(argv)
    assert par.load_reference(par.reference_path(tmp_path / "p", "speaker_embeddings", "speechbrain")).items


@pytest.mark.logic
def test_every_parity_call_validates_with_a_pinned_model() -> None:
    """The `TaskCall` for each backend of each smoke benchmark validates; a bare model id does not."""
    importlib.import_module("senselab")
    call = importlib.import_module("senselab_core.runtime.call")
    catalogue = call.packaged().catalogue
    suite = par.harness().Suite.load(ROOT / "benchmarks" / "suites" / "smoke.yaml")
    built = 0
    for bench in suite.benchmarks:
        if bench.task.startswith("signal."):
            continue
        for backend in call.packaged().registry.backends(bench.task):
            ref = par.model_of(bench.model, backend, catalogue)
            task_call = call.TaskCall.model_validate({"task": bench.task, "inputs": {}, "params": {}, "model": ref})
            assert task_call.model == ref
            built += 1
            if ref is not None:
                with pytest.raises(ValueError, match="explicit revision"):
                    call.TaskCall.parse_model(ref.id)
    assert built >= 4


@pytest.mark.logic
def test_run_takes_its_backend_params_from_the_items_file_it_was_made_for(tmp_path: Path) -> None:
    """The items file is the one source of the backend params; one of another benchmark, backend or items is refused."""
    ids = _ids("heldout", 2)
    document = {
        "benchmark": "asr",
        "backend": "transformers",
        "items": dict.fromkeys(ids, {}),
        "backend_params": {"b": 2},
    }
    assert par.recorded_backend_params(document, "asr", "transformers") == {"b": 2}
    par.require_recorded_items(document, ids[::-1])
    for benchmark, backend, named in (
        ("other", "transformers", "benchmark 'asr'; this run is of 'other'"),
        ("asr", "crisperwhisper", "backend 'transformers'; this run is of 'crisperwhisper'"),
    ):
        with pytest.raises(par.ParityError, match=named):
            par.recorded_backend_params(document, benchmark, backend)
    with pytest.raises(par.ParityError, match="this run holds"):
        par.require_recorded_items(document, ids[:1])
    items = tmp_path / "items.json"
    items.write_text(json.dumps({**document, "backend_params": {"not_a_crisperwhisper_param": 1}}), "utf-8")
    run = ["run", "--benchmark", "asr", "--backend", "crisperwhisper", "--data-cache", str(tmp_path), "--work"]
    run += [str(tmp_path), "--out", str(tmp_path / "o.json"), "--sha", "a" * 40, "--job", "1", "--items", str(items)]
    run += ["--execution", "laptop"]
    with pytest.raises(par.ParityError, match="backend 'transformers'; this run is of 'crisperwhisper'"):
        par.main(run)


@pytest.mark.logic
def test_heldout_follows_the_adapters_split_unit() -> None:
    """Utterances of a held-out speaker pass `require_heldout` though their own ids hash to either split."""
    unit_of = par.harness().LibriSpeechTrials().unit_of
    held = next(s for s in (f"{n}" for n in range(200)) if par.split_of(s) == "heldout")
    cal = next(s for s in (f"{n}" for n in range(200)) if par.split_of(s) == "calibration")
    ids = [f"{held}-1-{n:04d}" for n in range(8)]
    assert len({par.split_of(i) for i in ids}) == 2
    par.require_heldout(ids, unit_of)
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.require_heldout([*ids, f"{cal}-1-0000"], unit_of)
    with pytest.raises(par.ParityError, match="held-out items only"):
        par.require_heldout(ids, _OWN)


@pytest.mark.logic
def test_cross_stack_speaker_embeddings_of_one_heldout_speaker_agree() -> None:
    """Held-out utterances of one speaker compare across stacks though their ids hash to either split."""
    held = next(str(n) for n in range(200) if par.split_of(str(n)) == "heldout")
    ids = [f"{held}-1-{n:04d}" for n in range(8)]
    assert len({par.split_of(i) for i in ids}) == 2
    fields = {"embedding.format": '"embedding@1.0"', "embedding.vector": [1.0, 2.0]}

    def stack(name: str) -> Any:  # noqa: ANN401
        items = tuple(par.OutputItem(id=i, split="heldout", batches={"1": fields}) for i in ids)
        return par.ParityOutputs(
            flatten=par.FLATTEN_VERSION,
            task="speaker_embeddings",
            backend=name,
            stack=name,
            device="cpu",
            model="m",
            sha="a" * 40,
            job="1",
            items=items,
        )

    assert par.compare_stacks(stack("a"), stack("b"), THRESHOLDS) == {}
