"""Held-out parity of 2.x with 1.x: export the items, convert 1.x outputs, run 2.x, and compare them.

Subcommands, in the order a campaign runs them:
`items` (2.x env, no model) writes the held-out items, task and backend params and model pin for
`parity_reference_1x.py`;
`convert` turns that script's output files into `parity-reference@1.0`;
`run` runs a 2.x backend over the same items, under the backend params of that `items` file, and writes
`parity-outputs@3.0`; a backend the profile cannot place on this node is recorded as `not_placeable` with no outputs,
never run;
`compare` writes `parity-comparison@1.0`: a run's outputs against the 1.x reference, or alone against one request,
by each kernel type's comparison (`runtime.compare`) under the thresholds of `--config`, a run config merged over the
packaged one (`Thresholds.load`), whose digest the record carries. `packages/senselab/tests/parity` asserts the same.
Calibration items are refused everywhere.
"""

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy
from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import (
    PARITY_COMPARISON_FORMAT,
    PARITY_OUTPUTS_FORMAT,
    PARITY_REFERENCE_FORMAT,
    FormatTag,
    Formatted,
)
from senselab_core.kernel.results import Embedding, Intervals, Transcript
from senselab_core.runtime.compare import (
    FLATTEN_VERSION,
    Fields,
    Thresholds,
    Value,
    compare_fields,
    flatten,
    index_free,
    stale_flatten,
)

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = "heldout"
SINGLE, MANY = "1", "N"
NS = 1e9


class ParityError(ValueError):
    """A held-out rule broken, or a 1.x output file that is not one the converter reads."""


class ReferenceItem(Frozen):
    """A held-out item and the 1.x output fields, keyed as the 2.x output flattens."""

    id: NonEmptyStr
    split: NonEmptyStr
    fields: Fields


class ParityReference(Formatted):
    """The 1.x outputs of one backend's held-out items, from the `alpha` commit `alpha_sha`."""

    format: FormatTag = PARITY_REFERENCE_FORMAT
    flatten: NonEmptyStr
    task: NonEmptyStr
    backend: NonEmptyStr
    alpha_sha: NonEmptyStr
    job: NonEmptyStr
    items: tuple[ReferenceItem, ...]


class OutputItem(Frozen):
    """A held-out item and the 2.x output fields per batch shape (`1`: alone, `N`: every item in one request)."""

    id: NonEmptyStr
    split: NonEmptyStr
    batches: dict[str, Fields]


class ParityOutputs(Formatted):
    """The 2.x outputs of one backend's held-out items on one stack and device, from `model` (None: model-free).

    `not_placeable` is why the profile could not place the backend on this node, when it could not: no items were
    run and the record says nothing about parity on `device`.
    """

    format: FormatTag = PARITY_OUTPUTS_FORMAT
    flatten: NonEmptyStr
    task: NonEmptyStr
    backend: NonEmptyStr
    stack: NonEmptyStr
    device: NonEmptyStr
    model: NonEmptyStr | None
    sha: NonEmptyStr
    job: NonEmptyStr
    claims_batch_invariance: bool = False
    not_placeable: str | None = None
    items: tuple[OutputItem, ...]


class ParityComparison(Formatted):
    """One parity decision: a run's differences per item `against` the 1.x `reference` or its alone `batch`.

    `thresholds` is the canonical digest of the thresholds the decision was made under.
    """

    format: FormatTag = PARITY_COMPARISON_FORMAT
    task: NonEmptyStr
    backend: NonEmptyStr
    stack: NonEmptyStr
    device: NonEmptyStr
    against: NonEmptyStr
    thresholds: NonEmptyStr
    mismatches: dict[str, list[str]]

    @property
    def agree(self) -> bool:
        """Whether no item differs."""
        return not self.mismatches


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harness() -> ModuleType:
    """Return `benchmarks/harness.py`, the owner of the suites, the adapters and the item split."""
    return sys.modules.get("benchmark_harness") or _load("benchmark_harness", ROOT / "benchmarks" / "harness.py")


def split_of(item_id: str) -> str:
    """Return `calibration` or `heldout` as the harness splits ids."""
    return str(harness().split_of(item_id))


def resolve(task: str, backend_name: str) -> tuple[Any, Any]:
    """Return (task spec, backend spec) of the packaged declarations."""
    from senselab_core.runtime.call import packaged  # noqa: PLC0415 - needs the runtime

    backend = next(b for b in packaged().registry.backends(task) if b.name == backend_name)
    return packaged().registry.spec(task), backend


def load_items(suite: Path, benchmark: str, data_cache: Path, work: Path, split: str) -> tuple[Any, list[Any]]:
    """Return (benchmark, its adapter items of `split`), the data fetched into `data_cache`, files under `work`."""
    from senselab_core.runtime.fetch import DataManifest, fetch  # noqa: PLC0415 - only a real run needs the runtime

    [bench] = [b for b in harness().Suite.load(suite).benchmarks if b.name == benchmark]
    files = (
        {}
        if bench.manifest is None
        else fetch(
            DataManifest.load(ROOT / "benchmarks" / "manifests" / f"{bench.manifest}.yaml"), cache_root=data_cache
        )
    )
    items = harness().ADAPTERS[bench.adapter]().items(files, bench.items, work, bench.max_seconds)
    return bench, [i for i in items if split_of(i.id) == split]


def _wired(port: str, port_type: str, value: object) -> object:
    """Return the wire form of an item input as the call path binds it."""
    from senselab_core.kernel.wire import wire_form  # noqa: PLC0415 - only a real run needs the runtime
    from senselab_core.runtime.audio_io import READER  # noqa: PLC0415
    from senselab_core.runtime.call import bind_input  # noqa: PLC0415

    bound = bind_input(port, port_type, value)[1]
    return wire_form(bound, READER) if isinstance(bound, Frozen) else bound


def prepare(  # noqa: PLR0913 - what picks one benchmark's items and params
    suite: Path,
    benchmark: str,
    backend_name: str,
    backend_params: Mapping[str, Any],
    data_cache: Path,
    work: Path,
    split: str,
) -> tuple[Any, Any, dict[str, dict[str, object]], dict[str, Any]]:
    """Return (benchmark, backend spec, wired inputs of the `split` items by id, call params) for one benchmark."""
    bench, items = load_items(suite, benchmark, data_cache, work, split)
    spec, backend = resolve(bench.task, backend_name)
    wired = {i.id: {p: _wired(p, spec.inputs[p], v) for p, v in i.inputs.items()} for i in items}
    params = {
        **spec.params.model_validate(bench.params).model_dump(mode="json"),
        "backend_params": backend.params.model_validate(backend_params).model_dump(mode="json"),
    }
    return bench, backend, wired, params


def require_heldout(ids: Sequence[str]) -> None:
    """Raise ParityError if any of `ids` is not a held-out item."""
    other = sorted(i for i in ids if split_of(i) != HELDOUT)
    if other:
        raise ParityError(f"parity runs on held-out items only; not held out: {other}")


def _tag(kind: type[Formatted]) -> str:
    return str(kind.model_fields["format"].default)


def _turns(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns = sorted((round(float(ln["start"]) * NS), round(float(ln["end"]) * NS), str(ln["speaker"])) for ln in lines)
    return {
        "turns": {
            "format": _tag(Intervals),
            "start_ns": [s for s, _, _ in turns],
            "end_ns": [max(s, e) for s, e, _ in turns],
            "label": [label for _, _, label in turns],
        }
    }


def _transcript(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = " ".join(str(ln["text"]).strip() for ln in lines).strip()
    if not text:
        return {"transcript": {"format": _tag(Transcript), "segments": []}}
    segment: dict[str, Any] = {"text": text}
    logprob = lines[0].get("avg_logprob") if len(lines) == 1 else None
    if logprob is not None:
        segment["confidence"] = min(1.0, math.exp(float(logprob)))
    return {"transcript": {"format": _tag(Transcript), "segments": [segment]}}


def _embedding(vector: Sequence[Any]) -> dict[str, Any]:
    return {"embedding": {"format": _tag(Embedding), "vector": numpy.asarray(vector, dtype=float).ravel().tolist()}}


def project(record: Mapping[str, Any]) -> Fields:
    """Return the 2.x-keyed output fields of one 1.x output record (`{"task": ..., "lines" | "vector": ...}`)."""
    task = record.get("task")
    if task == "speech_to_text" and "lines" in record:
        shaped = _transcript(record["lines"])
    elif task == "diarize" and "lines" in record:
        shaped = _turns(record["lines"])
    elif task == "speaker_embeddings" and "vector" in record:
        shaped = _embedding(record["vector"])
    else:
        raise ParityError(
            f"a 1.x output record of task {task!r} with keys {sorted(record)} is not one the converter reads"
        )
    return flatten(shaped)


def convert(directory: Path, *, task: str, backend: str, alpha_sha: str, job: str) -> ParityReference:
    """Read the `<id>.json` files `parity_reference_1x.py` wrote under `directory`.

    Raises:
        ParityError: a file holds a calibration item or a record of another task.
    """
    records = {p.stem: json.loads(p.read_text("utf-8")) for p in sorted(directory.glob("*.json"))}
    require_heldout(list(records))
    wrong = sorted(i for i, r in records.items() if r.get("task") != task)
    if wrong:
        raise ParityError(f"records of another task than {task!r}: {wrong}")
    items = tuple(ReferenceItem(id=i, split=HELDOUT, fields=project(r)) for i, r in records.items())
    return ParityReference(
        flatten=FLATTEN_VERSION, task=task, backend=backend, alpha_sha=alpha_sha, job=job, items=items
    )


def load_reference(path: Path) -> ParityReference:
    """Read a 1.x reference file of the current format and flatten.

    Raises:
        ParityError: it is of another format or flatten; it names the file.
    """
    return ParityReference.model_validate(_current(path, PARITY_REFERENCE_FORMAT))


def load_outputs(path: Path) -> ParityOutputs:
    """Read a 2.x outputs file of the current format and flatten.

    Raises:
        ParityError: it is of another format or flatten; it names the file.
    """
    return ParityOutputs.model_validate(_current(path, PARITY_OUTPUTS_FORMAT))


def _current(path: Path, fmt: str) -> dict[str, Any]:
    raw: dict[str, Any] = json.loads(path.read_text("utf-8"))
    if (stale := stale_flatten(raw, fmt)) is not None:
        raise ParityError(f"{path}: {stale}; regenerate it")
    return raw


def compare(expected: Mapping[str, Value], actual: Mapping[str, Value], thresholds: Thresholds) -> list[str]:
    """Return how `actual` differs from `expected` (`compare_fields`), as `<path>: <detail> (<rule>)` lines.

    Only the leaves of `actual` whose index-free path `expected` holds are compared: a field 1.x never wrote is not
    judged, and an extra segment or turn still is.
    """
    held = {index_free(p) for p in expected}
    found = compare_fields(expected, {p: v for p, v in actual.items() if index_free(p) in held}, thresholds)
    return [f"{d.path}: {d.detail} ({d.rule}{f' {d.threshold}' if d.threshold else ''})" for d in found.differences]


def compare_items(
    expected: Mapping[str, Fields],
    actual: Mapping[str, Fields],
    thresholds: Thresholds,
) -> dict[str, list[str]]:
    """Return per-item differences over held-out items; an item `actual` lacks is a difference.

    Raises:
        ParityError: an item is not a held-out item.
    """
    require_heldout(list(expected))
    return {
        i: found
        for i in expected
        if (found := compare(expected[i], actual[i], thresholds) if i in actual else ["item has no output"])
    }


def decide(outputs: ParityOutputs, reference: ParityReference | None, thresholds: Thresholds) -> ParityComparison:
    """Return the parity decision on `outputs`: alone against `reference`, or with none, one request against alone.

    Raises:
        ParityError: `outputs` is a backend that was not placeable, or an item is not a held-out item.
    """
    if outputs.not_placeable is not None:
        raise ParityError(f"{outputs.backend} was not run on {outputs.device}: {outputs.not_placeable}")
    alone = {i.id: i.batches["1"] for i in outputs.items if "1" in i.batches}
    if reference is not None:
        against, found = "reference", compare_items({i.id: i.fields for i in reference.items}, alone, thresholds)
    else:
        together = {i.id: i.batches["N"] for i in outputs.items if "N" in i.batches}
        against, found = "batch", compare_items(alone, together, thresholds)
    return ParityComparison(
        task=outputs.task,
        backend=outputs.backend,
        stack=outputs.stack,
        device=outputs.device,
        against=against,
        thresholds=thresholds.canonical_digest(),
        mismatches=found,
    )


def placement_gap(task: str, model: Any, backend: Any, profile: Any) -> str | None:  # noqa: ANN401
    """Return why `profile` cannot place `backend` of `task` on `model` on this node, or None when it can.

    The same check the harness reports as "not placeable" (`placement_of`), with `profile`'s stack preference for
    `task` narrowed to the backend's stack; a placement on a different backend is a gap too.
    """
    from senselab_core.runtime.call import TaskCall, placement_of  # noqa: PLC0415 - needs the runtime
    from senselab_core.runtime.profiles import NoBackend, NoDevice  # noqa: PLC0415

    narrowed = profile.model_copy(update={"stack_preferences": {**profile.stack_preferences, task: (backend.stack,)}})
    try:
        placed = placement_of(task, TaskCall.parse_model(model), narrowed)
    except (NoDevice, NoBackend) as err:
        return " ".join(str(err).split())
    return None if placed.backend.name == backend.name else f"the profile places {task!r} on {placed.backend.name!r}"


def model_of(bench_model: str | None, backend: Any, catalogue: Any) -> Any:  # noqa: ANN401
    """Return the pinned `ModelRef` `backend` runs: the benchmark's when the backend serves it, else its own entry's."""
    from senselab_core.runtime.call import TaskCall  # noqa: PLC0415 - needs the runtime

    ref = TaskCall.parse_model(bench_model)
    if ref is not None and (not backend.models or ref.id in backend.models):
        return ref
    entry = catalogue.backend_entry(backend)
    return None if entry is None else entry.ref


def reference_path(parity_dir: Path, task: str, backend: str) -> Path:
    """Return where the 1.x reference of `backend` for `task` lives under a campaign directory."""
    return parity_dir / "reference" / task / f"{backend}.json"


def same_model_pairs(outputs: Sequence[ParityOutputs]) -> list[tuple[ParityOutputs, ParityOutputs]]:
    """Return the pairs of runs of one task and model on different stacks and one device, whatever the backends."""
    return [
        (a, b)
        for a, b in combinations(outputs, 2)
        if (a.task, a.model, a.device) == (b.task, b.model, b.device) and a.stack != b.stack
    ]


def run(  # noqa: PLR0913 - what one 2.x parity run fixes
    worker: Any,  # noqa: ANN401
    items: Mapping[str, Mapping[str, Any]],
    params: dict[str, Any],
    *,
    task: str,
    backend: str,
    stack: str,
    device: str,
    model: str | None,
    sha: str,
    job: str,
    claims_batch_invariance: bool = False,
) -> ParityOutputs:
    """Run each held-out item alone and all in one request; an item the worker fails on has no output.

    Raises:
        ParityError: `items` holds a calibration item.
    """
    require_heldout(list(items))
    seen: dict[str, dict[str, Fields]] = {i: {} for i in items}
    for batch, groups in ((SINGLE, [[i] for i in items]), (MANY, [list(items)])):
        for group in groups:
            for result in worker.run([{"id": i, "inputs": items[i]} for i in group], params):
                if "outputs" in result:
                    seen[result["id"]][batch] = flatten(result["outputs"])
    return ParityOutputs(
        flatten=FLATTEN_VERSION,
        task=task,
        backend=backend,
        stack=stack,
        device=device,
        model=model,
        sha=sha,
        job=job,
        claims_batch_invariance=claims_batch_invariance,
        items=tuple(OutputItem(id=i, split=HELDOUT, batches=b) for i, b in seen.items() if b),
    )


def recorded_backend_params(document: Mapping[str, Any], benchmark: str, backend: str, ids: Sequence[str]) -> Any:  # noqa: ANN401
    """Return the backend params an `items` document records, for a run of `backend` on `benchmark` over `ids`.

    Raises:
        ParityError: the document is of another benchmark or backend, or of other items.
    """
    for key, value in (("benchmark", benchmark), ("backend", backend)):
        if document.get(key) != value:
            raise ParityError(f"the items file is of {key} {document.get(key)!r}; this run is of {value!r}")
    if sorted(document["items"]) != sorted(ids):
        raise ParityError(f"the items file holds {sorted(document['items'])}; this run holds {sorted(ids)}")
    return document["backend_params"]


def _items_cli(args: argparse.Namespace) -> None:
    """Write the held-out items (inputs as strings), the task and backend params, and the pin of `args.backend`."""
    from senselab_core.runtime.call import packaged  # noqa: PLC0415 - only a real export needs the runtime

    bench, items = load_items(args.suite, args.benchmark, args.data_cache, args.work, HELDOUT)
    spec, backend = resolve(bench.task, args.backend)
    entry = packaged().catalogue.backend_entry(backend)
    document = {
        "benchmark": bench.name,
        "backend": args.backend,
        "task": bench.task,
        "params": spec.params.model_validate(bench.params).model_dump(mode="json"),
        "backend_params": backend.params.model_validate(args.backend_params).model_dump(mode="json"),
        "model": None if entry is None else {"id": entry.id, "revision": entry.revision},
        "items": {i.id: {k: str(v) for k, v in i.inputs.items()} for i in items},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True), "utf-8")


def _convert_cli(args: argparse.Namespace) -> None:
    reference = convert(args.records, task=args.task, backend=args.backend, alpha_sha=args.alpha_sha, job=args.job)
    out = reference_path(args.parity_dir, args.task, args.backend)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(reference.model_dump_json(indent=1), "utf-8")


def _run_cli(args: argparse.Namespace) -> None:
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - only a real run needs the runtime
    from senselab_core.testing.workers import RealWorkers  # noqa: PLC0415

    document = json.loads(args.items.read_text("utf-8"))
    bench, backend, wired, params = prepare(
        args.suite, args.benchmark, args.backend, document["backend_params"], args.data_cache, args.work, HELDOUT
    )
    tuning = recorded_backend_params(document, args.benchmark, args.backend, list(wired))
    profile = ExecutionProfile.load(args.execution)
    workers = RealWorkers(profile)
    stack = backend.stack + "".join(f" {k}={v}" for k, v in sorted(tuning.items()))
    from senselab_core.runtime.call import packaged  # noqa: PLC0415

    ref = model_of(bench.model, backend, packaged().catalogue)
    model = None if ref is None else str(ref)
    gap = placement_gap(bench.task, ref, backend, profile)
    try:
        if gap is not None:
            outputs = ParityOutputs(
                flatten=FLATTEN_VERSION,
                task=bench.task,
                backend=args.backend,
                stack=stack,
                device=workers.device,
                model=model,
                sha=args.sha,
                job=args.job,
                not_placeable=gap,
                items=(),
            )
        else:
            outputs = run(
                workers(bench.task, args.backend),
                wired,
                params,
                task=bench.task,
                backend=args.backend,
                stack=stack,
                device=workers.device,
                model=model,
                sha=args.sha,
                job=args.job,
                claims_batch_invariance=args.claim_batch_invariance,
            )
    finally:
        workers.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(outputs.model_dump_json(indent=1), "utf-8")


def _compare_cli(args: argparse.Namespace) -> None:
    reference = None if args.reference is None else load_reference(args.reference)
    decision = decide(load_outputs(args.outputs), reference, Thresholds.load(args.config))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(decision.model_dump_json(indent=1), "utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: `items`, `convert`, `run`, `compare`."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--suite", type=Path, default=ROOT / "benchmarks" / "suites" / "smoke.yaml")
        p.add_argument("--benchmark", required=True)
        p.add_argument("--backend", required=True)
        p.add_argument("--data-cache", type=Path, required=True)
        p.add_argument("--work", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)

    i = sub.add_parser("items")
    common(i)
    i.add_argument("--backend-params", type=json.loads, default={})
    i.set_defaults(func=_items_cli)
    c = sub.add_parser("convert")
    c.add_argument("records", type=Path)
    c.add_argument("--task", required=True)
    c.add_argument("--backend", required=True)
    c.add_argument("--alpha-sha", required=True)
    c.add_argument("--job", required=True)
    c.add_argument(
        "--parity-dir", type=Path, required=True, help="the campaign directory; writes reference/<task>/<backend>.json"
    )
    c.set_defaults(func=_convert_cli)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--execution", default="orcd")
    r.add_argument("--items", type=Path, required=True, help="the `items` file; its backend params are the run's")
    r.add_argument("--claim-batch-invariance", action="store_true")
    r.add_argument("--sha", required=True)
    r.add_argument("--job", required=True)
    r.set_defaults(func=_run_cli)
    m = sub.add_parser("compare")
    m.add_argument("outputs", type=Path)
    m.add_argument("--reference", type=Path)
    m.add_argument("--config", type=Path, help="a run config YAML whose thresholds are merged over the packaged ones")
    m.add_argument("--out", type=Path, required=True)
    m.set_defaults(func=_compare_cli)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
