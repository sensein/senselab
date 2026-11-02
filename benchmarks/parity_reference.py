"""Held-out parity of 2.x with 1.x: export the items, convert 1.x outputs, run 2.x, and compare within a profile.

Subcommands, in the order a campaign runs them:
`items` (2.x env, no model) writes the held-out items and model pin for `parity_reference_1x.py`;
`convert` turns that script's output files into `parity-reference@1.0`;
`run` runs a 2.x backend over the same items and writes `parity-outputs@1.0`.
`compare` is what `packages/senselab/tests/parity` asserts. Calibration items are refused everywhere and a bound is
never written here: the bound is the backend's `tolerance@1.0` profile.
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy
import yaml
from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import PARITY_OUTPUTS_FORMAT, PARITY_REFERENCE_FORMAT, FormatTag, Formatted

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = "heldout"
SINGLE, MANY = "1", "N"
NS = 1e9
Fields = dict[str, list[float] | str]


class Bounded(Protocol):
    """A tolerance profile as the comparison reads it: `derivation` and `fields` (each a `path` and a `tolerance`)."""

    derivation: Any
    fields: Sequence[Any]


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
    """The 2.x outputs of one backend's held-out items on one stack and device."""

    format: FormatTag = PARITY_OUTPUTS_FORMAT
    task: NonEmptyStr
    backend: NonEmptyStr
    stack: NonEmptyStr
    device: NonEmptyStr
    sha: NonEmptyStr
    job: NonEmptyStr
    claims_batch_invariance: bool = False
    items: tuple[OutputItem, ...]


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tolerance_module() -> ModuleType:
    """Return `scripts/generate_tolerance.py`, the owner of `flatten`, the profile model and item preparation."""
    return sys.modules.get("generate_tolerance") or _load(
        "generate_tolerance", ROOT / "scripts" / "generate_tolerance.py"
    )


def split_of(item_id: str) -> str:
    """Return `calibration` or `heldout` as the harness splits ids."""
    return str(tolerance_module()._harness().split_of(item_id))


def require_heldout(ids: Sequence[str]) -> None:
    """Raise ParityError if any of `ids` is not a held-out item."""
    other = sorted(i for i in ids if split_of(i) != HELDOUT)
    if other:
        raise ParityError(f"parity runs on held-out items only; not held out: {other}")


def _turns(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    turns = sorted((round(float(ln["start"]) * NS), round(float(ln["end"]) * NS), str(ln["speaker"])) for ln in lines)
    return {
        "turns": {
            "start_ns": [s for s, _, _ in turns],
            "end_ns": [max(s, e) for s, e, _ in turns],
            "label": [label for _, _, label in turns],
        }
    }


def _transcript(lines: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    text = " ".join(str(ln["text"]).strip() for ln in lines).strip()
    if not text:
        return {"transcript": {"segments": []}}
    segment: dict[str, Any] = {"text": text}
    logprob = lines[0].get("avg_logprob") if len(lines) == 1 else None
    if logprob is not None:
        segment["confidence"] = min(1.0, math.exp(float(logprob)))
    return {"transcript": {"segments": [segment]}}


def _embedding(vector: Sequence[Any]) -> dict[str, Any]:
    return {"embedding": {"vector": numpy.asarray(vector, dtype=float).ravel().tolist()}}


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
    fields: Fields = tolerance_module().flatten(shaped)
    return fields


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
    return ParityReference(task=task, backend=backend, alpha_sha=alpha_sha, job=job, items=items)


def load_profile(path: Path) -> Bounded:
    """Read a `tolerance@1.0` file, refusing one not derived from calibration items alone."""
    tol = tolerance_module()
    profile: Bounded = tol.ToleranceProfile.model_validate(yaml.safe_load(path.read_text("utf-8")))
    held = sorted(i for i in profile.derivation.items if split_of(i) != tol.CALIBRATION)
    if profile.derivation.split != tol.CALIBRATION or held:
        raise ParityError(f"{path.name} is not a calibration-only profile; held-out items: {held}")
    return profile


_LABEL = re.compile(r"turns\.label\[(\d+)\]")


def relabel_turns(expected: Mapping[str, list[float] | str], actual: Mapping[str, list[float] | str]) -> dict[str, Any]:
    """Return `actual` with its speaker labels renamed to `expected`'s by the one-to-one map of most shared time.

    The map is the one DER scores under. A label with no counterpart is renamed to one no reference label can equal.
    `actual` is unchanged where either side has no turns.
    """

    def keys(fields: Mapping[str, Any]) -> list[str]:
        found = [(int(m.group(1)), k) for k in fields if (m := _LABEL.fullmatch(k))]
        return [k for _, k in sorted(found)]

    ek, ak = keys(expected), keys(actual)
    times = ("turns.start_ns", "turns.end_ns")
    if not ek or not ak or any(t not in expected or t not in actual for t in times):
        return dict(actual)
    elabel = [json.loads(str(expected[k])) for k in ek]
    alabel = [json.loads(str(actual[k])) for k in ak]
    rows, cols = sorted(set(elabel)), sorted(set(alabel))
    overlap = numpy.zeros((len(rows), len(cols)))
    spans = [[list(f[t]) for t in times] for f in (expected, actual)]
    for i, (es, ee) in enumerate(zip(*spans[0], strict=True)):
        for j, (a_s, a_e) in enumerate(zip(*spans[1], strict=True)):
            overlap[rows.index(elabel[i]), cols.index(alabel[j])] += max(0.0, min(ee, a_e) - max(es, a_s))
    linear_sum_assignment = importlib.import_module("scipy.optimize").linear_sum_assignment
    mapped = {
        cols[c]: rows[r]
        for r, c in zip(*linear_sum_assignment(overlap, maximize=True), strict=True)
        if overlap[r, c] > 0
    }
    out = dict(actual)
    for k, label in zip(ak, alabel, strict=True):
        out[k] = json.dumps(mapped.get(label, f"<unmatched:{label}>"))
    return out


def compare(
    expected: Mapping[str, list[float] | str], actual: Mapping[str, list[float] | str], profile: Bounded
) -> list[str]:
    """Return what differs, `expected` against `actual` over the paths of `expected`, each judged by `profile`.

    A path is a mismatch when `actual` lacks it, its shape differs, the profile has no tolerance for it, or the
    difference exceeds the profile's tolerance (an exact field must be equal).
    """
    bound = {f.path: f.tolerance for f in profile.fields}
    actual = relabel_turns(expected, actual)
    out = []
    for path in sorted(expected):
        a, b = expected[path], actual.get(path)
        if b is None:
            out.append(f"{path}: missing from the compared output")
        elif path not in bound:
            out.append(f"{path}: the profile has no tolerance for it")
        elif isinstance(a, str) or isinstance(b, str):
            if a != b:
                out.append(f"{path}: {a!r} != {b!r}")
        elif len(a) != len(b):
            out.append(f"{path}: {len(a)} values against {len(b)}")
        elif a and (gap := float(numpy.max(numpy.abs(numpy.asarray(a) - numpy.asarray(b))))) > bound[path]:
            out.append(f"{path}: difference {gap:g} exceeds tolerance {bound[path]:g}")
    return out


def compare_items(
    expected: Mapping[str, Fields],
    actual: Mapping[str, Fields],
    profile: Bounded,  # noqa: ANN401
) -> dict[str, list[str]]:
    """Return per-item mismatches over held-out items; an item `actual` lacks is a mismatch.

    Raises:
        ParityError: an item is not a held-out item.
    """
    require_heldout(list(expected))
    return {
        i: found
        for i in expected
        if (found := compare(expected[i], actual[i], profile) if i in actual else ["item has no output"])
    }


def run(  # noqa: PLR0913 - what one 2.x parity run fixes
    worker: Any,  # noqa: ANN401
    items: Mapping[str, Mapping[str, Any]],
    params: dict[str, Any],
    *,
    task: str,
    backend: str,
    stack: str,
    device: str,
    sha: str,
    job: str,
    claims_batch_invariance: bool = False,
) -> ParityOutputs:
    """Run each held-out item alone and all in one request; an item the worker fails on has no output.

    Raises:
        ParityError: `items` holds a calibration item.
    """
    require_heldout(list(items))
    flatten = tolerance_module().flatten
    seen: dict[str, dict[str, Fields]] = {i: {} for i in items}
    for batch, groups in ((SINGLE, [[i] for i in items]), (MANY, [list(items)])):
        for group in groups:
            for result in worker.run([{"id": i, "inputs": items[i]} for i in group], params):
                if "outputs" in result:
                    seen[result["id"]][batch] = flatten(result["outputs"])
    return ParityOutputs(
        task=task,
        backend=backend,
        stack=stack,
        device=device,
        sha=sha,
        job=job,
        claims_batch_invariance=claims_batch_invariance,
        items=tuple(OutputItem(id=i, split=HELDOUT, batches=b) for i, b in seen.items() if b),
    )


def _items_cli(args: argparse.Namespace) -> None:
    """Write the held-out items (path or text inputs, as strings) and the model pin of `args.backend`."""
    from senselab_core.runtime.call import packaged  # noqa: PLC0415 - only a real export needs the runtime

    tol = tolerance_module()
    bench, items = tol.load_items(args.suite, args.benchmark, args.data_cache, args.work, HELDOUT)
    _, backend = tol.resolve(bench.task, args.backend)
    entry = packaged().catalogue.entry(backend.models[0]) if backend.models else None
    document = {
        "task": bench.task,
        "model": None if entry is None else {"id": entry.id, "revision": entry.revision},
        "items": {i.id: {k: str(v) for k, v in i.inputs.items()} for i in items},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=1, sort_keys=True), "utf-8")


def _convert_cli(args: argparse.Namespace) -> None:
    reference = convert(args.records, task=args.task, backend=args.backend, alpha_sha=args.alpha_sha, job=args.job)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(reference.model_dump_json(indent=1), "utf-8")


def _run_cli(args: argparse.Namespace) -> None:
    from senselab_core.runtime.profiles import ExecutionProfile  # noqa: PLC0415 - only a real run needs the runtime
    from senselab_core.testing.workers import RealWorkers  # noqa: PLC0415

    bench, backend, wired, params = tolerance_module().prepare(
        args.suite, args.benchmark, args.backend, args.backend_params, args.data_cache, args.work, HELDOUT
    )
    workers = RealWorkers(ExecutionProfile.load(args.execution))
    try:
        outputs = run(
            workers(bench.task, args.backend),
            wired,
            params,
            task=bench.task,
            backend=args.backend,
            stack=backend.stack + "".join(f" {k}={v}" for k, v in sorted(args.backend_params.items())),
            device=workers.device,
            sha=args.sha,
            job=args.job,
            claims_batch_invariance=args.claim_batch_invariance,
        )
    finally:
        workers.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(outputs.model_dump_json(indent=1), "utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: `items`, `convert`, `run`."""
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
    i.set_defaults(func=_items_cli)
    c = sub.add_parser("convert")
    c.add_argument("records", type=Path)
    c.add_argument("--task", required=True)
    c.add_argument("--backend", required=True)
    c.add_argument("--alpha-sha", required=True)
    c.add_argument("--job", required=True)
    c.add_argument("--out", type=Path, required=True)
    c.set_defaults(func=_convert_cli)
    r = sub.add_parser("run")
    common(r)
    r.add_argument("--execution", default="orcd")
    r.add_argument("--backend-params", type=json.loads, default={})
    r.add_argument("--claim-batch-invariance", action="store_true")
    r.add_argument("--sha", required=True)
    r.add_argument("--job", required=True)
    r.set_defaults(func=_run_cli)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
