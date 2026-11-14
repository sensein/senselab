"""Markdown of the plan, run, executor, dataset-object and profile reference pages and the LibriSpeech example.

Every entry is read from the code: signatures and docstrings through `inspect`, profile keys from the packaged
profiles, the example's dataset terms from the benchmark manifest and its model from the packaged catalogue.
"""

import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from senselab_core.kernel.dataset import Dataset as DatasetProtocol
from senselab_core.kernel.dataset import InputManifest, Record
from senselab_core.kernel.plan import LogicalPlan, plan_hash
from senselab_core.runtime import planner
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.execution import run as run_module
from senselab_core.runtime.execution.events import RunStatus
from senselab_core.runtime.execution.executor import Executor
from senselab_core.runtime.execution.in_process import InProcess
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.substrates import Local, Slurm, Substrate, substrate_of
from senselab_core.runtime.fetch import DataManifest
from senselab_core.runtime.profiles import ExecutionProfile, packaged_profiles
from senselab_core.runtime.site_placement import PlacementAnswer, PlacementRequest

import senselab
from senselab._api import facade
from senselab._api.packaged import CATALOGUE
from senselab.reference import _api
from senselab.reference._render import BADGES, _require_root, exit_checks, table

PLAN_REFERENCE = "plans.html"
"""The page of the plan reference, which the other pages link to."""


def plans_page() -> str:
    """Return the plan reference: `plan`, `Plan`, `Plan.run`, `Plan.explain`, `resume` and the logical plan."""
    return f"""{BADGES["2x"]}

{_api.module_summary(planner)}

A plan has two parts. The logical plan is the graph with every resolution it runs under: pinned models, profile
slices, the input manifest, the input binding and the reuse policy. It holds nothing about where or how it runs, and
its hash is the run id, the same on every host. The placement is where each node runs under an execution profile on
this host: backend, stack, env, device, batch size and resource class. Two plans of one task over one dataset under
different profiles have one hash and differ in their placement alone.

## Planning

{_api.entry("senselab.plan", facade.plan)}
{_api.entry("senselab_core.runtime.planner.plan", planner.plan)}
## `Plan`

{_api.doc(planner.Plan)}

Fields: {_api.field_names(planner.Plan)}.

{_api.members(planner.Plan, ("run", "explain"))}
## Resuming

{_api.entry("senselab.resume", facade.resume)}
{_api.entry("senselab_core.runtime.planner.resume", planner.resume)}
## The logical plan

{_api.doc(LogicalPlan)}

Fields: {_api.field_names(LogicalPlan)}.

{_api.entry("plan_hash", plan_hash, level=3)}
The input manifest a plan holds:

{_api.doc(InputManifest)}

Fields: {_api.field_names(InputManifest)}.
"""


def runs_page() -> str:
    """Return the run reference: `run`, `Run`, `RunStatus` and `ResultSet`."""
    return f"""{BADGES["2x"]}

{_api.module_summary(run_module)}

A run is the execution of a plan, started by `Plan.run` or `senselab.run`. It runs in the background and writes
everything it does under its run directory, named by the plan hash: its plan, its input sources, its event log and its
placements. Any process can open that directory with `Run(run_dir)`, read the status, wait for the run and read its
results, whether or not it started the run.

Running a plan needs the `pydra` extra of `senselab-core`, which compiles the plan to a workflow; without it `Plan.run`
raises `MissingExtra`. Planning, `explain` and `describe` need no extra.

## Running

{_api.entry("senselab.run", facade.run)}
{_api.entry("senselab.use", facade.use)}
## `Run`

{_api.doc(run_module.Run)}

{_api.members(run_module.Run, ("wait", "results", "cancel", "status", "events", "manifest"))}
Properties:

{_api.properties(run_module.Run, ("plan", "cancelled"))}

## `RunStatus`

{_api.doc(RunStatus)}

Fields: {_api.field_names(RunStatus)}.

## `ResultSet`

{_api.doc(run_module.ResultSet)}

{_api.members(run_module.ResultSet, ("outputs", "to_frame"))}
Properties:

{_api.properties(run_module.ResultSet, ("failures", "shared"))}

## Resuming a run

`senselab.resume` starts a run again from its run directory by its plan hash; see the
[plan reference](plans.html#resuming).
"""


def executors_page() -> str:
    """Return the executor reference: `Executor`, `InProcess`, `Pool` and the substrates."""
    return f"""{BADGES["2x"]}

An executor runs the fragments of an admitted plan. `Plan.run` takes one, and the session default is set with
`senselab.use(executor=..., store=...)`. With none, a plan under a local substrate runs on `InProcess`, and a plan
under a `slurm` substrate on a `Pool` over `Slurm`.

| Executor | Runs jobs | Workers |
|---|---|---|
| `InProcess` | in this process, before `submit` returns | this process's own worker set |
| `Pool` | on queues in the run directory | pool workers that a substrate starts: `Local` processes or `Slurm` jobs |

## `Executor`

{_api.doc(Executor)}

{_api.members(Executor, ("admit", "release", "fragment_size", "submit", "status", "cancel", "poll", "bind_epoch"))}
{_api.entry("InProcess", InProcess, level=2)}
{_api.entry("Pool", Pool, level=2)}
## Substrates

{_api.doc(Substrate)}

{_api.members(Substrate, ("start", "live", "running", "stop"))}
{_api.entry("Local", Local, level=3)}
{_api.entry("Slurm", Slurm, level=3)}
{_api.entry("substrate_of", substrate_of, level=3)}
## Site placement

senselab holds no site scheduling policy. A group that runs on a scheduler writes its policy as a placement hook;
see [profile keys](profiles.html#site-profiles).

{_api.doc(PlacementRequest)}

{_api.doc(PlacementAnswer)}
"""


def _flatten(node: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            flat.update(_flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


_KEY = re.compile(r"^( *)([\w-]+):")


def _comments(text: str) -> dict[str, str]:
    """Return the `#` comment above each key of a profile file, by dotted key path."""
    found: dict[str, str] = {}
    stack: list[str] = []
    pending: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            note = stripped.lstrip("#").strip()
            if "specs/" not in note and not note.startswith("Derivations:"):
                pending.append(note)
            continue
        key = _KEY.match(line)
        if key:
            depth = len(key[1]) // 2
            del stack[depth:]
            stack.append(key[2])
            if pending:
                found.setdefault(".".join(stack), " ".join(pending))
        pending = []
    return found


def _shown(value: object) -> str:
    unfilled = re.fullmatch(r"<fill: (.+)>", value) if isinstance(value, str) else None
    if unfilled:
        return f"you set: {unfilled[1]}"
    if value is None:
        return "none"
    return f"`{value}`"


def profile_rows(name: str) -> dict[str, Any]:
    """Return the packaged profile `name` as a flat mapping of dotted key path to its value in the file."""
    return _flatten(load_yaml(_profile_file(name), f"packaged profile {name}"))


def _profile_file(name: str) -> Any:  # noqa: ANN401 - a package resource
    return resources.files("senselab_core.runtime").joinpath("data", "profiles", f"{name}.yaml")


def _section_rows(
    values: Mapping[str, Mapping[str, Any]], meaning: Mapping[str, str]
) -> dict[str, list[tuple[str, ...]]]:
    keys = list(dict.fromkeys((*values["laptop"], *values["slurm"])))
    first: dict[str, int] = {}
    for i, key in enumerate(keys):
        first.setdefault(".".join(key.split(".")[:2]), i)
    keys.sort(key=lambda k: first[".".join(k.split(".")[:2])])
    rows: dict[str, list[tuple[str, ...]]] = {}
    for key in keys:
        rows.setdefault(key.split(".")[0], []).append(
            (
                f"`{key}`",
                _shown(values["laptop"].get(key, "-")),
                _shown(values["slurm"].get(key, "-")),
                meaning.get(key, ""),
            )
        )
    return rows


def profiles_page() -> str:
    """Return the execution profile reference: every key with its laptop value, its slurm template value and meaning."""
    laptop, slurm = (_profile_file(n).read_text("utf-8") for n in ("laptop", "slurm"))
    meaning = {**_comments(slurm), **_comments(laptop)}
    values = {n: profile_rows(n) for n in ("laptop", "slurm")}
    header = ["Key", "laptop", "slurm template"]
    rows = _section_rows(values, meaning)
    scalars = [
        (*r[:3], r[3] or meaning.get(r[0].strip("`").split(".")[0], ""))
        for section in rows.values()
        if len(section) == 1
        for r in section
    ]
    sections = [
        f"### `{top}`\n\n{meaning.get(top, '')}\n\n"
        + table(
            [*header, *(["Meaning"] if any(r[3] for r in section) else [])],
            [r if any(x[3] for x in section) else r[:3] for r in section],
        )
        + "\n"
        for top, section in rows.items()
        if len(section) > 1
    ]
    return f"""{BADGES["2x"]}

An execution profile (`execution-profile@5`) says where and how a plan runs: devices, resource classes, the substrate
that starts workers, the pool, calibration, leases and fault budgets. It is recorded with each result and never
reaches a key, so a plan has one hash under every profile. The packaged profiles are
{", ".join(f"`{p}`" for p in packaged_profiles())}. `ExecutionProfile.load(name_or_path)` reads one; a mapping or file
given with it is deep-merged over the profile it names, and a key the format does not define is refused.

{_api.entry("ExecutionProfile.load", ExecutionProfile.load, level=2)}
## Site profiles

senselab ships generic profiles only. The `laptop` profile runs on one machine. The `slurm` profile is a template, not
a site profile: every `<fill: ...>` in it is yours to set, and it refuses to load until none is left.

- A group writes its own site profile file: copy `slurm`, fill the values, rename it and give the file as the
  execution profile, `ExecutionProfile.load("my-site.yaml")`.
- Placement reads no scheduler. A QOS limit on submitted jobs or GPUs is invisible to it, and a run's coordinators are
  jobs too, so they count against a per-user submit limit that no placement request shows. The site's placement hook,
  a `module:callable` named in `substrate.placement_hook`, is where a group accounts for such limits: it receives a
  `PlacementRequest` and answers a `PlacementAnswer`.
- A group calibrates its own resource with `benchmarks/scale/ladder.py run SUITE --execution PROFILE --measures-root
  PATH`, then lists that shared measures root in `calibration.measures_roots`. A unit measured there is read on an
  identity match, so one calibration serves every member of the group.
- There is no packaged resource table: figures for a resource come from measures made on it.

## Keys

Values are those of the two packaged files; `-` means that file does not set the key. A key given in your profile
replaces the value.

### Single keys

{table([*header, "Meaning"], scalars)}

{chr(10).join(sections)}"""


def datasets_api_section() -> str:
    """Return the section of the datasets page on the dataset objects `Dataset.bids` and `Dataset.from_records`."""
    return f"""## Dataset objects

A plan runs over a dataset: records of named inputs. `senselab.Dataset` holds the two constructors. A BIDS tree is
read where it lies; the records of a dataset in memory are given as rows.

{_api.entry("senselab.Dataset.bids", senselab.Dataset.bids)}
{_api.entry("senselab.Dataset.from_records", senselab.Dataset.from_records)}
### The dataset protocol

{_api.doc(DatasetProtocol)}

{_api.members(DatasetProtocol, ("records", "entities", "filter", "shard", "locate"))}
A record:

{_api.doc(Record)}

Fields: {_api.field_names(Record)}.
"""


EXAMPLE_STEP = "stt"
"""The scenario of the exit checks that the example is: its task, data manifest and number of utterances."""
EXAMPLE_HASH = "97b8bdc549994632929072d206841a2dc282070ad1bc6235057f3eaa43bbff47"
"""The plan hash recorded for the example by the Phase 3 exit run: `SpeechToText()` over the 1000-utterance tree."""
EXAMPLE_RUN = {
    "commit": "ab62fd60",
    "model": "openai/whisper-large-v3-turbo@41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
    "items": 1000,
    "committed": 1000,
    "interrupted_at": 421,
}
"""What the exit run recorded, with the commit it ran at: the default model it planned with, the items, the items
committed, and the items committed when the interrupted run was stopped. A test fails when the catalogue default or the
exit scenario no longer equals it, so the page is not shown against a run that is not its own."""


def example_page(root: Path | None = None) -> str:
    """Return the LibriSpeech example: fetch, lay a BIDS tree, `.map` a task over it, its plan and `explain()`.

    Raises:
        FileNotFoundError: no checkout holds the manifest.
    """
    root = _require_root(root)
    step = exit_checks(root)["steps"][EXAMPLE_STEP]
    items, name = step["items"], step["manifest"]
    manifest = DataManifest.load(root / "benchmarks" / "manifests" / f"{name}.yaml")
    default = CATALOGUE.default("speech_to_text")
    run = EXAMPLE_RUN
    return f"""{BADGES["2x"]}

A speech-to-text plan over {items} LibriSpeech test-clean utterances: the dataset as a BIDS tree, the task
mapped over it, the plan and its `explain()`. Everything here is the code a user writes; the plan hash and counts are
those recorded by the Phase 3 exit run on a GPU worker.

## Data

LibriSpeech test-clean, licensed {manifest.license.expression}. {" ".join(manifest.terms.split())}

```bash
uv run senselab data fetch benchmarks/manifests/{name}.yaml
```

The archive lands in the data cache, every hash verified. `benchmarks/scale/bids_tree.py` lays the first {items}
utterances, in archive order, as `sub-<speaker>/ses-01/audio/sub-<speaker>_ses-01_task-read_run-<k>_audio.flac` under a
BIDS root. It fetches the archive itself, so this one command does both:

```bash
uv run python -c "import sys; sys.path.insert(0, 'benchmarks/scale'); import bids_tree, pathlib; \\
bids_tree.from_librispeech(pathlib.Path('work/data/librispeech-{items}'), \\
pathlib.Path('benchmarks/manifests/{name}.yaml'), {items})"
```

A BIDS tree of your own is read the same way.

## The task over the dataset

```python
from pathlib import Path

import senselab

dataset = senselab.Dataset.bids(Path("work/data/librispeech-{items}"))
task = senselab.SpeechToText()

with senselab.use(execution="laptop"):
    placed = senselab.plan(task, dataset)
    print(placed.hash)
    print(placed.explain())

results = task.map(dataset)
```

`task.map(dataset)` plans the task over every record and runs it on the session's executor; the `ResultSet` it
returns holds one entry per utterance, in record order. `SpeechToText()` takes the default model of the task,
`{default.id}`, pinned at commit `{default.revision[:12]}`, licensed {default.license.expression}.

## The plan

The plan of this task over this tree has the hash

```
{EXAMPLE_HASH}
```

It is the same under the `laptop` profile and under a site profile of a group's own: the two plans differ in their
placement and nowhere else. The profile is a user file, so the example runs under the packaged `laptop` profile; a
cluster run gives its own file, copied from the `slurm` template (see [profile keys](profiles.html#site-profiles)), as
`execution=`.

## `explain()`

`explain()` prints the plan hash, the envs to build, then a line per node: its backend, stack, env, device, batch size,
resource class and partition, the licenses of its env and model, and why the placement chose them. Figures for wall
time, GPU hours and memory are labelled `measured`, with the source, or `unmeasured`, with the reason; a
resource class says how many items the result store would serve and how many compute. See
[`Plan.explain`](plans.html) for each line.

## Running it

Running needs the `pydra` extra of `senselab-core`. On one GPU worker, {run["committed"]} of {run["items"]}
utterances were committed (commit `{run["commit"]}`). A run stopped with {run["interrupted_at"]} of {run["items"]}
committed, a third of the items being the target, and resumed (`senselab.resume`) served the committed items from the
store and computed only the rest, under the same plan hash. Both are in the [run reference](runs.html).
"""
