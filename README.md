# senselab 2.x

senselab processes behavioural data, primarily voice and speech, through reproducible pipelines.
In 2.x a user or an agent declares what to compute, a planner turns that into an inspectable plan for
a site and its hardware, and an executor runs it. Every task records provenance and caches its
results the same way, with models pinned to commits and backends run in locked, isolated
environments.

## Status

2.x is in Phase 2 and has no release yet.

- Phase 1 delivered `senselab-worker`, the subprocess worker runtime (standard library and numpy), and
  `senselab-core`, the kernel (values, identity, formats) and the runtime (result store, data fetch, locks,
  provenance, worker client).
- Phase 2 adds `senselab`, the public package: declared tasks and backends (`speech_to_text`, `diarize`,
  `speaker_embeddings`, `pii_spans`, `signal.resample`, and the metric tasks), a catalogue of models pinned by
  commit, the benchmark harness and its smoke suite. A backend is a code path; a model is a catalogue entry.
- Pipelines are not ported yet.
- The API reference and the generated 2.x status page are built from the declarations and the specs; see
  `senselab.reference` in the pull-request preview.
- Environments are built with uv-managed Python interpreters only.
- 1.x is developed on the `alpha` branch and released on PyPI as `senselab` 1.x. On this branch
  `src/` is the frozen 1.x source pool that use cases are ported from, one at a time; triage's
  tasks and pipelines port like any other use case, and its decision layer is superseded by the
  flywheel.

## Development

The repository is a uv workspace; the packages are under `packages/`. The interpreter is pinned in
`.python-version`.

```bash
git clone https://github.com/sensein/senselab.git
cd senselab
git checkout 2.x
uv sync --all-extras --group dev   # uv sync removes what it is not told to keep: pass the full set
uv run pre-commit install
```

Run the tests of the package you changed, serially:

```bash
uv run pytest packages/senselab-core/tests/runtime -q
uv run pytest packages/senselab-worker/tests -q
```

Each test carries one tier marker (`logic`, `contract`, `limits`, `usecase`, `env`); select tiers
with `-m`, for example `-m "not network"`. Do not use `pytest -n auto`.

Lint with `uv run lint-imports` and `uv run pre-commit run --all-files`.

## Design and documentation

- Working state: [`specs/20261030-140022-senselab-2x-api/STATE.md`](specs/20261030-140022-senselab-2x-api/STATE.md)
- Principles: [`principles.md`](specs/20261030-140022-senselab-2x-api/principles.md)
- Decisions: [the ADR index](specs/20261030-140022-senselab-2x-api/adr/ADR-000-index.md)
- API docs of a pull request: `https://sensein.group/senselab/2x/pr-<N>/`

## License

Apache License 2.0; see [`LICENSE.md`](LICENSE.md).
