"""Reference for senselab 2.x.

Pick the page that answers your question.

- [tasks](reference/tasks.html): what each task takes, returns and accepts as params, and which backends run it.
- [models](reference/models.html): the pinned models, with their licenses.
- [datasets](reference/datasets.html): the public datasets examples and benchmarks run on.
- [status](reference/status.html): what 2.x has built, benchmarked and ported.

A backend is a code path, declared in code, that runs a task in an environment. A model is a catalogue entry, data
and not code, pinned by commit. Environments are built with uv-managed Python interpreters only; a system Python is
never used.

Each page carries a status badge: available in 2.x, 1.x only, or planned. Pages are generated from the
declarations, the catalogue and the manifests, so they cannot drift from the code.
"""
