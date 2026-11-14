"""Reference for senselab 2.x.

Pick the page that answers your question.

- [tasks](reference/tasks.html): what each task takes, returns and accepts as params, and which backends run it.
- [models](reference/models.html): the pinned models, with their licenses.
- [datasets](reference/datasets.html): the public datasets examples and benchmarks run on, and the dataset objects.
- [plans](reference/plans.html): planning a task over a dataset, the plan and its hash, `explain` and `resume`.
- [runs](reference/runs.html): starting a run, its status and wait, its results, and the session defaults.
- [executors](reference/executors.html): `InProcess`, `Pool` and the substrates that start pool workers.
- [profiles](reference/profiles.html): every execution profile key, and how a group writes a site profile.
- [librispeech_example](reference/librispeech_example.html): a speech-to-text plan over LibriSpeech utterances.
- [status](reference/status.html): what 2.x has built, benchmarked and ported.

A backend is a code path, declared in code, that runs a task in an environment. A model is a catalogue entry, data
and not code, pinned by commit. Environments are built with uv-managed Python interpreters only; a system Python is
never used.

Each page carries a status badge: available in 2.x, 1.x only, or planned. Pages are generated from the
declarations, the catalogue and the manifests, so they cannot drift from the code.
"""
