"""A run's jobs on pydra's warm pool: their formation, keys, store, providers, events and lifecycle.

`executor` holds the documents of a run's jobs, `fragments` and `job` their formation and keys, `portable` the store
they publish to, `providers` and `warm` the pool session they run on, `runner` the driver of pydra's `GraphRunner`,
`events` a run's event log and status, `run` its lifecycle and results, and `calls` one task call.
"""
