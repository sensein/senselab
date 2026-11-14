"""The job engine behind the `Executor` protocol: plan fragments of jobs run, served or stored, and recorded.

`executor` holds the protocol and its documents, `job` the one engine (`execute`), `workers` the worker set it runs
on, `in_process` the `InProcess` executor, `events` a run's event log and status, and `run` its lifecycle and results.
"""
