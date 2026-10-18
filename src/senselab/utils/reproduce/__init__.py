"""Reproduce an analysis from someone else's repository: pinned code, a locked venv, a sandboxed run.

Each step of an analysis names a repository commit, a set of hashed locks, a command and the
aggregate metrics it should produce. The driver checks the code out at that commit, builds the venv
from the locks ahead of time, runs the command in a subprocess inside a sandbox (no network,
read-only data, an allowlisted environment), records provenance, and skips any step whose inputs
have not changed. The design is in ``specs/20261007-reproduce-driver/design.md``.
"""
