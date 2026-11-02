"""The license records of the packages of each env lock, read from a packaged ``env-licenses`` file.

The file is written by ``scripts/check_licenses.py --lock NAME`` (or ``--host``) from package metadata. Its envs are
the lock names, and ``host`` for the root lock; `lock_records_from` gives `contract_cases` its ``lock_records``.
"""

import yaml

from senselab_core.kernel.declare import BackendSpec
from senselab_core.kernel.license import License
from senselab_core.runtime.licenses import license_record
from senselab_core.testing.contracts import LockRecords

ENV_LICENSES_FORMAT = "env-licenses@1.0"
HOST_ENV = "host"


def env_name(backend: BackendSpec) -> str:
    """Return the env whose records cover `backend`: its lock name, or ``host``."""
    return HOST_ENV if backend.env.kind == "host" else str(backend.env.name)


def lock_records_from(text: str) -> LockRecords:
    """Return the provider of ``package -> License`` over the ``env-licenses`` document `text`.

    Raises:
        ValueError: if the format is not ``env-licenses@1.0``, or a backend's env has no records.
    """
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict) or raw.get("format") != ENV_LICENSES_FORMAT:
        raise ValueError(f"env licenses must carry format {ENV_LICENSES_FORMAT!r}")
    envs = raw["envs"]

    def records(backend: BackendSpec) -> dict[str, License | None]:
        name = env_name(backend)
        if name not in envs:
            raise ValueError(f"no license records for env {name!r}; run scripts/check_licenses.py --lock {name}")
        return {
            package: license_record(
                entry["expression"], source=entry["source"], verified=entry["verified"], gated=False
            )
            for package, entry in envs[name].items()
        }

    return records
