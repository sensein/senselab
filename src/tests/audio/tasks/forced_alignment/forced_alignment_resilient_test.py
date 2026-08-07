"""Behavior test: the MMS forced aligner loads via the SHA-pinning resilient loader.

The auto-align stage loads facebook/mms-1b-all in-process for every timestamp-less
ASR output, so it must route through load_hf_resilient (resolve->SHA-pin +
local_files_only) rather than a bare from_pretrained that HEADs the Hub each call.
"""

from unittest.mock import MagicMock

import pytest

from senselab.utils.data_structures import DeviceType


def test_mms_aligner_loads_via_load_hf_resilient(monkeypatch: pytest.MonkeyPatch) -> None:
    """_load_mms_aligner resolves + SHA-pins the MMS model (processor + model) via load_hf_resilient."""
    import senselab.audio.tasks.forced_alignment.forced_alignment as fa

    repos: list = []

    def spy(loader: object, *args: object, repo_id: str, revision: str = "main", **k: object) -> MagicMock:
        repos.append(repo_id)
        return MagicMock()

    monkeypatch.setattr(fa, "load_hf_resilient", spy)

    cache: dict = {}
    fa._load_mms_aligner("eng", DeviceType.CPU, cache)

    # Both the processor and the model are loaded through the resilient loader, pinned to MMS.
    assert repos == [fa.MMS_MODEL_ID, fa.MMS_MODEL_ID]
    assert (fa.MMS_MODEL_ID, "eng") in cache
