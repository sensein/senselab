"""The ppgs checkpoint is loaded from a resolved commit, never through a ref."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict

import pytest

from senselab.audio.tasks.features_extraction import ppg

SHA = "407a830baa843ec7f5dbc1cbcc7a819d721c1d4f"


def test_checkpoint_is_downloaded_at_the_resolved_commit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``ppgs_checkpoint`` resolves the ref first and downloads the file at the SHA, not the ref."""
    calls: Dict[str, Any] = {}

    def fake_resolve(repo_id: str, ref: str = "main", **_: Any) -> str:  # noqa: ANN401
        calls["resolve"] = (repo_id, ref)
        return SHA

    def fake_download(repo_id: str, filename: str, *, revision: str) -> str:
        calls["download"] = (repo_id, filename, revision)
        return str(tmp_path / "snapshots" / revision / filename)

    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", fake_resolve)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)

    sha, path = ppg.ppgs_checkpoint()
    assert sha == SHA
    assert calls["resolve"] == (ppg.PPGS_REPO_ID, "main")
    assert calls["download"] == (ppg.PPGS_REPO_ID, ppg.PPGS_CHECKPOINT_FILE, SHA)
    assert path.parent.name == SHA
    assert ppg.ppgs_model_revision() == SHA


def test_worker_loads_the_staged_checkpoint_and_makes_no_hub_call() -> None:
    """The worker passes the parent's checkpoint path to ``ppgs.load.model`` and never imports huggingface_hub."""
    tree = ast.parse(ppg._WORKER_SCRIPT)  # noqa: SLF001 -- the worker string is what is under test
    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "model"
    ]
    assert loads, "the worker no longer calls ppgs.load.model"
    for call in loads:
        keywords = {k.arg: k.value for k in call.keywords}
        assert "checkpoint" in keywords, "ppgs.load.model without checkpoint= downloads through refs/main"
        assert ast.unparse(keywords["checkpoint"]) == "args['checkpoint']"
    assert "huggingface_hub" not in ppg._WORKER_SCRIPT  # noqa: SLF001
