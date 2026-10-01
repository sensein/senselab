"""The second-opinion driver: SECOND_OPINION over a finished corpus, resumable, re-folding what it changes.

``specs/20261001-nimble-second-opinion/design.md`` is the design.
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Iterator

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.vocabulary import NIMBLE_OPINION, SECOND_OPINION_DISAGREES
from senselab.utils.prov_store import ProvStore
from tests.scripts.extend_llm_review_test import _finished_run, _hints, _manifest, _seed_verdicts

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLI = _REPO_ROOT / "scripts" / "extend_second_opinion.py"
_spec = importlib.util.spec_from_file_location("extend_second_opinion_under_test", _CLI)
assert _spec is not None and _spec.loader is not None, f"could not load {_CLI}"  # noqa: S101
cli = importlib.util.module_from_spec(_spec)
sys.modules["extend_second_opinion_under_test"] = cli
_spec.loader.exec_module(cli)

ON = "second_opinion:\n  enabled: true\n"


def _config(tmp_path: Path) -> Any:  # noqa: ANN401 — TriageConfig
    path = tmp_path / "override.yaml"
    path.write_text(ON, encoding="utf-8")
    return load_triage_config(path)


def _answers(other: float) -> dict[str, Any]:
    return {
        "other_voice": {"choice": "one", "probabilities": {"one": 1 - other, "more_than_one": other}},
        "instructions_spoken": {"noul": 0.02},
        "named_diagnosis": {"noul": 0.01},
        "safe_harbor_identifier_present": {"noul": 0.03},
    }


class _Opener:
    """An :data:`OpenAsk` counting how often the server was started and asked."""

    def __init__(self, other: float = 0.02, fail: bool = False) -> None:
        self.other = other
        self.fail = fail
        self.opened = 0
        self.asked = 0

    @contextlib.contextmanager
    def __call__(self) -> Iterator[Any]:
        self.opened += 1
        if self.fail:
            raise RuntimeError("ollama serve exited with 1 before answering")

        def ask(state: Any, questions: Any) -> dict[str, Any]:  # noqa: ANN401
            self.asked += 1
            return _answers(self.other)

        yield ask


def _run(tmp_path: Path, run_root: Path, opener: _Opener, **kw: Any) -> dict[str, Any]:  # noqa: ANN401
    return cli.run_slice(
        _manifest(tmp_path, run_root),
        slice_index=0,
        slice_count=1,
        config=_config(tmp_path),
        log_dir=tmp_path,
        open_ask=opener,
        **kw,
    )


def _opinions(run_root: Path) -> list[dict[str, Any]]:
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    return [
        dict(entity.attributes)
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == NIMBLE_OPINION and not store.is_invalidated(entity.id)
    ]


def test_an_opinion_lands_and_a_second_pass_starts_no_server(tmp_path: Path) -> None:
    """Resumable by the store: a standing ok opinion is present and the model is not served."""
    run_root = _finished_run(tmp_path / "corpus")
    first = _Opener()
    assert _run(tmp_path, run_root, first)["counts"] == {"ok": 1}
    assert first.opened == 1 and len(_opinions(run_root)) == 1
    again = _Opener()
    summary = _run(tmp_path, run_root, again)
    assert summary["counts"] == {"present": 1} and again.opened == 0 and summary["server_started"] is False


def test_force_asks_again_and_keeps_one_live_opinion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed answer retires the one it replaces."""
    run_root = _finished_run(tmp_path / "corpus")
    _run(tmp_path, run_root, _Opener(other=0.02))
    monkeypatch.setenv("SENSELAB_RESULT_CACHE", str(tmp_path / "fresh-cache"))
    _run(tmp_path, run_root, _Opener(other=0.6), force=True)
    live = _opinions(run_root)
    assert len(live) == 1 and live[0]["probabilities"]["other_voice"] == 0.6


def test_a_server_that_will_not_start_is_an_error_on_every_row_and_starts_once(tmp_path: Path) -> None:
    """Nothing is written, and the failed start is not retried row after row."""
    roots = [_finished_run(tmp_path / f"corpus{index}") for index in range(2)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            f'{{"stem": "{root.name}{index}", "enhanced": "{root / "run" / "streams" / "enhanced.flac"}"}}\n'
            for index, root in enumerate(roots)
        ),
        encoding="utf-8",
    )
    opener = _Opener(fail=True)
    summary = cli.run_slice(
        manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path, open_ask=opener
    )
    assert summary["counts"] == {"error": 2} and opener.opened == 1
    assert not any(_opinions(root) for root in roots)


def test_a_landed_opinion_is_folded_into_the_verdict(tmp_path: Path) -> None:
    """With hints, VERDICT is decided again and carries the opinion it compared."""
    run_root = _finished_run(tmp_path / "corpus")
    _seed_verdicts(run_root)
    summary = _run(tmp_path, run_root, _Opener(other=0.95), hints=_hints(tmp_path, run_root))
    assert summary["counts"] == {"ok": 1}
    assert list(summary["refolds"])[0].startswith(("flag/", "pass/", "discard/"))
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    folded = [
        entity
        for entity in store.entities("verdict")
        if entity.attributes.get("node") == "VERDICT" and not store.is_invalidated(entity.id)
    ]
    assert folded, "the re-fold wrote no live VERDICT"
    assert folded[-1].attributes["second_opinion"]["status"] == "ok"


def test_the_cli_refuses_a_config_that_leaves_it_off(tmp_path: Path) -> None:
    """Writing disabled into every store is not a pass."""
    run_root = _finished_run(tmp_path / "corpus")
    code = cli.main(
        [
            str(_manifest(tmp_path, run_root)),
            "--slice-index",
            "0",
            "--slice-count",
            "1",
            "--no-refold",
            "--ollama-binary",
            str(tmp_path / "ollama"),
            "--ollama-models",
            str(tmp_path / "models"),
        ]
    )
    assert code == 2


def test_the_ground_is_controlled_vocabulary() -> None:
    """The flag ground names the comparison, not the model."""
    assert "second-opinion" in SECOND_OPINION_DISAGREES
