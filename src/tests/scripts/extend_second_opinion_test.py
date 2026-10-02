"""The second-opinion driver: SECOND_OPINION over a finished corpus, resumable, concurrent, folding nothing.

``specs/20261001-nimble-second-opinion/design.md`` is the design.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
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


def _live_verdicts(run_root: Path) -> list[Any]:
    store = ProvStore.read_jsonl(run_root / "run" / "store.jsonl", run_id=run_root.name)
    return [
        entity
        for entity in store.entities("verdict")
        if entity.attributes.get("node") == "VERDICT" and not store.is_invalidated(entity.id)
    ]


def test_the_driver_writes_the_opinion_and_leaves_the_verdict_to_the_refold(tmp_path: Path) -> None:
    """No VERDICT is decided here; extend_refold.py then folds the opinion that landed."""
    run_root = _finished_run(tmp_path / "corpus")
    _seed_verdicts(run_root)
    seeded = [entity.id for entity in _live_verdicts(run_root)]
    hints = _hints(tmp_path, run_root)
    summary = _run(tmp_path, run_root, _Opener(other=0.95), hints=hints)
    assert summary["counts"] == {"ok": 1} and "refolds" not in summary
    assert [entity.id for entity in _live_verdicts(run_root)] == seeded

    refold = _load_refold()
    refold.run_slice(
        _manifest(tmp_path, run_root),
        slice_index=0,
        slice_count=1,
        config=_config(tmp_path),
        log_dir=tmp_path / "refold",
        hints=hints,
    )
    folded = _live_verdicts(run_root)
    assert folded and folded[-1].id not in seeded
    assert folded[-1].attributes["second_opinion"]["status"] == "ok"


def _load_refold() -> Any:  # noqa: ANN401 — a module
    path = _REPO_ROOT / "scripts" / "extend_refold.py"
    spec = importlib.util.spec_from_file_location("extend_refold_for_second_opinion_test", path)
    assert spec is not None and spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    sys.modules["extend_refold_for_second_opinion_test"] = module
    spec.loader.exec_module(module)
    return module


def _corpus(tmp_path: Path, count: int) -> tuple[list[Path], Path]:
    roots = [_finished_run(tmp_path / f"corpus{index}", words=("hello", f"row{index}")) for index in range(count)]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(
            f'{{"stem": "{root.name}{index}", "enhanced": "{root / "run" / "streams" / "enhanced.flac"}"}}\n'
            for index, root in enumerate(roots)
        ),
        encoding="utf-8",
    )
    return roots, manifest


def test_workers_ask_every_row_once_and_keep_the_record_order(tmp_path: Path) -> None:
    """Rows run concurrently; each store is asked once, the summary keeps manifest order, the server starts once."""
    roots, manifest = _corpus(tmp_path, 6)
    opener = _Opener()
    summary = cli.run_slice(
        manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path, open_ask=opener, workers=3
    )
    assert summary["counts"] == {"ok": 6} and summary["workers"] == 3
    assert opener.opened == 1 and opener.asked == 6
    for root in roots:
        opinions = _opinions(root)
        assert len(opinions) == 1 and opinions[0]["num_parallel"] == 3
    rows = (tmp_path / "slices" / "second-opinion-slice-0-of-1.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 6


def test_process_returns_records_in_row_order_under_workers(tmp_path: Path) -> None:
    """Out-of-order completion still returns one record per row, in the rows' order."""
    roots, manifest = _corpus(tmp_path, 5)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records = cli.process(rows, _config(tmp_path), cli.LazyAsk(_Opener()), force=False, workers=4)
    assert [record["stem"] for record in records] == [row["stem"] for row in rows]


def test_the_opinion_records_its_parallelism_but_the_cache_key_does_not(tmp_path: Path) -> None:
    """A serial ask after a parallel one is served from the cache: parallelism is provenance, not identity."""
    roots, manifest = _corpus(tmp_path, 2)
    parallel = _Opener()
    cli.run_slice(
        manifest, slice_index=0, slice_count=1, config=_config(tmp_path), log_dir=tmp_path, open_ask=parallel, workers=2
    )
    serial = _Opener()
    _run(tmp_path, roots[0], serial, force=True)
    assert serial.asked == 0
    assert _opinions(roots[0])[0]["num_parallel"] == 1


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
            "--hints",
            str(_hints(tmp_path, run_root)),
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
