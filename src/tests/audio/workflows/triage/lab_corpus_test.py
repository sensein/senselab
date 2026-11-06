"""The lab corpus: a made-up dataset, one invented task per kind of task triage reads (plan step 8c).

Its profile (``profiles/lab_corpus``) is written from senselab's template, builds on the reference profile and takes
its corpus statistics from a ``fitted.yaml``. Every recording has a plain name and declares its identity in its
hint, so these tests run with no file, task or value from the fixture profile. The kinds ``effort`` and
``per_sentence`` are left out: no branch has a matcher for them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pytest
import soundfile
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.dataset_profile import (
    deactivate_profile,
    missing_dataset_values,
    unfitted_values,
)
from senselab.audio.workflows.triage.identity import recorded_identity
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS, Result, branch_params, dispatch
from senselab.audio.workflows.triage.nodes.gates import conformance_gate_names, load_gate_bounds
from senselab.audio.workflows.triage.task_catalogue import FROM_TASK_LIST, recorded_task, task_list
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore
from tests.audio.workflows.triage.conftest import hint_for
from tests.conftest import FIXTURE_PROFILE

LAB_CORPUS = Path(__file__).parent / "profiles" / "lab_corpus"
"""The made-up dataset's profile."""

KINDS = {
    "hold-a-tone": ("VOICE", "sustained"),
    "slide-up": ("VOICE", "glide"),
    "read-the-card": ("SPEECH", "ordered_tokens"),
    "tell-a-story": ("SPEECH", "free_response"),
    "name-some-fruit": ("SPEECH", "item_list"),
    "pa-pa-pa": ("SPEECH", "syllable_train"),
    "pa-ta-ka": ("SPEECH", "syllable_sequence"),
    "three-coughs": ("AIRWAY", "event_series"),
    "cough-then-breathe": ("AIRWAY", "event_alternation"),
    "quiet-breathing": ("AIRWAY", "sound_coverage"),
}
"""Each invented task, its branch and its kind (the task list's ``expects.pattern``)."""


@pytest.fixture
def lab_corpus(without_triage_profile: None) -> Iterator[TriageConfig]:
    """The lab corpus's profile, active for the test, and no profile active after it."""
    config = load_triage_config(profile=LAB_CORPUS)
    yield config
    deactivate_profile()


def _admitted(config: TriageConfig, wav: Path, task: str) -> ProvStore:
    """A plainly named recording through the real ADMIT node, its identity declared in its hint."""
    store = ProvStore(run_id=wav.stem)
    t = np.arange(16000) / 16000
    soundfile.write(wav, (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32), 16000)
    result = admit(store, wav, config, hint_for("p07", "s2", task), run_dir=wav.parent)
    assert result.verdict.outcome is Outcome.PASS
    return store


def test_its_profile_is_all_a_run_needs(lab_corpus: TriageConfig) -> None:
    """The start check finds nothing missing and nothing unfitted; the task list is the lab's ten tasks."""
    assert missing_dataset_values(lab_corpus.values) == []
    assert unfitted_values() == []
    assert set(task_list()) == set(KINDS)


def test_it_reads_no_file_of_the_fixture_profile(lab_corpus: TriageConfig) -> None:
    """No task of the fixture's is the lab's, and the active profile chain does not include the fixture."""
    fixture_tasks = set((yaml.safe_load((FIXTURE_PROFILE / "tasks.yaml").read_text()) or {})["tasks"])
    assert not fixture_tasks & set(task_list())
    assert lab_corpus.profile is not None
    assert all(layer.root != FIXTURE_PROFILE.resolve() for layer in lab_corpus.profile.chain())
    fixture_files = {path.read_bytes() for path in FIXTURE_PROFILE.rglob("*") if path.is_file()}
    copied = [
        str(path.relative_to(LAB_CORPUS))
        for path in LAB_CORPUS.rglob("*")
        if path.is_file() and path.read_bytes() in fixture_files
    ]
    assert copied == [], f"copied from the fixture: {copied}"


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kind_is_admitted_with_its_declared_identity(lab_corpus: TriageConfig, tmp_path: Path, task: str) -> None:
    """ADMIT records the hint's participant, session and task from a plain file name."""
    store = _admitted(lab_corpus, tmp_path / "rec-0001.wav", task)
    identity = recorded_identity(store)
    assert identity is not None and (identity.participant, identity.session, identity.task_id) == ("p07", "s2", task)
    recorded = recorded_task(store)
    assert recorded is not None and (recorded.name, recorded.source, recorded.branch) == (
        task,
        FROM_TASK_LIST,
        KINDS[task][0],
    )


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kind_is_read_by_its_branchs_matcher(lab_corpus: TriageConfig, tmp_path: Path, task: str) -> None:
    """The owning branch aligns the recording with the task's matcher; the other branches detect."""
    store = _admitted(lab_corpus, tmp_path / "rec-0002.wav", task)
    owner, kind = KINDS[task]
    assert EXPECTATIONS[owner][task].pattern.value == kind
    params = branch_params(lab_corpus)
    for branch in ("AIRWAY", "SPEECH", "VOICE"):
        ran: list[str] = []
        dispatch(
            branch,
            store,
            params,
            hint_for("p07", "s2", task),
            align=lambda family, *_: ran.append(f"align {family}") or Result([], []),
            detect=lambda *_: ran.append("detect") or Result([], []),
        )
        assert ran == ([f"align {task}"] if branch == owner else ["detect"]), branch


@pytest.mark.parametrize("task", sorted(KINDS))
def test_each_kinds_gates_resolve(lab_corpus: TriageConfig, task: str) -> None:
    """Every gate the task's kind reads as its conformance has a bound, from the reference profile's layers."""
    owner, _ = KINDS[task]
    pattern = EXPECTATIONS[owner][task].pattern
    bounds = load_gate_bounds(lab_corpus, pattern, task)
    missing = [name for name in conformance_gate_names(pattern) if bounds.bounds.get(name) is None]
    assert missing == [], f"{task}: no bound for {missing}"
