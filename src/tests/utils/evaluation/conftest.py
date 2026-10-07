"""Markers and the synthetic cohort for the evaluation tests.

``slow`` tests take more than a few seconds, ``network`` tests need to reach the internet and ``gpu``
tests need CUDA. The fast suite deselects all three with ``-m "not slow and not network and not gpu"``.
A ``gpu`` test also carries ``pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")``,
so selecting it on a CPU-only host skips it rather than failing.

``imbalanced_cohort`` is a seeded synthetic cohort; nothing in it comes from real data.
"""

from dataclasses import dataclass

import numpy as np
import pytest

# Frozen after the red-phase measurements recorded in the specs note (Split methods, fixture).
COHORT_SEED = 0
N_SUBJECTS = 60
CLIPS_PER_SUBJECT = (4, 8, 16, 32, 64)
SUBJECT_RATES = (0.0, 0.05, 0.10, 0.30)
SUBJECT_RATE_WEIGHTS = (0.3, 0.3, 0.2, 0.2)
MULTILABEL_RATES = (0.5, 0.2, 0.05)


def pytest_configure(config: pytest.Config) -> None:
    """Register the three markers so ``--strict-markers`` accepts them."""
    config.addinivalue_line("markers", "slow: takes more than a few seconds")
    config.addinivalue_line("markers", "network: needs to reach the internet")
    config.addinivalue_line("markers", "gpu: needs a CUDA device")


@dataclass(frozen=True)
class Cohort:
    """One row per clip.

    Attributes:
        labels: Binary label per clip (0/1).
        subjects: Subject key per clip.
        recordings: Recording key per clip, unique only within a subject.
        clips: Clip key per clip, unique only within a subject.
        timepoints: ``t1`` or ``t2`` per clip; each recording belongs to one timepoint.
        multilabel: ``(n, 3)`` 0/1 matrix.
    """

    labels: np.ndarray
    subjects: np.ndarray
    recordings: np.ndarray
    clips: np.ndarray
    timepoints: np.ndarray
    multilabel: np.ndarray


def make_cohort(seed: int = COHORT_SEED) -> Cohort:
    """Build the synthetic cohort.

    60 subjects with 4-64 clips each in 2-4 recordings, alternating timepoints by recording. A
    subject's positive rate is one of 0, 0.05, 0.10 or 0.30 (about 0.10 overall), realised exactly as
    ``round(rate * n_clips)`` positive clips. The three multi-label columns have per-subject rates
    drawn around 0.5, 0.2 and 0.05.
    """
    rng = np.random.default_rng(seed)
    labels, subjects, recordings, clips, timepoints, multilabel = [], [], [], [], [], []
    for s in range(N_SUBJECTS):
        n = int(rng.choice(CLIPS_PER_SUBJECT))
        n_rec = int(rng.integers(2, 5))
        rate = float(rng.choice(SUBJECT_RATES, p=SUBJECT_RATE_WEIGHTS))
        y = np.zeros(n, dtype=int)
        y[rng.permutation(n)[: round(rate * n)]] = 1
        rec = rng.permutation(np.arange(n) % n_rec)
        ml_rates = [min(1.0, rng.uniform(0.0, 2.0 * p)) for p in MULTILABEL_RATES]
        ml = np.column_stack([rng.random(n) < p for p in ml_rates]).astype(int)
        labels.append(y)
        subjects += [f"s{s:02d}"] * n
        recordings += [f"r{r}" for r in rec]
        clips += [f"c{i}" for i in range(n)]
        timepoints += ["t1" if r % 2 == 0 else "t2" for r in rec]
        multilabel.append(ml)
    return Cohort(
        labels=np.concatenate(labels),
        subjects=np.array(subjects),
        recordings=np.array(recordings),
        clips=np.array(clips),
        timepoints=np.array(timepoints),
        multilabel=np.vstack(multilabel),
    )


@pytest.fixture(scope="session")
def imbalanced_cohort() -> Cohort:
    """The frozen synthetic cohort."""
    return make_cohort()


def _cells(*keys: np.ndarray) -> dict[tuple, np.ndarray]:
    """Row indices per distinct key tuple, in sorted key order."""
    combined = list(zip(*(k.tolist() for k in keys)))
    cells: dict[tuple, list[int]] = {}
    for i, key in enumerate(combined):
        cells.setdefault(key, []).append(i)
    return {key: np.array(cells[key]) for key in sorted(cells)}


def clip_level_holdout(
    labels: np.ndarray,
    subjects: np.ndarray,
    strata: np.ndarray,
    seed: int,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> dict[str, np.ndarray]:
    """A clip-level 60/20/20 holdout, stratified by label within each (subject, stratum) cell.

    A reimplementation, on synthetic data, of a seen-subject protocol that splits clips rather than
    recordings: within a cell, each label's clips are shuffled and cut by the fractions, and at least
    one clip stays in train.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for rows in _cells(subjects, strata, labels).values():
        rows = rng.permutation(rows)
        n_test = round(len(rows) * fractions[2])
        n_val = round(len(rows) * fractions[1])
        if n_test + n_val >= len(rows):
            n_test, n_val = max(0, len(rows) - 1 - n_val), min(n_val, len(rows) - 1)
        parts["test"] += rows[:n_test].tolist()
        parts["val"] += rows[n_test : n_test + n_val].tolist()
        parts["train"] += rows[n_test + n_val :].tolist()
    return {name: np.sort(np.array(rows, dtype=int)) for name, rows in parts.items()}


def within_class_subject_split(
    labels: np.ndarray, subjects: np.ndarray, seed: int, train_fraction: float = 0.8
) -> dict[str, np.ndarray]:
    """A mutant: clips split within each (label, subject) group, ignoring recordings.

    Each group's clips are shuffled; ``max(1, round(train_fraction * n))`` go to train, the rest to test.
    """
    rng = np.random.default_rng(seed)
    parts: dict[str, list[int]] = {"train": [], "test": []}
    for rows in _cells(labels, subjects).values():
        rows = rng.permutation(rows)
        n_train = max(1, round(train_fraction * len(rows)))
        parts["train"] += rows[:n_train].tolist()
        parts["test"] += rows[n_train:].tolist()
    return {name: np.sort(np.array(rows, dtype=int)) for name, rows in parts.items()}
