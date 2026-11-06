"""The study's recruited conditions, read from the dataset profile, against phrases the r6 reviewer wrote."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.recruited_conditions import (
    OTHER,
    RECRUITED,
    condition_kind_of,
    ledger_value,
    load_recruited_conditions,
)

PROFILE = "adult_speech_fixture_2026-09-04"


@pytest.mark.parametrize(
    ("text", "diagnosis"),
    [
        ("parkinson's", "parkinsons_disease"),
        ("Parkinson’s disease", "parkinsons_disease"),
        ("pd", "parkinsons_disease"),
        ("levodopa carbidopa", "parkinsons_disease"),
        ("deep deep brain stimulation", "parkinsons_disease"),
        ("idiopathic subglottic stenosis", "airway_stenosis"),
        ("idiopathic sublotic stenosis", "airway_stenosis"),
        ("surgery on my throat to reconstruct my windpipe", "airway_stenosis"),
        ("spasmodic dysphonia", "laryngeal_dystonia"),
        ("botox injections", "laryngeal_dystonia"),
        ("mtd", "muscle_tension_dysphonia"),
        ("paralyzed vocal cord", "unilateral_vocal_fold_paralysis"),
        ("one vocal cord has stopped moving", "unilateral_vocal_fold_paralysis"),
        ("nodules on my vocal cords", "benign_lesions"),
        ("recurrent respiratory papillomatosis", "precancerous_lesions"),
        ("essential tremors", "essential_tremor"),
        ("alzheimer's", "cognitive_impairment"),
        ("huntington's disease", "huntingtons_disease"),
        ("tos crónica", "unexplained_chronic_cough"),
    ],
)
def test_a_recruited_diagnosis_is_recognised(text: str, diagnosis: str) -> None:
    """The study's own diagnoses, their spellings in the transcripts, and treatments the release ties to them."""
    profile = load_recruited_conditions(PROFILE)
    assert profile.diagnosis(text) == diagnosis
    assert profile.kind(text) == RECRUITED


@pytest.mark.parametrize(
    "text",
    [
        "synovial joint cyst",
        "epidermoid cyst on the brain",
        "thyroid cancer",
        "sleep apnea",
        "multiple sclerosis",
        "vocal cord dysfunction",
        "a very rare voice disorder",
        "had a big growth on the side of my thyroid",
    ],
)
def test_a_condition_outside_the_recruited_ones_is_other(text: str) -> None:
    """Conditions the release names no diagnosis file for, including a joint cyst and a thyroid growth."""
    assert load_recruited_conditions(PROFILE).kind(text) == OTHER


def test_every_diagnosis_file_of_the_release_is_recruited_except_control() -> None:
    """The profile's entries are the release's phenotype/diagnosis/ files, control excluded."""
    names = {name for name, _ in load_recruited_conditions(PROFILE).conditions}
    assert len(names) == 20 and "control" not in names


def test_an_unknown_profile_is_refused() -> None:
    """A mistyped name is an error, not an empty list."""
    with pytest.raises(FileNotFoundError):
        load_recruited_conditions("no-such-study")


def test_a_store_from_before_the_rename_reads_under_the_new_names() -> None:
    """Plan step 9: ``cohort`` was the recruited kind, and its keys were ``cohort_*``; both still read."""
    assert condition_kind_of("cohort") == RECRUITED and condition_kind_of(OTHER) == OTHER
    old = {"cohort_diagnoses": ["essential_tremor"], "counts": {"cohort_condition_n": 2}}
    assert ledger_value(old, "recruited_diagnoses") == ["essential_tremor"]
    assert ledger_value(old["counts"], "recruited_condition_n") == 2
    assert ledger_value({"recruited_diagnoses": []}, "recruited_diagnoses") == []
