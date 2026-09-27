"""The study's cohort conditions, read from the packaged profile, against phrases the r6 reviewer wrote."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.cohort import COHORT, OTHER, load_cohort_profile

PROFILE = "bridge2ai_voice_adult_2026-09-04"


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
def test_a_cohort_diagnosis_is_recognised(text: str, diagnosis: str) -> None:
    """The study's own diagnoses, their spellings in the transcripts, and treatments the release ties to them."""
    profile = load_cohort_profile(PROFILE)
    assert profile.diagnosis(text) == diagnosis
    assert profile.kind(text) == COHORT


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
def test_a_condition_outside_the_cohorts_is_other(text: str) -> None:
    """Conditions the release names no diagnosis file for, including a joint cyst and a thyroid growth."""
    assert load_cohort_profile(PROFILE).kind(text) == OTHER


def test_every_diagnosis_file_of_the_release_is_a_cohort_except_control() -> None:
    """The profile's entries are the release's phenotype/diagnosis/ files, control excluded."""
    names = {name for name, _ in load_cohort_profile(PROFILE).conditions}
    assert len(names) == 20 and "control" not in names


def test_an_unknown_profile_is_refused() -> None:
    """A mistyped name is an error, not an empty cohort."""
    with pytest.raises(FileNotFoundError):
        load_cohort_profile("no-such-study")
