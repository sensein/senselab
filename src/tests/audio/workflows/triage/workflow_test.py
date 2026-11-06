"""Which stages a run drives, and which a dataset's workflow may leave off (plan step 11)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.workflow import (
    OFF_IN_WORKFLOW,
    OPTIONAL_STAGES,
    STAGE_NAMES,
    STAGES,
    check_stages,
    listed_stages,
    off_notes,
    require_stage,
    running_stages,
)


def _without(*off: str) -> dict:
    return {"workflow": {"stages": [name for name in STAGE_NAMES if name not in off]}}


class TestTheStageTable:
    """The table is the graph the runner has always driven."""

    def test_a_stage_needs_only_stages_before_it(self) -> None:
        """A need later in the order could not have run yet."""
        for index, stage in enumerate(STAGES):
            assert set(stage.needs) <= set(STAGE_NAMES[:index]), stage.name

    def test_the_stages_every_later_stage_reads_may_not_be_off(self) -> None:
        """ADMIT through routing and VERDICT carry the graph; the branches and the rest are a dataset's choice."""
        assert set(STAGE_NAMES) - OPTIONAL_STAGES == {
            "ADMIT",
            "PREPROCESS",
            "SESSION",
            "BACKGROUND",
            "TAXONOMY",
            "routing",
            "VERDICT",
        }

    def test_senselab_runs_every_stage_by_default(self) -> None:
        """Without a profile the workflow is the whole graph, so a dataset's runs are unchanged."""
        config = load_triage_config()
        assert listed_stages(config.values) == STAGE_NAMES
        assert off_notes(config.values) == {}


class TestAWorkflow:
    """What a ``workflow.stages`` list turns off, and what it refuses."""

    def test_a_stage_left_out_is_off(self) -> None:
        """The note is the one the run records."""
        assert off_notes(_without("AIRWAY", "COHORT")) == {"AIRWAY": OFF_IN_WORKFLOW, "COHORT": OFF_IN_WORKFLOW}

    def test_a_stage_whose_need_is_off_is_off_too(self) -> None:
        """No SPEECH, so nothing to redact or review, and no review for a second opinion."""
        notes = off_notes(_without("SPEECH"))
        assert notes["REDACT"] == notes["REVIEW"] == f"{OFF_IN_WORKFLOW}: needs SPEECH"
        assert notes["SECOND_OPINION"] == f"{OFF_IN_WORKFLOW}: needs REVIEW"
        assert "SPEECH" not in running_stages(_without("SPEECH"))

    def test_an_unknown_stage_is_refused(self) -> None:
        """A misspelled stage is not silently dropped."""
        with pytest.raises(ValueError, match="not triage stages"):
            check_stages([*STAGE_NAMES, "DDK"])

    def test_a_stage_that_may_not_be_off_is_refused(self) -> None:
        """A workflow without routing has no branches to run."""
        with pytest.raises(ValueError, match="routing"):
            check_stages([name for name in STAGE_NAMES if name != "routing"])

    def test_the_order_written_does_not_matter(self) -> None:
        """The graph's order is the order run, whatever order a profile lists."""
        assert check_stages(list(reversed(STAGE_NAMES))) == STAGE_NAMES

    def test_a_configuration_with_a_bad_workflow_does_not_load(self, tmp_path: Path) -> None:
        """Refused at load, before any recording is read."""
        override = tmp_path / "override.yaml"
        override.write_text(yaml.safe_dump({"workflow": {"stages": ["ADMIT", "VERDICT"]}}))
        with pytest.raises(ValueError, match="may be off"):
            load_triage_config(override)

    def test_a_corpus_pass_for_a_stage_that_is_off_is_refused(self) -> None:
        """extend_cohort over a workflow without COHORT has nothing to write."""
        with pytest.raises(ValueError, match="COHORT is off in this workflow"):
            require_stage(_without("COHORT"), "COHORT")
        require_stage(_without("AIRWAY"), "COHORT")
