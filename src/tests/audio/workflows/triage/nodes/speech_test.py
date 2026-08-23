"""SPEECH node tests. Every model call is faked at the node module; DSP and the store run real."""

import pytest

from senselab.audio.workflows.triage.config import load_triage_config


class TestConfigKeys:
    """The Task 5 config additions: present, overridable, and refusing while unmeasured."""

    def test_new_speech_keys_exist_and_the_unmeasured_ones_raise(self) -> None:
        """Null keys are present (overridable) and refuse to be read as values."""
        cfg = load_triage_config()
        assert cfg.get("yamnet.top_k") == 521
        for key in (
            "speech.word_gap_ms",
            "speech.second_diarizer",
            "speech.target_match_cosine",
            "speech.agreement_flag_floor",
            "speech.speech_test_stoi_floor",
        ):
            with pytest.raises(ValueError, match="benchmarks/open.md|no value"):
                cfg.require(key)
