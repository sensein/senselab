"""Tests functionality for interfacing with TalkBank datasets."""

import pytest

from senselab.utils.data_structures.script_line import ScriptLine
from senselab.utils.data_structures.talk_bank_helpers import chats_to_script_lines
from tests.utils.conftest import CHA_TALK_BANK_PATH

try:
    import pylangacq

    PYLANGACQ_AVAILABLE = True
except ImportError:
    PYLANGACQ_AVAILABLE = False


@pytest.mark.skipif(PYLANGACQ_AVAILABLE, reason="pylangacq is installed")
def test_chats_to_script_lines_import_error() -> None:
    """Tests that an ImportError is raised when pylangacq is not installed."""
    with pytest.raises(ImportError):
        chats_to_script_lines(CHA_TALK_BANK_PATH)


@pytest.mark.skipif(not PYLANGACQ_AVAILABLE, reason="pylangacq is not installed")
def test_chats_to_script_lines() -> None:
    """Tests the conversion of a TalkBank CHAT file to ScriptLines."""
    exp_line_1 = ScriptLine(
        text="a boy named jack was watching his frog last night.", start=0.001, end=34.881, speaker="CHI"
    )
    exp_line_2 = ScriptLine(
        text="when he went to bed he forgot to close the lid on the jar that the frog was in.",
        start=34.881,
        end=50.681,
        speaker="CHI",
    )
    exp_line_3 = ScriptLine(
        text="so in the morning when he woke up the frog wasn't in the jar.", start=50.681, end=56.088, speaker="CHI"
    )

    exp_result = {CHA_TALK_BANK_PATH: [exp_line_1, exp_line_2, exp_line_3]}

    actual_result = chats_to_script_lines(CHA_TALK_BANK_PATH)
    assert exp_result == actual_result
