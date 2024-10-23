"""Tests functionality for interfacing with TalkBank datasets."""

from senselab.utils.data_structures.script_line import ScriptLine
from senselab.utils.data_structures.talk_bank_helpers import chats_to_script_lines
from tests.utils.conftest import CHA_TALK_BANK_PATH


def test_chats_to_script_lines() -> None:
    """Tests the conversion of a TalkBank CHAT file to ScriptLines."""
    exp_line_1 = ScriptLine(
        text="a boy named jack was watching his frog last night.", start=0.001, end=34.881, speaker="CHI"
    )
    exp_line_2 = ScriptLine(
        text="when he went to bed he forgot to close the life on the jar that the frog was in.",
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
