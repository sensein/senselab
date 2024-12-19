"""Provides evaluation functions alignments."""

from senselab.utils.data_structures.script_line import ScriptLine


def check_alignment_differences(
    alignment_one: ScriptLine, alignment_two: ScriptLine, difference_tolerance: float = 0.1
) -> None:
    """Check if two alignments are within the specified difference tolerance.

    Args:
        alignment_one (ScriptLine): The first alignment segment.
        alignment_two (ScriptLine): The second alignment segment.
        difference_tolerance (float): Allowed difference in start and end times (seconds).

    Raises:
        AssertionError: If the start or end times differ by more than the tolerance.
    """
    print(f"Texts:    {alignment_one.text} | {alignment_two.text}")
    assert alignment_one.text == alignment_two.text
    if alignment_one.start and alignment_two.start:
        assert abs(alignment_one.start - alignment_two.start) < difference_tolerance
    if alignment_one.end and alignment_two.end:
        assert abs(alignment_one.end - alignment_two.end) < difference_tolerance
    if alignment_one.chunks and alignment_two.chunks and len(alignment_one.chunks) == len(alignment_two.chunks):
        for a1, a2 in zip(alignment_one.chunks, alignment_two.chunks):
            check_alignment_differences(a1, a2)
