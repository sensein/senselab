"""Provides alignment evaluation functions."""

from senselab.utils.data_structures.script_line import ScriptLine


def compare_alignments(alignment_one: ScriptLine, alignment_two: ScriptLine, difference_tolerance: float = 0.1) -> None:
    """Check if two alignments are within the specified difference tolerance.

    Args:
        alignment_one (ScriptLine): The first alignment segment.
        alignment_two (ScriptLine): The second alignment segment.
        difference_tolerance (float): Allowed difference in start and end times (seconds).

    Raises:
        AssertionError: If the start or end times differ by more than the tolerance.
    """
    print(f"Texts:    {alignment_one.text} | {alignment_two.text}")

    if alignment_one.text and alignment_two.text:
        assert (
            alignment_one.text.lower() == alignment_two.text.lower()
        ), f"Text mismatch: '{alignment_one.text}' != '{alignment_two.text}'"

    if alignment_one.start is not None and alignment_two.start is not None:
        assert (
            abs(alignment_one.start - alignment_two.start) < difference_tolerance
        ), f"(difference: {abs(alignment_one.start - alignment_two.start):.3f}s, \
            tolerance: {difference_tolerance:.3f}s)"

    if alignment_one.end is not None and alignment_two.end is not None:
        assert (
            abs(alignment_one.end - alignment_two.end) < difference_tolerance
        ), f"(difference: {abs(alignment_one.end - alignment_two.end):.3f}s, \
            tolerance: {difference_tolerance:.3f}s)"

    if alignment_one.chunks and alignment_two.chunks and len(alignment_one.chunks) == len(alignment_two.chunks):
        for i, (a1, a2) in enumerate(zip(alignment_one.chunks, alignment_two.chunks)):
            print(f"Comparing chunk {i + 1}/{len(alignment_one.chunks)}")
            compare_alignments(a1, a2)
