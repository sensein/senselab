"""This module implements plotting methods for utilities."""

import matplotlib.pyplot as plt

from senselab.utils.data_structures.script_line import ScriptLine


def plot_transcript(transcript: ScriptLine) -> None:
    """Plots the transcript visualization over time.

    Args:
        transcript (ScriptLine): The transcript object containing chunks of text with start and end times.

    Returns:
        None

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
        - Add check if transcript contains chunks with time information
    """
    if transcript.chunks is None:
        raise ValueError("The transcript does not contain any chunks.")

    chunks = transcript.chunks

    texts = [chunk.text for chunk in chunks]
    start_times = []
    end_times = []
    for chunk in chunks:
        # Ensure that chunks have start and end times
        if chunk.start is None or chunk.end is None:
            raise ValueError("Each chunk must have both start and end times.")
        else:
            start_times.append(chunk.start)
            end_times.append(chunk.end)

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(12, 6))

    # Plot each text segment and add text label
    for i, text in enumerate(texts):
        if start_times[i] is not None and end_times[i] is not None:
            ax.plot([start_times[i], end_times[i]], [i, i], marker="o")
            ax.text((start_times[i] + end_times[i]) / 2, i, text, ha="center", va="bottom")

    # Setting labels and title
    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Transcript Visualization Over Time")

    # Show the plot
    plt.show()
