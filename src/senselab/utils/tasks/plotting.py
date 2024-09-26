"""This module implements plotting methods for utilities."""

from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from senselab.utils.data_structures import ScriptLine


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
            if text:
                ax.text((start_times[i] + end_times[i]) / 2, i, text, ha="center", va="bottom")

    # Setting labels and title
    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels([])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Transcript Visualization Over Time")

    # Show the plot
    plt.show()


def plot_segment(segments: List[ScriptLine]) -> None:
    """Plots the segments of the transcript over time.

    Args:
        segments (List[ScriptLine]): The segments object containing segments with start and end times and a label.

    Returns:
        None

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
        - Add check if transcript contains segments with time information
    """
    start_times = []
    end_times = []
    labels = []

    for segment in segments:
        # Ensure that segments have start and end times and a label
        if segment.start is None or segment.end is None or segment.speaker is None:
            raise ValueError("Each segment must have start and end times and a label.")
        else:
            start_times.append(segment.start)
            end_times.append(segment.end)
            labels.append(segment.speaker)

    # Create a figure and axis
    _, ax = plt.subplots(figsize=(12, 6))

    # Create a color map based on unique labels
    unique_labels = list(set(labels))
    color_map = cm.get_cmap("tab10", len(unique_labels))  # 'tab10' provides 10 distinct colors
    label_to_color = {label: color_map(i) for i, label in enumerate(unique_labels)}
    label_to_y_value = {label: i for i, label in enumerate(unique_labels)}  # Assign y-value based on label index

    # Plot each segment and add text label with color
    for i, label in enumerate(labels):
        color = label_to_color[label]
        y_value = label_to_y_value[label]  # Get y-value based on label
        ax.plot([start_times[i], end_times[i]], [y_value, y_value], marker="o", color=color, linewidth=2)
        ax.text((start_times[i] + end_times[i]) / 2, y_value, label, ha="center", va="bottom", color=color)

    # Setting labels and title
    ax.set_yticks(range(len(unique_labels)))
    ax.set_yticklabels(unique_labels)
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Segment Visualization Over Time")

    # Show the plot
    plt.show()
