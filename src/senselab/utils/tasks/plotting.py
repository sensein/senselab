"""Plotting utilities for transcripts and diarization segments.

These helpers produce simple timeline visualizations using Matplotlib and
return the created `matplotlib.figure.Figure`. Plots are shown in a
non-blocking way (`plt.show(block=False)`), so you can continue execution and
optionally call `fig.savefig(...)` afterwards.
"""

from typing import List

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure

from senselab.utils.data_structures import ScriptLine


def plot_transcript(transcript: ScriptLine) -> Figure:
    """Visualize a transcript's word/segment chunks over time.

    Expects a `ScriptLine` where `chunks` is a list of child `ScriptLine`
    objects, each with `start`, `end`, and `text`. Each chunk is rendered as
    a horizontal segment positioned along the time axis, with its text label.

    Args:
        transcript (ScriptLine):
            Transcript whose `chunks` will be plotted. Every chunk must have
            both `start` and `end` timestamps (in seconds). `text` may be empty.

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError:
            If `transcript.chunks` is `None`, or any chunk lacks `start` or `end`.

    Example:
        >>> from senselab.utils.data_structures import ScriptLine
        >>> chunks = [
        ...     ScriptLine(text="hello", start=0.0, end=0.6),
        ...     ScriptLine(text="world", start=0.7, end=1.4),
        ... ]
        >>> transcript = ScriptLine(text="hello world", chunks=chunks)
        >>> fig = plot_transcript(transcript)
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
    figure, ax = plt.subplots(figsize=(12, 6))

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
    plt.show(block=False)
    return figure


def plot_segment(segments: List[ScriptLine]) -> Figure:
    """Visualize diarization (or generic labeled) segments over time.

    Each `ScriptLine` in `segments` must provide `start`, `end`, and `speaker`
    (used here as a categorical label). Segments are drawn as colored horizontal
    bars, stacked by label on the y-axis.

    Args:
        segments (list[ScriptLine]):
            List of labeled time segments. Each item must have
            `start` (sec), `end` (sec), and `speaker` (label) set.

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError:
            If any segment is missing `start`, `end`, or `speaker`.

    Example:
        >>> from senselab.utils.data_structures import ScriptLine
        >>> segs = [
        ...     ScriptLine(speaker="spk1", start=0.0, end=1.2),
        ...     ScriptLine(speaker="spk2", start=1.0, end=2.0),
        ...     ScriptLine(speaker="spk1", start=2.1, end=3.0),
        ... ]
        >>> fig = plot_segment(segs)
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
    figure, ax = plt.subplots(figsize=(12, 6))

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
    plt.show(block=False)
    return figure
