"""This module implements plotting methods for utilities."""


import matplotlib.pyplot as plt

from senselab.utils.data_structures.script_line import ScriptLine


def plot_transcript(transcript: ScriptLine) -> None:
    """Plots the transcript visualization over time.

    Args:
        transcript (Any): The transcript object containing chunks of text with start and end times.

    Returns:
        None

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
        - Add check if transcript contains chunks with time information
    """
    chunks = transcript.chunks

    # Extracting data for visualization
    texts = [chunk.text for chunk in chunks]
    start_times = [chunk.start for chunk in chunks]
    end_times = [chunk.end for chunk in chunks]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot each text segment and add text label
    for i, text in enumerate(texts):
        ax.plot([start_times[i], end_times[i]], [i, i], marker='o')
        ax.text((start_times[i] + end_times[i]) / 2, i, text, ha='center', va='bottom')

    # Setting labels and title
    ax.set_yticks(range(len(texts)))
    ax.set_yticklabels([])
    ax.set_xlabel('Time (seconds)')
    ax.set_title('Transcript Visualization Over Time')

    # Show the plot
    plt.show()