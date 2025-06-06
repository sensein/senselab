"""Evaluation utilities for bioacoustic quality control."""

import os
from typing import Callable, Optional, Union

import pandas as pd

from senselab.audio.data_structures import Audio


def get_metric(
    audio_or_path: Union[Audio, str],
    metric_function: Callable[[Audio], float],
    df: Optional[pd.DataFrame] = None,
) -> float:
    """Return a metric, using a cached DataFrame when possible.

    Args:
        audio_or_path: An Audio instance or filepath to the audio file.
        metric_function: The metric function, e.g. ``zero_crossing_rate_metric``.
        df: Optional DataFrame that already contains pre-computed metrics.
            The DataFrame must have:
              * a column ``'audio_path_or_id'`` holding file names, and
              * a column named exactly ``metric_function.__name__``.

    Returns:
        The metric value for this ``audio`` item. If ``df`` is provided and
        contains the value, that cached value is returned; otherwise the metric
        is computed directly and optionally added to ``df``.
    """
    metric_name = metric_function.__name__

    filepath = None
    if isinstance(audio_or_path, str):
        filepath = audio_or_path
    else:
        filepath = audio_or_path.filepath()

    metric = None
    if df is not None and filepath:
        audio_file_name = os.path.basename(filepath)
        row = df[df["audio_path_or_id"] == audio_file_name]
        if not row.empty and metric_name in row.columns:
            metric = row[metric_name].iloc[0]

    if metric is None and isinstance(audio_or_path, Audio):
        metric = metric_function(audio_or_path)
        if df is not None and filepath:
            audio_file_name = os.path.basename(filepath)
            if metric_name not in df.columns:
                df[metric_name] = pd.NA
            df.loc[df["audio_path_or_id"] == audio_file_name, metric_name] = metric

    if metric is None:
        raise ValueError("Expected metric to be non-None.")

    return metric
