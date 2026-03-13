"""This script contains a function for extracting health measurements from audio files.

The initial implementation of this features extraction was started by Nicholas Cummins
from King's College London and has since been further developed and maintained
by the senselab community.
"""

import os
from typing import Any, Dict, List, Literal, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import extract_praat_parselmouth_features_from_audios


def extract_health_measurements(
    audios: List[Audio],
    n_jobs: int = 1,
    backend: Literal["threading", "loky", "multiprocessing", "sequential"] = "sequential",
    verbose: int = 0,
    cache_dir: Optional[str | os.PathLike] = None,
) -> List[Dict[str, Any]]:
    """Extract health measurements from audio files.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_jobs (int, optional):
            Number of parallel jobs to run (default: 1).
        backend (str, optional):
            Backend to use for parallelization.
            - “sequential” (used by default) is a serial backend.
            - “loky” can induce some communication and memory overhead
            when exchanging input and output data with the worker Python processes.
            On some rare systems (such as Pyiodide), the loky backend may not be available.
            - “multiprocessing” previous process-based backend based on multiprocessing.Pool.
            Less robust than loky.
            - “threading” is a very low-overhead backend but it suffers from
            the Python Global Interpreter Lock if the called function relies
            a lot on Python objects. “threading” is mostly useful when the execution
            bottleneck is a compiled extension that explicitly releases the GIL
            (for instance a Cython loop wrapped in a “with nogil” block or an expensive
            call to a library such as NumPy).
        verbose (int, optional):
            Verbosity (default: 0).
            If non zero, progress messages are printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
        cache_dir (str | os.PathLike, optional):
            Path to cache directory. If None is given, no caching is done.


    Returns:
        List[Dict[str, Any]]: List of dictionaries containing speech and voice metrics
        that may be used for health monitoring.
            Metrics include:

            - speaking_rate
            - articulation_rate
            - phonation_ratio
            - pause_rate
            - mean_pause_duration
            - mean_f0_hertz
            - std_f0_hertz
            - mean_intensity_db
            - std_intensity_db
            - range_ratio_intensity_db
            - mean_hnr_db
            - std_hnr_db
            - spectral_slope
            - spectral_tilt
            - cepstral_peak_prominence_mean
            - cepstral_peak_prominence_std
            - mean_f1_loc
            - std_f1_loc
            - mean_b1_loc
            - std_b1_loc
            - mean_f2_loc
            - std_f2_loc
            - mean_b2_loc
            - std_b2_loc
            - spectral_gravity
            - spectral_std_dev
            - spectral_skewness
            - spectral_kurtosis

    Examples:
        >>> audios = [Audio(filepath="sample.wav")]
        >>> extract_health_measurements(audios)
        [{'speaking_rate': 3.874983349680919,
        'articulation_rate': 3.874983349680919,
        'phonation_ratio': 1.0,
        'pause_rate': 0.0,
        'mean_pause_duration': 0.0,
        'mean_f0_hertz': 118.59917806814313,
        'std_f0_hertz': 30.232960797931817,
        'mean_intensity_db': 69.76277128148347,
        'std_intensity_db': 58.54414165935646,
        'range_ratio_intensity_db': -0.25736445047981316,
        'mean_hnr_db': 3.3285614070654375,
        'std_hnr_db': 3.36490968797237,
        'spectral_slope': -13.982306776816046,
        'spectral_tilt': -0.004414961849917737,
        'cepstral_peak_prominence_mean': 7.0388038514346825,
        'cepstral_peak_prominence_std': 1.5672438573255245,
        'mean_f1_loc': 613.4664268420964,
        'std_f1_loc': 303.98235579059883,
        'mean_b1_loc': 401.96960219300837,
        'std_b1_loc': 400.9001719378358,
        'mean_f2_loc': 1701.7755281579418,
        'std_f2_loc': 325.4405394017738,
        'mean_b2_loc': 434.542188503193,
        'std_b2_loc': 380.8914612651878,
        'spectral_gravity': 579.587511962247,
        'spectral_std_dev': 651.3025011919739,
        'spectral_skewness': 3.5879707548251045,
        'spectral_kurtosis': 19.991495997865282}]
    """
    return extract_praat_parselmouth_features_from_audios(
        audios=audios,
        duration=False,
        jitter=False,
        shimmer=False,
        n_jobs=n_jobs,
        backend=backend,
        verbose=verbose,
        cache_dir=cache_dir,
    )
