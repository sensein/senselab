"""This script contains a function for extracting health measurements from audio files.

The initial implementation of this features extraction was started by Nicholas Cummins
from King's College London and has since been further developed and maintained
by the senselab community.
"""

from typing import Any, Dict, List, Optional

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import extract_praat_parselmouth_features_from_audios


def extract_health_measurements(
    audios: List[Audio], plugin: str = "serial", plugin_args: Dict[str, Any] = {}, cache_dir: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Extract health measurements from audio files.

    Args:
        audios (List[Audio]): List of Audio objects.
        plugin (str): Plugin to use for feature extraction. Defaults to "serial".
        plugin_args (Dict[str, Any]): Dictionary of arguments for the feature extraction plugin.
        cache_dir (Optional[str]): Directory to use for caching by pydra. Defaults to None.

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
        >>> audios = [Audio.from_filepath("sample.wav")]
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
        cache_dir=cache_dir,
        plugin=plugin,
        plugin_args=plugin_args,
        duration=False,
        jitter=False,
        shimmer=False,
    )
