"""Taxonomy of bioacoustic activities emphasizing human activities."""

from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check
from senselab.audio.tasks.bioacoustic_qc.metrics import (
    amplitude_headroom_metric,
    amplitude_interquartile_range_metric,
    amplitude_kurtosis_metric,
    amplitude_modulation_depth_metric,
    amplitude_skew_metric,
    clipping_present_metric,
    crest_factor_metric,
    dynamic_range_metric,
    mean_absolute_amplitude_metric,
    mean_absolute_deviation_metric,
    peak_snr_from_spectral_metric,
    phase_correlation_metric,
    proportion_clipped_metric,
    proportion_silence_at_beginning_metric,
    proportion_silence_at_end_metric,
    proportion_silent_metric,
    root_mean_square_energy_metric,
    shannon_entropy_amplitude_metric,
    signal_variance_metric,
    spectral_gating_snr_metric,
    zero_crossing_rate_metric,
)

BIOACOUSTIC_ACTIVITY_TAXONOMY = {
    "bioacoustic": {
        "checks": [audio_length_positive_check, audio_intensity_positive_check],
        "metrics": [
            proportion_silent_metric,
            proportion_silence_at_beginning_metric,
            proportion_silence_at_end_metric,
            amplitude_headroom_metric,
            spectral_gating_snr_metric,
            proportion_clipped_metric,
            clipping_present_metric,
            amplitude_modulation_depth_metric,
            root_mean_square_energy_metric,
            zero_crossing_rate_metric,
            signal_variance_metric,
            dynamic_range_metric,
            mean_absolute_amplitude_metric,
            mean_absolute_deviation_metric,
            shannon_entropy_amplitude_metric,
            crest_factor_metric,
            peak_snr_from_spectral_metric,
            amplitude_skew_metric,
            amplitude_kurtosis_metric,
            amplitude_interquartile_range_metric,
            phase_correlation_metric,
        ],
        "subclass": {
            "human": {
                "checks": [],
                "metrics": [],
                "subclass": {
                    "respiration": {
                        "checks": [],
                        "metrics": [],
                        "subclass": {
                            "breathing": {
                                "checks": [],
                                "metrics": [],
                                "subclass": {
                                    "quiet": {"checks": [], "metrics": [], "subclass": None},
                                    "deep": {"checks": [], "metrics": [], "subclass": None},
                                    "rapid": {"checks": [], "metrics": [], "subclass": None},
                                    "sigh": {"checks": [], "metrics": [], "subclass": None},
                                },
                            },
                            "exhalation": {
                                "checks": [],
                                "metrics": [],
                                "subclass": {
                                    "cough": {
                                        "checks": [],
                                        "metrics": [],
                                        "subclass": {
                                            "voluntary": {"checks": [], "metrics": [], "subclass": None},
                                            "reflexive": {"checks": [], "metrics": [], "subclass": None},
                                        },
                                    }
                                },
                            },
                            "inhalation": {
                                "checks": [],
                                "metrics": [],
                                "subclass": {
                                    "sniff": {"checks": [], "metrics": [], "subclass": None},
                                    "gasp": {"checks": [], "metrics": [], "subclass": None},
                                },
                            },
                        },
                    },
                    "vocalization": {
                        "checks": [],
                        "metrics": [],
                        "subclass": {
                            "speech": {
                                "checks": [],
                                "metrics": [],
                                "subclass": {
                                    "spontaneous_speech": {"checks": [], "metrics": [], "subclass": None},
                                    "read_speech": {"checks": [], "metrics": [], "subclass": None},
                                    "repetitive_speech": {
                                        "checks": [],
                                        "metrics": [],
                                        "subclass": {
                                            "diadochokinesis": {"checks": [], "metrics": [], "subclass": None},
                                            "counting": {"checks": [], "metrics": [], "subclass": None},
                                        },
                                    },
                                    "sustained_phonation": {"checks": [], "metrics": [], "subclass": None},
                                },
                            },
                            "non_speech": {
                                "checks": [],
                                "metrics": [],
                                "subclass": {
                                    "laughter": {"checks": [], "metrics": [], "subclass": None},
                                    "crying": {"checks": [], "metrics": [], "subclass": None},
                                    "humming": {"checks": [], "metrics": [], "subclass": None},
                                    "throat_clearing": {"checks": [], "metrics": [], "subclass": None},
                                },
                            },
                        },
                    },
                },
            }
        },
    }
}


COMPUTATIONAL_COMPLEXITY_TO_CHECK = {
    "low": [audio_intensity_positive_check, audio_length_positive_check],
    "medium": [],
    "high": [],
}
