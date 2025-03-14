"""Taxonomy of bioacoustic activities emphasizing human activities."""

from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check

BIOACOUSTIC_ACTIVITY_TAXONOMY = {
    "bioacoustic": {
        "checks": [audio_length_positive_check, audio_intensity_positive_check],
        "metrics": None,
        "subclass": {
            "human": {
                "checks": None,
                "metrics": None,
                "subclass": {
                    "respiration": {
                        "checks": None,
                        "metrics": None,
                        "subclass": {
                            "breathing": {
                                "checks": None,
                                "metrics": None,
                                "subclass": {
                                    "quiet": {"checks": None, "metrics": None, "subclass": None},
                                    "deep": {"checks": None, "metrics": None, "subclass": None},
                                    "rapid": {"checks": None, "metrics": None, "subclass": None},
                                    "sigh": {"checks": None, "metrics": None, "subclass": None},
                                },
                            },
                            "exhalation": {
                                "checks": None,
                                "metrics": None,
                                "subclass": {
                                    "cough": {
                                        "checks": None,
                                        "metrics": None,
                                        "subclass": {
                                            "voluntary": {"checks": None, "metrics": None, "subclass": None},
                                            "reflexive": {"checks": None, "metrics": None, "subclass": None},
                                        },
                                    }
                                },
                            },
                            "inhalation": {
                                "checks": None,
                                "metrics": None,
                                "subclass": {
                                    "sniff": {"checks": None, "metrics": None, "subclass": None},
                                    "gasp": {"checks": None, "metrics": None, "subclass": None},
                                },
                            },
                        },
                    },
                    "vocalization": {
                        "checks": None,
                        "metrics": None,
                        "subclass": {
                            "speech": {
                                "checks": None,
                                "metrics": None,
                                "subclass": {
                                    "spontaneous_speech": {"checks": None, "metrics": None, "subclass": None},
                                    "read_speech": {"checks": None, "metrics": None, "subclass": None},
                                    "repetitive_speech": {
                                        "checks": None,
                                        "metrics": None,
                                        "subclass": {
                                            "diadochokinesis": {"checks": None, "metrics": None, "subclass": None},
                                            "counting": {"checks": None, "metrics": None, "subclass": None},
                                        },
                                    },
                                    "sustained_phonation": {"checks": None, "metrics": None, "subclass": None},
                                },
                            },
                            "non_speech": {
                                "checks": None,
                                "metrics": None,
                                "subclass": {
                                    "laughter": {"checks": None, "metrics": None, "subclass": None},
                                    "crying": {"checks": None, "metrics": None, "subclass": None},
                                    "humming": {"checks": None, "metrics": None, "subclass": None},
                                    "throat_clearing": {"checks": None, "metrics": None, "subclass": None},
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
