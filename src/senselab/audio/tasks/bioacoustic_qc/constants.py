"""Taxonomy of bioacoustic activities emphasizing human activities."""

from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check

BIOACOUSTIC_ACTIVITY_TAXONOMY = {
    "bioacoustic": {
        "checks": [audio_length_positive_check, audio_intensity_positive_check],
        "metrics": [],
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
