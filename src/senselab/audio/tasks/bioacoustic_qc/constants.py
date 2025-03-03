"""Taxonomy of bioacoustic tasks emphasizing human tasks."""

from senselab.audio.tasks.bioacoustic_qc.checks import audio_intensity_positive_check, audio_length_positive_check

BIOACOUSTIC_TASK_TREE = {
    "bioacoustic": {
        "checks": [audio_length_positive_check, audio_intensity_positive_check],
        "subclass": {
            "human": {
                "checks": None,
                "subclass": {
                    "respiration": {
                        "checks": None,
                        "subclass": {
                            "breathing": {
                                "checks": None,
                                "subclass": {
                                    "quiet": {"checks": None, "subclass": None},
                                    "deep": {"checks": None, "subclass": None},
                                    "rapid": {"checks": None, "subclass": None},
                                    "sigh": {"checks": None, "subclass": None},
                                },
                            },
                            "exhalation": {
                                "checks": None,
                                "subclass": {
                                    "cough": {
                                        "checks": None,
                                        "subclass": {
                                            "voluntary": {"checks": None, "subclass": None},
                                            "reflexive": {"checks": None, "subclass": None},
                                        },
                                    }
                                },
                            },
                            "inhalation": {
                                "checks": None,
                                "subclass": {
                                    "sniff": {"checks": None, "subclass": None},
                                    "gasp": {"checks": None, "subclass": None},
                                },
                            },
                        },
                    },
                    "vocalization": {
                        "checks": None,
                        "subclass": {
                            "speech": {
                                "checks": None,
                                "subclass": {
                                    "spontaneous_speech": {"checks": None, "subclass": None},
                                    "read_speech": {"checks": None, "subclass": None},
                                    "repetitive_speech": {
                                        "checks": None,
                                        "subclass": {
                                            "diadochokinesis": {"checks": None, "subclass": None},
                                            "counting": {"checks": None, "subclass": None},
                                        },
                                    },
                                    "sustained_phonation": {"checks": None, "subclass": None},
                                },
                            },
                            "non_speech": {
                                "checks": None,
                                "subclass": {
                                    "laughter": {"checks": None, "subclass": None},
                                    "crying": {"checks": None, "subclass": None},
                                    "humming": {"checks": None, "subclass": None},
                                    "throat_clearing": {"checks": None, "subclass": None},
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
