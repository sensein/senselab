"""Taxonomy of bioacoustic tasks emphasizing human tasks."""

BIOACOUSTIC_TASK_TREE = {
    "bioacoustic": {
        "checks": None,
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
